from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset


# Filename index 1~4 corresponds to these four frequencies.
FREQUENCIES = (1, 4, 16, 64)


def _load_gray_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    image = np.asarray(Image.open(path), dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale image at {path}, "
            f"but got shape {image.shape}."
        )

    if image.max() > 1.0:
        image = image / 255.0

    return image


def _load_mat_array(path: Path, key: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    mat = loadmat(path)

    if key not in mat:
        available = [
            k for k in mat.keys()
            if not k.startswith("__")
        ]
        raise KeyError(
            f"{path}: expected key '{key}', "
            f"found {available}."
        )

    array = np.asarray(mat[key], dtype=np.float32)

    if array.ndim != 2:
        raise ValueError(
            f"{path}: expected 2-D array, "
            f"got shape {array.shape}."
        )

    return array


class HDRSLDataset(Dataset):
    """
    Dataset for the released HDRSL metal dataset.

    Returned tensors
    ----------------
    ldr:
        [8, H, W]

        Channel order:
        [
            10ms-f1, 40ms-f1,
            10ms-f4, 40ms-f4,
            10ms-f16, 40ms-f16,
            10ms-f64, 40ms-f64,
        ]

    hdr_gt:
        [4, H, W]

        Channel order:
        [f1, f4, f16, f64]

    sc_gt:
        [8, H, W]

        Channel order:
        [
            S1, C1,
            S4, C4,
            S16, C16,
            S64, C64,
        ]

    absolute_phase:
        [H, W]

    sample_id:
        str
    """

    def __init__(
        self,
        root: str | Path,
        sample_ids: Sequence[str] | None = None,
    ):
        self.root = Path(root)

        self.hdr_dir = (
            self.root
            / "images_input"
            / "images_GT"
        )

        self.ldr_10ms_dir = (
            self.root
            / "input_LDR_10ms"
            / "images_low"
        )

        self.ldr_40ms_dir = (
            self.root
            / "input_LDR_40ms"
            / "images_4"
        )

        self.sine_dir = (
            self.root
            / "Sine_component"
            / "fenzi_GT_mat_2"
        )

        self.cosine_dir = (
            self.root
            / "Cosine_component"
            / "fenmu_GT_mat_2"
        )

        self.absolute_phase_dir = (
            self.root
            / "Absolute_phase"
            / "Phases_GT_mat"
        )

        if not self.hdr_dir.is_dir():
            raise FileNotFoundError(self.hdr_dir)

        if sample_ids is None:
            sample_ids = [
                p.name
                for p in self.hdr_dir.iterdir()
                if p.is_dir()
            ]

            sample_ids = sorted(
                sample_ids,
                key=lambda x: int(x),
            )

        self.sample_ids = [
            str(sample_id)
            for sample_id in sample_ids
        ]

        if not self.sample_ids:
            raise RuntimeError(
                f"No samples found in {self.hdr_dir}."
            )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]

        ldr_channels = []
        hdr_channels = []
        sc_channels = []

        for file_index in range(1, 5):
            # ------------------------------------------------
            # Short / long exposure LDR
            # ------------------------------------------------

            ldr_10ms = _load_gray_image(
                self.ldr_10ms_dir
                / sample_id
                / f"{sample_id}_{file_index}.bmp"
            )  # 短曝光，4个频段的四张照片

            ldr_40ms = _load_gray_image(
                self.ldr_40ms_dir
                / sample_id
                / f"{sample_id}_{file_index}.bmp"
            ) # 长曝光，4个频段的四张照片

            ldr_channels.extend(
                [ldr_10ms, ldr_40ms]
            )

            # ------------------------------------------------
            # HDR GT
            # ------------------------------------------------

            hdr = _load_gray_image(
                self.hdr_dir
                / sample_id
                / f"{sample_id}_{file_index}.bmp"
            ) # HDR Ground Truth，4个频段的四张照片

            hdr_channels.append(hdr)

            # ------------------------------------------------
            # S / C GT
            # ------------------------------------------------

            sine = _load_mat_array(
                self.sine_dir
                / f"{sample_id}-{file_index}.mat",
                "numerator",
            ) # 正弦分量，4个频段的四张照片

            cosine = _load_mat_array(
                self.cosine_dir
                / f"{sample_id}-{file_index}.mat",
                "denominator",
            ) # 余弦分量，4个频段的四张照片

            sc_channels.extend([sine, cosine])

        absolute_phase = _load_mat_array(
            self.absolute_phase_dir
            / f"{sample_id}.mat",
            "phase",
        ) # 绝对相位，一张照片

        ldr = np.stack(
            ldr_channels,
            axis=0,
        )

        hdr_gt = np.stack(
            hdr_channels,
            axis=0,
        )

        sc_gt = np.stack(
            sc_channels,
            axis=0,
        )

        spatial_shape = ldr.shape[-2:]

        if hdr_gt.shape[-2:] != spatial_shape:
            raise ValueError(
                f"Sample {sample_id}: "
                f"HDR shape {hdr_gt.shape[-2:]} "
                f"!= LDR shape {spatial_shape}"
            )

        if sc_gt.shape[-2:] != spatial_shape:
            raise ValueError(
                f"Sample {sample_id}: "
                f"S/C shape {sc_gt.shape[-2:]} "
                f"!= LDR shape {spatial_shape}"
            )

        if absolute_phase.shape != spatial_shape:
            raise ValueError(
                f"Sample {sample_id}: "
                f"absolute phase shape "
                f"{absolute_phase.shape} "
                f"!= LDR shape {spatial_shape}"
            )

        return {
            "sample_id": sample_id,

            "ldr": torch.from_numpy(
                ldr
            ).float().contiguous(),  # 短曝光和长曝光的LDR，共8个通道，shape为[8, H, W]

            "hdr_gt": torch.from_numpy(
                hdr_gt
            ).float().contiguous(),  # HDR GT，4个频段的四张照片，shape为[4, H, W]

            "sc_gt": torch.from_numpy(
                sc_gt
            ).float().contiguous(),  # S/C GT，多步相移图计算出来的 S/C 数值矩阵，shape为[8, H, W]

            "absolute_phase": torch.from_numpy(
                absolute_phase
            ).float().contiguous(),  # 绝对相位，以最高频率 f64 的相位精度为基础，经过相位解模糊后的一张二维数值矩阵，shape为[H, W]
        }