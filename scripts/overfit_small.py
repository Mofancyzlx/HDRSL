from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from unet import UNet, UNet_attention
from loss import SSIM, MseDirectionLoss


FREQUENCIES = [1, 4, 16, 64]
TWO_PI = 2.0 * np.pi


# ============================================================
# Dataset
# ============================================================

def load_gray(path: Path):
    x = np.asarray(Image.open(path), dtype=np.float32)
    if x.max() > 1:
        x /= 255.0
    return x


def load_mat_value(path: Path, key: str):
    mat = loadmat(path)
    assert key in mat, f"{path}: expected key '{key}'"
    return np.asarray(mat[key], dtype=np.float32)


class HDRSLDataset(Dataset):
    def __init__(self, root: Path, sample_ids):
        self.root = root
        self.sample_ids = list(sample_ids)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sid = self.sample_ids[index]

        # ----------------------------------------------------
        # 8ch LDR:
        # [10ms-f1, 40ms-f1, ..., 10ms-f4, 40ms-f4]
        # ----------------------------------------------------
        ldr = []

        for i in range(1, 5):
            low = load_gray(
                self.root
                / "input_LDR_10ms"
                / "images_low"
                / sid
                / f"{sid}_{i}.bmp"
            )

            high = load_gray(
                self.root
                / "input_LDR_40ms"
                / "images_4"
                / sid
                / f"{sid}_{i}.bmp"
            )

            ldr.extend([low, high])

        ldr = np.stack(ldr, axis=0)

        # ----------------------------------------------------
        # 4ch HDR GT
        # ----------------------------------------------------
        hdr = []

        for i in range(1, 5):
            x = load_gray(
                self.root
                / "images_input"
                / "images_GT"
                / sid
                / f"{sid}_{i}.bmp"
            )
            hdr.append(x)

        hdr = np.stack(hdr, axis=0)

        # ----------------------------------------------------
        # 8ch S/C GT:
        # [S1,C1,S2,C2,S3,C3,S4,C4]
        # ----------------------------------------------------
        sc = []

        for i in range(1, 5):
            s = load_mat_value(
                self.root
                / "Sine_component"
                / "fenzi_GT_mat_2"
                / f"{sid}-{i}.mat",
                "numerator",
            )

            c = load_mat_value(
                self.root
                / "Cosine_component"
                / "fenmu_GT_mat_2"
                / f"{sid}-{i}.mat",
                "denominator",
            )

            sc.extend([s, c])

        sc = np.stack(sc, axis=0)

        # ----------------------------------------------------
        # Absolute phase GT
        # ----------------------------------------------------
        absolute = load_mat_value(
            self.root
            / "Absolute_phase"
            / "Phases_GT_mat"
            / f"{sid}.mat",
            "phase",
        )

        return {
            "id": sid,
            "ldr": torch.from_numpy(ldr),
            "hdr": torch.from_numpy(hdr),
            "sc": torch.from_numpy(sc),
            "absolute": torch.from_numpy(absolute),
        }


# ============================================================
# S/C -> Absolute Phase
#
# 已由官方 GT 闭环验证：
#
# wrapped = -atan2(S,C) + pi
# 1 -> 4 -> 16 -> 64
# ============================================================

def sc_to_absolute(sc):
    s = sc[:, 0::2]
    c = sc[:, 1::2]

    wrapped = -torch.atan2(s, c) + torch.pi

    absolute = wrapped[:, 0]

    for i in range(1, 4):
        ratio = FREQUENCIES[i] / FREQUENCIES[i - 1]

        k = torch.round(
            (ratio * absolute - wrapped[:, i])
            / (2.0 * torch.pi)
        )

        absolute = wrapped[:, i] + 2.0 * torch.pi * k

    return absolute


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def evaluate(student, phase_net, loader, device, use_amp):
    student.eval()
    phase_net.eval()

    hdr_error_sum = 0.0
    hdr_count = 0

    sc_error_sum = 0.0
    sc_count = 0

    phase_error_sum = 0.0
    phase_count = 0

    for batch in loader:
        ldr = batch["ldr"].to(device)
        hdr_gt = batch["hdr"].to(device)
        sc_gt = batch["sc"].to(device)
        absolute_gt = batch["absolute"].to(device)

        with autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            hdr_pred, _ = student(ldr)

            phase_input = torch.cat(
                [hdr_pred, ldr],
                dim=1,
            )

            sc_pred, _ = phase_net(phase_input)

        # HDR MAE
        hdr_error_sum += torch.abs(
            hdr_pred.float() - hdr_gt
        ).sum().item()

        hdr_count += hdr_gt.numel()

        # S/C MAE
        sc_error_sum += torch.abs(
            sc_pred.float() - sc_gt
        ).sum().item()

        sc_count += sc_gt.numel()

        # Absolute phase MAE
        absolute_pred = sc_to_absolute(sc_pred.float())

        valid = (
            torch.isfinite(absolute_pred)
            & torch.isfinite(absolute_gt)
        )

        phase_error_sum += torch.abs(
            absolute_pred[valid] - absolute_gt[valid]
        ).sum().item()

        phase_count += valid.sum().item()

    return {
        "hdr_mae": hdr_error_sum / hdr_count,
        "sc_mae": sc_error_sum / sc_count,
        "absolute_phase_mae": phase_error_sum / phase_count,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--teacher-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_root = ROOT / "dataset"

    # --------------------------------------------------------
    # Select fixed samples
    # --------------------------------------------------------
    hdr_root = dataset_root / "images_input" / "images_GT"

    all_ids = sorted(
        [p.name for p in hdr_root.iterdir() if p.is_dir()],
        key=lambda x: int(x),
    )

    rng = random.Random(args.seed)
    selected_ids = rng.sample(all_ids, args.samples)
    selected_ids = sorted(selected_ids, key=lambda x: int(x))

    print("Selected sample IDs:")
    print(selected_ids)

    dataset = HDRSLDataset(
        dataset_root,
        selected_ids,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    eval_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = args.amp and device.type == "cuda"

    print(f"\nDevice: {device}")
    print(f"AMP: {use_amp}")

    # --------------------------------------------------------
    # Networks
    # --------------------------------------------------------
    student = UNet_attention(
        8,
        4,
        False,
    ).to(device)

    teacher = UNet(
        4,
        4,
        False,
    ).to(device)

    phase_net = UNet(
        12,
        8,
        False,
    ).to(device)

    mse = nn.MSELoss()
    mae = nn.L1Loss()
    ssim_loss = SSIM(device=device)

    distillation_loss_fn = MseDirectionLoss(0.1)

    # ========================================================
    # Initial evaluation
    # ========================================================
    metrics = evaluate(
        student,
        phase_net,
        eval_loader,
        device,
        use_amp,
    )

    print("\n=== Initial ===")
    print(
        f"HDR MAE          = {metrics['hdr_mae']:.6f}\n"
        f"S/C MAE          = {metrics['sc_mae']:.6f}\n"
        f"Absolute phase MAE = "
        f"{metrics['absolute_phase_mae']:.6f}"
    )

    # ========================================================
    # Stage A: Teacher warm-up
    # HDR GT -> Teacher -> HDR GT
    # ========================================================
    teacher_optimizer = torch.optim.Adam(
        teacher.parameters(),
        lr=args.lr,
    )

    teacher_scaler = GradScaler(
        "cuda",
        enabled=use_amp,
    )

    print("\n=== Teacher warm-up ===")

    for epoch in range(1, args.teacher_epochs + 1):
        teacher.train()

        running = 0.0

        for batch in loader:
            hdr_gt = batch["hdr"].to(device)

            teacher_optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                teacher_pred, _ = teacher(hdr_gt)

                teacher_loss = (
                    torch.sqrt(
                        mse(teacher_pred, hdr_gt) + 1e-12
                    )
                    + mae(teacher_pred, hdr_gt)
                    + ssim_loss(teacher_pred, hdr_gt)
                )

            teacher_scaler.scale(
                teacher_loss
            ).backward()

            teacher_scaler.step(
                teacher_optimizer
            )

            teacher_scaler.update()

            running += teacher_loss.item()

        print(
            f"Teacher epoch {epoch:02d}: "
            f"loss={running / len(loader):.6f}"
        )

    # Freeze teacher
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad_(False)

    # ========================================================
    # Stage B: Student + Phase Network
    # ========================================================
    optimizer = torch.optim.Adam(
        list(student.parameters())
        + list(phase_net.parameters()),
        lr=args.lr,
    )

    scaler = GradScaler(
        "cuda",
        enabled=use_amp,
    )

    print("\n=== Student + Phase overfit ===")

    for epoch in range(1, args.epochs + 1):

        student.train()
        phase_net.train()

        total_running = 0.0
        hdr_running = 0.0
        sc_running = 0.0
        distill_running = 0.0

        for batch in loader:
            ldr = batch["ldr"].to(device)
            hdr_gt = batch["hdr"].to(device)
            sc_gt = batch["sc"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                # --------------------------------------------
                # HDR Generation
                # --------------------------------------------
                hdr_pred, student_features = student(ldr)

                with torch.no_grad():
                    _, teacher_features = teacher(hdr_gt)

                hdr_reconstruction_loss = (
                    torch.sqrt(
                        mse(hdr_pred, hdr_gt) + 1e-12
                    )
                    + mae(hdr_pred, hdr_gt)
                    + ssim_loss(hdr_pred, hdr_gt)
                )

                distill_loss = distillation_loss_fn(
                    student_features,
                    teacher_features,
                )

                hdr_loss = (
                    hdr_reconstruction_loss
                    + distill_loss
                )

                # --------------------------------------------
                # Phase Calculation
                # --------------------------------------------
                phase_input = torch.cat(
                    [hdr_pred, ldr],
                    dim=1,
                )

                sc_pred, _ = phase_net(phase_input)

                sc_loss = (
                    torch.sqrt(
                        mse(sc_pred, sc_gt) + 1e-12
                    )
                    + mae(sc_pred, sc_gt)
                )

                total_loss = hdr_loss + sc_loss

            scaler.scale(total_loss).backward()

            scaler.step(optimizer)
            scaler.update()

            total_running += total_loss.item()
            hdr_running += hdr_reconstruction_loss.item()
            sc_running += sc_loss.item()
            distill_running += distill_loss.item()

        metrics = evaluate(
            student,
            phase_net,
            eval_loader,
            device,
            use_amp,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"total={total_running / len(loader):.6f} | "
            f"hdr={hdr_running / len(loader):.6f} | "
            f"distill={distill_running / len(loader):.6f} | "
            f"sc={sc_running / len(loader):.6f} | "
            f"HDR_MAE={metrics['hdr_mae']:.6f} | "
            f"SC_MAE={metrics['sc_mae']:.6f} | "
            f"ABS_MAE={metrics['absolute_phase_mae']:.6f}"
        )

    # ========================================================
    # Save diagnostic checkpoint
    # ========================================================
    output_dir = ROOT / "outputs" / "overfit_small"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "phase_net": phase_net.state_dict(),
            "sample_ids": selected_ids,
            "args": vars(args),
        },
        output_dir / "checkpoint.pt",
    )

    print(
        f"\nCheckpoint saved to: "
        f"{output_dir / 'checkpoint.pt'}"
    )


if __name__ == "__main__":
    main()