from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from unet import UNet, UNet_attention


DATASET = ROOT / "dataset"
SAMPLE = "1"


def load_gray(path: Path):
    img = np.asarray(Image.open(path), dtype=np.float32)

    if img.max() > 1:
        img /= 255.0

    return img


# ============================================================
# 1. Load one sample:
#
# channel order:
# [10ms-f1, 40ms-f1,
#  10ms-f2, 40ms-f2,
#  10ms-f3, 40ms-f3,
#  10ms-f4, 40ms-f4]
# ============================================================

ldr_channels = []

for freq_index in range(1, 5):
    low_path = (
        DATASET
        / "input_LDR_10ms"
        / "images_low"
        / SAMPLE
        / f"{SAMPLE}_{freq_index}.bmp"
    )

    high_path = (
        DATASET
        / "input_LDR_40ms"
        / "images_4"
        / SAMPLE
        / f"{SAMPLE}_{freq_index}.bmp"
    )

    ldr_channels.append(load_gray(low_path))
    ldr_channels.append(load_gray(high_path))


ldr_np = np.stack(ldr_channels, axis=0)

# [C,H,W] -> [B,C,H,W]
ldr = torch.from_numpy(ldr_np).unsqueeze(0).float()


# ============================================================
# 2. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Input LDR shape: {tuple(ldr.shape)}")

ldr = ldr.to(device)


# ============================================================
# 3. Build the paper-version network skeleton
#
# HDR Generation:
#     8 -> 4
#
# Phase Calculation:
#     (8 LDR + 4 HDR) = 12 -> 8 S/C
# ============================================================

hdr_net = UNet_attention(
    n_channels=8,
    n_classes=4,
    bilinear=False,
).to(device)

phase_net = UNet(
    n_channels=12,
    n_classes=8,
    bilinear=False,
).to(device)

hdr_net.eval()
phase_net.eval()


# ============================================================
# 4. Random-initialized forward
# ============================================================

with torch.inference_mode():

    # [1,8,H,W] -> [1,4,H,W]
    hdr_pred, hdr_features = hdr_net(ldr)

    # [1,4,H,W] + [1,8,H,W] -> [1,12,H,W]
    phase_input = torch.cat(
        [hdr_pred, ldr],
        dim=1,
    )

    # [1,12,H,W] -> [1,8,H,W]
    sc_pred, phase_features = phase_net(phase_input)


# ============================================================
# 5. Validation
# ============================================================

assert ldr.ndim == 4
assert ldr.shape[1] == 8

assert hdr_pred.shape[1] == 4

assert phase_input.shape[1] == 12

assert sc_pred.shape[1] == 8

assert torch.isfinite(hdr_pred).all()
assert torch.isfinite(sc_pred).all()


# ============================================================
# 6. Print result
# ============================================================

print("\n=== Forward smoke test ===")

print(f"LDR input       : {tuple(ldr.shape)}")
print(f"HDR prediction  : {tuple(hdr_pred.shape)}")
print(f"Phase input     : {tuple(phase_input.shape)}")
print(f"S/C prediction  : {tuple(sc_pred.shape)}")

print("\nHDR encoder features:")
for i, x in enumerate(hdr_features):
    print(f"  feature[{i}] = {tuple(x.shape)}")

print("\nPhase encoder features:")
for i, x in enumerate(phase_features):
    print(f"  feature[{i}] = {tuple(x.shape)}")

print("\nRandom output statistics:")

print(
    "HDR prediction : "
    f"min={hdr_pred.min().item():.6f}, "
    f"max={hdr_pred.max().item():.6f}, "
    f"mean={hdr_pred.mean().item():.6f}"
)

print(
    "S/C prediction : "
    f"min={sc_pred.min().item():.6f}, "
    f"max={sc_pred.max().item():.6f}, "
    f"mean={sc_pred.mean().item():.6f}"
)

print("\nPASS: HDRSL forward data flow is valid.")