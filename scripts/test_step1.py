from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from utils.hdrsl_dataset import HDRSLDataset
from utils.phase import sc_to_absolute_phase


dataset = HDRSLDataset(
    ROOT / "dataset",
    sample_ids=["1"],
)

sample = dataset[0]

print("sample_id      :", sample["sample_id"])
print("ldr            :", tuple(sample["ldr"].shape))
print("hdr_gt         :", tuple(sample["hdr_gt"].shape))
print("sc_gt          :", tuple(sample["sc_gt"].shape))
print(
    "absolute_phase :",
    tuple(sample["absolute_phase"].shape),
)

absolute_reconstructed = sc_to_absolute_phase(
    sample["sc_gt"]
)

error = torch.abs(
    absolute_reconstructed
    - sample["absolute_phase"]
)

print()
print("Absolute phase verification:")
print(
    f"MAE       = {error.mean().item():.10f}"
)
print(
    f"RMSE      = "
    f"{torch.sqrt(torch.mean(error ** 2)).item():.10f}"
)
print(
    f"Max error = {error.max().item():.10f}"
)