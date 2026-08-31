from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
SAMPLE = "1"

FREQUENCIES = [1, 4, 16, 64]
TWO_PI = 2.0 * np.pi


def load_mat(path: Path, key: str):
    mat = loadmat(path)

    keys = [k for k in mat.keys() if not k.startswith("__")]
    assert key in mat, f"{path}: expected key '{key}', found {keys}"

    return np.asarray(mat[key], dtype=np.float64)


# ============================================================
# 1. Load S / C ground truth
# ============================================================

sine_list = []
cosine_list = []

for i in range(1, 5):
    sine = load_mat(
        DATASET
        / "Sine_component"
        / "fenzi_GT_mat_2"
        / f"{SAMPLE}-{i}.mat",
        "numerator",
    )

    cosine = load_mat(
        DATASET
        / "Cosine_component"
        / "fenmu_GT_mat_2"
        / f"{SAMPLE}-{i}.mat",
        "denominator",
    )

    sine_list.append(sine)
    cosine_list.append(cosine)

sine = np.stack(sine_list, axis=0)
cosine = np.stack(cosine_list, axis=0)


# ============================================================
# 2. Reconstruct four wrapped phases
#
# This follows the convention used by the released
# psp_get_phase.py and test.py:
#
#     phi = -atan2(S, C)
# ============================================================

wrapped = -np.arctan2(sine, cosine)

print("=== Wrapped phase ===")

for i, freq in enumerate(FREQUENCIES):
    print(
        f"f={freq:<2d}: "
        f"min={wrapped[i].min(): .6f}, "
        f"max={wrapped[i].max(): .6f}, "
        f"mean={wrapped[i].mean(): .6f}"
    )


# ============================================================
# 3. Standard hierarchical multi-frequency unwrapping
#
# Phi_high =
#     phi_high
#     + 2*pi*round(
#         (n*Phi_low - phi_high) / (2*pi)
#       )
#
# n = f_high / f_low
# ============================================================

def hierarchical_unwrap(wrapped_phase, first_to_2pi=False):
    if first_to_2pi:
        absolute = np.mod(wrapped_phase[0], TWO_PI)
    else:
        absolute = wrapped_phase[0].copy()

    for i in range(1, len(FREQUENCIES)):
        ratio = FREQUENCIES[i] / FREQUENCIES[i - 1]

        k = np.rint(
            (ratio * absolute - wrapped_phase[i])
            / TWO_PI
        )

        absolute = wrapped_phase[i] + TWO_PI * k

    return absolute


# Version A:
# directly use lowest-frequency phase in [-pi, pi]
absolute_raw = hierarchical_unwrap(
    wrapped,
    first_to_2pi=False,
)

# Version B:
# map lowest-frequency phase into [0, 2pi)
absolute_2pi = hierarchical_unwrap(
    wrapped,
    first_to_2pi=True,
)


# ============================================================
# 4. Also test the literal form of Eq. (8) as parsed from PDF:
#
# round((n*Phi_low - phi_high) / (2*pi*n))
#
# We test rather than assume whether this is the actual
# implementation convention.
# ============================================================

def hierarchical_unwrap_literal_eq8(wrapped_phase):
    absolute = np.mod(wrapped_phase[0], TWO_PI)

    for i in range(1, len(FREQUENCIES)):
        ratio = FREQUENCIES[i] / FREQUENCIES[i - 1]

        k = np.rint(
            (ratio * absolute - wrapped_phase[i])
            / (TWO_PI * ratio)
        )

        absolute = wrapped_phase[i] + TWO_PI * k

    return absolute


absolute_literal = hierarchical_unwrap_literal_eq8(wrapped)


# ============================================================
# 5. Load official absolute-phase GT
# ============================================================

absolute_gt = load_mat(
    DATASET
    / "Absolute_phase"
    / "Phases_GT_mat"
    / f"{SAMPLE}.mat",
    "phase",
)


# ============================================================
# 6. Compare
# ============================================================

def report(name, pred, gt):
    diff = pred - gt
    abs_diff = np.abs(diff)

    print(f"\n=== {name} ===")
    print(
        f"pred: min={pred.min():.6f}, "
        f"max={pred.max():.6f}, "
        f"mean={pred.mean():.6f}"
    )

    print(f"MAE       = {abs_diff.mean():.10f}")
    print(f"RMSE      = {np.sqrt(np.mean(diff ** 2)):.10f}")
    print(f"Max error = {abs_diff.max():.10f}")
    print(f"Mean diff = {diff.mean():.10f}")
    print(f"Median diff = {np.median(diff):.10f}")


print("\n=== Official absolute phase GT ===")
print(
    f"shape={absolute_gt.shape}, "
    f"min={absolute_gt.min():.6f}, "
    f"max={absolute_gt.max():.6f}, "
    f"mean={absolute_gt.mean():.6f}"
)

report(
    "A: raw lowest phase [-pi, pi]",
    absolute_raw,
    absolute_gt,
)

report(
    "B: lowest phase mapped to [0, 2pi)",
    absolute_2pi,
    absolute_gt,
)

report(
    "C: literal parsed Eq.(8)",
    absolute_literal,
    absolute_gt,
)

def hierarchical_unwrap_shift_pi(wrapped_phase):
    # Convert [-pi, pi] representation to [0, 2pi]
    shifted = wrapped_phase + np.pi

    absolute = shifted[0].copy()

    for i in range(1, len(FREQUENCIES)):
        ratio = FREQUENCIES[i] / FREQUENCIES[i - 1]

        k = np.rint(
            (ratio * absolute - shifted[i])
            / TWO_PI
        )

        absolute = shifted[i] + TWO_PI * k

    return absolute


absolute_shift_pi = hierarchical_unwrap_shift_pi(wrapped)

report(
    "D: all wrapped phases shifted by +pi",
    absolute_shift_pi,
    absolute_gt,
)