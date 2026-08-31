from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet, UNet_attention

from utils.hdrsl_dataset import HDRSLDataset
from utils.hdrsl_split import split_sample_ids
from utils.phase import (
    sc_to_wrapped_phase,
    sc_to_absolute_phase,
)


def per_sample_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Return one MAE value for every sample in the batch.
    """

    error = torch.abs(
        prediction - target
    )

    return error.flatten(1).mean(dim=1)


def per_sample_finite_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample MAE using finite pixels only.
    """

    error = torch.abs(
        prediction - target
    )

    valid = (
        torch.isfinite(prediction)
        & torch.isfinite(target)
    )

    error = error.flatten(1)
    valid = valid.flatten(1)

    numerator = (
        error * valid
    ).sum(dim=1)

    denominator = (
        valid.sum(dim=1)
        .clamp_min(1)
    )

    return numerator / denominator


def circular_phase_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Wrapped-phase MAE using shortest angular distance.

    This avoids treating values close to 0 and 2*pi
    as having an artificial ~2*pi error.
    """

    delta = prediction - target

    circular_error = torch.abs(
        torch.atan2(
            torch.sin(delta),
            torch.cos(delta),
        )
    )

    return circular_error.flatten(1).mean(dim=1)


def summarize(values: list[float]):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "num_samples": int(len(x)),
    }


def load_checkpoint(
    checkpoint_path: Path,
    student,
    phase_net,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "student" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'student'."
        )

    if "phase_net" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'phase_net'."
        )

    student.load_state_dict(
        checkpoint["student"]
    )

    phase_net.load_state_dict(
        checkpoint["phase_net"]
    )


@torch.inference_mode()
def evaluate(
    student,
    phase_net,
    loader,
    device,
    use_amp,
):
    student.eval()
    phase_net.eval()

    results = {
        "hdr_mae": [],
        "sine_mae": [],
        "cosine_mae": [],
        "wrapped_phase_mae": [],
        "wrapped_phase_direct_mae": [],
        "absolute_phase_mae": [],
    }

    per_sample_results = []

    for batch in tqdm(
        loader,
        desc="Evaluating HDRSL",
    ):
        ldr = batch["ldr"].to(
            device,
            non_blocking=True,
        )

        hdr_gt = batch["hdr_gt"].to(
            device,
            non_blocking=True,
        )

        sc_gt = batch["sc_gt"].to(
            device,
            non_blocking=True,
        )

        absolute_gt = (
            batch["absolute_phase"]
            .to(
                device,
                non_blocking=True,
            )
        )

        # ----------------------------------------------------
        # Network forward
        # ----------------------------------------------------

        with autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            hdr_pred, _ = student(ldr)

            phase_input = torch.cat(
                [hdr_pred, ldr],
                dim=1,
            )

            sc_pred, _ = phase_net(
                phase_input
            )

        # All evaluation calculations use float32/64
        # rather than AMP output precision.

        hdr_pred = hdr_pred.float()
        sc_pred = sc_pred.float()

        # ----------------------------------------------------
        # HDR fringe MAE
        # ----------------------------------------------------

        hdr_mae = per_sample_mae(
            hdr_pred,
            hdr_gt,
        )

        # ----------------------------------------------------
        # Sine / Cosine MAE
        #
        # channel:
        # S1,C1,S4,C4,S16,C16,S64,C64
        # ----------------------------------------------------

        sine_pred = sc_pred[:, 0::2]
        cosine_pred = sc_pred[:, 1::2]

        sine_gt = sc_gt[:, 0::2]
        cosine_gt = sc_gt[:, 1::2]

        sine_mae = per_sample_mae(
            sine_pred,
            sine_gt,
        )

        cosine_mae = per_sample_mae(
            cosine_pred,
            cosine_gt,
        )

        # ----------------------------------------------------
        # Wrapped phase
        # ----------------------------------------------------

        wrapped_pred = sc_to_wrapped_phase(
            sc_pred
        )

        wrapped_gt = sc_to_wrapped_phase(
            sc_gt
        )

        # Primary wrapped-phase metric:
        # shortest angular difference.
        wrapped_mae = circular_phase_mae(
            wrapped_pred,
            wrapped_gt,
        )

        # Auxiliary direct MAE, kept only for diagnosis.
        wrapped_direct_mae = per_sample_mae(
            wrapped_pred,
            wrapped_gt,
        )

        # ----------------------------------------------------
        # Absolute phase
        # ----------------------------------------------------

        absolute_pred = (
            sc_to_absolute_phase(
                sc_pred
            )
        )

        absolute_mae = (
            per_sample_finite_mae(
                absolute_pred,
                absolute_gt.double(),
            )
        )

        # ----------------------------------------------------
        # Store per-sample results
        # ----------------------------------------------------

        for key, values in [
            ("hdr_mae", hdr_mae),
            ("sine_mae", sine_mae),
            ("cosine_mae", cosine_mae),
            (
                "wrapped_phase_mae",
                wrapped_mae,
            ),
            (
                "wrapped_phase_direct_mae",
                wrapped_direct_mae,
            ),
            (
                "absolute_phase_mae",
                absolute_mae,
            ),
        ]:
            results[key].extend(
                values.detach()
                .cpu()
                .tolist()
            )

        # ----------------------------------------------------
        # Per-sample metrics
        # ----------------------------------------------------

        sample_ids = batch["sample_id"]

        hdr_values = hdr_mae.detach().cpu().tolist()
        sine_values = sine_mae.detach().cpu().tolist()
        cosine_values = cosine_mae.detach().cpu().tolist()
        wrapped_values = wrapped_mae.detach().cpu().tolist()
        wrapped_direct_values = (
            wrapped_direct_mae.detach().cpu().tolist()
        )
        absolute_values = (
            absolute_mae.detach().cpu().tolist()
        )

        for i, sample_id in enumerate(sample_ids):
            per_sample_results.append(
                {
                    "sample_id": str(sample_id),
                    "hdr_mae": float(hdr_values[i]),
                    "sine_mae": float(sine_values[i]),
                    "cosine_mae": float(cosine_values[i]),
                    "wrapped_phase_mae": float(
                        wrapped_values[i]
                    ),
                    "wrapped_phase_direct_mae": float(
                        wrapped_direct_values[i]
                    ),
                    "absolute_phase_mae": float(
                        absolute_values[i]
                    ),
                }
            )

    summary = {
        key: summarize(values)
        for key, values
        in results.items()
    }

    return summary, per_sample_results


@torch.inference_mode()
def evaluate_gt_self_check(
    loader,
    device,
):
    per_sample_results = []

    all_mae = []
    all_rmse = []
    all_max_error = []

    total_valid_pixels = 0
    total_cycle_jump_pixels = 0
    total_nonfinite_wrapped = 0

    global_wrapped_min = float("inf")
    global_wrapped_max = float("-inf")

    two_pi = 2.0 * np.pi
    cycle_tolerance = 1e-3

    for batch in tqdm(
        loader,
        desc="GT self-check",
    ):
        sample_ids = batch["sample_id"]

        sc_gt = batch["sc_gt"].to(
            device,
            non_blocking=True,
        )

        absolute_gt = (
            batch["absolute_phase"]
            .to(
                device,
                non_blocking=True,
            )
            .double()
        )

        # ----------------------------------------------------
        # S/C GT -> wrapped phase
        # ----------------------------------------------------

        wrapped_gt = sc_to_wrapped_phase(
            sc_gt
        )

        finite_wrapped = torch.isfinite(
            wrapped_gt
        )

        total_nonfinite_wrapped += (
            (~finite_wrapped).sum().item()
        )

        if finite_wrapped.any():
            global_wrapped_min = min(
                global_wrapped_min,
                wrapped_gt[
                    finite_wrapped
                ].min().item(),
            )

            global_wrapped_max = max(
                global_wrapped_max,
                wrapped_gt[
                    finite_wrapped
                ].max().item(),
            )

        # ----------------------------------------------------
        # S/C GT -> reconstructed absolute phase
        # ----------------------------------------------------

        absolute_reconstructed = (
            sc_to_absolute_phase(
                sc_gt
            )
        )

        error_signed = (
            absolute_reconstructed
            - absolute_gt
        )

        error_abs = torch.abs(
            error_signed
        )

        valid = (
            torch.isfinite(
                absolute_reconstructed
            )
            & torch.isfinite(
                absolute_gt
            )
        )

        # ----------------------------------------------------
        # Detect integer 2pi fringe-order jumps
        # ----------------------------------------------------

        cycle_index = torch.round(
            error_signed
            / two_pi
        )

        cycle_residual = torch.abs(
            error_signed
            - cycle_index * two_pi
        )

        cycle_jump_mask = (
            valid
            & (cycle_index != 0)
            & (
                cycle_residual
                < cycle_tolerance
            )
        )

        # ----------------------------------------------------
        # Per-sample statistics
        # ----------------------------------------------------

        for i, sample_id in enumerate(
            sample_ids
        ):
            valid_i = valid[i]

            if not valid_i.any():
                raise RuntimeError(
                    f"Sample {sample_id}: "
                    "no valid phase pixels."
                )

            error_i = error_abs[i][
                valid_i
            ]

            mae = error_i.mean().item()

            rmse = torch.sqrt(
                torch.mean(
                    error_i ** 2
                )
            ).item()

            max_error = error_i.max().item()

            cycle_jump_count = (
                cycle_jump_mask[i]
                .sum()
                .item()
            )

            valid_pixel_count = (
                valid_i.sum().item()
            )

            max_cycle_jump = 0

            if cycle_jump_count > 0:
                max_cycle_jump = int(
                    torch.abs(
                        cycle_index[i][
                            cycle_jump_mask[i]
                        ]
                    )
                    .max()
                    .item()
                )

            per_sample_results.append(
                {
                    "sample_id":
                        str(sample_id),

                    "absolute_phase_mae":
                        float(mae),

                    "absolute_phase_rmse":
                        float(rmse),

                    "absolute_phase_max_error":
                        float(max_error),

                    "valid_pixels":
                        int(valid_pixel_count),

                    "cycle_jump_pixels":
                        int(cycle_jump_count),

                    "max_cycle_jump":
                        int(max_cycle_jump),
                }
            )

            all_mae.append(mae)
            all_rmse.append(rmse)
            all_max_error.append(
                max_error
            )

            total_valid_pixels += (
                valid_pixel_count
            )

            total_cycle_jump_pixels += (
                cycle_jump_count
            )

    # --------------------------------------------------------
    # Find worst sample
    # --------------------------------------------------------

    worst_sample = max(
        per_sample_results,
        key=lambda x:
            x["absolute_phase_mae"],
    )

    summary = {
        "sample_count":
            len(per_sample_results),

        "absolute_phase_mae":
            summarize(all_mae),

        "absolute_phase_rmse":
            summarize(all_rmse),

        "absolute_phase_max_error":
            summarize(all_max_error),

        "wrapped_phase_min":
            float(global_wrapped_min),

        "wrapped_phase_max":
            float(global_wrapped_max),

        "wrapped_nonfinite_pixels":
            int(
                total_nonfinite_wrapped
            ),

        "valid_pixels":
            int(total_valid_pixels),

        "cycle_jump_pixels":
            int(
                total_cycle_jump_pixels
            ),

        "cycle_jump_ratio":
            float(
                total_cycle_jump_pixels
                / max(
                    total_valid_pixels,
                    1,
                )
            ),

        "worst_sample":
            worst_sample,
    }

    return (
        summary,
        per_sample_results,
    )

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HDRSL on a deterministic "
            "train/val/test split."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--gt-self-check",
        action="store_true",
        help=(
            "Bypass the network and verify "
            "S/C GT -> Absolute Phase GT."
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "dataset",
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--amp",
        action="store_true",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if (
        not args.gt_self_check
        and args.checkpoint is None
    ):
        parser.error(
            "--checkpoint is required "
            "unless --gt-self-check is used."
        )

    # --------------------------------------------------------
    # Dataset / deterministic split
    # --------------------------------------------------------

    full_dataset = HDRSLDataset(
        args.dataset_root
    )

    splits = split_sample_ids(
        full_dataset.sample_ids,
        seed=args.split_seed,
    )

    sample_ids = splits[
        args.split
    ]

    if args.max_samples is not None:
        sample_ids = sample_ids[
            :args.max_samples
        ]

    dataset = HDRSLDataset(
        args.dataset_root,
        sample_ids=sample_ids,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(
        f"Dataset total : "
        f"{len(full_dataset)}"
    )

    print(
        "Split sizes   : "
        f"train={len(splits['train'])}, "
        f"val={len(splits['val'])}, "
        f"test={len(splits['test'])}"
    )

    print(
        f"Evaluating    : "
        f"{args.split} "
        f"({len(dataset)} samples)"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu_id}"
        )
    else:
        device = torch.device("cpu")

    use_amp = (
        args.amp
        and device.type == "cuda"
    )

    print(f"Device        : {device}")
    print(f"AMP           : {use_amp}")

        # ========================================================
    # GT self-check mode
    # ========================================================

    if args.gt_self_check:
        print(
            "\nRunning GT self-check:"
        )

        print(
            "S/C GT -> Wrapped Phase "
            "-> Absolute Phase"
        )

        (
            gt_metrics,
            gt_per_sample,
        ) = evaluate_gt_self_check(
            loader,
            device,
        )

        print(
            "\n=== HDRSL GT Self-Check ==="
        )

        phase_mae = (
            gt_metrics[
                "absolute_phase_mae"
            ]
        )

        phase_rmse = (
            gt_metrics[
                "absolute_phase_rmse"
            ]
        )

        print(
            "Absolute Phase MAE     : "
            f"{phase_mae['mean']:.10f} "
            f"± {phase_mae['std']:.10f}"
        )

        print(
            "Absolute Phase RMSE    : "
            f"{phase_rmse['mean']:.10f} "
            f"± {phase_rmse['std']:.10f}"
        )

        print(
            "Wrapped phase range    : "
            f"["
            f"{gt_metrics['wrapped_phase_min']:.6f}, "
            f"{gt_metrics['wrapped_phase_max']:.6f}"
            f"]"
        )

        print(
            "Wrapped non-finite     : "
            f"{gt_metrics['wrapped_nonfinite_pixels']}"
        )

        print(
            "2pi cycle-jump pixels  : "
            f"{gt_metrics['cycle_jump_pixels']}"
        )

        print(
            "2pi cycle-jump ratio   : "
            f"{gt_metrics['cycle_jump_ratio']:.10e}"
        )

        worst = (
            gt_metrics[
                "worst_sample"
            ]
        )

        print(
            "Worst sample           : "
            f"{worst['sample_id']} "
            f"(MAE="
            f"{worst['absolute_phase_mae']:.10f})"
        )

        # ----------------------------------------------------
        # Save GT self-check JSON
        # ----------------------------------------------------

        if args.output is None:
            output = (
                ROOT
                / "outputs"
                / "evaluation"
                / (
                    f"{args.split}"
                    "_gt_self_check.json"
                )
            )
        else:
            output = args.output

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "dataset_root":
                str(args.dataset_root),

            "split":
                args.split,

            "split_seed":
                args.split_seed,

            "sample_count":
                len(dataset),

            "gt_self_check":
                gt_metrics,

            "per_sample":
                gt_per_sample,
        }

        with output.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                payload,
                f,
                indent=2,
            )

        # ----------------------------------------------------
        # Save per-sample CSV
        # ----------------------------------------------------

        csv_output = output.with_name(
            output.stem
            + "_per_sample.csv"
        )

        fieldnames = [
            "sample_id",
            "absolute_phase_mae",
            "absolute_phase_rmse",
            "absolute_phase_max_error",
            "valid_pixels",
            "cycle_jump_pixels",
            "max_cycle_jump",
        ]

        with csv_output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                gt_per_sample
            )

        print(
            f"\nSaved GT self-check to: "
            f"{output}"
        )

        print(
            "Saved GT per-sample to: "
            f"{csv_output}"
        )

        return

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    student = UNet_attention(
        8,
        4,
        False,
    ).to(device)

    phase_net = UNet(
        12,
        8,
        False,
    ).to(device)

    load_checkpoint(
        args.checkpoint,
        student,
        phase_net,
    )

    print(
        f"Checkpoint    : "
        f"{args.checkpoint}"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    metrics, per_sample_results = evaluate(
        student,
        phase_net,
        loader,
        device,
        use_amp,
    )

    print(
        "\n=== HDRSL Evaluation ==="
    )

    display_names = {
        "hdr_mae":
            "HDR MAE",

        "sine_mae":
            "Sine MAE",

        "cosine_mae":
            "Cosine MAE",

        "wrapped_phase_mae":
            "Wrapped Phase MAE",

        "wrapped_phase_direct_mae":
            "Wrapped Direct MAE",

        "absolute_phase_mae":
            "Absolute Phase MAE",
    }

    for key, metric in metrics.items():
        print(
            f"{display_names[key]:22s}: "
            f"{metric['mean']:.6f} "
            f"± {metric['std']:.6f}"
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    if args.output is None:
        output = (
            ROOT
            / "outputs"
            / "evaluation"
            / f"{args.split}_metrics.json"
        )
    else:
        output = args.output

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "split_seed": args.split_seed,
        "sample_count": len(dataset),
        "metrics": metrics,
        "per_sample": per_sample_results,
    }

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print(
        f"\nSaved metrics to: "
        f"{output}"
    )

    csv_output = output.with_name(
        output.stem + "_per_sample.csv"
        )

    fieldnames = [
        "sample_id",
        "hdr_mae",
        "sine_mae",
        "cosine_mae",
        "wrapped_phase_mae",
        "wrapped_phase_direct_mae",
        "absolute_phase_mae",
    ]

    with csv_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            per_sample_results
        )

    print(
        f"Saved per-sample metrics to: "
        f"{csv_output}"
    )


if __name__ == "__main__":
    main()