from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import SSIM, MseDirectionLoss
from unet import UNet, UNet_attention

from utils.hdrsl_dataset import HDRSLDataset
from utils.hdrsl_split import split_sample_ids
from utils.phase import (
    sc_to_wrapped_phase,
    sc_to_absolute_phase,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serialize_args(args):
    result = {}

    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value

    return result


# ============================================================
# Loss
# ============================================================

def hdr_output_loss(
    prediction,
    target,
    mse,
    mae,
    ssim,
):
    """
    Paper Eq. (14):

        Loutput =
            MAE
            + sqrt(MSE)
            + (1 - SSIM)
    """

    l2 = torch.sqrt(
        mse(prediction, target)
    )

    l1 = mae(
        prediction,
        target,
    )

    ssim_value = ssim(
        prediction,
        target,
    )

    total = l1 + l2 + ssim_value

    return total, l1, l2, ssim_value


def sc_output_loss(
    prediction,
    target,
    mse,
    mae,
):
    """
    Phase Calculation Module:
        sqrt(MSE) + MAE
    """

    l2 = torch.sqrt(
        mse(prediction, target)
    )

    l1 = mae(
        prediction,
        target,
    )

    return l1 + l2, l1, l2


# ============================================================
# Validation
# ============================================================

def circular_phase_error(
    prediction,
    target,
):
    delta = prediction - target

    return torch.abs(
        torch.atan2(
            torch.sin(delta),
            torch.cos(delta),
        )
    )


@torch.inference_mode()
def validate(
    student,
    phase_net,
    loader,
    device,
    use_amp,
):
    student.eval()
    phase_net.eval()

    hdr_values = []
    sc_values = []
    sine_values = []
    cosine_values = []
    wrapped_values = []
    absolute_values = []

    for batch in tqdm(
        loader,
        desc="Validation",
        leave=False,
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
            .double()
        )

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

            sc_pred, phase_features = (
                phase_net(phase_input)
            )

            del phase_features

        hdr_pred = hdr_pred.float()
        sc_pred = sc_pred.float()

        # ----------------------------------------------------
        # HDR
        # ----------------------------------------------------

        hdr_mae = (
            torch.abs(
                hdr_pred - hdr_gt
            )
            .flatten(1)
            .mean(dim=1)
        )

        # ----------------------------------------------------
        # S / C
        # ----------------------------------------------------

        sc_mae = (
            torch.abs(
                sc_pred - sc_gt
            )
            .flatten(1)
            .mean(dim=1)
        )

        sine_mae = (
            torch.abs(
                sc_pred[:, 0::2]
                - sc_gt[:, 0::2]
            )
            .flatten(1)
            .mean(dim=1)
        )

        cosine_mae = (
            torch.abs(
                sc_pred[:, 1::2]
                - sc_gt[:, 1::2]
            )
            .flatten(1)
            .mean(dim=1)
        )

        # ----------------------------------------------------
        # Wrapped phase
        # ----------------------------------------------------

        wrapped_pred = (
            sc_to_wrapped_phase(
                sc_pred
            )
        )

        wrapped_gt = (
            sc_to_wrapped_phase(
                sc_gt
            )
        )

        wrapped_mae = (
            circular_phase_error(
                wrapped_pred,
                wrapped_gt,
            )
            .flatten(1)
            .mean(dim=1)
        )

        # ----------------------------------------------------
        # Absolute phase
        # ----------------------------------------------------

        absolute_pred = (
            sc_to_absolute_phase(
                sc_pred
            )
        )

        for i in range(
            absolute_pred.shape[0]
        ):
            valid = (
                torch.isfinite(
                    absolute_pred[i]
                )
                & torch.isfinite(
                    absolute_gt[i]
                )
            )

            phase_mae = torch.mean(
                torch.abs(
                    absolute_pred[i][valid]
                    - absolute_gt[i][valid]
                )
            )

            absolute_values.append(
                phase_mae.item()
            )

        hdr_values.extend(
            hdr_mae.cpu().tolist()
        )

        sc_values.extend(
            sc_mae.cpu().tolist()
        )

        sine_values.extend(
            sine_mae.cpu().tolist()
        )

        cosine_values.extend(
            cosine_mae.cpu().tolist()
        )

        wrapped_values.extend(
            wrapped_mae.cpu().tolist()
        )

    return {
        "hdr_mae":
            float(np.mean(hdr_values)),

        "sc_mae":
            float(np.mean(sc_values)),

        "sine_mae":
            float(np.mean(sine_values)),

        "cosine_mae":
            float(np.mean(cosine_values)),

        "wrapped_phase_mae":
            float(np.mean(wrapped_values)),

        "absolute_phase_mae":
            float(np.mean(absolute_values)),
    }


# ============================================================
# Teacher warm-up
# ============================================================

def train_teacher_pass(
    teacher,
    loader,
    optimizer,
    scaler,
    mse,
    mae,
    ssim,
    device,
    use_amp,
):
    teacher.train()

    total_loss = 0.0

    for batch in tqdm(
        loader,
        desc="Teacher warm-up",
        leave=False,
    ):
        hdr_gt = batch["hdr_gt"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            teacher_pred, features = (
                teacher(hdr_gt)
            )

            del features

            (
                teacher_loss,
                _,
                _,
                _,
            ) = hdr_output_loss(
                teacher_pred,
                hdr_gt,
                mse,
                mae,
                ssim,
            )

        scaler.scale(
            teacher_loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += (
            teacher_loss.item()
        )

    return total_loss / len(loader)


# ============================================================
# Main training epoch
# ============================================================

def train_epoch(
    epoch,
    student,
    teacher,
    phase_net,
    loader,
    optimizer,
    teacher_optimizer,
    scaler,
    mse,
    mae,
    ssim,
    distillation_loss_fn,
    device,
    use_amp,
    teacher_warmup_epochs,
):
    student.train()
    teacher.train()
    phase_net.train()

    running = {
        "total_loss": 0.0,
        "hdr_output_loss": 0.0,
        "distillation_loss": 0.0,
        "phase_loss": 0.0,
        "teacher_loss": 0.0,
    }

    num_batches = 0

    for batch in tqdm(
        loader,
        desc=f"Train epoch {epoch}",
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

        optimizer.zero_grad(
            set_to_none=True
        )

        if epoch > teacher_warmup_epochs:
            teacher_optimizer.zero_grad(
                set_to_none=True
            )

        with autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            # =================================================
            # Student HDR Generation
            # =================================================

            (
                hdr_pred,
                student_features,
            ) = student(ldr)

            # =================================================
            # Teacher
            #
            # Official released training logic:
            # - epoch <= 10:
            #       teacher separately warmed-up
            #       but not updated in main student loop
            #
            # - epoch > 10:
            #       teacher participates in joint update
            # =================================================

            if epoch <= teacher_warmup_epochs:
                with torch.no_grad():
                    (
                        teacher_pred,
                        teacher_features,
                    ) = teacher(hdr_gt)

                teacher_loss = None

            else:
                (
                    teacher_pred,
                    teacher_features,
                ) = teacher(hdr_gt)

                (
                    teacher_loss,
                    _,
                    _,
                    _,
                ) = hdr_output_loss(
                    teacher_pred,
                    hdr_gt,
                    mse,
                    mae,
                    ssim,
                )

            # =================================================
            # HDR reconstruction loss
            # =================================================

            (
                hdr_loss,
                _,
                _,
                _,
            ) = hdr_output_loss(
                hdr_pred,
                hdr_gt,
                mse,
                mae,
                ssim,
            )

            # =================================================
            # Four-layer feature distillation
            # =================================================

            distill_loss = (
                distillation_loss_fn(
                    student_features,
                    teacher_features,
                )
            )

            # =================================================
            # Phase Calculation Network
            # =================================================

            phase_input = torch.cat(
                [hdr_pred, ldr],
                dim=1,
            )

            (
                sc_pred,
                phase_features,
            ) = phase_net(
                phase_input
            )

            del phase_features

            (
                phase_loss,
                _,
                _,
            ) = sc_output_loss(
                sc_pred,
                sc_gt,
                mse,
                mae,
            )

            # =================================================
            # Total
            # =================================================

            total_loss = (
                hdr_loss
                + distill_loss
                + phase_loss
            )

            if teacher_loss is not None:
                total_loss = (
                    total_loss
                    + teacher_loss
                )

        scaler.scale(
            total_loss
        ).backward()

        scaler.step(
            optimizer
        )

        if epoch > teacher_warmup_epochs:
            scaler.step(
                teacher_optimizer
            )

        scaler.update()

        running["total_loss"] += (
            total_loss.item()
        )

        running[
            "hdr_output_loss"
        ] += hdr_loss.item()

        running[
            "distillation_loss"
        ] += distill_loss.item()

        running[
            "phase_loss"
        ] += phase_loss.item()

        if teacher_loss is not None:
            running[
                "teacher_loss"
            ] += teacher_loss.item()

        num_batches += 1

    return {
        key: value / num_batches
        for key, value
        in running.items()
    }


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    epoch,
    student,
    teacher,
    phase_net,
    optimizer,
    teacher_optimizer,
    scheduler,
    teacher_scheduler,
    scaler,
    best_sc_mae,
    args,
    splits,
    train_generator,
):
    checkpoint = {
        "epoch":
            epoch,

        "student":
            student.state_dict(),

        "teacher":
            teacher.state_dict(),

        "phase_net":
            phase_net.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "teacher_optimizer":
            teacher_optimizer.state_dict(),

        "scheduler":
            scheduler.state_dict(),

        "teacher_scheduler":
            teacher_scheduler.state_dict(),

        "scaler":
            scaler.state_dict(),

        "best_sc_mae":
            best_sc_mae,

        "args":
            serialize_args(args),

        "splits":
            splits,

        "train_generator_state":
            train_generator.get_state(),
    }

    torch.save(
        checkpoint,
        path,
    )


def load_resume(
    path,
    student,
    teacher,
    phase_net,
    optimizer,
    teacher_optimizer,
    scheduler,
    teacher_scheduler,
    scaler,
    train_generator,
    splits,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint["splits"] != splits:
        raise ValueError(
            "Current train/val/test split does not "
            "match the resumed checkpoint."
        )

    student.load_state_dict(
        checkpoint["student"]
    )

    teacher.load_state_dict(
        checkpoint["teacher"]
    )

    phase_net.load_state_dict(
        checkpoint["phase_net"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    teacher_optimizer.load_state_dict(
        checkpoint["teacher_optimizer"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler"]
    )

    teacher_scheduler.load_state_dict(
        checkpoint[
            "teacher_scheduler"
        ]
    )

    scaler.load_state_dict(
        checkpoint["scaler"]
    )

    train_generator.set_state(
        checkpoint[
            "train_generator_state"
        ]
    )

    start_epoch = (
        int(checkpoint["epoch"]) + 1
    )

    best_sc_mae = float(
        checkpoint["best_sc_mae"]
    )

    return (
        start_epoch,
        best_sc_mae,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Formal HDRSL training."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "dataset",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "train_hdrsl"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--eta-min",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--teacher-warmup-epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--teacher-passes",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--distill-mse-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        "--resume",
        type=Path,
        default=None,
    )

    # Diagnostic / integration-test controls
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # ========================================================
    # Dataset / split
    # ========================================================

    full_dataset = HDRSLDataset(
        args.dataset_root
    )

    splits = split_sample_ids(
        full_dataset.sample_ids,
        seed=args.split_seed,
    )

    train_ids = splits["train"]
    val_ids = splits["val"]

    if args.max_train_samples is not None:
        train_ids = train_ids[
            :args.max_train_samples
        ]

    if args.max_val_samples is not None:
        val_ids = val_ids[
            :args.max_val_samples
        ]

    # For a diagnostic run, the effective split used by the
    # checkpoint must reflect the truncated IDs.
    effective_splits = {
        "train": train_ids,
        "val": val_ids,
        "test": splits["test"],
    }

    train_dataset = HDRSLDataset(
        args.dataset_root,
        sample_ids=train_ids,
    )

    val_dataset = HDRSLDataset(
        args.dataset_root,
        sample_ids=val_ids,
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # ========================================================
    # Device
    # ========================================================

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

    print(
        f"Device        : {device}"
    )

    print(
        f"AMP           : {use_amp}"
    )

    print(
        f"Dataset       : {len(full_dataset)}"
    )

    print(
        "Official split: "
        f"train={len(splits['train'])}, "
        f"val={len(splits['val'])}, "
        f"test={len(splits['test'])}"
    )

    print(
        "This run      : "
        f"train={len(train_dataset)}, "
        f"val={len(val_dataset)}"
    )

    # ========================================================
    # Networks
    # ========================================================

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
    ).to(device) # 8 通道 LDR images + 4 通道 HDR images = 12 通道输入，输出 8 通道的 S/C 图像

    # ========================================================
    # Loss
    # ========================================================

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    ssim = SSIM(
        device=device
    )

    distillation_loss_fn = (
        MseDirectionLoss(
            args.distill_mse_weight
        )
    )

    # ========================================================
    # Optimizers
    # ========================================================

    optimizer = torch.optim.Adam(
        list(student.parameters())
        + list(phase_net.parameters()),
        lr=args.learning_rate,
    )

    teacher_optimizer = (
        torch.optim.Adam(
            teacher.parameters(),
            lr=args.learning_rate,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.eta_min,
        )
    )

    teacher_scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            teacher_optimizer,
            T_max=args.epochs,
            eta_min=args.eta_min,
        )
    )

    scaler = GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # ========================================================
    # Output
    # ========================================================

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = (
        args.output_dir
        / "config.json"
    )

    split_path = (
        args.output_dir
        / "split.json"
    )

    log_path = (
        args.output_dir
        / "train_log.jsonl"
    )

    if (
        args.resume is None
        and log_path.exists()
    ):
        raise RuntimeError(
            f"{args.output_dir} already contains "
            "a training log. Use a new output "
            "directory or --resume."
        )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            serialize_args(args),
            f,
            indent=2,
        )

    with split_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            effective_splits,
            f,
            indent=2,
        )

    # ========================================================
    # Resume
    # ========================================================

    start_epoch = 1
    best_sc_mae = float("inf")

    if args.resume is not None:
        (
            start_epoch,
            best_sc_mae,
        ) = load_resume(
            args.resume,
            student,
            teacher,
            phase_net,
            optimizer,
            teacher_optimizer,
            scheduler,
            teacher_scheduler,
            scaler,
            train_generator,
            effective_splits,
        )

        print(
            f"Resume        : {args.resume}"
        )

        print(
            f"Start epoch   : {start_epoch}"
        )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        epoch_start = time.time()

        lr_used = (
            optimizer.param_groups[0]["lr"]
        )

        teacher_lr_used = (
            teacher_optimizer
            .param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # Official teacher warm-up schedule:
        #
        # first 10 epochs:
        #     3 full teacher passes / epoch
        # ----------------------------------------------------

        teacher_warmup_loss = None

        if (
            epoch
            <= args.teacher_warmup_epochs
        ):
            warmup_losses = []

            for pass_index in range(
                args.teacher_passes
            ):
                print(
                    f"\nEpoch {epoch} "
                    f"Teacher pass "
                    f"{pass_index + 1}/"
                    f"{args.teacher_passes}"
                )

                loss_value = (
                    train_teacher_pass(
                        teacher,
                        train_loader,
                        teacher_optimizer,
                        scaler,
                        mse,
                        mae,
                        ssim,
                        device,
                        use_amp,
                    )
                )

                warmup_losses.append(
                    loss_value
                )

            teacher_warmup_loss = (
                float(
                    np.mean(
                        warmup_losses
                    )
                )
            )

        # ----------------------------------------------------
        # Student + Phase (+ Teacher after warm-up)
        # ----------------------------------------------------

        train_metrics = train_epoch(
            epoch,
            student,
            teacher,
            phase_net,
            train_loader,
            optimizer,
            teacher_optimizer,
            scaler,
            mse,
            mae,
            ssim,
            distillation_loss_fn,
            device,
            use_amp,
            args.teacher_warmup_epochs,
        )

        # ----------------------------------------------------
        # Validation
        #
        # IMPORTANT:
        # test split is never used here.
        # ----------------------------------------------------

        val_metrics = validate(
            student,
            phase_net,
            val_loader,
            device,
            use_amp,
        )

        is_best = (
            val_metrics["sc_mae"]
            < best_sc_mae
        )

        if is_best:
            best_sc_mae = (
                val_metrics["sc_mae"]
            )

        # Scheduler advances once per epoch,
        # matching the released training code.
        scheduler.step()
        teacher_scheduler.step()

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        log_record = {
            "epoch":
                epoch,

            "lr":
                lr_used,

            "teacher_lr":
                teacher_lr_used,

            "teacher_warmup_loss":
                teacher_warmup_loss,

            "train":
                train_metrics,

            "val":
                val_metrics,

            "best_val_sc_mae":
                best_sc_mae,

            "is_best":
                is_best,

            "epoch_seconds":
                epoch_seconds,
        }

        with log_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    log_record
                )
                + "\n"
            )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        last_path = (
            args.output_dir
            / "last.pt"
        )

        save_checkpoint(
            last_path,
            epoch,
            student,
            teacher,
            phase_net,
            optimizer,
            teacher_optimizer,
            scheduler,
            teacher_scheduler,
            scaler,
            best_sc_mae,
            args,
            effective_splits,
            train_generator,
        )

        if is_best:
            best_path = (
                args.output_dir
                / "best.pt"
            )

            save_checkpoint(
                best_path,
                epoch,
                student,
                teacher,
                phase_net,
                optimizer,
                teacher_optimizer,
                scheduler,
                teacher_scheduler,
                scaler,
                best_sc_mae,
                args,
                effective_splits,
                train_generator,
            )

        print(
            f"\n=== Epoch {epoch} ==="
        )

        print(
            f"Train total       : "
            f"{train_metrics['total_loss']:.6f}"
        )

        print(
            f"Train HDR         : "
            f"{train_metrics['hdr_output_loss']:.6f}"
        )

        print(
            f"Train distill     : "
            f"{train_metrics['distillation_loss']:.6f}"
        )

        print(
            f"Train phase       : "
            f"{train_metrics['phase_loss']:.6f}"
        )

        print(
            f"Val HDR MAE       : "
            f"{val_metrics['hdr_mae']:.6f}"
        )

        print(
            f"Val S/C MAE       : "
            f"{val_metrics['sc_mae']:.6f}"
        )

        print(
            f"Val wrapped MAE   : "
            f"{val_metrics['wrapped_phase_mae']:.6f}"
        )

        print(
            f"Val absolute MAE  : "
            f"{val_metrics['absolute_phase_mae']:.6f}"
        )

        print(
            f"Best val S/C MAE  : "
            f"{best_sc_mae:.6f}"
        )

        print(
            f"Epoch time        : "
            f"{epoch_seconds:.1f}s"
        )


if __name__ == "__main__":
    main()