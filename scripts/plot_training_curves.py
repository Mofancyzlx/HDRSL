from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to train_log.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output figures",
    )

    return parser.parse_args()


def load_records(log_path):
    records = []

    with log_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def plot_training_losses(
    records,
    output_dir,
):
    epochs = [
        r["epoch"]
        for r in records
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        [
            r["train"]["total_loss"]
            for r in records
        ],
        label="Total",
    )

    plt.plot(
        epochs,
        [
            r["train"]["hdr_output_loss"]
            for r in records
        ],
        label="HDR",
    )

    plt.plot(
        epochs,
        [
            r["train"]["distillation_loss"]
            for r in records
        ],
        label="Distillation",
    )

    plt.plot(
        epochs,
        [
            r["train"]["phase_loss"]
            for r in records
        ],
        label="Phase",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = (
        output_dir
        / "training_losses.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_validation_metrics(
    records,
    output_dir,
):
    epochs = [
        r["epoch"]
        for r in records
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        [
            r["val"]["hdr_mae"]
            for r in records
        ],
        label="HDR MAE",
    )

    plt.plot(
        epochs,
        [
            r["val"]["sc_mae"]
            for r in records
        ],
        label="S/C MAE",
    )

    plt.plot(
        epochs,
        [
            r["val"]["wrapped_phase_mae"]
            for r in records
        ],
        label="Wrapped Phase MAE",
    )

    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = (
        output_dir
        / "validation_metrics.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved: {output_path}")


def main():
    args = parse_args()

    if not args.log.exists():
        raise FileNotFoundError(
            f"Log file not found: {args.log}"
        )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.log.parent
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_records(
        args.log
    )

    if not records:
        raise RuntimeError(
            "No training records found."
        )

    print(
        f"Loaded {len(records)} epochs"
    )

    plot_training_losses(
        records,
        output_dir,
    )

    plot_validation_metrics(
        records,
        output_dir,
    )


if __name__ == "__main__":
    main()