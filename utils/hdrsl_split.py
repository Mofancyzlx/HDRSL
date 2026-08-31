from __future__ import annotations

from typing import Sequence

import torch


def split_sample_ids(
    sample_ids: Sequence[str],
    seed: int = 0,
) -> dict[str, list[str]]:
    """
    Deterministic 8:1:1 split.

    For the released 1700-sample metal dataset:

        train = 1360
        val   = 170
        test  = 170

    The implementation follows the behavior of the
    official code's random_split(... manual_seed(0)).
    """

    sample_ids = list(sample_ids)

    n_total = len(sample_ids)

    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    generator = torch.Generator()
    generator.manual_seed(seed)

    permutation = torch.randperm(
        n_total,
        generator=generator,
    ).tolist()

    train_indices = permutation[:n_train]

    val_indices = permutation[
        n_train:n_train + n_val
    ]

    test_indices = permutation[
        n_train + n_val:
    ]

    return {
        "train": [
            sample_ids[i]
            for i in train_indices
        ],
        "val": [
            sample_ids[i]
            for i in val_indices
        ],
        "test": [
            sample_ids[i]
            for i in test_indices
        ],
    }