from __future__ import annotations

import torch


FREQUENCIES = (1, 4, 16, 64)
TWO_PI = 2.0 * torch.pi


def sc_to_wrapped_phase(
    sc: torch.Tensor,
) -> torch.Tensor:

    if sc.shape[-3] != 8:
        raise ValueError(
            f"Expected 8 S/C channels, "
            f"got shape {tuple(sc.shape)}."
        )

    # Phase unwrapping contains integer fringe-order decisions.
    # Use float64 for numerical stability near round() boundaries.
    sc = sc.to(torch.float64)

    channel_dim = sc.ndim - 3

    indices_s = torch.tensor(
        [0, 2, 4, 6],
        device=sc.device,
    )

    indices_c = torch.tensor(
        [1, 3, 5, 7],
        device=sc.device,
    )

    sine = torch.index_select(
        sc,
        channel_dim,
        indices_s,
    )

    cosine = torch.index_select(
        sc,
        channel_dim,
        indices_c,
    )

    wrapped = (
        -torch.atan2(sine, cosine)
        + torch.pi
    )

    return wrapped


def unwrap_multifrequency(
    wrapped: torch.Tensor,
    frequencies=FREQUENCIES,
) -> torch.Tensor:
    """
    Hierarchical temporal phase unwrapping.

    Frequency order:
        1 -> 4 -> 16 -> 64

    Parameters
    ----------
    wrapped:
        [B, 4, H, W] or [4, H, W]

    Returns
    -------
    absolute_phase:
        [B, H, W] or [H, W]
    """

    if wrapped.shape[-3] != len(frequencies):
        raise ValueError(
            f"Expected {len(frequencies)} "
            f"wrapped-phase channels, "
            f"got shape {tuple(wrapped.shape)}."
        )

    channel_dim = wrapped.ndim - 3

    phases = torch.unbind(
        wrapped,
        dim=channel_dim,
    )

    absolute = phases[0]

    for i in range(1, len(frequencies)):
        ratio = (
            frequencies[i]
            / frequencies[i - 1]
        )

        wrapped_high = phases[i]

        fringe_order = torch.round(
            (
                ratio * absolute
                - wrapped_high
            )
            / TWO_PI
        )

        absolute = (
            wrapped_high
            + TWO_PI * fringe_order
        )

    return absolute


def sc_to_absolute_phase(
    sc: torch.Tensor,
) -> torch.Tensor:
    """
    Full HDRSL phase conversion:

        S/C
         ↓
        -atan2(S,C) + pi
         ↓
        1 -> 4 -> 16 -> 64 unwrap
         ↓
        absolute phase
    """

    wrapped = sc_to_wrapped_phase(sc)

    return unwrap_multifrequency(
        wrapped
    )


def absolute_phase_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Mean absolute error over finite pixels.
    """

    valid = (
        torch.isfinite(prediction)
        & torch.isfinite(target)
    )

    if not torch.any(valid):
        raise ValueError(
            "No finite pixels available "
            "for phase MAE."
        )

    return torch.mean(
        torch.abs(
            prediction[valid]
            - target[valid]
        )
    )