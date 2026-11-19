"""Denoising loss: squared error between a prediction and the clean point."""

from __future__ import annotations

from torch import Tensor


def denoising_loss(denoised: Tensor, x0: Tensor) -> Tensor:
    """Squared error per point, shape (n,): the loss a denoiser is trained
    on. Over all functions its expectation is smallest for E[x0 | x], and
    that smallest value is the expected trace of Cov(x0 | x)."""
    if denoised.shape != x0.shape:
        raise ValueError(
            f"denoised shape {tuple(denoised.shape)} must match "
            f"x0 shape {tuple(x0.shape)}."
        )
    # Sum squared coordinate errors so each point gets one scalar loss value.
    return (denoised - x0).square().sum(dim=-1)
