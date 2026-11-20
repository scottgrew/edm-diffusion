"""Classifier guidance via exact mixture responsibilities."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from toy.diffusion import _noise_column, _posterior_parts, ideal_denoiser
from toy.mixture import GaussianMixture
from toy.sample import Denoise


def _validate_blob_index(mixture: GaussianMixture, blob: int) -> None:
    """Reject blob indices outside the mixture's component list."""
    component_count = mixture.weights.shape[0]
    if blob < 0 or blob >= component_count:
        raise ValueError(
            f"blob index {blob} out of range for K={component_count} "
            "components."
        )


def _log_density_gradients(
    mixture: GaussianMixture, x: Tensor, sigma: float | Tensor
) -> tuple[Tensor, Tensor]:
    """Per-blob grad log density and responsibilities at noisy points x."""
    responsibility, _, chol_factor = _posterior_parts(mixture, x, sigma)
    means_column = mixture.means.unsqueeze(0).unsqueeze(-1)
    offset_from_centre = x.unsqueeze(1).unsqueeze(-1) - means_column
    # Blob j's score is minus inverse covariance times offset from its mean.
    gradient = -torch.cholesky_solve(offset_from_centre, chol_factor).squeeze(
        -1
    )
    return responsibility, gradient


def log_responsibility_gradient(
    mixture: GaussianMixture, x: Tensor, sigma: float | Tensor, blob: int
) -> Tensor:
    """grad_x log r_k(x) for the noised mixture, shape (n, 2), in closed
    form: g_k - sum_j r_j g_j with g_j = -(S_j + s^2 I)^-1 (x - mu_j), the
    gradient of blob j's log density. One Cholesky factor per blob."""
    _validate_blob_index(mixture, blob)
    responsibility, gradient = _log_density_gradients(mixture, x, sigma)
    # Classifier log-probability gradient reweights blob scores by r_j.
    target_gradient = gradient[:, blob]
    mixture_gradient = (responsibility.unsqueeze(-1) * gradient).sum(dim=1)
    return target_gradient - mixture_gradient


def guided_denoiser(
    mixture: GaussianMixture, blob: int, scale: float
) -> Denoise:
    """A Denoise (x, sigma (n,)) -> (n, 2) whose score is
    score(x) + scale * grad log r_blob(x): D = x + sigma^2 times that."""
    _validate_blob_index(mixture, blob)
    if not math.isfinite(scale):
        raise ValueError(f"scale must be finite, got {scale}.")

    def denoise(x: Tensor, sigma: Tensor) -> Tensor:
        gradient = log_responsibility_gradient(mixture, x, sigma, blob)
        unguided = ideal_denoiser(mixture, x, sigma)
        noise_variance = _noise_column(sigma, x.shape[0]) ** 2
        # Tweedie: bump the denoiser by sigma^2 times the guidance gradient.
        return unguided + noise_variance * scale * gradient

    return denoise
