"""EDM preconditioning (Karras et al., 2022, Elucidating the Design Space of
Diffusion-Based Generative Models): per-sigma coefficients, loss weights, and
network input scaling so training sees unit variance at every noise level."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from toy.mixture import GaussianMixture


class Coefficients(NamedTuple):
    """EDM's four per-sigma coefficients, each shaped like sigma."""

    c_skip: Tensor
    c_out: Tensor
    c_in: Tensor
    c_noise: Tensor


def sigma_data_of(mixture: GaussianMixture) -> float:
    """Root mean square of one coordinate of a clean point:
    sqrt(E|x0|^2 / 2) = sqrt(sum_k w_k (trace S_k + |mu_k|^2) / 2), in closed
    form from the parameters. EDM's coefficients assume this scale."""
    # Per-component trace(S_k) is the sum of diagonal entries, shape (K,).
    component_traces = mixture.covariances.diagonal(dim1=-2, dim2=-1).sum(
        dim=-1
    )
    # Each blob adds its centre's squared distance from the origin to E|x0|^2.
    mean_squared_norm = (mixture.means**2).sum(dim=-1)
    # E|x0|^2 over the mixture is the weighted sum over blobs.
    expected_radius_squared = (
        mixture.weights * (component_traces + mean_squared_norm)
    ).sum()
    # sigma_data is the per-coordinate RMS scale EDM uses in table 1.
    return math.sqrt((expected_radius_squared / 2.0).item())


def _raise_if_sigma_or_sigma_data_invalid(
    sigma: Tensor, sigma_data: float
) -> None:
    """Reject non-finite or non-positive sigma entries and sigma_data."""
    # sigma_data must be positive because EDM's formulas divide by its scale.
    if not math.isfinite(sigma_data) or sigma_data <= 0:
        raise ValueError(
            f"sigma_data must be finite and positive, got {sigma_data}."
        )
    # Any bad level breaks the unit-variance preconditioning at that draw.
    bad_mask = (~torch.isfinite(sigma)) | (sigma <= 0)
    if bad_mask.any():
        # First bad index is named so callers can locate corrupt training data.
        bad_index = int(torch.where(bad_mask)[0][0].item())
        raise ValueError(
            f"sigma at index {bad_index} must be finite and positive, "
            f"got {sigma[bad_index].item()}."
        )


def coefficients(sigma: Tensor, sigma_data: float) -> Coefficients:
    """c_skip = sd^2 / (s^2 + sd^2); c_out = s sd / sqrt(s^2 + sd^2);
    c_in = 1 / sqrt(s^2 + sd^2); c_noise = log(s) / 4 (EDM table 1)."""
    _raise_if_sigma_or_sigma_data_invalid(sigma, sigma_data)
    sigma_data_squared = sigma_data * sigma_data
    # s^2 + sd^2 is shared by skip, out, and in; kept as (n,) only.
    variance_sum = sigma.square() + sigma_data_squared
    sqrt_variance_sum = variance_sum.sqrt()
    # Each coefficient is one elementwise pass over the n noise levels.
    c_skip = sigma_data_squared / variance_sum
    c_out = sigma * sigma_data / sqrt_variance_sum
    c_in = 1.0 / sqrt_variance_sum
    c_noise = sigma.log() / 4.0
    return Coefficients(c_skip=c_skip, c_out=c_out, c_in=c_in, c_noise=c_noise)


def loss_weight(sigma: Tensor, sigma_data: float) -> Tensor:
    """lambda = (s^2 + sd^2) / (s sd)^2, which equals 1 / c_out^2, so the
    weighted loss is the plain squared error of F against its target."""
    _raise_if_sigma_or_sigma_data_invalid(sigma, sigma_data)
    # Dividing by c_out^2 undoes the network's output scale, so the weighted
    # loss is the plain squared error of F against a unit-variance target.
    sigma_data_squared = sigma_data * sigma_data
    variance_sum = sigma.square() + sigma_data_squared
    return variance_sum / (sigma * sigma_data).square()


def network_target(
    x0: Tensor, x: Tensor, sigma: Tensor, sigma_data: float
) -> Tensor:
    """What F must output for D to return x0: (x0 - c_skip x) / c_out."""
    # Batched x0 and x must align before any per-sigma broadcast in training.
    if x0.shape != x.shape:
        raise ValueError(
            f"x0 shape {tuple(x0.shape)} must match x shape "
            f"{tuple(x.shape)}."
        )
    coef = coefficients(sigma, sigma_data)
    # (n,) coefficients widen to (n, 1) for broadcasting onto plane points.
    c_skip_column = coef.c_skip.unsqueeze(-1)
    c_out_column = coef.c_out.unsqueeze(-1)
    # The part of x0 the skip path cannot supply, rescaled to unit variance.
    return (x0 - c_skip_column * x) / c_out_column


def denoiser_from_network(
    network_output: Tensor, x: Tensor, sigma: Tensor, sigma_data: float
) -> Tensor:
    """D = c_skip x + c_out F."""
    coef = coefficients(sigma, sigma_data)
    # Same (n, 1) broadcast pattern as network_target for EDM's denoiser sum.
    c_skip_column = coef.c_skip.unsqueeze(-1)
    c_out_column = coef.c_out.unsqueeze(-1)
    # The skip path passes the noisy point through; the network adds the rest.
    return c_skip_column * x + c_out_column * network_output


def sample_sigma(
    n: int,
    generator: torch.Generator,
    *,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> Tensor:
    """Training noise levels: log sigma ~ Normal(p_mean, p_std) (EDM table 1),
    shape (n,), float64."""
    # Log-normal noise levels concentrate training where denoising is hardest.
    log_sigma = p_mean + p_std * torch.randn(
        n, generator=generator, dtype=torch.float64
    )
    # exp maps log sigma to strictly positive sigma, shape (n,) float64.
    return log_sigma.exp()
