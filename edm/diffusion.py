"""Forward noising, the ideal denoiser, and the score for a Gaussian mixture."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from edm.mixture import GaussianMixture


def _noise_column(sigma: float | Tensor, n: int) -> Tensor:
    """Turn sigma into a float64 column for broadcasting onto plane points."""
    if isinstance(sigma, float):
        return torch.tensor([[sigma]], dtype=torch.float64)
    if sigma.dim() == 0:
        return sigma.to(dtype=torch.float64).reshape(1, 1)
    if sigma.dim() == 1:
        if sigma.shape[0] != n:
            raise ValueError(
                f"sigma length {sigma.shape[0]} must match batch size n={n}."
            )
        return sigma.to(dtype=torch.float64).reshape(n, 1)
    raise ValueError("sigma must be a float or shape (n,) tensor.")


def _log_gaussian(offset: Tensor, chol_factor: Tensor) -> Tensor:
    """Log density at an offset from a Gaussian's centre, given the lower
    Cholesky factor of its covariance (the caller factors once and reuses it).
    """
    # Whiten the offset so squared length is the Mahalanobis term.
    whitened_offset = torch.linalg.solve_triangular(
        chol_factor, offset, upper=False
    )
    squared_mahalanobis = (whitened_offset**2).sum(dim=(-2, -1))
    # Log-determinant of a covariance is twice the sum of the logs of its
    # Cholesky factor's diagonal, and it measures how spread out the blob is.
    chol_diagonal = torch.diagonal(chol_factor, dim1=-2, dim2=-1)
    log_det = 2.0 * chol_diagonal.log().sum(dim=-1)
    plane_dimension = 2
    return -0.5 * (
        plane_dimension * math.log(2.0 * math.pi)
        + log_det
        + squared_mahalanobis
    )


def _posterior_parts(
    mixture: GaussianMixture, x: Tensor, sigma: float | Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Per blob: its responsibility for each point (n, K), its posterior
    mean for each point (n, K, 2), and the lower Cholesky factor of its
    widened covariance (1 or n, K, 2, 2), which posterior_covariance reuses
    so the denoiser alone never pays for the blob covariances.
    """
    batch_size = x.shape[0]
    noise_column = _noise_column(sigma, batch_size)
    # Extra trailing unit axes so noise variance aligns with (K, 2, 2) blocks.
    noise_variance = (noise_column**2)[..., None, None]
    identity = torch.eye(2, dtype=torch.float64, device=x.device)
    # Each blob's covariance after independent widening by the noise level.
    widened_covariance = (
        mixture.covariances.unsqueeze(0) + noise_variance * identity
    )
    means_column = mixture.means.unsqueeze(0).unsqueeze(-1)
    # Offsets measure how far each noisy point sits from each blob centre.
    offset_from_centre = x.unsqueeze(1).unsqueeze(-1) - means_column
    # One Cholesky factorisation per widened blob serves the density, the
    # posterior means, and the posterior covariances below.
    chol_factor = torch.linalg.cholesky(widened_covariance)
    log_blob_density = _log_gaussian(offset_from_centre, chol_factor)
    # Each blob's share of the blame: how likely it is that this noisy dot
    # came from that blob, counting both the blob's weight and how far the dot
    # sits from it once the blob is widened by the noise.
    log_weights = mixture.weights.log().unsqueeze(0)
    responsibility = torch.softmax(log_weights + log_blob_density, dim=-1)
    solved_offset = torch.cholesky_solve(offset_from_centre, chol_factor)
    correction = torch.matmul(mixture.covariances.unsqueeze(0), solved_offset)
    # Within one blob the best guess pulls the dot back toward the blob's
    # centre; the pull is strong when the noise is large compared with the
    # blob's own spread, and weak when it is small.
    posterior_mean = (means_column + correction).squeeze(-1)
    return responsibility, posterior_mean, chol_factor


def add_noise(
    x0: Tensor, sigma: float | Tensor, generator: torch.Generator
) -> Tensor:
    """Corrupt clean points with isotropic Gaussian noise of scale sigma."""
    batch_size = x0.shape[0]
    noise_scale = _noise_column(sigma, batch_size)
    standard_normal = torch.randn(
        x0.shape, generator=generator, dtype=x0.dtype, device=x0.device
    )
    # This is the whole forward process in this convention: the clean point
    # plus noise of size sigma. A sampler later runs it backwards, from
    # large sigma down to zero.
    return x0 + noise_scale * standard_normal


def ideal_denoiser(
    mixture: GaussianMixture, x: Tensor, sigma: float | Tensor
) -> Tensor:
    """Exact conditional mean of the clean point given a noisy observation."""
    responsibility, posterior_mean, _ = _posterior_parts(mixture, x, sigma)
    # The final guess blends each blob's posterior mean using responsibilities.
    return (responsibility.unsqueeze(-1) * posterior_mean).sum(dim=1)


def posterior_covariance(
    mixture: GaussianMixture, x: Tensor, sigma: float | Tensor
) -> Tensor:
    """Cov(x0 | x) for each noisy point, shape (n, 2, 2): the uncertainty
    about the clean point that remains after seeing the noisy one."""
    responsibility, posterior_mean, chol_factor = _posterior_parts(
        mixture, x, sigma
    )
    # What one blob alone still leaves uncertain after seeing the noisy point,
    # S - S C^-1 S, from the same Cholesky factor; it depends on the noise
    # level but not on where the point is. Shape (1 or n, K, 2, 2).
    blob_covariance = mixture.covariances.unsqueeze(0)
    blob_posterior_covariance = blob_covariance - blob_covariance @ (
        torch.cholesky_solve(blob_covariance, chol_factor)
    )
    # The blended posterior mean is the same weighted sum as the ideal denoiser.
    denoiser_mean = (responsibility.unsqueeze(-1) * posterior_mean).sum(dim=1)
    # Law of total covariance: average blob second moments minus the outer
    # product of the blended mean.
    mean_outer = posterior_mean[..., :, None] * posterior_mean[..., None, :]
    weighted_second_moment = (
        responsibility[..., None, None]
        * (blob_posterior_covariance + mean_outer)
    ).sum(dim=1)
    denoiser_outer = denoiser_mean[:, :, None] * denoiser_mean[:, None, :]
    return weighted_second_moment - denoiser_outer


def score(mixture: GaussianMixture, x: Tensor, sigma: float | Tensor) -> Tensor:
    """Score via Tweedie's formula linking denoising to log-density gradient."""
    batch_size = x.shape[0]
    denoised = ideal_denoiser(mixture, x, sigma)
    noise_variance = _noise_column(sigma, batch_size) ** 2
    # Tweedie's formula: the best guess of the clean point and the gradient
    # of the noisy log-density carry the same information about x_0.
    return (denoised - x) / noise_variance
