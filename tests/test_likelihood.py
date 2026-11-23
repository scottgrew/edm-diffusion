"""Probability-flow likelihood against the exact density and a divergence."""

import pytest
import torch
from torch import Tensor

from edm.diffusion import ideal_denoiser
from edm.likelihood import drift_and_divergence, log_likelihood
from edm.mixture import THREE_BLOBS, GaussianMixture
from edm.sample import Denoise, karras_sigmas


def _ideal_denoise_for(mixture: GaussianMixture) -> Denoise:
    def denoise(x: Tensor, sigma: Tensor) -> Tensor:
        return ideal_denoiser(mixture, x, sigma)

    return denoise


def _echo_denoise(x: Tensor, _sigma: Tensor) -> Tensor:
    return x


def _linear_divergence_oracle(covariance: Tensor, sigma: float) -> float:
    """Closed-form div for a single-blob linear ideal denoiser at noise s."""
    plane_dimension = 2
    identity = torch.eye(plane_dimension, dtype=torch.float64)
    widened = covariance + (sigma**2) * identity
    kalman_gain = covariance @ torch.linalg.inv(widened)
    trace_term = torch.trace(identity - kalman_gain)
    return (trace_term / sigma).item()


def test_log_likelihood_matches_exact_density() -> None:
    """Ideal denoiser: integrated log p matches the mixture at sigma 0.002."""
    generator = torch.Generator().manual_seed(0)
    low_noise = THREE_BLOBS.noised(0.002)
    points = low_noise.sample(2_000, generator)
    denoise = _ideal_denoise_for(THREE_BLOBS)
    high_noise = THREE_BLOBS.noised(80.0)
    prior = high_noise.log_prob
    exact = low_noise.log_prob(points)

    def max_error(step_count: int) -> float:
        sigmas = karras_sigmas(step_count).flip(0)[1:]
        estimated = log_likelihood(denoise, points, sigmas, prior)
        return (estimated - exact).abs().max().item()

    error_sixty_four = max_error(64)
    error_one_twenty_eight = max_error(128)
    assert error_one_twenty_eight < 0.03
    ratio = error_sixty_four / error_one_twenty_eight
    assert 3.5 <= ratio <= 5.0


def test_divergence_matches_linear_closed_form() -> None:
    """Single blob: autograd div matches trace(I - S(S + s^2 I)^-1) / s."""
    single_blob = GaussianMixture(
        weights=torch.tensor([1.0], dtype=torch.float64),
        means=torch.tensor([[1.0, -2.0]], dtype=torch.float64),
        covariances=torch.tensor(
            [[[0.5, 0.2], [0.2, 0.3]]], dtype=torch.float64
        ),
    )
    generator = torch.Generator().manual_seed(1)
    points = torch.randn(64, 2, generator=generator, dtype=torch.float64)
    sigma = 4.0
    denoise = _ideal_denoise_for(single_blob)
    _, computed = drift_and_divergence(denoise, points, sigma)
    oracle_value = _linear_divergence_oracle(single_blob.covariances[0], sigma)
    oracle = torch.full_like(computed, oracle_value)
    assert torch.allclose(computed, oracle, rtol=0.0, atol=1e-12)


def _zero_prior(batch: Tensor) -> Tensor:
    return torch.zeros(batch.shape[0], dtype=torch.float64)


def test_log_likelihood_validation_raises() -> None:
    """Strictly increasing, positive sigmas are required for likelihood."""
    points = torch.zeros(4, 2, dtype=torch.float64)
    decreasing = karras_sigmas(18)[:-1]
    with pytest.raises(ValueError, match="strictly increasing"):
        log_likelihood(_echo_denoise, points, decreasing, _zero_prior)
    non_positive = torch.tensor([-0.01, 0.002, 1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="positive"):
        log_likelihood(_echo_denoise, points, non_positive, _zero_prior)
    at_most_one = torch.tensor([0.002], dtype=torch.float64)
    with pytest.raises(ValueError, match="length at least 2"):
        log_likelihood(_echo_denoise, points, at_most_one, _zero_prior)
