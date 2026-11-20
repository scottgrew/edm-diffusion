"""EDM schedules and samplers against closed-form and moment oracles."""

import math

import pytest
import torch
from torch import Tensor

from toy.diffusion import ideal_denoiser
from toy.mixture import THREE_BLOBS, GaussianMixture
from toy.sample import (
    Denoise,
    heun_sample,
    initial_noise,
    karras_sigmas,
    stochastic_sample,
)


def _mixture_total_covariance(mixture: GaussianMixture) -> Tensor:
    """Oracle: Cov(x) from weights, means, and component covariances."""
    weighted_second = (
        mixture.weights[..., None, None]
        * (
            mixture.covariances
            + mixture.means[..., :, None] * mixture.means[..., None, :]
        )
    ).sum(dim=0)
    mixture_mean = (mixture.weights.unsqueeze(-1) * mixture.means).sum(dim=0)
    mean_outer = mixture_mean[:, None] * mixture_mean[None, :]
    return weighted_second - mean_outer


def _sample_moment_errors(
    samples: Tensor, mixture: GaussianMixture
) -> tuple[Tensor, Tensor]:
    """Absolute mean error and max covariance error vs the mixture."""
    sample_mean = samples.mean(dim=0)
    # torch.cov wants one variable per row; shape (2, 2), not (n, n).
    sample_covariance = torch.cov(samples.T)
    analytic_mean = (mixture.weights.unsqueeze(-1) * mixture.means).sum(dim=0)
    analytic_covariance = _mixture_total_covariance(mixture)
    mean_error = (sample_mean - analytic_mean).abs()
    covariance_error = (sample_covariance - analytic_covariance).abs().max()
    return mean_error, covariance_error


def _responsibility_shares(samples: Tensor, mixture: GaussianMixture) -> Tensor:
    """Fraction of samples per blob by clean mixture responsibility."""
    component_count = mixture.weights.shape[0]
    log_density = torch.stack(
        [
            torch.distributions.MultivariateNormal(
                mixture.means[index], mixture.covariances[index]
            ).log_prob(samples)
            for index in range(component_count)
        ],
        dim=-1,
    )
    log_weights = mixture.weights.log()
    responsibility = torch.softmax(log_weights + log_density, dim=-1)
    assignments = responsibility.argmax(dim=-1)
    # Empirical weights are compared to the mixture's true blob proportions.
    counts = torch.bincount(assignments, minlength=component_count)
    return counts.to(dtype=torch.float64) / samples.shape[0]


def _probability_flow_closed_form(
    noisy_points: Tensor,
    sigma_from: float,
    sigma_to: float,
    mean: Tensor,
    covariance: Tensor,
) -> Tensor:
    """Oracle: one Gaussian blob along the probability-flow ODE."""
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

    def scale_matrix(noise_variance: float) -> Tensor:
        scaled_roots = torch.sqrt(eigenvalues + noise_variance)
        return eigenvectors @ torch.diag(scaled_roots) @ eigenvectors.T

    def inv_scale_matrix(noise_variance: float) -> Tensor:
        inverse_roots = 1.0 / torch.sqrt(eigenvalues + noise_variance)
        return eigenvectors @ torch.diag(inverse_roots) @ eigenvectors.T

    forward_factor = scale_matrix(sigma_to**2) @ inv_scale_matrix(sigma_from**2)
    centered = noisy_points - mean.unsqueeze(0)
    return mean.unsqueeze(0) + centered @ forward_factor.T


def _ideal_denoise_for(mixture: GaussianMixture) -> Denoise:
    def denoise(x: Tensor, sigma: Tensor) -> Tensor:
        return ideal_denoiser(mixture, x, sigma)

    return denoise


def _echo_denoise(x: Tensor, _sigma: Tensor) -> Tensor:
    return x


class _CountingDenoiser:
    """Wrap a denoiser and count how many times it is called."""

    def __init__(self, denoise: Denoise) -> None:
        self._denoise = denoise
        self.call_count = 0

    def __call__(self, x: Tensor, sigma: Tensor) -> Tensor:
        self.call_count += 1
        return self._denoise(x, sigma)


def test_karras_sigmas_schedule_reference() -> None:
    """Schedule length, endpoints, monotonicity, rho spacing (refs)."""
    schedule = karras_sigmas(18)
    assert schedule.shape == (19,)
    assert abs(schedule[0].item() - 80.0) / 80.0 < 1e-12
    assert abs(schedule[17].item() - 0.002) / 0.002 < 1e-12
    assert schedule[18].item() == 0.0
    assert (schedule[1:] < schedule[:-1]).all()
    rho = 7.0
    roots = schedule[:18] ** (1.0 / rho)
    spacing = roots[1:] - roots[:-1]
    assert torch.allclose(
        spacing, spacing[0].expand_as(spacing), rtol=1e-12, atol=1e-12
    )


def test_heun_sample_matches_gaussian_probability_flow() -> None:
    """Single blob: Heun vs closed-form ODE; second order shrinks ~4x."""
    single_blob = GaussianMixture(
        weights=torch.tensor([1.0], dtype=torch.float64),
        means=torch.tensor([[1.0, -2.0]], dtype=torch.float64),
        covariances=torch.tensor(
            [[[0.5, 0.2], [0.2, 0.3]]], dtype=torch.float64
        ),
    )
    generator = torch.Generator().manual_seed(0)
    sigma_start = 80.0
    sigma_stop = 0.002
    noisy_cloud = sigma_start * torch.randn(
        1_000, 2, generator=generator, dtype=torch.float64
    )
    denoise = _ideal_denoise_for(single_blob)
    mean = single_blob.means[0]
    covariance = single_blob.covariances[0]
    oracle = _probability_flow_closed_form(
        noisy_cloud, sigma_start, sigma_stop, mean, covariance
    )

    def max_error(step_count: int) -> float:
        sigmas = karras_sigmas(step_count)[:-1]
        integrated = heun_sample(denoise, noisy_cloud.clone(), sigmas)
        return (integrated - oracle).abs().max().item()

    error_forty = max_error(40)
    error_twenty = max_error(20)
    assert error_forty < 0.03
    ratio = error_twenty / error_forty
    assert 3.5 <= ratio <= 5.0


def test_heun_sample_reproduces_three_blobs_mixture() -> None:
    """Ideal denoiser plus Heun: moments and shares match THREE_BLOBS."""
    generator = torch.Generator().manual_seed(1)
    sample_count = 100_000
    sigma_max = 80.0
    started = initial_noise(sample_count, sigma_max, generator)
    denoise = _ideal_denoise_for(THREE_BLOBS)
    sigmas = karras_sigmas(64)
    samples = heun_sample(denoise, started, sigmas)
    mean_error, covariance_error = _sample_moment_errors(samples, THREE_BLOBS)
    sample_std = samples.std(dim=0, unbiased=True)
    mean_threshold = 5.0 * (sample_std / (sample_count**0.5)).max()
    assert mean_error.max() <= mean_threshold
    assert covariance_error.item() < 0.03
    shares = _responsibility_shares(samples, THREE_BLOBS)
    for observed, expected in zip(
        shares.tolist(), THREE_BLOBS.weights.tolist(), strict=True
    ):
        standard_error = math.sqrt(expected * (1.0 - expected) / sample_count)
        assert abs(observed - expected) <= 5.0 * standard_error


def test_stochastic_sample_reproduces_three_blobs_mixture() -> None:
    """Stochastic Heun: moments match THREE_BLOBS within Monte Carlo limits."""
    generator = torch.Generator().manual_seed(2)
    sample_count = 100_000
    sigma_max = 80.0
    started = initial_noise(sample_count, sigma_max, generator)
    denoise = _ideal_denoise_for(THREE_BLOBS)
    sigmas = karras_sigmas(128)
    samples = stochastic_sample(denoise, started, sigmas, generator)
    mean_error, covariance_error = _sample_moment_errors(samples, THREE_BLOBS)
    sample_std = samples.std(dim=0, unbiased=True)
    mean_threshold = 5.0 * (sample_std / (sample_count**0.5)).max()
    assert mean_error.max() <= mean_threshold
    assert covariance_error.item() < 0.05


def test_denoiser_call_counts_match_edm() -> None:
    """Each integrator: two denoiser evals per step except the last."""
    step_count = 32
    sigmas = karras_sigmas(step_count)
    generator = torch.Generator().manual_seed(0)
    points = initial_noise(8, 80.0, generator)
    expected_calls = 2 * step_count - 1

    counting = _CountingDenoiser(_echo_denoise)
    heun_sample(counting, points.clone(), sigmas)
    assert counting.call_count == expected_calls

    counting.call_count = 0
    stochastic_sample(counting, points.clone(), sigmas, generator)
    assert counting.call_count == expected_calls


@pytest.mark.parametrize(
    ("step_count", "keyword_arguments"),
    [
        (1, {}),
        (18, {"sigma_min": 0.0}),
        (18, {"sigma_max": 0.001}),
        (18, {"rho": 0.0}),
    ],
)
def test_karras_sigmas_validation_raises(
    step_count: int, keyword_arguments: dict
) -> None:
    """Each karras_sigmas validation rule raises ValueError."""
    with pytest.raises(ValueError):
        karras_sigmas(step_count, **keyword_arguments)


def test_sampler_validation_raises() -> None:
    """Strictly decreasing sigmas and a non-negative terminal are required."""
    generator = torch.Generator().manual_seed(0)
    points = torch.zeros(2, 2, dtype=torch.float64)
    increasing = torch.tensor([0.002, 1.0, 80.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="strictly decreasing"):
        heun_sample(_echo_denoise, points, increasing)
    with pytest.raises(ValueError, match="strictly decreasing"):
        stochastic_sample(_echo_denoise, points, increasing, generator)
    negative_end = torch.tensor([80.0, 0.002, -0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="end in 0"):
        heun_sample(_echo_denoise, points, negative_end)
    with pytest.raises(ValueError, match="end in 0"):
        stochastic_sample(_echo_denoise, points, negative_end, generator)
