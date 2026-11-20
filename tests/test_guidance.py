"""Classifier guidance: gradients, w=1 score, sampling, sharpening."""

import math

import pytest
import torch
from torch import Tensor

from toy.guidance import guided_denoiser, log_responsibility_gradient
from toy.mixture import THREE_BLOBS, GaussianMixture
from toy.sample import heun_sample, initial_noise, karras_sigmas


def _autograd_log_responsibility_gradient(
    mixture: GaussianMixture, x: Tensor, sigma: float, blob: int
) -> Tensor:
    """Oracle: autograd grad_x log r_k from torch.distributions densities."""
    detached = x.detach().clone().requires_grad_(True)
    identity = torch.eye(2, dtype=torch.float64, device=x.device)
    widened_covariance = mixture.covariances + (sigma**2) * identity
    log_blob = torch.stack(
        [
            torch.distributions.MultivariateNormal(
                mixture.means[index], widened_covariance[index]
            ).log_prob(detached)
            + mixture.weights[index].log()
            for index in range(mixture.weights.shape[0])
        ],
        dim=-1,
    )
    log_responsibility = torch.log_softmax(log_blob, dim=-1)
    target = log_responsibility[:, blob]
    (gradient,) = torch.autograd.grad(target.sum(), detached)
    return gradient


def _widened_blob_score(
    mixture: GaussianMixture, x: Tensor, sigma: float, blob: int
) -> Tensor:
    """Oracle: score of blob k alone after isotropic widening by sigma."""
    identity = torch.eye(2, dtype=torch.float64, device=x.device)
    widened = mixture.covariances[blob] + (sigma**2) * identity
    chol_factor = torch.linalg.cholesky(widened)
    offset = (x - mixture.means[blob]).unsqueeze(-1)
    return -torch.cholesky_solve(offset, chol_factor).squeeze(-1)


def _guided_score_at_scale_one(
    mixture: GaussianMixture, x: Tensor, sigma: float, blob: int
) -> Tensor:
    """Score implied by guided_denoiser at unit guidance strength."""
    denoise = guided_denoiser(mixture, blob, 1.0)
    batch_sigma = torch.full(
        (x.shape[0],), sigma, dtype=torch.float64, device=x.device
    )
    denoised = denoise(x, batch_sigma)
    noise_variance = sigma**2
    return (denoised - x) / noise_variance


def _blob_moment_errors(
    samples: Tensor, mean: Tensor, covariance: Tensor
) -> tuple[Tensor, Tensor]:
    """Absolute mean error and max covariance error vs one blob."""
    sample_mean = samples.mean(dim=0)
    # torch.cov wants one variable per row; shape (2, 2), not (n, n).
    sample_covariance = torch.cov(samples.T)
    mean_error = (sample_mean - mean).abs()
    covariance_error = (sample_covariance - covariance).abs().max()
    return mean_error, covariance_error


@pytest.mark.parametrize("sigma", [0.05, 0.5, 3.0])
def test_log_responsibility_gradient_matches_autograd(sigma: float) -> None:
    """Closed form equals autograd grad log r_k at 1,000 points, every k."""
    generator = torch.Generator().manual_seed(0)
    point_count = 1_000
    cloud = torch.randn(
        point_count, 2, generator=generator, dtype=torch.float64
    )
    component_count = THREE_BLOBS.weights.shape[0]
    for blob in range(component_count):
        closed = log_responsibility_gradient(THREE_BLOBS, cloud, sigma, blob)
        oracle = _autograd_log_responsibility_gradient(
            THREE_BLOBS, cloud, sigma, blob
        )
        assert torch.allclose(closed, oracle, rtol=1e-8, atol=1e-10)


def test_unit_scale_guided_score_is_conditional_blob_score() -> None:
    """At w=1 the guided score matches the widened single-blob score."""
    generator = torch.Generator().manual_seed(1)
    point_count = 500
    cloud = torch.randn(
        point_count, 2, generator=generator, dtype=torch.float64
    )
    for sigma in (0.05, 0.5, 3.0):
        for blob in range(THREE_BLOBS.weights.shape[0]):
            guided = _guided_score_at_scale_one(THREE_BLOBS, cloud, sigma, blob)
            oracle = _widened_blob_score(THREE_BLOBS, cloud, sigma, blob)
            assert torch.allclose(guided, oracle, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("blob", [0, 1, 2])
def test_heun_guidance_at_unit_scale_hits_target_blob(blob: int) -> None:
    """w=1 Heun sampling concentrates on blob k's mean and covariance."""
    generator = torch.Generator().manual_seed(4)
    sample_count = 20_000
    sigma_max = 80.0
    started = initial_noise(sample_count, sigma_max, generator)
    denoise = guided_denoiser(THREE_BLOBS, blob, 1.0)
    sigmas = karras_sigmas(64)
    samples = heun_sample(denoise, started, sigmas)
    target_mean = THREE_BLOBS.means[blob]
    target_covariance = THREE_BLOBS.covariances[blob]
    mean_error, covariance_error = _blob_moment_errors(
        samples, target_mean, target_covariance
    )
    # 64 Heun steps leave a discretisation bias of about 0.015 in the mean
    # (director's reference), above five standard errors at this sample
    # size, so the mean gets the same absolute bound as the covariance.
    assert mean_error.max() < 0.03
    assert covariance_error.item() < 0.03


def test_strong_guidance_sharpens_blob_one() -> None:
    """Scale 3 lowers sample covariance trace versus blob 1 alone."""
    generator = torch.Generator().manual_seed(4)
    sample_count = 100_000
    sigma_max = 80.0
    started = initial_noise(sample_count, sigma_max, generator)
    denoise = guided_denoiser(THREE_BLOBS, 1, 3.0)
    sigmas = karras_sigmas(64)
    samples = heun_sample(denoise, started, sigmas)
    # torch.cov wants one variable per row; shape (2, 2), not (n, n).
    sample_covariance = torch.cov(samples.T)
    blob_trace = torch.trace(THREE_BLOBS.covariances[1])
    sample_trace = torch.trace(sample_covariance)
    trace_ratio = sample_trace / blob_trace
    # Outside answer: measured trace ratio goes in the task report.
    assert sample_trace < blob_trace
    assert trace_ratio < 1.0


def test_log_responsibility_gradient_validates_blob() -> None:
    """Out-of-range blob index raises ValueError."""
    points = torch.zeros(2, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="out of range"):
        log_responsibility_gradient(THREE_BLOBS, points, 0.5, -1)
    with pytest.raises(ValueError, match="out of range"):
        log_responsibility_gradient(THREE_BLOBS, points, 0.5, 3)


def test_guided_denoiser_validates_scale_and_blob() -> None:
    """Non-finite scale and out-of-range blob raise ValueError."""
    with pytest.raises(ValueError, match="out of range"):
        guided_denoiser(THREE_BLOBS, 5, 1.0)
    with pytest.raises(ValueError, match="finite"):
        guided_denoiser(THREE_BLOBS, 0, math.inf)
    with pytest.raises(ValueError, match="finite"):
        guided_denoiser(THREE_BLOBS, 0, float("nan"))
