"""Tests for Gaussian mixtures against independent oracles."""

import pytest
import torch

from toy.diffusion import add_noise
from toy.mixture import THREE_BLOBS, GaussianMixture


def _assert_moments_match(
    samples: torch.Tensor, mixture: GaussianMixture
) -> None:
    """Compare sample moments to the closed form from weights, means, and S."""
    sample_count = samples.shape[0]
    sample_mean = samples.mean(dim=0)
    # torch.cov wants one variable per row, so the (n, 2) samples go in
    # transposed; untransposed it would build an n-by-n matrix.
    sample_covariance = torch.cov(samples.T)
    # Analytic mean is the weighted average of component centres.
    analytic_mean = (mixture.weights.unsqueeze(-1) * mixture.means).sum(dim=0)
    # Analytic covariance uses the law of total variance in matrix form.
    outer_per_component = torch.einsum(
        "ki,kj->kij", mixture.means, mixture.means
    )
    weighted_second_moment = (
        mixture.weights.unsqueeze(-1).unsqueeze(-1)
        * (mixture.covariances + outer_per_component)
    ).sum(dim=0)
    analytic_covariance = weighted_second_moment - torch.outer(
        analytic_mean, analytic_mean
    )
    sample_std = samples.std(dim=0, unbiased=True)
    standard_error_of_mean = sample_std / (sample_count**0.5)
    mean_threshold = 5.0 * standard_error_of_mean.max()
    # Sampling noise should not move means or covariances past these bounds.
    assert torch.allclose(sample_mean, analytic_mean, atol=mean_threshold)
    assert torch.allclose(sample_covariance, analytic_covariance, atol=0.02)


def test_log_prob_integrates_to_one() -> None:
    """Grid integral near 1 checks that log_prob is a proper density."""
    grid_step = 0.02
    axis = torch.arange(-10.0, 10.0, grid_step, dtype=torch.float64)
    # A square grid covers the plane where almost all mass lives.
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
    grid_points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    # PyTorch's MixtureSameFamily is the oracle, not the hand-written denoiser.
    log_density = THREE_BLOBS.log_prob(grid_points)
    cell_area = grid_step**2
    integral = torch.exp(log_density).sum().item() * cell_area
    assert abs(integral - 1.0) < 1e-4


def test_sample_moments_match_three_blobs() -> None:
    """Large-sample mean and covariance match formulas from the parameters."""
    generator = torch.Generator().manual_seed(0)
    # Many draws make Monte Carlo error small enough to test moments tightly.
    samples = THREE_BLOBS.sample(400_000, generator)
    _assert_moments_match(samples, THREE_BLOBS)


def test_noised_sample_moments_match_widened_mixture() -> None:
    """Noising samples matches sampling the analytically widened mixture."""
    noise_level = 0.8
    generator = torch.Generator().manual_seed(0)
    clean_samples = THREE_BLOBS.sample(400_000, generator)
    # Forward noising should match drawing from widened covariances.
    noisy_samples = add_noise(clean_samples, noise_level, generator)
    _assert_moments_match(noisy_samples, THREE_BLOBS.noised(noise_level))


def test_post_init_rejects_shape_mismatch() -> None:
    """Mismatched tensor ranks are caught before any sampling runs."""
    with pytest.raises(ValueError, match="means must have shape"):
        GaussianMixture(
            weights=torch.tensor([0.5, 0.5], dtype=torch.float64),
            means=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
            covariances=torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                dtype=torch.float64,
            ),
        )


def test_post_init_rejects_non_positive_weight() -> None:
    """Zero or negative weights cannot define a categorical over components."""
    with pytest.raises(ValueError, match="strictly positive"):
        GaussianMixture(
            weights=torch.tensor([0.0, 1.0], dtype=torch.float64),
            means=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            covariances=torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                dtype=torch.float64,
            ),
        )


def test_post_init_rejects_weights_not_summing_to_one() -> None:
    """Weights that do not sum to one break normalization of the mixture."""
    with pytest.raises(ValueError, match="sum to 1"):
        GaussianMixture(
            weights=torch.tensor([0.5, 0.4], dtype=torch.float64),
            means=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            covariances=torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
                dtype=torch.float64,
            ),
        )


def test_post_init_rejects_asymmetric_covariance() -> None:
    """Only symmetric matrices can be valid Gaussian covariances."""
    bad_covariance = torch.tensor([[1.0, 0.5], [0.0, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="symmetric"):
        GaussianMixture(
            weights=torch.tensor([1.0], dtype=torch.float64),
            means=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
            covariances=bad_covariance.unsqueeze(0),
        )


def test_post_init_rejects_indefinite_covariance() -> None:
    """A matrix with a negative eigenvalue cannot be Cholesky-factorized."""
    indefinite = torch.tensor([[1.0, 0.0], [0.0, -0.1]], dtype=torch.float64)
    with pytest.raises(ValueError, match="positive definite"):
        GaussianMixture(
            weights=torch.tensor([1.0], dtype=torch.float64),
            means=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
            covariances=indefinite.unsqueeze(0),
        )
