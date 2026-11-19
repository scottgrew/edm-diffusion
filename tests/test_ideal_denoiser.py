"""Tests for the ideal denoiser and score against independent oracles."""

import pytest
import torch

from toy.diffusion import _noise_column, ideal_denoiser, score
from toy.mixture import THREE_BLOBS


@pytest.mark.parametrize("noise_level", (0.05, 0.5, 3.0))
def test_score_matches_autograd_log_prob_gradient(noise_level: float) -> None:
    """Tweedie: score matches the gradient of the noisy mixture log-density."""
    generator = torch.Generator().manual_seed(42)
    sample_count = 1_000
    clean = THREE_BLOBS.sample(sample_count, generator)
    # Synthetic noisy points let autograd differentiate the noisy log-density.
    noisy = clean + noise_level * torch.randn(
        sample_count, 2, generator=generator, dtype=torch.float64
    )
    noisy.requires_grad_(True)
    widened = THREE_BLOBS.noised(noise_level)
    # Autograd on PyTorch's log_prob is independent of our closed-form score.
    log_density = widened.log_prob(noisy).sum()
    log_density.backward()
    autograd_score = noisy.grad
    closed_form_score = score(THREE_BLOBS, noisy.detach(), noise_level)
    assert torch.allclose(
        closed_form_score, autograd_score, rtol=1e-8, atol=1e-10
    )


def test_denoiser_near_identity_at_tiny_noise() -> None:
    """When noise is negligible the posterior mean should equal the input."""
    generator = torch.Generator().manual_seed(7)
    points = THREE_BLOBS.sample(200, generator)
    tiny_sigma = 1e-6
    # Almost no noise leaves the posterior mean equal to the observation.
    denoised = ideal_denoiser(THREE_BLOBS, points, tiny_sigma)
    assert torch.allclose(denoised, points, atol=1e-6)


def test_denoiser_returns_mixture_mean_at_huge_noise() -> None:
    """Huge noise washes out components so the guess is the global mean."""
    generator = torch.Generator().manual_seed(7)
    points = THREE_BLOBS.sample(200, generator)
    huge_sigma = 1e6
    denoised = ideal_denoiser(THREE_BLOBS, points, huge_sigma)
    # The weighted centre is the limit when blobs are indistinguishable.
    mixture_mean = (THREE_BLOBS.weights.unsqueeze(-1) * THREE_BLOBS.means).sum(
        dim=0
    )
    assert torch.allclose(
        denoised, mixture_mean.unsqueeze(0).expand_as(denoised), atol=1e-6
    )


def test_denoiser_per_point_sigma_matches_scalar_calls() -> None:
    """A vector of noise levels should match calling the scalar path per row."""
    generator = torch.Generator().manual_seed(99)
    batch_size = 50
    points = THREE_BLOBS.sample(batch_size, generator)
    per_point_sigma = torch.linspace(0.1, 2.0, batch_size, dtype=torch.float64)
    batched = ideal_denoiser(THREE_BLOBS, points, per_point_sigma)
    # Each scalar call is the reference oracle for one row.
    reference_rows = [
        ideal_denoiser(
            THREE_BLOBS, points[row : row + 1], per_point_sigma[row].item()
        )
        for row in range(batch_size)
    ]
    reference = torch.cat(reference_rows, dim=0)
    assert torch.allclose(batched, reference)


def test_noise_column_float_sigma() -> None:
    """A Python float becomes a single (1, 1) column in float64."""
    result = _noise_column(0.5, 10)
    expected = torch.tensor([[0.5]], dtype=torch.float64)
    assert result.shape == (1, 1)
    assert torch.equal(result, expected)


def test_noise_column_zero_dim_tensor() -> None:
    """A zero-dimensional tensor is reshaped like a scalar sigma."""
    sigma = torch.tensor(0.5, dtype=torch.float64)
    result = _noise_column(sigma, 10)
    expected = torch.tensor([[0.5]], dtype=torch.float64)
    assert result.shape == (1, 1)
    assert torch.equal(result, expected)


def test_noise_column_per_point_vector() -> None:
    """A length-n vector becomes an (n, 1) column of noise levels."""
    batch_size = 4
    sigma = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    result = _noise_column(sigma, batch_size)
    assert result.shape == (batch_size, 1)
    assert torch.equal(result, sigma.reshape(batch_size, 1))


def test_noise_column_length_mismatch_raises() -> None:
    """Sigma length must match the batch size named in the error."""
    sigma = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    with pytest.raises(
        ValueError, match="sigma length 3 must match batch size n=5"
    ):
        _noise_column(sigma, 5)


def test_noise_column_rejects_wrong_rank() -> None:
    """Only floats and length-n vectors are valid sigma shapes."""
    sigma = torch.tensor([[0.5]], dtype=torch.float64)
    with pytest.raises(ValueError, match="sigma must be a float or shape"):
        _noise_column(sigma, 1)
