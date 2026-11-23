"""Posterior covariance and the denoising loss, against independent oracles."""

import math

import pytest
import torch
import torch.autograd.functional as autograd_functional
from torch import Tensor

from edm.diffusion import ideal_denoiser, posterior_covariance
from edm.loss import denoising_loss
from edm.mixture import THREE_BLOBS, GaussianMixture


def _tweedie_posterior_covariance(
    noisy_points: Tensor, noise_level: float
) -> Tensor:
    """Oracle: s^2 I + s^4 times the Hessian of the noisy log-density."""
    # Widened mixture and identity feed the Tweedie oracle, not our closed form.
    widened = THREE_BLOBS.noised(noise_level)
    identity = torch.eye(2, dtype=torch.float64)

    # Scalar log-density per point is what PyTorch differentiates twice.
    def log_density_single(point: Tensor) -> Tensor:
        return widened.log_prob(point.unsqueeze(0)).squeeze()

    # Row-wise Hessians of log p(x) are the independent second-order oracle.
    hessian_rows = [
        autograd_functional.hessian(log_density_single, noisy_points[row])
        for row in range(noisy_points.shape[0])
    ]
    # Stack to (n, 2, 2): each row is the curvature of log p(x) at one point.
    hessian = torch.stack(hessian_rows, dim=0)
    # s^2 I + s^4 H is the Tweedie covariance, built only from autograd.
    return (noise_level**2) * identity + (noise_level**4) * hessian


def _mixture_total_covariance(mixture: GaussianMixture) -> Tensor:
    """Oracle: Cov(x) from weights, means, and component covariances."""
    # Mixture second moments from weights and component parameters, not from
    # posterior_covariance, are the limit when noise washes out the observation.
    weighted_second = (
        mixture.weights[..., None, None]
        * (
            mixture.covariances
            + mixture.means[..., :, None] * mixture.means[..., None, :]
        )
    ).sum(dim=0)
    # Subtracting the outer product of the mean yields total covariance Cov(x).
    mixture_mean = (mixture.weights.unsqueeze(-1) * mixture.means).sum(dim=0)
    mean_outer = mixture_mean[:, None] * mixture_mean[None, :]
    return weighted_second - mean_outer


@pytest.mark.parametrize("noise_level", (0.05, 0.5, 3.0))
def test_posterior_covariance_matches_second_order_tweedie(
    noise_level: float,
) -> None:
    """Second-order Tweedie: Cov(x0|x) equals s^2 I + s^4 H from autograd."""
    # Fixed seed makes clean and noisy draws reproducible for both sides.
    generator = torch.Generator().manual_seed(0)
    sample_count = 200
    clean = THREE_BLOBS.sample(sample_count, generator)
    # Forward diffusion noise supplies x; the oracle Hessian uses the same x.
    noisy = clean + noise_level * torch.randn(
        sample_count, 2, generator=generator, dtype=torch.float64
    )
    # Closed form is under test; autograd Tweedie is the outside covariance.
    closed_form = posterior_covariance(THREE_BLOBS, noisy, noise_level)
    oracle = _tweedie_posterior_covariance(noisy, noise_level)
    assert torch.allclose(closed_form, oracle, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("noise_level", (0.05, 0.5, 3.0))
def test_denoising_loss_minus_trace_posterior_mean_near_zero(
    noise_level: float,
) -> None:
    """The minimum: ideal loss minus trace Cov(x0 | x) averages to 0 (5 SE)."""
    # Many samples sharpen whether Bayes-optimal loss minus trace averages to 0.
    generator = torch.Generator().manual_seed(1)
    sample_count = 400_000
    clean = THREE_BLOBS.sample(sample_count, generator)
    # Noisy points are the conditioning variable in both loss and posterior.
    noisy = clean + noise_level * torch.randn(
        sample_count, 2, generator=generator, dtype=torch.float64
    )
    # E[x0 | x] is the reference denoiser; it is defined outside denoising_loss.
    denoised = ideal_denoiser(THREE_BLOBS, noisy, noise_level)
    per_point_loss = denoising_loss(denoised, clean)
    post_cov = posterior_covariance(THREE_BLOBS, noisy, noise_level)
    # Trace Cov(x0 | x) is irreducible error; oracle from posterior_covariance.
    trace_post = post_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    # Per point, squared error of the best guess equals leftover posterior
    # uncertainty, so this difference averages to zero; a non-Bayes-optimal
    # denoiser would leave a positive mean.
    residual = per_point_loss - trace_post
    # Sample mean and spread estimate how far the residual average is from zero.
    sample_mean = residual.mean().item()
    sample_std = residual.std(unbiased=True).item()
    # Five standard errors: a correct identity fails less than once per million.
    standard_error = sample_std / math.sqrt(sample_count)
    assert abs(sample_mean) <= 5.0 * standard_error


@pytest.mark.parametrize("noise_level", (0.05, 0.5, 3.0))
def test_biased_denoiser_raises_mean_loss_by_bias_norm_squared(
    noise_level: float,
) -> None:
    """The minimiser: shifting the denoiser by c adds |c|^2 to the loss."""
    # Same large-n setup as the minimum test, for a sharp mean-loss comparison.
    generator = torch.Generator().manual_seed(1)
    sample_count = 400_000
    clean = THREE_BLOBS.sample(sample_count, generator)
    # Synthetic noise gives x while ground truth stays available for the loss.
    noisy = clean + noise_level * torch.randn(
        sample_count, 2, generator=generator, dtype=torch.float64
    )
    # Optimal predictions come from ideal_denoiser, not from the loss routine.
    denoised = ideal_denoiser(THREE_BLOBS, noisy, noise_level)
    optimal_loss = denoising_loss(denoised, clean)
    # Fixed bias c is the outside perturbation; |c|^2 is the predicted increase.
    bias = torch.tensor([0.1, -0.2], dtype=torch.float64)
    # Shift by c adds |c|^2 on average: optimal error is uncorrelated with c,
    # which is why E[x0|x] minimises loss; |c|^2 is the outside prediction.
    shifted_loss = denoising_loss(denoised + bias, clean)
    loss_increase = (shifted_loss - optimal_loss).mean().item()
    # Pointwise excess over |c|^2 should fluctuate around zero at the minimiser.
    residual = (shifted_loss - optimal_loss) - bias.square().sum().item()
    sample_std = residual.std(unbiased=True).item()
    # Five standard errors: mean increase should match |c|^2 except rarely.
    standard_error = sample_std / math.sqrt(sample_count)
    # |c|^2 from the bias tensor is the outside closed-form prediction.
    expected_increase = bias.square().sum().item()
    assert abs(loss_increase - expected_increase) <= 5.0 * standard_error


def test_posterior_covariance_near_zero_at_tiny_noise() -> None:
    """Limits: negligible noise leaves almost no uncertainty about x0."""
    # Fixed points exercise the low-noise limit of the posterior covariance.
    generator = torch.Generator().manual_seed(7)
    points = THREE_BLOBS.sample(200, generator)
    # Almost noiseless observations should leave almost no posterior spread.
    tiny_sigma = 1e-6
    covariance = posterior_covariance(THREE_BLOBS, points, tiny_sigma)
    # Zero matrix is the outside limit: knowing x means Cov(x0 | x) vanishes.
    zero = torch.zeros(2, 2, dtype=torch.float64)
    assert torch.allclose(covariance, zero, atol=1e-10)


def test_posterior_covariance_matches_total_mixture_at_huge_noise() -> None:
    """Huge noise makes Cov(x0 | x) the mixture's total covariance."""
    # Observations at fixed locations probe the high-noise posterior limit.
    generator = torch.Generator().manual_seed(7)
    points = THREE_BLOBS.sample(200, generator)
    # When sigma dwarfs the data, x carries almost no information about x0.
    huge_sigma = 1e6
    covariance = posterior_covariance(THREE_BLOBS, points, huge_sigma)
    # Total mixture covariance from weights means is independent of the formula.
    oracle_total = _mixture_total_covariance(THREE_BLOBS)
    expected = oracle_total.unsqueeze(0).expand_as(covariance)
    assert torch.allclose(covariance, expected, atol=1e-6)


def test_posterior_covariance_symmetric_positive_semidefinite() -> None:
    """Every posterior covariance is symmetric with no negative eigenvalue."""
    # Batched points at moderate sigma stress structural laws of Cov(x0 | x).
    generator = torch.Generator().manual_seed(12)
    points = THREE_BLOBS.sample(100, generator)
    # Mid noise is a generic regime where covariances must behave as laws.
    covariance = posterior_covariance(THREE_BLOBS, points, 0.5)
    # Batch layout (n, 2, 2) is what training code expects for diffusion covs.
    assert covariance.shape == (100, 2, 2)
    # Symmetry is checked via transpose, required of any covariance matrix.
    transpose = covariance.transpose(-2, -1)
    assert torch.allclose(covariance, transpose, atol=1e-15)
    # Nonnegative eigenvalues are the outside PSD check for diffusion covs.
    eigenvalues = torch.linalg.eigvalsh(covariance)
    assert (eigenvalues >= -1e-15).all()


def test_posterior_covariance_per_point_sigma_matches_scalar_calls() -> None:
    """Per-point sigma should match calling the scalar path per row."""
    # Same batch size as ideal_denoiser API tests for apples-to-apples coverage.
    generator = torch.Generator().manual_seed(99)
    batch_size = 50
    points = THREE_BLOBS.sample(batch_size, generator)
    # A vector of noise levels exercises batched sigma without changing x0 | x.
    per_point_sigma = torch.linspace(0.1, 2.0, batch_size, dtype=torch.float64)
    batched = posterior_covariance(THREE_BLOBS, points, per_point_sigma)
    # Row-wise scalar calls are the reference oracle for the batched API.
    reference_rows = [
        posterior_covariance(
            THREE_BLOBS, points[row : row + 1], per_point_sigma[row].item()
        )
        for row in range(batch_size)
    ]
    # Concatenated scalar results are the independent batched reference.
    reference = torch.cat(reference_rows, dim=0)
    assert torch.allclose(batched, reference)


def test_denoising_loss_by_hand() -> None:
    """denoising_loss: fixed tensors give [5, 25] per coordinate sums."""
    # Hand-picked tensors make the squared-error sum obvious without sampling.
    denoised = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64)
    clean = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float64)
    result = denoising_loss(denoised, clean)
    # [5, 25] is the outside arithmetic: (1^2+2^2) and (3^2+4^2) per row.
    expected = torch.tensor([5.0, 25.0], dtype=torch.float64)
    assert torch.equal(result, expected)


def test_denoising_loss_shape_mismatch_raises() -> None:
    """Mismatched shapes must name both tensor shapes in the error."""
    # Deliberately mismatched batch sizes should be caught before any diffusion.
    denoised = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    clean = torch.tensor([[0.0, 0.0], [3.0, 4.0]], dtype=torch.float64)
    # The regex is the outside contract on how shape errors are reported.
    with pytest.raises(
        ValueError,
        match=r"denoised shape \(1, 2\) must match x0 shape \(2, 2\)",
    ):
        denoising_loss(denoised, clean)
