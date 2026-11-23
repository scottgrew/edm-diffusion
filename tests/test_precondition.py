"""EDM preconditioning: identities, unit variance, round trip, and oracles."""

import math

import pytest
import torch

from edm.diffusion import add_noise, ideal_denoiser, posterior_covariance
from edm.loss import denoising_loss
from edm.mixture import THREE_BLOBS, GaussianMixture
from edm.precondition import (
    Coefficients,
    coefficients,
    denoiser_from_network,
    loss_weight,
    network_target,
    sample_sigma,
    sigma_data_of,
)


def _mixture_clean_second_moment(mixture: GaussianMixture) -> float:
    """Oracle: E|x0|^2 from weights, means, and traces, not sigma_data_of."""
    # Per-blob trace and mean norm are the closed-form ingredients of E|x0|^2.
    component_traces = mixture.covariances.diagonal(dim1=-2, dim2=-1).sum(
        dim=-1
    )
    mean_squared_norm = (mixture.means**2).sum(dim=-1)
    # Weighted sum is independent of the precondition module under test.
    return (
        (mixture.weights * (component_traces + mean_squared_norm)).sum().item()
    )


def test_loss_weight_times_c_out_squared_is_one_on_log_grid() -> None:
    """Weight identity: loss_weight * c_out^2 == 1 on a log grid 0.002..80."""
    # Log-spaced sigmas stress the identity across orders of magnitude in s.
    sigma = torch.logspace(
        math.log10(0.002), math.log10(80.0), steps=200, dtype=torch.float64
    )
    sd = sigma_data_of(THREE_BLOBS)
    coef = coefficients(sigma, sd)
    weight = loss_weight(sigma, sd)
    # Outside claim: lambda * c_out^2 equals 1 exactly for EDM table 1.
    product = weight * coef.c_out.square()
    # Max deviation from 1 is the scalar oracle; reference is exactly 0.
    max_deviation = (product - 1.0).abs().max().item()
    assert max_deviation <= 1e-15


@pytest.mark.parametrize("noise_level", (0.002, 0.5, 80.0))
def test_network_target_has_unit_variance_per_coordinate(
    noise_level: float,
) -> None:
    """Unit-variance target: mean |network_target|^2 is 2 within 5 SE."""
    # Fixed seed matches director reference draws for clean and Gaussian noise.
    generator = torch.Generator().manual_seed(0)
    sample_count = 400_000
    clean = THREE_BLOBS.sample(sample_count, generator)
    # Constant sigma per row matches the director's reference Monte Carlo setup.
    sigma_tensor = torch.full((sample_count,), noise_level, dtype=torch.float64)
    # One noise level per row exercises the batched (n,) sigma API.
    noisy = add_noise(clean, sigma_tensor, generator)
    sd = sigma_data_of(THREE_BLOBS)
    target = network_target(clean, noisy, sigma_tensor, sd)
    # Mean |target|^2 over draws should be 2 when each coordinate has unit var.
    mean_squared_norm = target.square().sum(dim=-1).mean().item()
    pointwise_sq = target.square().sum(dim=-1)
    sample_std = pointwise_sq.std(unbiased=True).item()
    # Five SE gates false failures when EDM preconditioning is correct.
    standard_error = sample_std / math.sqrt(sample_count)
    assert abs(mean_squared_norm - 2.0) <= 5.0 * standard_error
    # Independent oracle: E|x0|^2 = 2 sd^2 from mixture parameters alone.
    second_moment = _mixture_clean_second_moment(THREE_BLOBS)
    assert abs(second_moment - 2.0 * sd * sd) <= 1e-12


@pytest.mark.parametrize("noise_level", (0.002, 0.5, 80.0))
def test_scaled_input_has_unit_variance_per_coordinate(
    noise_level: float,
) -> None:
    """Unit-variance input: mean |c_in x|^2 is 2 within 5 SE at each sigma."""
    generator = torch.Generator().manual_seed(0)
    sample_count = 400_000
    clean = THREE_BLOBS.sample(sample_count, generator)
    # Same (n,) sigma layout as the network-target variance check.
    sigma_tensor = torch.full((sample_count,), noise_level, dtype=torch.float64)
    noisy = add_noise(clean, sigma_tensor, generator)
    sd = sigma_data_of(THREE_BLOBS)
    coef = coefficients(sigma_tensor, sd)
    # c_in scales the noisy point; only (n, 1) broadcast multiplies (n, 2) x.
    scaled = coef.c_in.unsqueeze(-1) * noisy
    pointwise_sq = scaled.square().sum(dim=-1)
    mean_squared_norm = pointwise_sq.mean().item()
    sample_std = pointwise_sq.std(unbiased=True).item()
    standard_error = sample_std / math.sqrt(sample_count)
    assert abs(mean_squared_norm - 2.0) <= 5.0 * standard_error
    # Outside expectation: E|x|^2 = 2 (sd^2 + s^2) for isotropic 2-D noise.
    expected_noisy_second_moment = 2.0 * (sd * sd + noise_level * noise_level)
    empirical = noisy.square().sum(dim=-1).mean().item()
    empirical_std = noisy.square().sum(dim=-1).std(unbiased=True).item()
    empirical_se = empirical_std / math.sqrt(sample_count)
    assert abs(empirical - expected_noisy_second_moment) <= 5.0 * empirical_se


@pytest.mark.parametrize("noise_level", (0.002, 0.5, 80.0))
def test_denoiser_round_trip_recovers_clean_within_tolerance(
    noise_level: float,
) -> None:
    """Round trip: denoiser_from_network(network_target(...)) returns x0."""
    generator = torch.Generator().manual_seed(0)
    sample_count = 400
    clean = THREE_BLOBS.sample(sample_count, generator)
    # Fewer points suffice because the round trip is exact in floating point.
    sigma_tensor = torch.full((sample_count,), noise_level, dtype=torch.float64)
    noisy = add_noise(clean, sigma_tensor, generator)
    sd = sigma_data_of(THREE_BLOBS)
    # F target then D reconstruction is the algebraic identity D(x0|F) = x0.
    network_out = network_target(clean, noisy, sigma_tensor, sd)
    reconstructed = denoiser_from_network(network_out, noisy, sigma_tensor, sd)
    # A relative tolerance allows for large coordinates at sigma 80.
    relative_error = (
        (
            (reconstructed - clean).norm(dim=-1)
            / clean.norm(dim=-1).clamp(min=1e-12)
        )
        .max()
        .item()
    )
    assert relative_error <= 1e-12


@pytest.mark.parametrize("noise_level", (0.002, 0.5, 80.0))
def test_weighted_ideal_network_loss_minus_trace_averages_to_zero(
    noise_level: float,
) -> None:
    """Ideal network: weighted loss minus weighted trace posterior mean is 0."""
    generator = torch.Generator().manual_seed(0)
    sample_count = 400_000
    clean = THREE_BLOBS.sample(sample_count, generator)
    sigma_tensor = torch.full((sample_count,), noise_level, dtype=torch.float64)
    noisy = add_noise(clean, sigma_tensor, generator)
    sd = sigma_data_of(THREE_BLOBS)
    # EDM lambda weights the denoising loss so all sigmas contribute equally.
    weight = loss_weight(sigma_tensor, sd)
    denoised = ideal_denoiser(THREE_BLOBS, noisy, sigma_tensor)
    per_point_loss = denoising_loss(denoised, clean)
    post_cov = posterior_covariance(THREE_BLOBS, noisy, sigma_tensor)
    # trace(Cov) is irreducible error; diagonal sum -> (n,) per point.
    trace_post = post_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    residual = weight * per_point_loss - weight * trace_post
    # Weighted residual mean near 0 means lambda lines up with the Bayes limit.
    sample_mean = residual.mean().item()
    sample_std = residual.std(unbiased=True).item()
    standard_error = sample_std / math.sqrt(sample_count)
    assert abs(sample_mean) <= 5.0 * standard_error


def test_sample_sigma_log_moments_match_edm_priors() -> None:
    """sample_sigma: log-mean -1.2 and log-sd 1.2 within 5 SE, all positive."""
    generator = torch.Generator().manual_seed(3)
    sample_count = 400_000
    sigmas = sample_sigma(sample_count, generator)
    # log sigma is what we compare to Normal(-1.2, 1.2) training prior.
    log_sigma = sigmas.log()
    log_mean = log_sigma.mean().item()
    log_std = log_sigma.std(unbiased=True).item()
    # SE of the sample mean and sd are closed forms for the director check.
    mean_se = 1.2 / math.sqrt(sample_count)
    std_se = 1.2 / math.sqrt(2.0 * sample_count)
    assert abs(log_mean - (-1.2)) <= 5.0 * mean_se
    assert abs(log_std - 1.2) <= 5.0 * std_se
    assert (sigmas > 0).all()


def test_sigma_data_of_matches_samples_and_closed_form() -> None:
    """sigma_data_of equals sampling oracle and 1.6224980739587953."""
    closed = sigma_data_of(THREE_BLOBS)
    # Director reference value is the outside closed-form anchor.
    reference = 1.6224980739587953
    assert abs(closed - reference) <= 1e-12
    generator = torch.Generator().manual_seed(0)
    sample_count = 400_000
    clean = THREE_BLOBS.sample(sample_count, generator)
    # Sample mean |x0|^2 / 2 is independent of sigma_data_of's formula path.
    radius_sq_mean = clean.square().sum(dim=-1).mean().item() / 2.0
    sample_sd = math.sqrt(radius_sq_mean)
    pointwise = clean.square().sum(dim=-1) / 2.0
    sample_std = pointwise.std(unbiased=True).item()
    standard_error = sample_std / math.sqrt(sample_count)
    assert abs(sample_sd - closed) <= 5.0 * standard_error


def test_c_noise_equals_log_sigma_over_four() -> None:
    """c_noise matches log(sigma) / 4 within 1e-15 on a log grid."""
    sigma = torch.logspace(
        math.log10(0.002), math.log10(80.0), steps=100, dtype=torch.float64
    )
    sd = sigma_data_of(THREE_BLOBS)
    coef = coefficients(sigma, sd)
    # log(s)/4 is the outside definition from EDM table 1.
    expected = sigma.log() / 4.0
    assert (coef.c_noise - expected).abs().max().item() <= 1e-15


def test_coefficients_shape_on_valid_sigma() -> None:
    """Decision table: finite positive sigma yields (n,) coefficient blocks."""
    sigma = torch.tensor([0.1, 0.5, 2.0], dtype=torch.float64)
    sd = sigma_data_of(THREE_BLOBS)
    coef = coefficients(sigma, sd)
    # Each coefficient has one entry per noise level, like sigma.
    assert isinstance(coef, Coefficients)
    assert coef.c_skip.shape == (3,)
    assert coef.c_out.shape == (3,)
    assert coef.c_in.shape == (3,)
    assert coef.c_noise.shape == (3,)


def test_coefficients_raises_on_non_positive_sigma() -> None:
    """Decision table: non-positive sigma names first bad index and value."""
    sigma = torch.tensor([0.5, -0.1, 1.0], dtype=torch.float64)
    sd = sigma_data_of(THREE_BLOBS)
    with pytest.raises(
        ValueError,
        match=r"sigma at index 1 must be finite and positive, got -0.1",
    ):
        coefficients(sigma, sd)


def test_coefficients_raises_on_inf_sigma() -> None:
    """Decision table: inf sigma names the first bad index and value."""
    sigma = torch.tensor([0.5, float("inf"), 1.0], dtype=torch.float64)
    sd = sigma_data_of(THREE_BLOBS)
    with pytest.raises(
        ValueError,
        match=r"sigma at index 1 must be finite and positive, got inf",
    ):
        coefficients(sigma, sd)


def test_coefficients_raises_on_nan_sigma() -> None:
    """Decision table: NaN sigma names the first bad index."""
    sigma = torch.tensor([0.5, float("nan"), 1.0], dtype=torch.float64)
    sd = sigma_data_of(THREE_BLOBS)
    with pytest.raises(
        ValueError,
        match=r"sigma at index 1 must be finite and positive, got nan",
    ):
        coefficients(sigma, sd)


def test_coefficients_raises_on_invalid_sigma_data() -> None:
    """Decision table: sigma_data must be finite and positive."""
    sigma = torch.tensor([0.5], dtype=torch.float64)
    with pytest.raises(
        ValueError, match=r"sigma_data must be finite and positive, got -1.0"
    ):
        coefficients(sigma, -1.0)


def test_network_target_shape_mismatch_raises() -> None:
    """Decision table: network_target names both shapes on mismatch."""
    sigma = torch.tensor([0.5, 0.5], dtype=torch.float64)
    x0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    x = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    sd = sigma_data_of(THREE_BLOBS)
    with pytest.raises(
        ValueError, match=(r"x0 shape \(1, 2\) must match x shape \(2, 2\)")
    ):
        network_target(x0, x, sigma, sd)
