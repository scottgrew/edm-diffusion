"""EDM sampling: Karras noise schedules, Heun and stochastic integrators."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

Denoise = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class Churn:
    """EDM algorithm 2's settings (Karras et al. 2022, table 5 defaults
    for ImageNet-64): how much noise to add back, between which levels,
    and how much to inflate it."""

    amount: float = 40.0
    sigma_min: float = 0.05
    sigma_max: float = 50.0
    noise_inflation: float = 1.003


DEFAULT_CHURN = Churn()


def _validate_sigmas(sigmas: Tensor) -> int:
    """Reject schedules that cannot be integrated along sigma downward."""
    # Integration needs at least one segment between two noise levels.
    if sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("sigmas must be a length at least 2 vector.")
    # Noise levels must fall strictly step to step along the reverse path.
    if not (sigmas[1:] < sigmas[:-1]).all():
        raise ValueError("sigmas must be strictly decreasing.")
    # The last entry says whether we end at clean data or keep sigma_min.
    terminal_sigma = sigmas[-1].item()
    # Truncated schedules may stop at sigma_min; a final 0 is optional.
    if terminal_sigma >= 0.0:
        return sigmas.numel() - 1
    # A negative terminal level is not a valid endpoint for sampling.
    raise ValueError("sigmas must end in 0.")


def karras_sigmas(
    steps: int,
    *,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> Tensor:
    """EDM's noise levels: steps values from sigma_max down to sigma_min,
    evenly spaced in sigma^(1/rho), then a final 0; shape (steps + 1,)."""
    # We need at least two steps to walk from high noise toward clean data.
    if steps < 2:
        raise ValueError(f"steps must be at least 2, got {steps}.")
    # The smallest level must stay positive so log-variance stays finite.
    if sigma_min <= 0:
        raise ValueError(f"sigma_min must be positive, got {sigma_min}.")
    # The schedule must span a real interval of noise, not a single level.
    if sigma_max <= sigma_min:
        raise ValueError(
            f"sigma_max must exceed sigma_min, got {sigma_max} and "
            f"{sigma_min}."
        )
    # rho controls how tightly mid-range noise levels pack on the path.
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}.")
    # Even spacing in sigma^(1/rho) is EDM's time reparameterisation.
    ramp = torch.linspace(0.0, 1.0, steps, dtype=torch.float64)
    min_root = sigma_min ** (1.0 / rho)
    max_root = sigma_max ** (1.0 / rho)
    sigmas = (max_root + ramp * (min_root - max_root)) ** rho
    # Sampling finishes at zero noise, the clean limit of the diffusion.
    terminal = torch.zeros(1, dtype=torch.float64)
    return torch.cat((sigmas, terminal))


def initial_noise(
    n: int, sigma_max: float, generator: torch.Generator
) -> Tensor:
    """sigma_max times standard normal, shape (n, 2), float64."""
    # The path starts from a wide Gaussian cloud at the largest noise level.
    return sigma_max * torch.randn(
        n, 2, generator=generator, dtype=torch.float64
    )


def _sigma_level_tensor(
    sigma: float, batch_size: int, device: torch.device
) -> Tensor:
    """One scalar noise level broadcast for every row of a batch."""
    # The denoiser expects one sigma per batch row at a shared noise level.
    return torch.full((batch_size,), sigma, dtype=torch.float64, device=device)


def _probability_flow_slope(
    denoise: Denoise, x: Tensor, sigma: float
) -> Tensor:
    """Slope (x - D(x; sigma)) / sigma of the probability-flow ODE."""
    # Pack the scalar level so the denoiser sees a full batch at sigma.
    batch_size = x.shape[0]
    sigma_batch = _sigma_level_tensor(sigma, batch_size, x.device)
    denoised = denoise(x, sigma_batch)
    # (x - D) / sigma points from the clean guess back toward noise.
    return (x - denoised) / sigma


def _heun_step(
    denoise: Denoise, x: Tensor, sigma_from: float, sigma_to: float
) -> Tensor:
    """One Heun step along sigma from sigma_from down to sigma_to."""
    # Step size is negative because sigma falls along the generative path.
    step_size = sigma_to - sigma_from
    # Slope at the start of the segment sets the Euler direction in x-space.
    slope_at_start = _probability_flow_slope(denoise, x, sigma_from)
    # Euler step follows (x - D) / sigma, from the guess back toward noise.
    euler_point = x + step_size * slope_at_start
    # At sigma 0 the slope is undefined; the last step is Euler only.
    if sigma_to == 0.0:
        return euler_point
    # Slope at the trial point supplies the Heun correction at sigma_to.
    slope_at_end = _probability_flow_slope(denoise, euler_point, sigma_to)
    # Averaging both slopes cancels first-order error; halving steps cuts
    # error about four-fold, which is the Heun second-order gain.
    return x + step_size * (0.5 * slope_at_start + 0.5 * slope_at_end)


def heun_sample(denoise: Denoise, x: Tensor, sigmas: Tensor) -> Tensor:
    """EDM algorithm 1: one Euler step from sigma_i to sigma_{i+1}, then a
    trapezoidal correction using the slope at sigma_{i+1}, except on the last
    step to sigma 0 (Euler only, the slope there is undefined)."""
    # Validate the schedule before integrating along sigma.
    step_count = _validate_sigmas(sigmas)
    # Walk from high noise to clean data, one schedule segment per step.
    for step_index in range(step_count):
        # Read segment endpoints, then advance x one Heun step along sigma.
        sigma_current = sigmas[step_index].item()
        sigma_next = sigmas[step_index + 1].item()
        x = _heun_step(denoise, x, sigma_current, sigma_next)
    return x


def stochastic_sample(
    denoise: Denoise,
    x: Tensor,
    sigmas: Tensor,
    generator: torch.Generator,
    churn: Churn = DEFAULT_CHURN,
) -> Tensor:
    """EDM algorithm 2: at each step with sigma_min <= sigma_i <= sigma_max,
    raise the level to sigma_hat = sigma_i (1 + gamma) with gamma =
    min(amount / steps, sqrt(2) - 1) by adding noise of standard deviation
    noise_inflation * sqrt(sigma_hat^2 - sigma_i^2), then take the Heun step
    from sigma_hat to sigma_{i+1} (see churn for table 5 defaults)."""
    # Validate the schedule before integrating along sigma.
    step_count = _validate_sigmas(sigmas)
    # gamma never exceeds sqrt(2) - 1 so sigma_hat stays within EDM's bound.
    churn_cap = math.sqrt(2.0) - 1.0
    # Algorithm 1's Heun loop, with optional churn noise at each step.
    for step_index in range(step_count):
        # Segment endpoints in sigma set the Heun step and churn band.
        sigma_current = sigmas[step_index].item()
        sigma_next = sigmas[step_index + 1].item()
        working = x
        sigma_hat = sigma_current
        # Noise plus a higher level help the sampler forget earlier errors.
        if churn.sigma_min <= sigma_current <= churn.sigma_max:
            # gamma lifts sigma_hat above sigma_i before noise is added.
            gamma = min(churn.amount / step_count, churn_cap)
            sigma_hat = sigma_current * (1.0 + gamma)
            # Inflate noise to match variance of sigma_hat versus sigma_i.
            variance_gain = sigma_hat**2 - sigma_current**2
            noise_scale = churn.noise_inflation * math.sqrt(variance_gain)
            # Perturb x to the lifted level, then integrate down in sigma.
            standard_normal = torch.randn(
                x.shape, generator=generator, dtype=x.dtype, device=x.device
            )
            working = x + noise_scale * standard_normal
        # Heun step from sigma_hat (or sigma_i) down to sigma_{i+1}.
        x = _heun_step(denoise, working, sigma_hat, sigma_next)
    return x
