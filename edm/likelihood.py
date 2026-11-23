"""Exact log-density via the probability-flow ODE and its divergence."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from edm.sample import Denoise

PLANE_DIMENSION = 2


def _sigma_level_tensor(
    sigma: float, batch_size: int, device: torch.device
) -> Tensor:
    """One scalar noise level broadcast for every row of a batch."""
    # The denoiser expects one sigma per batch row at a shared noise level.
    return torch.full((batch_size,), sigma, dtype=torch.float64, device=device)


def _validate_likelihood_sigmas(sigmas: Tensor) -> int:
    """Reject schedules that cannot integrate likelihood along sigma upward."""
    # Integration needs at least one segment between two noise levels.
    if sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("sigmas must be a length at least 2 vector.")
    # Likelihood integration walks from low sigma up toward high sigma.
    if not (sigmas[1:] > sigmas[:-1]).all():
        raise ValueError("sigmas must be strictly increasing.")
    # Every level must stay positive so the drift division stays finite.
    if (sigmas <= 0).any():
        raise ValueError("sigmas must be positive.")
    return sigmas.numel() - 1


def drift_and_divergence(
    denoise: Denoise, x: Tensor, sigma: float
) -> tuple[Tensor, Tensor]:
    """The ODE drift dx/dsigma = (x - D(x; sigma)) / sigma, shape (n, 2),
    and its divergence (trace of d drift / dx) per point, shape (n,), exact
    by autograd: one backward pass per coordinate (two in 2-D)."""
    # Each row needs its own Jacobian diagonal for the divergence trace.
    batch_size = x.shape[0]
    with_grad = x.detach().requires_grad_(True)
    sigma_batch = _sigma_level_tensor(sigma, batch_size, with_grad.device)
    denoised = denoise(with_grad, sigma_batch)
    # Drift pushes noisy points along the probability-flow direction in x.
    drift = (with_grad - denoised) / sigma
    divergence = torch.zeros(batch_size, dtype=torch.float64, device=x.device)
    for coordinate in range(PLANE_DIMENSION):
        # Pick one drift component and backprop to its diagonal sensitivity.
        unit_mask = torch.zeros_like(drift)
        unit_mask[:, coordinate] = 1.0
        retain = coordinate < PLANE_DIMENSION - 1
        (partials,) = torch.autograd.grad(
            drift, with_grad, grad_outputs=unit_mask, retain_graph=retain
        )
        divergence = divergence + partials[:, coordinate]
    return drift.detach(), divergence.detach()


def _heun_likelihood_step(
    denoise: Denoise, x: Tensor, sigma_from: float, sigma_to: float
) -> tuple[Tensor, Tensor, Tensor]:
    """One Heun step upward in sigma; drifts (n, 2) and divergences (n,)."""
    # Step size is positive because sigma rises along the likelihood path.
    step_size = sigma_to - sigma_from
    drift_at_start, div_at_start = drift_and_divergence(denoise, x, sigma_from)
    euler_point = x + step_size * drift_at_start
    drift_at_end, div_at_end = drift_and_divergence(
        denoise, euler_point, sigma_to
    )
    # Heun averages endpoint drifts for second-order flow accuracy.
    averaged_drift = 0.5 * drift_at_start + 0.5 * drift_at_end
    advanced = x + step_size * averaged_drift
    return advanced, div_at_start, div_at_end


def log_likelihood(
    denoise: Denoise,
    x: Tensor,
    sigmas: Tensor,
    prior_log_prob: Callable[[Tensor], Tensor],
) -> Tensor:
    """log p_{sigmas[0]}(x) for each point, shape (n,): Heun steps along the
    increasing sigmas, accumulating the trapezoidal integral of the
    divergence, then prior_log_prob at the last sigma's end point."""
    step_count = _validate_likelihood_sigmas(sigmas)
    state = x
    # Toward higher sigma the flow spreads points; div adds to log p uphill.
    div_integral = torch.zeros(
        state.shape[0], dtype=torch.float64, device=state.device
    )
    for step_index in range(step_count):
        sigma_current = sigmas[step_index].item()
        sigma_next = sigmas[step_index + 1].item()
        state, div_start, div_end = _heun_likelihood_step(
            denoise, state, sigma_current, sigma_next
        )
        # Detach state so one denoiser autograd graph is active per step.
        state = state.detach()
        segment = sigmas[step_index + 1] - sigmas[step_index]
        div_integral = div_integral + 0.5 * (div_start + div_end) * segment
    return prior_log_prob(state) + div_integral
