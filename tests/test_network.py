"""Neural denoiser training against independent ideal-denoiser oracles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from edm.diffusion import add_noise, ideal_denoiser, posterior_covariance
from edm.mixture import THREE_BLOBS
from edm.network import Denoiser, TrainConfig, train_denoiser
from edm.precondition import coefficients, sample_sigma, sigma_data_of

if TYPE_CHECKING:
    from torch import Tensor

_HELD_OUT_COUNT = 100_000
_TRAIN_STEPS_FULL = 1000
_TRAIN_STEPS_SHORT = 20


@pytest.fixture(scope="module", name="trained_model_and_losses")
def fixture_trained_model_and_losses() -> tuple[Denoiser, Tensor]:
    """One 1000-step run shared by convergence and learning-rate tests."""
    return train_denoiser(THREE_BLOBS, TrainConfig(steps=_TRAIN_STEPS_FULL))


def _gap_over_trace(model: Denoiser, noise_level: float) -> float:
    """Mean |D_net - D_ideal|^2 / mean trace Cov(x0|x) on held-out draws."""
    generator = torch.Generator().manual_seed(99)
    clean = THREE_BLOBS.sample(_HELD_OUT_COUNT, generator)
    sigma_tensor = torch.full(
        (_HELD_OUT_COUNT,), noise_level, dtype=torch.float64
    )
    noisy = add_noise(clean, sigma_tensor, generator)
    with torch.no_grad():
        d_net = model(noisy, sigma_tensor)
        d_ideal = ideal_denoiser(THREE_BLOBS, noisy, sigma_tensor)
        sq_gap = (d_net - d_ideal).square().sum(dim=-1)
        post_cov = posterior_covariance(THREE_BLOBS, noisy, sigma_tensor)
        trace_post = post_cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        return (sq_gap.mean() / trace_post.mean()).item()


def test_zero_network_is_skip_path() -> None:
    """All-zero weights make D equal c_skip x; F enters only through c_out."""
    sd = sigma_data_of(THREE_BLOBS)
    model = Denoiser(sd, hidden_width=128, hidden_layers=3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    generator = torch.Generator().manual_seed(0)
    points = THREE_BLOBS.sample(100, generator)
    sigma = sample_sigma(100, generator)
    with torch.no_grad():
        denoised = model(points, sigma)
        coef = coefficients(sigma, sd)
        expected = coef.c_skip.unsqueeze(-1) * points
        assert torch.allclose(denoised, expected, atol=1e-15)


@pytest.mark.parametrize("noise_level", (0.05, 0.5, 3.0))
def test_learns_ideal_denoiser(
    trained_model_and_losses: tuple[Denoiser, Tensor], noise_level: float
) -> None:
    """Gap over irreducible trace stays below 0.03 at each held-out sigma."""
    model, _losses = trained_model_and_losses
    model.eval()
    ratio = _gap_over_trace(model, noise_level)
    assert ratio < 0.03


def test_repeatable_training() -> None:
    """Same config gives identical weights and loss traces across two runs."""
    config = TrainConfig(steps=_TRAIN_STEPS_SHORT)
    model_a, losses_a = train_denoiser(THREE_BLOBS, config)
    model_b, losses_b = train_denoiser(THREE_BLOBS, config)
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key])
    assert torch.equal(losses_a, losses_b)


def test_learning_happens(
    trained_model_and_losses: tuple[Denoiser, Tensor],
) -> None:
    """Late training loss is below early loss in the 1000-step reference run."""
    _model, losses = trained_model_and_losses
    first_hundred = losses[:100].mean().item()
    last_hundred = losses[-100:].mean().item()
    assert last_hundred < first_hundred


def test_train_config_rejects_steps_below_one() -> None:
    """Decision table: steps < 1 names the field."""
    with pytest.raises(ValueError, match="steps"):
        TrainConfig(steps=0)


def test_train_config_rejects_batch_size_below_one() -> None:
    """Decision table: batch_size < 1 names the field."""
    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig(batch_size=0)


def test_train_config_rejects_hidden_width_below_one() -> None:
    """Decision table: hidden_width < 1 names the field."""
    with pytest.raises(ValueError, match="hidden_width"):
        TrainConfig(hidden_width=0)


def test_train_config_rejects_hidden_layers_below_one() -> None:
    """Decision table: hidden_layers < 1 names the field."""
    with pytest.raises(ValueError, match="hidden_layers"):
        TrainConfig(hidden_layers=0)


def test_train_config_rejects_non_positive_learning_rate() -> None:
    """Decision table: learning_rate <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig(learning_rate=0.0)


def test_train_config_rejects_non_finite_learning_rate() -> None:
    """Decision table: non-finite learning_rate raises ValueError."""
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig(learning_rate=float("nan"))
