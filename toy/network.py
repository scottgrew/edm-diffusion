"""Train a small MLP to match the ideal denoiser on mixture samples."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from toy.diffusion import add_noise
from toy.mixture import GaussianMixture
from toy.precondition import (
    coefficients,
    denoiser_from_network,
    network_target,
    sample_sigma,
    sigma_data_of,
)


@dataclass(frozen=True)
class TrainConfig:
    """Everything that decides a training run; the seed makes it repeatable."""

    steps: int = 4000
    batch_size: int = 4096
    learning_rate: float = 1e-3
    hidden_width: int = 128
    hidden_layers: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject hyperparameters that cannot define a valid training run."""
        if self.steps < 1:
            raise ValueError(f"steps must be at least 1, got {self.steps}.")
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be at least 1, got {self.batch_size}."
            )
        if self.hidden_width < 1:
            raise ValueError(
                f"hidden_width must be at least 1, got {self.hidden_width}."
            )
        if self.hidden_layers < 1:
            raise ValueError(
                f"hidden_layers must be at least 1, got {self.hidden_layers}."
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be finite and positive, "
                f"got {self.learning_rate}."
            )


def _build_mlp(hidden_width: int, hidden_layers: int) -> nn.Sequential:
    """SiLU MLP: 3 inputs, two outputs, float64 weights throughout."""
    modules: list[nn.Module] = [
        nn.Linear(3, hidden_width, dtype=torch.float64),
        nn.SiLU(),
    ]
    # Each extra hidden layer lets F bend more sharply between the blobs.
    for _ in range(hidden_layers - 1):
        modules.append(
            nn.Linear(hidden_width, hidden_width, dtype=torch.float64)
        )
        modules.append(nn.SiLU())
    modules.append(nn.Linear(hidden_width, 2, dtype=torch.float64))
    return nn.Sequential(*modules)


class Denoiser(torch.nn.Module):
    """D(x; sigma) = c_skip x + c_out F(c_in x, c_noise), with F a SiLU MLP
    on the three numbers (c_in x, c_noise) and two outputs, float64."""

    def __init__(
        self, sigma_data: float, hidden_width: int, hidden_layers: int
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.mlp = _build_mlp(hidden_width, hidden_layers)

    def network(self, x: Tensor, sigma: Tensor) -> Tensor:
        """F's output, shape (n, 2): what network_target asks for."""
        coef = coefficients(sigma, self.sigma_data)
        # c_in scales x; (n,) coeffs broadcast onto both coordinates.
        scaled = coef.c_in.unsqueeze(-1) * x
        # F sees unit-variance input coords plus the log-noise conditioning.
        noise_level = coef.c_noise.unsqueeze(-1)
        features = torch.cat([scaled, noise_level], dim=-1)
        return self.mlp(features)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """The denoised point D(x; sigma), shape (n, 2)."""
        return denoiser_from_network(
            self.network(x, sigma), x, sigma, self.sigma_data
        )


def train_denoiser(
    mixture: GaussianMixture, config: TrainConfig
) -> tuple[Denoiser, Tensor]:
    """Trains on fresh mixture samples every step and returns the model and
    the loss of every step, shape (steps,)."""
    sd = sigma_data_of(mixture)
    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        model = Denoiser(sd, config.hidden_width, config.hidden_layers)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )
    # One generator for every training draw; seed+1 is separate from init.
    generator = torch.Generator().manual_seed(config.seed + 1)
    # One scalar loss per step, shape (steps,).
    losses = torch.empty(config.steps, dtype=torch.float64)
    model.train()
    for step in range(config.steps):
        # Fresh clean points, noise levels and noise every step: the mixture
        # is an endless supply of data, so the network never sees a repeat.
        x0 = mixture.sample(config.batch_size, generator)
        sigma = sample_sigma(config.batch_size, generator)
        x = add_noise(x0, sigma, generator)
        prediction = model.network(x, sigma)
        target = network_target(x0, x, sigma, sd)
        # Mean |F - target|^2; the target already carries EDM scaling.
        per_point = (prediction - target).square().sum(dim=-1)
        loss = per_point.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[step] = loss.detach()
    return model, losses
