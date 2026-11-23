"""Gaussian mixtures: sampling, density, and widening blobs with noise."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GaussianMixture:
    """A weighted sum of two-dimensional Gaussian components."""

    weights: Tensor
    means: Tensor
    covariances: Tensor

    def __post_init__(self) -> None:
        """Reject parameters that cannot define a valid mixture density."""
        number_of_components = self.weights.shape[0]
        if self.means.shape != (number_of_components, 2):
            raise ValueError(
                "means must have shape (K, 2) matching weights length K="
                f"{number_of_components}, got {tuple(self.means.shape)}."
            )
        if self.covariances.shape != (number_of_components, 2, 2):
            raise ValueError(
                "covariances must have shape (K, 2, 2) matching weights "
                f"length K={number_of_components}, got "
                f"{tuple(self.covariances.shape)}."
            )
        if (self.weights <= 0).any():
            raise ValueError("every mixture weight must be strictly positive.")
        weight_sum = self.weights.sum().item()
        if abs(weight_sum - 1.0) > 1e-12:
            raise ValueError(
                "mixture weights must sum to 1 within 1e-12, "
                f"got sum {weight_sum}."
            )
        for index in range(number_of_components):
            covariance = self.covariances[index]
            if not torch.allclose(covariance, covariance.T):
                raise ValueError(
                    f"covariance at component {index} must be symmetric."
                )
            try:
                torch.linalg.cholesky(covariance)
            except RuntimeError as error:
                raise ValueError(
                    f"covariance at component {index} must be positive "
                    f"definite (Cholesky failed): {error}"
                ) from error

    def sample(self, n: int, generator: torch.Generator) -> Tensor:
        """Draw n points by picking a component then sampling that Gaussian."""
        # Which blob each draw comes from, according to the mixture weights.
        component_index = torch.multinomial(
            self.weights, n, replacement=True, generator=generator
        )
        # Factor each blob's covariance once (K factorisations, not n), then
        # look up the chosen blob's centre and factor for every draw.
        cholesky_factors = torch.linalg.cholesky(self.covariances)
        chosen_means = self.means[component_index]
        chol_factor = cholesky_factors[component_index]
        # A standard normal pushed through the blob's Cholesky factor has
        # exactly that blob's covariance.
        standard_normal = torch.randn(
            n, 2, generator=generator, dtype=torch.float64
        ).unsqueeze(-1)
        offset = (chol_factor @ standard_normal).squeeze(-1)
        return chosen_means + offset

    def log_prob(self, x: Tensor) -> Tensor:
        """Log density at each row of x via PyTorch's mixture distribution."""
        # PyTorch's mixture class is the trusted density reference.
        component_distribution = torch.distributions.MultivariateNormal(
            loc=self.means, covariance_matrix=self.covariances
        )
        categorical = torch.distributions.Categorical(probs=self.weights)
        mixture = torch.distributions.MixtureSameFamily(
            categorical, component_distribution
        )
        return mixture.log_prob(x)

    def noised(self, sigma: float) -> GaussianMixture:
        """Return the mixture after isotropic noise of scale sigma."""
        # A sum of independent Gaussians is Gaussian, so noising each blob
        # only widens its covariance by sigma squared along both axes.
        identity = torch.eye(2, dtype=torch.float64)
        widened_covariances = self.covariances + (sigma**2) * identity
        return GaussianMixture(
            weights=self.weights,
            means=self.means,
            covariances=widened_covariances,
        )


THREE_BLOBS = GaussianMixture(
    weights=torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64),
    means=torch.tensor(
        [[-2.0, 0.0], [2.0, 1.0], [0.0, -2.5]], dtype=torch.float64
    ),
    covariances=torch.tensor(
        [
            [[0.3, 0.1], [0.1, 0.2]],
            [[0.15, 0.0], [0.0, 0.4]],
            [[0.25, -0.08], [-0.08, 0.25]],
        ],
        dtype=torch.float64,
    ),
)
