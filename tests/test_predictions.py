"""k-means prediction collapse against mixture oracles."""

import pytest
import torch
from torch import Tensor

from toy.mixture import THREE_BLOBS, GaussianMixture
from toy.predictions import Predictions, kmeans_predictions


def _centre_blob_ownership(centres: Tensor, mixture: GaussianMixture) -> Tensor:
    """Oracle: blob index of highest per-blob density at each centre."""
    component_count = mixture.weights.shape[0]
    log_density = torch.stack(
        [
            torch.distributions.MultivariateNormal(
                mixture.means[blob], mixture.covariances[blob]
            ).log_prob(centres)
            for blob in range(component_count)
        ],
        dim=-1,
    )
    return log_density.argmax(dim=-1)


def _confidence_mass_per_blob(
    predictions: Predictions, mixture: GaussianMixture
) -> Tensor:
    """Sum confidences of centres owned by each blob (oracle grouping)."""
    component_count = mixture.weights.shape[0]
    ownership = _centre_blob_ownership(predictions.centres, mixture)
    masses = torch.zeros(component_count, dtype=torch.float64)
    for blob in range(component_count):
        blob_mask = ownership == blob
        masses[blob] = predictions.confidences[blob_mask].sum()
    return masses


def _mean_squared_nearest_distance(
    samples: Tensor, predictions: Predictions
) -> float:
    """Oracle objective: mean squared distance to the nearest centre."""
    distances = torch.cdist(samples, predictions.centres, p=2)
    nearest_distance = distances.min(dim=1).values
    return nearest_distance.square().mean().item()


def _centres_sorted_by_nearest_mean(centres: Tensor, means: Tensor) -> Tensor:
    """Order centres by the index of the closest blob mean (test oracle)."""
    nearest_mean = torch.cdist(centres, means).argmin(dim=1)
    return centres[torch.argsort(nearest_mean)]


def test_confidence_masses_match_blob_weights() -> None:
    """Six modes on THREE_BLOBS: grouped confidences track blob weights."""
    sample_count = 20_000
    generator = torch.Generator().manual_seed(0)
    samples = THREE_BLOBS.sample(sample_count, generator)
    generator = torch.Generator().manual_seed(0)
    predictions = kmeans_predictions(samples, generator, k=6)
    masses = _confidence_mass_per_blob(predictions, THREE_BLOBS)
    ownership = _centre_blob_ownership(predictions.centres, THREE_BLOBS)
    component_count = THREE_BLOBS.weights.shape[0]
    for blob in range(component_count):
        assert (ownership == blob).any()
        assert abs(masses[blob] - THREE_BLOBS.weights[blob]) < 0.02
    # Outside answer: summed masses ~0.4948, 0.3028, 0.2024 at n=20000.
    assert abs(predictions.confidences.sum().item() - 1.0) < 1e-12


def test_k_three_finds_blob_means() -> None:
    """k=3 collapses to the three blob centres on THREE_BLOBS."""
    sample_count = 20_000
    generator = torch.Generator().manual_seed(1)
    samples = THREE_BLOBS.sample(sample_count, generator)
    generator = torch.Generator().manual_seed(1)
    predictions = kmeans_predictions(samples, generator, k=3)
    sorted_centres = _centres_sorted_by_nearest_mean(
        predictions.centres, THREE_BLOBS.means
    )
    centre_error = (sorted_centres - THREE_BLOBS.means).abs().max().item()
    assert centre_error < 0.05
    # Outside answer: centre_error max ~0.008 after sorting.


def test_lloyd_objective_is_nonincreasing() -> None:
    """More Lloyd steps never raise mean squared distance to centres."""
    sample_count = 5_000
    generator = torch.Generator().manual_seed(2)
    samples = THREE_BLOBS.sample(sample_count, generator)
    objectives: list[float] = []
    for iteration_count in range(1, 11):
        run_generator = torch.Generator().manual_seed(2)
        predictions = kmeans_predictions(
            samples, run_generator, k=6, max_iterations=iteration_count
        )
        objectives.append(_mean_squared_nearest_distance(samples, predictions))
    for step in range(1, len(objectives)):
        assert objectives[step] <= objectives[step - 1] + 1e-12


def test_kmeans_predictions_is_repeatable() -> None:
    """Equal seeds reproduce centres and confidences exactly."""
    samples = THREE_BLOBS.sample(1_000, torch.Generator().manual_seed(3))
    first = kmeans_predictions(samples, torch.Generator().manual_seed(4))
    second = kmeans_predictions(samples, torch.Generator().manual_seed(4))
    assert torch.equal(first.centres, second.centres)
    assert torch.equal(first.confidences, second.confidences)


def test_confidences_sorted_largest_first() -> None:
    """Mode shares are reported in non-increasing order."""
    samples = THREE_BLOBS.sample(2_000, torch.Generator().manual_seed(5))
    predictions = kmeans_predictions(
        samples, torch.Generator().manual_seed(5), k=6
    )
    if predictions.confidences.shape[0] > 1:
        descending = predictions.confidences[:-1] >= predictions.confidences[1:]
        assert descending.all()


def test_kmeans_predictions_validates_inputs() -> None:
    """Each validation rule raises ValueError."""
    generator = torch.Generator().manual_seed(0)
    valid = THREE_BLOBS.sample(10, generator)
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        kmeans_predictions(valid[:, :1], generator)
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        kmeans_predictions(valid.flatten(), generator)
    with pytest.raises(ValueError, match="at least k=6"):
        kmeans_predictions(valid[:5], generator, k=6)
    with pytest.raises(ValueError, match="k must be at least 1"):
        kmeans_predictions(valid, generator, k=0)
    with pytest.raises(ValueError, match="max_iterations must be at least 1"):
        kmeans_predictions(valid, generator, max_iterations=0)
