"""Collapse many diffusion samples into a small set of mode predictions."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class Predictions(NamedTuple):
    """k cluster centres (k, 2) and their shares of the samples (k,),
    summing to 1, sorted by share, largest first."""

    centres: Tensor
    confidences: Tensor


def _validate_kmeans_inputs(
    samples: Tensor, k: int, max_iterations: int
) -> None:
    """Reject sample batches and hyperparameters k-means cannot use."""
    if samples.dim() != 2 or samples.shape[-1] != 2:
        raise ValueError(
            f"samples must have shape (n, 2), got {tuple(samples.shape)}."
        )
    sample_count = samples.shape[0]
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}.")
    if sample_count < k:
        raise ValueError(f"need at least k={k} samples, got n={sample_count}.")
    if max_iterations < 1:
        raise ValueError(
            f"max_iterations must be at least 1, got {max_iterations}."
        )


def _kmeans_plus_plus_centres(
    samples: Tensor, k: int, generator: torch.Generator
) -> Tensor:
    """Pick k initial centres via k-means++ on the diffusion sample cloud."""
    sample_count = samples.shape[0]
    first_index = torch.randint(sample_count, (1,), generator=generator).item()
    chosen = [first_index]
    # Squared distance from each sample to its nearest chosen seed, (n,);
    # kept up to date with one new column per seed, not recomputed.
    nearest_squared = (samples - samples[first_index]).square().sum(dim=1)
    # Each new seed favours points far from the seeds chosen so far.
    for _ in range(1, k):
        weights = nearest_squared
        if weights.sum().item() == 0.0:
            # Every sample sits on a seed: any further seed is as good.
            weights = torch.ones_like(nearest_squared)
        next_index = torch.multinomial(weights, 1, generator=generator).item()
        chosen.append(next_index)
        to_new = (samples - samples[next_index]).square().sum(dim=1)
        nearest_squared = torch.minimum(nearest_squared, to_new)
    return samples[chosen]


def _assign(samples: Tensor, centres: Tensor) -> Tensor:
    """Index of each sample's nearest centre, (n,); distances are (n, k)."""
    return torch.cdist(samples, centres).argmin(dim=1)


def _lloyd_iterate(
    samples: Tensor, centres: Tensor, max_iterations: int
) -> tuple[Tensor, Tensor]:
    """Lloyd assignment and update until labels stabilize or the cap; the
    returned labels always belong to the returned centres."""
    cluster_count = centres.shape[0]
    assignments = _assign(samples, centres)
    for _ in range(max_iterations):
        # Each centre moves to the mean of its members: one pass sums the
        # members per cluster and one counts them.
        sums = torch.zeros_like(centres).index_add_(0, assignments, samples)
        counts = torch.bincount(assignments, minlength=cluster_count)
        occupied = counts > 0
        centres = centres.clone()
        # An empty cluster keeps its previous centre.
        centres[occupied] = sums[occupied] / counts[occupied].unsqueeze(-1)
        updated = _assign(samples, centres)
        if torch.equal(updated, assignments):
            break
        assignments = updated
    return centres, assignments


def kmeans_predictions(
    samples: Tensor,
    generator: torch.Generator,
    *,
    k: int = 6,
    max_iterations: int = 100,
) -> Predictions:
    """k-means++ seeds (Arthur and Vassilvitskii 2007: each new seed drawn
    with probability proportional to its squared distance from the nearest
    seed so far), then Lloyd iterations until no sample changes cluster or
    max_iterations; an empty cluster keeps its previous centre."""
    _validate_kmeans_inputs(samples, k, max_iterations)
    centres = _kmeans_plus_plus_centres(samples, k, generator)
    centres, assignments = _lloyd_iterate(samples, centres, max_iterations)
    sample_count = samples.shape[0]
    # Each confidence is that mode's share of the posterior sample cloud.
    cluster_counts = torch.bincount(assignments, minlength=k).to(
        dtype=samples.dtype
    )
    confidences = cluster_counts / sample_count
    largest_first = torch.argsort(confidences, descending=True)
    return Predictions(
        centres=centres[largest_first], confidences=confidences[largest_first]
    )
