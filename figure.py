"""Draw docs/figure.png: the data, the trained sampler, and guided samples with
their six k-means predictions, all on the THREE_BLOBS mixture."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from toy.guidance import guided_denoiser
from toy.mixture import THREE_BLOBS
from toy.network import TrainConfig, train_denoiser
from toy.predictions import kmeans_predictions
from toy.sample import heun_sample, initial_noise, karras_sigmas

POINTS = 2000
SAMPLER_STEPS = 32
GUIDED_BLOB = 0
GUIDANCE_SCALE = 1.0


def density_grid(extent: float, cells: int = 200):
    """The mixture's log density on a square grid, for contour lines."""
    axis = torch.linspace(-extent, extent, cells, dtype=torch.float64)
    xs, ys = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)
    log_p = THREE_BLOBS.log_prob(points).reshape(cells, cells)
    return xs.numpy(), ys.numpy(), log_p.numpy()


def sample_with(denoise, seed: int) -> torch.Tensor:
    """POINTS samples from noise by the Heun probability-flow sampler."""
    generator = torch.Generator().manual_seed(seed)
    sigmas = karras_sigmas(SAMPLER_STEPS)
    x = initial_noise(POINTS, float(sigmas[0]), generator)
    return heun_sample(denoise, x, sigmas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/figure.png"))
    parser.add_argument("--train-steps", type=int, default=2000)
    args = parser.parse_args()

    torch.set_num_threads(2)
    data = THREE_BLOBS.sample(POINTS, torch.Generator().manual_seed(0))
    model, _ = train_denoiser(THREE_BLOBS, TrainConfig(steps=args.train_steps))
    model.eval()
    with torch.no_grad():
        learned = sample_with(model, seed=1)
    guided = sample_with(
        guided_denoiser(THREE_BLOBS, GUIDED_BLOB, GUIDANCE_SCALE), seed=2
    )
    predictions = kmeans_predictions(guided, torch.Generator().manual_seed(3))

    extent = float(data.abs().max()) * 1.15
    xs, ys, log_p = density_grid(extent)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    panels = [
        (data, "Data: three-blob Gaussian mixture"),
        (learned, f"Trained denoiser, Heun sampler ({SAMPLER_STEPS} steps)"),
        (guided, f"Guided to blob {GUIDED_BLOB} + six k-means predictions"),
    ]
    for ax, (points, title) in zip(axes, panels, strict=True):
        ax.contour(xs, ys, log_p, levels=12, colors="0.75", linewidths=0.6)
        ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.45, color="C0")
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
    centres = predictions[0]
    axes[2].scatter(
        centres[:, 0],
        centres[:, 1],
        marker="*",
        s=220,
        color="C3",
        edgecolor="k",
        label="predictions",
    )
    axes[2].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
