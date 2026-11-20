"""Train a denoising network on mixture samples and save the checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from toy.mixture import THREE_BLOBS
from toy.network import TrainConfig, train_denoiser


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the EDM denoiser MLP on THREE_BLOBS."
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    overrides: dict[str, int] = {}
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.seed is not None:
        overrides["seed"] = args.seed
    config = TrainConfig(**overrides)
    model, _losses = train_denoiser(THREE_BLOBS, config)
    torch.save(
        {"config": dataclasses.asdict(config), "state": model.state_dict()},
        args.out,
    )


if __name__ == "__main__":
    main()
