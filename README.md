# Diffusion on a 2-D toy

A small diffusion model (EDM-style denoiser, deterministic and stochastic
samplers, exact likelihood, classifier guidance, and six predictions with
confidences) built on a two-dimensional Gaussian mixture, where every quantity
the model approximates also has a closed form. Each component is tested
against that exact answer before the same code moves to trajectory
prediction on the Waymo Open Motion Dataset.

![Data, trained sampler, guided samples with six predictions](docs/figure.png)

Left: 2,000 samples of the three-blob mixture over its log-density contours.
Middle: 2,000 samples from the trained denoiser, Heun sampler, 32 noise
levels. Right: the same sampler guided toward one blob, with the six k-means
predictions a planner would consume (`make figure`, 2,000 training steps,
about 50 s on two CPU threads).

## Why a toy

On a Gaussian mixture the ideal denoiser, the score, the noised density and
the probability-flow trajectories are known exactly. So instead of judging a
diffusion model by how its samples look, every piece is compared with the
right answer:

| Component | Checked against |
|---|---|
| Noising, ideal denoiser, score | closed form of the noised mixture; score equals autograd of the exact log density; limits at tiny and huge noise |
| Training loss and EDM preconditioning | the skip path for a zero network; the trained denoiser matches the ideal one within the test tolerance |
| Heun probability-flow sampler | the closed-form flow on a single Gaussian (second-order convergence); mixture moments reproduced |
| Stochastic sampler | mixture reproduced; denoiser call counts equal EDM's |
| Log-likelihood through the probability-flow ODE | the exact density of the noised mixture; divergence against a linear closed form |
| Six predictions with confidences | k-means on samples, confidences summing to one |
| Classifier guidance | the exact mixture classifier's gradient against autograd; unit-scale guidance equals the target blob's own score |

`make test` runs all 97 tests (about 2 minutes on a laptop CPU).

## Run it

```
python -m pip install torch matplotlib pytest
make test                           # the exact-answer tests
make figure                         # writes docs/figure.png
python train.py --out denoiser.pt   # train the denoiser MLP alone
```

## Layout

One-way dependencies: `mixture` -> `diffusion` -> `loss` -> `precondition` ->
`network` -> `sample` -> `likelihood`, `predictions`, `guidance`, all in
`toy/`. Tests run in float64 and batch over points and blobs; nothing loops
over samples.

The denoiser is a 3 x 128 SiLU MLP with EDM's input, output and noise
preconditioning, trained on fresh mixture samples each step (Adam, learning
rate 1e-3, batch 4,096).
