# Diffusion on a 2-D toy

An EDM-style diffusion model on a three-component 2-D Gaussian mixture:
denoiser, deterministic and stochastic samplers, log-likelihood through the
probability-flow ODE, classifier guidance, and six predictions with
confidences by k-means. Each component is tested against the mixture's closed
form.

![Data, trained sampler, guided samples with six predictions](docs/figure.png)

Left: 2,000 mixture samples over log-density contours. Middle: 2,000 samples
from the trained denoiser, Heun sampler, 32 noise levels. Right: samples
guided to one component, with the six k-means predictions.

## Tests

| Component | Reference |
|---|---|
| Noising, ideal denoiser, score | closed form of the noised mixture; autograd of the exact log density; limits at tiny and huge noise |
| Loss and EDM preconditioning | skip path for a zero network; trained denoiser against the ideal one, within tolerance |
| Heun sampler | closed-form flow on a single Gaussian; mixture moments |
| Stochastic sampler | mixture moments; denoiser call counts |
| Log-likelihood (probability-flow ODE) | exact density of the noised mixture; divergence against a linear closed form |
| Predictions | k-means centres and confidences summing to one |
| Classifier guidance | exact mixture classifier gradient against autograd; unit-scale guidance against the component's score |

## Run

```
python -m pip install torch matplotlib pytest
make test                           # 97 tests
make figure                         # docs/figure.png
python train.py --out denoiser.pt   # train the denoiser
```

## Layout

`toy/`: `mixture` -> `diffusion` -> `loss` -> `precondition` -> `network` ->
`sample` -> `likelihood`, `predictions`, `guidance`. Denoiser: 3 x 128 SiLU
MLP with EDM preconditioning, Adam 1e-3, batch 4,096, fresh mixture samples
each step.
