---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: ptgp
---

# Random Walk Kernel

{ref}`Non-stationary <non-stationary>`, {ref}`1-D <one-d>`, {ref}`Rough <rough>`

The random walk kernel — the covariance function of standard Brownian motion (the Wiener process) — is the textbook example of a non-stationary GP kernel. Its covariance depends not just on the distance between inputs but on their actual locations: $k(x, y) = \min(x, y)$ means that variance grows linearly with the input value, and two points are correlated only via their shared "history" from the origin.

Use this kernel for processes that genuinely accumulate randomness over time: drifting hardware sensors, financial paths, particle trajectories. Sample paths are continuous but nowhere differentiable, qualitatively similar to Matérn 1/2 but with an explicit notion of an origin from which uncertainty grows.

## Key properties and parameters

```{eval-rst}
=================  ===================================================
Domain             :math:`x \in \mathbb{R}^+` (positive scalars, 1-D)
Stationary         **No** — variance grows linearly with input
Sample smoothness  :math:`C^0` — continuous, nowhere differentiable
Variance           :math:`k(x, x) = x`
=================  ===================================================
```

Constructor: [`ptgp.kernels.RandomWalk(input_dim=1, active_dims=None)`](../../api/generated/ptgp.kernels.RandomWalk.rst). The kernel has no shape parameters of its own; scale it with `eta**2 * RandomWalk()`. It operates on a single input column — pass `active_dims=[k]` to pick a column when `input_dim > 1`.

### Covariance function

$$
k(x, y) = \min(x, y)
$$

for $x, y \in \mathbb{R}^+$. Two inputs share the variance contributed by their common interval from the origin to the smaller of the two; beyond that point their increments are independent.

::::::{tab-set}
:::::{tab-item} Prior samples

```{jupyter-execute}
:hide-code:
import ptgp as pg
from ptgp.plotting import plot_prior_samples
plot_prior_samples(
    [("Random Walk", pg.kernels.RandomWalk(input_dim=1))],
    n_samples=4,
    x_range=(0.05, 5.0),
)
```
:::::

:::::{tab-item} Posterior given 3 observations

```{jupyter-execute}
:hide-code:
import numpy as np
from ptgp.plotting import plot_conditional
plot_conditional(
    [("Random Walk", pg.kernels.RandomWalk(input_dim=1))],
    n_draws=3,
    x_range=(0.05, 5.0),
    obs_x=np.array([[0.6], [1.6], [4.0]]),
    obs_y=np.array([0.45, 0.55, -1.10]),
)
```
:::::

:::::{tab-item} Code

```{jupyter-execute}
import numpy as np
import pytensor
import pytensor.tensor as pt
import ptgp as pg

kernel = pg.kernels.RandomWalk(input_dim=1)

X_sym = pt.matrix("X", shape=(None, 1))
K_fn = pytensor.function([X_sym], kernel(X_sym))

X = np.linspace(0.05, 5.0, 200)[:, None]
K = K_fn(X)
print(K.shape, K.diagonal()[:3])
```
:::::
::::::

Prior samples fan out from the origin; the posterior pinches around each observation and the uncertainty between them is bounded by how the Brownian variance accumulates between data points. Far past the last observation, the posterior reverts toward the prior and uncertainty grows without bound.

```{seealso}
- [Matern12](matern12.md) — stationary kernel that also produces nowhere-differentiable sample paths, but without the Brownian "variance grows from an origin" structure.
- [Matern52](matern52.md) — smooth stationary alternative; use this when "drift over time" is better modeled as a smooth trend than as accumulating randomness.
```

The random walk kernel and the broader connection between GPs and stochastic processes are covered in {cite:t}`rasmussen-williams-2006` Chapter 4.

```{bibliography}
:filter: docname in docnames
```
