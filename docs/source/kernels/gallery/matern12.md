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

# Matérn 1/2 (Exponential) Kernel

[Stationary](../gallery_tags.rst#stationary), [Isotropic](../gallery_tags.rst#isotropic), [Very Rough](../gallery_tags.rst#very-rough)

The Matérn 1/2 kernel — also known as the **exponential kernel** — produces sample functions that are continuous but nowhere differentiable. The resulting GP is equivalent to an Ornstein–Uhlenbeck process: a mean-reverting random walk in continuous time. Sample paths look like noisy zig-zag trajectories rather than smooth curves.

This kernel is rarely appropriate as a model of a deterministic underlying function — its sample paths are simply too rough for most real-world signals — but it is useful as a *control* against which to compare smoother kernels, and as a building block for compositions where rapid local variation is desired on top of a smoother trend.

## Key properties and parameters

```{eval-rst}
=================  =================================================
Domain             :math:`x \in \mathbb{R}^D`
Stationary         Yes
Sample smoothness  :math:`C^0` — continuous, nowhere differentiable
Variance           :math:`k(x, x) = 1`
=================  =================================================
```

Constructor: [`ptgp.kernels.Matern12(input_dim, ls, active_dims=None)`](../../api/generated/ptgp.kernels.Matern12.rst). The lengthscale `ls` can be scalar (isotropic) or a vector of length `len(active_dims)` (per-dimension). Scale with `eta**2 * Matern12(...)`.

### Covariance function

$$
k(x, y) = \exp\!\left( -r \right),
\qquad r = \frac{\lVert x - y \rVert}{\ell}
$$

where $\ell$ is the lengthscale (`ls`). This is the slowest-decaying member of the Matérn family at long distances.

::::::{tab-set}
:::::{tab-item} Kernel decay
:sync: ls

```{jupyter-execute}
:hide-code:
import ptgp as pg
from ptgp.plotting import plot_kernel_decay
plot_kernel_decay(
    [
        (r"$\ell=0.3$", pg.kernels.Matern12(input_dim=1, ls=0.3)),
        (r"$\ell=1.0$", pg.kernels.Matern12(input_dim=1, ls=1.0)),
        (r"$\ell=3.0$", pg.kernels.Matern12(input_dim=1, ls=3.0)),
    ],
    max_distance=5.0,
)
```
:::::

:::::{tab-item} Prior samples
:sync: ls

```{jupyter-execute}
:hide-code:
from ptgp.plotting import plot_prior_samples
plot_prior_samples(
    [
        (r"$\ell=0.3$", pg.kernels.Matern12(input_dim=1, ls=0.3)),
        (r"$\ell=1.0$", pg.kernels.Matern12(input_dim=1, ls=1.0)),
        (r"$\ell=3.0$", pg.kernels.Matern12(input_dim=1, ls=3.0)),
    ],
    n_samples=4,
)
```
:::::

:::::{tab-item} Posterior given 3 observations
:sync: ls

```{jupyter-execute}
:hide-code:
from ptgp.plotting import plot_conditional
plot_conditional(
    [
        (r"$\ell=0.3$", pg.kernels.Matern12(input_dim=1, ls=0.3)),
        (r"$\ell=1.0$", pg.kernels.Matern12(input_dim=1, ls=1.0)),
        (r"$\ell=3.0$", pg.kernels.Matern12(input_dim=1, ls=3.0)),
    ],
    n_draws=3,
)
```
:::::

:::::{tab-item} Code

```{jupyter-execute}
import numpy as np
import pytensor
import pytensor.tensor as pt
import ptgp as pg

kernel = pg.kernels.Matern12(input_dim=1, ls=1.0)

X_sym = pt.matrix("X", shape=(None, 1))
K_fn = pytensor.function([X_sym], kernel(X_sym))

X = np.linspace(-3, 3, 200)[:, None]
K = K_fn(X)
print(K.shape, K.diagonal()[:3])
```
:::::
::::::

```{seealso}
- [Matern32](matern32.md) — same family at $\nu = 3/2$. Once mean-square differentiable; visibly smoother draws.
- [Matern52](matern52.md) — same family at $\nu = 5/2$. Twice differentiable; the standard recommended default.
- [RandomWalk](randomwalk.md) — also produces nowhere-differentiable sample paths, but non-stationary (variance grows with input).
```

The Matérn family and its smoothness-tuning role in GP modeling is discussed in {cite:t}`rasmussen-williams-2006` Chapter 4.

```{bibliography}
:filter: docname in docnames
```
