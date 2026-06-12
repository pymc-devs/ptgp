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

# Matérn 3/2 Kernel

[Stationary](../gallery_tags.rst#stationary), [Isotropic](../gallery_tags.rst#isotropic), [Rough](../gallery_tags.rst#rough)

The Matérn 3/2 kernel produces sample functions that are once mean-square differentiable but no smoother — the function itself and its first derivative exist and are continuous, but the second derivative does not. This is the right kernel when the process you are modeling has a clearly defined velocity but acceleration is a meaningless concept (e.g. abrupt regime changes, piecewise-linear-ish behavior on small scales).

Matérn 3/2 sits between Matérn 5/2 (twice differentiable, the default smooth kernel) and Matérn 1/2 (the exponential kernel, nowhere differentiable). Compared to Matérn 5/2, draws look visibly more jittery while still respecting overall trends.

## Key properties and parameters

```{eval-rst}
=================  =================================================
Domain             :math:`x \in \mathbb{R}^D`
Stationary         Yes
Sample smoothness  :math:`C^1` — once mean-square differentiable
Variance           :math:`k(x, x) = 1`
=================  =================================================
```

Constructor: [`ptgp.kernels.Matern32(input_dim, ls, active_dims=None)`](../../api/generated/ptgp.kernels.Matern32.rst). The lengthscale `ls` can be scalar (isotropic) or a vector of length `len(active_dims)` (per-dimension). Scale with `eta**2 * Matern32(...)`.

### Covariance function

$$
k(x, y) = \left( 1 + \sqrt{3}\, r \right) \exp\!\left( -\sqrt{3}\, r \right),
\qquad r = \frac{\lVert x - y \rVert}{\ell}
$$

where $\ell$ is the lengthscale (`ls`).

::::::{tab-set}
:::::{tab-item} Kernel decay
:sync: ls

```{jupyter-execute}
:hide-code:
import ptgp as pg
from ptgp.plotting import plot_kernel_decay
plot_kernel_decay(
    [
        (r"$\ell=0.3$", pg.kernels.Matern32(input_dim=1, ls=0.3)),
        (r"$\ell=1.5$", pg.kernels.Matern32(input_dim=1, ls=1.5)),
        (r"$\ell=5.0$", pg.kernels.Matern32(input_dim=1, ls=5.0)),
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
        (r"$\ell=0.3$", pg.kernels.Matern32(input_dim=1, ls=0.3)),
        (r"$\ell=1.5$", pg.kernels.Matern32(input_dim=1, ls=1.5)),
        (r"$\ell=5.0$", pg.kernels.Matern32(input_dim=1, ls=5.0)),
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
        (r"$\ell=0.3$", pg.kernels.Matern32(input_dim=1, ls=0.3)),
        (r"$\ell=1.5$", pg.kernels.Matern32(input_dim=1, ls=1.5)),
        (r"$\ell=5.0$", pg.kernels.Matern32(input_dim=1, ls=5.0)),
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

kernel = pg.kernels.Matern32(input_dim=1, ls=1.0)

X_sym = pt.matrix("X", shape=(None, 1))
K_fn = pytensor.function([X_sym], kernel(X_sym))

X = np.linspace(-3, 3, 200)[:, None]
K = K_fn(X)
print(K.shape, K.diagonal()[:3])
```
:::::
::::::

```{seealso}
- [Matern52](matern52.md) — same family at $\nu = 5/2$. One additional order of mean-square differentiability; the standard "smooth-but-not-analytic" choice.
- [Matern12](matern12.md) — the exponential kernel at $\nu = 1/2$. Continuous but nowhere differentiable.
- [ExpQuad](expquad.md) — the limit $\nu \to \infty$. Sample functions are infinitely differentiable.
```

The Matérn family and its smoothness-tuning role in GP modeling is discussed in {cite:t}`rasmussen-williams-2006` Chapter 4.

```{bibliography}
:filter: docname in docnames
```
