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

# Matérn 5/2 Kernel

{ref}`Stationary <stationary>`, {ref}`Isotropic <isotropic>`, {ref}`Smooth <smooth>`

The Matérn 5/2 kernel is the smoothest member of the Matérn family that is *not* infinitely differentiable. Sample functions are twice mean-square differentiable — the function and its first derivative are well-defined and continuous, but the curvature can have jumps. This is enough smoothness for most real-world processes and avoids the overconfident extrapolation that the [exponentiated quadratic](expquad.md) tends to produce when the true function is not analytic.

In practice Matérn 5/2 is the recommended default kernel when you don't have a specific reason to assume infinite differentiability. It is the standard choice across Bayesian optimization, geostatistics, and surrogate modeling literatures.

## Key properties and parameters

```{eval-rst}
=================  =================================================
Domain             :math:`x \in \mathbb{R}^D`
Stationary         Yes
Sample smoothness  :math:`C^2` — twice mean-square differentiable
Variance           :math:`k(x, x) = 1`
=================  =================================================
```

Constructor: [`ptgp.kernels.Matern52(input_dim, ls, active_dims=None)`](../../api/generated/ptgp.kernels.Matern52.rst). The lengthscale `ls` can be scalar (isotropic) or a vector of length `len(active_dims)` (per-dimension). Scale with `eta**2 * Matern52(...)`.

### Covariance function

$$
k(x, y) = \left( 1 + \sqrt{5}\, r + \tfrac{5}{3} r^2 \right) \exp\!\left( -\sqrt{5}\, r \right),
\qquad r = \frac{\lVert x - y \rVert}{\ell}
$$

where $\ell$ is the lengthscale (`ls`). The covariance decays slower than the exponentiated quadratic at moderate distances and faster than Matérn 3/2.

::::::{tab-set}
:::::{tab-item} Kernel decay
:sync: ls

```{jupyter-execute}
:hide-code:
import ptgp as pg
from ptgp.plotting import plot_kernel_decay
plot_kernel_decay(
    [
        (r"$\ell=0.3$", pg.kernels.Matern52(input_dim=1, ls=0.3)),
        (r"$\ell=1.5$", pg.kernels.Matern52(input_dim=1, ls=1.5)),
        (r"$\ell=5.0$", pg.kernels.Matern52(input_dim=1, ls=5.0)),
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
        (r"$\ell=0.3$", pg.kernels.Matern52(input_dim=1, ls=0.3)),
        (r"$\ell=1.5$", pg.kernels.Matern52(input_dim=1, ls=1.5)),
        (r"$\ell=5.0$", pg.kernels.Matern52(input_dim=1, ls=5.0)),
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
        (r"$\ell=0.3$", pg.kernels.Matern52(input_dim=1, ls=0.3)),
        (r"$\ell=1.5$", pg.kernels.Matern52(input_dim=1, ls=1.5)),
        (r"$\ell=5.0$", pg.kernels.Matern52(input_dim=1, ls=5.0)),
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

kernel = pg.kernels.Matern52(input_dim=1, ls=1.0)

X_sym = pt.matrix("X", shape=(None, 1))
K_fn = pytensor.function([X_sym], kernel(X_sym))

X = np.linspace(-3, 3, 200)[:, None]
K = K_fn(X)
print(K.shape, K.diagonal()[:3])
```
:::::
::::::

```{seealso}
- [ExpQuad](expquad.md) — the limit $\nu \to \infty$ of the Matérn family. Sample functions are infinitely differentiable; encodes a stronger smoothness assumption.
- [Matern32](matern32.md) — same family at $\nu = 3/2$. Sample functions are only once mean-square differentiable; visibly rougher draws.
- [Matern12](matern12.md) — the exponential kernel at $\nu = 1/2$. Continuous but nowhere differentiable.
```

The Matérn family and its smoothness-tuning role in GP modeling is discussed in {cite:t}`rasmussen-williams-2006` Chapter 4.

```{bibliography}
:filter: docname in docnames
```
