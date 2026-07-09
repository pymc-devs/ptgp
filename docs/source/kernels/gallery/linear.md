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

# Linear Kernel

{ref}`Non-stationary <non-stationary>`, {ref}`Isotropic <isotropic>`

The linear (dot-product) kernel produces sample functions that are straight lines through the center point $c$. It is the GP equivalent of Bayesian linear regression: the posterior mean is a linear function of the inputs and the posterior variance reflects uncertainty in the slope.

Because the covariance grows without bound as inputs move away from the center, a GP with a pure linear kernel has widening uncertainty far from the data.  A GP with a linear kernel is equivalent to a linear regression $y = \beta_0 + \beta_1 x$.

## Key properties and parameters

```{eval-rst}
=================  ===================================================
Domain             :math:`x \in \mathbb{R}^D`
Stationary         **No** — covariance depends on absolute position
Sample smoothness  Linear (straight lines in 1-D)
Variance           :math:`k(x, x) = \lVert x - c \rVert^2`
=================  ===================================================
```

Constructor: [`ptgp.kernels.Linear(input_dim, c=0.0, active_dims=None)`](../../api/generated/ptgp.kernels.Linear.rst). The center `c` can be a scalar (same center for every active dimension) or a vector of length `len(active_dims)`. Scale the kernel with multiplication: `eta**2 * Linear(...)`.

### Covariance function

$$
k(x, y) = (x - c)^\top (y - c)
$$

where $c$ is the center. When $c = 0$ this reduces to the standard dot product $x^\top y$. Shifting $c$ moves the pivot point around which the linear functions rotate.

::::::{tab-set}
:::::{tab-item} Prior samples
:sync: c

```{jupyter-execute}
:hide-code:
import ptgp as pg
from ptgp.plotting import plot_prior_samples
plot_prior_samples(
    [
        (r"$c=-1$", pg.kernels.Linear(input_dim=1, c=-1.0)),
        (r"$c=0$", pg.kernels.Linear(input_dim=1, c=0.0)),
        (r"$c=1$", pg.kernels.Linear(input_dim=1, c=1.0)),
    ],
    n_samples=4,
)
```
:::::

:::::{tab-item} Posterior given 3 observations
:sync: c

```{jupyter-execute}
:hide-code:
from ptgp.plotting import plot_conditional
plot_conditional(
    [
        (r"$c=-1$", pg.kernels.Linear(input_dim=1, c=-1.0)),
        (r"$c=0$", pg.kernels.Linear(input_dim=1, c=0.0)),
        (r"$c=1$", pg.kernels.Linear(input_dim=1, c=1.0)),
    ],
    n_draws=3,
    y_range=(-2.0, 2.0),
)
```
:::::

:::::{tab-item} Code

```{jupyter-execute}
import numpy as np
import pytensor
import pytensor.tensor as pt
import ptgp as pg

kernel = pg.kernels.Linear(input_dim=1, c=0.0)

X_sym = pt.matrix("X", shape=(None, 1))
K_fn = pytensor.function([X_sym], kernel(X_sym))

X = np.linspace(-3, 3, 200)[:, None]
K = K_fn(X)
print(K.shape, K.diagonal()[:3])
```
:::::
::::::

The three centers show the kernel's effect on prior beliefs:

- **$c = -1$** — lines pivot around $x = -1$; uncertainty is smallest near $-1$ and grows in both directions.
- **$c = 0$** — the standard dot-product kernel; lines pass through the origin.
- **$c = 1$** — lines pivot around $x = 1$.

```{seealso}
- [ExpQuad](expquad.md) — the default stationary kernel. Combine with Linear to model a smooth trend plus local structure.
- [Random Walk](randomwalk.md) — another non-stationary kernel, but with rough (nowhere-differentiable) sample paths rather than straight lines.
```

The linear kernel and its connection to Bayesian linear regression are discussed in {cite:t}`rasmussen-williams-2006` Chapter 2.

```{bibliography}
:filter: docname in docnames
```
