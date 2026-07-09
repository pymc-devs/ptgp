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

# Exponentiated Quadratic Kernel

{ref}`Stationary <stationary>`, {ref}`Isotropic <isotropic>`, {ref}`Smooth <smooth>`, {ref}`Universal <universal>`

The exponentiated quadratic kernel — also known as the **squared exponential**, **RBF**, or **Gaussian** kernel — measures covariance with the squared Euclidean distance between inputs, decaying as a Gaussian in distance. It is the default first choice in most GP applications.

Sample functions drawn from a GP with an ExpQuad covariance are infinitely differentiable, the smoothest functions a GP can produce. This makes the kernel appropriate when the function being modeled is genuinely smooth, but it also encodes a strong prior assumption: when the truth is rougher than ExpQuad expects, posterior credible intervals between observations are overconfident. The [Matérn 5/2](matern52.md) kernel is a common less aggressive substitute that preserves twice-differentiability without ExpQuad's strong smoothness assumption.

## Key properties and parameters

```{eval-rst}
=================  =================================================
Domain             :math:`x \in \mathbb{R}^D`
Stationary         Yes
Sample smoothness  :math:`C^\infty` — infinitely differentiable
Variance           :math:`k(x, x) = 1`
=================  =================================================
```

Constructor: [`ptgp.kernels.ExpQuad(input_dim, ls, active_dims=None)`](../../api/generated/ptgp.kernels.ExpQuad.rst). The lengthscale `ls` can be either a scalar (isotropic, one length-scale shared across input dimensions) or a vector of length `len(active_dims)` (one lengthscale per dimension — the standard GP recipe for letting the model down-weight dimensions that don't carry signal). Scale the kernel with multiplication: `eta**2 * ExpQuad(...)`; per ptgp convention `eta` is always squared.

### Covariance function

$$
k(x, y) = \exp\left( -\frac{\lVert x - y \rVert^2}{2\, \ell^2} \right)
$$

where $\ell$ is the lengthscale (`ls`). The kernel is `1` when $x = y$ and decays to `0` as $\lVert x - y \rVert / \ell$ grows. By a distance of $3\ell$, the covariance is already below `0.012`.

::::::{tab-set}
:::::{tab-item} Kernel decay
:sync: ls

```{jupyter-execute}
:hide-code:
import ptgp as pg
from ptgp.plotting import plot_kernel_decay
plot_kernel_decay(
    [
        (r"$\ell=0.3$", pg.kernels.ExpQuad(input_dim=1, ls=0.3)),
        (r"$\ell=1.5$", pg.kernels.ExpQuad(input_dim=1, ls=1.5)),
        (r"$\ell=5.0$", pg.kernels.ExpQuad(input_dim=1, ls=5.0)),
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
        (r"$\ell=0.3$", pg.kernels.ExpQuad(input_dim=1, ls=0.3)),
        (r"$\ell=1.5$", pg.kernels.ExpQuad(input_dim=1, ls=1.5)),
        (r"$\ell=5.0$", pg.kernels.ExpQuad(input_dim=1, ls=5.0)),
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
        (r"$\ell=0.3$", pg.kernels.ExpQuad(input_dim=1, ls=0.3)),
        (r"$\ell=1.5$", pg.kernels.ExpQuad(input_dim=1, ls=1.5)),
        (r"$\ell=5.0$", pg.kernels.ExpQuad(input_dim=1, ls=5.0)),
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

kernel = pg.kernels.ExpQuad(input_dim=1, ls=1.0)

X_sym = pt.matrix("X", shape=(None, 1))
K_fn = pytensor.function([X_sym], kernel(X_sym))

X = np.linspace(-3, 3, 200)[:, None]
K = K_fn(X)
print(K.shape, K.diagonal()[:3])
```
:::::
::::::

The three lengthscales show the kernel's effect on prior beliefs:

- **$\ell=0.3$** — short lengthscale, wiggly draws, posterior reverts to the prior quickly outside data.
- **$\ell=1.5$** — moderate lengthscale, smooth interpolation between observations.
- **$\ell=5.0$** — long lengthscale, very smooth, draws nearly straight between observations and extrapolate confidently.

```{seealso}
- [Matern52](matern52.md) — same family at smoothness $\nu = 5/2$. Sample functions are twice mean-square differentiable. Recommended default when ExpQuad's $C^\infty$ assumption is too strong.
- [Matern32](matern32.md) — once mean-square differentiable; visibly rougher samples.
- [Matern12](matern12.md) — the exponential kernel ($\nu = 1/2$). Sample functions are continuous but nowhere differentiable.
```

The exponentiated quadratic is the canonical GP covariance and is discussed across all standard treatments; {cite:t}`rasmussen-williams-2006` Chapter 4 is the standard reference.

```{bibliography}
:filter: docname in docnames
```
