# PTGP

A Gaussian process library for building GP models that solve real-world problems.

## Who this is for

PTGP is for practitioners who need flexible, well-supported GP modeling. The goal of PTGP is to be fully batteries-included and ready to work on real-world problems:

- **Practical GP algorithms:** exact GP, VFE with collapsed bound, SVGP with minibatch training, VFF (Variational Fourier Features)
- **Full kernel library:** ExpQuad, Matern52/32/12, RandomWalk, Gibbs, WarpedInput, categorical kernels for multi-class or categorical input variables, composition via `+` and `*`, `active_dims` for dimension selection
- **Non-Gaussian likelihoods:** Bernoulli, Poisson, NegativeBinomial, StudentT
- **PyMC priors:** set priors on any hyperparameter; use PyMC distributions for mean functions and noise models; MAP training by default
- **Training tools:** L-BFGS-B and Adam optimizers, per-parameter learning rates, staged optimization, frozen variables, inducing point initialization strategies, diagnostic-guided workflows; more are being added, such as carefully monitored training to help diagnose issues early
- **Skill files:** LLM-readable guides for debugging training issues and folk wisdom; currently have helpers for VFE, more planned for other approximations
- **More coming:** see the [issues](https://github.com/bwengals/ptgp/issues)

Researchers benefit from the underlying design: PTGP is built on PyTensor's symbolic graph and rewrite system, so you write GP math directly (`pt.linalg.inv(K)`, `pt.linalg.slogdet(K)`) and the compiler chooses efficient algorithms based on declared matrix structure. This makes it straightforward to implement new GP approximations and create custom models, and will eventually allow matrix structure like Kronecker, Toeplitz, and sparse to be taken advantage of automatically.

## Models

| Model | Scale | Best for |
|-------|-------|----------|
| `gp.Unapproximated` | N < ~2,000 | Exact inference, model comparison |
| `gp.VFE` | N < ~50,000 | Medium-scale data with inducing points |
| `gp.SVGP` | N up to ~500,000 | Large data, non-Gaussian likelihoods, minibatch training |
| `FourierFeatures1D` | 1D Matern kernels | Structured Kuu via Fourier basis; no inducing point placement |

## Quick start

```python
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import ptgp as pg

X = np.random.randn(200, 1)
y = np.sin(X.ravel()) + 0.1 * np.random.randn(200)
Z = np.linspace(-2, 2, 20)[:, None]

with pm.Model() as model:
    ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
    eta = pm.Exponential("eta", lam=1.0)
    kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)

    vp = pg.gp.init_variational_params(M=20)
    svgp = pg.gp.SVGP(
        kernel=kernel,
        likelihood=pg.likelihoods.Gaussian(sigma=0.1),
        inducing_variable=pg.inducing.Points(pt.as_tensor(Z)),
        variational_params=vp,
    )

X_var = pt.matrix("X", shape=(None, 1))
y_var = pt.vector("y", shape=(None,))

step, shared_params, shared_extras = pg.optim.compile_training_step(
    lambda gp, X, y: pg.objectives.elbo(gp, X, y).elbo,
    svgp, X_var, y_var,
    model=model,
    extra_vars=vp.extra_vars,
    extra_init=vp.extra_init,
    learning_rate=1e-2,
)

for i in range(500):
    loss = step(X, y)

predict_fn = pg.optim.compile_predict(svgp, pt.matrix("X_new"), model, shared_params,
                                       extra_vars=vp.extra_vars, shared_extras=shared_extras)
mean, var = predict_fn(np.linspace(-3, 3, 100)[:, None])
```

Training uses MAP by default: the PyMC log-prior is added to the objective. Pass `include_prior=False` for pure ELBO. For exact GPs and VFE, use `compile_scipy_objective` with L-BFGS-B instead. For now, see [`notebooks/demo.ipynb`](notebooks/demo.ipynb) for examples covering all models. Better documentation and examples are planned.

## How it works

PTGP is built on PyTensor's symbolic graph. Kernels, likelihoods, and GP models return symbolic tensors with naive linear algebra like `pt.linalg.inv(K)` that PyTensor's rewrite system automatically lowers to efficient Cholesky-based code using declared matrix properties. All models compile their full forward+gradient step down to the minimum number of cubic factorizations. Iterative solvers are planned for the future.

PTGP tries to distill some of the approaches of existing GP libraries and make them more accessible, mainly [GPJax](https://github.com/JaxGaussianProcesses/GPJax), [GPflow](https://github.com/GPflow/GPflow), and [GPyTorch](https://github.com/cornellius-gp/gpytorch).

## Install

```bash
pip install -e .
```

Requires PyTensor from main and PyMC >= 6.0:

```bash
pip install git+https://github.com/pymc-devs/pytensor@main
pip install pymc
```

## Status

Prototype. See the [issues](https://github.com/bwengals/ptgp/issues) for what's being worked on. Feel free to propose issues, feature requests, or use cases you've been hoping could be made easier. PRs always welcome.
