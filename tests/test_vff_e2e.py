"""End-to-end VFF SVGP smoke test: build, train, predict."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ptgp import FourierFeatures1D
from ptgp.gp.svgp import SVGP, init_variational_params
from ptgp.kernels.stationary import Matern32
from ptgp.likelihoods.gaussian import Gaussian
from ptgp.objectives import elbo
from ptgp.optim.training import compile_predict, compile_training_step


def test_vff_svgp_trains_and_predicts():
    rng = np.random.default_rng(0)
    N = 200
    X = np.sort(rng.uniform(0, 1, N))[:, None]
    y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * rng.standard_normal(N)

    f = FourierFeatures1D.from_data(X, num_frequencies=10)
    M = f.num_inducing
    vp = init_variational_params(M)

    with pm.Model() as model:
        ls = pm.Exponential("ls", lam=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        kernel = eta**2 * Matern32(input_dim=1, ls=ls)
        svgp = SVGP(
            kernel=kernel,
            likelihood=Gaussian(sigma=0.1),
            inducing_variable=f,
            variational_params=vp,
            whiten=True,
        )

    X_var = pt.matrix("X")
    y_var = pt.vector("y")
    step, shared_params, shared_extras = compile_training_step(
        lambda gp, X, y: elbo(gp, X, y).elbo,
        svgp,
        X_var,
        y_var,
        model=model,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
        learning_rate=1e-2,
    )
    X_new_var = pt.matrix("X_new")
    pred = compile_predict(
        svgp,
        X_new_var,
        model,
        shared_params,
        extra_vars=vp.extra_vars,
        shared_extras=shared_extras,
    )

    loss0 = float(step(X, y))
    for _ in range(40):
        step(X, y)
    loss1 = float(step(X, y))
    assert loss1 < loss0  # negative-ELBO drops

    m, v = pred(X)
    assert m.shape == (N,) and v.shape == (N,)
    assert np.all(np.isfinite(m)) and np.all(v >= 0)
