"""CompositeLikelihood tests: per-group dispatch, re-rooting, and end-to-end use with VGP."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor_ml.optim import adam

import ptgp as pg

from ptgp.likelihoods import CompositeLikelihood, Gaussian, Poisson, StudentT


def _inputs(rng, N):
    """Column 0 is a feature, column 1 the group label (interleaved, non-contiguous)."""
    return np.c_[rng.uniform(-1, 1, N), np.arange(N) % 2].astype(float)


def _hetero(X):
    return 0.2 + 0.1 * X[:, 0] ** 2


def _eval(outputs, inputs, values):
    fn = pytensor.function(inputs, outputs, on_unused_input="ignore")
    return fn(*values)


def _per_group(X_num, fn0, fn1):
    return np.where(X_num[:, 1] == 0, fn0, fn1)


def test_variational_expectation_equals_per_group():
    """Each row gets its own group's expectation, with input-dependent sigma
    evaluated at that group's rows."""
    rng = np.random.default_rng(0)
    N = 10
    X_num = _inputs(rng, N)
    y, mu, var = rng.standard_normal(N), rng.standard_normal(N), rng.uniform(0.1, 1.0, N)
    X = pt.matrix("X", shape=(None, 2))
    lik = CompositeLikelihood(
        [StudentT(nu=4.0, sigma=0.5), Gaussian(_hetero(X), x=X)], group=X[:, 1], x=X
    )
    yv, muv, varv = pt.vector("y"), pt.vector("mu"), pt.vector("var")
    out = pytensor.function([X, yv, muv, varv], lik.variational_expectation(yv, muv, varv))(
        X_num, y, mu, var
    )

    def ve(lik0):
        return pytensor.function([yv, muv, varv], lik0.variational_expectation(yv, muv, varv))(
            y, mu, var
        )

    expected = _per_group(
        X_num, ve(StudentT(nu=4.0, sigma=0.5)), ve(Gaussian(_hetero(pt.as_tensor(X_num))))
    )
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_at_recomputes_groups_for_new_inputs():
    """lik.at(X_new) re-roots the labels and the components onto the new inputs."""
    rng = np.random.default_rng(1)
    X = pt.matrix("X", shape=(None, 2))
    lik = CompositeLikelihood([Poisson(), Gaussian(_hetero(X), x=X)], group=X[:, 1], x=X)
    X_new = _inputs(rng, 7)[::-1].copy()
    mu, var = rng.standard_normal(7), rng.uniform(0.1, 1.0, 7)
    X_new_t, muv, varv = pt.matrix("X_new", shape=(None, 2)), pt.vector("mu"), pt.vector("var")
    m, v = lik.at(X_new_t).predict_mean_and_var(muv, varv)
    m, v = pytensor.function([X_new_t, muv, varv], [m, v])(X_new, mu, var)

    pm_, pv = _eval(list(Poisson().predict_mean_and_var(muv, varv)), [muv, varv], [mu, var])
    g = Gaussian(_hetero(pt.as_tensor(X_new)))
    gm, gv = _eval(list(g.predict_mean_and_var(muv, varv)), [muv, varv], [mu, var])
    np.testing.assert_allclose(m, _per_group(X_new, pm_, gm), atol=1e-10)
    np.testing.assert_allclose(v, _per_group(X_new, pv, gv), atol=1e-10)


def test_rejects_static_and_out_of_range_labels():
    X = pt.matrix("X", shape=(None, 2))
    with pytest.raises(ValueError, match="computed from x"):
        CompositeLikelihood([Gaussian(0.3), Poisson()], group=np.array([0, 1]), x=X)

    lik = CompositeLikelihood([Gaussian(0.3), Poisson()], group=X[:, 1], x=X)
    muv = pt.vector("mu")
    fn = pytensor.function([X, muv], lik.predict_mean_and_var(muv, muv)[0])
    with pytest.raises(AssertionError, match="group labels"):
        fn(np.array([[0.0, 0.0], [0.0, 2.0]]), np.zeros(2))


def test_vgp_with_composite_likelihood_trains_and_predicts():
    """VGP with Gaussian and Student-t groups trains, and predicts at new inputs."""
    rng = np.random.default_rng(3)
    N = 40
    X = _inputs(rng, N)
    y = np.sin(3 * X[:, 0]) + 0.15 * rng.standard_normal(N)
    y[X[:, 1] == 1][:3] += 4.0  # outliers in the Student-t group

    X_var, y_var = pt.matrix("X", shape=(None, 2)), pt.vector("y")
    lik = CompositeLikelihood(
        [Gaussian(0.2), StudentT(nu=3.0, sigma=0.2)], group=X_var[:, 1], x=X_var
    )
    vp = pg.gp.init_vgp_params(N)
    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        kernel = eta**2 * pg.kernels.Matern52(input_dim=2, ls=ls, active_dims=[0])
        vgp = pg.gp.VGP(kernel=kernel, likelihood=lik, variational_params=vp)

    train_step, shared_params, shared_extras = pg.optim.compile_training_step(
        pg.objectives.vgp_elbo, vgp, X_var, y_var, model=model, optimizer=adam(1e-2)
    )
    losses = [float(train_step(X, y)) for _ in range(300)]
    assert losses[-1] < losses[0], "VGP + CompositeLikelihood loss should decrease"

    X_new = _inputs(rng, 6)
    predict = pg.optim.compile_predict(
        vgp,
        pt.matrix("X_new", shape=(None, 2)),
        model,
        shared_params,
        shared_extras=shared_extras,
        incl_lik=True,
        X_train=X,
    )
    mean, var = predict(X_new)
    assert np.all(np.isfinite(mean)) and np.all(var > 0)
