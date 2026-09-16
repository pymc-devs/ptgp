"""Every model graph and training path keeps ``floatX`` when it is float32."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

import ptgp as pg

M = 6
Z0 = np.linspace(0, 5, M)[:, None]


@pytest.fixture(autouse=True)
def float32():
    """``ptgp`` was imported at float64 above, so anything fixed at import time shows up here."""
    with pytensor.config.change_flags(floatX="float32"):
        yield


def _data():
    rng = np.random.default_rng(0)
    X = np.sort(rng.uniform(0, 5, 30))[:, None]
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 30)
    return X, y


def _shared_dtypes(shared_params, shared_extras):
    return {str(sv.get_value().dtype) for sv in [*shared_params.values(), *shared_extras]}


def _model(X, likelihood=None, Z_var=None, Z_init=None):
    """SVGP, VFE, and Unapproximated sharing one prior container.

    Inducing points come from ``kmeans_init`` unless ``Z_var`` names a placeholder, which is
    frozen by the caller or made trainable through ``Z_init``.
    """
    vp = pg.gp.init_variational_params(M)
    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        kernel = pg.kernels.Matern52(input_dim=1, ls=ls)
        if Z_var is None:
            ind, _ = pg.inducing.kmeans_init(X, M)
        else:
            ind = pg.inducing.Points(Z_var, Z_init=Z_init)
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.Gaussian(sigma=sigma) if likelihood is None else likelihood,
            inducing_variable=ind,
            variational_params=vp,
        )
        vfe = pg.gp.VFE(kernel=kernel, sigma=sigma, inducing_variable=ind)
        full = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)
    return model, svgp, vfe, full, vp


@pytest.mark.parametrize(
    "likelihood",
    [None, pg.likelihoods.Bernoulli(), pg.likelihoods.Poisson()],
    ids=["gaussian", "bernoulli", "poisson"],
)
def test_svgp_elbo_stays_float32(likelihood):
    X_np, _ = _data()
    X = pt.matrix("X", shape=(None, 1))
    y = pt.vector("y")
    _, svgp, _, _, _ = _model(X_np, likelihood=likelihood)

    assert svgp.inducing_variable.Z.dtype == "float32"
    assert pg.objectives.elbo(svgp, X, y).elbo.dtype == "float32"
    assert {term.dtype for term in svgp.predict_marginal(X)} == {"float32"}


def test_gaussian_objectives_stay_float32():
    X_np, _ = _data()
    X = pt.matrix("X", shape=(None, 1))
    y = pt.vector("y")
    _, _, vfe, full, _ = _model(X_np)

    assert pg.objectives.collapsed_elbo(vfe, X, y).elbo.dtype == "float32"
    assert pg.objectives.fitc_log_marginal_likelihood(vfe, X, y).fitc.dtype == "float32"
    assert pg.objectives.marginal_log_likelihood(full, X, y).mll.dtype == "float32"
    assert {term.dtype for term in pg.objectives.vfe_diagnostics(vfe, X, y)} == {"float32"}


def test_training_step_with_frozen_z_takes_float64_data():
    X_np, y_np = _data()
    Z_var = pt.matrix("Z", shape=(M, 1))
    model, svgp, _, _, vp = _model(X_np, Z_var=Z_var)

    train_step, shared_params, shared_extras = pg.optim.compile_training_step(
        lambda gp, X, y: pg.objectives.elbo(gp, X, y).elbo,
        svgp,
        pt.matrix("X", shape=(None, 1)),
        pt.vector("y"),
        model=model,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
        frozen_vars={Z_var: Z0},
    )
    loss = train_step(X_np, y_np)

    assert np.isfinite(loss)
    assert str(loss.dtype) == "float32"
    assert _shared_dtypes(shared_params, shared_extras) == {"float32"}


def test_scipy_objective_with_trainable_z_round_trips_float64_theta():
    X_np, y_np = _data()
    Z_var = pt.matrix("Z", shape=(M, 1))
    model, svgp, _, _, vp = _model(X_np, Z_var=Z_var, Z_init=Z0)
    X_var = pt.matrix("X", shape=(None, 1))
    y_var = pt.vector("y")
    ind = svgp.inducing_variable

    fun, theta0, unpack_to_shared, shared_params, shared_extras = pg.optim.compile_scipy_objective(
        lambda gp, X, y: pg.objectives.elbo(gp, X, y).elbo,
        svgp,
        X_var,
        y_var,
        model=model,
        extra_vars=[*vp.extra_vars, *ind.extra_vars],
        extra_init=[*vp.extra_init, *ind.extra_init],
    )
    theta = theta0.astype(np.float64) + 0.1
    loss, grad = fun(theta, X_np, y_np)
    unpack_to_shared(theta)
    unpacked = np.concatenate(
        [sv.get_value().ravel() for sv in [*shared_params.values(), *shared_extras]]
    )

    assert np.isfinite(loss)
    assert grad.shape == theta0.shape
    assert _shared_dtypes(shared_params, shared_extras) == {"float32"}
    np.testing.assert_allclose(unpacked, theta, rtol=1e-6)

    predict = pg.optim.compile_predict(
        svgp,
        X_var,
        model=model,
        shared_params=shared_params,
        extra_vars=[*vp.extra_vars, *ind.extra_vars],
        shared_extras=shared_extras,
    )
    assert {str(out.dtype) for out in predict(X_np)} == {"float32"}


def test_scipy_diagnostics_with_frozen_sigma_compile_and_run():
    """A frozen value enters the graph at the variable's dtype, so ``graph_replace`` accepts it."""
    X_np, y_np = _data()
    model, _, vfe, _, _ = _model(X_np)
    X_var = pt.matrix("X", shape=(None, 1))
    y_var = pt.vector("y")
    frozen = {model.rvs_to_values[model["sigma"]]: np.log(0.1)}

    _, theta0, _, _, _ = pg.optim.compile_scipy_objective(
        lambda gp, X, y: pg.objectives.collapsed_elbo(gp, X, y).elbo,
        vfe,
        X_var,
        y_var,
        model=model,
        frozen_vars=frozen,
    )
    diag_fn = pg.optim.compile_scipy_diagnostics(
        pg.objectives.vfe_diagnostics,
        vfe,
        X_var,
        y_var,
        model=model,
        frozen_vars=frozen,
    )
    terms = diag_fn(theta0.astype(np.float64), X_np, y_np)

    assert np.isfinite(terms.elbo)
    assert terms.sigma == pytest.approx(0.1, rel=1e-5)
