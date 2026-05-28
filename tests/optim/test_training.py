"""Tests for ptgp.optim.training — native PyTensor training and prediction."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

import ptgp as pg


@pytest.fixture
def gp_data():
    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 5, 40))[:, None]
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 40)
    return X, y


@pytest.fixture
def svgp_data():
    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 5, 80))[:, None]
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 80)
    return X, y


def test_compile_training_step_gp(gp_data):
    """GP trains and loss decreases."""
    X, y = gp_data

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        sigma = pm.Exponential("sigma", lam=1.0)

        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        gp = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)

    X_var = pt.matrix("X")
    y_var = pt.vector("y")

    train_step, shared_params, shared_extras = pg.optim.compile_training_step(
        lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
        gp,
        X_var,
        y_var,
        model=model,
        learning_rate=1e-2,
    )

    losses = []
    for i in range(100):
        loss = train_step(X, y)
        losses.append(float(loss))

    assert losses[-1] < losses[0], "Loss should decrease during training"


def test_compile_predict_gp(gp_data):
    """GP prediction returns reasonable shapes and values after training."""
    X, y = gp_data

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        sigma = pm.Exponential("sigma", lam=1.0)

        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        gp = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)

    X_var = pt.matrix("X")
    y_var = pt.vector("y")

    train_step, shared_params, shared_extras = pg.optim.compile_training_step(
        lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
        gp,
        X_var,
        y_var,
        model=model,
        learning_rate=1e-2,
    )

    for _ in range(200):
        train_step(X, y)

    X_new_var = pt.matrix("X_new")
    predict_fn = pg.optim.compile_predict(
        gp,
        X_new_var,
        model,
        shared_params,
        X_train=X,
        y_train=y,
    )

    X_test = np.linspace(0, 5, 20)[:, None]
    fmean, fvar = predict_fn(X_test)

    assert fmean.shape == (20,)
    assert fvar.shape == (20,)
    assert np.all(fvar >= 0), "Predictive variance should be non-negative"
    # Predictions should roughly follow sin(x) after training
    assert np.corrcoef(fmean, np.sin(X_test[:, 0]))[0, 1] > 0.9


def test_compile_training_step_svgp(svgp_data):
    """SVGP trains with variational parameters as extra_vars."""
    X, y = svgp_data
    M = 15

    rng = np.random.default_rng(0)
    Z_init = rng.choice(X[:, 0], M, replace=False)
    Z_init = np.sort(Z_init)[:, None]

    vp = pg.gp.init_variational_params(M)

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)

        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.Gaussian(sigma=0.1),
            inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z_init)),
            variational_params=vp,
        )

    X_var = pt.matrix("X")
    y_var = pt.vector("y")

    train_step, shared_params, shared_extras = pg.optim.compile_training_step(
        lambda gp, X, y: pg.objectives.elbo(gp, X, y).elbo,
        svgp,
        X_var,
        y_var,
        model=model,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
        learning_rate=1e-2,
    )

    losses = []
    for i in range(100):
        loss = train_step(X, y)
        losses.append(float(loss))

    assert losses[-1] < losses[0], "SVGP loss should decrease during training"


def test_prior_shifts_optimum(gp_data):
    """Training with include_prior=True converges to a different point
    than include_prior=False when the prior is strong enough to pull
    the optimum away from the MLE.
    """
    X, y = gp_data

    def build():
        with pm.Model() as model:
            # Tight prior on ls, far from whatever the MLE would pick.
            ls = pm.Normal("ls", mu=5.0, sigma=0.01)
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.Exponential("sigma", lam=1.0)
            kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
            gp = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)
        return model, gp

    def train(include_prior):
        model, gp = build()
        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        train_step, shared_params, _ = pg.optim.compile_training_step(
            lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
            gp,
            X_var,
            y_var,
            model=model,
            learning_rate=1e-2,
            include_prior=include_prior,
        )
        for _ in range(500):
            train_step(X, y)
        return pg.optim.get_trained_params(model, shared_params)

    mle = train(include_prior=False)
    map_ = train(include_prior=True)

    # The tight prior should pin ls near 5.0 in constrained space.
    assert abs(map_["ls"] - 5.0) < 0.1
    # The MLE is free to wander far from the prior mean.
    assert abs(mle["ls"] - 5.0) > 0.5


def test_sgd_optimizer(gp_data):
    """SGD optimizer works as alternative to adam."""
    X, y = gp_data

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        sigma = pm.Exponential("sigma", lam=1.0)

        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        gp = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)

    X_var = pt.matrix("X")
    y_var = pt.vector("y")

    train_step, shared_params, shared_extras = pg.optim.compile_training_step(
        lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
        gp,
        X_var,
        y_var,
        model=model,
        optimizer_fn=pg.optim.sgd,
        learning_rate=1e-3,
    )

    losses = []
    for i in range(50):
        loss = train_step(X, y)
        losses.append(float(loss))

    assert losses[-1] < losses[0], "SGD should also reduce loss"


def test_tracked_minimize_interrupts_gracefully():
    """KeyboardInterrupt during scipy optimization returns a synthesized
    OptimizeResult carrying the last theta seen by the callback."""
    n_calls = [0]
    raise_after = 10

    def fun(theta, *args):
        n_calls[0] += 1
        if n_calls[0] >= raise_after:
            raise KeyboardInterrupt
        # Rosenbrock — slow to converge, scipy will keep iterating.
        x = np.asarray(theta)
        f = float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
        g = np.zeros_like(x)
        g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return f, g

    theta0 = np.array([0.0, 0.0, 0.0])

    # diag_fn appends one entry per callback so we can count iterations seen.
    from collections import namedtuple

    Term = namedtuple("Term", ["loss"])

    def diag_fn(theta, *args):
        return Term(loss=float(theta @ theta))

    # Disable convergence so scipy runs until we raise.
    result, history = pg.optim.tracked_minimize(
        fun,
        theta0,
        args=(),
        diag_fn=diag_fn,
        options={"maxiter": 100, "ftol": 0, "gtol": 0},
    )

    assert result.status == 99
    assert result.success is False
    assert "KeyboardInterrupt" in result.message
    # last_theta is updated inside the callback before fun is called next, so
    # the returned x reflects the last completed iteration's iterate.
    assert result.x.shape == theta0.shape
    # history grew, but stopped before raise_after.
    assert len(history) >= 1
    assert len(history) < raise_after


def test_minimize_staged_vfe_interrupts_in_phase1(gp_data):
    """KeyboardInterrupt during phase 1 returns from minimize_staged_vfe with
    only phase1 labels in the history and no later phases."""
    X, y = gp_data
    M = 10
    rng = np.random.default_rng(0)
    Z_init = np.sort(rng.choice(X[:, 0], M, replace=False))[:, None]

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        sigma = pm.Exponential("sigma", lam=1.0)
        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        vfe = pg.gp.VFE(
            kernel=kernel,
            sigma=sigma,
            inducing_variable=pg.inducing.Points(pt.matrix("Z", shape=(M, 1))),
        )

    Z_var = vfe.inducing_variable.Z
    X_var = pt.matrix("X", shape=(None, 1))
    y_var = pt.vector("y", shape=(None,))

    # Patch tracked_minimize to raise KeyboardInterrupt on the 2nd phase-1
    # callback. We do this by patching the diag_fn passed inside the staged
    # routine, but simplest: monkey-patch tracked_minimize directly to inject
    # an interrupt after a few iters.
    import ptgp.optim.training as training_mod

    orig = training_mod.tracked_minimize
    state = {"calls": 0}

    def raising_tracked_minimize(fun, theta0, args, diag_fn=None, print_every=None, **kwargs):
        # On the first call (phase 1), wrap fun to raise after a few invocations.
        state["calls"] += 1
        if state["calls"] == 1:
            inner_count = [0]

            def f_wrapped(theta, *a):
                inner_count[0] += 1
                if inner_count[0] >= 4:
                    raise KeyboardInterrupt
                return fun(theta, *a)

            return orig(
                f_wrapped, theta0, args=args, diag_fn=diag_fn, print_every=print_every, **kwargs
            )
        return orig(fun, theta0, args=args, diag_fn=diag_fn, print_every=print_every, **kwargs)

    training_mod.tracked_minimize = raising_tracked_minimize
    try:
        result, history, phase_labels, unpack, sp, se = pg.optim.minimize_staged_vfe(
            lambda gp_, X_, y_: pg.objectives.collapsed_elbo(gp_, X_, y_).elbo,
            vfe,
            X_var,
            y_var,
            X,
            y,
            model,
            sigma_init=0.1,
            Z_var=Z_var,
            Z_init=Z_init,
            phase1_freeze_Z=False,
            phase1_maxiter=20,
            phase2_cycles=1,
            phase2_maxiter_Z=5,
            phase2_maxiter_hyper=5,
            phase3_maxiter=5,
        )
    finally:
        training_mod.tracked_minimize = orig

    assert result.status == 99
    # Only phase1 labels should appear; phases 2a/2b/3 must not have run.
    assert set(phase_labels) <= {"phase1"}
    assert callable(unpack)
    assert isinstance(sp, dict) and len(sp) > 0
    assert len(se) == 1  # Z's shared var
