"""Tests for compile_scipy_objective."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.optimize

import ptgp as pg


class TestScipyObjectiveGP:
    def test_loss_and_grad_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 1))
        y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(20)

        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        with pm.Model() as model:
            ls = pm.HalfFlat("ls")
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            gp = pg.gp.Unapproximated(
                kernel=eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls),
                sigma=sigma,
            )

        fun, theta0, _, _, _ = pg.optim.compile_scipy_objective(
            lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
            gp,
            X_var,
            y_var,
            model=model,
        )
        assert theta0.shape == (3,)
        loss, grad = fun(theta0, X, y)
        assert np.isfinite(loss)
        assert grad.shape == (3,)

    def test_lbfgs_decreases_loss(self):
        rng = np.random.default_rng(0)
        X = np.sort(rng.uniform(0, 5, 40))[:, None]
        y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 40)

        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        with pm.Model() as model:
            ls = pm.HalfFlat("ls")
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            gp = pg.gp.Unapproximated(
                kernel=eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls),
                sigma=sigma,
            )

        fun, theta0, unpack_to_shared, shared, _ = pg.optim.compile_scipy_objective(
            lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
            gp,
            X_var,
            y_var,
            model=model,
        )
        loss0, _ = fun(theta0, X, y)
        result = scipy.optimize.minimize(
            fun,
            theta0,
            args=(X, y),
            jac=True,
            method="L-BFGS-B",
        )
        assert result.success
        assert result.fun < loss0

        unpack_to_shared(result.x)
        params = pg.optim.get_trained_params(model, shared)
        assert params["sigma"] < 0.3  # recovered noise close to 0.1

    def test_predict_after_unpack(self):
        rng = np.random.default_rng(0)
        X = np.sort(rng.uniform(0, 5, 30))[:, None]
        y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 30)

        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        X_new_var = pt.matrix("X_new")
        with pm.Model() as model:
            ls = pm.HalfFlat("ls")
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            gp = pg.gp.Unapproximated(
                kernel=eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls),
                sigma=sigma,
            )

        fun, theta0, unpack_to_shared, shared, _ = pg.optim.compile_scipy_objective(
            lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
            gp,
            X_var,
            y_var,
            model=model,
        )
        result = scipy.optimize.minimize(
            fun,
            theta0,
            args=(X, y),
            jac=True,
            method="L-BFGS-B",
        )
        unpack_to_shared(result.x)

        predict = pg.optim.compile_predict(
            gp,
            X_new_var,
            model,
            shared,
            X_train=X,
            y_train=y,
        )
        X_new = np.linspace(0, 5, 10)[:, None]
        mu, var = predict(X_new)
        assert mu.shape == (10,)
        assert np.all(np.isfinite(mu))
        assert np.all(var > 0)


class TestScipyObjectiveVFE:
    def test_extra_vars_in_theta(self):
        """Z_var shows up in theta0 and gets optimized."""
        rng = np.random.default_rng(0)
        X = np.sort(rng.uniform(0, 5, 40))[:, None]
        y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 40)
        M = 6
        Z_init = np.linspace(0, 5, M)[:, None]

        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        Z_var = pt.matrix("Z")
        with pm.Model() as model:
            ls = pm.HalfFlat("ls")
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            vfe = pg.gp.VFE(
                kernel=eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls),
                sigma=sigma,
                inducing_variable=pg.inducing.Points(Z_var),
            )

        fun, theta0, unpack_to_shared, _, extras = pg.optim.compile_scipy_objective(
            lambda gp, X, y: pg.objectives.collapsed_elbo(gp, X, y).elbo,
            vfe,
            X_var,
            y_var,
            model=model,
            extra_vars=[Z_var],
            extra_init=[Z_init],
        )
        assert theta0.shape == (3 + M,)

        result = scipy.optimize.minimize(
            fun,
            theta0,
            args=(X, y),
            jac=True,
            method="L-BFGS-B",
        )
        assert result.success
        unpack_to_shared(result.x)
        Z_final = extras[0].get_value()
        assert np.linalg.norm(Z_final - Z_init) > 0


class TestScipyObjectiveFrozenVars:
    def test_frozen_var_not_in_theta(self):
        """frozen_vars are excluded from theta0 and their gradient."""
        rng = np.random.default_rng(0)
        X = np.sort(rng.uniform(0, 5, 30))[:, None]
        y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 30)
        M = 5
        Z_frozen = np.linspace(0, 5, M)[:, None]

        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        Z_var = pt.matrix("Z")
        with pm.Model() as model:
            ls = pm.HalfFlat("ls")
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            vfe = pg.gp.VFE(
                kernel=eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls),
                sigma=sigma,
                inducing_variable=pg.inducing.Points(Z_var),
            )

        fun, theta0, _, _, _ = pg.optim.compile_scipy_objective(
            lambda gp, X, y: pg.objectives.collapsed_elbo(gp, X, y).elbo,
            vfe,
            X_var,
            y_var,
            model=model,
            frozen_vars={Z_var: Z_frozen},
        )
        # Only the 3 hyperparameters in theta; no Z.
        assert theta0.shape == (3,)
        loss, grad = fun(theta0, X, y)
        assert grad.shape == (3,)


class TestNamedtupleObjective:
    def test_namedtuple_objective_matches_scalar(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 1))
        y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(20)

        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        with pm.Model():
            ls = pm.HalfFlat("ls")
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            gp = pg.gp.Unapproximated(
                kernel=eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls),
                sigma=sigma,
            )

            fun_nt, theta0, *_ = pg.optim.compile_scipy_objective(
                pg.objectives.marginal_log_likelihood, gp, X_var, y_var
            )
            fun_sc, _, *_ = pg.optim.compile_scipy_objective(
                lambda g, X, y: pg.objectives.marginal_log_likelihood(g, X, y).mll,
                gp,
                X_var,
                y_var,
            )
        loss_nt, grad_nt = fun_nt(theta0, X, y)
        loss_sc, grad_sc = fun_sc(theta0, X, y)
        np.testing.assert_allclose(loss_nt, loss_sc)
        np.testing.assert_allclose(grad_nt, grad_sc)
