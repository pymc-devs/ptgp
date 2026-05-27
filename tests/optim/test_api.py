"""Tests for the user-facing convenience API (pg.fit / pg.predict)."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

import ptgp as pg


def _toy_data(N=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(0, 5, N))[:, None]
    y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(N)
    return X, y


class TestFitPredictUnapproximated:
    def test_round_trip(self):
        X, y = _toy_data()
        with pm.Model():
            ls = pm.HalfFlat("ls")
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            gp = pg.gp.Unapproximated(
                kernel=eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls),
                sigma=sigma,
            )
            fit = pg.fit(gp, X, y)
        assert fit.result.success
        assert set(fit.params) == {"ls", "eta", "sigma"}

        X_test = np.linspace(0, 5, 12)[:, None]
        mu, var = pg.predict(gp, X_test, fit, X_train=X, y_train=y)
        assert mu.shape == (12,)
        assert np.all(np.isfinite(mu))
        assert np.all(var > 0)

    def test_predict_without_training_data_raises(self):
        X, y = _toy_data(N=20)
        with pm.Model():
            ls = pm.HalfFlat("ls")
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            gp = pg.gp.Unapproximated(kernel=pg.kernels.Matern52(input_dim=1, ls=ls), sigma=sigma)
            fit = pg.fit(gp, X, y, options={"maxiter": 1})

        with pytest.raises(ValueError, match="X_train"):
            pg.predict(gp, X, fit)


class TestFitPredictVFE:
    def test_round_trip_moves_inducing_points(self):
        X, y = _toy_data()
        M = 8
        Z_init = np.linspace(0, 5, M)[:, None]
        Z_var = pt.matrix("Z", shape=(M, 1))
        with pm.Model():
            ls = pm.HalfFlat("ls")
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            vfe = pg.gp.VFE(
                kernel=eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls),
                sigma=sigma,
                inducing_variable=pg.inducing.Points(Z_var, Z_init=Z_init),
            )
            fit = pg.fit(vfe, X, y)
        assert fit.result.success
        Z_final = fit.shared_extras[0].get_value()
        assert np.linalg.norm(Z_final - Z_init) > 0

        X_test = np.linspace(0, 5, 12)[:, None]
        mu, var = pg.predict(vfe, X_test, fit, X_train=X, y_train=y)
        assert mu.shape == (12,)


class TestFitPredictSVGP:
    def test_round_trip(self):
        X, y = _toy_data(N=80)
        M = 8
        Z_init = np.linspace(0, 5, M)[:, None]
        Z_var = pt.matrix("Z", shape=(M, 1))
        vp = pg.gp.init_variational_params(M)
        with pm.Model():
            ls = pm.HalfFlat("ls")
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            svgp = pg.gp.SVGP(
                kernel=eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls),
                likelihood=pg.likelihoods.Gaussian(sigma),
                inducing_variable=pg.inducing.Points(Z_var, Z_init=Z_init),
                variational_params=vp,
            )
            fit = pg.fit(svgp, X, y, options={"maxiter": 100})

        X_test = np.linspace(0, 5, 12)[:, None]
        mu, var = pg.predict(svgp, X_test, fit)
        assert mu.shape == (12,)
        assert np.all(np.isfinite(mu))
        assert np.all(var > 0)
