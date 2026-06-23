"""Tests for the objectives in ptgp.objectives."""

import numpy as np
import pytensor
import pytensor.assumptions as pta
import pytensor.tensor as pt
import pytest

from ptgp.gp import SVGP, VFE, Unapproximated, VariationalParams
from ptgp.inducing import Points
from ptgp.kernels import ExpQuad
from ptgp.likelihoods import Gaussian
from ptgp.mean import Constant, Linear, Zero
from ptgp.objectives import (
    collapsed_elbo,
    elbo,
    marginal_log_likelihood,
    variance_budget,
    vfe_diagnostics,
)


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(0, 5, 20))[:, None].astype(np.float64)
    y = np.sin(X.ravel()) + 0.1 * rng.standard_normal(20)
    return X, y


@pytest.fixture
def inducing_points():
    return np.linspace(0.5, 4.5, 5)[:, None].astype(np.float64)


class TestMarginalLogLikelihood:
    def test_finite(self, regression_data):
        X, y = regression_data
        gp = Unapproximated(kernel=ExpQuad(input_dim=1, ls=1.0), mean=Zero(), sigma=0.1)
        mll = _eval(
            marginal_log_likelihood(gp, pt.as_tensor_variable(X), pt.as_tensor_variable(y)).mll
        )
        assert np.isfinite(mll)

    def test_better_fit_higher_mll(self, regression_data):
        """A kernel with reasonable params should have higher MLL than a bad one."""
        X, y = regression_data
        gp_good = Unapproximated(kernel=ExpQuad(input_dim=1, ls=1.0), mean=Zero(), sigma=0.1)
        gp_bad = Unapproximated(kernel=ExpQuad(input_dim=1, ls=0.01), mean=Zero(), sigma=10.0)

        mll_good = _eval(
            marginal_log_likelihood(gp_good, pt.as_tensor_variable(X), pt.as_tensor_variable(y)).mll
        )
        mll_bad = _eval(
            marginal_log_likelihood(gp_bad, pt.as_tensor_variable(X), pt.as_tensor_variable(y)).mll
        )

        assert mll_good > mll_bad


class TestELBO:
    def _identity_vp(self, M):
        return VariationalParams(
            q_mu=pt.zeros(M),
            q_sqrt=pta.assume(pt.eye(M), lower_triangular=True),
        )

    def test_finite(self, regression_data, inducing_points):
        X, y = regression_data
        svgp = SVGP(
            kernel=ExpQuad(input_dim=1, ls=1.0),
            mean=Zero(),
            likelihood=Gaussian(sigma=0.1),
            inducing_variable=Points(pt.as_tensor_variable(inducing_points)),
            variational_params=self._identity_vp(5),
        )
        elbo_val = _eval(elbo(svgp, pt.as_tensor_variable(X), pt.as_tensor_variable(y)).elbo)
        assert np.isfinite(elbo_val)

    def test_unwhitened_finite(self, regression_data, inducing_points):
        X, y = regression_data
        svgp = SVGP(
            kernel=ExpQuad(input_dim=1, ls=1.0),
            mean=Zero(),
            likelihood=Gaussian(sigma=0.1),
            inducing_variable=Points(pt.as_tensor_variable(inducing_points)),
            variational_params=self._identity_vp(5),
            whiten=False,
        )
        elbo_val = _eval(elbo(svgp, pt.as_tensor_variable(X), pt.as_tensor_variable(y)).elbo)
        assert np.isfinite(elbo_val)

    def test_whitened_and_unwhitened_agree_at_prior(self, regression_data, inducing_points):
        """With q=prior (q_mu=0, q_sqrt=I for whitened; q_mu=0, q_sqrt=Luu for unwhitened),
        both parameterizations should give the same ELBO."""
        X, y = regression_data
        Z = pt.as_tensor_variable(inducing_points)
        kernel = ExpQuad(input_dim=1, ls=1.0)

        # Whitened: q_mu=0, q_sqrt=I is the prior q(v)=N(0,I)
        svgp_w = SVGP(
            kernel=kernel,
            mean=Zero(),
            likelihood=Gaussian(sigma=0.1),
            inducing_variable=Points(Z),
            variational_params=self._identity_vp(5),
            whiten=True,
        )
        elbo_w = _eval(elbo(svgp_w, pt.as_tensor_variable(X), pt.as_tensor_variable(y)))

        # Unwhitened: q_mu=0, q_sqrt=Luu is the prior q(u)=N(0, Kuu+jit·I)
        # Jitter Kuu before Cholesky to match InducingVariables._jittered_Kuu,
        # so the unwhitened path's q_sqrt corresponds to the same prior as the
        # whitened path's identity q_sqrt.
        Kuu = _eval(kernel(Z))
        Luu = np.linalg.cholesky(Kuu + 1e-6 * np.eye(5))
        vp_u = VariationalParams(
            q_mu=pt.zeros(5),
            q_sqrt=pta.assume(pt.as_tensor_variable(Luu), lower_triangular=True),
        )
        svgp_u = SVGP(
            kernel=kernel,
            mean=Zero(),
            likelihood=Gaussian(sigma=0.1),
            inducing_variable=Points(Z),
            variational_params=vp_u,
            whiten=False,
        )
        elbo_u = _eval(elbo(svgp_u, pt.as_tensor_variable(X), pt.as_tensor_variable(y)))

        np.testing.assert_allclose(elbo_w, elbo_u, atol=1e-6)

    def test_elbo_less_than_mll(self, regression_data, inducing_points):
        """ELBO should be a lower bound on the marginal log likelihood."""
        X, y = regression_data
        ls, sigma = 1.0, 0.1
        kernel = ExpQuad(input_dim=1, ls=ls)

        gp = Unapproximated(kernel=kernel, mean=Zero(), sigma=sigma)
        mll_val = _eval(
            marginal_log_likelihood(gp, pt.as_tensor_variable(X), pt.as_tensor_variable(y)).mll
        )

        svgp = SVGP(
            kernel=kernel,
            mean=Zero(),
            likelihood=Gaussian(sigma=sigma),
            inducing_variable=Points(pt.as_tensor_variable(inducing_points)),
            variational_params=self._identity_vp(5),
        )
        elbo_val = _eval(elbo(svgp, pt.as_tensor_variable(X), pt.as_tensor_variable(y)).elbo)

        assert elbo_val <= mll_val + 1e-6  # ELBO <= MLL


class TestCollapsedELBO:
    def test_finite(self, regression_data, inducing_points):
        X, y = regression_data
        vfe_model = VFE(
            kernel=ExpQuad(input_dim=1, ls=1.0),
            mean=Zero(),
            sigma=0.1,
            inducing_variable=Points(pt.as_tensor_variable(inducing_points)),
        )
        celbo = _eval(
            collapsed_elbo(vfe_model, pt.as_tensor_variable(X), pt.as_tensor_variable(y)).elbo
        )
        assert np.isfinite(celbo)

    def test_collapsed_elbo_less_than_mll(self, regression_data, inducing_points):
        """Collapsed ELBO should be a lower bound on the marginal log likelihood."""
        X, y = regression_data
        ls, sigma = 1.0, 0.1
        kernel = ExpQuad(input_dim=1, ls=ls)

        gp = Unapproximated(kernel=kernel, mean=Zero(), sigma=sigma)
        mll_val = _eval(
            marginal_log_likelihood(gp, pt.as_tensor_variable(X), pt.as_tensor_variable(y)).mll
        )

        vfe_model = VFE(
            kernel=kernel,
            mean=Zero(),
            sigma=sigma,
            inducing_variable=Points(pt.as_tensor_variable(inducing_points)),
        )
        celbo = _eval(
            collapsed_elbo(vfe_model, pt.as_tensor_variable(X), pt.as_tensor_variable(y)).elbo
        )

        assert celbo <= mll_val + 1e-6  # collapsed ELBO <= MLL


class TestVarianceBudget:
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(0)
        X = np.sort(rng.uniform(0, 5, 30))[:, None].astype(np.float64)
        # nonzero offset (2.0) to exercise mean-invariance
        y = np.sin(X.ravel()) + 0.1 * rng.standard_normal(30) + 2.0
        return X, y

    def _gp(self, eta=1.0, sigma=0.3, mean=None):
        return Unapproximated(
            kernel=eta**2 * ExpQuad(input_dim=1, ls=1.0),
            mean=mean if mean is not None else Zero(),
            sigma=sigma,
        )

    def _budget(self, gp, X, y):
        b = variance_budget(gp, pt.as_tensor_variable(X), pt.as_tensor_variable(y))
        return b._make(_eval(*b))

    def test_fractions_sum_to_one(self, data):
        X, y = data
        b = self._budget(self._gp(), X, y)
        assert np.isclose(b.frac_mean + b.frac_signal + b.frac_noise, 1.0)

    def test_mean_invariance(self, data):
        X, y = data
        b0 = self._budget(self._gp(), X, y)
        b1 = self._budget(self._gp(), X, y + 10.0)
        for f in ("frac_mean", "frac_signal", "frac_noise", "var_ratio"):
            assert np.isclose(getattr(b0, f), getattr(b1, f)), f

    def test_scale_invariance(self, data):
        X, y = data
        a = 7.0
        b0 = self._budget(self._gp(eta=1.0, sigma=0.3), X, y)
        # scale y, and the model's amplitude and noise, by a
        b1 = self._budget(self._gp(eta=a, sigma=a * 0.3), X, a * y)
        for f in ("frac_mean", "frac_signal", "frac_noise", "var_ratio"):
            assert np.isclose(getattr(b0, f), getattr(b1, f)), f

    def test_linear_mean_contributes_variance(self, data):
        X, y = data
        gp_lin = self._gp(mean=Linear(coeffs=pt.as_tensor_variable(np.array([1.0]))))
        assert self._budget(gp_lin, X, y).frac_mean > 0.0
        # a constant mean sets the level, not the variance -> 0 contribution
        gp_const = self._gp(mean=Constant(c=5.0))
        assert np.isclose(self._budget(gp_const, X, y).frac_mean, 0.0)

    def test_heteroskedastic_sigma(self, data):
        X, y = data
        Xt = pt.as_tensor_variable(X)
        sigma = 0.1 + 0.05 * Xt[:, 0] ** 2  # length-N vector
        gp = Unapproximated(kernel=ExpQuad(input_dim=1, ls=1.0), mean=Zero(), sigma=sigma)
        b = variance_budget(gp, Xt, pt.as_tensor_variable(y))._make(
            _eval(*variance_budget(gp, Xt, pt.as_tensor_variable(y)))
        )
        expected_noise = float(np.mean((0.1 + 0.05 * X[:, 0] ** 2) ** 2))
        assert np.isclose(b.noise_var, expected_noise)
        assert np.isclose(b.frac_mean + b.frac_signal + b.frac_noise, 1.0)


class TestVFEDiagnostics:
    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(1)
        X = np.sort(rng.uniform(0, 5, 30))[:, None].astype(np.float64)
        y = np.sin(X.ravel()) + 0.1 * rng.standard_normal(30) + 3.0  # offset
        Z = np.linspace(0.5, 4.5, 6)[:, None].astype(np.float64)
        return X, y, Z

    def _diag(self, X, y, Z, eta=1.0, sigma=0.3):
        vfe = VFE(
            kernel=eta**2 * ExpQuad(input_dim=1, ls=1.0),
            mean=Zero(),
            sigma=sigma,
            inducing_variable=Points(pt.as_tensor_variable(Z)),
        )
        d = vfe_diagnostics(vfe, pt.as_tensor_variable(X), pt.as_tensor_variable(y))
        return d._make(_eval(*d))

    def test_budget_fields_present(self, setup):
        X, y, Z = setup
        d = self._diag(X, y, Z)
        assert np.isclose(d.frac_mean + d.frac_signal + d.frac_noise, 1.0)

    def test_excess_fit_scale_invariant(self, setup):
        X, y, Z = setup
        a = 5.0
        d0 = self._diag(X, y, Z, eta=1.0, sigma=0.3)
        d1 = self._diag(X, a * y, Z, eta=a, sigma=a * 0.3)
        assert np.isclose(d0.excess_fit_per_n, d1.excess_fit_per_n)
