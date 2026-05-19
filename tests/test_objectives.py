"""Tests for the objectives in ptgp.objectives."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

import pytensor.assumptions as pta
from ptgp.gp import SVGP, VFE, Unapproximated, VariationalParams
from ptgp.inducing import Points
from ptgp.kernels import ExpQuad
from ptgp.likelihoods import Gaussian
from ptgp.mean import Zero
from ptgp.objectives import collapsed_elbo, elbo, marginal_log_likelihood


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
