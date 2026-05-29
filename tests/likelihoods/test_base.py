"""Tests for the likelihood base class — configurable inverse-link behavior."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp.likelihoods import Bernoulli, Gaussian, NegativeBinomial, Poisson, StudentT


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


def _hetero(X):
    return 0.5 + 0.1 * X[:, 0] ** 2


def _provider(arr, kind):
    if kind == "symbolic":
        return pt.matrix("Xtr", shape=(None, arr.shape[1]))
    if kind == "baked":
        return pt.as_tensor_variable(arr)
    return pm.Data("Xtr", arr)


class TestConfigurableLink:
    def test_bernoulli_logit_link(self):
        """Bernoulli with logit link should differ from probit but still be valid."""
        mu, var = np.array([0.0, 1.0]), np.array([0.5, 0.5])
        y = np.array([1.0, 0.0])

        ve_probit = _eval(
            Bernoulli().variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        ve_logit = _eval(
            Bernoulli(invlink=pt.sigmoid).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )

        assert np.all(ve_probit < 0) and np.all(ve_logit < 0)
        assert not np.allclose(ve_probit, ve_logit)

    def test_poisson_custom_link_uses_quadrature(self):
        """Poisson with non-exp link should fall back to quadrature and still work."""
        mu, var = np.array([1.0]), np.array([0.1])
        y = np.array([2.0])

        def softplus(f):
            return pt.log1p(pt.exp(f))

        ve = _eval(
            Poisson(invlink=softplus).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        assert np.isfinite(ve).all()


class TestCloneReplaceData:
    @pytest.mark.parametrize("kind", ["symbolic", "baked", "pm_data"])
    def test_reroot_matches_direct_build(self, kind):
        """Re-rooting yields the parameter evaluated at the new inputs, however
        the training data was supplied. ``train`` and ``new`` differ in length so
        a baked array's statically frozen row dimension cannot be re-imposed."""
        train = np.linspace(-1.0, 1.0, 5)[:, None]
        new = np.array([[0.0], [1.0], [2.0]])
        X_new = pt.matrix("X_new", shape=(None, 1))
        with pm.Model():
            rerooted = Gaussian(_hetero(_provider(train, kind))).clone_replace_data(X_new).sigma
        got = pytensor.function([X_new], rerooted)(new)
        np.testing.assert_allclose(got, _hetero(new))

    def test_reroot_matches_multidim_input(self):
        new = np.array([[0.0, 1.0], [2.0, 3.0]])
        X = pt.matrix("X", shape=(None, 2))
        X_new = pt.matrix("X_new", shape=(None, 2))
        rerooted = Gaussian(0.1 + pt.sum(X**2, axis=1)).clone_replace_data(X_new).sigma
        got = pytensor.function([X_new], rerooted)(new)
        np.testing.assert_allclose(got, 0.1 + np.sum(new**2, axis=1))

    def test_reroots_all_tensor_params(self):
        X = pt.matrix("X", shape=(None, 1))
        new = np.array([[1.0], [2.0], [3.0]])
        out = StudentT(nu=3.0 + X[:, 0] ** 2, sigma=_hetero(X)).clone_replace_data(
            pt.as_tensor_variable(new)
        )
        nu_val, sigma_val = _eval(out.nu, out.sigma)
        np.testing.assert_allclose(nu_val, 3.0 + new[:, 0] ** 2)
        np.testing.assert_allclose(sigma_val, _hetero(new))

    def test_negative_binomial_alpha(self):
        X = pt.matrix("X", shape=(None, 1))
        new = np.array([[0.0], [2.0]])
        out = NegativeBinomial(alpha=_hetero(X)).clone_replace_data(pt.as_tensor_variable(new))
        np.testing.assert_allclose(_eval(out.alpha), _hetero(new))

    def test_leaves_scalar_params_untouched(self):
        X = pt.matrix("X", shape=(None, 1))
        lik = StudentT(nu=4.0, sigma=_hetero(X))
        out = lik.clone_replace_data(pt.matrix("X_new", shape=(None, 1)))
        assert out.nu is lik.nu

    def test_returns_copy_leaving_original_untouched(self):
        X = pt.matrix("X", shape=(None, 1))
        lik = Gaussian(_hetero(X))
        original = lik.sigma
        out = lik.clone_replace_data(pt.matrix("X_new", shape=(None, 1)))
        assert lik.sigma is original
        assert out.sigma is not original

    def test_preserves_non_parameter_attributes(self):
        X = pt.matrix("X", shape=(None, 1))
        lik = StudentT(nu=_hetero(X), sigma=0.5, n_points=33)
        assert lik.clone_replace_data(pt.matrix("X_new", shape=(None, 1))).n_points == 33

    def test_scalar_hyperparam_survives_reroot(self):
        """A parameter mixing data with a scalar placeholder re-roots only the
        data; the scalar still flows through to the result."""
        X = pt.matrix("X", shape=(None, 1))
        hyp = pt.scalar("hyp")
        new = np.array([[0.0], [1.0], [2.0]])
        X_new = pt.matrix("X_new", shape=(None, 1))
        out = Gaussian(hyp + 0.05 * X[:, 0] ** 2).clone_replace_data(X_new)
        got = pytensor.function([X_new, hyp], out.sigma)(new, 2.0)
        np.testing.assert_allclose(got, 2.0 + 0.05 * new[:, 0] ** 2)

    def test_eager_numpy_parameter_is_not_rerooted(self):
        """Pure-numpy arithmetic collapses the data dependence to a fixed vector
        before pytensor sees it, leaving nothing to re-root."""
        arr = np.linspace(-1.0, 1.0, 5)[:, None]
        sigma = pt.as_tensor_variable(0.1 + 0.05 * arr[:, 0] ** 2)
        out = Gaussian(sigma).clone_replace_data(pt.matrix("X_new", shape=(None, 1)))
        assert out.sigma is sigma

    def test_mismatched_feature_count_is_not_rerooted(self):
        sigma = 0.1 + pt.sum(pt.matrix("Xtr", shape=(None, 3)) ** 2, axis=1)
        out = Gaussian(sigma).clone_replace_data(pt.matrix("X_new", shape=(None, 1)))
        assert out.sigma is sigma

    def test_one_dimensional_X_is_rejected(self):
        """A 1D X is indistinguishable from a per-observation parameter, so
        re-rooting against it would risk replacing the parameter itself."""
        X = pt.matrix("X", shape=(None, 1))
        lik = Gaussian(0.1 + 0.05 * X[:, 0] ** 2)
        with pytest.raises(ValueError, match=r"requires a 2D design matrix"):
            lik.clone_replace_data(pt.vector("X_new", shape=(None,)))
