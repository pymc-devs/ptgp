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
            Bernoulli(link="logit").variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )

        assert np.all(ve_probit < 0) and np.all(ve_logit < 0)
        assert not np.allclose(ve_probit, ve_logit)

    def test_bernoulli_cloglog_link_uses_quadrature(self):
        """cloglog has no closed-form predictive, so it routes through quadrature."""
        mu, var, y = np.array([0.0, 1.0]), np.array([0.5, 0.5]), np.array([1.0, 0.0])
        ve = _eval(
            Bernoulli(link="cloglog").variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        assert np.all(ve < 0)

    def test_unsupported_link_raises(self):
        with pytest.raises(ValueError, match="does not support link"):
            Poisson(link="logit")


class TestCloneReplaceData:
    @pytest.mark.parametrize("kind", ["symbolic", "pm_data"])
    def test_reroot_matches_direct_build(self, kind):
        """Re-rooting yields the parameter evaluated at the new inputs, for the
        variable-length data providers used in prediction. ``train`` and ``new``
        differ in length to exercise re-rooting onto a different number of
        points (a symbolic placeholder or pm.Data is ``(None, D)``)."""
        train = np.linspace(-1.0, 1.0, 5)[:, None]
        new = np.array([[0.0], [1.0], [2.0]])
        X_new = pt.matrix("X_new", shape=(None, 1))
        with pm.Model():
            Xtr = _provider(train, kind)
            rerooted = Gaussian(_hetero(Xtr), x=Xtr).at(X_new).sigma
        got = pytensor.function([X_new], rerooted)(new)
        np.testing.assert_allclose(got, _hetero(new))

    def test_baked_constant_x_reroots_at_matching_length(self):
        """A frozen numpy constant as ``x`` re-roots cleanly only at its own
        length: its static row dimension propagates into the parameter graph at
        build time, and re-rooting (which uses ``graph_replace`` to preserve the
        identity of any RV hyperparameters) cannot un-specialize it afterward.
        Variable-length re-rooting requires a symbolic placeholder or pm.Data."""
        arr = np.linspace(-1.0, 1.0, 5)[:, None]
        same_len = np.linspace(2.0, 3.0, 5)[:, None]
        X_new = pt.matrix("X_new", shape=(None, 1))
        Xc = pt.as_tensor_variable(arr)  # one node, reused as both param input and x=
        rerooted = Gaussian(_hetero(Xc), x=Xc).at(X_new).sigma
        got = pytensor.function([X_new], rerooted)(same_len)
        np.testing.assert_allclose(got, _hetero(same_len))

    def test_reroot_matches_multidim_input(self):
        new = np.array([[0.0, 1.0], [2.0, 3.0]])
        X = pt.matrix("X", shape=(None, 2))
        X_new = pt.matrix("X_new", shape=(None, 2))
        rerooted = Gaussian(0.1 + pt.sum(X**2, axis=1), x=X).at(X_new).sigma
        got = pytensor.function([X_new], rerooted)(new)
        np.testing.assert_allclose(got, 0.1 + np.sum(new**2, axis=1))

    def test_reroots_all_tensor_params(self):
        X = pt.matrix("X", shape=(None, 1))
        new = np.array([[1.0], [2.0], [3.0]])
        out = StudentT(nu=3.0 + X[:, 0] ** 2, sigma=_hetero(X), x=X).at(pt.as_tensor_variable(new))
        nu_val, sigma_val = _eval(out.nu, out.sigma)
        np.testing.assert_allclose(nu_val, 3.0 + new[:, 0] ** 2)
        np.testing.assert_allclose(sigma_val, _hetero(new))

    def test_negative_binomial_alpha(self):
        X = pt.matrix("X", shape=(None, 1))
        new = np.array([[0.0], [2.0]])
        out = NegativeBinomial(alpha=_hetero(X), x=X).at(pt.as_tensor_variable(new))
        np.testing.assert_allclose(_eval(out.alpha), _hetero(new))

    def test_leaves_scalar_params_untouched(self):
        """A data-independent parameter keeps its value through re-rooting (it
        shares the likelihood node, so its output object is rebuilt, but the
        value it computes is unchanged)."""
        X = pt.matrix("X", shape=(None, 1))
        lik = StudentT(nu=4.0, sigma=_hetero(X), x=X)
        out = lik.at(pt.as_tensor_variable(np.array([[0.0], [1.0], [2.0]])))
        np.testing.assert_allclose(_eval(out.nu), 4.0)

    def test_returns_copy_leaving_original_untouched(self):
        X = pt.matrix("X", shape=(None, 1))
        lik = Gaussian(_hetero(X), x=X)
        original = lik.sigma
        out = lik.at(pt.matrix("X_new", shape=(None, 1)))
        assert lik.sigma is original
        assert out.sigma is not original

    def test_preserves_non_parameter_attributes(self):
        X = pt.matrix("X", shape=(None, 1))
        lik = StudentT(nu=_hetero(X), sigma=0.5, n_points=33, x=X)
        assert lik.at(pt.matrix("X_new", shape=(None, 1))).n_points == 33

    def test_scalar_hyperparam_survives_reroot(self):
        """A parameter mixing data with a scalar placeholder re-roots only the
        data; the scalar leaf keeps its identity and flows through to the
        result (graph_replace preserves sibling leaves)."""
        X = pt.matrix("X", shape=(None, 1))
        hyp = pt.scalar("hyp")
        new = np.array([[0.0], [1.0], [2.0]])
        X_new = pt.matrix("X_new", shape=(None, 1))
        out = Gaussian(hyp + 0.05 * X[:, 0] ** 2, x=X).at(X_new)
        got = pytensor.function([X_new, hyp], out.sigma)(new, 2.0)
        np.testing.assert_allclose(got, 2.0 + 0.05 * new[:, 0] ** 2)

    def test_eager_numpy_parameter_is_not_rerooted(self):
        """Pure-numpy arithmetic collapses the data dependence to a fixed vector;
        with no ``x=`` the likelihood carries no design matrix, so ``at`` is a
        no-op and the value is preserved."""
        arr = np.linspace(-1.0, 1.0, 5)[:, None]
        expected = 0.1 + 0.05 * arr[:, 0] ** 2
        lik = Gaussian(pt.as_tensor_variable(expected))
        out = lik.at(pt.matrix("X_new", shape=(None, 1)))
        assert out.sigma is lik.sigma  # at did nothing (no design matrix)
        np.testing.assert_allclose(_eval(out.sigma), expected)

    def test_no_x_means_not_rerooted(self):
        """Omitting ``x=`` is an explicit opt-out: the likelihood carries no
        design matrix, so ``at`` leaves the parameter untouched."""
        sigma = 0.1 + pt.sum(pt.matrix("Xtr", shape=(None, 3)) ** 2, axis=1)
        lik = Gaussian(sigma)
        out = lik.at(pt.matrix("X_new", shape=(None, 3)))
        assert out.sigma is lik.sigma

    def test_one_dimensional_X_new_is_supported(self):
        """With an explicit, identity-keyed handle there is no parameter/data
        ambiguity, so the replacement inputs need not be 2D — a 1D X_new
        substitutes cleanly."""
        X = pt.vector("X", shape=(None,))
        new = np.array([0.0, 1.0, 2.0])
        X_new = pt.vector("X_new", shape=(None,))
        out = Gaussian(0.1 + 0.05 * X**2, x=X).at(X_new)
        got = pytensor.function([X_new], out.sigma)(new)
        np.testing.assert_allclose(got, 0.1 + 0.05 * new**2)


class TestGradientThroughLikelihoodNode:
    """The likelihood node is a pass-through, so gradients must flow through it
    to the parameters' hyperparameters — for both homoskedastic (no design
    matrix) and heteroskedastic nodes."""

    def _ve_grad(self, lik, wrt, givens):
        y = pt.as_tensor_variable(np.array([0.5, -0.3]))
        mu = pt.as_tensor_variable(np.array([0.0, 0.2]))
        var = pt.as_tensor_variable(np.array([1.0, 0.5]))
        ve = pt.sum(lik.variational_expectation(y, mu, var))
        g = pytensor.grad(ve, wrt)
        return pytensor.function(list(givens), g)

    def test_grad_homoskedastic(self):
        alpha = pt.scalar("alpha")
        lik = Gaussian(alpha)  # node, has_data=False
        val = self._ve_grad(lik, alpha, [alpha])(0.7)
        assert np.isfinite(val)

    def test_grad_heteroskedastic(self):
        X = pt.matrix("X", shape=(None, 1))
        alpha = pt.scalar("alpha")
        lik = Gaussian(alpha + 0.05 * X[:, 0] ** 2, x=X)  # node, has_data=True
        fn = self._ve_grad(lik, alpha, [X, alpha])
        val = fn(np.array([[0.0], [1.0]]), 0.7)
        assert np.isfinite(val)
