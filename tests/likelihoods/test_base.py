"""Tests for likelihood nodes: inverse links, re-rooting, quadrature, and gradients."""

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


class TestAt:
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

    @pytest.mark.parametrize("kind", ["symbolic", "pm_data"])
    def test_data_dependent_parameter_without_x_raises(self, kind):
        with pm.Model():
            Xtr = _provider(np.zeros((5, 1)), kind)
            with pytest.raises(ValueError, match="no design matrix was given"):
                Gaussian(_hetero(Xtr))

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

    Y = np.array([0.5, -0.3])
    MU = np.array([0.0, 0.2])
    VAR = np.array([1.0, 0.5])

    def _ve_grad(self, lik, wrt, givens):
        args = [pt.as_tensor_variable(a) for a in (self.Y, self.MU, self.VAR)]
        ve = pt.sum(lik.variational_expectation(*args))
        return pytensor.function(list(givens), pytensor.grad(ve, wrt))

    def _expected(self, s):
        """d/dalpha of the Gaussian expected log-likelihood, with ds/dalpha = 1."""
        return np.sum(-1.0 / s + ((self.Y - self.MU) ** 2 + self.VAR) / s**3)

    def test_grad_homoskedastic(self):
        alpha = pt.scalar("alpha")
        lik = Gaussian(alpha)  # node, has_data=False
        val = self._ve_grad(lik, alpha, [alpha])(0.7)
        np.testing.assert_allclose(val, self._expected(0.7))

    def test_grad_heteroskedastic(self):
        X = pt.matrix("X", shape=(None, 1))
        alpha = pt.scalar("alpha")
        lik = Gaussian(alpha + 0.05 * X[:, 0] ** 2, x=X)  # node, has_data=True
        Xv = np.array([[0.0], [1.0]])
        val = self._ve_grad(lik, alpha, [X, alpha])(Xv, 0.7)
        np.testing.assert_allclose(val, self._expected(0.7 + 0.05 * Xv[:, 0] ** 2))


@pytest.mark.parametrize("n", [3, 20], ids=["n_ne_points", "n_eq_points"])
@pytest.mark.parametrize("family", ["student_t", "negative_binomial"])
def test_quadrature_with_per_point_parameters(family, n):
    """Per-point parameters give the same quadrature as building each point separately."""
    rng = np.random.default_rng(0)
    y = rng.poisson(2.0, n).astype(float)
    mu, var = rng.normal(0.5, 0.3, n), rng.uniform(0.1, 0.5, n)
    p = rng.uniform(1.0, 3.0, n)
    X = pt.matrix("X", shape=(None, 1))
    Xv = p[:, None]

    def make(param, x=None):
        if family == "student_t":
            return StudentT(nu=3.0 + param, sigma=param, x=x)
        return NegativeBinomial(alpha=param, x=x)

    def evaluate(lik, args, givens=()):
        args = [pt.as_tensor_variable(a) for a in args]
        outs = [
            lik.variational_expectation(*args),
            *lik.predict_mean_and_var(*args[1:]),
            lik.predict_log_density(*args),
        ]
        return pytensor.function(list(givens), outs)(*[Xv for _ in givens])

    got = evaluate(make(X[:, 0], x=X), (y, mu, var), givens=[X])
    want = [evaluate(make(p[i]), (y[i : i + 1], mu[i : i + 1], var[i : i + 1])) for i in range(n)]
    for k, out in enumerate(got):
        np.testing.assert_allclose(out, np.concatenate([w[k] for w in want]), rtol=1e-10)
