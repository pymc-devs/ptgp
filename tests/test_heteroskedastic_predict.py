import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from ptgp.gp import SVGP, VFE, Unapproximated, init_variational_params
from ptgp.inducing import Points
from ptgp.kernels import ExpQuad
from ptgp.likelihoods import Gaussian
from ptgp.mean import Zero
from ptgp.objectives import collapsed_elbo


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


def _data():
    rng = np.random.default_rng(0)
    X_train = np.sort(rng.uniform(-1.5, 1.5, 25))[:, None].astype(np.float64)
    y_train = np.sin(X_train.ravel()) + 0.1 * rng.standard_normal(25)
    X_new = np.linspace(-2.0, 2.0, 10)[:, None].astype(np.float64)
    return X_train, y_train, X_new


def _hetero_sigma(X):
    return 0.1 + 0.05 * X[:, 0] ** 2


class TestUnapproximatedHeteroskedastic:
    def test_incl_lik_adds_pointwise_noise(self):
        X_train, y_train, X_new = _data()
        X = pt.matrix("X", shape=(None, 1))
        gp = Unapproximated(
            kernel=ExpQuad(input_dim=1, ls=1.0), mean=Zero(), sigma=_hetero_sigma(X), x=X
        )
        X_new_t = pt.matrix("X_new", shape=(None, 1))
        y_t = pt.vector("y", shape=(None,))

        fmean, fvar = gp.predict_marginal(X_new_t, X, y_t)
        ymean, yvar = gp.predict_marginal(X_new_t, X, y_t, incl_lik=True)
        fn = pytensor.function([X, X_new_t, y_t], [fmean, fvar, ymean, yvar])
        m, v, ym, yv = fn(X_train, X_new, y_train)

        np.testing.assert_allclose(ym, m, atol=1e-12)
        expected_sigma_new = 0.1 + 0.05 * X_new[:, 0] ** 2
        np.testing.assert_allclose(yv, v + expected_sigma_new**2, atol=1e-10)


class TestVFEHeteroskedastic:
    def _build(self, X):
        Z = np.linspace(-1.5, 1.5, 6)[:, None].astype(np.float64)
        return VFE(
            kernel=ExpQuad(input_dim=1, ls=1.0),
            mean=Zero(),
            sigma=_hetero_sigma(X), x=X,
            inducing_variable=Points(pt.as_tensor_variable(Z)),
        )

    def test_incl_lik_adds_pointwise_noise(self):
        X_train, y_train, X_new = _data()
        X = pt.matrix("X", shape=(None, 1))
        vfe = self._build(X)
        X_new_t = pt.matrix("X_new", shape=(None, 1))
        y_t = pt.vector("y", shape=(None,))

        fmean, fvar = vfe.predict_marginal(X_new_t, X, y_t)
        ymean, yvar = vfe.predict_marginal(X_new_t, X, y_t, incl_lik=True)
        fn = pytensor.function([X, X_new_t, y_t], [fmean, fvar, ymean, yvar])
        m, v, ym, yv = fn(X_train, X_new, y_train)

        np.testing.assert_allclose(ym, m, atol=1e-12)
        expected_sigma_new = 0.1 + 0.05 * X_new[:, 0] ** 2
        np.testing.assert_allclose(yv, v + expected_sigma_new**2, atol=1e-10)


class TestSVGPHeteroskedastic:
    def test_incl_lik_adds_pointwise_noise(self):
        X_train, y_train, X_new = _data()
        M = 6
        Z = np.linspace(-1.5, 1.5, M)[:, None].astype(np.float64)
        vp = init_variational_params(M)
        X = pt.matrix("X", shape=(None, 1))
        svgp = SVGP(
            kernel=ExpQuad(input_dim=1, ls=1.0),
            mean=Zero(),
            likelihood=Gaussian(_hetero_sigma(X), x=X),
            inducing_variable=Points(pt.as_tensor_variable(Z)),
            variational_params=vp,
        )
        X_new_t = pt.matrix("X_new", shape=(None, 1))

        _, fvar = svgp.predict_marginal(X_new_t)
        _, yvar = svgp.predict_marginal(X_new_t, incl_lik=True)
        fn = pytensor.function([X_new_t, *vp.extra_vars], [fvar, yvar], on_unused_input="ignore")
        v, yv = fn(X_new, *vp.extra_init)

        expected_sigma_new = 0.1 + 0.05 * X_new[:, 0] ** 2
        np.testing.assert_allclose(yv - v, expected_sigma_new**2, atol=1e-10)


class TestScalarSigmaUnaffected:
    """Regression: scalar sigma (no graph dependence on X) still works."""

    def test_unapproximated_scalar(self):
        X_train, y_train, X_new = _data()
        gp = Unapproximated(kernel=ExpQuad(input_dim=1, ls=1.0), mean=Zero(), sigma=0.3)
        fmean, fvar = gp.predict_marginal(
            pt.as_tensor_variable(X_new),
            pt.as_tensor_variable(X_train),
            pt.as_tensor_variable(y_train),
            incl_lik=True,
        )
        m, v = _eval(fmean, fvar)
        assert m.shape == (X_new.shape[0],)
        # Predictive variance must include scalar 0.09 noise on every point.
        # Lower bound: at least sigma^2 = 0.09 added.
        fmean_no_lik, fvar_no_lik = gp.predict_marginal(
            pt.as_tensor_variable(X_new),
            pt.as_tensor_variable(X_train),
            pt.as_tensor_variable(y_train),
        )
        v_no_lik = _eval(fvar_no_lik)
        np.testing.assert_allclose(v - v_no_lik, 0.09, atol=1e-10)

    def test_vfe_scalar(self):
        X_train, y_train, X_new = _data()
        Z = np.linspace(-1.5, 1.5, 6)[:, None].astype(np.float64)
        vfe = VFE(
            kernel=ExpQuad(input_dim=1, ls=1.0),
            mean=Zero(),
            sigma=0.3,
            inducing_variable=Points(pt.as_tensor_variable(Z)),
        )
        fmean, fvar = vfe.predict_marginal(
            pt.as_tensor_variable(X_new),
            pt.as_tensor_variable(X_train),
            pt.as_tensor_variable(y_train),
        )
        fmean2, fvar2 = vfe.predict_marginal(
            pt.as_tensor_variable(X_new),
            pt.as_tensor_variable(X_train),
            pt.as_tensor_variable(y_train),
            incl_lik=True,
        )
        v, v2 = _eval(fvar, fvar2)
        np.testing.assert_allclose(v2 - v, 0.09, atol=1e-10)


def test_pm_data_sigma_is_detected_as_data_dependent():
    """sigma built against a pm.Data SharedVariable must still be re-rooted.

    The design matrix handle is keyed by identity (the ``x=`` argument), so a
    pm.Data SharedVariable works exactly like a symbolic placeholder — passing
    it as ``x=`` lets at swap it for the test inputs.
    """
    X_train_arr = np.linspace(-1.5, 1.5, 10)[:, None].astype(np.float64)
    X_new_arr = np.linspace(-2.0, 2.0, 5)[:, None].astype(np.float64)
    y_train_arr = np.sin(X_train_arr.ravel()).astype(np.float64)
    with pm.Model(coords={"obs": np.arange(10)}):
        X = pm.Data("X", X_train_arr, dims=("obs", "feat"))
        gp = Unapproximated(
            kernel=ExpQuad(input_dim=1, ls=1.0),
            mean=Zero(),
            sigma=_hetero_sigma(X), x=X,
        )
    X_new_t = pt.matrix("X_new", shape=(None, 1))
    y_t = pt.vector("y", shape=(None,))
    fmean, fvar = gp.predict_marginal(X_new_t, X, y_t)
    ymean, yvar = gp.predict_marginal(X_new_t, X, y_t, incl_lik=True)
    fn = pytensor.function([X_new_t, y_t], [fmean, fvar, ymean, yvar])
    m, v, ym, yv = fn(X_new_arr, y_train_arr)
    expected_sigma_new = 0.1 + 0.05 * X_new_arr[:, 0] ** 2
    np.testing.assert_allclose(yv, v + expected_sigma_new**2, atol=1e-10)


def test_collapsed_elbo_with_heteroskedastic_sigma():
    """Training-side path: collapsed_elbo must run and produce a finite ELBO
    when sigma is a vector graph built against X. Guards the dropped
    callable-sigma shim from regressing.
    """
    X_train, y_train, _ = _data()
    X = pt.matrix("X", shape=(None, 1))
    y = pt.vector("y", shape=(None,))
    Z = np.linspace(-1.5, 1.5, 6)[:, None].astype(np.float64)
    vfe = VFE(
        kernel=ExpQuad(input_dim=1, ls=1.0),
        mean=Zero(),
        sigma=_hetero_sigma(X), x=X,
        inducing_variable=Points(pt.as_tensor_variable(Z)),
    )
    elbo = collapsed_elbo(vfe, X, y).elbo
    val = pytensor.function([X, y], elbo)(X_train, y_train)
    assert np.isfinite(val)
