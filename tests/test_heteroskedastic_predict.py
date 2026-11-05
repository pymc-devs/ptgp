import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp.gp import SVGP, VFE, Unapproximated, init_variational_params
from ptgp.inducing import Points
from ptgp.kernels import ExpQuad
from ptgp.likelihoods import Gaussian
from ptgp.mean import Zero
from ptgp.objectives import collapsed_elbo, elbo


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


def _k(A, B):
    """ExpQuad with ls=1 in numpy."""
    return np.exp(-0.5 * (A[:, 0:1] - B[:, 0:1].T) ** 2)


def _np_predict(which, X, y, X_new, Z, s2):
    """Reference posterior f-marginals for the exact GP and VFE with noise variances ``s2``."""
    Ksf, kss = _k(X_new, X), np.ones(len(X_new))
    if which == "gp":
        K = _k(X, X) + np.diag(s2)
        return Ksf @ np.linalg.solve(K, y), kss - np.sum(Ksf * np.linalg.solve(K, Ksf.T).T, 1)
    Kuu, Kuf, Kus = _k(Z, Z), _k(Z, X), _k(Z, X_new)
    Sigma = Kuu + (Kuf / s2) @ Kuf.T
    m = Kus.T @ np.linalg.solve(Sigma, (Kuf / s2) @ y)
    q = np.sum(Kus * np.linalg.solve(Kuu, Kus), 0) - np.sum(Kus * np.linalg.solve(Sigma, Kus), 0)
    return m, kss - q


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


def test_pm_data_x_reroots():
    """A pm.Data design matrix passed as ``x=`` re-roots like a symbolic placeholder."""
    X_train_arr = np.linspace(-1.5, 1.5, 10)[:, None].astype(np.float64)
    X_new_arr = np.linspace(-2.0, 2.0, 5)[:, None].astype(np.float64)
    y_train_arr = np.sin(X_train_arr.ravel()).astype(np.float64)
    with pm.Model(coords={"obs": np.arange(10)}):
        X = pm.Data("X", X_train_arr, dims=("obs", "feat"))
        gp = Unapproximated(
            kernel=ExpQuad(input_dim=1, ls=1.0),
            mean=Zero(),
            sigma=_hetero_sigma(X),
            x=X,
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
        sigma=_hetero_sigma(X),
        x=X,
        inducing_variable=Points(pt.as_tensor_variable(Z)),
    )
    elbo = collapsed_elbo(vfe, X, y).elbo
    val = pytensor.function([X, y], elbo)(X_train, y_train)
    assert np.isfinite(val)


@pytest.mark.parametrize("which", ["gp", "vfe", "svgp"])
def test_compile_predict_heteroskedastic(which):
    """compile_predict with numeric training data evaluates sigma at X_new for
    incl_lik, and at X_train for the noise in the training covariance."""
    from ptgp.optim import compile_predict

    X_train, y_train, X_new = _data()
    X = pt.matrix("X", shape=(None, 1))
    sigma = _hetero_sigma(X)
    Z_np = np.linspace(-1.5, 1.5, 6)[:, None]
    Z = Points(pt.as_tensor_variable(Z_np))  # cond(Kuu) ~ 2e3
    with pm.Model() as model:
        kernel = ExpQuad(input_dim=1, ls=1.0)
        gp = {
            "gp": lambda: Unapproximated(kernel=kernel, sigma=sigma, x=X),
            "vfe": lambda: VFE(kernel=kernel, sigma=sigma, inducing_variable=Z, x=X),
            "svgp": lambda: SVGP(
                kernel=kernel,
                likelihood=Gaussian(sigma, x=X),
                inducing_variable=Z,
                variational_params=init_variational_params(6),
            ),
        }[which]()
    data = {} if which == "svgp" else {"X_train": X_train, "y_train": y_train}
    extras = [pytensor.shared(v) for v in getattr(gp, "extra_init", ())]
    X_new_t = pt.matrix("X_new", shape=(None, 1))
    kw = dict(extra_vars=gp.extra_vars, shared_extras=extras, **data)
    m, v = compile_predict(gp, X_new_t, model, {}, **kw)(X_new)
    ym, yv = compile_predict(gp, X_new_t, model, {}, incl_lik=True, **kw)(X_new)
    np.testing.assert_allclose(ym, m, atol=1e-12)
    np.testing.assert_allclose(yv, v + _hetero_sigma(X_new) ** 2, atol=1e-10)

    if which != "svgp":
        m_ref, v_ref = _np_predict(
            which, X_train, y_train, X_new, Z_np, _hetero_sigma(X_train) ** 2
        )
        np.testing.assert_allclose(m, m_ref, atol=1e-8)
        np.testing.assert_allclose(v, v_ref, atol=1e-8)


def test_minibatch_elbo_heteroskedastic():
    """``elbo`` evaluates sigma at the minibatch inputs and scales the data term by N / b."""
    X_train, y_train, _ = _data()
    N, idx = len(X_train), np.arange(0, 25, 3)
    M = 6
    rng = np.random.default_rng(1)
    Z_np = np.linspace(-1.5, 1.5, M)[:, None]
    q_mu = rng.normal(0, 0.5, M)
    q_sqrt = np.tril(rng.normal(0, 0.1, (M, M)), -1) + np.diag(rng.uniform(0.3, 0.8, M))
    vp = init_variational_params(M, q_mu_init=q_mu, q_sqrt_init=q_sqrt)
    X = pt.matrix("X", shape=(None, 1))
    X_mb = pt.matrix("X_mb", shape=(None, 1))
    y_mb = pt.vector("y_mb", shape=(None,))
    svgp = SVGP(
        kernel=ExpQuad(input_dim=1, ls=1.0),
        likelihood=Gaussian(_hetero_sigma(X), x=X),
        inducing_variable=Points(pt.as_tensor_variable(Z_np)),
        variational_params=vp,
    )
    terms = elbo(svgp, X_mb, y_mb, n_data=N)
    fn = pytensor.function([X_mb, y_mb, *vp.extra_vars], list(terms))
    got_elbo, got_ve, got_kl = fn(X_train[idx], y_train[idx], *vp.extra_init)

    # Whitened SVGP q(f) at the minibatch inputs, in numpy.
    Xb, yb = X_train[idx], y_train[idx]
    A = np.linalg.solve(np.linalg.cholesky(_k(Z_np, Z_np) + 1e-6 * np.eye(M)), _k(Z_np, Xb))
    m = A.T @ q_mu
    v = 1.0 - np.sum(A**2, 0) + np.sum((A.T @ q_sqrt) ** 2, 1)
    s2 = _hetero_sigma(Xb) ** 2
    ve = -0.5 * (np.log(2 * np.pi * s2) + ((yb - m) ** 2 + v) / s2)
    np.testing.assert_allclose(got_ve, N / len(idx) * ve.sum(), rtol=1e-8)
    np.testing.assert_allclose(got_elbo, got_ve - got_kl, rtol=1e-10)


@pytest.mark.parametrize("which", ["gp", "vfe", "svgp"])
def test_fit_and_predict_heteroskedastic(which):
    """pg.fit builds its own symbolic X; the objectives re-root sigma onto it."""
    import ptgp as pg

    X_train, y_train, X_new = _data()
    X = pt.matrix("X", shape=(None, 1))
    Z = Points(pt.as_tensor_variable(np.linspace(-1.5, 1.5, 6)[:, None]))
    with pm.Model():
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        s0 = pm.HalfNormal("s0", sigma=0.5)
        sigma = s0 + 0.05 * X[:, 0] ** 2
        kernel = ExpQuad(input_dim=1, ls=ls)
        gp = {
            "gp": lambda: Unapproximated(kernel=kernel, sigma=sigma, x=X),
            "vfe": lambda: VFE(kernel=kernel, sigma=sigma, inducing_variable=Z, x=X),
            "svgp": lambda: SVGP(
                kernel=kernel,
                likelihood=Gaussian(sigma, x=X),
                inducing_variable=Z,
                variational_params=init_variational_params(6),
            ),
        }[which]()
        fit = pg.fit(gp, X_train, y_train, options={"maxiter": 5})
    data = {} if which == "svgp" else {"X_train": X_train, "y_train": y_train}
    _, v = pg.predict(gp, X_new, fit, **data)
    _, yv = pg.predict(gp, X_new, fit, incl_lik=True, **data)
    expected = (fit.params["s0"] + 0.05 * X_new[:, 0] ** 2) ** 2
    np.testing.assert_allclose(yv - v, expected, atol=1e-10)
