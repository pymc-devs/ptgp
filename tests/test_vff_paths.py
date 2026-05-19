"""Op-routing and equivalence tests for VFF on the SVGP path."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp.gp.svgp import SVGP
from ptgp.inducing_fourier import FourierFeatures1D
from ptgp.kernels.stationary import Matern32
from ptgp.likelihoods.gaussian import Gaussian


def _make(whiten, num_frequencies=8):
    f = FourierFeatures1D(0, 1, num_frequencies=num_frequencies)
    k = 1.0 * Matern32(input_dim=1, ls=0.4)
    return SVGP(
        kernel=k,
        likelihood=Gaussian(sigma=0.1),
        inducing_variable=f,
        whiten=whiten,
    )


def test_whitened_path_uses_sqrt_solve(monkeypatch):
    counts = {"sqrt": 0, "K_uu": 0, "solve": 0}
    orig_sqrt = FourierFeatures1D.Kuu_sqrt_solve

    def spy(self, kernel, rhs):
        counts["sqrt"] += 1
        return orig_sqrt(self, kernel, rhs)

    def forbid(name):
        def _f(self, *a, **k):
            counts[name] += 1
            raise AssertionError(f"{name} called on whitened path")

        return _f

    monkeypatch.setattr(FourierFeatures1D, "Kuu_sqrt_solve", spy)
    monkeypatch.setattr(FourierFeatures1D, "K_uu", forbid("K_uu"))
    monkeypatch.setattr(FourierFeatures1D, "Kuu_solve", forbid("solve"))

    svgp = _make(whiten=True)
    _ = svgp.predict_marginal(pt.matrix("_X"))
    _ = svgp.prior_kl()
    assert counts["sqrt"] >= 1
    assert counts["K_uu"] == 0 and counts["solve"] == 0


def test_unwhitened_path_uses_solve_and_logdet(monkeypatch):
    counts = {"sqrt": 0, "K_uu": 0, "solve": 0, "logdet": 0}

    def forbid(name):
        def _f(self, *a, **k):
            counts[name] += 1
            raise AssertionError(f"{name} called on unwhitened path")

        return _f

    orig_solve = FourierFeatures1D.Kuu_solve
    orig_logdet = FourierFeatures1D.Kuu_logdet

    def solve_spy(self, kernel, rhs):
        counts["solve"] += 1
        return orig_solve(self, kernel, rhs)

    def logdet_spy(self, kernel):
        counts["logdet"] += 1
        return orig_logdet(self, kernel)

    monkeypatch.setattr(FourierFeatures1D, "Kuu_sqrt_solve", forbid("sqrt"))
    monkeypatch.setattr(FourierFeatures1D, "K_uu", forbid("K_uu"))
    monkeypatch.setattr(FourierFeatures1D, "Kuu_solve", solve_spy)
    monkeypatch.setattr(FourierFeatures1D, "Kuu_logdet", logdet_spy)

    svgp = _make(whiten=False)
    _ = svgp.predict_marginal(pt.matrix("_X"))
    _ = svgp.prior_kl()
    assert counts["solve"] >= 1 and counts["logdet"] >= 1
    assert counts["K_uu"] == 0 and counts["sqrt"] == 0


def test_no_dense_cholesky_on_Kuu():
    f = FourierFeatures1D(0, 1, num_frequencies=12)
    svgp = _make(whiten=True, num_frequencies=12)
    N = 50
    X_sym = pt.tensor("_X", shape=(N, 1), dtype="float64")
    fmean, fvar = svgp.predict_marginal(X_sym)
    fn = pytensor.function([X_sym], [fmean, fvar])

    M = f.num_inducing
    fg = fn.maker.fgraph
    shape_of = fg.shape_feature.shape_of
    for node in fg.toposort():
        if node.op.__class__.__name__ == "Cholesky":
            in_shape = shape_of[node.inputs[0]]
            resolved = tuple(int(s.data) if hasattr(s, "data") else None for s in in_shape)
            assert resolved != (M, M), f"dense MxM Cholesky on Kuu at {node}"


@pytest.mark.parametrize("kind", ["diagonal", "full"])
def test_whitened_vs_unwhitened_agreement(kind):
    f = FourierFeatures1D(0, 1, num_frequencies=10)
    k = 1.0 * Matern32(input_dim=1, ls=0.3)
    rng = np.random.default_rng(0)
    M = f.num_inducing

    m_u = rng.standard_normal(M)
    if kind == "diagonal":
        q_sqrt_u = np.diag(np.exp(0.1 * rng.standard_normal(M)))
    else:
        A = rng.standard_normal((M, M))
        q_sqrt_u = np.linalg.cholesky(A @ A.T + 1e-3 * np.eye(M))
    S_u = q_sqrt_u @ q_sqrt_u.T

    R_inv = f.Kuu_sqrt_solve(k, pt.as_tensor(np.eye(M))).eval()

    q_mu_v = R_inv @ m_u
    S_v = R_inv @ S_u @ R_inv.T
    q_sqrt_v = np.linalg.cholesky(S_v + 1e-10 * np.eye(M))

    svgp_u = SVGP(
        kernel=k,
        likelihood=Gaussian(sigma=0.1),
        inducing_variable=f,
        whiten=False,
        q_mu=pt.as_tensor(m_u),
        q_sqrt=pt.as_tensor(q_sqrt_u),
    )
    svgp_w = SVGP(
        kernel=k,
        likelihood=Gaussian(sigma=0.1),
        inducing_variable=f,
        whiten=True,
        q_mu=pt.as_tensor(q_mu_v),
        q_sqrt=pt.as_tensor(q_sqrt_v),
    )

    X_test = np.linspace(f.a + 0.05, f.b - 0.05, 30)[:, None]
    m_u_out, v_u_out = [t.eval() for t in svgp_u.predict_marginal(pt.as_tensor(X_test))]
    m_w_out, v_w_out = [t.eval() for t in svgp_w.predict_marginal(pt.as_tensor(X_test))]
    np.testing.assert_allclose(m_u_out, m_w_out, atol=1e-6)
    np.testing.assert_allclose(v_u_out, v_w_out, atol=1e-6)

    np.testing.assert_allclose(svgp_u.prior_kl().eval(), svgp_w.prior_kl().eval(), atol=1e-6)
