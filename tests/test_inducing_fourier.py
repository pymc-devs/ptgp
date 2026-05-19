import numpy as np
import pytensor.tensor as pt
import pytest

from ptgp.gp.svgp import SVGP
from ptgp.inducing_fourier import FourierFeatures1D
from ptgp.kernels.stationary import ExpQuad, Matern12, Matern32, Matern52
from ptgp.likelihoods.gaussian import Gaussian
from tests._fixtures.vff_kuu_oracle import (
    oracle_kuf_no_edges,
    oracle_kuu_matern12,
    oracle_kuu_matern32,
    oracle_kuu_matern52,
)


def test_init_validates_a_lt_b():
    with pytest.raises(ValueError, match="a < b"):
        FourierFeatures1D(a=1.0, b=0.0, num_frequencies=8)


def test_init_validates_num_frequencies():
    with pytest.raises(ValueError, match="num_frequencies"):
        FourierFeatures1D(a=0.0, b=1.0, num_frequencies=0)


def test_num_inducing_is_2k_plus_1():
    f = FourierFeatures1D(a=0.0, b=1.0, num_frequencies=8)
    assert f.num_inducing == 17


def test_from_data_validates_shape():
    with pytest.raises(ValueError, match="shape"):
        FourierFeatures1D.from_data(np.zeros((10, 2)), num_frequencies=8)


def test_from_data_buffer():
    X = np.linspace(0, 1, 100)[:, None]
    f = FourierFeatures1D.from_data(X, num_frequencies=8, buffer=0.1)
    assert f.a < 0.0
    assert f.b > 1.0


def _scale_eval(s):
    return float(s.eval()) if hasattr(s, "eval") else float(s)


def test_resolve_bare_matern():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    s, base = f._resolve_scaled_matern(Matern32(input_dim=1, ls=1.0))
    assert isinstance(base, Matern32)
    assert _scale_eval(s) == 1.0


def test_resolve_canonical_eta_squared():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    eta = pt.as_tensor(2.0)
    s, base = f._resolve_scaled_matern(eta**2 * Matern32(input_dim=1, ls=1.0))
    assert isinstance(base, Matern32)
    np.testing.assert_allclose(_scale_eval(s), 4.0, atol=1e-12)


def test_resolve_nested_product_chain():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    eta = pt.as_tensor(2.0)
    s, base = f._resolve_scaled_matern((eta**2 * 1.5) * Matern32(input_dim=1, ls=1.0))
    assert isinstance(base, Matern32)
    np.testing.assert_allclose(_scale_eval(s), 6.0, atol=1e-12)


def test_resolve_rejects_non_matern():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    with pytest.raises(NotImplementedError, match="Matern"):
        f._resolve_scaled_matern(ExpQuad(input_dim=1, ls=1.0))


def test_resolve_rejects_two_kernel_product():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    k = Matern12(input_dim=1, ls=1.0) * Matern32(input_dim=1, ls=1.0)
    with pytest.raises(NotImplementedError, match="separable/product VFF"):
        f._resolve_scaled_matern(k)


def test_resolve_rejects_sum_kernel_as_additive_vff():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    k = Matern12(input_dim=1, ls=1.0) + Matern32(input_dim=1, ls=1.0)
    with pytest.raises(NotImplementedError, match="additive VFF"):
        f.K_uf(k, pt.as_tensor(np.array([[0.5]])))


def test_structured_kuu_base_shapes():
    for cls, expected_R in [(Matern12, 1), (Matern32, 2), (Matern52, 3)]:
        f = FourierFeatures1D(0, 1, num_frequencies=5)
        k = cls(input_dim=1, ls=0.5)
        d, U = f._structured_Kuu_base(k)
        d_v, U_v = d.eval(), U.eval()
        assert d_v.shape == (11,)
        assert U_v.shape == (11, expected_R)


def test_structured_kuu_base_matches_oracle_matern12():
    f = FourierFeatures1D(a=-0.5, b=1.5, num_frequencies=10)
    k = Matern12(input_dim=1, ls=0.3)
    d, U = [t.eval() for t in f._structured_Kuu_base(k)]
    Kuu_struct = np.diag(d) + U @ U.T
    Kuu_oracle = oracle_kuu_matern12(a=-0.5, b=1.5, ms=np.arange(11), ls=0.3)
    np.testing.assert_allclose(Kuu_struct, Kuu_oracle, atol=1e-10)


def test_structured_kuu_base_matches_oracle_matern32():
    f = FourierFeatures1D(a=-0.5, b=1.5, num_frequencies=10)
    k = Matern32(input_dim=1, ls=0.3)
    d, U = [t.eval() for t in f._structured_Kuu_base(k)]
    Kuu_struct = np.diag(d) + U @ U.T
    Kuu_oracle = oracle_kuu_matern32(a=-0.5, b=1.5, ms=np.arange(11), ls=0.3)
    np.testing.assert_allclose(Kuu_struct, Kuu_oracle, atol=1e-10)


def test_structured_kuu_base_matches_oracle_matern52():
    f = FourierFeatures1D(a=-0.5, b=1.5, num_frequencies=10)
    k = Matern52(input_dim=1, ls=0.3)
    d, U = [t.eval() for t in f._structured_Kuu_base(k)]
    Kuu_struct = np.diag(d) + U @ U.T
    Kuu_oracle = oracle_kuu_matern52(a=-0.5, b=1.5, ms=np.arange(11), ls=0.3)
    np.testing.assert_allclose(Kuu_struct, Kuu_oracle, atol=1e-10)


@pytest.mark.parametrize(
    "kernel_cls,bad_K",
    [(Matern32, 2), (Matern52, 3)],
)
def test_structured_Kuu_rank_deficient_guard(kernel_cls, bad_K):
    """num_inducing must exceed R; raise with a clear message otherwise."""
    f = FourierFeatures1D(0, 1, num_frequencies=bad_K - 1)
    with pytest.raises(ValueError, match="num_frequencies"):
        f._structured_Kuu(kernel_cls(input_dim=1, ls=1.0))


def test_Kuu_reference_scale_convention():
    f = FourierFeatures1D(0, 1, num_frequencies=8)
    base = Matern32(input_dim=1, ls=0.5)
    eta = pt.as_tensor(2.0)
    K_scaled = f.K_uu(eta**2 * base).eval()
    K_base = f.K_uu(base).eval()
    np.testing.assert_allclose(K_scaled, K_base / 4.0, atol=1e-10)


def test_Kuu_matches_oracle_with_reference_scale():
    """K_uu(eta**2 * Matern) matches oracle / eta**2."""
    f = FourierFeatures1D(a=-0.5, b=1.5, num_frequencies=10)
    base = Matern32(input_dim=1, ls=0.3)
    eta = pt.as_tensor(1.5)
    K = f.K_uu(eta**2 * base).eval()
    K_oracle = oracle_kuu_matern32(a=-0.5, b=1.5, ms=np.arange(11), ls=0.3) / (1.5**2)
    np.testing.assert_allclose(K, K_oracle, atol=1e-10)


def test_Kuf_shape_and_reference_scale_convention():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    k = Matern32(input_dim=1, ls=0.5)
    X = pt.as_tensor(np.linspace(0.1, 0.9, 7)[:, None])
    Kuf_base = f.K_uf(k, X).eval()
    Kuf_scaled = f.K_uf(2.0 * k, X).eval()
    assert Kuf_base.shape == (9, 7)
    np.testing.assert_allclose(Kuf_scaled, Kuf_base, atol=1e-10)


def test_Kuf_active_dims():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    k = Matern32(input_dim=2, active_dims=[1], ls=0.5)
    X = pt.as_tensor(np.column_stack([np.linspace(5, 10, 5), np.linspace(0.1, 0.9, 5)]))
    Kuf = f.K_uf(k, X).eval()
    assert Kuf.shape == (9, 5)


def test_Kuf_matches_oracle_no_edges():
    """K_uf on the interior matches the no-edges basis evaluation."""
    f = FourierFeatures1D(a=-0.5, b=1.5, num_frequencies=10)
    k = Matern32(input_dim=1, ls=0.3)
    X_np = np.linspace(0.0, 1.0, 13)[:, None]
    X = pt.as_tensor(X_np)
    Kuf = f.K_uf(k, X).eval()
    Kuf_oracle = oracle_kuf_no_edges(a=-0.5, b=1.5, ms=np.arange(11), X=X_np)
    np.testing.assert_allclose(Kuf, Kuf_oracle, atol=1e-12)


def test_Kuf_matern12_reference_edges():
    f = FourierFeatures1D(a=0.0, b=1.0, num_frequencies=3, allow_extrapolation=True)
    k = Matern12(input_dim=1, ls=0.5)
    X_np = np.array([[-0.2], [0.25], [1.3]])
    Kuf = f.K_uf(k, pt.as_tensor(X_np)).eval()

    omegas = 2.0 * np.pi * np.arange(4) / (f.b - f.a)
    interior = np.cos(omegas * (0.25 - f.a))
    left_edge = np.exp(-abs(-0.2 - f.a) / 0.5)
    right_edge = np.exp(-abs(1.3 - f.b) / 0.5)
    np.testing.assert_allclose(Kuf[:4, 0], left_edge, atol=1e-12)
    np.testing.assert_allclose(Kuf[:4, 1], interior, atol=1e-12)
    np.testing.assert_allclose(Kuf[:4, 2], right_edge, atol=1e-12)
    np.testing.assert_allclose(Kuf[4:, [0, 2]], 0.0, atol=1e-12)


def test_Kuf_matern32_reference_edges():
    f = FourierFeatures1D(a=0.0, b=1.0, num_frequencies=3, allow_extrapolation=True)
    k = Matern32(input_dim=1, ls=0.5)
    X_np = np.array([[-0.2], [0.25], [1.3]])
    Kuf = f.K_uf(k, pt.as_tensor(X_np)).eval()

    omegas = 2.0 * np.pi * np.arange(4) / (f.b - f.a)
    omegas_sin = omegas[1:]
    interior_cos = np.cos(omegas * (0.25 - f.a))
    interior_sin = np.sin(omegas_sin * (0.25 - f.a))
    arg_left = np.sqrt(3.0) * abs(-0.2 - f.a) / 0.5
    arg_right = np.sqrt(3.0) * abs(1.3 - f.b) / 0.5
    left_cos = (1.0 + arg_left) * np.exp(-arg_left)
    right_cos = (1.0 + arg_right) * np.exp(-arg_right)
    left_sin = (-0.2 - f.a) * np.exp(-arg_left) * omegas_sin
    right_sin = (1.3 - f.b) * np.exp(-arg_right) * omegas_sin

    np.testing.assert_allclose(Kuf[:4, 0], left_cos, atol=1e-12)
    np.testing.assert_allclose(Kuf[:4, 1], interior_cos, atol=1e-12)
    np.testing.assert_allclose(Kuf[:4, 2], right_cos, atol=1e-12)
    np.testing.assert_allclose(Kuf[4:, 0], left_sin, atol=1e-12)
    np.testing.assert_allclose(Kuf[4:, 1], interior_sin, atol=1e-12)
    np.testing.assert_allclose(Kuf[4:, 2], right_sin, atol=1e-12)


def test_Kuf_rejects_multi_active_dim():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    k = Matern32(input_dim=2, active_dims=[0, 1], ls=0.5)
    X = pt.as_tensor(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="active_dims"):
        f.K_uf(k, X)


def test_Kuu_solve_matches_dense():
    f = FourierFeatures1D(0, 1, num_frequencies=8)
    k = 1.5 * Matern32(input_dim=1, ls=0.4)
    rhs_np = np.random.default_rng(0).standard_normal((17, 3))
    out = f.Kuu_solve(k, pt.as_tensor(rhs_np)).eval()
    K = f.K_uu(k).eval()
    np.testing.assert_allclose(out, np.linalg.solve(K, rhs_np), atol=1e-6)


def test_Kuu_logdet_matches_dense():
    f = FourierFeatures1D(0, 1, num_frequencies=8)
    k = 1.5 * Matern52(input_dim=1, ls=0.4)
    K = f.K_uu(k).eval()
    np.testing.assert_allclose(f.Kuu_logdet(k).eval(), np.linalg.slogdet(K)[1], atol=1e-6)


def test_Kuu_sqrt_solve_R_inv_identity():
    """R_inv @ Kuu @ R_inv.T = I."""
    f = FourierFeatures1D(0, 1, num_frequencies=8)
    k = 1.0 * Matern32(input_dim=1, ls=0.5)
    M = f.num_inducing
    R_inv = f.Kuu_sqrt_solve(k, pt.as_tensor(np.eye(M))).eval()
    K = f.K_uu(k).eval()
    np.testing.assert_allclose(R_inv @ K @ R_inv.T, np.eye(M), atol=1e-6)


def test_Kuu_sqrt_solve_quadratic_form():
    """rhs.T @ R_inv.T @ R_inv @ rhs = rhs.T @ Kuu^{-1} @ rhs."""
    f = FourierFeatures1D(0, 1, num_frequencies=8)
    k = 1.0 * Matern52(input_dim=1, ls=0.3)
    rhs = np.random.default_rng(0).standard_normal((f.num_inducing, 4))
    R_inv_rhs = f.Kuu_sqrt_solve(k, pt.as_tensor(rhs)).eval()
    K = f.K_uu(k).eval()
    np.testing.assert_allclose(R_inv_rhs.T @ R_inv_rhs, rhs.T @ np.linalg.solve(K, rhs), atol=1e-6)


def test_domain_check_uses_active_column():
    f = FourierFeatures1D(0, 1, num_frequencies=4)
    k = Matern32(input_dim=2, active_dims=[1], ls=0.5)
    X_ok = np.column_stack([np.array([5.0, 10.0]), np.array([0.2, 0.8])])
    f._domain_check(X_ok, k)
    X_bad = np.column_stack([np.array([0.1, 0.2]), np.array([0.5, 1.5])])
    with pytest.raises(ValueError, match=r"column 1|X\[:, 1\]"):
        f._domain_check(X_bad, k)


def test_wrap_helper_validates_at_input_index():
    from ptgp.inducing_fourier import _maybe_wrap_with_domain_check

    f = FourierFeatures1D(0, 1, num_frequencies=4)
    k = Matern32(input_dim=1, ls=0.5)

    class _Model:
        pass

    m = _Model()
    m.inducing_variable = f
    m.kernel = k

    def fn(theta, X, y):
        return float(theta[0])

    wrapped = _maybe_wrap_with_domain_check(fn, m, input_index=1)

    assert wrapped(np.zeros(3), np.array([[0.5]]), np.zeros(1)) == 0.0
    with pytest.raises(ValueError):
        wrapped(np.zeros(3), np.array([[5.0]]), np.zeros(1))
    assert wrapped(np.array([42.0, 99.0]), np.array([[0.5]]), np.zeros(1)) == 42.0


def test_wrap_helper_noop_for_non_vff():
    from ptgp.inducing_fourier import _maybe_wrap_with_domain_check

    class _Model:
        pass

    m = _Model()
    m.inducing_variable = None
    m.kernel = None

    def fn(x):
        return x

    assert _maybe_wrap_with_domain_check(fn, m, input_index=0) is fn


def test_domain_check_opt_out():
    f = FourierFeatures1D(0, 1, num_frequencies=4, allow_extrapolation=True)
    k = Matern32(input_dim=1, ls=0.5)
    f._domain_check(np.array([[10.0]]), k)


def test_domain_check_rejects_matern52_extrapolation_even_when_opted_out():
    f = FourierFeatures1D(0, 1, num_frequencies=4, allow_extrapolation=True)
    k = Matern52(input_dim=1, ls=0.5)
    with pytest.raises(ValueError, match=r"Matern52.*outside"):
        f._domain_check(np.array([[10.0]]), k)


def test_Kuu_sqrt_solve_finite_at_small_lambda():
    """Long ls vs domain forces small Gram eigenvalues; delta must stay finite."""
    f = FourierFeatures1D(0, 1, num_frequencies=15)
    k = 1.0 * Matern32(input_dim=1, ls=10.0)
    M = f.num_inducing
    R_inv = f.Kuu_sqrt_solve(k, pt.as_tensor(np.eye(M))).eval()
    assert np.all(np.isfinite(R_inv))
    K = f.K_uu(k).eval()
    np.testing.assert_allclose(R_inv @ K @ R_inv.T, np.eye(M), atol=1e-5)


def test_gauss_kl_structured_scalar_for_vff():
    f = FourierFeatures1D(0, 1, num_frequencies=8)
    k = 1.0 * Matern32(input_dim=1, ls=0.4)
    svgp = SVGP(
        kernel=k,
        likelihood=Gaussian(sigma=0.1),
        inducing_variable=f,
        whiten=False,
    )
    out = svgp.prior_kl().eval()
    assert out.ndim == 0 and np.isfinite(out)


def test_eta_squared_matern32_predictions_finite():
    f = FourierFeatures1D(0, 1, num_frequencies=8)
    base = Matern32(input_dim=1, ls=0.4)
    eta = pt.as_tensor(2.0)

    rng = np.random.default_rng(0)
    M = f.num_inducing
    q_mu = rng.standard_normal(M)
    q_sqrt = np.eye(M)
    svgp_scaled = SVGP(
        kernel=eta**2 * base,
        likelihood=Gaussian(sigma=0.1),
        inducing_variable=f,
        whiten=True,
        q_mu=pt.as_tensor(q_mu),
        q_sqrt=pt.as_tensor(q_sqrt),
    )
    X = pt.as_tensor(np.linspace(0.1, 0.9, 20)[:, None])
    m_s, v_s = [t.eval() for t in svgp_scaled.predict_marginal(X)]
    assert np.all(np.isfinite(m_s)) and np.all(v_s >= 0)


def test_rank_deficient_matern52_raises():
    f = FourierFeatures1D(0, 1, num_frequencies=2)  # M=5, R=6 for Matern52 → fail
    with pytest.raises(ValueError, match="num_frequencies >= 3"):
        f._structured_Kuu(Matern52(input_dim=1, ls=0.5))


def test_vff_converges_to_exact_gp_sanity():
    rng = np.random.default_rng(0)
    N = 200
    X = np.sort(rng.uniform(0, 1, N))[:, None]
    _ = np.sin(2 * np.pi * X[:, 0]) + 0.05 * rng.standard_normal(N)

    f = FourierFeatures1D.from_data(X, num_frequencies=64, buffer=0.2)
    k = 1.0 * Matern32(input_dim=1, ls=0.2)
    svgp = SVGP(
        kernel=k,
        likelihood=Gaussian(sigma=0.05),
        inducing_variable=f,
        whiten=True,
    )
    fmean, fvar = [t.eval() for t in svgp.predict_marginal(pt.as_tensor(X))]
    assert np.all(np.isfinite(fmean))
    assert np.all(fvar >= 0)
