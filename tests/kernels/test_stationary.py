"""Stationary kernel tests against closed-form analytic references.

Each kernel is checked against its textbook formula evaluated in NumPy, so the
suite pins down that the kernels are mathematically correct without depending on
another GP library.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp.kernels import ExpQuad, Matern12, Matern32, Matern52

# ExpQuad and Matern12 evaluate exactly in float64, so the comparison is tight.
ATOL = 1e-10
# Matern52/32 carry a sqrt(5)/sqrt(3) constant that PTGP evaluates via pt.sqrt,
# which pytensor computes in floatX (float32 by default). That injects ~3e-8 of
# error relative to the exact formula, so the Matern comparisons use a looser
# tolerance — still far tighter than any real formula error would survive.
MATERN_ATOL = 1e-6


def _ptgp_eval(kernel, X_np, Y_np=None):
    """Evaluate a PTGP kernel symbolically and compile to a numeric result."""
    X_pt = pt.as_tensor_variable(X_np)
    Y_pt = pt.as_tensor_variable(Y_np) if Y_np is not None else None
    K_sym = kernel(X_pt, Y_pt)
    f = pytensor.function([], K_sym)
    return f()


def _scaled_dist(X, Y, ls):
    """Euclidean distance between rows of X and Y after dividing by lengthscale.

    ``ls`` may be a scalar (isotropic) or a per-dimension vector (ARD).
    """
    ls = np.asarray(ls, dtype=float)
    Xs, Ys = X / ls, Y / ls
    sqd = np.sum(Xs**2, axis=1)[:, None] + np.sum(Ys**2, axis=1)[None, :] - 2.0 * Xs @ Ys.T
    return np.sqrt(np.maximum(sqd, 0.0))


def _analytic(X, Y, ls, eta, kind):
    """Closed-form covariance matrix for a stationary kernel, in NumPy."""
    r = _scaled_dist(X, Y, ls)
    if kind == "expquad":
        k = np.exp(-0.5 * r**2)
    elif kind == "matern52":
        k = (1.0 + np.sqrt(5.0) * r + 5.0 * r**2 / 3.0) * np.exp(-np.sqrt(5.0) * r)
    elif kind == "matern32":
        k = (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)
    elif kind == "matern12":
        k = np.exp(-r)
    else:  # pragma: no cover - guards against typos in test parametrization
        raise ValueError(f"unknown kernel kind {kind!r}")
    return eta**2 * k


@pytest.fixture
def X_1d():
    return np.linspace(0.0, 5.0, 20)[:, None].astype(np.float64)


@pytest.fixture
def X_1d_other():
    return np.linspace(1.0, 3.0, 10)[:, None].astype(np.float64)


@pytest.fixture
def X_2d():
    rng = np.random.default_rng(42)
    return rng.standard_normal((15, 2)).astype(np.float64)


@pytest.fixture
def X_2d_other():
    rng = np.random.default_rng(99)
    return rng.standard_normal((8, 2)).astype(np.float64)


class TestExpQuad:
    def test_gram_1d(self, X_1d):
        ls, eta = 1.5, 2.0
        ptgp_k = eta**2 * ExpQuad(input_dim=1, ls=ls)
        ref = _analytic(X_1d, X_1d, ls, eta, "expquad")
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d), ref, atol=ATOL)

    def test_cross_1d(self, X_1d, X_1d_other):
        ls, eta = 1.5, 2.0
        ptgp_k = eta**2 * ExpQuad(input_dim=1, ls=ls)
        ref = _analytic(X_1d, X_1d_other, ls, eta, "expquad")
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d, X_1d_other), ref, atol=ATOL)

    def test_gram_2d(self, X_2d):
        ls, eta = 0.8, 1.0
        ptgp_k = ExpQuad(input_dim=2, ls=ls)
        ref = _analytic(X_2d, X_2d, ls, eta, "expquad")
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_2d), ref, atol=ATOL)

    def test_symmetric_annotation(self, X_1d):
        X_pt = pt.as_tensor_variable(X_1d)
        K = ExpQuad(input_dim=1, ls=1.0)(X_pt)
        from pytensor.assumptions.core import FactState
        from pytensor.assumptions.specify import SpecifyAssumptions

        assert isinstance(K.owner.op, SpecifyAssumptions)
        assert ("symmetric", FactState.TRUE) in K.owner.op.assumptions
        assert ("positive_definite", FactState.TRUE) in K.owner.op.assumptions

    def test_cross_no_annotation(self, X_1d, X_1d_other):
        X_pt = pt.as_tensor_variable(X_1d)
        Y_pt = pt.as_tensor_variable(X_1d_other)
        K = ExpQuad(input_dim=1, ls=1.0)(X_pt, Y_pt)
        from pytensor.assumptions.specify import SpecifyAssumptions

        assert not isinstance(K.owner.op, SpecifyAssumptions)


class TestMatern52:
    def test_gram_1d(self, X_1d):
        ls, eta = 1.2, 1.5
        ptgp_k = eta**2 * Matern52(input_dim=1, ls=ls)
        ref = _analytic(X_1d, X_1d, ls, eta, "matern52")
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d), ref, atol=MATERN_ATOL)

    def test_cross_1d(self, X_1d, X_1d_other):
        ls, eta = 1.2, 1.5
        ptgp_k = eta**2 * Matern52(input_dim=1, ls=ls)
        ref = _analytic(X_1d, X_1d_other, ls, eta, "matern52")
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d, X_1d_other), ref, atol=MATERN_ATOL)

    def test_gram_2d(self, X_2d):
        ls, eta = 0.5, 2.0
        ptgp_k = eta**2 * Matern52(input_dim=2, ls=ls)
        ref = _analytic(X_2d, X_2d, ls, eta, "matern52")
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_2d), ref, atol=MATERN_ATOL)


class TestMatern32:
    def test_gram_1d(self, X_1d):
        ls, eta = 2.0, 1.0
        ptgp_k = Matern32(input_dim=1, ls=ls)
        ref = _analytic(X_1d, X_1d, ls, eta, "matern32")
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d), ref, atol=MATERN_ATOL)

    def test_cross_2d(self, X_2d, X_2d_other):
        ls, eta = 0.7, 1.3
        ptgp_k = eta**2 * Matern32(input_dim=2, ls=ls)
        ref = _analytic(X_2d, X_2d_other, ls, eta, "matern32")
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_2d, X_2d_other), ref, atol=MATERN_ATOL)


class TestMatern12:
    def test_gram_1d(self, X_1d):
        ls, eta = 1.0, 1.0
        ref = _analytic(X_1d, X_1d, ls, eta, "matern12")
        np.testing.assert_allclose(_ptgp_eval(Matern12(input_dim=1, ls=ls), X_1d), ref, atol=ATOL)

    def test_gram_symmetry(self, X_1d):
        K = _ptgp_eval(Matern12(input_dim=1, ls=1.0), X_1d)
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_diagonal_is_one(self, X_1d):
        K = _ptgp_eval(Matern12(input_dim=1, ls=1.0), X_1d)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-14)

    def test_positive_definite(self, X_1d):
        eigvals = np.linalg.eigvalsh(_ptgp_eval(Matern12(input_dim=1, ls=1.0), X_1d))
        assert np.all(eigvals > -1e-10)

    def test_cross_shape(self, X_1d, X_1d_other):
        assert _ptgp_eval(Matern12(input_dim=1, ls=1.0), X_1d, X_1d_other).shape == (20, 10)


class TestActiveDims:
    def test_active_dims_selects_columns(self, X_2d):
        k_2d = ExpQuad(input_dim=2, ls=1.0, active_dims=[0])
        k_1d = ExpQuad(input_dim=1, ls=1.0)
        np.testing.assert_allclose(
            _ptgp_eval(k_2d, X_2d), _ptgp_eval(k_1d, X_2d[:, :1]), atol=1e-14
        )

    def test_active_dims_out_of_range(self):
        with pytest.raises(ValueError, match="active_dims"):
            ExpQuad(input_dim=2, ls=1.0, active_dims=[0, 5])


class TestARD:
    def test_scalar_and_vector_ls_match_when_equal(self, X_2d):
        """Scalar ls=0.5 and vector ls=[0.5, 0.5] should produce identical kernels."""
        k_iso = Matern52(input_dim=2, ls=0.5)
        k_ard = Matern52(input_dim=2, ls=np.array([0.5, 0.5]))
        np.testing.assert_allclose(_ptgp_eval(k_iso, X_2d), _ptgp_eval(k_ard, X_2d), atol=1e-14)

    def test_ard_matches_analytic(self, X_2d):
        """ARD Matern52 with per-dim lengthscales matches the closed-form reference."""
        ls = np.array([0.5, 1.2])
        ptgp_k = Matern52(input_dim=2, ls=ls)
        ref = _analytic(X_2d, X_2d, ls, 1.0, "matern52")
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_2d), ref, atol=MATERN_ATOL)
