"""Stationary kernel tests against GPJax reference implementation."""

import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from gpjax.kernels.stationary import (
    RBF as GPJaxRBF,
)
from gpjax.kernels.stationary import (
    Matern32 as GPJaxMatern32,
)
from gpjax.kernels.stationary import (
    Matern52 as GPJaxMatern52,
)

from ptgp.kernels import ExpQuad, Matern12, Matern32, Matern52

# GPJax uses float32 internally, so comparisons are limited to ~1e-6 precision.
ATOL = 1e-5


def _ptgp_eval(kernel, X_np, Y_np=None):
    """Evaluate a PTGP kernel symbolically and compile to a numeric result."""
    X_pt = pt.as_tensor_variable(X_np)
    Y_pt = pt.as_tensor_variable(Y_np) if Y_np is not None else None
    K_sym = kernel(X_pt, Y_pt)
    f = pytensor.function([], K_sym)
    return f()


def _gpjax_gram(kernel, X_np):
    X_jnp = jnp.array(X_np, dtype=jnp.float32)
    return np.array(kernel.gram(X_jnp).to_dense())


def _gpjax_cross(kernel, X_np, Y_np):
    return np.array(
        kernel.cross_covariance(
            jnp.array(X_np, dtype=jnp.float32),
            jnp.array(Y_np, dtype=jnp.float32),
        )
    )


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
        gpjax_k = GPJaxRBF(lengthscale=jnp.array(ls), variance=jnp.array(eta**2))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d), _gpjax_gram(gpjax_k, X_1d), atol=ATOL)

    def test_cross_1d(self, X_1d, X_1d_other):
        ls, eta = 1.5, 2.0
        ptgp_k = eta**2 * ExpQuad(input_dim=1, ls=ls)
        gpjax_k = GPJaxRBF(lengthscale=jnp.array(ls), variance=jnp.array(eta**2))
        np.testing.assert_allclose(
            _ptgp_eval(ptgp_k, X_1d, X_1d_other), _gpjax_cross(gpjax_k, X_1d, X_1d_other), atol=ATOL
        )

    def test_gram_2d(self, X_2d):
        ptgp_k = ExpQuad(input_dim=2, ls=0.8)
        gpjax_k = GPJaxRBF(lengthscale=jnp.array(0.8), variance=jnp.array(1.0))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_2d), _gpjax_gram(gpjax_k, X_2d), atol=ATOL)

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
        gpjax_k = GPJaxMatern52(lengthscale=jnp.array(ls), variance=jnp.array(eta**2))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d), _gpjax_gram(gpjax_k, X_1d), atol=ATOL)

    def test_cross_1d(self, X_1d, X_1d_other):
        ls, eta = 1.2, 1.5
        ptgp_k = eta**2 * Matern52(input_dim=1, ls=ls)
        gpjax_k = GPJaxMatern52(lengthscale=jnp.array(ls), variance=jnp.array(eta**2))
        np.testing.assert_allclose(
            _ptgp_eval(ptgp_k, X_1d, X_1d_other), _gpjax_cross(gpjax_k, X_1d, X_1d_other), atol=ATOL
        )

    def test_gram_2d(self, X_2d):
        ls, eta = 0.5, 2.0
        ptgp_k = eta**2 * Matern52(input_dim=2, ls=ls)
        gpjax_k = GPJaxMatern52(lengthscale=jnp.array(ls), variance=jnp.array(eta**2))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_2d), _gpjax_gram(gpjax_k, X_2d), atol=ATOL)


class TestMatern32:
    def test_gram_1d(self, X_1d):
        ptgp_k = Matern32(input_dim=1, ls=2.0)
        gpjax_k = GPJaxMatern32(lengthscale=jnp.array(2.0), variance=jnp.array(1.0))
        np.testing.assert_allclose(_ptgp_eval(ptgp_k, X_1d), _gpjax_gram(gpjax_k, X_1d), atol=ATOL)

    def test_cross_2d(self, X_2d, X_2d_other):
        ls, eta = 0.7, 1.3
        ptgp_k = eta**2 * Matern32(input_dim=2, ls=ls)
        gpjax_k = GPJaxMatern32(lengthscale=jnp.array(ls), variance=jnp.array(eta**2))
        np.testing.assert_allclose(
            _ptgp_eval(ptgp_k, X_2d, X_2d_other), _gpjax_cross(gpjax_k, X_2d, X_2d_other), atol=ATOL
        )


class TestMatern12:
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

    def test_ard_vs_pymc(self, X_2d):
        """ARD Matern52 with per-dim lengthscales matches PyMC's implementation."""
        import pymc as pm

        ls = np.array([0.5, 1.2])
        ptgp_k = Matern52(input_dim=2, ls=ls)
        pymc_k = pm.gp.cov.Matern52(input_dim=2, ls=ls)

        K_ptgp = _ptgp_eval(ptgp_k, X_2d)
        K_pymc = pymc_k(pt.as_tensor_variable(X_2d)).eval()
        # PyMC and PTGP use different NaN-safe sqrt strategies near zero, so
        # allow small numerical differences.
        np.testing.assert_allclose(K_ptgp, K_pymc, atol=1e-8)
