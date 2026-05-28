"""Categorical kernel tests."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp.kernels import ExpQuad, LowRankCategorical, Overlap


def _ptgp_eval(kernel, X_np, Y_np=None):
    """Evaluate a PTGP kernel symbolically and compile to a numeric result."""
    X_pt = pt.as_tensor_variable(X_np)
    Y_pt = pt.as_tensor_variable(Y_np) if Y_np is not None else None
    K_sym = kernel(X_pt, Y_pt)
    f = pytensor.function([], K_sym)
    return f()


def _ptgp_diag(kernel, X_np):
    """Compile a kernel's diag(X) to a numeric result."""
    X_pt = pt.as_tensor_variable(X_np)
    f = pytensor.function([], kernel.diag(X_pt))
    return f()


class TestOverlap:
    def test_single_column(self):
        """k(x, y) = 1[x == y] with one active categorical column."""
        X = np.array([[0.0], [1.0], [0.0], [2.0]])
        expected = (X == X.T).astype(float)
        np.testing.assert_allclose(_ptgp_eval(Overlap(input_dim=1), X), expected, atol=1e-14)

    def test_multi_column_mean(self):
        """With multiple active columns, kernel is the mean equality."""
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        eq = (X[:, None, :] == X[None, :, :]).astype(float)
        expected = eq.mean(axis=-1)
        np.testing.assert_allclose(_ptgp_eval(Overlap(input_dim=2), X), expected, atol=1e-14)

    def test_active_dims_selects_column(self):
        """Only the active columns contribute; other columns are ignored."""
        X = np.array([[0.0, 99.0], [1.0, 99.0], [0.0, 42.0]])
        k = Overlap(input_dim=2, active_dims=[0])
        expected = (X[:, :1] == X[:, :1].T).astype(float)
        np.testing.assert_allclose(_ptgp_eval(k, X), expected, atol=1e-14)

    def test_scaling(self):
        X = np.array([[0.0], [1.0], [1.0]])
        expected = 4.0 * (X == X.T).astype(float)
        np.testing.assert_allclose(_ptgp_eval(4.0 * Overlap(input_dim=1), X), expected, atol=1e-14)

    def test_cross(self):
        X = np.array([[0.0], [1.0]])
        Y = np.array([[1.0], [1.0], [2.0]])
        expected = (X == Y.T).astype(float)
        np.testing.assert_allclose(_ptgp_eval(Overlap(input_dim=1), X, Y), expected, atol=1e-14)

    def test_diag(self):
        X = np.array([[0.0], [1.0], [2.0]])
        np.testing.assert_allclose(_ptgp_diag(Overlap(input_dim=1), X), np.ones(3), atol=1e-14)

    def test_positive_definite(self):
        rng = np.random.default_rng(0)
        X = rng.integers(0, 5, size=(20, 2)).astype(float)
        eigvals = np.linalg.eigvalsh(_ptgp_eval(Overlap(input_dim=2), X))
        assert np.all(eigvals > -1e-10)


class TestLowRankCategorical:
    def _make(self, L, R, rng, input_dim=1, active_dims=None):
        W = rng.normal(size=(L, R))
        kappa = rng.uniform(0.1, 1.0, size=L)
        k = LowRankCategorical(
            input_dim=input_dim,
            num_levels=L,
            W=pt.as_tensor_variable(W),
            kappa=pt.as_tensor_variable(kappa),
            active_dims=active_dims,
        )
        return k, W, kappa

    def test_gram_values(self):
        rng = np.random.default_rng(0)
        L, R = 4, 2
        k, W, kappa = self._make(L, R, rng)
        X = np.array([[0.0], [1.0], [2.0], [3.0], [1.0]])
        B = W @ W.T + np.diag(kappa)
        xi = X[:, 0].astype(int)
        expected = B[np.ix_(xi, xi)]
        np.testing.assert_allclose(_ptgp_eval(k, X), expected, atol=1e-14)

    def test_diag(self):
        rng = np.random.default_rng(1)
        L, R = 5, 3
        k, W, kappa = self._make(L, R, rng)
        X = np.array([[0.0], [2.0], [4.0], [1.0]])
        xi = X[:, 0].astype(int)
        expected = (W**2).sum(axis=-1)[xi] + kappa[xi]
        np.testing.assert_allclose(_ptgp_diag(k, X), expected, atol=1e-14)

    def test_cross(self):
        rng = np.random.default_rng(2)
        L, R = 3, 2
        k, W, kappa = self._make(L, R, rng)
        X = np.array([[0.0], [2.0]])
        Y = np.array([[1.0], [0.0], [2.0]])
        B = W @ W.T + np.diag(kappa)
        expected = B[np.ix_(X[:, 0].astype(int), Y[:, 0].astype(int))]
        np.testing.assert_allclose(_ptgp_eval(k, X, Y), expected, atol=1e-14)

    def test_positive_definite(self):
        rng = np.random.default_rng(3)
        L, R = 5, 2
        k, _, _ = self._make(L, R, rng)
        X = rng.integers(0, L, size=(20, 1)).astype(float)
        eigvals = np.linalg.eigvalsh(_ptgp_eval(k, X))
        assert np.all(eigvals > -1e-10)

    def test_product_with_continuous_kernel(self):
        """LowRankCategorical composes with ExpQuad via ProductKernel."""
        rng = np.random.default_rng(4)
        L, R = 3, 2
        W = rng.normal(size=(L, R))
        kappa = rng.uniform(0.1, 1.0, size=L)

        k_cat = LowRankCategorical(
            input_dim=2,
            num_levels=L,
            W=pt.as_tensor_variable(W),
            kappa=pt.as_tensor_variable(kappa),
            active_dims=[1],
        )
        k_cont = ExpQuad(input_dim=2, ls=0.8, active_dims=[0])
        k = k_cont * k_cat

        X = np.array([[0.1, 0.0], [0.3, 1.0], [0.5, 2.0], [0.7, 1.0]])
        K = _ptgp_eval(k, X)
        B = W @ W.T + np.diag(kappa)
        diff = X[:, :1] - X[:, :1].T
        K_cont = np.exp(-0.5 * (diff / 0.8) ** 2)
        xi = X[:, 1].astype(int)
        K_cat = B[np.ix_(xi, xi)]
        np.testing.assert_allclose(K, K_cont * K_cat, atol=1e-14)

    def test_rank1_matches_onehot_expquad_ard(self):
        """Rank-1 LowRankCategorical matches one-hot + ExpQuad with ARD ls.

        With w_l = exp(-1 / (2 * ls_l**2)) and kappa_l = 1 - w_l**2, the
        kernel B[i, j] = w_i * w_j + kappa_i * delta_ij equals ExpQuad on
        one-hot encoded inputs with ARD lengthscales ls.
        """
        rng = np.random.default_rng(5)
        L = 4
        ls = rng.uniform(0.5, 2.0, size=L)
        w = np.exp(-1.0 / (2.0 * ls**2))
        kappa = 1.0 - w**2

        k_cat = LowRankCategorical(
            input_dim=1,
            num_levels=L,
            W=pt.as_tensor_variable(w[:, None]),
            kappa=pt.as_tensor_variable(kappa),
        )
        X_codes = np.array([[0.0], [1.0], [2.0], [3.0], [1.0]])
        K_cat = _ptgp_eval(k_cat, X_codes)

        X_onehot = np.eye(L)[X_codes[:, 0].astype(int)]
        k_cont = ExpQuad(input_dim=L, ls=pt.as_tensor_variable(ls))
        K_cont = _ptgp_eval(k_cont, X_onehot)

        np.testing.assert_allclose(K_cat, K_cont, atol=1e-14)

    def test_rejects_multidim_active_dims(self):
        W = pt.as_tensor_variable(np.zeros((3, 2)))
        kappa = pt.as_tensor_variable(np.ones(3))
        with pytest.raises(ValueError, match="length 1"):
            LowRankCategorical(input_dim=2, num_levels=3, W=W, kappa=kappa, active_dims=[0, 1])
