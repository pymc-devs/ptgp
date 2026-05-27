"""Non-stationary kernel tests."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp.kernels import ExpQuad, Gibbs, Linear, Matern52, RandomWalk, WarpedInput


def _ptgp_eval(kernel, X_np, Y_np=None):
    """Evaluate a PTGP kernel symbolically and compile to a numeric result."""
    X_pt = pt.as_tensor_variable(X_np)
    Y_pt = pt.as_tensor_variable(Y_np) if Y_np is not None else None
    K_sym = kernel(X_pt, Y_pt)
    f = pytensor.function([], K_sym)
    return f()


class TestLinear:
    def test_gram_values(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(_ptgp_eval(Linear(input_dim=2), X), X @ X.T, atol=1e-14)

    def test_cross_shape_and_values(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y = np.array([[0.5, 1.0], [2.0, -1.0], [4.0, 0.0]])
        K = _ptgp_eval(Linear(input_dim=2), X, Y)
        np.testing.assert_allclose(K, X @ Y.T, atol=1e-14)

    def test_centering(self):
        """k(x, y) = (x - c)(y - c) with non-zero c."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = 2.5
        np.testing.assert_allclose(
            _ptgp_eval(Linear(input_dim=2, c=c), X), (X - c) @ (X - c).T, atol=1e-14
        )

    def test_vs_pymc(self):
        """Centered Linear matches PyMC's pm.gp.cov.Linear."""
        import pymc as pm

        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 3))
        c = 0.7
        K_ptgp = _ptgp_eval(Linear(input_dim=3, c=c), X)
        K_pymc = pm.gp.cov.Linear(input_dim=3, c=c)(pt.as_tensor_variable(X)).eval()
        np.testing.assert_allclose(K_ptgp, K_pymc, atol=1e-12)

    def test_active_dims(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        K = _ptgp_eval(Linear(input_dim=2, active_dims=[0]), X)
        np.testing.assert_allclose(K, X[:, :1] @ X[:, :1].T, atol=1e-14)

    def test_scaling(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(
            _ptgp_eval(3.0 * Linear(input_dim=2), X), 3.0 * (X @ X.T), atol=1e-14
        )

    def test_diag_matches_gram_diag(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((8, 2))
        k = Linear(input_dim=2, c=0.3)
        X_pt = pt.as_tensor_variable(X)
        np.testing.assert_allclose(k.diag(X_pt).eval(), np.diag(k(X_pt).eval()), atol=1e-12)


class TestRandomWalk:
    def test_gram_values(self):
        X = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(_ptgp_eval(RandomWalk(), X), np.minimum(X, X.T), atol=1e-14)

    def test_scaling(self):
        X = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(
            _ptgp_eval(4.0 * RandomWalk(), X), 4.0 * np.minimum(X, X.T), atol=1e-14
        )

    def test_cross(self):
        X = np.array([[1.0], [3.0]])
        Y = np.array([[2.0], [4.0]])
        K = _ptgp_eval(RandomWalk(), X, Y)
        np.testing.assert_allclose(K, np.minimum(X, Y.T), atol=1e-14)

    def test_positive_definite(self):
        X = np.linspace(0.1, 5.0, 20)[:, None]
        eigvals = np.linalg.eigvalsh(_ptgp_eval(RandomWalk(), X))
        assert np.all(eigvals > -1e-10)


class TestGibbs:
    def test_constant_ls_matches_expquad(self):
        """Gibbs with constant l(x) = l0 reduces to ExpQuad(ls=l0)."""
        X = np.linspace(0.0, 5.0, 15)[:, None]
        ls0 = 1.3
        gibbs = Gibbs(lengthscale_func=lambda X: pt.fill(X[:, 0], ls0))
        expquad = ExpQuad(input_dim=1, ls=ls0)
        np.testing.assert_allclose(_ptgp_eval(gibbs, X), _ptgp_eval(expquad, X), atol=1e-14)

    def test_vs_pymc(self):
        """Gibbs with varying l(x) matches PyMC's pm.gp.cov.Gibbs."""
        import pymc as pm

        X = np.linspace(0.0, 5.0, 15)[:, None]

        def ls_func(X):
            return 0.5 + 0.3 * X[:, 0]

        gibbs = Gibbs(lengthscale_func=ls_func)
        pymc_gibbs = pm.gp.cov.Gibbs(input_dim=1, lengthscale_func=ls_func)

        K_ptgp = _ptgp_eval(gibbs, X)
        K_pymc = pymc_gibbs(pt.as_tensor_variable(X)).eval()
        np.testing.assert_allclose(K_ptgp, K_pymc, atol=1e-10)

    def test_rejects_multidim_active_dims(self):
        with pytest.raises(ValueError, match="length 1"):
            Gibbs(lengthscale_func=lambda X: X[:, 0], active_dims=[0, 0])


class TestWarpedInput:
    def test_identity_warp_matches_inner(self):
        """Warping with the identity should give the same kernel as the inner kernel."""
        X = np.linspace(0.0, 5.0, 15)[:, None]
        inner = Matern52(input_dim=1, ls=0.8)
        warped = WarpedInput(input_dim=1, kernel_func=inner, warp_func=lambda X: X)
        np.testing.assert_allclose(_ptgp_eval(warped, X), _ptgp_eval(inner, X), atol=1e-14)

    def test_vs_pymc(self):
        """WarpedInput with a nonlinear warp matches PyMC's pm.gp.cov.WarpedInput."""
        import pymc as pm

        X = np.linspace(0.1, 3.0, 15)[:, None]

        def warp(X):
            return X**2

        inner = ExpQuad(input_dim=1, ls=1.0)
        pymc_inner = pm.gp.cov.ExpQuad(input_dim=1, ls=1.0)

        warped = WarpedInput(input_dim=1, kernel_func=inner, warp_func=warp)
        pymc_warped = pm.gp.cov.WarpedInput(input_dim=1, cov_func=pymc_inner, warp_func=warp)

        K_ptgp = _ptgp_eval(warped, X)
        K_pymc = pymc_warped(pt.as_tensor_variable(X)).eval()
        np.testing.assert_allclose(K_ptgp, K_pymc, atol=1e-10)
