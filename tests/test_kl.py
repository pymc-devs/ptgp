"""Tests for gauss_kl in ptgp.kl."""

import numpy as np
import pytensor
import pytensor.tensor as pt

from ptgp.kl import gauss_kl, gauss_kl_structured


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


class TestGaussKL:
    def test_whitened_zero_mean_identity_cov(self):
        """KL[N(0, I) || N(0, I)] = 0."""
        M = 3
        kl = _eval(gauss_kl(pt.zeros(M), pt.eye(M), K=None))
        np.testing.assert_allclose(kl, 0.0, atol=1e-10)

    def test_whitened_nonzero_mean(self):
        """KL should be positive for non-trivial q."""
        q_mu = pt.as_tensor_variable(np.array([1.0, 0.5, -0.3]))
        q_sqrt = pt.as_tensor_variable(np.eye(3) * 0.5)
        kl = _eval(gauss_kl(q_mu, q_sqrt, K=None))
        assert kl > 0.0

    def test_unwhitened_matches_whitened_with_identity_prior(self):
        """With K=I, unwhitened should match whitened."""
        q_mu = pt.as_tensor_variable(np.array([0.5, -0.5]))
        q_sqrt = pt.as_tensor_variable(np.array([[0.8, 0.0], [0.2, 0.6]]))
        kl_w = _eval(gauss_kl(q_mu, q_sqrt, K=None))
        kl_u = _eval(gauss_kl(q_mu, q_sqrt, K=pt.eye(2)))
        np.testing.assert_allclose(kl_w, kl_u, atol=1e-10)

    def test_unwhitened_positive(self):
        M = 3
        rng = np.random.default_rng(0)
        L = np.tril(rng.standard_normal((M, M)))
        K = L @ L.T + 0.1 * np.eye(M)
        q_mu = rng.standard_normal(M)
        q_sqrt = np.eye(M) * 0.5

        kl = _eval(
            gauss_kl(
                pt.as_tensor_variable(q_mu),
                pt.as_tensor_variable(q_sqrt),
                K=pt.as_tensor_variable(K),
            )
        )
        assert kl > 0.0


def test_gauss_kl_structured_matches_dense():
    rng = np.random.default_rng(0)
    M = 8
    A = rng.standard_normal((M, M))
    K_dense = A @ A.T + np.eye(M)
    q_mu = rng.standard_normal(M)
    q_sqrt = np.linalg.cholesky(np.eye(M) * 0.5)

    K_t = pt.as_tensor(K_dense)
    expected = gauss_kl(pt.as_tensor(q_mu), pt.as_tensor(q_sqrt), K=K_t).eval()

    def K_solve(rhs):
        return pt.linalg.solve(K_t, rhs)

    _, K_logdet = pt.linalg.slogdet(K_t)
    actual = gauss_kl_structured(pt.as_tensor(q_mu), pt.as_tensor(q_sqrt), K_solve, K_logdet).eval()

    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_gauss_kl_structured_returns_scalar():
    M = 5
    q_mu = pt.as_tensor(np.zeros(M))
    q_sqrt = pt.as_tensor(np.eye(M))
    K = pt.as_tensor(np.eye(M))
    out = gauss_kl_structured(
        q_mu, q_sqrt, lambda x: pt.linalg.solve(K, x), pt.linalg.slogdet(K)[1]
    ).eval()
    assert out.ndim == 0
    assert np.isfinite(out)
