"""Independent whitened-SVGP ELBO reference (numpy + scipy, no PTGP internals).

Shared by the SVGP ELBO-match tests. The reference assembles the ELBO from
first principles (whitened predict by hand, scipy.integrate.quad for the
variational expectation, closed-form whitened KL) and only needs a numpy
``log_prob(f, y)`` for the likelihood, so it is library-agnostic.

The leading underscore keeps pytest from collecting this module as a test file.
"""

import numpy as np
import scipy.linalg

from scipy import integrate


def _matern52_numpy(X1, X2, ls, eta):
    """Matern52 kernel in numpy: k(r) = eta^2 (1 + sqrt(5)r + 5r^2/3) exp(-sqrt(5)r)."""
    sqd = np.sum(X1**2, axis=-1)[:, None] + np.sum(X2**2, axis=-1)[None, :] - 2.0 * X1 @ X2.T
    r = np.sqrt(np.maximum(sqd, 0.0)) / ls
    s5 = np.sqrt(5.0)
    return eta**2 * (1.0 + s5 * r + 5.0 * r**2 / 3.0) * np.exp(-s5 * r)


def whitened_predict(X, Z, ls, eta, q_mu, q_sqrt):
    """Whitened-SVGP marginal predictive mean and variance at X."""
    Kzz = _matern52_numpy(Z, Z, ls, eta)
    Kzx = _matern52_numpy(Z, X, ls, eta)
    K_diag = np.full(X.shape[0], eta**2)  # Matern52 diag is eta^2
    Lz = scipy.linalg.cholesky(Kzz, lower=True)
    A = scipy.linalg.solve_triangular(Lz, Kzx, lower=True)  # (M, N)
    mu_f = A.T @ q_mu
    var_f = K_diag - np.sum(A**2, axis=0) + np.sum((A.T @ q_sqrt) ** 2, axis=1)
    return mu_f, var_f


def whitened_kl(q_mu, q_sqrt):
    """Closed-form KL[N(q_mu, q_sqrt q_sqrt^T) || N(0, I)]."""
    M = q_mu.size
    logdet_S = 2.0 * np.sum(np.log(np.abs(np.diag(q_sqrt))))
    return 0.5 * (np.sum(q_sqrt**2) + q_mu @ q_mu - M - logdet_S)


def variational_expectation_quad(log_prob_fn, y, mu_f, var_f):
    """Sum of per-point E_{q(f_n)}[log p(y_n|f_n)] via adaptive quadrature.

    Integrates in the standard-normal z-scale: f = mu + sqrt(var) * z. The
    [-30, 30] z-range is well past the numerical support of exp(-z^2/2).
    """
    total = 0.0
    for yn, m, v in zip(y, mu_f, var_f):
        sd = np.sqrt(v)

        def integrand(z, yn=yn, m=m, sd=sd):
            return log_prob_fn(m + sd * z, yn) * np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)

        val, _ = integrate.quad(integrand, -30.0, 30.0)
        total += val
    return total


def reference_elbo(X, y, Z, q_mu, q_sqrt, ls, eta, log_prob_fn):
    """Whitened-SVGP ELBO assembled from numpy + scipy."""
    mu_f, var_f = whitened_predict(X, Z, ls, eta, q_mu, q_sqrt)
    ve = variational_expectation_quad(log_prob_fn, y, mu_f, var_f)
    kl = whitened_kl(q_mu, q_sqrt)
    return ve - kl


def fixed_config(rng, N=40, M=8, x_range=(-2.0, 2.0)):
    """Shared fixed configuration for ELBO-match tests."""
    X = np.sort(rng.uniform(x_range[0], x_range[1], N))[:, None]
    Z = np.linspace(x_range[0], x_range[1], M)[:, None]
    q_mu = rng.normal(0, 0.3, M)
    L = np.tril(rng.normal(0, 0.2, (M, M)))
    L[np.arange(M), np.arange(M)] = np.abs(L[np.arange(M), np.arange(M)]) + 0.5
    return X, Z, q_mu, L
