"""Bernoulli likelihood tests against an independent quadrature reference."""

import numpy as np
import pytensor
import pytensor.tensor as pt

from scipy.special import erf

from ptgp.likelihoods import Bernoulli

ATOL = 1e-10


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


def _bernoulli_ve_reference(y, mu, var, n_points=20):
    """Gauss-Hermite quadrature of E_{q(f)}[log p(y | f)] for the probit Bernoulli.

    Reimplements the same quadrature rule PTGP uses (physicist Hermite nodes,
    weights divided by sqrt(pi), f = mu + sqrt(2 var) t) and the same clamped
    probit link, in plain NumPy. This validates PTGP's pytensor implementation
    against a from-scratch reference rather than against another GP library.
    """
    jitter = 1e-3  # mirrors ptgp.likelihoods.bernoulli.inv_probit clamping
    nodes, weights = np.polynomial.hermite.hermgauss(n_points)
    weights = weights / np.sqrt(np.pi)
    sd = np.sqrt(var)[:, None]
    F = mu[:, None] + np.sqrt(2.0) * sd * nodes[None, :]
    p = 0.5 * (1.0 + erf(F / np.sqrt(2.0))) * (1.0 - 2.0 * jitter) + jitter
    log_prob = y[:, None] * np.log(p) + (1.0 - y[:, None]) * np.log(1.0 - p)
    return np.sum(log_prob * weights[None, :], axis=1)


class TestBernoulli:
    def test_ve_matches_quadrature(self):
        mu, var = np.array([0.0, 1.0, -1.0]), np.array([0.25, 0.5, 1.0])
        y = np.array([1.0, 1.0, 0.0])

        ve = _eval(
            Bernoulli(n_points=20).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )

        expected = _bernoulli_ve_reference(y, mu, var, n_points=20)

        np.testing.assert_allclose(ve, expected, atol=ATOL)

    def test_predict_mean_and_var_closed_form(self):
        mu, var = np.array([0.0, 2.0, -2.0]), np.array([0.1, 0.5, 1.0])
        pm_val, pv_val = _eval(
            *Bernoulli().predict_mean_and_var(pt.as_tensor_variable(mu), pt.as_tensor_variable(var))
        )
        assert np.all(pm_val >= 0.0) and np.all(pm_val <= 1.0)
        assert np.all(pv_val >= 0.0) and np.all(pv_val <= 0.25)

    def test_ve_negative(self):
        mu, var = np.array([0.0, 2.0, -2.0]), np.array([0.1, 0.5, 1.0])
        y = np.array([1.0, 0.0, 1.0])
        ve = _eval(
            Bernoulli().variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        assert np.all(ve <= 0.0)
