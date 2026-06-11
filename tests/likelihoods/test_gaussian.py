"""Gaussian likelihood tests against closed-form analytic results."""

import numpy as np
import pytensor
import pytensor.tensor as pt

from ptgp.likelihoods import Gaussian

ATOL = 1e-12


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


class TestGaussian:
    def test_ve_closed_form(self):
        mu, var = np.array([0.0, 0.5, -1.0]), np.array([0.1, 0.5, 1.0])
        y, sigma = np.array([0.1, 0.3, -0.8]), 0.5

        ve = _eval(
            Gaussian(sigma).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )

        # E_{q(f)}[log N(y | f, sigma^2)] for q(f) = N(mu, var) has the closed form
        #   -0.5 log(2 pi sigma^2) - 0.5 ((y - mu)^2 + var) / sigma^2
        expected = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((y - mu) ** 2 + var) / sigma**2

        np.testing.assert_allclose(ve, expected, atol=ATOL)

    def test_zero_var_matches_log_prob(self):
        mu, y, sigma = np.array([0.0, 1.0]), np.array([0.1, 0.9]), 0.3
        lik = Gaussian(sigma)
        ve = _eval(
            lik.variational_expectation(
                pt.as_tensor_variable(y),
                pt.as_tensor_variable(mu),
                pt.as_tensor_variable(np.zeros(2)),
            )
        )
        lp = _eval(lik._log_prob(pt.as_tensor_variable(mu), pt.as_tensor_variable(y)))
        np.testing.assert_allclose(ve, lp, atol=1e-12)

    def test_predict_mean_and_var(self):
        mu, var, sigma = np.array([1.0, 2.0]), np.array([0.5, 1.0]), 0.3
        pm, pv = _eval(
            *Gaussian(sigma).predict_mean_and_var(
                pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        np.testing.assert_allclose(pm, mu, atol=1e-12)
        np.testing.assert_allclose(pv, var + sigma**2, atol=1e-12)

    def test_predict_log_density(self):
        mu, var = np.array([0.0, 1.0]), np.array([0.1, 0.5])
        y, sigma = np.array([0.1, 0.8]), 0.5
        pld = _eval(
            Gaussian(sigma).predict_log_density(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        # Manual: log N(y; mu, var + sigma^2)
        total_var = var + sigma**2
        expected = -0.5 * (np.log(2 * np.pi * total_var) + (y - mu) ** 2 / total_var)
        np.testing.assert_allclose(pld, expected, atol=1e-12)
