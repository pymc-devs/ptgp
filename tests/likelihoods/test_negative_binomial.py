"""Negative binomial likelihood tests."""

import numpy as np
import pytensor
import pytensor.tensor as pt

from ptgp.likelihoods import NegativeBinomial, Poisson


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


class TestNegativeBinomial:
    def test_quadrature_convergence(self):
        mu, var = np.array([0.5, 1.0]), np.array([0.2, 0.5])
        y = np.array([2.0, 5.0])

        ve_20 = _eval(
            NegativeBinomial(alpha=5.0, n_points=20).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        ve_50 = _eval(
            NegativeBinomial(alpha=5.0, n_points=50).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        np.testing.assert_allclose(ve_20, ve_50, atol=1e-6)

    def test_converges_to_poisson(self):
        """NB with large alpha should approach Poisson."""
        mu, var = np.array([0.5, 1.0]), np.array([0.1, 0.3])
        y = np.array([1.0, 3.0])

        # Use quadrature for both so comparison is apples-to-apples
        poisson_lik = Poisson(n_points=50)
        pop = poisson_lik.owner.op
        ve_poisson = _eval(
            pop._gauss_hermite(
                pop._log_prob,
                pt.as_tensor_variable(y),
                pt.as_tensor_variable(mu),
                pt.as_tensor_variable(var),
            )
        )
        ve_nb = _eval(
            NegativeBinomial(alpha=1e4, n_points=50).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )

        np.testing.assert_allclose(ve_nb, ve_poisson, atol=1e-2)
