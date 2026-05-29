"""Poisson likelihood tests."""

import numpy as np
import pytensor
import pytensor.tensor as pt

from ptgp.likelihoods import Poisson


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


class TestPoisson:
    def test_closed_form_matches_quadrature(self):
        """Poisson with log link has a closed-form VE — should match quadrature."""
        mu, var = np.array([0.0, 1.0, -0.5]), np.array([0.1, 0.5, 1.0])
        y = np.array([1.0, 3.0, 0.0])

        ve_closed = _eval(
            Poisson().variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        # Use base-op quadrature via _gauss_hermite directly
        lik = Poisson(n_points=50)
        ve_quad = _eval(
            lik._op._gauss_hermite(
                lik._op._log_prob,
                pt.as_tensor_variable(y),
                pt.as_tensor_variable(mu),
                pt.as_tensor_variable(var),
            )
        )

        np.testing.assert_allclose(ve_closed, ve_quad, atol=1e-6)

    def test_ve_values(self):
        """Spot check: with mu=0, var=0, y=1, VE = 1*0 - exp(0) - log(1!) = -1."""
        ve = _eval(
            Poisson().variational_expectation(
                pt.as_tensor_variable(np.array([1.0])),
                pt.as_tensor_variable(np.array([0.0])),
                pt.as_tensor_variable(np.array([0.0])),
            )
        )
        np.testing.assert_allclose(ve, np.array([-1.0]), atol=1e-12)
