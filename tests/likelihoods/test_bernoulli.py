"""Bernoulli likelihood tests against GPJax reference and analytical results."""

import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt

from gpjax.integrators import GHQuadratureIntegrator
from gpjax.likelihoods import Bernoulli as GPJaxBernoulli

from ptgp.likelihoods import Bernoulli
from ptgp.likelihoods.base import LikelihoodOp

ATOL = 1e-5


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


class TestBernoulli:
    def test_ve_against_gpjax(self):
        mu, var = np.array([0.0, 1.0, -1.0]), np.array([0.25, 0.5, 1.0])
        y = np.array([1.0, 1.0, 0.0])

        ve = _eval(
            Bernoulli(n_points=20).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )

        gpjax_ve = np.array(
            GPJaxBernoulli(
                num_datapoints=3,
                integrator=GHQuadratureIntegrator(num_points=20),
            ).expected_log_likelihood(
                y=jnp.array(y)[:, None],
                mean=jnp.array(mu)[:, None],
                variance=jnp.array(var)[:, None],
            )
        )

        np.testing.assert_allclose(ve, gpjax_ve, atol=ATOL)

    def test_probit_closed_form_predict_matches_quadrature(self):
        """The probit closed-form predictive should agree with the base quadrature
        path it overrides — a correctness check, not just bounds."""
        mu, var = np.array([0.0, 0.7, -1.2]), np.array([0.2, 0.5, 1.0])
        op = Bernoulli(n_points=64).owner.op
        m_closed, v_closed = op.predict_mean_and_var(
            [], pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
        )
        m_quad, v_quad = LikelihoodOp.predict_mean_and_var(
            op, [], pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
        )
        mc, vc, mq, vq = _eval(m_closed, v_closed, m_quad, v_quad)
        np.testing.assert_allclose(mc, mq, atol=1e-4)
        np.testing.assert_allclose(vc, vq, atol=1e-4)

    def test_ve_negative(self):
        mu, var = np.array([0.0, 2.0, -2.0]), np.array([0.1, 0.5, 1.0])
        y = np.array([1.0, 0.0, 1.0])
        ve = _eval(
            Bernoulli().variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        assert np.all(ve <= 0.0)
