"""Student-t likelihood tests."""

import numpy as np
import pytensor
import pytensor.tensor as pt

from ptgp.likelihoods import Gaussian, StudentT


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


class TestStudentT:
    def test_converges_to_gaussian(self):
        mu, var = np.array([0.0, 0.5]), np.array([0.1, 0.3])
        y, sigma = np.array([0.1, 0.4]), 0.5

        ve_gauss = _eval(
            Gaussian(sigma).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        ve_student = _eval(
            StudentT(nu=1000.0, sigma=sigma).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )

        np.testing.assert_allclose(ve_student, ve_gauss, atol=1e-2)

    def test_quadrature_convergence(self):
        mu, var, y = np.array([0.0]), np.array([1.0]), np.array([0.5])
        ve_20 = _eval(
            StudentT(nu=5.0, sigma=1.0, n_points=20).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        ve_50 = _eval(
            StudentT(nu=5.0, sigma=1.0, n_points=50).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        np.testing.assert_allclose(ve_20, ve_50, atol=1e-6)

    def test_predict_variance_uses_studentt_formula(self):
        """At zero latent variance the predictive variance is the Student-T
        variance sigma**2 * nu / (nu - 2)."""
        mu, var = np.array([0.5, -1.0]), np.zeros(2)
        nu, sigma = 5.0, 0.7
        pm, pv = _eval(
            *StudentT(nu=nu, sigma=sigma).predict_mean_and_var(
                pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        np.testing.assert_allclose(pm, mu, atol=1e-6)
        np.testing.assert_allclose(pv, sigma**2 * nu / (nu - 2.0), atol=1e-6)

    def test_heavier_tails_than_gaussian_for_outlier(self):
        """A far-from-mean observation is penalized less under Student-T than
        Gaussian at matched scale — the heavy-tail behavior."""
        mu, var, y = np.array([0.0]), np.array([0.1]), np.array([10.0])
        sigma = 1.0
        ve_studentt = _eval(
            StudentT(nu=3.0, sigma=sigma).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        ve_gaussian = _eval(
            Gaussian(sigma).variational_expectation(
                pt.as_tensor_variable(y), pt.as_tensor_variable(mu), pt.as_tensor_variable(var)
            )
        )
        assert ve_studentt > ve_gaussian
