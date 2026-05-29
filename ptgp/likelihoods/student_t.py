import pytensor.tensor as pt

from ptgp.likelihoods.base import Likelihood


class StudentT(Likelihood):
    """Student-T likelihood p(y|f) = StudentT(y; f, sigma, nu).

    Variational expectation via Gauss-Hermite quadrature.

    Parameters
    ----------
    nu : tensor or PyMC random variable
        Degrees of freedom.
    sigma : tensor or PyMC random variable
        Scale parameter.
    n_points : int
        Number of Gauss-Hermite quadrature points (default 20).
    """

    param_names = ("nu", "sigma")

    def __init__(self, nu, sigma, n_points=20):
        self.nu = pt.as_tensor_variable(nu)
        self.sigma = pt.as_tensor_variable(sigma)
        self.n_points = n_points

    def _log_prob(self, f, y):
        nu, sigma = self.nu, self.sigma
        z = (y - f) / sigma
        return (
            pt.gammaln((nu + 1.0) / 2.0)
            - pt.gammaln(nu / 2.0)
            - 0.5 * pt.log(nu * pt.pi * sigma**2)
            - 0.5 * (nu + 1.0) * pt.log1p(z**2 / nu)
        )

    def _conditional_mean(self, f):
        return f

    def _conditional_variance(self, f):
        return pt.ones_like(f) * self.sigma**2 * self.nu / (self.nu - 2.0)
