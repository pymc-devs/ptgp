import pytensor.tensor as pt

from ptgp.likelihoods.base import Likelihood

LOG2PI = pt.log(2.0 * pt.pi)


class Gaussian(Likelihood):
    """Gaussian likelihood p(y|f) = N(y; f, sigma^2).

    All methods have closed-form expressions (no quadrature needed).

    Parameters
    ----------
    sigma : tensor or PyMC random variable
        Observation noise standard deviation.
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def _log_prob(self, f, y):
        return -0.5 * (LOG2PI + pt.log(self.sigma**2) + (y - f) ** 2 / self.sigma**2)

    def _conditional_mean(self, f):
        return f

    def _conditional_variance(self, f):
        return pt.ones_like(f) * self.sigma**2

    def variational_expectation(self, y, mu, var):
        return -0.5 * (LOG2PI + pt.log(self.sigma**2) + ((y - mu) ** 2 + var) / self.sigma**2)

    def predict_mean_and_var(self, mu, var):
        return mu, var + self.sigma**2

    def predict_log_density(self, y, mu, var):
        total_var = var + self.sigma**2
        return -0.5 * (LOG2PI + pt.log(total_var) + (y - mu) ** 2 / total_var)
