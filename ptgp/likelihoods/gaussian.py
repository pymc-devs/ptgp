import pytensor.tensor as pt

from ptgp.likelihoods.base import Likelihood, LikelihoodOp, _param_property

LOG2PI = pt.log(2.0 * pt.pi)


class GaussianOp(LikelihoodOp):
    """Gaussian likelihood Op: closed-form expectations (no quadrature)."""

    def _log_prob(self, f, y, sigma):
        return -0.5 * (LOG2PI + pt.log(sigma**2) + (y - f) ** 2 / sigma**2)

    def _conditional_mean(self, f, sigma):
        return f

    def _conditional_variance(self, f, sigma):
        return pt.ones_like(f) * sigma**2

    def variational_expectation(self, params, y, mu, var):
        (sigma,) = params
        return -0.5 * (LOG2PI + pt.log(sigma**2) + ((y - mu) ** 2 + var) / sigma**2)

    def predict_mean_and_var(self, params, mu, var):
        (sigma,) = params
        return mu, var + sigma**2

    def predict_log_density(self, params, y, mu, var):
        (sigma,) = params
        total_var = var + sigma**2
        return -0.5 * (LOG2PI + pt.log(total_var) + (y - mu) ** 2 / total_var)


class Gaussian(Likelihood):
    """Gaussian likelihood p(y|f) = N(y; f, sigma^2).

    All methods have closed-form expressions (no quadrature needed).

    Parameters
    ----------
    sigma : tensor
        Observation noise standard deviation. May be a scalar (homoskedastic)
        or a vector built against a design matrix ``X`` (heteroskedastic).
    x : tensor, optional
        The design matrix ``sigma`` was built against. Pass it when ``sigma`` is
        heteroskedastic so :meth:`Likelihood.at` can re-root
        sigma onto the test inputs at predict time.
    """

    op_cls = GaussianOp
    param_names = ("sigma",)
    sigma = _param_property("sigma")

    def __init__(self, sigma, x=None):
        super().__init__(x=x, sigma=sigma)
