import pytensor.tensor as pt

from ptgp.likelihoods.base import LikelihoodOp, to_inputs

LOG2PI = pt.log(2.0 * pt.pi)


class GaussianOp(LikelihoodOp):
    """Gaussian likelihood Op: closed-form expectations (no quadrature)."""

    param_names = ("sigma",)

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


def Gaussian(sigma, x=None):
    """Build a Gaussian likelihood p(y|f) = N(y; f, sigma^2).

    Returns a :class:`~ptgp.likelihoods.base.LikelihoodVariable` — a graph node
    exposing ``.sigma``, ``.at``, ``.predict_mean_and_var``, etc.

    Parameters
    ----------
    sigma : tensor
        Observation noise standard deviation. Scalar (homoskedastic) or a vector
        built against a design matrix ``X`` (heteroskedastic).
    x : tensor, optional
        The design matrix ``sigma`` was built against; pass it for
        heteroskedastic noise so ``.at`` can re-root sigma onto test inputs.
    """
    op = GaussianOp(has_data=x is not None)
    return op(*to_inputs([sigma], x))
