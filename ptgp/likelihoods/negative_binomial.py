import pytensor.tensor as pt

from ptgp.likelihoods.base import Likelihood


class NegativeBinomial(Likelihood):
    """Negative binomial likelihood: p(y|f) = NB(y; invlink(f), alpha).

    Parameterized as NB(y; mu, alpha) where mu = invlink(f) and alpha is the
    overdispersion parameter.  Variance is mu + mu^2 / alpha.
    Default link is log (invlink=exp).

    Parameters
    ----------
    alpha : tensor or PyMC random variable
        Overdispersion parameter.
    invlink : callable, optional
        Inverse link function (default: exp).
    n_points : int
        Number of Gauss-Hermite quadrature points (default 20).
    """

    param_names = ("alpha",)

    def __init__(self, alpha, invlink=None, n_points=20):
        self.alpha = pt.as_tensor_variable(alpha)
        self.invlink = invlink or pt.exp
        self.n_points = n_points

    def _log_prob(self, f, y):
        mu = self.invlink(f)
        alpha = self.alpha
        return (
            pt.gammaln(y + alpha)
            - pt.gammaln(alpha)
            - pt.gammaln(y + 1.0)
            + alpha * pt.log(alpha / (alpha + mu))
            + y * pt.log(mu / (alpha + mu))
        )

    def _conditional_mean(self, f):
        return self.invlink(f)

    def _conditional_variance(self, f):
        mu = self.invlink(f)
        return mu + mu**2 / self.alpha
