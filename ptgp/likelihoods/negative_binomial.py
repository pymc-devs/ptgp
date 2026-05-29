import pytensor.tensor as pt

from ptgp.likelihoods.base import Likelihood, LikelihoodOp, _param_property


class NegativeBinomialOp(LikelihoodOp):
    """Negative binomial likelihood Op. Expectations via quadrature."""

    def _log_prob(self, f, y, alpha):
        mu = self.invlink(f)
        return (
            pt.gammaln(y + alpha)
            - pt.gammaln(alpha)
            - pt.gammaln(y + 1.0)
            + alpha * pt.log(alpha / (alpha + mu))
            + y * pt.log(mu / (alpha + mu))
        )

    def _conditional_mean(self, f, alpha):
        return self.invlink(f)

    def _conditional_variance(self, f, alpha):
        mu = self.invlink(f)
        return mu + mu**2 / alpha


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
    x : tensor, optional
        The design matrix ``alpha`` was built against, for a heteroskedastic
        parameter re-rooted onto the test inputs at predict time.
    """

    op_cls = NegativeBinomialOp
    param_names = ("alpha",)
    alpha = _param_property("alpha")

    def __init__(self, alpha, invlink=None, n_points=20, x=None):
        super().__init__(x=x, n_points=n_points, invlink=invlink or pt.exp, alpha=alpha)
