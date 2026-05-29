import pytensor.tensor as pt

from ptgp.likelihoods.base import LikelihoodOp, build


class NegativeBinomialOp(LikelihoodOp):
    """Negative binomial likelihood Op. Expectations via quadrature."""

    param_names = ("alpha",)
    default_invlink = staticmethod(pt.exp)

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


def NegativeBinomial(alpha, invlink=None, n_points=20, x=None):
    """Build a negative binomial likelihood NB(y; invlink(f), alpha).

    Returns a :class:`~ptgp.likelihoods.base.LikelihoodVariable`. Variance is
    ``mu + mu**2 / alpha`` with ``mu = invlink(f)``; default link is log
    (``invlink=exp``).

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
        parameter re-rooted onto test inputs via ``.at``.
    """
    return build(NegativeBinomialOp, [alpha], x=x, n_points=n_points, invlink=invlink)
