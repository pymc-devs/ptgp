import pytensor.tensor as pt

from ptgp.likelihoods.base import LikelihoodOp, to_inputs


class NegativeBinomialOp(LikelihoodOp):
    """Negative binomial likelihood Op. Expectations via quadrature."""

    param_names = ("alpha",)
    allowed_links = ("log", "cloglog")

    def _log_prob(self, f, y, alpha):
        mu = self._invlink(f)
        return (
            pt.gammaln(y + alpha)
            - pt.gammaln(alpha)
            - pt.gammaln(y + 1.0)
            + alpha * pt.log(alpha / (alpha + mu))
            + y * pt.log(mu / (alpha + mu))
        )

    def _conditional_mean(self, f, alpha):
        return self._invlink(f)

    def _conditional_variance(self, f, alpha):
        mu = self._invlink(f)
        return mu + mu**2 / alpha


def NegativeBinomial(alpha, link=None, n_points=20, x=None):
    """Build a negative binomial likelihood NB(y; invlink(f), alpha).

    Returns a :class:`~ptgp.likelihoods.base.LikelihoodVariable`. Variance is
    ``mu + mu**2 / alpha`` with ``mu = invlink(f)``; all links use quadrature.

    Parameters
    ----------
    alpha : tensor or PyMC random variable
        Overdispersion parameter.
    link : str, optional
        Inverse link name, one of ``"log"`` (default) or ``"cloglog"``.
    n_points : int
        Number of Gauss-Hermite quadrature points (default 20).
    x : tensor, optional
        The design matrix ``alpha`` was built against, for a heteroskedastic
        parameter re-rooted onto test inputs via ``.at``.
    """
    op = NegativeBinomialOp(n_points=n_points, link=link, has_data=x is not None)
    return op(*to_inputs([alpha], x))
