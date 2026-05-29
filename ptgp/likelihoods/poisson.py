import pytensor.tensor as pt

from ptgp.likelihoods.base import LikelihoodOp


class PoissonOp(LikelihoodOp):
    """Poisson likelihood Op. Closed-form variational expectation for log link."""

    param_names = ()
    allowed_links = ("log",)

    def _log_prob(self, f, y):
        lam = self._invlink(f)
        return y * pt.log(lam) - lam - pt.gammaln(y + 1.0)

    def _conditional_mean(self, f):
        return self._invlink(f)

    def _conditional_variance(self, f):
        return self._invlink(f)

    def variational_expectation(self, params, y, mu, var):
        if self.link == "log":
            return y * mu - pt.exp(mu + var / 2.0) - pt.gammaln(y + 1.0)
        return super().variational_expectation(params, y, mu, var)


def Poisson(link=None, n_points=20):
    """Build a Poisson likelihood p(y|f) with rate ``invlink(f)``.

    Returns a :class:`~ptgp.likelihoods.base.LikelihoodVariable`. The log link
    (the default and only supported link) has a closed-form variational
    expectation.

    Parameters
    ----------
    link : str, optional
        Inverse link name; only ``"log"`` is supported (the default).
    n_points : int
        Number of Gauss-Hermite quadrature points (default 20).
    """
    return PoissonOp(n_points=n_points, link=link)()
