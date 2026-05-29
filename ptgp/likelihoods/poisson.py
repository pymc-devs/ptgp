import pytensor.tensor as pt

from ptgp.likelihoods.base import LikelihoodOp, build


class PoissonOp(LikelihoodOp):
    """Poisson likelihood Op. Closed-form variational expectation for log link."""

    param_names = ()
    default_invlink = staticmethod(pt.exp)

    def _log_prob(self, f, y):
        lam = self.invlink(f)
        return y * pt.log(lam) - lam - pt.gammaln(y + 1.0)

    def _conditional_mean(self, f):
        return self.invlink(f)

    def _conditional_variance(self, f):
        return self.invlink(f)

    def variational_expectation(self, params, y, mu, var):
        if self.invlink is pt.exp:
            return y * mu - pt.exp(mu + var / 2.0) - pt.gammaln(y + 1.0)
        return super().variational_expectation(params, y, mu, var)


def Poisson(invlink=None, n_points=20):
    """Build a Poisson likelihood p(y|f) = Poisson(y; invlink(f)).

    Returns a :class:`~ptgp.likelihoods.base.LikelihoodVariable`. Default link is
    log (closed-form variational expectation); other links fall back to quadrature.
    """
    return build(PoissonOp, [], n_points=n_points, invlink=invlink)
