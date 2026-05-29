import pytensor.tensor as pt

from ptgp.likelihoods.base import LikelihoodOp, build


def inv_probit(x):
    """Probit link: Phi(x) = 0.5 * (1 + erf(x / sqrt(2))), clamped to (jitter, 1-jitter)."""
    jitter = 1e-3
    return 0.5 * (1.0 + pt.erf(x / pt.sqrt(2.0))) * (1.0 - 2.0 * jitter) + jitter


class BernoulliOp(LikelihoodOp):
    """Bernoulli likelihood Op. Closed-form predictive for the probit link."""

    param_names = ()
    default_invlink = staticmethod(inv_probit)

    def _log_prob(self, f, y):
        p = self.invlink(f)
        return y * pt.log(p) + (1.0 - y) * pt.log(1.0 - p)

    def _conditional_mean(self, f):
        return self.invlink(f)

    def _conditional_variance(self, f):
        p = self.invlink(f)
        return p * (1.0 - p)

    def predict_mean_and_var(self, params, mu, var):
        if self.invlink is inv_probit:
            p = inv_probit(mu / pt.sqrt(1.0 + var))
            return p, p - p**2
        return super().predict_mean_and_var(params, mu, var)


def Bernoulli(invlink=None, n_points=20):
    """Build a Bernoulli likelihood p(y=1|f) = invlink(f).

    Returns a :class:`~ptgp.likelihoods.base.LikelihoodVariable`. Default link is
    probit (closed-form predictive); pass ``invlink=pt.sigmoid`` for logit.
    """
    return build(BernoulliOp, [], n_points=n_points, invlink=invlink)
