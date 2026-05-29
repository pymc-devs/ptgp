import pytensor.tensor as pt

from ptgp.likelihoods.base import Likelihood, LikelihoodOp


def inv_probit(x):
    """Probit link: Phi(x) = 0.5 * (1 + erf(x / sqrt(2))), clamped to (jitter, 1-jitter)."""
    jitter = 1e-3
    return 0.5 * (1.0 + pt.erf(x / pt.sqrt(2.0))) * (1.0 - 2.0 * jitter) + jitter


class BernoulliOp(LikelihoodOp):
    """Bernoulli likelihood Op. Closed-form predictive for the probit link."""

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


class Bernoulli(Likelihood):
    """Bernoulli likelihood: p(y=1|f) = invlink(f).

    Default link is probit. Variational expectation via Gauss-Hermite quadrature.

    Parameters
    ----------
    invlink : callable, optional
        Inverse link function (default: probit). Use ``pt.sigmoid`` for logit link.
    n_points : int
        Number of Gauss-Hermite quadrature points (default 20).
    """

    op_cls = BernoulliOp
    param_names = ()

    def __init__(self, invlink=None, n_points=20):
        super().__init__(n_points=n_points, invlink=invlink or inv_probit)
