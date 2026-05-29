import pytensor.tensor as pt

from ptgp.likelihoods.base import LikelihoodOp, inv_probit


class BernoulliOp(LikelihoodOp):
    """Bernoulli likelihood Op. Closed-form predictive for the probit link."""

    param_names = ()
    allowed_links = ("probit", "logit", "cloglog")

    def _log_prob(self, f, y):
        p = self._invlink(f)
        return y * pt.log(p) + (1.0 - y) * pt.log(1.0 - p)

    def _conditional_mean(self, f):
        return self._invlink(f)

    def _conditional_variance(self, f):
        p = self._invlink(f)
        return p * (1.0 - p)

    def predict_mean_and_var(self, params, mu, var):
        if self.link == "probit":
            p = inv_probit(mu / pt.sqrt(1.0 + var))
            return p, p - p**2
        return super().predict_mean_and_var(params, mu, var)


def Bernoulli(link=None, n_points=20):
    """Build a Bernoulli likelihood p(y=1|f) = invlink(f).

    Returns a :class:`~ptgp.likelihoods.base.LikelihoodVariable`. The probit link
    (default) has a closed-form predictive; other links fall back to quadrature.

    Parameters
    ----------
    link : str, optional
        Inverse link name, one of ``"probit"`` (default), ``"logit"``, or
        ``"cloglog"``.
    n_points : int
        Number of Gauss-Hermite quadrature points (default 20).
    """
    return BernoulliOp(n_points=n_points, link=link)()
