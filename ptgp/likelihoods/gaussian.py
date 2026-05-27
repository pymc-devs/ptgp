import pytensor.assumptions as pta
import pytensor.tensor as pt

from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors

from ptgp.likelihoods.base import Likelihood

LOG2PI = pt.log(2.0 * pt.pi)


class Gaussian(Likelihood):
    """Gaussian likelihood p(y|f) = N(y; f, sigma^2).

    All methods have closed-form expressions (no quadrature needed).

    Parameters
    ----------
    sigma : tensor
        Observation noise standard deviation. May be a scalar (homoskedastic)
        or a vector built against a symbolic ``X`` (heteroskedastic). For the
        heteroskedastic case, set ``sigma.tag.requires_data = True`` so that
        :meth:`sigma_at` substitutes X via ``graph_replace`` at predict time
        and raises on a wiring error (sigma built against the wrong X).
    """

    def __init__(self, sigma):
        sigma = pt.as_tensor_variable(sigma)
        if getattr(sigma.tag, "requires_data", False):
            self.sigma.tag.requires_data = True

    def sigma_at(self, X_train, X_new):
        """Return ``sigma`` with ``X_train`` substituted by ``X_new``.

        - If ``sigma.tag.requires_data`` is set, runs ``graph_replace`` and
          checks that the substitution actually fired (raises with a clear
          message if ``X_train`` is not in sigma's graph).
        - Otherwise sigma is treated as independent of X and returned as-is
          (no graph_replace, which would error on unused substitutions in
          strict mode).
        """
        if not getattr(self.sigma.tag, "requires_data", False):
            return self.sigma
        if X_train not in list(ancestors([self.sigma])):
            raise ValueError(
                "sigma is tagged requires_data=True but X_train is not in its graph. "
                "Build sigma against the same X you pass to predict_marginal."
            )
        return graph_replace(self.sigma, {X_train: X_new})

    def _log_prob(self, f, y):
        return -0.5 * (LOG2PI + pt.log(self.sigma**2) + (y - f) ** 2 / self.sigma**2)

    def _conditional_mean(self, f):
        return f

    def _conditional_variance(self, f):
        return pt.ones_like(f) * self.sigma**2

    def variational_expectation(self, y, mu, var):
        return -0.5 * (LOG2PI + pt.log(self.sigma**2) + ((y - mu) ** 2 + var) / self.sigma**2)

    def predict_mean_and_var(self, mu, var):
        return mu, var + self.sigma**2

    def predict_log_density(self, y, mu, var):
        total_var = var + self.sigma**2
        return -0.5 * (LOG2PI + pt.log(total_var) + (y - mu) ** 2 / total_var)
