import pytensor.tensor as pt

from pytensor.graph.basic import Constant
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors
from pytensor.tensor.type import TensorType

from ptgp.likelihoods.base import Likelihood

LOG2PI = pt.log(2.0 * pt.pi)


class Gaussian(Likelihood):
    """Gaussian likelihood p(y|f) = N(y; f, sigma^2).

    All methods have closed-form expressions (no quadrature needed).

    Parameters
    ----------
    sigma : tensor
        Observation noise standard deviation. May be a scalar (homoskedastic)
        or a vector built against a symbolic ``X`` (heteroskedastic). At
        predict time, :meth:`sigma_at` substitutes the training ``X`` for
        ``X_new`` in sigma's graph; mismatches surface as a strict-mode
        ``graph_replace`` error.
    """

    def __init__(self, sigma):
        self.sigma = pt.as_tensor_variable(sigma)

    def sigma_at(self, X_train, X_new):
        """Return ``sigma`` with ``X_train`` substituted by ``X_new``.

        Sigma is treated as data-dependent iff it has any free symbolic
        tensor input in its graph (covers both raw ``pt.matrix`` X and
        ``pm.Data`` X — the latter is a SharedVariable but its type is
        ``TensorType``). For data-independent sigma the call is a no-op.
        For data-dependent sigma, strict-mode ``graph_replace`` raises a
        clear error if ``X_train`` is not in sigma's graph.
        """
        has_data_dep = any(
            v.owner is None and not isinstance(v, Constant) and isinstance(v.type, TensorType)
            for v in ancestors([self.sigma])
        )
        if not has_data_dep:
            return self.sigma
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
