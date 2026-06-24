import numpy as np
import pytensor.tensor as pt


class Likelihood:
    """Base class for all PTGP likelihoods.

    Subclasses must implement ``_log_prob(f, y)`` which returns the symbolic
    log-likelihood log p(y|f).

    Gaussian likelihoods override ``variational_expectation`` and
    ``predict_mean_and_var`` with closed-form expressions.  Non-Gaussian
    likelihoods inherit the default Gauss-Hermite quadrature implementations.

    Parameters
    ----------
    invlink : callable, optional
        Inverse link function mapping latent f to the natural parameter.
        Subclasses set a default (e.g. exp for Poisson, sigmoid for Bernoulli).
    n_points : int
        Number of Gauss-Hermite quadrature points (default 20).
    """

    n_points: int = 20
    invlink = None

    def _log_prob(self, f, y):
        """Symbolic log p(y|f). Subclasses must implement."""
        raise NotImplementedError

    def _conditional_mean(self, f):
        """E[y|f]. Subclasses must implement."""
        raise NotImplementedError

    def _conditional_variance(self, f):
        """Var[y|f]. Subclasses must implement."""
        raise NotImplementedError

    def variational_expectation(self, y, mu, var):
        """E_{q(f)}[log p(y|f)] where q(f) = N(mu, var).

        Default: Gauss-Hermite quadrature.
        """
        return self._gauss_hermite(lambda f, y: self._log_prob(f, y), y, mu, var)

    def predict_mean_and_var(self, mu, var):
        """Predictive mean and variance: E_{q(f)}[E[y|f]], E_{q(f)}[Var[y|f]] + Var_{q(f)}[E[y|f]].

        Default: Gauss-Hermite quadrature.
        """
        E_mean = self._gauss_hermite(
            lambda f, _: self._conditional_mean(f), pt.zeros_like(mu), mu, var
        )
        E_mean_sq = self._gauss_hermite(
            lambda f, _: self._conditional_mean(f) ** 2, pt.zeros_like(mu), mu, var
        )
        E_var = self._gauss_hermite(
            lambda f, _: self._conditional_variance(f), pt.zeros_like(mu), mu, var
        )
        return E_mean, E_var + E_mean_sq - E_mean**2

    def predict_log_density(self, y, mu, var):
        """log E_{q(f)}[p(y|f)], the predictive log-density at test points.

        Default: Gauss-Hermite quadrature in log-space for numerical stability.
        """
        return self._gauss_hermite_logspace(lambda f, y: self._log_prob(f, y), y, mu, var)

    def _gauss_hermite(self, func, y, mu, var):
        """E_{q(f)}[func(f, y)] via Gauss-Hermite quadrature."""
        gh_points, gh_weights = np.polynomial.hermite.hermgauss(self.n_points)
        gh_points = pt.as_tensor_variable(gh_points)
        gh_weights = pt.as_tensor_variable(gh_weights / np.sqrt(np.pi))

        # f = mu + sqrt(2 * var) * t_j, shape (N, n_points)
        sd = pt.sqrt(var)[:, None]
        F = mu[:, None] + pt.sqrt(2.0) * sd * gh_points[None, :]

        # func(f, y) at each quadrature point, shape (N, n_points)
        vals = func(F, y[:, None])

        return pt.sum(vals * gh_weights[None, :], axis=1)

    def _gauss_hermite_logspace(self, func, y, mu, var):
        """log E_{q(f)}[exp(func(f, y))] via Gauss-Hermite quadrature.

        Uses logsumexp for numerical stability.
        """
        gh_points, gh_weights = np.polynomial.hermite.hermgauss(self.n_points)
        gh_points = pt.as_tensor_variable(gh_points)
        log_weights = pt.as_tensor_variable(np.log(gh_weights / np.sqrt(np.pi)))

        sd = pt.sqrt(var)[:, None]
        F = mu[:, None] + pt.sqrt(2.0) * sd * gh_points[None, :]

        log_vals = func(F, y[:, None]) + log_weights[None, :]

        return pt.logsumexp(log_vals, axis=1)
