import copy

import numpy as np
import pytensor.tensor as pt

from pytensor.graph.replace import clone_replace
from pytensor.graph.traversal import ancestors
from pytensor.tensor.type import TensorType


def _data_inputs(value, X):
    """Leaf nodes in ``value``'s graph that look like the design matrix ``X``.

    A leaf (no owner) with the same ndim as ``X`` and matching feature
    dimensions is taken to be a copy of the model input — whether it is a
    symbolic placeholder, a ``pm.Data`` / shared variable, or a constant array
    baked in via ``pt.as_tensor_variable``. Matching is by type, so ``pm.Data``
    (a SharedVariable whose type is ``TensorType``) is caught. PyMC random
    variables and other computed nodes carry an owner and are excluded; scalars
    never match, so literal constants like ``0.1`` are left alone.
    """
    feat = X.type.shape[1:]
    return [
        v
        for v in ancestors([value])
        if v.owner is None
        and isinstance(v.type, TensorType)
        and v.type.ndim == X.type.ndim
        and v.type.ndim > 0
        and all(a is None or b is None or a == b for a, b in zip(v.type.shape[1:], feat))
    ]


def _reroot(value, X):
    """Re-root ``value`` onto ``X`` by replacing design-matrix-shaped leaves.

    The input(s) to replace are discovered from the graph, so the caller only
    supplies the new data. ``rebuild_strict=False`` lets a baked constant array
    (whose row dimension is statically frozen) be replaced by ``X`` without
    coercing ``X`` back to that frozen shape. A data-independent ``value``
    (scalar or PyMC RV) is returned unchanged.

    ``X`` must be a 2D design matrix ``(N, D)``. The discovery in
    :func:`_data_inputs` keys on dimensionality: parameters are per-observation
    1D vectors, so a 2D ``X`` is what distinguishes the data from the parameter
    itself. A 1D ``X`` would collapse that gap and could match the parameter, so
    it is rejected.
    """
    if X.type.ndim < 2:
        raise ValueError(
            "Re-rooting a likelihood parameter requires a 2D design matrix (N, D); "
            f"got X with ndim={X.type.ndim}. Parameters are per-observation 1D "
            "vectors, so a 1D X cannot be distinguished from the parameter itself."
        )
    value = pt.as_tensor_variable(value)
    inputs = _data_inputs(value, X)
    if not inputs:
        return value
    return clone_replace(value, {v: X for v in inputs}, rebuild_strict=False)


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
    param_names: tuple = ()

    def clone_replace_data(self, X):
        """Return a copy with every data-dependent parameter re-rooted onto X.

        Each parameter named in ``param_names`` whose graph depends on a free
        data input is re-expressed over ``X``; data-independent parameters are
        left untouched. Used at predict time to evaluate parameters at the test
        inputs rather than the training inputs they were built against.
        """
        new = copy.copy(self)
        for name in self.param_names:
            setattr(new, name, _reroot(getattr(self, name), X))
        return new

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
        """log E_{q(f)}[p(y|f)] — predictive log-density at test points.

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
