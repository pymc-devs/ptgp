import copy

import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import Apply
from pytensor.graph.null_type import NullType
from pytensor.graph.op import Op
from pytensor.graph.replace import graph_replace

_NULL = NullType()


class LikelihoodOp(Op):
    """Structural marker keeping a likelihood inside the PyTensor graph.

    The Apply node consumes the design matrix (optional; input 0 when present)
    and the parameters as its *inputs*, and emits a single ``NoneType`` handle as
    output. It computes nothing — its sole purpose is to be a real graph node, so
    ``graph_replace`` / ``fgraph.clone`` / PyMC conditioning carry the likelihood
    (and re-root its parameters) instead of it being Python-side state those
    operations would silently drop. The Op *type* (the subclass) encodes the
    family, so behaviour is recoverable via ``owner.op``; the parameters are the
    node's inputs.

    Because the parameters are *inputs* and the handle output is never consumed
    by the loss, the node is never on the differentiation path nor in a compiled
    loss graph — so it needs no gradient, view, shape, or strip machinery. To
    keep it alive across graph operations, hold its output (or register it as a
    PyMC named variable).

    Parameters
    ----------
    param_names : tuple of str
        Parameter names, in input order (after the optional design matrix).
    n_points : int
        Gauss-Hermite quadrature points for the default expectations.
    invlink : callable or None
        Inverse link function (set by subclasses that need one).
    has_data : bool
        Whether input 0 is the design matrix ``X`` (i.e. parameters depend on it).
    """

    __props__ = ("param_names", "n_points", "invlink", "has_data")

    def __init__(self, param_names=(), n_points=20, invlink=None, has_data=False):
        self.param_names = tuple(param_names)
        self.n_points = n_points
        self.invlink = invlink
        self.has_data = has_data

    def make_node(self, *args):
        args = [pt.as_tensor_variable(a) for a in args]
        return Apply(self, args, [_NULL()])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = None

    # --- symbolic behaviour: subclasses implement these three ---

    def _log_prob(self, f, y, *params):
        """Symbolic log p(y|f). Subclasses must implement."""
        raise NotImplementedError

    def _conditional_mean(self, f, *params):
        """E[y|f]. Subclasses must implement."""
        raise NotImplementedError

    def _conditional_variance(self, f, *params):
        """Var[y|f]. Subclasses must implement."""
        raise NotImplementedError

    # --- expectations (defaults via quadrature; subclasses may override) ---

    def variational_expectation(self, params, y, mu, var):
        """E_{q(f)}[log p(y|f)] where q(f) = N(mu, var). Default: quadrature."""
        return self._gauss_hermite(lambda f, yy: self._log_prob(f, yy, *params), y, mu, var)

    def predict_mean_and_var(self, params, mu, var):
        """Predictive mean and variance. Default: Gauss-Hermite quadrature."""
        E_mean = self._gauss_hermite(
            lambda f, _: self._conditional_mean(f, *params), pt.zeros_like(mu), mu, var
        )
        E_mean_sq = self._gauss_hermite(
            lambda f, _: self._conditional_mean(f, *params) ** 2, pt.zeros_like(mu), mu, var
        )
        E_var = self._gauss_hermite(
            lambda f, _: self._conditional_variance(f, *params), pt.zeros_like(mu), mu, var
        )
        return E_mean, E_var + E_mean_sq - E_mean**2

    def predict_log_density(self, params, y, mu, var):
        """log E_{q(f)}[p(y|f)]. Default: log-space Gauss-Hermite quadrature."""
        return self._gauss_hermite_logspace(
            lambda f, yy: self._log_prob(f, yy, *params), y, mu, var
        )

    def _gauss_hermite(self, func, y, mu, var):
        """E_{q(f)}[func(f, y)] via Gauss-Hermite quadrature."""
        gh_points, gh_weights = np.polynomial.hermite.hermgauss(self.n_points)
        gh_points = pt.as_tensor_variable(gh_points)
        gh_weights = pt.as_tensor_variable(gh_weights / np.sqrt(np.pi))

        sd = pt.sqrt(var)[:, None]
        F = mu[:, None] + pt.sqrt(2.0) * sd * gh_points[None, :]
        vals = func(F, y[:, None])
        return pt.sum(vals * gh_weights[None, :], axis=1)

    def _gauss_hermite_logspace(self, func, y, mu, var):
        """log E_{q(f)}[exp(func(f, y))] via quadrature, using logsumexp."""
        gh_points, gh_weights = np.polynomial.hermite.hermgauss(self.n_points)
        gh_points = pt.as_tensor_variable(gh_points)
        log_weights = pt.as_tensor_variable(np.log(gh_weights / np.sqrt(np.pi)))

        sd = pt.sqrt(var)[:, None]
        F = mu[:, None] + pt.sqrt(2.0) * sd * gh_points[None, :]
        log_vals = func(F, y[:, None]) + log_weights[None, :]
        return pt.logsumexp(log_vals, axis=1)


# --- Purely functional API ------------------------------------------------
#
# A likelihood is fully described by its node: ``owner.op`` is the family (with
# behaviour), and the node's *inputs* are the optional design matrix followed by
# the parameters. These free functions take such a node (a "lik" — the NoneType
# handle) and dispatch through ``owner.op``, so a likelihood can be used with no
# holder at all. The :class:`Likelihood` holder is a thin view built on these.


def _params_of(node):
    """The parameter inputs of a likelihood node (after the optional X slot)."""
    off = 1 if node.op.has_data else 0
    return list(node.inputs[off:])


def op_of(lik):
    """The :class:`LikelihoodOp` behind a likelihood node (``owner.op``)."""
    return lik.owner.op


def param(lik, name):
    """The named parameter input of a likelihood node (e.g. ``"sigma"``)."""
    node = lik.owner
    off = 1 if node.op.has_data else 0
    return node.inputs[off + node.op.param_names.index(name)]


def at(lik, X):
    """Re-root a likelihood node onto design matrix ``X``.

    A no-op for a likelihood with no design matrix (``op.has_data`` is False) —
    there is no ``X`` input to swap. Otherwise returns a new handle whose
    parameter inputs are re-expressed over ``X`` (RV-safe via ``graph_replace``).
    """
    node = lik.owner
    if not node.op.has_data:
        return lik
    return graph_replace([lik], {node.inputs[0]: X}, strict=False)[0]


def variational_expectation(lik, y, mu, var):
    """E_{q(f)}[log p(y|f)] for a likelihood node, dispatched via ``owner.op``."""
    return lik.owner.op.variational_expectation(_params_of(lik.owner), y, mu, var)


def predict_mean_and_var(lik, mu, var):
    """Predictive mean and variance for a likelihood node."""
    return lik.owner.op.predict_mean_and_var(_params_of(lik.owner), mu, var)


def predict_log_density(lik, y, mu, var):
    """Predictive log-density for a likelihood node."""
    return lik.owner.op.predict_log_density(_params_of(lik.owner), y, mu, var)


def _param_property(name):
    """Expose a named parameter as ``self.<name>`` (the parameter input)."""
    return property(lambda self: self._params[type(self).param_names.index(name)])


class Likelihood:
    """Public likelihood object — a thin view over a :class:`LikelihoodOp` node.

    Every likelihood is a node: construction builds ``op(x?, *params)`` whose
    single ``NoneType`` output (``self._lik``) is the handle, and whose inputs
    are the optional design matrix followed by the parameter graphs. The handle
    keeps the likelihood in the PyTensor graph (carried by ``graph_replace`` /
    ``clone``); the methods read the parameters off the node's inputs and
    dispatch behaviour through ``owner.op`` — the same calls the free functions
    above make.

    The node is never consumed by the loss (the loss uses the parameter graphs
    directly, via ``.sigma`` etc.), so it stays off the gradient path and out of
    compiled graphs. :meth:`at` re-roots the parameters when there's a design
    matrix; it is a no-op otherwise (homoskedastic or parameterless).

    Subclasses set ``op_cls``, ``param_names``, and expose each parameter via
    :func:`_param_property`.
    """

    op_cls = LikelihoodOp
    param_names = ()
    n_points = 20

    def __init__(self, x=None, n_points=20, invlink=None, **params):
        self.n_points = n_points
        self.invlink = invlink
        has_data = bool(self.param_names) and x is not None
        self._op = self.op_cls(self.param_names, n_points, invlink, has_data=has_data)
        ordered = [pt.as_tensor_variable(params[name]) for name in self.param_names]
        args = ([pt.as_tensor_variable(x)] if has_data else []) + ordered
        self._lik = self._op(*args)  # NoneType handle — every likelihood is a node

    @property
    def _params(self):
        return _params_of(self._lik.owner)

    def at(self, X):
        """Return a copy with parameters re-rooted onto ``X`` (no-op without a design matrix)."""
        new = copy.copy(self)
        new._lik = at(self._lik, X)
        return new

    def variational_expectation(self, y, mu, var):
        """E_{q(f)}[log p(y|f)] where q(f) = N(mu, var)."""
        return self._op.variational_expectation(self._params, y, mu, var)

    def predict_mean_and_var(self, mu, var):
        """Predictive mean and variance under q(f) = N(mu, var)."""
        return self._op.predict_mean_and_var(self._params, mu, var)

    def predict_log_density(self, y, mu, var):
        """Predictive log-density log E_{q(f)}[p(y|f)] at test points."""
        return self._op.predict_log_density(self._params, y, mu, var)
