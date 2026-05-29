import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.graph.replace import graph_replace
from pytensor.graph.type import Type


class LikelihoodVariable(Variable):
    """A likelihood that lives in the PyTensor graph *and* carries the accessor API.

    It is the output of a :class:`LikelihoodOp` node, so ``owner.op`` recovers
    the family (with behaviour) and ``clone`` / ``graph_replace`` carry it. It
    also exposes the methods we used to put on a holder, each dispatched through
    ``owner.op`` (the free functions below):

    - parameter access — ``lik.sigma`` / ``lik.nu`` / ... (the node's inputs),
    - :meth:`at` — re-root the parameters onto new design inputs,
    - :meth:`variational_expectation` / :meth:`predict_mean_and_var` /
      :meth:`predict_log_density`.
    """

    def at(self, X):
        """Re-root onto design matrix ``X`` (no-op without a design matrix)."""
        return at(self, X)

    def variational_expectation(self, y, mu, var):
        """E_{q(f)}[log p(y|f)] where q(f) = N(mu, var)."""
        return variational_expectation(self, y, mu, var)

    def predict_mean_and_var(self, mu, var):
        """Predictive mean and variance under q(f) = N(mu, var)."""
        return predict_mean_and_var(self, mu, var)

    def predict_log_density(self, y, mu, var):
        """Predictive log-density log E_{q(f)}[p(y|f)] at test points."""
        return predict_log_density(self, y, mu, var)

    def __getattr__(self, name):
        # Only fires for attributes not found normally. Expose the named
        # parameters (node inputs) and a little Op config; everything else
        # (incl. PyTensor internals) raises AttributeError as usual.
        if name.startswith("_") or name in ("owner", "type", "index", "tag", "name", "auto_name"):
            raise AttributeError(name)
        owner = self.__dict__.get("owner")
        if owner is None:
            raise AttributeError(name)
        op = owner.op
        if name in op.param_names:
            off = 1 if op.has_data else 0
            return owner.inputs[off + op.param_names.index(name)]
        if name in ("n_points", "invlink", "param_names", "has_data"):
            return getattr(op, name)
        raise AttributeError(name)


class LikelihoodType(Type):
    """Type of a likelihood handle. Carries no runtime value — the node it comes
    from is a structural marker, never consumed by the loss or compiled."""

    def filter(self, value, strict=False, allow_downcast=None):
        return value

    def make_variable(self, name=None):
        return LikelihoodVariable(self, None, name=name)

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


_LIK_TYPE = LikelihoodType()


class LikelihoodOp(Op):
    """Structural marker keeping a likelihood inside the PyTensor graph.

    The Apply node consumes the design matrix (optional; input 0 when present)
    and the parameters as its *inputs*, and emits a single :class:`LikelihoodType`
    handle (a :class:`LikelihoodVariable`) as output. It computes nothing — it is
    a real graph node purely so ``graph_replace`` / ``fgraph.clone`` / PyMC
    conditioning carry the likelihood (and re-root its parameters). The Op *type*
    (the subclass) encodes the family; behaviour is recoverable via ``owner.op``.

    Because the parameters are *inputs* and the handle output is never consumed
    by the loss, the node is never on the differentiation path nor in a compiled
    loss graph — no gradient, view, shape, or strip machinery is needed.

    Subclasses set the class attributes ``param_names`` and (optionally)
    ``default_invlink`` and implement the behaviour methods.
    """

    __props__ = ("param_names", "n_points", "invlink", "has_data")
    param_names = ()
    default_invlink = None

    def __init__(self, n_points=20, invlink=None, has_data=False):
        self.param_names = type(self).param_names
        self.n_points = n_points
        self.invlink = invlink if invlink is not None else type(self).default_invlink
        self.has_data = has_data

    def make_node(self, *args):
        args = [pt.as_tensor_variable(a) for a in args]
        return Apply(self, args, [_LIK_TYPE()])

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


def build(op_cls, params, x=None, n_points=20, invlink=None):
    """Build a :class:`LikelihoodVariable` from an Op class and ordered params.

    ``params`` is the list of parameter graphs in ``op_cls.param_names`` order
    (empty for parameterless likelihoods). ``x`` (the design matrix) is prepended
    as input 0 when given, marking the parameters as data-dependent. Used by the
    family helpers (:func:`~ptgp.likelihoods.Gaussian`, etc.).
    """
    has_data = bool(op_cls.param_names) and x is not None
    op = op_cls(n_points=n_points, invlink=invlink, has_data=has_data)
    args = [pt.as_tensor_variable(p) for p in params]
    if has_data:
        args = [pt.as_tensor_variable(x), *args]
    return op(*args)


# --- Purely functional API ------------------------------------------------
#
# Operate on a likelihood node (a :class:`LikelihoodVariable`) and dispatch
# through ``owner.op``. The variable's methods are thin wrappers over these, so
# either style works; the free functions are handy when a dummy node's outputs
# are plain TensorVariables rather than a LikelihoodVariable.


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

    A no-op for a likelihood with no design matrix (``op.has_data`` is False).
    Otherwise returns a new handle whose parameter inputs are re-expressed over
    ``X`` (RV-safe via ``graph_replace``).
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
