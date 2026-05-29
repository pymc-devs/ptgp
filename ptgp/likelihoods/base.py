import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.type import Type
from pytensor.tensor.rewriting.basic import register_specialize


def inv_probit(x):
    """Probit inverse link Phi(x), clamped to (jitter, 1 - jitter)."""
    jitter = 1e-3
    return 0.5 * (1.0 + pt.erf(x / pt.sqrt(2.0))) * (1.0 - 2.0 * jitter) + jitter


def inv_cloglog(x):
    """Complementary log-log inverse link, clamped to (jitter, 1 - jitter)."""
    jitter = 1e-3
    return (1.0 - pt.exp(-pt.exp(x))) * (1.0 - 2.0 * jitter) + jitter


LINKS = {
    "log": pt.exp,
    "logit": pt.sigmoid,
    "probit": inv_probit,
    "cloglog": inv_cloglog,
}


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
        # The guard keeps __getattr__ from recursing on its own attribute access
        # and lets dunders and PyTensor internals fall through as AttributeError.
        if name.startswith("_") or name in ("owner", "type", "index", "tag", "name", "auto_name"):
            raise AttributeError(name)
        owner = self.__dict__.get("owner")
        if owner is None:
            raise AttributeError(name)
        op = owner.op
        if name in op.param_names:
            return param(self, name)
        if name in ("n_points", "link", "param_names", "has_data"):
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

    Subclasses set the class attributes ``param_names`` and ``allowed_links``
    and implement the behaviour methods.
    """

    __props__ = ("param_names", "n_points", "link", "has_data")
    param_names = ()
    allowed_links = ()

    def __init__(self, n_points=20, link=None, has_data=False):
        self.param_names = type(self).param_names
        self.n_points = n_points
        self.link = link if link is not None else self.default_link
        if self.link is not None and self.link not in self.allowed_links:
            raise ValueError(
                f"{type(self).__name__} does not support link {self.link!r}; "
                f"choose from {self.allowed_links}."
            )
        self.has_data = has_data

    @property
    def default_link(self):
        """Default link name: the first of ``allowed_links`` (``None`` if empty)."""
        return self.allowed_links[0] if self.allowed_links else None

    def _invlink(self, f):
        """Map the latent ``f`` to the natural parameter via the inverse link."""
        return LINKS[self.link](f)

    def make_node(self, *args):
        args = [pt.as_tensor_variable(a) for a in args]
        return Apply(self, args, [_LIK_TYPE()])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = None

    def _log_prob(self, f, y, *params):
        """Symbolic log p(y|f). Subclasses must implement."""
        raise NotImplementedError

    def _conditional_mean(self, f, *params):
        """E[y|f]. Subclasses must implement."""
        raise NotImplementedError

    def _conditional_variance(self, f, *params):
        """Var[y|f]. Subclasses must implement."""
        raise NotImplementedError

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


class GPData(Op):
    """Opaque identity barrier marking the design matrix in a parameter graph.

    The design matrix is routed through this Op before it reaches the
    parameters (``sigma = f(gp_data(X))``), which keeps a *stable* handle to it:
    rewrites cannot see through the barrier, so an algebraic cancellation
    (e.g. ``exp(log(x)) -> x``) cannot dissolve the path from the parameter back
    to ``X`` and sever the handle. It is also the likelihood node's design-matrix
    input (input 0), so it is structurally findable and clone-safe.

    It carries no value of its own (identity passthrough) and is stripped at the
    *specialize* stage (:func:`_strip_gp_data`) — i.e. only when building the
    final executable, after re-rooting, never during the canonicalization passes
    a model graph might go through while the handle still matters.
    """

    __props__ = ()
    view_map = {0: [0]}

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = inputs[0]

    def infer_shape(self, fgraph, node, input_shapes):
        return input_shapes

    def pullback(self, inputs, outputs, cotangents):
        return list(cotangents)  # identity


_gp_data_op = GPData()


def gp_data(x):
    """Wrap the design matrix ``x`` in an opaque :class:`GPData` barrier."""
    return _gp_data_op(pt.as_tensor_variable(x))


@register_specialize
@node_rewriter([GPData])
def _strip_gp_data(fgraph, node):
    """Elide the design-matrix barrier when building the final executable.

    Registered at the *specialize* stage (not canonicalize), so it does not fire
    during the canonicalization passes the barrier is meant to survive — it only
    drops the identity Op once all graph transforms (incl. re-rooting) are done.
    """
    return [node.inputs[0]]


def to_inputs(params, x):
    """Node inputs for ``params`` and an optional design matrix ``x``.

    Without ``x`` the inputs are just the parameters. With ``x``, it is wrapped in
    a :class:`GPData` barrier and the parameters are re-expressed over it, so the
    barrier becomes input 0 — the data demarcator that :func:`at` re-roots.
    """
    params = [pt.as_tensor_variable(p) for p in params]
    if x is None:
        return params
    x = pt.as_tensor_variable(x)
    xd = gp_data(x)
    return [xd, *graph_replace(params, {x: xd}, strict=False)]


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
    return _params_of(node)[node.op.param_names.index(name)]


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
