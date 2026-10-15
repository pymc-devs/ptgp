import pytensor.tensor as pt

from pytensor.graph.basic import Apply
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors
from pytensor.raise_op import Assert

from ptgp.likelihoods.base import _LIK_TYPE, LikelihoodOp, LikelihoodType, _params_of, gp_data


class CompositeOp(LikelihoodOp):
    """Likelihood node that applies a different sub-likelihood to each group of observations.

    Inputs are ``[GPData(x), group, *components]``, where ``group`` is a graph of
    ``x`` giving each row's integer label and ``components`` are likelihood
    variables. Because ``group`` and the components hang off the same ``GPData``
    barrier, :func:`~ptgp.likelihoods.at` re-roots all of them onto new inputs.

    Each expectation selects the rows of group ``k`` with ``nonzero`` and
    evaluates component ``k`` on them, slicing its per-point parameters to those
    rows. The subsets have data-dependent sizes, so the JAX backend cannot
    compile this Op (#67).
    """

    param_names = ("group",)

    def make_node(self, xd, group, *components):
        xd = pt.as_tensor_variable(xd)
        group = pt.as_tensor_variable(group)
        for c in components:
            if not isinstance(c.type, LikelihoodType):
                raise TypeError(
                    f"CompositeLikelihood components must be likelihood variables; got {c.type}. "
                    "Build them with Gaussian, Bernoulli, and the other likelihood builders."
                )
        return Apply(self, [xd, group, *components], [_LIK_TYPE()])

    def _groups(self, params, mu):
        group, *components = params
        labels = pt.cast(pt.round(group), "int64")
        n = len(components)
        in_range = pt.all((labels >= 0) & (labels < n))
        check = Assert(f"group labels must be integers in 0..{n - 1}.")
        labels = check(labels, in_range)
        for k, comp in enumerate(components):
            idx = pt.nonzero(pt.eq(labels, k))[0]
            sub = [pt.broadcast_to(p, mu.shape)[idx] for p in _params_of(comp.owner)]
            yield idx, comp.owner.op, sub

    def variational_expectation(self, params, y, mu, var):
        out = pt.zeros_like(mu)
        for idx, op, sub in self._groups(params, mu):
            out = pt.set_subtensor(
                out[idx], op.variational_expectation(sub, y[idx], mu[idx], var[idx])
            )
        return out

    def predict_mean_and_var(self, params, mu, var):
        mean_out, var_out = pt.zeros_like(mu), pt.zeros_like(var)
        for idx, op, sub in self._groups(params, mu):
            m, v = op.predict_mean_and_var(sub, mu[idx], var[idx])
            mean_out = pt.set_subtensor(mean_out[idx], m)
            var_out = pt.set_subtensor(var_out[idx], v)
        return mean_out, var_out

    def predict_log_density(self, params, y, mu, var):
        out = pt.zeros_like(mu)
        for idx, op, sub in self._groups(params, mu):
            out = pt.set_subtensor(out[idx], op.predict_log_density(sub, y[idx], mu[idx], var[idx]))
        return out


def CompositeLikelihood(likelihoods, group, x):
    """Build a likelihood that applies ``likelihoods[k]`` to the rows labeled ``k``.

    The ELBO data term is a sum over independent points, so the combined
    expectations are each group's expectations scattered back into a length-N
    vector. Group labels must be computed from the inputs, e.g.
    ``group=X[:, kind_col]`` with the kernel's ``active_dims`` excluding that
    column, so they follow minibatches and are recomputed for new inputs at
    prediction.

    Parameters
    ----------
    likelihoods : list of LikelihoodVariable
        One sub-likelihood per group. Input-dependent parameters (e.g. a
        heteroskedastic ``sigma``) must be built against the same ``x``; each
        group sees its own rows of them.
    group : tensor, shape (N,)
        Integer labels in ``0..len(likelihoods) - 1``, computed from ``x``.
    x : tensor, shape (N, D)
        The design matrix ``group`` is built against.

    Returns
    -------
    LikelihoodVariable
    """
    x = pt.as_tensor_variable(x)
    group = pt.as_tensor_variable(group)
    if x not in ancestors([group]):
        raise ValueError(
            "group must be computed from x. Static labels cannot follow minibatches "
            "or new inputs at prediction."
        )
    xd = gp_data(x)
    group, *components = graph_replace([group, *likelihoods], {x: xd}, strict=False)
    return CompositeOp(has_data=True)(xd, group, *components)
