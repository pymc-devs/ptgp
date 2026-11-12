"""Expectations of a likelihood under a Gaussian approximation of its latents.

Given a PyMC likelihood and, for each latent it reads, the mean and variance of that latent
under q, compute E_q[log p(y | latents)] per row. Three routes, tried in order:

1. **closed form**, when the likelihood is a known family on a linear predictor
   (Normal with an affine mean, Poisson with ``mu = exp(affine)``);
2. **1-D Gauss-Hermite quadrature over the linear predictor**, when the latents reach the
   likelihood only through one vector that is affine in them;
3. **tensor-product quadrature over the GP values**, otherwise.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pymc.variational.minibatch_rv import get_scaling
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.basic import BernoulliRV, NormalRV, PoissonRV, StudentTRV

LOG2PI = np.log(2.0 * np.pi)


def gauss_hermite(n):
    t, w = np.polynomial.hermite.hermgauss(n)
    return t, w / np.sqrt(np.pi)


# ---------------------------------------------------------------- graph helpers


def substitute(outputs, mapping):
    """``outputs`` with ``mapping`` applied, re-making only the nodes that change.

    Unlike ``clone_replace`` it never copies an unchanged subgraph (so model RVs stay the model's),
    and unlike ``graph_replace`` it lets static shapes follow the new inputs.
    """
    memo = dict(mapping)

    def rebuild(v):
        if v in memo:
            return memo[v]
        node = v.owner
        if node is None:
            out = v
        else:
            inputs = [rebuild(i) for i in node.inputs]
            if all(a is b for a, b in zip(inputs, node.inputs)):
                out = v
            else:
                out = node.op.make_node(*inputs).outputs[v.index]
        memo[v] = out
        return out

    return [rebuild(o) for o in outputs]


def value_ancestors(outputs):
    """Ancestors reached through values, not only through shapes.

    The gradient of ``eps[idx]`` is ``inc_subtensor(zeros_like(eps), ...)``: it depends on
    ``eps`` only for its shape (``Shape``, or the first input of ``Second``, i.e. ``fill``).
    """
    seen, stack = set(), list(outputs)
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        node = v.owner
        if node is None:
            continue
        name = type(node.op).__name__
        if name in ("Shape", "Shape_i"):
            continue
        ins = node.inputs
        if getattr(node.op, "scalar_op", None).__class__.__name__ == "Second":
            ins = ins[1:]
        if name == "Alloc":
            ins = ins[:1]
        stack.extend(ins)
    return seen


def _nearest_first(v):
    seen, order, frontier = {v}, [], [v]
    while frontier:
        nxt = []
        for x in frontier:
            order.append(x)
            if x.owner is not None:
                for i in x.owner.inputs:
                    if i not in seen:
                        seen.add(i)
                        nxt.append(i)
        frontier = nxt
    return order


# ---------------------------------------------------------------- the linear predictor


def is_affine(expr, latents):
    """Whether the vector ``expr`` is affine in ``latents``: its gradient doesn't depend on them."""
    if getattr(expr.type, "ndim", None) != 1:
        return False
    used = [v for v in latents if v in ancestors([expr])]
    if not used:
        return False
    grads = pt.grad(expr.sum(), wrt=used, disconnected_inputs="ignore")
    return not (set(latents) & value_ancestors(grads))


def linear_predictor(exprs, latents):
    """The vector that all of ``latents`` reach ``exprs`` through, if it's affine in them.

    Searches from ``exprs[0]`` toward its inputs for the nearest vector ``eta`` such that every
    latent feeds it, nothing reaches ``exprs`` except through it, and it's affine in the latents.
    Returns ``eta``, or None when there is no such vector.
    """
    latents = list(latents)
    lat = set(latents)
    for v in _nearest_first(exprs[0]):
        if (v in lat and len(lat) > 1) or getattr(v.type, "ndim", None) != 1:
            continue
        if not lat <= set(ancestors([v])):
            continue
        cut = graph_replace(exprs, {v: v.type()}, strict=False)
        if lat & set(ancestors(cut)):
            continue
        if is_affine(v, latents):
            return v
    return None


def predictor_moments(eta, q):
    """Mean and variance of an affine ``eta`` under independent latents with moments ``q``.

    ``q`` maps each latent's value variable to its ``(mean, variance)``. The variance adds each
    latent's contribution separately, assuming each row of ``eta`` reads one element of each
    latent (as in ``f(X) + a[group]``).
    """
    q = {v: mv for v, mv in q.items() if v in ancestors([eta])}
    zeros = {v: pt.zeros_like(m) for v, (m, _) in q.items()}  # not zeros_like(v): that keeps v
    mean = substitute([eta], {v: m for v, (m, _) in q.items()})[0]
    base = substitute([eta], zeros)[0]
    var = 0.0
    for v, (_, s2) in q.items():
        one = substitute([eta], {**zeros, v: pt.sqrt(s2)})[0]
        var = var + (one - base) ** 2
    return mean, var


def integrate(exprs, eta, mean, var, n_points, how="mean"):
    """E over ``eta ~ N(mean, var)`` of each expr, per row; ``how="logmeanexp"`` gives
    log E[exp(expr)] instead (a log predictive density)."""
    t, w = gauss_hermite(n_points)
    e = eta.type()
    cut = graph_replace(exprs, {eta: e}, strict=False)
    grid = mean[None, :] + pt.sqrt(2.0 * var)[None, :] * pt.as_tensor(t)[:, None]
    vals = vectorize_graph(cut, {e: grid})
    if how == "mean":
        return [pt.sum(v * pt.as_tensor(w)[:, None], axis=0) for v in vals]
    return [pt.logsumexp(v + pt.as_tensor(np.log(w))[:, None], axis=0) for v in vals]


def grid_quadrature(graph, values, moments, n_points):
    """E over independent N(m_i, v_i) of ``graph`` (elementwise in ``values``), per row, on a
    tensor-product grid: ``n_points ** len(values)`` evaluations per row."""
    t, w = gauss_hermite(n_points)
    k = len(values)
    grids, weight = {}, 1.0
    for i, (lv, (m, v)) in enumerate(zip(values, moments)):
        shape = [1] * k
        shape[i] = n_points
        grids[lv] = m + pt.sqrt(2.0 * v) * pt.as_tensor(t.reshape([*shape, 1]))
        weight = np.multiply.outer(weight, w) if i else w
    vals = vectorize_graph(graph, grids)
    W = pt.as_tensor(np.asarray(weight).reshape([*list(np.shape(weight)), 1]))
    return pt.sum(vals * W, axis=tuple(range(k)))


# ---------------------------------------------------------------- closed forms


def _normal(y, params, q):
    mu, sigma = params
    if set(q) & set(ancestors([sigma])) or not is_affine(mu, list(q)):
        return None
    m, v = predictor_moments(mu, q)
    return -0.5 * LOG2PI - pt.log(sigma) - ((y - m) ** 2 + v) / (2.0 * sigma**2)


def _poisson(y, params, q):
    (mu,) = params
    op = mu.owner.op if mu.owner is not None else None
    if getattr(op, "scalar_op", None).__class__.__name__ != "Exp":
        return None
    eta = mu.owner.inputs[0]
    if not is_affine(eta, list(q)):
        return None
    m, v = predictor_moments(eta, q)
    return y * m - pt.exp(m + 0.5 * v) - pt.gammaln(y + 1.0)


CLOSED_FORMS = {NormalRV: _normal, PoissonRV: _poisson}


def _closed_form(model, y, y_value, q):
    node, scale = y.owner, 1.0
    if type(node.op).__name__ == "MinibatchRandomVariable":  # pm.Minibatch with total_size
        scale = get_scaling(node.inputs[1:], y_value.shape)
        node = node.inputs[0].owner
    fn = CLOSED_FORMS.get(type(node.op))
    if fn is None:
        return None
    params = model.replace_rvs_by_values(list(node.op.dist_params(node)))
    out = fn(y_value, params, q)
    return None if out is None else out * scale


# ---------------------------------------------------------------- the three routes


def expected_loglik(model, y, q, gp_values, n_points, closed_form=True, rewrite=None):
    """Per-row E_q[log p(y | latents)], and which route computed it; ``(None, None)`` if ``y``
    reads no approximated latent.

    ``q`` maps the value variable of each approximated latent to its ``(mean, variance)``;
    ``gp_values`` are the ones that are GP evaluations (the fallback grid is over those).
    ``rewrite(graphs) -> (graphs, moments)`` may replace parts of the graph (a sum of GPs, say)
    by single variables with known moments.
    """
    y_value = model.rvs_to_values[y]
    (y_new,), extra = rewrite([y]) if rewrite is not None else ([y], {})
    if extra:
        y, q, gp_values = y_new, {**q, **extra}, set(gp_values) | set(extra)
        (lp,) = model.replace_rvs_by_values([pm.logp(y, y_value, warn_rvs=False)])
    else:
        (lp,) = model.logp(vars=[y], sum=False)  # elementwise, with any total_size scaling
    used = {v: q[v] for v in q if v in ancestors([lp])}
    if not used:
        return None, None
    if closed_form:
        out = _closed_form(model, y, y_value, used)
        if out is not None:
            return out, "closed form"
    eta = linear_predictor([lp], list(used))
    if eta is not None:
        mean, var = predictor_moments(eta, used)
        return integrate([lp], eta, mean, var, n_points)[0], "1-D quadrature"
    if set(used) - set(gp_values):
        raise ValueError(
            f"{y.name} reads a MeanField variable through something that isn't a linear predictor. "
            "MeanField variables have to enter the likelihood through a vector that's affine in them."
        )
    return grid_quadrature(
        lp, list(used), list(used.values()), n_points
    ), f"{len(used)}-D quadrature"


# ---------------------------------------------------------------- moments of a new observation

MOMENTS = {
    NormalRV: lambda mu, sigma: (mu, sigma**2),
    StudentTRV: lambda nu, mu, sigma: (mu, sigma**2 * nu / (nu - 2.0)),
    BernoulliRV: lambda p: (p, p * (1.0 - p)),
    PoissonRV: lambda mu: (mu, mu),
}


def predictive_moments(model, rv, q, n_points, rewrite=None):
    """Mean and variance of the unobserved variable ``rv`` with its latents integrated over ``q``.

    ``rewrite(graphs) -> (graphs, moments)`` may replace parts of the graph (a sum of GPs, say)
    by single variables with known moments.
    """
    op = type(rv.owner.op)
    if op not in MOMENTS:
        raise ValueError(
            f"No mean and variance known for {rv.name}'s distribution ({op.__name__})."
        )
    graphs = list(MOMENTS[op](*rv.owner.op.dist_params(rv.owner)))
    if rewrite is not None:
        graphs, extra = rewrite(graphs)
        q = {**q, **extra}
    m, v = model.replace_rvs_by_values(graphs)
    used = {x: q[x] for x in q if x in ancestors([m, v])}
    if not used:
        return m, v
    eta = linear_predictor([m, v], list(used))
    if eta is not None:
        mean, var = predictor_moments(eta, used)
        Em, Ev, Em2 = integrate([m, v, m**2], eta, mean, var, n_points)
    else:
        Em, Ev, Em2 = (
            grid_quadrature(x, list(used), list(used.values()), n_points) for x in (m, v, m**2)
        )
    return Em, Ev + Em2 - Em**2


def log_predictive(model, rv, value, q, n_points, rewrite=None):
    """Per-row log density of ``value`` under ``rv`` with its latents integrated over ``q``."""
    graphs = [pm.logp(rv, pt.as_tensor(value))]
    if rewrite is not None:
        graphs, extra = rewrite(graphs)
        q = {**q, **extra}
    lp = model.replace_rvs_by_values(graphs)[0]
    used = {x: q[x] for x in q if x in ancestors([lp])}
    if not used:
        return lp
    eta = linear_predictor([lp], list(used))
    if eta is None:
        raise ValueError(
            f"{rv.name} doesn't read its latents through a linear predictor, "
            "which the log predictive density needs."
        )
    mean, var = predictor_moments(eta, used)
    return integrate([lp], eta, mean, var, n_points, how="logmeanexp")[0]
