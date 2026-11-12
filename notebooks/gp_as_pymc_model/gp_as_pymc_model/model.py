"""The model layer: a GP is one Op in a PyMC model.

``f = GP("f", kernel)`` is a process. ``f(X)`` is its values at the rows of ``X``: a random
variable in the current ``pm.Model``, and one ``GPRV`` node in the graph.
"""

from dataclasses import dataclass

import pymc as pm
import pytensor.tensor as pt

from pymc.distributions.distribution import SymbolicRandomVariable, _support_point
from pymc.pytensorf import normalize_rng_param
from pytensor.graph.basic import Constant
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.basic import multivariate_normal

from ptgp.kernels.base import Kernel

JITTER = 1e-6


def _parameters(outputs, rows):
    """The variables ``outputs`` are built from that don't depend on ``rows``: the GP's parameters.

    These are the hyperparameters (model RVs, data, anything else) that enter the kernel and
    mean. Constants stay inside the graph.
    """
    rows = set(rows)
    depends = {}

    def dep(v):
        if v not in depends:
            depends[v] = v in rows or (v.owner is not None and any(dep(i) for i in v.owner.inputs))
        return depends[v]

    found = []

    def walk(v):
        if v.owner is None:
            return
        for i in v.owner.inputs:
            if dep(i):
                walk(i)
            elif not isinstance(i, Constant) and i not in found:
                found.append(i)

    for out in outputs:
        if dep(out):
            walk(out)
        elif not isinstance(out, Constant) and out not in found:
            found.append(out)
    return found


class GPRV(SymbolicRandomVariable):
    """The values of the GP ``process`` at some rows: one node in the graph.

    Inputs are ``(rng, rows, *parameters)`` for the first evaluation of a process, and
    ``(rng, rows, given_rows, given_values, *parameters)`` for a later one, which is the GP
    conditional on the earlier evaluations. The inner graph draws a multivariate normal, so PyMC
    samples it and derives its log density with no extra code.

    The Op carries the process's kernel and mean as graphs on its parameters, so the kernel and
    mean at a node's *current* inputs (after ``pm.do``, say) are read off the node, with nothing
    to search for.
    """

    _print_name = ("GP", "\\operatorname{GP}")
    inline_logprob = True

    def __init__(
        self,
        *args,
        process,
        index,
        kernel,
        mean_graph,
        mean_rows,
        parameters,
        conditional,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.process, self.index = process, index
        self._kernel, self._mean_graph, self._mean_rows = kernel, mean_graph, mean_rows
        self._parameters, self.conditional = parameters, conditional

    def rows(self, node):
        return node.inputs[1]

    def _now(self, node):
        """Build-time parameters -> this node's parameter inputs."""
        current = node.inputs[4 if self.conditional else 2 :]
        return {old: new for old, new in zip(self._parameters, current) if old is not new}

    def kernel(self, node):
        """The process's kernel at this node's parameters, as a ptgp ``Kernel``."""
        rep = self._now(node)
        if not rep:
            return self._kernel
        return Kernel(self._kernel.input_dim, graph_replace(self._kernel.graph, rep, strict=False))

    def mean(self, node):
        """Return the process's mean function at this node's parameters, a callable on rows."""
        rep = self._now(node)
        return lambda A: graph_replace(self._mean_graph, {**rep, self._mean_rows: A}, strict=False)


@_support_point.register(GPRV)
def _gp_support_point(op, rv, rng, rows, *rest):
    return pt.zeros(rows.shape[0])


class GP:
    """A Gaussian process. ``f(X)`` is its values at the rows of ``X``, a random variable.

    Every evaluation belongs to the same process: the first is drawn from the prior, each later one
    from the conditional given all earlier ones, so together they're one draw of the GP at all the
    inputs. Calling ``f`` again with the same ``X`` returns the same variable. Which evaluations are
    observed (feed a likelihood) and which are predictions is read from the model, not declared.
    """

    def __init__(self, name, kernel, mean=None, jitter=JITTER):
        self.name, self.kernel, self.jitter = name, kernel, jitter
        left, right = kernel.rows
        self._R = pt.matrix("_rows", shape=(None, kernel.input_dim), dtype=left.dtype)
        self._mean_graph = (
            pt.zeros(self._R.shape[0]) if mean is None else pt.as_tensor(mean(self._R))
        )
        self._parameters = _parameters([kernel.graph, self._mean_graph], [left, right, self._R])
        self._evals = []  # (X, rv)

    def _mean(self, A):
        return graph_replace(self._mean_graph, {self._R: A}, strict=False)

    def _jitter(self, A):
        return self.jitter * pt.eye(A.shape[0])

    def __call__(self, X, name=None):
        for X_, rv in self._evals:
            if X_ is X:
                return rv
        k = len(self._evals)
        if name is None:
            name = f"{self.name}({X.name})" if getattr(X, "name", None) else f"{self.name}_{k}"
        X_in = pt.as_tensor(X)
        K = self.kernel
        rng = normalize_rng_param(None)
        if k == 0:
            given, sig = [], "[rng],(n,d)"
            mu, cov = self._mean(X_in), K(X_in) + self._jitter(X_in)
        else:
            rows_g = pt.concatenate([pt.as_tensor(X_) for X_, _ in self._evals], axis=0)
            f_g = pt.concatenate([rv for _, rv in self._evals])
            given, sig = [rows_g, f_g], "[rng],(n,d),(m,d),(m)"
            Kgg = K(rows_g) + self._jitter(rows_g)
            Kng = K(X_in, rows_g)
            A = pt.linalg.solve(Kgg, Kng.T, assume_a="pos")
            mu = self._mean(X_in) + A.T @ (f_g - self._mean(rows_g))
            cov = K(X_in) - Kng @ A + self._jitter(X_in)
        next_rng, draws = multivariate_normal(mu, cov, rng=rng, return_next_rng=True)
        for i, p in enumerate(self._parameters):
            sig += ",(" + ",".join(f"p{i}_{j}" for j in range(p.ndim)) + ")"
        inputs = [rng, X_in, *given, *self._parameters]
        op = GPRV(
            inputs=inputs,
            outputs=[next_rng, draws],
            extended_signature=sig + "->[rng],(n)",
            process=self.name,
            index=k,
            kernel=K,
            mean_graph=self._mean_graph,
            mean_rows=self._R,
            parameters=list(self._parameters),
            conditional=k > 0,
        )
        rv = pm.modelcontext(None).register_rv(op(*inputs), name)
        self._evals.append((X, rv))
        return rv


@dataclass
class Process:
    """A GP read from a model: its kernel and mean, and its evaluations, split by role."""

    name: str
    kernel: Kernel
    mean: object  # A -> mean(A)
    observed: list  # [(rv, rows)]: evaluations a likelihood reads
    predictive: list  # [(rv, rows)]: the rest


def _direct(v, stops):
    """Members of ``stops`` that ``v``'s inputs reach without passing through another member."""
    found, seen, stack = set(), set(), list(v.owner.inputs)
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        if x in stops:
            found.add(x)
        elif x.owner is not None:
            stack.extend(x.owner.inputs)
    return found


def processes(model):
    """The GPs in ``model``, each with its evaluations sorted into observed and not observed."""
    by_name = {}
    for rv in model.free_RVs:
        if rv.owner is not None and isinstance(rv.owner.op, GPRV):
            by_name.setdefault(rv.owner.op.process, {})[rv.owner.op.index] = rv
    evals = {rv for d in by_name.values() for rv in d.values()}
    reads = set()
    for y in model.observed_RVs:
        reads |= _direct(y, evals)
    out = []
    for name, d in by_name.items():
        first = d[min(d)].owner
        observed, predictive = [], []
        for k in sorted(d):
            rv = d[k]
            (observed if rv in reads else predictive).append((rv, rv.owner.op.rows(rv.owner)))
        out.append(
            Process(name, first.op.kernel(first), first.op.mean(first), observed, predictive)
        )
    return out


# ---------------------------------------------------------------- algebra on GP nodes


def _is(v, scalar_op):
    return (
        v.owner is not None
        and getattr(v.owner.op, "scalar_op", None).__class__.__name__ == scalar_op
    )


def _scalar(v):
    return all(v.type.broadcastable)


def _add_terms(v):
    if _is(v, "Add"):
        for i in v.owner.inputs:
            yield from _add_terms(i)
    else:
        yield v


def _gp_term(v, evals):
    """``v`` as ``coefficient * (a GP evaluation)``: ``(coefficient or None, rv)``, else None."""
    if v in evals:
        return None, v
    if _is(v, "Mul"):
        hits = [(i, _gp_term(i, evals)) for i in v.owner.inputs]
        gp = [(i, t) for i, t in hits if t is not None]
        rest = [i for i, t in hits if t is None]
        if len(gp) == 1 and all(_scalar(i) for i in rest):
            coef, rv = gp[0][1]
            for i in rest:
                coef = i.squeeze() if coef is None else coef * i.squeeze()
            return coef, rv
    return None


@dataclass
class GPSum:
    """A node of the form ``offset + sum_k coefficient_k * f_k(rows)`` found in a graph."""

    var: object  # the Add node
    terms: dict  # GP name -> (coefficient or None, rv)
    offset: object  # everything in the sum that isn't a GP (0 if nothing)
    rows: object


def find_gp_sums(graphs, evals):
    """Sums of GP evaluations at common rows, in ``graphs``.

    ``evals`` maps each ``GPRV`` variable of interest to its GP's name. Returns one ``GPSum`` for
    every outermost ``Add`` node that adds one evaluation of each of those GPs, all at the same
    rows, each optionally times a scalar. This is plain pattern matching on typed nodes.
    """
    names, found, seen = set(evals.values()), [], set()

    def visit(v):
        if v in seen or v.owner is None:
            return
        seen.add(v)
        if _is(v, "Add"):
            terms, offset = {}, []
            for t in _add_terms(v):
                gp = _gp_term(t, evals)
                if gp is None:
                    offset.append(t)
                elif evals[gp[1]] in terms:
                    terms = None
                    break
                else:
                    terms[evals[gp[1]]] = gp
            rows = {rv.owner.inputs[1] for _, rv in (terms or {}).values()}
            clean = not any(e in ancestors(offset) for e in evals) if offset else True
            if terms and set(terms) == names and len(rows) == 1 and clean:
                found.append(
                    GPSum(
                        v,
                        terms,
                        pt.add(*offset) if len(offset) > 1 else (offset[0] if offset else 0.0),
                        rows.pop(),
                    )
                )
                return
        for i in v.owner.inputs:
            visit(i)

    for g in graphs:
        visit(g)
    return found
