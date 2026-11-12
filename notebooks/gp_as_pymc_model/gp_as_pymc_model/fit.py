"""Fitting and prediction, shaped like ptgp's: ``fit`` (scipy) and ``compile_training_step``
(pytensor-ml optimizers, per-group, with a loop the user owns). Both give a ``Fit``.

A ``Fit`` holds the fitted values, the history of every objective term and of ptgp's
diagnostics during training, and the prediction methods.
"""

from dataclasses import dataclass, field

import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc.pytensorf import compile as pm_compile
from pymc.pytensorf import reseed_rngs
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.type import RandomGeneratorType
from pytensor_ml.optim import adam, compile_train
from pytensor_ml.optim.base import Gradients, Steps

from ptgp.gp.svgp import _softplus_lower_triangular
from ptgp.optim.training import tracked_minimize

from . import expectations as ex
from .model import processes
from .objective import average_blocks, mix_blocks


def _check_no_stray_rvs(obj):
    """A random draw in the loss that isn't the model's means a model variable was copied."""
    own = set(obj.model.basic_RVs)
    stray = [
        v
        for v in ancestors([obj.loss])
        if v.owner
        and isinstance(v.owner.op, RandomVariable)
        and v not in own
        and v.owner.outputs[1] is v
        and "float" in v.dtype
    ]
    if stray:
        raise ValueError(
            f"The objective draws {len(stray)} random variable(s) that aren't the model's. "
            "A model variable was copied while building it."
        )


def _recorded(obj):
    """Everything recorded during a fit: labels, graphs, and how to rebuild the diagnostics."""
    labels, graphs, layout = ["loss"], [obj.loss], {}
    for k, t in obj.terms.items():
        labels.append(k)
        graphs.append(t)
    for k, d in obj.diagnostics.items():
        layout[k] = (type(d), len(graphs), len(d))
        graphs.extend(d)
    return labels, graphs, layout


def _variables(obj, start):
    point = {**obj.model.initial_point(), **(start or {})}
    variables = [*obj.inputs, *[p for p, _ in obj.params.values()]]
    values = [np.array(point[v.name], dtype=float) for v in obj.inputs]
    values += [np.array(init, dtype=float) for _, init in obj.params.values()]
    return point, variables, values


@dataclass
class Fit:
    """A fitted objective."""

    objective: object
    point: dict  # values of the model's variables (unconstrained), fitted where they were fitted
    params: dict  # fitted variational parameters, by name
    result: object = None  # scipy's OptimizeResult, from fit
    _labels: list = field(default_factory=list, repr=False)
    _layout: dict = field(default_factory=dict, repr=False)
    _records: list = field(default_factory=list, repr=False)  # one row per recorded step
    _cache: dict = field(default_factory=dict, repr=False)

    def __repr__(self):
        n = len(self._records)
        loss = f"loss {self.trace['loss'][-1]:.4f}; " if n else ""
        return f"Fit({n} recorded steps; {loss}estimates {self.estimates})"

    # -- what happened during training

    @property
    def trace(self):
        """Loss and every objective term at each recorded step: label -> array."""
        records = np.asarray(self._records).reshape(len(self._records), -1)
        return {k: records[:, i] for i, k in enumerate(self._labels)}

    @property
    def history(self):
        """ptgp's diagnostics at each recorded step: label -> list of namedtuples
        (``VFEDiagnostics``, ``UnapproximatedDiagnostics``, ``FITCTerms``, ``ELBOTerms``)."""
        return {
            k: [cls(*row[i : i + n]) for row in self._records]
            for k, (cls, i, n) in self._layout.items()
        }

    @property
    def diagnostics(self):
        """ptgp's diagnostics at the last recorded step: label -> namedtuple."""
        return {k: v[-1] for k, v in self.history.items()}

    @property
    def terms(self):
        """Every objective term at the last recorded step."""
        return {k: float(v[-1]) for k, v in self.trace.items() if k != "loss"}

    @property
    def estimates(self):
        """The point-estimated variables on their own (constrained) scale."""
        if "estimates" not in self._cache:
            model, names, graphs = self.objective.model, [], []
            for rv in model.free_RVs:
                value = model.rvs_to_values[rv]
                if value in self.objective.inputs:
                    transform = model.rvs_to_transforms[rv]
                    names.append(rv.name)
                    graphs.append(
                        value if transform is None else transform.backward(value, *rv.owner.inputs)
                    )
            graphs = model.replace_rvs_by_values(graphs) if graphs else []
            vals = self.objective.function(graphs)(self.point, self.params) if graphs else []
            self._cache["estimates"] = {n: np.round(v, 4).tolist() for n, v in zip(names, vals)}
        return self._cache["estimates"]

    # -- prediction

    def evaluate(self, graphs):
        """Evaluate graphs of the objective's inputs and parameters at the fitted values."""
        return self.objective.function(list(graphs))(self.point, self.params)

    def predict(self, gp, X):
        """Posterior mean and variance of the GP named ``gp`` at the rows of the array ``X``.
        Nothing is added to the model. For GPs fitted as a sum, ``gp`` is one of them, or the sum
        (``"f_a+f_b"``)."""
        obj = self.objective
        if gp not in obj.gps:
            raise ValueError(f"{gp} isn't a GP in this objective. Its GPs are {list(obj.gps)}.")
        if ("gp", gp) not in self._cache:
            state = obj.gps[gp]
            Xs = pt.matrix(
                "_X", shape=(None, state.gp.kernel.input_dim), dtype=pytensor.config.floatX
            )
            outs = mix_blocks(*obj.model.replace_rvs_by_values(list(state.moments(Xs))), obj.blocks)
            values = [v for v in obj.model.value_vars if v in ancestors(outs)]
            ps = [p for p, _ in obj.params.values()]
            self._cache["gp", gp] = (
                values,
                pytensor.function([*values, *ps, Xs], outs, on_unused_input="ignore"),
            )
        obj.check_training_data()
        values, fn = self._cache["gp", gp]
        X = np.asarray(X, dtype=pytensor.config.floatX)
        X = X[:, None] if X.ndim == 1 else X
        return fn(*[self.point[v.name] for v in values], *[self.params[n] for n in obj.params], X)

    def predict_new(self, name):
        """Posterior mean and variance of the model's unobserved variable ``name`` (a new
        observation such as ``y_new``, or a GP evaluation), at the current ``pm.Data``."""
        if ("new", name) not in self._cache:
            self._cache["new", name] = self.objective.function(
                predict_new_graph(self.objective, name)
            )
        return self._cache["new", name](self.point, self.params)

    def log_predictive(self, name, value):
        """Per-row log density of ``value`` under the predicted distribution of ``name``."""
        return self.evaluate([log_predictive_graph(self.objective, name, value)])[0]


# ---------------------------------------------------------------- prediction graphs


def _q(obj):
    """Moments under q of everything a new observation may read: the GPs at their unobserved
    evaluations (re-read from the model, so ones added after fitting count), and MeanField variables."""
    model = obj.model
    q = dict(obj.meanfield)
    current = {p.name: p for p in processes(model)}
    for name, state in obj.gps.items():
        for rv, X in current[name].predictive if name in current else []:
            q[model.rvs_to_values[rv]] = state.moments(X)
    return q


def _rewrite(obj):
    def rewrite(graphs):
        q = {}
        for fn in obj.rewriters:
            graphs, extra = fn(graphs)
            q.update(extra)
        return graphs, q

    return rewrite


def predict_new_graph(obj, name):
    model, rv, q = obj.model, obj.model[name], _q(obj)
    if model.rvs_to_values.get(rv) in q:
        mean, var = q[model.rvs_to_values[rv]]
    else:
        mean, var = ex.predictive_moments(model, rv, q, obj.n_points, _rewrite(obj))
    return mix_blocks(*model.replace_rvs_by_values([mean, var]), obj.blocks)


def log_predictive_graph(obj, name, value):
    model = obj.model
    lpd = ex.log_predictive(model, model[name], value, _q(obj), obj.n_points, _rewrite(obj))
    return average_blocks(model.replace_rvs_by_values([lpd])[0], obj.blocks, how="logmeanexp")


# ---------------------------------------------------------------- fitting


def fit(obj, method="L-BFGS-B", start=None, **scipy_kwargs):
    """Minimize the loss with ``scipy.optimize.minimize`` (L-BFGS-B by default), the analogue of
    ``ptgp.fit``. Records every term and diagnostic at each iteration, and returns a ``Fit``.

    ``start`` maps model variable names to starting values (in the unconstrained space); other
    keyword arguments go to scipy (``options``, ``tol``, ...).
    """
    _check_no_stray_rvs(obj)
    obj.record_training_data()
    point, variables, values = _variables(obj, start)
    labels, graphs, layout = _recorded(obj)
    shapes = [v.shape for v in values]
    sizes = [int(np.prod(s)) for s in shapes]

    def split(theta):
        out, i = [], 0
        for size, shape in zip(sizes, shapes):
            out.append(theta[i : i + size].reshape(shape))
            i += size
        return out

    loss_and_grad = pm_compile(variables, [obj.loss, *pt.grad(obj.loss, variables)])
    record = pm_compile(variables, graphs, on_unused_input="ignore")

    def fun(theta):
        out = loss_and_grad(*split(theta))
        return float(out[0]), np.concatenate([np.ravel(g) for g in out[1:]])

    if not values:  # nothing to fit (every variable fixed, say by pm.do): just record the terms
        return _finish(obj, point, [], labels, layout, [np.array(record(), dtype=float)], None)
    theta0 = np.concatenate([v.ravel() for v in values])
    result, records = tracked_minimize(
        fun,
        theta0,
        args=(),
        diag_fn=lambda theta: np.array(record(*split(theta)), dtype=float),
        method=method,
        **scipy_kwargs,
    )
    records.append(np.array(record(*split(result.x)), dtype=float))
    return _finish(obj, point, split(result.x), labels, layout, records, result)


def _finish(obj, point, values, labels, layout, records, result):
    n = len(obj.inputs)
    fitted = dict(zip([v.name for v in obj.inputs], values[:n]))
    params = dict(zip(obj.params, values[n:]))
    return Fit(obj, {**point, **fitted}, params, result, labels, layout, list(records))


# ---------------------------------------------------------------- training steps


def _grouped_updates(obj, optimizer, param_groups, variables, shared, loss):
    """Apply each group's transform to its shared variables and merge the updates."""
    by_name = dict(zip(obj.parameter_names, shared))
    group_of, merged = {}, {}
    for name, group in param_groups.items():
        for n in group:
            if n not in by_name:
                raise ValueError(
                    f"param_groups[{name!r}] contains {n!r}, which isn't a parameter of this "
                    f"objective. Its parameters are {list(by_name)}."
                )
            if n in group_of:
                raise ValueError(
                    f"{n} appears in multiple groups ({group_of[n]!r} and {name!r}). "
                    "Each parameter can belong to one group only."
                )
            group_of[n] = name
        updates = optimizer[name](loss, [by_name[n] for n in group])
        if isinstance(updates, Gradients):
            raise ValueError(
                f"The transform for group {name!r} returned gradients rather than steps. "
                "Put a rule such as adam(1e-3) in the group's chain."
            )
        merged.update(updates)
    if missing := [n for n in by_name if n not in group_of]:
        raise ValueError(
            f"No group contains the parameter(s) {missing}. "
            "The groups together have to cover every parameter (see obj.parameter_names)."
        )
    return Steps(merged)


def compile_training_step(
    obj, optimizer=None, param_groups=None, start=None, random_seed=0, compile_kwargs=None
):
    """Compile one training step, the analogue of ``ptgp.optim.compile_training_step``.

    Parameters
    ----------
    optimizer : Transform or dict[str, Transform], optional
        A ``pytensor_ml.optim`` transform such as ``adam(1e-2)`` (the default), or a dict mapping
        group name to transform, with ``param_groups``. ``natural_gradient(gamma)`` is a transform
        for the ``q(u)`` parameters of ``SVGP``. Two rules of the same kind in one dict need
        distinct ``namespace`` arguments.
    param_groups : dict[str, list[str]], optional
        Group name -> parameter names: point-estimated model variables (``"ls"``) and variational
        parameters (``"f_q_mu"``). Required when ``optimizer`` is a dict; the groups together must
        cover every name in ``obj.parameter_names``.
    start : dict, optional
        Model variable name -> starting value, in the unconstrained space.

    Returns
    -------
    train_step : callable
        ``train_step() -> loss``. Draws a new ``pm.Minibatch``, updates the parameters in place,
        and records every term and diagnostic. The training loop is yours.
    fit : Fit
        Reads the parameters ``train_step`` updates, so it predicts from the latest values at
        any point of the loop, and holds the recorded history.
    """
    if isinstance(optimizer, dict) != (param_groups is not None):
        raise ValueError(
            "optimizer as a dict and param_groups go together. "
            "Pass both for per-group optimizers, or a single transform and no groups."
        )
    if isinstance(optimizer, dict) and set(optimizer) != set(param_groups):
        raise ValueError(
            f"optimizer keys {sorted(optimizer)} do not match param_groups keys "
            f"{sorted(param_groups)}. Each group needs exactly one transform."
        )
    _check_no_stray_rvs(obj)
    obj.record_training_data()
    point, variables, values = _variables(obj, start)
    labels, graphs, layout = _recorded(obj)
    shared = [
        pytensor.shared(val, name=var.name, shape=val.shape) for var, val in zip(variables, values)
    ]
    loss, *recorded = graph_replace(graphs, dict(zip(variables, shared)), strict=False)
    if isinstance(optimizer, dict):
        rule = _grouped_updates(obj, optimizer, param_groups, variables, shared, loss)
    else:
        rule = optimizer or adam(1e-2)
    step = compile_train(
        loss,
        rule,
        parameters=shared,
        inputs=[],
        extra_outputs=recorded,
        compile_kwargs=compile_kwargs,
    )
    rngs = [
        v
        for v in ancestors([loss])
        if isinstance(v.type, RandomGeneratorType) and hasattr(v, "set_value")
    ]
    reseed_rngs(rngs, random_seed)
    result = _finish(obj, point, values, labels, layout, [], None)
    n = len(obj.inputs)

    def train_step():
        record = np.array(step(), dtype=float)
        result._records.append(record)
        for var, sv in zip(variables[:n], shared[:n]):
            result.point[var.name] = sv.get_value()
        for name, sv in zip(obj.params, shared[n:]):
            result.params[name] = sv.get_value()
        result._cache.pop("estimates", None)
        return float(record[0])

    return train_step, result


def natural_gradient(gamma=0.1):
    """A transform taking natural-gradient steps on the ``q(u)`` of ``SVGP`` families.

    Its group holds ``q_mu`` and ``q_sqrt`` of one or more GPs (``"f_q_mu", "f_q_sqrt"``). The
    step is ``theta <- theta - gamma * dLoss/d eta`` for natural parameters ``theta = (S^-1 m,
    -S^-1 / 2)`` and expectation parameters ``eta = (m, S + m m^T)``. It's built from the loss
    graph, so it holds for any likelihood. With a Normal likelihood and the other parameters fixed,
    one step with ``gamma = 1`` gives the optimal ``q(u)``: the collapsed (VFE) bound.

    A step that's too large for a non-Normal likelihood can leave the covariance not positive
    definite; use a smaller ``gamma`` then.
    """

    def transform(loss, parameters):
        mus = [p for p in parameters if p.name.endswith("_q_mu")]
        flats = {p.name[: -len("_q_sqrt")]: p for p in parameters if p.name.endswith("_q_sqrt")}
        if len(mus) + len(flats) != len(parameters) or {
            m.name[: -len("_q_mu")] for m in mus
        } != set(flats):
            raise ValueError(
                "natural_gradient applies to the q_mu and q_sqrt of SVGP families, in pairs, and got "
                f"{[p.name for p in parameters]}. Put other parameters in another group."
            )
        updates = {}
        for q_mu in mus:
            q_flat = flats[q_mu.name[: -len("_q_mu")]]
            M = q_mu.type.shape[0]
            rows, cols = np.tril_indices(M)
            diag = np.arange(M)

            def to_flat(L):  # the inverse of ptgp's softplus lower-triangular parameterization
                return pt.set_subtensor(L[diag, diag], pt.log(pt.expm1(L[diag, diag])))[rows, cols]

            # the loss as a function of the expectation parameters, and its gradient there
            e1, e2 = pt.vector("_eta1", shape=(M,)), pt.matrix("_eta2", shape=(M, M))
            L_eta = pt.linalg.cholesky(e2 - pt.outer(e1, e1))
            loss_eta = graph_replace(loss, {q_mu: e1, q_flat: to_flat(L_eta)}, strict=False)
            g1, g2 = pt.grad(loss_eta, [e1, e2])
            L = _softplus_lower_triangular(q_flat, M)
            S = L @ L.T
            g1, g2 = graph_replace([g1, g2], {e1: q_mu, e2: S + pt.outer(q_mu, q_mu)}, strict=False)
            # the step, in the natural parameters
            S_inv = pt.linalg.inv(S)
            theta1 = S_inv @ q_mu - gamma * g1
            theta2 = -0.5 * S_inv - gamma * 0.5 * (g2 + g2.T)
            S_new = pt.linalg.inv(-2.0 * theta2)
            S_new = 0.5 * (S_new + S_new.T)
            updates[q_mu] = S_new @ theta1
            updates[q_flat] = to_flat(pt.linalg.cholesky(S_new))
        return Steps(updates)

    return transform
