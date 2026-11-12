"""The objective layer: ``objective(model, approx)`` reads a model and returns named terms.

The analogue of ``model.logp(vars=..., sum=False)`` for a fitting criterion. ``approx`` maps
each GP's name, and any other variable's name (or a tuple of names), to a family from
``families``. Variables not listed are point estimates. Each family builds its own terms; this
module only finds the pieces, asks the families, and assembles the result.
"""

from dataclasses import dataclass, field

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.graph.replace import vectorize_graph
from pytensor.graph.traversal import ancestors

import ptgp as pg

from .expectations import expected_loglik, value_ancestors
from .model import _direct, processes


def average_blocks(graph, blocks, how="mean"):
    """``graph`` averaged over the fixed draws of every Gaussian block it depends on."""
    rep = {}
    for values, loc, logsd, draws in blocks:
        i = 0
        for v in values:
            n = int(np.prod(v.type.shape)) if v.ndim else 1
            if v in ancestors([graph]):
                d = loc[i : i + n][None, :] + pt.exp(logsd[i : i + n])[None, :] * pt.as_tensor(
                    draws[:, i : i + n]
                )
                rep[v] = d.reshape((draws.shape[0], *v.type.shape)) if v.ndim else d[:, 0]
            i += n
    if not rep:
        return graph
    batched = vectorize_graph(graph, rep)
    if how == "mean":
        return pt.mean(batched, axis=0)
    return pt.logsumexp(batched, axis=0) - np.log(next(iter(rep.values())).shape[0])


def mix_blocks(mean, var, blocks):
    """Mean and variance after averaging over the blocks' draws: E[m], E[v + m^2] - E[m]^2."""
    m = average_blocks(mean, blocks)
    return [m, average_blocks(var + mean**2, blocks) - m**2]


@dataclass
class Objective:
    """A fitting criterion read from a model, term by term."""

    model: object
    terms: dict  # label -> scalar graph
    kinds: dict  # label -> "bound" | "penalty"
    routes: dict  # label -> how a data term was computed ("closed form", "1-D quadrature", ...)
    pointwise: dict  # label -> per-row data term
    diagnostics: dict  # label -> namedtuple of scalar graphs (ptgp's diagnostics types)
    params: dict  # variational parameter name -> (variable, initial value)
    inputs: list  # value variables of the point-estimated model variables
    gps: dict  # GP name -> the bound family, which knows the GP's posterior
    blocks: list  # Gaussian blocks: (value variables, loc, logsd, fixed draws)
    meanfield: dict  # value variable -> (mean, variance) under q
    n_points: int = 20
    rewriters: list = field(
        default_factory=list
    )  # graph -> (graph, moments): sums of GPs, for prediction
    fitted_data: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict)

    @property
    def init(self):
        """Initial values of the variational parameters, by name."""
        return {n: init for n, (_, init) in self.params.items()}

    @property
    def parameter_names(self):
        """Names of everything a fit optimizes: the point-estimated model variables, then the
        variational parameters. These are the names ``param_groups`` uses."""
        values_to_rvs = {v: rv for rv, v in self.model.rvs_to_values.items()}
        return [values_to_rvs[v].name for v in self.inputs] + list(self.params)

    @property
    def bound(self):
        return pt.sum([t for k, t in self.terms.items() if self.kinds[k] == "bound"])

    @property
    def loss(self):
        return -pt.sum(list(self.terms.values()))

    def __repr__(self):
        rows = [
            f"  {k:26s} {self.kinds[k]:8s} {self.routes.get(k, '')}".rstrip() for k in self.terms
        ]
        return "Objective with terms:\n" + "\n".join(rows)

    def function(self, outputs):
        """``f(point, params)`` evaluating graphs of the objective's inputs and parameters."""
        values = [v for v in self.model.value_vars if v in ancestors(outputs)]
        ps = [p for p, _ in self.params.values()]
        fn = pytensor.function([*values, *ps], outputs, on_unused_input="ignore")

        def call(point, params=None):
            self.check_training_data()
            params = params or {}
            try:
                return fn(*[point[v.name] for v in values], *[params[n] for n in self.params])
            except ValueError as err:
                self._explain_rows(outputs, err)
                raise

        return call

    # -- the fit fixes the training data

    def _training_data(self):
        used = set(ancestors(list(self.terms.values())))
        return [d for d in self.model.data_vars if d in used]

    def record_training_data(self):
        """Remember the training data, so prediction can refuse if it changes. Fitting calls this."""
        self.fitted_data = {
            d.name: np.array(d.get_value(), copy=True) for d in self._training_data()
        }

    def check_training_data(self):
        for name, value in self.fitted_data.items():
            now = self.model[name].get_value()
            if np.shape(now) != np.shape(value) or not np.array_equal(now, value):
                raise ValueError(
                    f"The training data {name} has changed since the fit. Predictions condition on the training "
                    "data, so set it back, or refit."
                )

    def _explain_rows(self, outputs, err):
        training = set(self._training_data())
        rows = {
            d.name: np.shape(d.get_value())[0]
            for d in self.model.data_vars
            if d in ancestors(outputs) and d not in training and np.ndim(d.get_value()) > 0
        }
        if len(set(rows.values())) > 1:
            raise ValueError(
                f"The prediction inputs have different numbers of rows: {rows}. "
                "Set them together, in one pm.set_data call."
            ) from err


class Builder:
    """What a family writes into while the objective is assembled."""

    def __init__(self, model, n_points, closed_form):
        self.model, self.n_points, self.closed_form = model, n_points, closed_form
        self.initial_point = model.initial_point()
        self.terms, self.kinds, self.routes, self.pointwise, self.diagnostics = {}, {}, {}, {}, {}
        self.params, self.gps, self.blocks, self.meanfield = {}, {}, [], {}
        self.q = {}  # value variable -> (mean, variance): every latent the likelihoods integrate over
        self.gp_values = set()
        self.handled = set()  # names of free variables that already have their terms
        self.rewriters = []  # graphs -> (graphs, moments): sums of GPs replaced by one variable

    def value(self, rv):
        return self.model.rvs_to_values[rv]

    def param(self, name, shape, init):
        """A new variational parameter: a symbolic variable and its initial value."""
        var = pt.tensor(name, shape=shape, dtype=pytensor.config.floatX)
        self.params[name] = (var, np.asarray(init, dtype=pytensor.config.floatX))
        return var

    def add(self, label, graph, kind="bound", route=None):
        self.terms[label], self.kinds[label] = graph, kind
        if route:
            self.routes[label] = route

    def rewrite(self, graphs):
        q = {}
        for fn in self.rewriters:
            graphs, extra = fn(graphs)
            q.update(extra)
        return graphs, q

    def logp(self, rv):
        """The log prior of a free variable, with its transform's Jacobian."""
        return pt.sum(self.model.logp(vars=[rv], jacobian=True, sum=False)[0])


def _families(model, approx, procs):
    from .families import GPFamily, Point

    gp_fams, var_fams, sums = {}, [], []
    for key, fam in approx.items():
        names = key if isinstance(key, tuple) else (key,)
        if isinstance(fam, GPFamily):
            for name in names:
                if name not in procs:
                    raise ValueError(
                        f"{name} is not a GP. GP families apply to GP processes, by the GP's name."
                    )
                gp_fams[name] = fam
            if len(names) > 1:  # a tuple of GPs: they enter a likelihood as a sum, fitted as one GP
                sums.append((names, fam))
        elif not isinstance(fam, Point):
            if bad := [n for n in names if n in procs]:
                raise ValueError(
                    f"{bad[0]} is a GP, and {type(fam).__name__} isn't a GP family. "
                    "Pass one of Exact, VFE, FITC, SVGP."
                )
            var_fams.append((names, fam))
    for name in procs:
        if name not in gp_fams:
            raise ValueError(
                f"No approximation for the GP {name}. Pass one of Exact, VFE, FITC, SVGP."
            )
    return gp_fams, var_fams, sums


def objective(model, approx, penalties=None, n_points=20, closed_form=True):
    """Read a fitting criterion from ``model``, term by term.

    Parameters
    ----------
    approx : dict
        GP name -> a GP family (``Exact``, ``VFE``, ``FITC``, ``SVGP``); a tuple of GP names -> one
        GP family, for GPs that are added together in a likelihood and are fitted as one GP;
        variable name, or a tuple of names, -> ``MeanField``, ``Gaussian`` or ``Point``. Unlisted
        variables are ``Point``.
    penalties : dict, optional
        Label -> ``callable(builder) -> scalar``: extra loss terms kept out of the bound.
    n_points : int
        Gauss-Hermite points for expectations that need quadrature.
    closed_form : bool
        Use closed-form expected log-likelihoods where they're known.
    """
    from .families import Gaussian, MeanField

    procs = {p.name: p for p in processes(model)}
    from .families import _SumMember

    gp_fams, var_fams, sums = _families(model, approx, procs)
    b = Builder(model, n_points, closed_form)

    # each GP's family: its parameters, its KL, and the moments of its observed values
    for name, gp in procs.items():
        if not gp.observed:
            raise ValueError(
                f"No likelihood reads the GP {name}. Evaluate it at the training inputs."
            )
        b.handled |= {rv.name for rv, _ in gp.observed}
    groups = []
    for names, fam in sums:
        group = fam.bind_sum(b, [procs[n] for n in names])
        b.gps[group.name] = group
        groups.append(group)
        for n in names:
            b.gps[n] = _SumMember(group, procs[n])
    for name, gp in procs.items():
        if name not in b.gps:
            b.gps[name] = gp_fams[name].bind(b, gp)

    # variables that exist only for prediction: unobserved evaluations and what depends on them
    evals = [rv for gp in procs.values() for rv, _ in gp.observed + gp.predictive]
    observed_evals = {rv for gp in procs.values() for rv, _ in gp.observed}
    predictive = [
        rv
        for rv in model.free_RVs
        if rv not in observed_evals
        and (rv in evals or set(evals) & set(ancestors(rv.owner.inputs)))
    ]
    b.handled |= {rv.name for rv in predictive}

    for names, fam in var_fams:
        if isinstance(fam, MeanField):
            fam.bind(b, names)

    # the likelihoods
    for y in model.observed_RVs:
        if bad := [p.name for p in predictive if p in _direct(y, set(predictive) | observed_evals)]:
            raise ValueError(
                f"{y.name} is observed but depends on the unobserved variable(s) {bad}. "
                "Observed data has to read the GP directly."
            )
        claimed = [s for s in b.gps.values() if s.claims(y)]
        if len(claimed) > 1:
            raise ValueError(
                f"{y.name} reads more than one GP that's integrated out exactly "
                f"({[s.gp.name for s in claimed]}). To fit a sum of GPs as one GP, give them one "
                "family as a tuple, e.g. {('f_a', 'f_b'): Exact()}."
            )
        if claimed:
            (state,) = claimed
            state.likelihood(b, y)
            continue
        ell, route = expected_loglik(model, y, b.q, b.gp_values, n_points, closed_form, b.rewrite)
        if ell is None:  # reads nothing that's approximated
            b.add(f"loglik[{y.name}]", pt.sum(model.logp(vars=[y], sum=False)[0]))
        else:
            b.pointwise[f"expected_loglik[{y.name}]"] = ell
            b.add(f"expected_loglik[{y.name}]", pt.sum(ell), route=route)

    for names, fam in var_fams:
        if isinstance(fam, Gaussian):
            fam.bind(b, names)
    for rv in model.free_RVs:  # everything else is a point estimate
        if rv.name not in b.handled:
            b.add(f"logp[{rv.name}]", b.logp(rv))
    for label, fn in (penalties or {}).items():
        b.add(f"penalty[{label}]", fn(b), kind="penalty")

    # model variables -> their values, then average over the Gaussian blocks' draws
    def finish(graphs):
        graphs = model.replace_rvs_by_values(list(graphs)) if graphs else []
        return [average_blocks(g, b.blocks) for g in graphs]

    terms = dict(zip(b.terms, finish(b.terms.values())))
    pointwise = dict(zip(b.pointwise, finish(b.pointwise.values())))
    diagnostics = {k: type(d)(*finish(d)) for k, d in b.diagnostics.items()}
    var_exp = [t for k, t in terms.items() if k.startswith("expected_loglik")]
    if var_exp:
        kl = [-t for k, t in terms.items() if k.startswith("kl[")]
        bound = pt.sum([t for k, t in terms.items() if b.kinds[k] == "bound"])
        diagnostics["elbo"] = pg.objectives.ELBOTerms(
            elbo=bound, var_exp=pt.sum(var_exp), kl=pt.sum(kl) if kl else pt.constant(0.0)
        )

    left = (
        set(b.q)
        | {v for block in b.blocks for v in block[0]}
        | {b.value(rv) for rv in observed_evals}
    )
    if left & value_ancestors(list(terms.values())):
        raise ValueError(
            "The objective still depends on an approximated variable's value. "
            "A latent reaches a likelihood in a way the expectation can't integrate."
        )
    inputs = [v for v in model.value_vars if v in ancestors(list(terms.values()))]
    return Objective(
        model,
        terms,
        b.kinds,
        b.routes,
        pointwise,
        diagnostics,
        b.params,
        inputs,
        b.gps,
        b.blocks,
        b.meanfield,
        n_points,
        rewriters=[g.rewrite for g in groups],
    )
