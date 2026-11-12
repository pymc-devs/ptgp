"""Families: how each variable is handled when a model is fitted. Each one builds its own terms.

GP families (``Exact``, ``VFE``, ``FITC``, ``SVGP``) are bound to one GP of the model, or to
several that enter a likelihood as a sum; the bound object adds the family's terms and knows the
GP's posterior, for prediction. ``MeanField`` and
``Gaussian`` apply to other variables. A new approximation is a new class here.
"""

import operator

from dataclasses import dataclass
from functools import reduce

import numpy as np
import pytensor.assumptions as pta
import pytensor.tensor as pt

from pytensor.graph.basic import equal_computations
from pytensor.graph.traversal import ancestors
from pytensor.tensor.random.basic import NormalRV

import ptgp as pg

from ptgp.conditionals import _DEFAULT_JITTER as _JITTER
from ptgp.conditionals import conditional_unwhitened, conditional_whitened
from ptgp.gp.svgp import _softplus_lower_triangular

from .expectations import substitute
from .model import Process, find_gp_sums


class GPFamily:
    """Base class of the GP families. ``bind(builder, gp)`` returns the family bound to ``gp``."""

    def bind(self, b, gp):
        raise NotImplementedError


def _inducing(b, name, Z, train):
    """A ptgp inducing-variable object and its size, from an array or from such an object."""
    if isinstance(Z, pg.inducing.InducingVariables):
        for i, (var, init) in enumerate(zip(Z.extra_vars, Z.extra_init)):  # its own trainable parts
            b.params[var.name or f"{name}_inducing_{i}"] = (var, np.asarray(init))
        M = Z.num_inducing
        if not isinstance(M, int):  # Points: read the number of rows of Z
            M = Z.Z.type.shape[0] if hasattr(Z.Z, "type") else Z.Z.shape[0]
        return Z, int(M)
    Z = np.asarray(Z, dtype=float)
    Z_var = b.param(f"{name}_Z", Z.shape, Z) if train else pt.as_tensor(Z)
    return pg.inducing.Points(Z_var), Z.shape[0]


# ---------------------------------------------------------------- GP integrated out analytically


class _Conjugate(GPFamily):
    """A GP integrated against one Normal likelihood whose mean is the GP."""

    def bind(self, b, gp):
        return _ConjugateGP(self, b, gp)

    def bind_sum(self, b, gps):
        """Bind to several GPs that enter one likelihood as a sum."""
        return _ConjugateSum(self, b, gps)


class _ConjugateGP:
    def __init__(self, family, b, gp):
        name = type(family).__name__
        if len(gp.observed) != 1:
            raise ValueError(
                f"{name} needs one observed evaluation of {gp.name}, and it has "
                f"{len(gp.observed)}. Use SVGP(Z) for a GP seen through several likelihoods."
            )
        self.family, self.gp, self.ptgp = family, gp, None
        ((self.f, self.rows),) = gp.observed

    def claims(self, y):
        return self.f in ancestors(y.owner.inputs)

    def likelihood(self, b, y):
        name = type(self.family).__name__
        if self.ptgp is not None:
            raise ValueError(
                f"{name} needs one likelihood on {self.f.name}, and {y.name} is a second one. Use SVGP(Z)."
            )
        if type(y.owner.op).__name__ == "MinibatchRandomVariable":
            raise ValueError(
                f"{name} integrates the GP against all of the data, and {y.name} is a minibatch "
                "(it has total_size). Use SVGP(Z) to train on minibatches."
            )
        mu, sigma = (
            y.owner.op.dist_params(y.owner) if isinstance(y.owner.op, NormalRV) else (None, None)
        )
        if mu is not self.f:
            raise ValueError(
                f"{name} needs a Normal likelihood whose mean is {self.f.name} itself, and {y.name} "
                "isn't one. Use SVGP(Z) for other likelihoods, or give a sum of GPs one family as a tuple."
            )
        if any(rv in ancestors([sigma]) for s in b.gps.values() for rv, _ in s.gp.observed):
            raise ValueError(
                f"{y.name}'s sigma depends on a GP, so {name} doesn't apply. Use SVGP(Z)."
            )
        self.y_value = b.value(y)
        self.ptgp = self.family.build(b, self.gp, sigma)
        label, diagnostics = self.family.terms(self.ptgp, self.rows, self.y_value)
        b.add(f"{label}[{y.name}]", diagnostics[0])
        b.diagnostics[self.gp.name] = diagnostics

    def moments(self, X):
        """Posterior mean and variance of the GP at the rows of ``X``."""
        return self.ptgp.predict_marginal(X, self.rows, self.y_value)


# ---------------------------------------------------------------- a sum of GPs, handled as one GP


class _GPSum:
    """Several GPs that enter a likelihood as a sum, handled as one GP.

    ``offset + sum_k c_k f_k(X)`` with independent ``f_k ~ GP(m_k, k_k)`` is ``offset`` plus one GP
    with mean ``sum_k c_k m_k`` and kernel ``sum_k c_k^2 k_k``. The sum is found by matching the
    ``GPRV`` nodes in the likelihood's graph; nothing in the model says the GPs belong together.
    """

    def __init__(self, family, b, gps):
        self.family, self.members = family, {g.name: g for g in gps}
        self.name = "+".join(self.members)
        self.evals = {rv: g.name for g in gps for rv, _ in g.observed + g.predictive}
        self.observed = {rv for g in gps for rv, _ in g.observed}
        self.gp = Process(self.name, gps[0].kernel, None, [], [])
        self.ptgp = None

    def _set(self, match):
        """Take the coefficients from a matched sum, and build the sum's kernel and mean."""
        self.coefs = {n: c for n, (c, _) in match.terms.items()}
        self.parts = {
            n: (g.kernel if self.coefs[n] is None else self.coefs[n] ** 2 * g.kernel)
            for n, g in self.members.items()
        }
        means = [(self.coefs[n], g.mean) for n, g in self.members.items()]
        mean = lambda A: sum(f(A) if c is None else c * f(A) for c, f in means)  # noqa: E731
        self.gp = Process(self.name, reduce(operator.add, self.parts.values()), mean, [], [])

    def _same(self, match):
        """Whether a matched sum has the coefficients this one was built with."""
        for n, (c, _) in match.terms.items():
            c0 = self.coefs[n]
            if (c is None) != (c0 is None) or (c is not None and not equal_computations([c], [c0])):
                return False
        return True

    def claims(self, y):
        return False

    def _part(self, X, part):
        """Posterior mean and variance at ``X`` of the zero-mean part of the sum with kernel ``part``."""
        raise NotImplementedError

    def member_moments(self, name, X):
        """Posterior mean and variance of one of the GPs at ``X``, read off the sum's posterior."""
        mean, var = self._part(X, self.parts[name])
        c = self.coefs[name]
        prior_mean = self.members[name].mean(X)
        return (prior_mean + mean, var) if c is None else (prior_mean + mean / c, var / c**2)

    def rewrite(self, graphs):
        """Replace every sum of these GPs in ``graphs`` by one variable, and give its moments.

        The GPs' posteriors are correlated, so anything that reads their sum has to use the sum's
        posterior, not the parts' separately.
        """
        q = {}
        for m in find_gp_sums(graphs, self.evals):
            if not self._same(m):
                raise ValueError(
                    f"The GPs {self.name} are fitted as one GP, and here they're added with different "
                    "coefficients than in the likelihood they were fitted through. Use the same ones."
                )
            S = pt.vector(f"_{self.name}")
            q[S] = self.moments(m.rows)
            graphs = substitute(graphs, {m.var: m.offset + S})
        return graphs, q


class _SumMember:
    """One GP of a sum: its own posterior, read off the sum's."""

    def __init__(self, group, gp):
        self.group, self.gp, self.family = group, gp, group.family

    @property
    def ptgp(self):
        return self.group.ptgp

    def claims(self, y):
        return False

    def moments(self, X):
        return self.group.member_moments(self.gp.name, X)


class _ConjugateSum(_GPSum):
    """A conjugate family over GPs that enter one Normal likelihood as a sum: integrated out as one GP."""

    def claims(self, y):
        return any(
            set(rv for _, rv in m.terms.values()) <= self.observed
            for m in find_gp_sums(list(y.owner.inputs), self.evals)
        )

    def likelihood(self, b, y):
        name = type(self.family).__name__
        if self.ptgp is not None:
            raise ValueError(
                f"{name} needs one likelihood on the sum {self.name}, and {y.name} is a second one."
            )
        if type(y.owner.op).__name__ == "MinibatchRandomVariable":
            raise ValueError(
                f"{name} integrates the GPs against all of the data, and {y.name} is a minibatch "
                "(it has total_size). Use SVGP(Z) to train on minibatches."
            )
        if not isinstance(y.owner.op, NormalRV):
            raise ValueError(
                f"{name} on the sum {self.name} needs a Normal likelihood, and {y.name} isn't one. "
                "Use SVGP(Z) instead."
            )
        mu, sigma = y.owner.op.dist_params(y.owner)
        matches = find_gp_sums([mu], self.evals)
        if len(matches) != 1 or matches[0].var is not mu:
            raise ValueError(
                f"{name} on the sum {self.name} needs {y.name}'s mean to be a sum of those GPs at the "
                "same inputs (each times a scalar at most), plus terms that don't involve a GP."
            )
        if any(rv in ancestors([sigma]) for rv in self.evals):
            raise ValueError(f"{y.name}'s sigma depends on a GP, so {name} doesn't apply.")
        (m,) = matches
        self._set(m)
        self.rows = m.rows
        self.y_value = b.value(y) - m.offset
        self.ptgp = self.family.build(b, self.gp, sigma)
        label, diagnostics = self.family.terms(self.ptgp, self.rows, self.y_value)
        b.add(f"{label}[{y.name}]", diagnostics[0])
        b.diagnostics[self.name] = diagnostics

    def moments(self, X):
        """Posterior mean and variance of the sum of the GPs (without the offset) at ``X``."""
        return self.ptgp.predict_marginal(X, self.rows, self.y_value)

    def _part(self, X, part):
        # The part and the data are jointly Gaussian with Cov(part(X), y) = part(X, X_train): the usual
        # GP posterior with the part's kernel in the cross-covariance and prior variance.
        gp, rows = self.ptgp, self.rows
        resid = self.y_value - gp.mean(rows)
        sigma = gp.likelihood.sigma * pt.ones(rows.shape[0])
        if isinstance(gp, pg.gp.Unapproximated):
            Knn = pta.assume(
                gp.kernel(rows) + pt.diag(sigma**2), positive_definite=True, symmetric=True
            )
            Kns = part(rows, X)
            return Kns.T @ pt.linalg.solve(Knn, resid, assume_a="pos"), part.diag(X) - pt.sum(
                Kns * pt.linalg.solve(Knn, Kns, assume_a="pos"), axis=0
            )
        Z = gp.inducing_variable.Z  # sparse: the same factored form as ptgp's VFE prediction
        M = Z.shape[0]
        Kuu = pta.assume(gp.kernel(Z) + _JITTER * pt.eye(M), positive_definite=True, symmetric=True)
        Lu = pt.linalg.cholesky(Kuu)
        B = pt.linalg.solve_triangular(Lu, gp.kernel(Z, rows), lower=True) / sigma[None, :]
        Aus = pt.linalg.solve_triangular(Lu, part(Z, X), lower=True)
        inner = pta.assume(pt.eye(M) + B @ B.T, positive_definite=True, symmetric=True)
        mean = Aus.T @ pt.linalg.solve(inner, B @ (resid / sigma), assume_a="pos")
        var = (
            part.diag(X)
            - pt.sum(Aus * Aus, axis=0)
            + pt.sum(Aus * pt.linalg.solve(inner, Aus, assume_a="pos"), axis=0)
        )
        return mean, var


@dataclass
class Exact(_Conjugate):
    """Integrate the GP out exactly: the log marginal likelihood. Cost O(N^3)."""

    def build(self, b, gp, sigma):
        return pg.gp.Unapproximated(kernel=gp.kernel, mean=gp.mean, sigma=sigma)

    def terms(self, obj_gp, X, y):
        return "mll", pg.objectives.unapproximated_diagnostics(obj_gp, X, y)


@dataclass
class VFE(_Conjugate):
    """Titsias' collapsed lower bound. ``Z`` is an array of inducing inputs (trained unless
    ``train_Z=False``) or a ``ptgp.inducing.Points`` object."""

    Z: object
    train_Z: bool = True

    def build(self, b, gp, sigma):
        inducing, _ = _inducing(b, gp.name, self.Z, self.train_Z)
        return pg.gp.VFE(kernel=gp.kernel, mean=gp.mean, sigma=sigma, inducing_variable=inducing)

    def terms(self, obj_gp, X, y):
        return "collapsed_elbo", pg.objectives.vfe_diagnostics(obj_gp, X, y)


@dataclass
class FITC(VFE):
    """The FITC approximation to the marginal likelihood (not a bound)."""

    def terms(self, obj_gp, X, y):
        return "fitc", pg.objectives.fitc_log_marginal_likelihood(obj_gp, X, y)


# ---------------------------------------------------------------- GP with an explicit q(u)


@dataclass
class SVGP(GPFamily):
    """q(u) = N(q_mu, q_sqrt q_sqrt^T) at inducing variables; any likelihoods, minibatches.

    ``Z`` is an array of inducing inputs, or a ptgp inducing-variable object such as
    ``ptgp.inducing.FourierFeatures1D`` (which then says what's trainable itself).
    """

    Z: object
    train_Z: bool = False
    whiten: bool = True

    def bind(self, b, gp):
        return _SVGPGP(self, b, gp)

    def bind_sum(self, b, gps):
        """Bind to several GPs that enter the likelihoods as a sum: one q(u) for the sum."""
        return _SVGPSum(self, b, gps)


def _svgp(b, family, gp):
    """ptgp's SVGP for ``gp``, with its variational parameters and its KL term added to ``b``."""
    inducing, M = _inducing(b, gp.name, family.Z, family.train_Z)
    q_mu = b.param(f"{gp.name}_q_mu", (M,), np.zeros(M))
    q_flat = b.param(
        f"{gp.name}_q_sqrt", (M * (M + 1) // 2,), pg.gp.init_variational_params(M).extra_init[1]
    )
    q_sqrt = pta.assume(_softplus_lower_triangular(q_flat, M), lower_triangular=True)
    out = pg.gp.SVGP(
        kernel=gp.kernel,
        mean=gp.mean,
        likelihood=None,
        inducing_variable=inducing,
        variational_params=pg.gp.VariationalParams(q_mu=q_mu, q_sqrt=q_sqrt),
        whiten=family.whiten,
    )
    b.add(f"kl[{gp.name}]", -out.prior_kl())
    return out


class _SVGPGP:
    def __init__(self, family, b, gp):
        self.family, self.gp = family, gp
        self.ptgp = _svgp(b, family, gp)
        # one latent over the stacked inputs of every observed evaluation, split back per evaluation
        rows = [r for _, r in gp.observed]
        mean, var = self.ptgp.predict_marginal(
            rows[0] if len(rows) == 1 else pt.concatenate(rows, axis=0)
        )
        start = 0
        for (rv, _), r in zip(gp.observed, rows):
            n = r.shape[0]
            b.q[b.value(rv)] = (mean[start : start + n], var[start : start + n])
            b.gp_values.add(b.value(rv))
            start = start + n

    def claims(self, y):
        return False

    def moments(self, X):
        """Posterior mean and variance of the GP at the rows of ``X``."""
        return self.ptgp.predict_marginal(X)


class _SVGPSum(_GPSum):
    """SVGP over GPs that enter the likelihoods as a sum: one set of inducing variables and one
    q(u) for the sum. Any likelihood, and minibatches."""

    def __init__(self, family, b, gps):
        super().__init__(family, b, gps)
        found = [
            m
            for m in find_gp_sums(list(b.model.observed_RVs), self.evals)
            if {rv for _, rv in m.terms.values()} <= self.observed
        ]
        if not found:
            raise ValueError(
                f"No likelihood reads the GPs {self.name} as a sum at the same inputs. "
                "Give each GP its own SVGP(Z) instead."
            )
        self._set(found[0])
        if not all(self._same(m) for m in found):
            raise ValueError(
                f"The likelihoods add the GPs {self.name} with different coefficients. "
                "Give each GP its own SVGP(Z) instead."
            )
        self.ptgp = _svgp(b, family, self.gp)
        b.rewriters.append(self.rewrite)

    def moments(self, X):
        """Posterior mean and variance of the sum of the GPs (without the offset) at ``X``."""
        return self.ptgp.predict_marginal(X)

    def _part(self, X, part):
        # Cov(part(X), u) = part(X, Z) for the sum's inducing values u: ptgp's conditional with the
        # part's kernel in the cross-covariance and prior variance.
        gp = self.ptgp
        ind, Kmn, Knn = gp.inducing_variable, gp.inducing_variable.K_uf(part, X), part.diag(X)
        if gp.whiten:
            return conditional_whitened(ind.Kuu_sqrt_solve(gp.kernel, Kmn), Knn, gp.q_mu, gp.q_sqrt)
        return conditional_unwhitened(ind.Kuu_solve(gp.kernel, Kmn), Kmn, Knn, gp.q_mu, gp.q_sqrt)


# ---------------------------------------------------------------- other variables


@dataclass
class MeanField:
    """q = N(m, s^2) per element of a variable with a Normal prior; analytic KL."""

    init_sd: float = 0.1

    def bind(self, b, names):
        for name in names:
            rv = b.model[name]
            if not isinstance(rv.owner.op, NormalRV) or b.model.rvs_to_transforms[rv] is not None:
                raise ValueError(
                    f"MeanField needs an untransformed Normal prior, and {name} isn't one."
                )
            value = b.value(rv)
            start = b.initial_point[value.name]
            m = b.param(f"{name}_m", value.type.shape, start)
            log_s = b.param(
                f"{name}_log_s", value.type.shape, np.full(np.shape(start), np.log(self.init_sd))
            )
            s2 = pt.exp(2.0 * log_s)
            b.q[value] = b.meanfield[value] = (m, s2)
            mu0, sd0 = rv.owner.op.dist_params(rv.owner)
            kl = pt.sum(pt.log(sd0) - log_s + (s2 + (m - mu0) ** 2) / (2.0 * sd0**2) - 0.5)
            b.add(f"kl[{name}]", -kl)
            b.handled.add(name)


@dataclass
class Gaussian:
    """A mean-field Gaussian over a group of variables (in their unconstrained space), with its
    expectations averaged over ``draws`` fixed standard-normal draws, and its entropy."""

    draws: int = 16
    seed: int = 0
    init_sd: float = 0.1

    def bind(self, b, names):
        rvs = [b.model[n] for n in names]
        values = [b.value(rv) for rv in rvs]
        starts = [np.atleast_1d(b.initial_point[v.name]).ravel() for v in values]
        d = sum(s.size for s in starts)
        label = "+".join(names)
        loc = b.param(f"{label}_loc", (d,), np.concatenate(starts))
        logsd = b.param(f"{label}_logsd", (d,), np.full(d, np.log(self.init_sd)))
        b.blocks.append(
            (values, loc, logsd, np.random.default_rng(self.seed).standard_normal((self.draws, d)))
        )
        for rv in rvs:
            b.add(f"logp[{rv.name}]", b.logp(rv))
        b.add(f"entropy[{label}]", pt.sum(logsd) + 0.5 * d * (1.0 + np.log(2 * np.pi)))
        b.handled |= set(names)


@dataclass
class Point:
    """A point estimate (MAP): the variable's log prior. The default for unlisted variables."""
