"""Diagnostic: print the compiled-graph op breakdown for each GP model.

For each of ``Unapproximated``, ``VFE``, ``SVGP``, builds the joint graph
(loss + all gradients), compiles it under three configurations, and prints
the op counts side-by-side:

- **pre-rewrite**: counted on the symbolic graph before any compilation.
- **ptgp on**: post-``fast_run``, with ptgp's full set of rewrites and
  assumption rules active.
- **ptgp off (baseline)**: post-``fast_run``, with ptgp's rules stripped
  from the assumption registry and ptgp's structural rewrites excluded
  from the optdb. Approximates "what PyTensor alone would do".

Also prints the max difference between ptgp-on and ptgp-off output
values across all outputs, as a sanity check that the rewrites preserve
numerical semantics.

Run from the project root with the project's ``ptgp`` env active:

    python scripts/joint_graph_analysis.py

This script is **not** part of the test suite — it's a diagnostic tool
for investigating compiled-graph structure (e.g., when
``tests/test_cubic_floor.py`` fails or when evaluating new rewrites).
The cubic-op floor pinning lives in ``tests/test_cubic_floor.py``.
"""

import collections

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.assumptions.core import ASSUMPTION_INFER_REGISTRY
from pytensor.graph.traversal import ancestors
from pytensor.tensor.blockwise import Blockwise

import ptgp.rewrites as R

from ptgp.gp import SVGP, VFE, Unapproximated, init_variational_params
from ptgp.gp.svgp import _matrix_to_softplus_flat_init
from ptgp.inducing import Points
from ptgp.kernels import ExpQuad
from ptgp.likelihoods import Gaussian
from ptgp.mean import Zero
from ptgp.objectives import collapsed_elbo, elbo, marginal_log_likelihood

CUBIC_OPS = {"Cholesky", "MatrixInverse", "Solve", "LUFactor", "SLogDet", "Det"}
TRACKED_OPS = [
    "Cholesky",
    "CholeskySolve",
    "SolveTriangular",
    "Solve",
    "MatrixInverse",
    "LUFactor",
    "SLogDet",
    "Det",
]


def _op_name(op):
    return type(op.core_op).__name__ if isinstance(op, Blockwise) else type(op).__name__


def _count_symbolic(outs):
    c = collections.Counter()
    for v in ancestors(outs):
        if v.owner is not None:
            c[_op_name(v.owner.op)] += 1
    return c


def _count_compiled(fn):
    return collections.Counter(_op_name(n.op) for n in fn.maker.fgraph.apply_nodes)


def _cubic_total(counts):
    return sum(counts.get(k, 0) for k in CUBIC_OPS)


# ---- ptgp on/off control ----

_ptgp_funcs = {
    getattr(R, n)
    for n in dir(R)
    if callable(getattr(R, n)) and getattr(getattr(R, n), "__module__", None) == "ptgp.rewrites"
}
_snap = {k: list(v) for k, v in ASSUMPTION_INFER_REGISTRY.items()}


def _disable_ptgp():
    for k, fns in ASSUMPTION_INFER_REGISTRY.items():
        ASSUMPTION_INFER_REGISTRY[k] = [f for f in fns if f not in _ptgp_funcs]


def _restore_ptgp():
    for k in list(ASSUMPTION_INFER_REGISTRY.keys()):
        del ASSUMPTION_INFER_REGISTRY[k]
    for k, v in _snap.items():
        ASSUMPTION_INFER_REGISTRY[k] = list(v)


_PTGP_OPTDB_REWRITES = (
    "merge_composites_with_shared_inputs",
    "merge_after_composite_dedup",
)


def _mode_ptgp_off():
    return pytensor.compile.mode.get_default_mode().excluding(*_PTGP_OPTDB_REWRITES)


# ---- Model graph builders ----


def build_unapproximated():
    X = pt.dmatrix("X")
    y = pt.dvector("y")
    sigma = pt.dscalar("sigma")
    ls = pt.dscalar("ls")
    gp = Unapproximated(kernel=ExpQuad(input_dim=1, ls=ls), mean=Zero(), sigma=sigma)
    loss = marginal_log_likelihood(gp, X, y)
    g_sigma, g_ls = pt.grad(loss, [sigma, ls])
    return [X, y, sigma, ls], [loss, g_sigma, g_ls]


def build_vfe(M=8):
    X = pt.dmatrix("X")
    y = pt.dvector("y")
    sigma = pt.dscalar("sigma")
    ls = pt.dscalar("ls")
    Z = pt.dmatrix("Z")
    vfe = VFE(
        kernel=ExpQuad(input_dim=1, ls=ls),
        mean=Zero(),
        sigma=sigma,
        inducing_variable=Points(Z),
    )
    loss = -collapsed_elbo(vfe, X, y)
    g_sigma, g_ls, g_Z = pt.grad(loss, [sigma, ls, Z])
    return [X, y, sigma, ls, Z], [loss, g_sigma, g_ls, g_Z]


def build_svgp(M=8):
    X = pt.dmatrix("X")
    y = pt.dvector("y")
    sigma = pt.dscalar("sigma")
    ls = pt.dscalar("ls")
    Z = pt.dmatrix("Z")
    vp = init_variational_params(M)
    svgp = SVGP(
        kernel=ExpQuad(input_dim=1, ls=ls),
        mean=Zero(),
        likelihood=Gaussian(sigma),
        inducing_variable=Points(Z),
        variational_params=vp,
    )
    loss = -elbo(svgp, X, y)
    g_sigma, g_ls, g_Z, g_q_mu, g_q_sqrt = pt.grad(loss, [sigma, ls, Z, vp.q_mu, vp.extra_vars[1]])
    return (
        [X, y, sigma, ls, Z, vp.q_mu, vp.extra_vars[1]],
        [loss, g_sigma, g_ls, g_Z, g_q_mu, g_q_sqrt],
    )


# ---- Per-model analysis ----


def _sample_inputs(name, N=20, M=8):
    rng = np.random.default_rng(0)
    if name == "Unapproximated":
        return [rng.standard_normal((N, 1)), rng.standard_normal(N), 0.5, 1.2]
    if name == "VFE":
        return [
            rng.standard_normal((N, 1)),
            rng.standard_normal(N),
            0.5,
            1.2,
            rng.standard_normal((M, 1)),
        ]
    # SVGP
    flat_init = _matrix_to_softplus_flat_init(np.eye(M), M)
    return [
        rng.standard_normal((N, 1)),
        rng.standard_normal(N),
        0.5,
        1.2,
        rng.standard_normal((M, 1)),
        np.zeros(M),
        flat_init,
    ]


def analyze(name, builder):
    print(f"\n{'=' * 80}\n  {name}\n{'=' * 80}")

    inputs, outputs = builder()
    pre = _count_symbolic(outputs)

    inputs, outputs = builder()
    fn_on = pytensor.function(inputs, outputs)
    on = _count_compiled(fn_on)

    _disable_ptgp()
    try:
        inputs, outputs = builder()
        fn_off = pytensor.function(inputs, outputs, mode=_mode_ptgp_off())
        off = _count_compiled(fn_off)
    finally:
        _restore_ptgp()

    header = ["state", *TRACKED_OPS, "cubic", "Dot", "Sum", "ExtractDiag"]
    rows = [
        ("pre-rewrite", pre),
        ("ptgp on", on),
        ("ptgp off (baseline)", off),
    ]
    col_w = [22] + [6] * len(TRACKED_OPS) + [5, 5, 5, 11]
    print()
    print("  " + " ".join(f"{h:>{w}}" for h, w in zip(header, col_w)))
    print("  " + "-" * (sum(col_w) + len(col_w)))
    for label, counts in rows:
        cubic = _cubic_total(counts)
        dot = counts.get("Dot", 0) + counts.get("Dot22", 0)
        s = counts.get("Sum", 0)
        ed = counts.get("ExtractDiag", 0)
        vals = [counts.get(k, 0) for k in TRACKED_OPS] + [cubic, dot, s, ed]
        print("  " + f"{label:>22} " + " ".join(f"{v:>{w}}" for v, w in zip(vals, col_w[1:])))

    vals = _sample_inputs(name)
    on_vals = fn_on(*vals)
    off_vals = fn_off(*vals)
    diffs = [
        float(np.max(np.abs(np.atleast_1d(a) - np.atleast_1d(b))))
        for a, b in zip(on_vals, off_vals)
    ]
    print()
    print(f"  numerical: max |ptgp_on - ptgp_off| across all outputs = {max(diffs):.2e}")


if __name__ == "__main__":
    analyze("Unapproximated", build_unapproximated)
    analyze("VFE", build_vfe)
    analyze("SVGP", build_svgp)
