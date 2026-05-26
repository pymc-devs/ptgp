"""Diagnostic: which cubic-factorization ops in each GP joint graph are in-place?

For each linalg op with an ``overwrite_a``/``overwrite_b`` attribute
(``Cholesky``, ``CholeskySolve``, ``Solve``, ``SolveTriangular``, etc.),
prints whether the inplace flag is set and — if not — whether it *could*
be set safely, or whether something legitimately blocks it.

A buffer is **eligible for inplace** when:

- It has only one consumer (this op).
- It is **not** an fgraph input (caller owns those — PyTensor refuses to
  overwrite them).
- It is **not** a ``Constant`` (overwriting would corrupt subsequent calls
  since Constants persist across function evaluations).
- It is **not** an fgraph output (would shadow the result the caller
  expects to read).

Cases marked **MISSED** are real opportunities the inplace pass didn't
take — typically because PyTensor's inplace analysis doesn't look through
``TypeCastingOp`` wrappers like ``SpecifyAssumptions`` to verify
aliasing.

Run from the project root with the project's ``ptgp`` env active:

    python scripts/inplace_audit.py

This script is **not** part of the test suite. For a stricter regression
check on inplace counts, write a test in ``tests/`` that walks the
compiled graph and asserts the count of ops with ``overwrite_a=True``
and ``overwrite_b=True``.
"""

import pytensor
import pytensor.tensor as pt

from pytensor.graph.basic import Constant
from pytensor.tensor.blockwise import Blockwise

from ptgp.gp import SVGP, VFE, Unapproximated, init_variational_params
from ptgp.inducing import Points
from ptgp.kernels import ExpQuad
from ptgp.likelihoods import Gaussian
from ptgp.mean import Zero
from ptgp.objectives import collapsed_elbo, elbo, marginal_log_likelihood

INPLACE_ATTR_NAMES = ("overwrite_a", "overwrite_b")


def _core(op):
    return op.core_op if isinstance(op, Blockwise) else op


def _input_status(fgraph, node, inp):
    """Return a one-word reason this input is or isn't eligible for inplace overwrite."""
    other_consumers = [c for c, _ in fgraph.clients.get(inp, ()) if c is not node]
    if other_consumers:
        return "blocked: shared with other consumers"
    if isinstance(inp, Constant):
        return "blocked: Constant (would corrupt subsequent calls)"
    if inp in fgraph.inputs:
        return "blocked: fgraph input (caller owns it)"
    if inp in fgraph.outputs:
        return "blocked: also an fgraph output"
    return "eligible"


def audit(name, fn):
    print(f"\n--- {name} ---")
    fgraph = fn.maker.fgraph
    found_any = False
    for node in fgraph.apply_nodes:
        core = _core(node.op)
        flags = {a: getattr(core, a, None) for a in INPLACE_ATTR_NAMES}
        flags = {k: v for k, v in flags.items() if v is not None}
        if not flags:
            continue
        found_any = True
        cls_name = type(core).__name__
        flag_str = ", ".join(f"{k}={v}" for k, v in flags.items())
        print(f"  {cls_name}({flag_str})")
        for attr_name, current in flags.items():
            # `overwrite_a` looks at input[0]; `overwrite_b` looks at input[1].
            input_idx = 0 if attr_name == "overwrite_a" else 1
            if input_idx >= len(node.inputs):
                continue
            inp = node.inputs[input_idx]
            origin = type(_core(inp.owner.op)).__name__ if inp.owner else "fgraph_input"
            status = _input_status(fgraph, node, inp)
            if current is True:
                marker = "✓ inplace"
            elif status == "eligible":
                marker = "✗ MISSED — could be inplace"
            else:
                marker = f"✗ {status}"
            print(f"    {attr_name}: input={origin}  →  {marker}")
    if not found_any:
        print("  (no linalg ops with overwrite_* attributes)")


# ---- Model graph builders ----


def build_unapproximated():
    X = pt.dmatrix("X")
    y = pt.dvector("y")
    sigma = pt.dscalar("sigma")
    ls = pt.dscalar("ls")
    gp = Unapproximated(kernel=ExpQuad(input_dim=1, ls=ls), mean=Zero(), sigma=sigma)
    loss = marginal_log_likelihood(gp, X, y)
    g_sigma, g_ls = pt.grad(loss, [sigma, ls])
    return pytensor.function([X, y, sigma, ls], [loss, g_sigma, g_ls])


def build_vfe():
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
    return pytensor.function([X, y, sigma, ls, Z], [loss, g_sigma, g_ls, g_Z])


def build_svgp():
    X = pt.dmatrix("X")
    y = pt.dvector("y")
    sigma = pt.dscalar("sigma")
    ls = pt.dscalar("ls")
    Z = pt.dmatrix("Z")
    vp = init_variational_params(8)
    svgp = SVGP(
        kernel=ExpQuad(input_dim=1, ls=ls),
        mean=Zero(),
        likelihood=Gaussian(sigma),
        inducing_variable=Points(Z),
        variational_params=vp,
    )
    loss = -elbo(svgp, X, y)
    g_sigma, g_ls, g_Z, g_q_mu, g_q_sqrt = pt.grad(loss, [sigma, ls, Z, vp.q_mu, vp.extra_vars[1]])
    return pytensor.function(
        [X, y, sigma, ls, Z, vp.q_mu, vp.extra_vars[1]],
        [loss, g_sigma, g_ls, g_Z, g_q_mu, g_q_sqrt],
    )


if __name__ == "__main__":
    audit("Unapproximated", build_unapproximated())
    audit("VFE", build_vfe())
    audit("SVGP", build_svgp())
