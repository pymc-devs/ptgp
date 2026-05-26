"""PTGP-local PyTensor rewrites and assumption rules.

Registered into PyTensor's global registries at import time. See REWRITES.md.

Scope after the 2026-05 ablation (audits/rewrites_ablation_plan.md): only the
rules that the empirical probe showed are load-bearing and not yet covered by
upstream pytensor remain here. The deleted rules (POSITIVE assumption key and
its consumers, ``slogdet_specialize``, ``_set_subtensor_psd``, the two
``Composite``-merge passes) were either dead code in pytensor 3.0.3 or were
subsumed by an explicit ``pta.assume(K, positive_definite=True, symmetric=True)``
annotation on ``marginal_log_likelihood``'s ``K``.
"""

import pytensor.tensor as pt

from pytensor.assumptions.core import (
    FactState,
    check_assumption,
    register_assumption,
)
from pytensor.assumptions.positive_definite import POSITIVE_DEFINITE
from pytensor.assumptions.triangular import LOWER_TRIANGULAR
from pytensor.graph.rewriting.basic import (
    copy_stack_trace,
    node_rewriter,
)
from pytensor.tensor.basic import ExtractDiag
from pytensor.tensor.blas import Dot22
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.solvers.general import Solve
from pytensor.tensor.linalg.solvers.psd import CholeskySolve, cho_solve
from pytensor.tensor.linalg.solvers.triangular import solve_triangular
from pytensor.tensor.linalg.summary import Det
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.basic import register_specialize
from pytensor.tensor.rewriting.blockwise import blockwise_of


# ---------------------------------------------------------------------------
# Helpers shared across the rules below.
# ---------------------------------------------------------------------------


def _matrix_transpose_of(var):
    """Return the underlying ``X`` if ``var`` is ``X.T``, else None.

    Works for both 2D and N-D tensors (matrix transpose = swap of last two axes).
    """
    owner = var.owner
    if owner is None:
        return None
    if isinstance(owner.op, DimShuffle) and owner.op.is_matrix_transpose:
        return owner.inputs[0]
    return None


def _unwrap_blockwise(op):
    """Unwrap ``Blockwise`` to the core op; pass other ops through."""
    return op.core_op if isinstance(op, Blockwise) else op


def _core_op_of(var):
    """Return ``var.owner.op``'s core op (unwrapping ``Blockwise``); None if no owner."""
    if var.owner is None:
        return None
    return _unwrap_blockwise(var.owner.op)


def _matches_core_op(var, *op_classes):
    """If ``var``'s core op is an instance of any in ``op_classes``, return that op; else None."""
    op = _core_op_of(var)
    return op if isinstance(op, op_classes) else None


def _try_AAT_factor(fgraph, M, lower_only=False):
    """If ``M = A @ A.T`` or ``M = A.T @ A`` for some matrix ``A``, return ``(A, form)``.

    ``form`` is ``"AAT"`` or ``"ATA"``. Recognizes both 2-D ``Dot``/``Dot22`` and
    Blockwise-wrapped versions, so batched matmul (``(B, N, K) @ (B, K, N)``)
    works without extra handling.

    If ``lower_only=True``, only return matches where ``A`` is annotated
    ``LOWER_TRIANGULAR`` — required by the slogdet/det/inverse fast paths.
    """
    if not isinstance(_core_op_of(M), Dot | Dot22):
        return None
    a, b = M.owner.inputs
    if _matrix_transpose_of(b) is a:
        if lower_only and not check_assumption(fgraph, a, LOWER_TRIANGULAR):
            return None
        return a, "AAT"
    if _matrix_transpose_of(a) is b:
        if lower_only and not check_assumption(fgraph, b, LOWER_TRIANGULAR):
            return None
        return b, "ATA"
    return None


def _existing_cholesky(fgraph, A):
    """Return an existing ``Cholesky(lower=True)(A)`` output already in *fgraph*, else None.

    Lets the inverse rewrite share a factor produced by an upstream Solve lowering
    instead of computing a second one.
    """
    for client, _ in fgraph.clients.get(A, ()):
        core_op = _unwrap_blockwise(client.op)
        if isinstance(core_op, Cholesky) and core_op.lower:
            return client.outputs[0]
    return None


# ---------------------------------------------------------------------------
# PSD recognition for ``X.T @ M^{-1} @ X`` expressed via ``Solve`` / ``CholeskySolve``.
# ``match_congruence`` in upstream only walks ``Dot`` nodes, so it cannot see the
# ``Solve`` / ``CholeskySolve`` acting as ``M^{-1}``. These two clauses fill that gap.
# ---------------------------------------------------------------------------


@register_assumption(POSITIVE_DEFINITE, Dot)
def _dot_xt_solve_x_psd(key, op, feature, fgraph, node, input_states):
    """``X.T @ Solve(M, X)`` ≡ ``X.T @ M^{-1} @ X`` is PSD when ``M`` is PSD."""
    a, b = node.inputs
    X = _matrix_transpose_of(a)
    if X is None or _matches_core_op(b, Solve) is None:
        return [FactState.UNKNOWN]
    M, X2 = b.owner.inputs
    if X2 is X and feature.check(M, POSITIVE_DEFINITE):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


@register_assumption(POSITIVE_DEFINITE, Dot)
def _dot_xt_chosolve_x_psd(key, op, feature, fgraph, node, input_states):
    """``X.T @ CholeskySolve(L, X)`` ≡ ``X.T @ M^{-1} @ X`` (M = L @ L.T) is PSD."""
    a, b = node.inputs
    X = _matrix_transpose_of(a)
    if X is None or _matches_core_op(b, CholeskySolve) is None:
        return [FactState.UNKNOWN]
    L, X2 = b.owner.inputs
    if X2 is X and isinstance(_core_op_of(L), Cholesky):
        return [FactState.TRUE]
    return [FactState.UNKNOWN]


# ---------------------------------------------------------------------------
# Det(L @ L.T) -> (prod(diag(L)))**2 for lower-triangular L.
#
# Identity: det(L @ L.T) = det(L)**2 = (prod(diag(L)))**2 (always >= 0).
# Eliminates the standalone Det Apply that survives upstream's slogdet rewrites
# when Det is also referenced by the gradient pullback (det(M) * inv(M).T).
# ---------------------------------------------------------------------------


@register_specialize
@node_rewriter([blockwise_of(Det)])
def det_of_LLT_to_diag_product(fgraph, node):
    """``Det(L @ L.T)`` or ``Det(L.T @ L)`` -> ``(prod(diag(L)))**2`` for lower-triangular ``L``.

    Identity: ``det(L L.T) = det(L.T L) = det(L)**2 = (prod(diag(L)))**2`` (always >= 0).
    """
    [A] = node.inputs
    aat = _try_AAT_factor(fgraph, A, lower_only=True)
    if aat is None:
        return None
    L, _form = aat
    diag_L = pt.diagonal(L, axis1=-2, axis2=-1)
    new_det = (diag_L.prod(axis=-1) ** 2).astype(node.outputs[0].dtype)
    copy_stack_trace(node.outputs[0], new_det)
    return [new_det]


# ---------------------------------------------------------------------------
# ExtractDiag(A @ A.T) -> sum(A**2, axis=-1) for any matrix A.
#
# Identity: (A @ A.T)[i,i] = sum_k A[i,k]^2 = ||A_row_i||^2. Folding this lets
# pt.trace(L @ L.T) compile to ||L||_F^2 directly, eliminating the M^2-element
# materialization of the L @ L.T outer product just to take its trace.
# Generally useful — not GP-specific. No assumption needed on A.
# ---------------------------------------------------------------------------


@register_specialize
@node_rewriter([ExtractDiag])
def diag_of_AAT_to_row_norms_squared(fgraph, node):
    """Diagonal of ``A @ A.T`` (or ``A.T @ A``) lowers to elementwise norms of ``A``.

    - ``ExtractDiag(A @ A.T)`` -> ``sum(A**2, axis=-1)`` (row norms squared)
    - ``ExtractDiag(A.T @ A)`` -> ``sum(A**2, axis=-2)`` (column norms squared)

    Generic — no assumption needed on ``A``. Handles batched matmul via
    ``Blockwise(Dot)`` automatically (the AAT factor matcher unwraps Blockwise).
    """
    extract_op = node.op
    if extract_op.offset != 0:
        return None
    [A_AT] = node.inputs
    ndim = A_AT.type.ndim
    if ndim is None or ndim < 2:
        return None
    # ExtractDiag must select the trailing two axes (the matrix axes).
    axis1 = extract_op.axis1 % ndim
    axis2 = extract_op.axis2 % ndim
    if {axis1, axis2} != {ndim - 2, ndim - 1}:
        return None
    aat = _try_AAT_factor(fgraph, A_AT)
    if aat is None:
        return None
    A, form = aat
    sum_axis = -1 if form == "AAT" else -2
    new_diag = pt.sum(pt.sqr(A), axis=sum_axis).astype(node.outputs[0].dtype)
    copy_stack_trace(node.outputs[0], new_diag)
    return [new_diag]


# ---------------------------------------------------------------------------
# MatrixInverse(PSD A) -> cho_solve(L, eye), reusing an existing Cholesky if
# present. Avoids the redundant cubic factorisation that pt.grad(slogdet)
# triggers via its standalone MatrixInverse cotangent.
# ---------------------------------------------------------------------------


@register_specialize
@node_rewriter([blockwise_of(MatrixInverse)])
def matrix_inverse_specialize(fgraph, node):
    """``MatrixInverse(A)`` -> simplified form when ``A`` has structure.

    Three paths:
    - ``A = L @ L.T`` (lower-triangular ``L``): ``cho_solve((L, True), eye)``,
      using ``L`` directly — no fresh Cholesky.
    - ``A = L.T @ L`` (lower-triangular ``L``): ``inv(L) @ inv(L.T)`` via two
      ``solve_triangular`` calls — also reuses ``L`` directly.
    - Otherwise (generic PSD ``A``): ``cho_solve((Cholesky(A), True), eye)``,
      sharing an existing Cholesky if already in the graph.
    """
    [A] = node.inputs
    eye = pt.eye(A.shape[-1], dtype=A.dtype)

    aat = _try_AAT_factor(fgraph, A, lower_only=True)
    if aat is not None:
        L, form = aat
        if form == "AAT":
            inv_A = cho_solve((L, True), eye)
        else:  # ATA: inv(L.T @ L) = inv(L) @ inv(L.T)
            inv_A = solve_triangular(L, solve_triangular(L, eye, lower=True, trans="T"), lower=True)
    else:
        if not check_assumption(fgraph, A, POSITIVE_DEFINITE):
            return None
        L = _existing_cholesky(fgraph, A)
        if L is None:
            L = cholesky(A, lower=True)
        inv_A = cho_solve((L, True), eye)

    copy_stack_trace(node.outputs[0], inv_A)
    return [inv_A]
