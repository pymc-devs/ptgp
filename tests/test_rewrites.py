"""Tests for the structural rewrites in ``ptgp.rewrites``.

Written so that, when the rewrites are upstreamed into PyTensor, the test
bodies move over unchanged — only the side-effect import that installs the
rules would shift from ``ptgp`` to ``pytensor``.

After the 2026-05 ablation (see ``audits/rewrites_ablation_plan.md``) the
scope of ``ptgp.rewrites`` shrank to three structural rewrites: H
(``Det(L @ L.T)``), I (``diag(A @ A.T)``), and J (``MatrixInverse(PSD A)``).
"""

import numpy as np
import pytensor.assumptions as pta
import pytensor.tensor as pt

from pytensor import function
from pytensor.graph import rewrite_graph
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.basic import ExtractDiag
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.summary import Det

# Install the rewrites under test (side-effect import).
import ptgp.rewrites  # noqa: F401

# ---------------------------------------------------------------------------
# Helpers for the structural rewrites below.
# ---------------------------------------------------------------------------


def _has_op(graph, op_type):
    fg = FunctionGraph(outputs=[graph] if not isinstance(graph, list) else graph, clone=False)
    for node in fg.apply_nodes:
        op = node.op
        core = op.core_op if isinstance(op, Blockwise) else op
        if isinstance(core, op_type):
            return True
    return False


# ---------------------------------------------------------------------------
# Det(L @ L.T) -> (prod(diag(L)))**2 rewrite (Rule H).
# ---------------------------------------------------------------------------


def test_det_of_LLT_takes_diag_product_when_L_lower_triangular():
    """``Det(L @ L.T)`` lowers to ``(prod(diag(L)))**2`` — no Det Apply remains."""
    from pytensor.tensor.linalg.summary import det

    L = pta.assume(pt.dmatrix("L"), lower_triangular=True)
    d = det(L @ L.T)
    rewritten = rewrite_graph(d, include=("fast_run",))
    assert not _has_op(rewritten, Det)


def test_det_of_LLT_diag_product_is_numerically_correct():
    from pytensor.tensor.linalg.summary import det

    L = pta.assume(pt.dmatrix("L"), lower_triangular=True)
    d = det(L @ L.T)
    f = function([L.owner.inputs[0]], d)

    rng = np.random.default_rng(0)
    M = 5
    L_val = np.tril(rng.standard_normal((M, M)))
    L_val[np.arange(M), np.arange(M)] = np.abs(L_val[np.arange(M), np.arange(M)]) + 0.5
    expected = float(np.linalg.det(L_val @ L_val.T))
    np.testing.assert_allclose(f(L_val), expected, atol=1e-10)


# ---------------------------------------------------------------------------
# ExtractDiag(A @ A.T) -> sum(A**2, axis=-1) rewrite (Rule I).
# ---------------------------------------------------------------------------


def test_diag_of_AAT_to_row_norms_squared_eliminates_dot():
    """``ExtractDiag(A @ A.T)`` lowers to ``sum(A**2, axis=-1)`` — no ExtractDiag."""
    A = pt.dmatrix("A")
    out = pt.diagonal(A @ A.T)
    rewritten = rewrite_graph(out, include=("fast_run",))
    assert not _has_op(rewritten, ExtractDiag)


def test_diag_of_AAT_is_numerically_correct():
    A = pt.dmatrix("A")
    out = pt.diagonal(A @ A.T)
    f = function([A], out)

    rng = np.random.default_rng(0)
    A_val = rng.standard_normal((6, 4))  # non-square — diagonal still defined for A @ A.T
    np.testing.assert_allclose(f(A_val), np.diag(A_val @ A_val.T), atol=1e-12)


def test_diag_of_ATA_lowers_to_column_norms():
    """``ExtractDiag(A.T @ A)`` lowers to ``sum(A**2, axis=-2)`` — no ExtractDiag."""
    A = pt.dmatrix("A")
    out = pt.diagonal(A.T @ A)
    rewritten = rewrite_graph(out, include=("fast_run",))
    assert not _has_op(rewritten, ExtractDiag)


def test_diag_of_ATA_is_numerically_correct():
    A = pt.dmatrix("A")
    out = pt.diagonal(A.T @ A)
    f = function([A], out)

    rng = np.random.default_rng(0)
    A_val = rng.standard_normal((6, 4))  # gives diag of (4, 4)
    np.testing.assert_allclose(f(A_val), np.diag(A_val.T @ A_val), atol=1e-12)


# ---------------------------------------------------------------------------
# MatrixInverse(PSD A) -> cho_solve(L, eye) rewrite (Rule J).
# ---------------------------------------------------------------------------


def test_matrix_inverse_of_LTL_uses_solve_triangular():
    """``MatrixInverse(L.T @ L)`` lowers to two solve_triangular calls — no fresh Cholesky."""
    L = pta.assume(pt.dmatrix("L"), lower_triangular=True)
    inv_M = pt.linalg.inv(L.T @ L)
    rewritten = rewrite_graph(inv_M, include=("fast_run",))
    assert not _has_op(rewritten, MatrixInverse)
    assert not _has_op(rewritten, Cholesky)


def test_matrix_inverse_of_LTL_is_numerically_correct():
    L = pta.assume(pt.dmatrix("L"), lower_triangular=True)
    inv_M = pt.linalg.inv(L.T @ L)
    f = function([L.owner.inputs[0]], inv_M)

    rng = np.random.default_rng(0)
    M = 5
    L_val = np.tril(rng.standard_normal((M, M)))
    L_val[np.arange(M), np.arange(M)] = np.abs(L_val[np.arange(M), np.arange(M)]) + 0.5
    expected = np.linalg.inv(L_val.T @ L_val)
    np.testing.assert_allclose(f(L_val), expected, atol=1e-10)


def test_matrix_inverse_of_psd_is_lowered_to_cholesky():
    K = pta.assume(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    inv_K = pt.linalg.inv(K)
    rewritten = rewrite_graph(inv_K, include=("fast_run",))
    assert not _has_op(rewritten, MatrixInverse)
    assert _has_op(rewritten, Cholesky)


def test_matrix_inverse_without_psd_is_not_lowered():
    K = pt.dmatrix("K")  # no PSD annotation
    inv_K = pt.linalg.inv(K)
    rewritten = rewrite_graph(inv_K, include=("fast_run",))
    assert not _has_op(rewritten, Cholesky)
    assert _has_op(rewritten, MatrixInverse)


def test_matrix_inverse_reuses_existing_cholesky():
    """Sibling Solve(K, b) and inv(K) should share one Cholesky factor."""
    K = pta.assume(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    b = pt.dvector("b")
    x = pt.linalg.solve(K, b, assume_a="pos")
    inv_K = pt.linalg.inv(K)
    rewritten = rewrite_graph([x, inv_K], include=("fast_run",))
    fg = FunctionGraph(outputs=rewritten, clone=False)
    n_chol = sum(
        1
        for node in fg.apply_nodes
        if isinstance(node.op.core_op if isinstance(node.op, Blockwise) else node.op, Cholesky)
    )
    assert n_chol == 1, f"expected MatrixInverse to share Cholesky, got {n_chol} factorisations"


def test_matrix_inverse_lowering_is_numerically_correct():
    K = pta.assume(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    inv_K = pt.linalg.inv(K)
    f = function([K.owner.inputs[0]], inv_K)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    K_val = A @ A.T + np.eye(6)
    np.testing.assert_allclose(f(K_val), np.linalg.inv(K_val), atol=1e-10)


# ---------------------------------------------------------------------------
# Joint-graph cubic-op floor pinning (was: Blocker A / merge_composites_*).
# These pin the invariant the deleted composite-merge passes were claimed to
# enforce; they pass without those passes because pytensor 3.0.3's fusion
# pipeline already arrives at the floor.
# ---------------------------------------------------------------------------


def _gp_mll_like_joint():
    """Build the joint forward+gradient graph for ``Unapproximated + marginal_log_likelihood``."""
    from ptgp.gp import Unapproximated
    from ptgp.kernels import ExpQuad
    from ptgp.mean import Zero
    from ptgp.objectives import marginal_log_likelihood

    X = pt.dmatrix("X")
    y = pt.dvector("y")
    sigma = pt.dscalar("sigma")
    ls = pt.dscalar("ls")
    gp = Unapproximated(kernel=ExpQuad(input_dim=1, ls=ls), mean=Zero(), sigma=sigma)
    loss = marginal_log_likelihood(gp, X, y).mll
    g_sigma, g_ls = pt.grad(loss, [sigma, ls])
    return [sigma, ls, X, y], [loss, g_sigma, g_ls]


def _count_in_compiled(fn, op_type):
    return sum(
        1
        for n in fn.maker.fgraph.apply_nodes
        if isinstance(n.op.core_op if isinstance(n.op, Blockwise) else n.op, op_type)
    )


def test_gp_joint_graph_hits_one_cholesky():
    """Joint forward+gradient compilation should hit the floor: exactly one Cholesky."""
    inputs, outputs = _gp_mll_like_joint()
    fn = function(inputs, outputs)
    assert _count_in_compiled(fn, Cholesky) == 1
    assert _count_in_compiled(fn, MatrixInverse) == 0


def test_gp_joint_graph_numerically_correct():
    """Joint forward+gradient values match an analytic reference and finite-difference grads."""
    inputs, outputs = _gp_mll_like_joint()
    fn = function(inputs, outputs)

    rng = np.random.default_rng(0)
    N = 8
    Xv = rng.standard_normal((N, 1))
    yv = rng.standard_normal(N)
    sigma_v, ls_v = 0.5, 1.2

    sqd = (Xv[:, 0:1] - Xv[:, 0:1].T) ** 2
    K = np.exp(-0.5 * np.maximum(sqd / ls_v**2, 0.0)) + sigma_v**2 * np.eye(N)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(K, yv)
    ref_loss = -0.5 * (yv @ alpha + 2 * np.sum(np.log(np.diag(L))) + N * np.log(2 * np.pi))

    got_loss, got_gs, got_gl = fn(sigma_v, ls_v, Xv, yv)
    np.testing.assert_allclose(got_loss, ref_loss, atol=1e-10)

    eps = 1e-6
    gs_fd = (fn(sigma_v + eps, ls_v, Xv, yv)[0] - fn(sigma_v - eps, ls_v, Xv, yv)[0]) / (2 * eps)
    gl_fd = (fn(sigma_v, ls_v + eps, Xv, yv)[0] - fn(sigma_v, ls_v - eps, Xv, yv)[0]) / (2 * eps)
    np.testing.assert_allclose(got_gs, gs_fd, atol=1e-6)
    np.testing.assert_allclose(got_gl, gl_fd, atol=1e-6)
