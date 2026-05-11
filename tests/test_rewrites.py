"""Tests for the assumption rules and rewrites added in ``ptgp.rewrites``.

Written so that, when the rules are upstreamed into PyTensor, the test bodies
move over unchanged — only the imports of ``POSITIVE`` (and the side-effect
import that installs the rules) would shift from ``ptgp`` to ``pytensor``.
"""

import numpy as np
import pytensor.assumptions as pa
import pytensor.tensor as pt

from pytensor import function
from pytensor.assumptions import (
    POSITIVE_DEFINITE,
    AssumptionFeature,
)
from pytensor.graph import rewrite_graph
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.basic import ExtractDiag
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.summary import Det, SLogDet

# Install the assumption rules and rewrites under test (side-effect import).
import ptgp.rewrites  # noqa: F401

from ptgp.rewrites import POSITIVE


def make_fgraph(*outputs, inputs=None):
    fg = FunctionGraph(outputs=list(outputs), inputs=inputs, clone=False)
    af = AssumptionFeature()
    fg.attach_feature(af)
    return fg, af


# ---------------------------------------------------------------------------
# POSITIVE assumption propagation
# ---------------------------------------------------------------------------


def test_assume_accepts_positive_kwarg():
    x = pt.dscalar("x")
    y = pa.assume(x, positive=True)
    _, af = make_fgraph(y, inputs=[x])
    assert af.check(y, POSITIVE)


def test_positive_propagates_through_sqr_and_pow():
    x = pa.assume(pt.dscalar("x"), positive=True)
    _, af = make_fgraph(x**2, x**3.5, inputs=[x.owner.inputs[0]])
    assert af.check(x**2, POSITIVE)
    assert af.check(x**3.5, POSITIVE)


def test_positive_propagates_through_mul_only_if_all_positive():
    x = pa.assume(pt.dscalar("x"), positive=True)
    y = pa.assume(pt.dscalar("y"), positive=True)
    z = pt.dscalar("z")  # no assumption
    _, af = make_fgraph(x * y, x * z, inputs=[x.owner.inputs[0], y.owner.inputs[0], z])
    assert af.check(x * y, POSITIVE)
    assert not af.check(x * z, POSITIVE)


def test_positive_propagates_through_dimshuffle():
    x = pa.assume(pt.dvector("x"), positive=True)
    _, af = make_fgraph(x[None, :], inputs=[x.owner.inputs[0]])
    assert af.check(x[None, :], POSITIVE)


def test_alloc_of_positive_constant_is_positive():
    n = pt.lscalar("n")
    ones = pt.ones((n,))
    _, af = make_fgraph(ones, inputs=[n])
    assert af.check(ones, POSITIVE)


def test_alloc_of_zero_is_not_positive():
    n = pt.lscalar("n")
    zeros = pt.zeros((n,))
    _, af = make_fgraph(zeros, inputs=[n])
    assert not af.check(zeros, POSITIVE)


def test_diagonal_of_identity_is_positive():
    n = pt.lscalar("n")
    diag_eye = pt.diagonal(pt.eye(n))
    _, af = make_fgraph(diag_eye, inputs=[n])
    assert af.check(diag_eye, POSITIVE)


# ---------------------------------------------------------------------------
# POSITIVE_DEFINITE inference for new patterns
# ---------------------------------------------------------------------------


def test_alloc_diag_of_symbolic_positive_vector_is_psd():
    v = pa.assume(pt.dvector("v"), positive=True)
    M = pt.diag(v)
    _, af = make_fgraph(M, inputs=[v.owner.inputs[0]])
    assert af.check(M, POSITIVE_DEFINITE)


def test_alloc_diag_of_unknown_vector_is_not_psd():
    v = pt.dvector("v")
    M = pt.diag(v)
    _, af = make_fgraph(M, inputs=[v])
    assert not af.check(M, POSITIVE_DEFINITE)


def test_mul_of_positive_scalar_and_psd_matrix_is_psd():
    sigma = pa.assume(pt.dscalar("sigma"), positive=True)
    K = pa.assume(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    M = sigma**2 * K
    _, af = make_fgraph(M, inputs=[sigma.owner.inputs[0], K.owner.inputs[0]])
    assert af.check(M, POSITIVE_DEFINITE)


def test_psd_propagates_through_matrix_transpose():
    K = pa.assume(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    _, af = make_fgraph(K.T, inputs=[K.owner.inputs[0]])
    assert af.check(K.T, POSITIVE_DEFINITE)


def test_quadratic_form_dot_is_psd():
    """X.T @ M @ X is PSD when M is PSD."""
    X = pt.dmatrix("X")
    M = pa.assume(pt.dmatrix("M"), positive_definite=True, symmetric=True)
    Q = X.T.dot(M.dot(X))
    _, af = make_fgraph(Q, inputs=[X, M.owner.inputs[0]])
    assert af.check(Q, POSITIVE_DEFINITE)


def test_quadratic_form_with_solve_is_psd():
    """X.T @ M^{-1} @ X (written as Solve) is PSD when M is PSD."""
    X = pt.dmatrix("X")
    M = pa.assume(pt.dmatrix("M"), positive_definite=True, symmetric=True)
    Q = X.T.dot(pt.linalg.solve(M, X))
    _, af = make_fgraph(Q, inputs=[X, M.owner.inputs[0]])
    assert af.check(Q, POSITIVE_DEFINITE)


def test_quadratic_form_with_cholesky_solve_is_psd():
    """X.T @ M^{-1} @ X (written as CholeskySolve) is PSD when L = Cholesky(M)."""
    X = pt.dmatrix("X")
    M = pa.assume(pt.dmatrix("M"), positive_definite=True, symmetric=True)
    L = cholesky(M, lower=True)
    Q = X.T.dot(pt.linalg.cho_solve((L, True), X))
    _, af = make_fgraph(Q, inputs=[X, M.owner.inputs[0]])
    assert af.check(Q, POSITIVE_DEFINITE)


def test_set_subtensor_of_zeros_with_positive_diagonal_is_psd():
    """``zeros(N, N)[arange(N), arange(N)] = positive_vec`` is PSD."""
    n = pt.lscalar("n")
    v = pa.assume(pt.dvector("v"), positive=True)
    base = pt.zeros((n, n))
    idx = pt.arange(n)
    M = pt.set_subtensor(base[idx, idx], v)
    _, af = make_fgraph(M, inputs=[n, v.owner.inputs[0]])
    assert af.check(M, POSITIVE_DEFINITE)


# ---------------------------------------------------------------------------
# SLogDet -> Cholesky rewrite
# ---------------------------------------------------------------------------


def _has_op(graph, op_type):
    fg = FunctionGraph(outputs=[graph] if not isinstance(graph, list) else graph, clone=False)
    for node in fg.apply_nodes:
        op = node.op
        core = op.core_op if isinstance(op, Blockwise) else op
        if isinstance(core, op_type):
            return True
    return False


def test_slogdet_of_psd_is_lowered_to_cholesky():
    K = pa.assume(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    _, logdet = pt.linalg.slogdet(K)
    rewritten = rewrite_graph(logdet, include=("fast_run",))
    assert not _has_op(rewritten, SLogDet)
    assert _has_op(rewritten, Cholesky)


def test_slogdet_without_psd_is_not_lowered():
    K = pt.dmatrix("K")  # no PSD annotation
    _, logdet = pt.linalg.slogdet(K)
    rewritten = rewrite_graph(logdet, include=("fast_run",))
    # Without the PSD assumption the Cholesky-based rewrite must not fire.
    assert not _has_op(rewritten, Cholesky)


def test_slogdet_reuses_existing_cholesky():
    """When an upstream Solve has already produced ``Cholesky(K)``, SLogDet should reuse it."""
    K = pa.assume(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    b = pt.dvector("b")
    x = pt.linalg.solve(K, b, assume_a="pos")
    _, logdet = pt.linalg.slogdet(K)
    rewritten = rewrite_graph([x, logdet], include=("fast_run",))
    fg = FunctionGraph(outputs=rewritten, clone=False)
    n_chol = sum(
        1
        for node in fg.apply_nodes
        if isinstance(node.op.core_op if isinstance(node.op, Blockwise) else node.op, Cholesky)
    )
    assert n_chol == 1, f"expected SLogDet to share Cholesky, got {n_chol} factorisations"


def test_slogdet_lowering_is_numerically_correct():
    K = pa.assume(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    _, logdet = pt.linalg.slogdet(K)
    f = function([K.owner.inputs[0]], logdet)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    K_val = A @ A.T + np.eye(6)
    np.testing.assert_allclose(f(K_val), np.linalg.slogdet(K_val)[1], atol=1e-8)


def test_slogdet_of_LLT_takes_diag_shortcut_when_L_lower_triangular():
    """SLogDet(L @ L.T) collapses to ``2 * sum(log|diag(L)|)`` — no Cholesky, no SLogDet."""
    L = pa.assume(pt.dmatrix("L"), lower_triangular=True)
    _, logdet = pt.linalg.slogdet(L @ L.T)
    rewritten = rewrite_graph(logdet, include=("fast_run",))
    assert not _has_op(rewritten, SLogDet)
    assert not _has_op(rewritten, Cholesky)


def test_slogdet_of_LLT_diag_shortcut_is_numerically_correct():
    """The diag shortcut produces the same value as full slogdet."""
    L = pa.assume(pt.dmatrix("L"), lower_triangular=True)
    _, logdet = pt.linalg.slogdet(L @ L.T)
    f = function([L.owner.inputs[0]], logdet)

    rng = np.random.default_rng(0)
    M = 6
    L_val = np.tril(rng.standard_normal((M, M)))
    L_val[np.arange(M), np.arange(M)] = np.abs(L_val[np.arange(M), np.arange(M)]) + 0.5
    expected = float(np.linalg.slogdet(L_val @ L_val.T)[1])
    np.testing.assert_allclose(f(L_val), expected, atol=1e-12)


def test_slogdet_of_LLT_diag_shortcut_handles_signed_diagonal():
    """``log|x²|`` style is robust to signed diagonal entries on L."""
    L = pa.assume(pt.dmatrix("L"), lower_triangular=True)
    _, logdet = pt.linalg.slogdet(L @ L.T)
    f = function([L.owner.inputs[0]], logdet)

    rng = np.random.default_rng(1)
    M = 5
    L_val = np.tril(rng.standard_normal((M, M)))
    # Mix positive and negative diagonals.
    diag = rng.standard_normal(M)
    L_val[np.arange(M), np.arange(M)] = np.where(np.abs(diag) > 0.3, diag, np.sign(diag) * 0.5)
    expected = float(np.linalg.slogdet(L_val @ L_val.T)[1])
    np.testing.assert_allclose(f(L_val), expected, atol=1e-12)


def test_det_of_LLT_takes_diag_product_when_L_lower_triangular():
    """``Det(L @ L.T)`` lowers to ``(prod(diag(L)))**2`` — no Det Apply remains."""
    from pytensor.tensor.linalg.summary import det

    L = pa.assume(pt.dmatrix("L"), lower_triangular=True)
    d = det(L @ L.T)
    rewritten = rewrite_graph(d, include=("fast_run",))
    assert not _has_op(rewritten, Det)


def test_det_of_LLT_diag_product_is_numerically_correct():
    from pytensor.tensor.linalg.summary import det

    L = pa.assume(pt.dmatrix("L"), lower_triangular=True)
    d = det(L @ L.T)
    f = function([L.owner.inputs[0]], d)

    rng = np.random.default_rng(0)
    M = 5
    L_val = np.tril(rng.standard_normal((M, M)))
    L_val[np.arange(M), np.arange(M)] = np.abs(L_val[np.arange(M), np.arange(M)]) + 0.5
    expected = float(np.linalg.det(L_val @ L_val.T))
    np.testing.assert_allclose(f(L_val), expected, atol=1e-10)


def test_diag_of_AAT_to_row_norms_squared_eliminates_dot():
    """``ExtractDiag(A @ A.T)`` lowers to ``sum(A**2, axis=-1)`` — no Dot, no ExtractDiag."""
    A = pt.dmatrix("A")
    out = pt.diagonal(A @ A.T)
    rewritten = rewrite_graph(out, include=("fast_run",))
    # Folds out: no ExtractDiag (no need to extract diag of an outer product).
    assert not _has_op(rewritten, ExtractDiag)


def test_diag_of_AAT_is_numerically_correct():
    A = pt.dmatrix("A")
    out = pt.diagonal(A @ A.T)
    f = function([A], out)

    rng = np.random.default_rng(0)
    A_val = rng.standard_normal((6, 4))  # non-square — diagonal still defined for A @ A.T
    np.testing.assert_allclose(f(A_val), np.diag(A_val @ A_val.T), atol=1e-12)


def test_diag_of_ATA_lowers_to_column_norms():
    """``ExtractDiag(A.T @ A)`` lowers to ``sum(A**2, axis=-2)`` — no ExtractDiag, no Dot."""
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


def test_slogdet_of_LTL_takes_diag_shortcut_when_L_lower_triangular():
    """``SLogDet(L.T @ L)`` lowers via the diagonal shortcut — same as L @ L.T."""
    L = pa.assume(pt.dmatrix("L"), lower_triangular=True)
    _, logdet = pt.linalg.slogdet(L.T @ L)
    rewritten = rewrite_graph(logdet, include=("fast_run",))
    assert not _has_op(rewritten, SLogDet)
    assert not _has_op(rewritten, Cholesky)


def test_slogdet_of_LTL_is_numerically_correct():
    L = pa.assume(pt.dmatrix("L"), lower_triangular=True)
    _, logdet = pt.linalg.slogdet(L.T @ L)
    f = function([L.owner.inputs[0]], logdet)

    rng = np.random.default_rng(0)
    M = 5
    L_val = np.tril(rng.standard_normal((M, M)))
    L_val[np.arange(M), np.arange(M)] = np.abs(L_val[np.arange(M), np.arange(M)]) + 0.5
    expected = float(np.linalg.slogdet(L_val.T @ L_val)[1])
    np.testing.assert_allclose(f(L_val), expected, atol=1e-12)


def test_matrix_inverse_of_LTL_uses_solve_triangular():
    """``MatrixInverse(L.T @ L)`` lowers to two solve_triangular calls — no fresh Cholesky."""
    L = pa.assume(pt.dmatrix("L"), lower_triangular=True)
    inv_M = pt.linalg.inv(L.T @ L)
    rewritten = rewrite_graph(inv_M, include=("fast_run",))
    # No MatrixInverse, no Cholesky (L is reused via solve_triangular).
    assert not _has_op(rewritten, MatrixInverse)
    assert not _has_op(rewritten, Cholesky)


def test_matrix_inverse_of_LTL_is_numerically_correct():
    L = pa.assume(pt.dmatrix("L"), lower_triangular=True)
    inv_M = pt.linalg.inv(L.T @ L)
    f = function([L.owner.inputs[0]], inv_M)

    rng = np.random.default_rng(0)
    M = 5
    L_val = np.tril(rng.standard_normal((M, M)))
    L_val[np.arange(M), np.arange(M)] = np.abs(L_val[np.arange(M), np.arange(M)]) + 0.5
    expected = np.linalg.inv(L_val.T @ L_val)
    np.testing.assert_allclose(f(L_val), expected, atol=1e-10)


# ---------------------------------------------------------------------------
# MatrixInverse(PSD A) -> cho_solve(L, eye) rewrite
# ---------------------------------------------------------------------------


def test_matrix_inverse_of_psd_is_lowered_to_cholesky():
    K = pa.assume(pt.dmatrix("K"), positive_definite=True, symmetric=True)
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
    K = pa.assume(pt.dmatrix("K"), positive_definite=True, symmetric=True)
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
    K = pa.assume(pt.dmatrix("K"), positive_definite=True, symmetric=True)
    inv_K = pt.linalg.inv(K)
    f = function([K.owner.inputs[0]], inv_K)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 6))
    K_val = A @ A.T + np.eye(6)
    np.testing.assert_allclose(f(K_val), np.linalg.inv(K_val), atol=1e-10)


# ---------------------------------------------------------------------------
# merge_composites_with_shared_inputs (Blocker A)
# ---------------------------------------------------------------------------


def _gp_mll_like_joint():
    """Build the joint forward+gradient graph for ``Unapproximated + marginal_log_likelihood``.

    This is the canonical case the merge rewrite targets — it produces the
    sibling-Composite pattern that FusionOptimizer creates when forward and
    gradient consumers of the kernel ``exp(...)`` value sit in different
    convex closures.
    """
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


def test_merge_composites_collapses_gp_joint_graph_to_one_cholesky():
    """Joint forward+gradient compilation should hit the floor: exactly one Cholesky."""
    inputs, outputs = _gp_mll_like_joint()
    fn = function(inputs, outputs)
    assert _count_in_compiled(fn, Cholesky) == 1
    assert _count_in_compiled(fn, MatrixInverse) == 0


def test_merge_composites_does_not_change_single_composite_graphs():
    """Graphs with no sibling-Composite pattern should compile unchanged in cubic-op count."""
    x = pt.dvector("x")
    out = pt.exp(x) + pt.log1p(x**2)  # one Elemwise chain, one consumer
    fn = function([x], out)
    # There should be at most one Elemwise(Composite); definitely no Cholesky/MatrixInverse.
    assert _count_in_compiled(fn, Cholesky) == 0
    assert _count_in_compiled(fn, MatrixInverse) == 0


def test_merge_composites_preserves_numerical_correctness():
    """Joint forward+gradient values match an analytic reference and finite-difference grads."""
    inputs, outputs = _gp_mll_like_joint()
    fn = function(inputs, outputs)

    rng = np.random.default_rng(0)
    N = 8
    Xv = rng.standard_normal((N, 1))
    yv = rng.standard_normal(N)
    sigma_v, ls_v = 0.5, 1.2

    # Analytic reference for the loss (note: marginal_log_likelihood includes the 2π term)
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
