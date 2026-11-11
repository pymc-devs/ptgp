import pytensor.assumptions as pta
import pytensor.tensor as pt

from pytensor.assumptions.core import check_assumption, register_assumption, true_if
from pytensor.assumptions.positive_definite import POSITIVE_DEFINITE
from pytensor.assumptions.symmetric import SYMMETRIC
from pytensor.assumptions.triangular import LOWER_TRIANGULAR
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor.basic import ExtractDiag
from pytensor.tensor.blas import Dot22
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.solvers.psd import cho_solve
from pytensor.tensor.linalg.solvers.triangular import solve_triangular
from pytensor.tensor.linalg.summary import Det
from pytensor.tensor.math import Dot
from pytensor.tensor.rewriting.basic import register_specialize, register_stabilize
from pytensor.tensor.rewriting.blockwise import blockwise_of

from ptgp.kernels.base import KernelOp


def _try_AAT_factor(fgraph, M, lower_only=False):
    """Detect a matmul of a matrix with its transpose.

    Parameters
    ----------
    fgraph : FunctionGraph
        Used for assumption lookups when ``lower_only=True``.
    M : TensorVariable
        Candidate matrix.
    lower_only : bool, optional
        Only return matches where the factor carries the ``LOWER_TRIANGULAR``
        assumption. Default False.

    Returns
    -------
    tuple of (TensorVariable, str) or None
        ``(A, "AAT")`` when ``M = A @ A.T``, ``(A, "ATA")`` when ``M = A.T @ A``,
        otherwise None.
    """
    match M.owner_op_and_inputs:
        case (Blockwise(Dot()) | Dot() | Dot22(), a, b):
            pass
        case _:
            return None

    match b.owner_op_and_inputs:
        case (DimShuffle(is_left_expanded_matrix_transpose=True), inner) if inner is a:
            if lower_only and not check_assumption(fgraph, a, LOWER_TRIANGULAR):
                return None
            return a, "AAT"

    match a.owner_op_and_inputs:
        case (DimShuffle(is_left_expanded_matrix_transpose=True), inner) if inner is b:
            if lower_only and not check_assumption(fgraph, b, LOWER_TRIANGULAR):
                return None
            return b, "ATA"

    return None


def _existing_cholesky(fgraph, A):
    """Find an existing ``Cholesky(A, lower=True)`` output already in the graph.

    Parameters
    ----------
    fgraph : FunctionGraph
    A : TensorVariable
        Matrix whose Cholesky factor to look up among its clients.

    Returns
    -------
    TensorVariable or None
        Output of the existing Cholesky Apply, or None.
    """
    for client, _ in fgraph.clients.get(A, ()):
        match client.op:
            case Blockwise(Cholesky(lower=True)) | Cholesky(lower=True):
                return client.outputs[0]
    return None


@register_specialize
@node_rewriter([blockwise_of(Det)])
def det_of_LLT_to_diag_product(fgraph, node):
    r"""Lower ``Det(L @ L.T)`` and ``Det(L.T @ L)`` to a diagonal product.

    Requires :math:`L` carrying the ``LOWER_TRIANGULAR`` assumption.

    .. math::

        \det(L L^\top) = \det(L^\top L) = \det(L)^2 = \left(\prod_i L_{ii}\right)^2
    """
    [A] = node.inputs
    aat = _try_AAT_factor(fgraph, A, lower_only=True)
    if aat is None:
        return None
    L, _ = aat
    diag_L = pt.diagonal(L, axis1=-2, axis2=-1)
    new_det = (diag_L.prod(axis=-1) ** 2).astype(node.outputs[0].dtype)
    copy_stack_trace(node.outputs[0], new_det)
    return [new_det]


@register_specialize
@node_rewriter([blockwise_of(MatrixInverse)])
def matrix_inverse_specialize(fgraph, node):
    r"""Lower ``MatrixInverse(A)`` when ``A`` has a known structure.

    - ``A = L @ L.T`` with lower-triangular ``L``: ``cho_solve((L, True), eye)``,
      reusing ``L`` directly.
    - ``A = L.T @ L`` with lower-triangular ``L``: two ``solve_triangular`` calls
      implementing :math:`L^{-1} L^{-\top}`.
    - ``A`` PSD: ``cho_solve((Cholesky(A), True), eye)``, sharing an existing
      Cholesky factor if one is in the graph.
    """
    [A] = node.inputs
    eye = pt.eye(A.shape[-1], dtype=A.dtype)

    aat = _try_AAT_factor(fgraph, A, lower_only=True)
    if aat is not None:
        L, form = aat
        if form == "AAT":
            inv_A = cho_solve((L, True), eye)
        else:
            # inv(L.T @ L) = inv(L) @ inv(L.T): two triangular solves on L.
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


def _kernel_rows_match(key, op, feature, fgraph, node, input_states):
    """``K(X, X)`` is symmetric and positive definite when both row inputs are one node."""
    return true_if(node.inputs[0] is node.inputs[1])


register_assumption(SYMMETRIC, KernelOp)(_kernel_rows_match)
register_assumption(POSITIVE_DEFINITE, KernelOp)(_kernel_rows_match)


def _normalized_kernel_body(node):
    """Kernel body on ``node.inputs``, after the passes it missed inside the Op.

    The pullback body has been through canonicalize and stabilize since grad time, so a body
    exposed without them neither merges with it nor gets the stabilized forms.
    """
    frozen_body = node.op.fgraph
    body_inputs = [variable.type() for variable in node.inputs]
    body_outputs = frozen_body.bind(dict(zip(frozen_body.inputs, body_inputs)))

    body_graph = FunctionGraph(body_inputs, body_outputs, clone=False)
    rewrite_graph(body_graph, include=("ShapeOpt", "canonicalize", "stabilize"))

    [body] = graph_replace(
        body_graph.outputs,
        replace=dict(zip(body_inputs, node.inputs)),
        strict=False,
    )
    copy_stack_trace(frozen_body.outputs, body)
    return body


@register_specialize("inline_ofg")
@node_rewriter([KernelOp])
def late_inline_kernel(fgraph, node):
    """Expose the kernel body once every kernel-targeted rewrite has had its turn.

    The body carries no annotation, so what the Op proved about ``K(X, X)`` moves onto it.
    """
    body = _normalized_kernel_body(node)
    left, right, *_ = node.inputs
    if left is right:
        return [pta.assume(body, symmetric=True, positive_definite=True)]
    return [body]


def _kernel_diagonal(kernel_op, rows, parameters, dtype):
    """Diagonal of ``K(rows, rows)``, from the Op's declaration or its body on paired rows."""
    if kernel_op.unit_diagonal:
        return pt.ones(rows.shape[0], dtype=dtype)

    body_left, body_right = rows.type(), rows.type()
    [body] = kernel_op.build_inner_graph(body_left, body_right, *parameters)

    paired_rows = rows[:, None, :]
    vectorized = vectorize_graph(body, replace={body_left: paired_rows, body_right: paired_rows})
    return vectorized[:, 0, 0]


@register_stabilize
@register_specialize
@node_rewriter([ExtractDiag])
def diagonal_of_kernel_to_kernel_diagonal(fgraph, node):
    """Lower ``diag(K(X, X))`` without building ``K``."""
    if node.op.offset != 0:
        return None

    match node.inputs[0].owner_op_and_inputs:
        case (KernelOp() as kernel_op, left, right, *parameters) if left is right:
            new_output = _kernel_diagonal(kernel_op, left, parameters, node.outputs[0].dtype)
            copy_stack_trace(node.outputs[0], new_output)
            return [new_output]
    return None
