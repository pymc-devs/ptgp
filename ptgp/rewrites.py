import pytensor.tensor as pt

from pytensor.assumptions.core import check_assumption
from pytensor.assumptions.positive_definite import POSITIVE_DEFINITE
from pytensor.assumptions.triangular import LOWER_TRIANGULAR
from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.scalar import Abs, Sqr
from pytensor.tensor.basic import ExtractDiag
from pytensor.tensor.blas import Dot22
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor.linalg.decomposition.lu import LU, LUFactor
from pytensor.tensor.linalg.decomposition.qr import QR
from pytensor.tensor.linalg.decomposition.svd import SVD
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.solvers.psd import cho_solve
from pytensor.tensor.linalg.solvers.triangular import solve_triangular
from pytensor.tensor.linalg.summary import Det, det
from pytensor.tensor.math import Dot, Prod, prod, sign
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.blockwise import blockwise_of

# Imported before the override of det_of_matrix_factorized_elsewhere below, so
# PyTensor's version is registered first and gets replaced rather than the reverse.
from pytensor.tensor.rewriting.linalg.utils import matrix_diagonal_product


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

    Requires :math:`L` carrying the ``LOWER_TRIANGULAR`` assumption. The square
    goes inside the product so that ``local_log_prod_to_sum_log`` can turn
    ``log(det)`` into a sum of logs, which does not underflow for large ``N``.

    .. math::

        \det(L L^\top) = \det(L^\top L) = \det(L)^2 = \prod_i L_{ii}^2
    """
    [A] = node.inputs
    aat = _try_AAT_factor(fgraph, A, lower_only=True)
    if aat is None:
        return None
    L, _ = aat
    diag_L = pt.diagonal(L, axis1=-2, axis2=-1)
    new_det = pt.sqr(diag_L).prod(axis=-1).astype(node.outputs[0].dtype)
    copy_stack_trace(node.outputs[0], new_det)
    return [new_det]


@register_stabilize("shape_unsafe", overwrite_existing=True)
@register_specialize("shape_unsafe", overwrite_existing=True)
@node_rewriter([det])
def det_of_matrix_factorized_elsewhere(fgraph, node):
    """
    If we have det(X) or abs(det(X)) and there is already a nice decomposition(X) floating around,
    use it to compute it more cheaply

    Copy of PyTensor's rewrite of the same name, registered over it. The only
    change is the Cholesky case, which returns ``prod(diag(L) ** 2)`` instead of
    ``prod(diag(L)) ** 2`` so that ``local_log_prod_to_sum_log`` can turn
    ``log(det)`` into a sum of logs. Remove once the change is upstream.
    """
    [det] = node.outputs
    [x] = node.inputs

    sign_not_needed = all(
        isinstance(client.op, Elemwise) and isinstance(client.op.scalar_op, Abs | Sqr)
        for client, _ in fgraph.clients[det]
    )

    new_det = None
    for client, _ in fgraph.clients[x]:
        core_op = client.op.core_op if isinstance(client.op, Blockwise) else client.op
        match core_op:
            case Cholesky():
                L = client.outputs[0]
                new_det = prod(pt.sqr(pt.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
            case LU():
                U = client.outputs[-1]
                new_det = matrix_diagonal_product(U)
            case LUFactor():
                LU_packed = client.outputs[0]
                new_det = matrix_diagonal_product(LU_packed)
            case _:
                if not sign_not_needed:
                    continue
                match core_op:
                    case SVD():
                        lmbda = client.outputs[1] if core_op.compute_uv else client.outputs[0]
                        new_det = prod(lmbda, axis=-1)
                    case QR():
                        R = client.outputs[-1]
                        # if mode == "economic", R may not be square and this rewrite could hide a shape error
                        # That's why it's tagged as `shape_unsafe`
                        new_det = matrix_diagonal_product(R)

        if new_det is not None:
            # found a match
            break
    else:  # no-break (i.e., no-match)
        return None

    [det] = node.outputs
    copy_stack_trace(det, new_det)
    return [new_det]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([sign])
def local_sign_prod(fgraph, node):
    r"""Rewrite ``sign(prod(x))`` as ``prod(sign(x))``.

    Looks through one ``DimShuffle`` between the ``sign`` and the ``prod``. The
    gradient of the ``abs`` in ``slogdet`` is ``sign(det)``. Without this
    rewrite, a determinant lowered to a diagonal product underflows to 0 for
    large ``N``, and ``sign(0) = 0`` silently drops the log-determinant term
    from the gradient.

    .. math::

        \operatorname{sign}\left(\prod_i x_i\right) = \prod_i \operatorname{sign}(x_i)
    """
    [arg] = node.inputs
    dimshuffle = None
    match arg.owner_op_and_inputs:
        case (DimShuffle() as dimshuffle, inner):
            arg = inner
    match arg.owner_op_and_inputs:
        case (Prod(axis=axis), x):
            new = sign(x).prod(axis=axis)
            if dimshuffle is not None:
                new = dimshuffle(new)
            new = new.astype(node.outputs[0].dtype)
            copy_stack_trace(node.outputs[0], new)
            return [new]


@register_specialize
@node_rewriter([ExtractDiag])
def diag_of_AAT_to_row_norms_squared(fgraph, node):
    r"""Lower the diagonal of ``A @ A.T`` or ``A.T @ A`` to elementwise norms.

    .. math::

        (A A^\top)_{ii} = \sum_k A_{ik}^2, \qquad
        (A^\top A)_{ii} = \sum_k A_{ki}^2

    ``ExtractDiag(A @ A.T)`` lowers to ``sum(A**2, axis=-1)``; ``ExtractDiag(A.T @ A)``
    lowers to ``sum(A**2, axis=-2)``. No assumption on ``A`` is required.
    """
    extract_op = node.op
    if extract_op.offset != 0:
        return None
    [A_AT] = node.inputs
    ndim = A_AT.type.ndim
    if ndim is None or ndim < 2:
        return None
    # The diag must run along the matrix axes; if ExtractDiag is reading other
    # axes the AAT pattern does not apply.
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
