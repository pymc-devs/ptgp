import pytensor.tensor as pt

from pytensor.assumptions.core import check_assumption
from pytensor.assumptions.positive_definite import POSITIVE_DEFINITE
from pytensor.assumptions.triangular import LOWER_TRIANGULAR
from pytensor.graph.rewriting.basic import copy_stack_trace, node_rewriter
from pytensor.scalar import Sqr
from pytensor.tensor.basic import ExtractDiag
from pytensor.tensor.blas import Dot22
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.linalg.decomposition.cholesky import Cholesky, cholesky
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.solvers.psd import cho_solve
from pytensor.tensor.linalg.solvers.triangular import solve_triangular
from pytensor.tensor.linalg.summary import Det
from pytensor.tensor.math import Dot, Prod, sign
from pytensor.tensor.rewriting.basic import (
    register_canonicalize,
    register_specialize,
    register_stabilize,
)
from pytensor.tensor.rewriting.blockwise import blockwise_of
from pytensor.tensor.rewriting.math import local_useless_abs


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


# TODO: Workarounds for pymc-devs/pytensor#2468. Remove the stabilize registration of
# local_useless_abs, local_pow2_to_sqr, local_sign_sqr and local_sign_prod once a pytensor
# release includes the fix, and bump the pytensor pin to that release.
#
# PyTensor lowers ``det(K)`` to ``prod(diag(L)) ** 2`` during stabilize. Its ``log(sqr(x))``
# rewrite then needs the ``abs`` from ``slogdet`` gone, but ``local_useless_abs`` only runs in
# canonicalize and specialize. Anything that differentiates the stabilized graph, like
# pytensor-ml's ``compile_train``, would otherwise see a product that underflows for large N.
register_stabilize(local_useless_abs)


@register_stabilize
@node_rewriter([pt.pow])
def local_pow2_to_sqr(fgraph, node):
    r"""Rewrite ``x ** 2`` as ``sqr(x)`` during stabilize.

    PyTensor does this only in specialize, after stabilize has lowered ``det(K)``
    to ``prod(diag(L)) ** 2``. PyTensor's ``log(sqr(x))`` rewrite needs the
    ``sqr`` form to turn ``log(det)`` into a sum of logs before the product
    underflows.
    """
    x, exponent = node.inputs
    if pt.get_underlying_scalar_constant_value(exponent, raise_not_constant=False) != 2:
        return None
    new = pt.sqr(x)
    if new.type.broadcastable != node.outputs[0].type.broadcastable:
        return None
    new = new.astype(node.outputs[0].dtype)
    copy_stack_trace(node.outputs[0], new)
    return [new]


@register_canonicalize
@register_stabilize
@register_specialize
@node_rewriter([sign])
def local_sign_sqr(fgraph, node):
    r"""Rewrite ``sign(sqr(x))`` as ``sqr(sign(x))``.

    .. math::

        \operatorname{sign}(x^2) = \operatorname{sign}(x)^2
    """
    match node.inputs[0].owner_op_and_inputs:
        case (Elemwise(Sqr()), x):
            new = pt.sqr(sign(x)).astype(node.outputs[0].dtype)
            copy_stack_trace(node.outputs[0], new)
            return [new]


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
