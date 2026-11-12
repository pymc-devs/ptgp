import pytensor.tensor as pt

from pytensor.graph.replace import graph_replace

from ptgp.kernels.base import (
    Kernel,
    KernelOp,
    WarpedRowsOp,
    leaf_graph,
    resolve_active_dims,
    row_placeholders,
)
from ptgp.kernels.stationary import _squared_distance


class LinearOp(KernelOp):
    def evaluate(self, left, right, c):
        return (left - c) @ (right - c).T


class RandomWalkOp(KernelOp):
    def evaluate(self, left, right):
        return pt.minimum(left, right.T)


class GibbsOp(KernelOp):
    """``left_ls`` and ``right_ls`` are per-row lengthscales, computed in the outer graph."""

    unit_diagonal = True

    def evaluate(self, left, right, left_ls, right_ls):
        squared_distance = _squared_distance(left, right)
        denominator = pt.reshape(pt.square(left_ls), (-1, 1)) + pt.reshape(
            pt.square(right_ls), (1, -1)
        )
        amplitude = pt.sqrt(2.0 * pt.outer(left_ls, right_ls) / denominator)
        return amplitude * pt.exp(-squared_distance / denominator)


class Linear(Kernel):
    r"""Linear (dot-product) kernel.

    .. math::
       k(x, y) = (x - c)^{\top} (y - c)

    Non-stationary: the covariance depends on the absolute input position, not
    just on :math:`x - y`. Scale with multiplication: eta**2 * Linear(input_dim=...).

    Parameters
    ----------
    input_dim : int
        Number of columns of ``X`` the kernel expects.
    c : scalar or 1-D tensor, default 0.0
        Center subtracted from the inputs. A scalar applies to every active
        column, and a length-``len(active_dims)`` vector applies per-column.
    active_dims : sequence of int, optional
        Columns of ``X`` this kernel operates on. Defaults to all columns.
    """

    def __init__(self, input_dim, c=0.0, active_dims=None):
        graph = leaf_graph(LinearOp, input_dim, active_dims, c)
        super().__init__(input_dim=input_dim, graph=graph)


class RandomWalk(Kernel):
    r"""Random walk (Brownian motion / Wiener process) kernel.

    .. math::
       k(x, y) = \min(x, y)

    Defined for positive inputs. Operates on a single input column.
    Scale with multiplication: eta**2 * RandomWalk()

    Parameters
    ----------
    input_dim : int
        Number of columns of ``X`` the kernel expects.
    active_dims : sequence of int, optional
        Column of ``X`` to operate on. Must have length 1. Required when
        ``input_dim > 1``. Defaults to ``[0]`` when ``input_dim == 1``.
    """

    def __init__(self, input_dim=1, active_dims=None):
        if len(resolve_active_dims(input_dim, active_dims)) != 1:
            raise ValueError(
                "RandomWalk kernel requires active_dims of length 1 when input_dim > 1."
            )

        graph = leaf_graph(RandomWalkOp, input_dim, active_dims)
        super().__init__(input_dim=input_dim, graph=graph)


class Gibbs(Kernel):
    r"""Gibbs kernel with a location-dependent lengthscale function.

    .. math::
       k(x, y) = \sqrt{\frac{2 \ell(x) \ell(y)}{\ell(x)^2 + \ell(y)^2}}
                 \exp\left(-\frac{(x - y)^2}{\ell(x)^2 + \ell(y)^2}\right)

    Operates on a single input column.

    Parameters
    ----------
    lengthscale_func : callable
        ``lengthscale_func(X) -> tensor`` returning per-point lengthscales of
        shape ``(N,)`` for input ``X`` of shape ``(N, 1)``.
    input_dim : int
        Number of columns of ``X`` the kernel expects.
    active_dims : sequence of int, optional
        Column of ``X`` to operate on. Must have length 1. Required when
        ``input_dim > 1``.
    """

    def __init__(self, lengthscale_func, input_dim=1, active_dims=None):
        columns = list(resolve_active_dims(input_dim, active_dims))
        if len(columns) != 1:
            raise ValueError("Gibbs kernel requires active_dims of length 1 when input_dim > 1.")
        if not callable(lengthscale_func):
            raise TypeError("lengthscale_func must be callable")

        left, right = row_placeholders(input_dim)
        graph = leaf_graph(
            GibbsOp,
            input_dim,
            active_dims,
            lengthscale_func(left[:, columns]),
            lengthscale_func(right[:, columns]),
        )
        super().__init__(input_dim=input_dim, graph=graph)


class WarpedInput(Kernel):
    r"""Warp the inputs of another kernel with an arbitrary function.

    .. math::
       k(x, y) = k_{\mathrm{inner}}(w(x), w(y))

    Parameters
    ----------
    input_dim : int
        Number of columns of ``X`` this kernel expects. May differ from
        ``kernel_func.input_dim``, because the warp may change dimensionality.
    kernel_func : Kernel
        Inner kernel applied to the warped inputs.
    warp_func : callable
        ``warp_func(X) -> X_warped``. PyTensor function.
    active_dims : sequence of int, optional
        Columns of ``X`` to warp. Defaults to all.
    """

    def __init__(self, input_dim, kernel_func, warp_func, active_dims=None):
        if not isinstance(kernel_func, Kernel):
            raise TypeError("kernel_func must be a Kernel")
        if not callable(warp_func):
            raise TypeError("warp_func must be callable")

        columns = list(resolve_active_dims(input_dim, active_dims))
        left, right = row_placeholders(input_dim)
        inner_left, inner_right = kernel_func.rows
        warped_left = pt.cast(warp_func(left[:, columns]), inner_left.dtype)
        warped_right = pt.cast(warp_func(right[:, columns]), inner_right.dtype)

        graph = graph_replace(
            kernel_func.graph,
            replace={
                inner_left: WarpedRowsOp()(warped_left, left),
                inner_right: WarpedRowsOp()(warped_right, right),
            },
        )
        super().__init__(input_dim=input_dim, graph=graph)
