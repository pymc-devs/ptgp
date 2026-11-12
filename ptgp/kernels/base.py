from functools import cache

import numpy as np
import pytensor
import pytensor.assumptions as pta
import pytensor.tensor as pt

from pytensor.gradient import pullback, pushforward
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors
from pytensor.tensor.symbolic import TensorSymbolicOp


@cache
def _row_placeholders(input_dim, dtype):
    return (
        pt.matrix("_left", shape=(None, input_dim), dtype=dtype),
        pt.matrix("_right", shape=(None, input_dim), dtype=dtype),
    )


def row_placeholders(input_dim):
    """The ``(left, right)`` row variables every kernel of ``input_dim`` is a graph on."""
    return _row_placeholders(input_dim, pytensor.config.floatX)


def resolve_active_dims(input_dim, active_dims):
    """Validate ``active_dims`` as a tuple of columns. ``None`` means all columns."""
    if active_dims is None:
        return tuple(range(input_dim))

    active_dims = np.asarray(active_dims, dtype=int)
    if active_dims.max() >= input_dim:
        raise ValueError(
            f"active_dims contains index {int(active_dims.max())}, but input_dim is {input_dim}"
        )
    return tuple(active_dims.tolist())


def leaf_graph(op_type, input_dim, active_dims, *parameters):
    """Apply a :class:`KernelOp` to the row placeholders of ``input_dim``."""
    kernel_op = op_type(active_dims=resolve_active_dims(input_dim, active_dims))
    return kernel_op(*row_placeholders(input_dim), *parameters)


class KernelOp(TensorSymbolicOp):
    """``K(left, right, *parameters)`` for one base kernel, as a graph node.

    Subclasses implement ``evaluate``, the kernel math on the active columns. Parameters are
    explicit inputs, so conditioning reaches them, and the type of the Op says which kernel
    this is, so rewrites can choose the best form for the situation. ``ptgp.rewrites`` inlines
    the body at specialize, after every kernel-targeted rewrite has had its turn.

    Parameters
    ----------
    active_dims : tuple of int
        Columns of the row inputs this kernel operates on.
    """

    __props__ = ("active_dims",)
    inline = False
    unit_diagonal = False

    def __str__(self):
        return type(self).__name__

    def evaluate(self, left, right, *parameters):
        """Raw kernel matrix on the active columns."""
        raise NotImplementedError

    def build_inner_graph(self, left, right, *parameters):
        columns = list(self.active_dims)
        return [self.evaluate(left[:, columns], right[:, columns], *parameters)]

    def pullback(self, inputs, outputs, output_grads):
        # K(X, X) passes one node in both row slots, and each slot needs its own partial.
        distinct_inputs = [variable.copy() for variable in inputs]
        return pullback(
            self.build_inner_graph(*distinct_inputs),
            distinct_inputs,
            output_grads,
            disconnected_inputs="ignore",
            return_disconnected="disconnected",
        )

    def pushforward(self, inputs, outputs, eval_points):
        distinct_inputs = [variable.copy() for variable in inputs]
        return pushforward(
            self.build_inner_graph(*distinct_inputs),
            distinct_inputs,
            eval_points,
            disconnected_outputs="ignore",
            return_disconnected="disconnected",
        )


class WarpedRowsOp(TensorSymbolicOp):
    """Warped rows, with the rows they came from as a second input.

    The warp stays in the outer graph, so conditioning reaches its parameters. This node only
    records where the rows came from, for :attr:`Kernel.rows`, so it inlines at compile time.
    """

    inline = True

    def __str__(self):
        return type(self).__name__

    def build_inner_graph(self, warped, rows):
        return [warped]


def _unwarped(rows):
    while isinstance(rows.owner_op, WarpedRowsOp):
        rows = rows.owner.inputs[1]
    return rows


def _rows_as_dtype(rows, dtype):
    """``rows`` in ``dtype``, recorded as an identity warp so :attr:`Kernel.rows` finds them."""
    if rows.dtype == dtype:
        return rows
    return WarpedRowsOp()(pt.cast(rows, dtype), rows)


class Kernel:
    """A covariance function: a graph on the row placeholders of ``input_dim``.

    The graph is the kernel. Its leaves are :class:`KernelOp` nodes, and sums and products
    of kernels are ordinary graph arithmetic on them, so nothing but the graph is stored.
    Calling a kernel replaces the row placeholders.

    Parameters
    ----------
    input_dim : int
        Number of columns of ``X`` the kernel expects.
    graph : TensorVariable
        ``K(left, right)`` on ``row_placeholders(input_dim)``.
    """

    def __init__(self, input_dim, graph):
        self.input_dim = input_dim
        self.graph = graph

    @property
    def rows(self):
        """The ``(left, right)`` placeholders this kernel's graph was built on."""
        kernel_node = next(
            (
                variable.owner
                for variable in ancestors([self.graph])
                if isinstance(variable.owner_op, KernelOp)
            ),
            None,
        )
        if kernel_node is None:
            raise ValueError(
                "Kernel graph has no KernelOp. Its rows are read from the KernelOp nodes."
            )

        left, right, *_ = kernel_node.inputs
        return _unwarped(left), _unwarped(right)

    def __call__(self, X, Y=None):
        """Return K(X, Y), or K(X, X) annotated symmetric and PD if ``Y`` is None.

        Rows are cast to the dtype of the kernel's rows.
        """
        left, right = self.rows

        X = pt.cast(X, left.dtype)
        if Y is None:
            K = graph_replace(self.graph, replace={left: X, right: X})
            return pta.assume(K, symmetric=True, positive_definite=True)

        Y = pt.cast(Y, right.dtype)
        return graph_replace(self.graph, replace={left: X, right: Y})

    def diag(self, X):
        """Return the diagonal of K(X, X)."""
        return pt.diagonal(self(X))

    def _combine(self, other, operation):
        """Apply ``operation`` to this graph and a scalar or another kernel's graph."""
        if not isinstance(other, Kernel):
            return Kernel(input_dim=self.input_dim, graph=operation(self.graph, other))

        if other.input_dim != self.input_dim:
            raise ValueError(
                f"Kernels in a combination must share input_dim "
                f"(got {self.input_dim} and {other.input_dim})"
            )

        left, right = self.rows
        other_left, other_right = other.rows
        other_graph = graph_replace(
            other.graph,
            replace={
                other_left: _rows_as_dtype(left, other_left.dtype),
                other_right: _rows_as_dtype(right, other_right.dtype),
            },
        )
        return Kernel(input_dim=self.input_dim, graph=operation(self.graph, other_graph))

    def __add__(self, other):
        """Return the sum of two kernels."""
        if not isinstance(other, Kernel):
            return NotImplemented
        return self._combine(other, pt.add)

    def __mul__(self, other):
        """Return the product of self and a kernel or a scalar."""
        return self._combine(other, pt.mul)

    def __rmul__(self, other):
        """Return the product of a scalar and self."""
        return self._combine(other, pt.mul)
