import pytensor
import pytensor.tensor as pt

from ptgp.kernels.base import Kernel, KernelOp, leaf_graph, resolve_active_dims


class OverlapOp(KernelOp):
    """Every level matches itself, so k(x, x) = 1."""

    unit_diagonal = True

    def evaluate(self, left, right):
        left_levels = pt.cast(left, "int64")
        right_levels = pt.cast(right, "int64")
        matches = pt.eq(left_levels[:, None, :], right_levels[None, :, :])
        return pt.mean(pt.cast(matches, pytensor.config.floatX), axis=-1)


class LowRankCategoricalOp(KernelOp):
    def evaluate(self, left, right, W, kappa):
        left_levels = pt.cast(left[:, 0], "int64")
        right_levels = pt.cast(right[:, 0], "int64")
        B = W @ W.T + pt.diag(kappa)
        return B[left_levels][:, right_levels]


class Overlap(Kernel):
    """Overlap (Hamming) kernel for integer-coded categorical inputs.

    k(x, y) = (1 / D) * sum_d 1[x_d == y_d]

    where D is the number of active categorical columns. The kernel returns
    the fraction of active columns whose levels match. With one active column,
    this is ``1[x == y]``.

    Values are expected to be non-negative integer level codes stored in a
    float matrix (the library-wide convention). Entries are cast to ``int64``
    inside the kernel; level codes are not range-checked.

    Parameters
    ----------
    input_dim : int
        Total number of columns in the design matrix.
    active_dims : sequence of int, optional
        Columns of ``X`` holding categorical level codes. Defaults to all
        columns.
    """

    def __init__(self, input_dim, active_dims=None):
        super().__init__(input_dim=input_dim, graph=leaf_graph(OverlapOp, input_dim, active_dims))


class LowRankCategorical(Kernel):
    """Categorical kernel with a low-rank-plus-diagonal level covariance.

    k(x, y) = B[x, y] with B = W @ W.T + diag(kappa)

    Also known as the Coregion / Intrinsic Coregionalization Model (ICM)
    kernel. Operates on a single categorical column. Level codes are expected
    to be non-negative integers stored in a float matrix (the library-wide
    convention). Codes are cast to ``int64`` inside the kernel;
    they are not range-checked; passing a code >= ``num_levels`` is user
    error.

    Parameters
    ----------
    input_dim : int
        Total number of columns in the design matrix.
    num_levels : int
        Number of distinct category levels L.
    W : tensor, shape (num_levels, rank)
        Shared latent-factor loadings across levels.
    kappa : tensor, shape (num_levels,)
        Per-level independent variance. Keeps B full rank when rank < L and
        prevents degeneracy when two levels share latent coordinates.
    active_dims : sequence of int, length 1
        Column of ``X`` holding the categorical level codes.

    Notes
    -----
    Rank-1 parameterization and one-hot equivalence. With ``rank=1``,
    ``B = w w.T + diag(kappa)`` for ``w`` of shape ``(num_levels,)``. This is
    equivalent to a one-hot encoding combined with an ``ExpQuad`` kernel with
    ARD lengthscales ``ls_l``, under the constraint
    ``w_l = exp(-1 / (2 * ls_l**2))`` and ``kappa_l = 1 - w_l**2``. Use
    ``rank=1`` when you expect a single shared trend across levels plus
    per-level independent variation.
    """

    def __init__(self, input_dim, num_levels, W, kappa, active_dims=None):
        if len(resolve_active_dims(input_dim, active_dims)) != 1:
            raise ValueError("LowRankCategorical requires active_dims of length 1")

        graph = leaf_graph(LowRankCategoricalOp, input_dim, active_dims, W, kappa)
        super().__init__(input_dim=input_dim, graph=graph)
