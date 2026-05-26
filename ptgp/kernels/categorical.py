import pytensor.tensor as pt

from ptgp.kernels.base import Kernel


class Overlap(Kernel):
    """Overlap (Hamming) kernel for integer-coded categorical inputs.

    k(x, y) = (1 / D) * sum_d 1[x_d == y_d]

    where D is the number of active categorical columns. The kernel returns
    the fraction of active columns whose levels match. With one active column,
    this is simply ``1[x == y]``.

    Values are expected to be non-negative integer level codes stored in a
    float matrix (the library-wide convention). Entries are cast to ``int64``
    inside ``_eval``; level codes are not range-checked.

    Parameters
    ----------
    input_dim : int
        Total number of columns in the design matrix.
    active_dims : sequence of int, optional
        Columns of ``X`` holding categorical level codes. Defaults to all
        columns.
    """

    def __init__(self, input_dim, active_dims=None):
        """Validate dimensions via the base class."""
        super().__init__(input_dim, active_dims)

    def _eval(self, X, Y):
        """Mean of elementwise equality across active columns."""
        Xa = pt.cast(X[:, self.active_dims], "int64")
        Ya = pt.cast(Y[:, self.active_dims], "int64")
        eq = pt.eq(Xa[:, None, :], Ya[None, :, :])
        return pt.mean(pt.cast(eq, "float64"), axis=-1)

    def diag(self, X):
        """Diagonal of K(X, X). k(x, x) = 1 since every level matches itself."""
        return pt.ones(X.shape[0])


class LowRankCategorical(Kernel):
    """Categorical kernel with a low-rank-plus-diagonal level covariance.

    k(x, y) = B[x, y] with B = W @ W.T + diag(kappa)

    Also known as the Coregion / Intrinsic Coregionalization Model (ICM)
    kernel. Operates on a single categorical column. Level codes are expected
    to be non-negative integers stored in a float matrix (the library-wide
    convention). Codes are cast to ``int64`` inside ``_eval`` and ``diag``;
    they are not range-checked — passing a code >= ``num_levels`` is user
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
        """Validate dimensions and store the factor loadings."""
        super().__init__(input_dim, active_dims)
        if len(self.active_dims) != 1:
            raise ValueError("LowRankCategorical requires active_dims of length 1")
        self.num_levels = num_levels
        self.W = W
        self.kappa = kappa

    def _eval(self, X, Y):
        """Gather ``B[xi, yi]`` for level codes in the active column."""
        xi = pt.cast(X[:, self.active_dims[0]], "int64")
        yi = pt.cast(Y[:, self.active_dims[0]], "int64")
        B = self.W @ self.W.T + pt.diag(self.kappa)
        return B[xi][:, yi]

    def diag(self, X):
        """Diagonal of K(X, X). k(x, x) = ||W[x]||**2 + kappa[x]."""
        xi = pt.cast(X[:, self.active_dims[0]], "int64")
        Bdiag = pt.sum(pt.square(self.W), axis=-1) + self.kappa
        return Bdiag[xi]
