import pytensor.tensor as pt

from ptgp.kernels.base import Kernel
from ptgp.kernels.stationary import _squared_distance


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
        Center subtracted from the inputs. Scalar applies to every active
        column; a length-``len(active_dims)`` vector applies per-column.
    active_dims : sequence of int, optional
        Columns of ``X`` this kernel operates on. Defaults to all columns.
    """

    def __init__(self, input_dim, c=0.0, active_dims=None):
        """Validate dimensions via the base class and store ``c``."""
        super().__init__(input_dim, active_dims)
        self.c = c

    def _eval(self, X, Y):
        """Centered dot product: ``(X - c) @ (Y - c).T``."""
        Xc = X[:, self.active_dims] - self.c
        Yc = Y[:, self.active_dims] - self.c
        return Xc @ Yc.T

    def diag(self, X):
        """Diagonal of K(X, X). k(x, x) = ||x - c||^2."""
        Xc = X[:, self.active_dims] - self.c
        return pt.sum(pt.square(Xc), axis=-1)


class RandomWalk(Kernel):
    """Random walk (Brownian motion / Wiener process) kernel.

    k(x, y) = min(x, y)

    Defined for positive inputs. Operates on a single input column.
    Scale with multiplication: eta**2 * RandomWalk()

    Parameters
    ----------
    input_dim : int
        Number of columns of ``X`` the kernel expects.
    active_dims : sequence of int, optional
        Column of ``X`` to operate on. Must have length 1. Required when
        ``input_dim > 1``; defaults to ``[0]`` when ``input_dim == 1``.
    """

    def __init__(self, input_dim=1, active_dims=None):
        """Require a single active column; any ``input_dim`` is allowed."""
        if active_dims is None and input_dim == 1:
            active_dims = [0]
        super().__init__(input_dim=input_dim, active_dims=active_dims)
        if len(self.active_dims) != 1:
            raise ValueError(
                "RandomWalk kernel requires active_dims of length 1 when input_dim > 1."
            )

    def _eval(self, X, Y):
        """Pairwise min over the active dim."""
        Xa = X[:, self.active_dims]
        Ya = Y[:, self.active_dims]
        return pt.minimum(Xa, Ya.T)

    def diag(self, X):
        """Diagonal of K(X, X). min(x, x) = x."""
        return X[:, self.active_dims[0]]


class Gibbs(Kernel):
    """Gibbs kernel with a location-dependent lengthscale function.

    k(x, y) = sqrt(2 * l(x) * l(y) / (l(x)^2 + l(y)^2))
             * exp(-(x - y)^2 / (l(x)^2 + l(y)^2))

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
        """Require a single active column; any ``input_dim`` is allowed."""
        if active_dims is None and input_dim == 1:
            active_dims = [0]
        super().__init__(input_dim=input_dim, active_dims=active_dims)
        if len(self.active_dims) != 1:
            raise ValueError("Gibbs kernel requires active_dims of length 1 when input_dim > 1.")
        if not callable(lengthscale_func):
            raise TypeError("lengthscale_func must be callable")
        self.lengthscale_func = lengthscale_func

    def _eval(self, X, Y):
        """Evaluate the Gibbs kernel with per-point lengthscales."""
        Xa = X[:, self.active_dims]
        Ya = Y[:, self.active_dims]
        rx = self.lengthscale_func(Xa)
        ry = self.lengthscale_func(Ya)
        r2 = _squared_distance(Xa, Ya)
        rx2 = pt.reshape(pt.square(rx), (-1, 1))
        ry2 = pt.reshape(pt.square(ry), (1, -1))
        denom = rx2 + ry2
        return pt.sqrt(2.0 * pt.outer(rx, ry) / denom) * pt.exp(-r2 / denom)

    def diag(self, X):
        """Diagonal of K(X, X). k(x, x) = 1 for any lengthscale function."""
        return pt.ones(X.shape[0])


class WarpedInput(Kernel):
    """Warp the inputs of another kernel with an arbitrary function.

    k(x, y) = k_inner(w(x), w(y))

    Parameters
    ----------
    input_dim : int
        Number of columns of ``X`` this kernel expects. May differ from
        ``kernel_func.input_dim``; the warp may change dimensionality.
    kernel_func : Kernel
        Inner kernel applied to the warped inputs.
    warp_func : callable
        ``warp_func(X) -> X_warped``. PyTensor function.
    active_dims : sequence of int, optional
        Columns of ``X`` to warp. Defaults to all.
    """

    def __init__(self, input_dim, kernel_func, warp_func, active_dims=None):
        """Validate dimensions via the base class and store the inner kernel and warp."""
        super().__init__(input_dim, active_dims)
        if not isinstance(kernel_func, Kernel):
            raise TypeError("kernel_func must be a Kernel")
        if not callable(warp_func):
            raise TypeError("warp_func must be callable")
        self.kernel_func = kernel_func
        self.warp_func = warp_func

    def _eval(self, X, Y):
        """Warp both inputs, then delegate to the inner kernel."""
        Xw = self.warp_func(X[:, self.active_dims])
        Yw = self.warp_func(Y[:, self.active_dims])
        return self.kernel_func(Xw, Yw)

    def diag(self, X):
        """Diagonal of K(X, X): warp the inputs, then delegate to inner kernel."""
        return self.kernel_func.diag(self.warp_func(X[:, self.active_dims]))
