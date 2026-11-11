import pytensor
import pytensor.tensor as pt

from ptgp.kernels.base import Kernel, KernelOp, leaf_graph


def _squared_distance(X, Y):
    """Squared Euclidean distance between rows of X and Y.

    Parameters
    ----------
    X : tensor, shape (N, D)
    Y : tensor, shape (M, D)

    Returns
    -------
    tensor, shape (N, M)
    """
    X2 = pt.sum(pt.square(X), axis=-1, keepdims=True)
    Y2 = pt.sum(pt.square(Y), axis=-1, keepdims=True)
    XY = X @ Y.T
    return pt.maximum(X2 - 2.0 * XY + Y2.T, 0.0)


def _euclidean_distance(X, Y):
    """Euclidean distance between rows of X and Y.

    Clamps the squared distance from below to avoid NaN gradients at zero.
    """
    return pt.sqrt(pt.maximum(_squared_distance(X, Y), 1e-36))


class StationaryOp(KernelOp):
    """Stationary kernel k(x, y) = f(||x - y|| / ls), so k(x, x) = 1 for every lengthscale."""

    unit_diagonal = True


class ExpQuadOp(StationaryOp):
    def evaluate(self, left, right, ls):
        return pt.exp(-0.5 * _squared_distance(left / ls, right / ls))


class Matern52Op(StationaryOp):
    def evaluate(self, left, right, ls):
        tau = _euclidean_distance(left / ls, right / ls)
        sqrt5 = pt.constant(5.0**0.5, dtype=pytensor.config.floatX)
        five_thirds = pt.constant(5.0 / 3.0, dtype=pytensor.config.floatX)
        return (1.0 + sqrt5 * tau + five_thirds * pt.square(tau)) * pt.exp(-sqrt5 * tau)


class Matern32Op(StationaryOp):
    def evaluate(self, left, right, ls):
        tau = _euclidean_distance(left / ls, right / ls)
        sqrt3 = pt.constant(3.0**0.5, dtype=pytensor.config.floatX)
        return (1.0 + sqrt3 * tau) * pt.exp(-sqrt3 * tau)


class Matern12Op(StationaryOp):
    def evaluate(self, left, right, ls):
        return pt.exp(-_euclidean_distance(left / ls, right / ls))


class Stationary(Kernel):
    """Base class for stationary kernels k(x, y) = f(||x - y|| / ls).

    Parameters
    ----------
    input_dim : int
        Number of columns of ``X`` the kernel expects.
    ls : scalar or 1-D tensor
        Lengthscale. Scalar → isotropic. Length-``len(active_dims)`` vector →
        ARD (per-dimension lengthscales).
    active_dims : sequence of int, optional
        Columns of ``X`` this kernel operates on. Defaults to all columns.
    """

    op_type: type[StationaryOp]

    def __init__(self, input_dim, ls, active_dims=None):
        graph = leaf_graph(self.op_type, input_dim, active_dims, ls)
        super().__init__(input_dim=input_dim, graph=graph)


class ExpQuad(Stationary):
    """Exponentiated quadratic (RBF / squared exponential) kernel.

    k(x, y) = exp(-0.5 * ||x - y||^2 / ls^2)

    Scale with multiplication: eta**2 * ExpQuad(input_dim=..., ls=ls)
    """

    op_type = ExpQuadOp


class Matern52(Stationary):
    """Matern 5/2 kernel.

    k(x, y) = (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
    where r = ||x - y|| / ls
    """

    op_type = Matern52Op


class Matern32(Stationary):
    """Matern 3/2 kernel.

    k(x, y) = (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
    where r = ||x - y|| / ls
    """

    op_type = Matern32Op


class Matern12(Stationary):
    """Matern 1/2 (Ornstein-Uhlenbeck) kernel.

    k(x, y) = exp(-r)
    where r = ||x - y|| / ls
    """

    op_type = Matern12Op
