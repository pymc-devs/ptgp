import pytensor.tensor as pt

from ptgp.kernels.base import Kernel


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

    def __init__(self, input_dim, ls, active_dims=None):
        """Validate dimensions via the base class and store ``ls``."""
        super().__init__(input_dim, active_dims)
        self.ls = ls

    def _slice_input(self, X):
        return X[:, self.active_dims]

    def _scaled_sq_dist(self, X, Y):
        X = self._slice_input(X) / self.ls
        Y = self._slice_input(Y) / self.ls
        return _squared_distance(X, Y)

    def _scaled_euclid_dist(self, X, Y):
        X = self._slice_input(X) / self.ls
        Y = self._slice_input(Y) / self.ls
        return _euclidean_distance(X, Y)

    def diag(self, X):
        """Diagonal of K(X, X). For any stationary kernel this is ones."""
        return pt.ones(X.shape[0])


class ExpQuad(Stationary):
    """Exponentiated quadratic (RBF / squared exponential) kernel.

    k(x, y) = exp(-0.5 * ||x - y||^2 / ls^2)

    Scale with multiplication: eta**2 * ExpQuad(input_dim=..., ls=ls)
    """

    def _eval(self, X, Y):
        return pt.exp(-0.5 * self._scaled_sq_dist(X, Y))


class Matern52(Stationary):
    """Matern 5/2 kernel.

    k(x, y) = (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
    where r = ||x - y|| / ls
    """

    def _eval(self, X, Y):
        tau = self._scaled_euclid_dist(X, Y)
        sqrt5 = pt.sqrt(5.0)
        return (1.0 + sqrt5 * tau + 5.0 / 3.0 * pt.square(tau)) * pt.exp(-sqrt5 * tau)


class Matern32(Stationary):
    """Matern 3/2 kernel.

    k(x, y) = (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
    where r = ||x - y|| / ls
    """

    def _eval(self, X, Y):
        tau = self._scaled_euclid_dist(X, Y)
        sqrt3 = pt.sqrt(3.0)
        return (1.0 + sqrt3 * tau) * pt.exp(-sqrt3 * tau)


class Matern12(Stationary):
    """Matern 1/2 (Ornstein-Uhlenbeck) kernel.

    k(x, y) = exp(-r)
    where r = ||x - y|| / ls
    """

    def _eval(self, X, Y):
        tau = self._scaled_euclid_dist(X, Y)
        return pt.exp(-tau)
