import numpy as np

from ptgp.rewrites import assume


class Kernel:
    """Base class for all PTGP kernels.

    Subclasses implement ``_eval(self, X, Y)`` (raw kernel math, both args
    always given) and ``diag(self, X)``. The base ``__call__`` handles the
    K(X, X) case and attaches symmetric/PSD assumptions.

    Parameters
    ----------
    input_dim : int
        Number of columns of ``X`` the kernel expects.
    active_dims : sequence of int, optional
        Columns of ``X`` this kernel operates on. Defaults to all columns.
    """

    def __init__(self, input_dim, active_dims=None):
        """Validate and store ``input_dim`` and ``active_dims``."""
        self.input_dim = input_dim
        if active_dims is None:
            self.active_dims = np.arange(input_dim)
        else:
            self.active_dims = np.asarray(active_dims, dtype=int)
            if self.active_dims.max() >= input_dim:
                raise ValueError(
                    f"active_dims contains index {int(self.active_dims.max())}, "
                    f"but input_dim is {input_dim}"
                )

    def __call__(self, X, Y=None):
        """K(X, Y); K(X, X) if Y is None, annotated symmetric and PSD."""
        if Y is None:
            K = self._eval(X, X)
            return assume(K, symmetric=True, positive_definite=True)
        return self._eval(X, Y)

    def _eval(self, X, Y):
        """Raw kernel matrix — subclasses implement."""
        raise NotImplementedError

    def diag(self, X):
        """Diagonal of K(X, X) — subclasses implement."""
        raise NotImplementedError

    def diag(self, X):
        """Diagonal of K(X, X). Default fallback: pt.diag(self(X))."""
        import pytensor.tensor as pt

        return pt.diag(self(X))

    def __add__(self, other):
        """Return a SumKernel combining self and other."""
        from ptgp.kernels.combination import SumKernel

        return SumKernel(self, other)

    def __mul__(self, other):
        """Return a ProductKernel combining self and other."""
        from ptgp.kernels.combination import ProductKernel

        return ProductKernel(self, other)

    def __rmul__(self, other):
        """Return a ProductKernel; supports ``scalar * kernel``."""
        from ptgp.kernels.combination import ProductKernel

        return ProductKernel(self, other)
