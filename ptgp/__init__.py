from ptgp import (
    gp,
    inducing,
    kernels,
    likelihoods,
    mean,
    objectives,
    optim,
    utils,
)
from ptgp.rewrites import assume  # also registers PyTensor rewrites at import

__all__ = [
    "assume",
    "gp",
    "inducing",
    "kernels",
    "likelihoods",
    "mean",
    "objectives",
    "optim",
    "utils",
]


try:
    from ptgp._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"
