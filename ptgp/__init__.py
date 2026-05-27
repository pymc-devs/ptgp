import logging
import sys

from ptgp import (
    gp,
    inducing,
    kernels,
    likelihoods,
    mean,
    objectives,
    optim,
    rewrites,  # noqa: F401  (registers PyTensor rewrites at import)
    utils,
)
from ptgp.inducing_fourier import FourierFeatures1D
from ptgp.optim.api import FitResult, fit, predict



class _StdoutHandler(logging.StreamHandler):
    """Handler that resolves sys.stdout at emit time, not at init time."""

    def __init__(self):
        super().__init__()

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _):
        pass


_logger = logging.getLogger("ptgp")
if not _logger.handlers:
    _handler = _StdoutHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

__all__ = [
    "FourierFeatures1D",
    "FitResult",
    "fit",
    "gp",
    "inducing",
    "kernels",
    "likelihoods",
    "mean",
    "objectives",
    "optim",
    "predict",
    "utils",
]


try:
    from ptgp._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"
