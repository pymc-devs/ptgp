"""GP models as PyMC models, with the fitting objective read from the model.

- ``GP``: a process; ``f(X)`` is its values at ``X``, a random variable in a ``pm.Model``.
- ``objective(model, approx)``: named terms of a fitting criterion, read from the model.
- Families: ``Exact``, ``VFE``, ``FITC``, ``SVGP`` for GPs; ``MeanField``, ``Gaussian``, ``Point``.
- ``fit(obj)`` (scipy, L-BFGS-B) and ``compile_training_step(obj, optimizer, param_groups)``
  (pytensor-ml optimizers, per group; ``natural_gradient`` for an SVGP's q(u)): both give a
  ``Fit``, which holds the history and predicts.
"""

from .families import FITC, SVGP, VFE, Exact, Gaussian, GPFamily, MeanField, Point
from .fit import Fit, compile_training_step, fit, natural_gradient
from .model import GP, GPRV, find_gp_sums, processes
from .objective import Objective, objective

__all__ = [
    "FITC",
    "GP",
    "GPRV",
    "SVGP",
    "VFE",
    "Exact",
    "Fit",
    "GPFamily",
    "Gaussian",
    "MeanField",
    "Objective",
    "Point",
    "compile_training_step",
    "find_gp_sums",
    "fit",
    "natural_gradient",
    "objective",
    "processes",
]
