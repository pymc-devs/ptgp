Optimization
============

.. currentmodule:: ptgp.optim

High-level API
--------------

.. autosummary::
    :toctree: generated/

    fit
    predict
    FitResult

Training compilers
------------------

.. autosummary::
    :toctree: generated/

    compile_training_step
    compile_scipy_objective
    compile_scipy_diagnostics
    compile_predict
    get_trained_params

Staged & tracked minimization
-----------------------------

.. autosummary::
    :toctree: generated/

    minimize_staged_vfe
    tracked_minimize
    phase_sort_key

Optimizers and schedules
------------------------

Optimizer rules, schedules, gradient clipping, and update transforms come
from `pytensor-ml <https://pytensor-ml.readthedocs.io>`_. Pass a configured
transform (e.g. ``pytensor_ml.optim.adam(1e-2)``) as the ``optimizer``
argument of :func:`ptgp.optim.compile_training_step`.
