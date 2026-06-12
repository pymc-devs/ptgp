Optimization
============

.. currentmodule:: ptgp.optim

Training compilers
------------------

.. autosummary::
    :toctree: generated/

    compile_training_step
    compile_scipy_objective
    compile_scipy_diagnostics
    compile_predict
    get_trained_params

Optimizers
----------

.. autosummary::
    :toctree: generated/

    adam
    sgd

Staged & tracked minimization
-----------------------------

.. autosummary::
    :toctree: generated/

    minimize_staged_vfe
    tracked_minimize
    phase_sort_key

Schedules
---------

.. currentmodule:: ptgp.optim.schedules

.. autosummary::
    :toctree: generated/

    constant
    exponential_decay
    cosine
