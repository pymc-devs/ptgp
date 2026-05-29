Objectives
==========

.. currentmodule:: ptgp.objectives

Each objective pairs with a specific GP model and returns a symbolic loss
suitable for the training compilers in :mod:`ptgp.optim`.

.. autosummary::
    :toctree: generated/

    marginal_log_likelihood
    elbo
    collapsed_elbo
    fitc_log_marginal_likelihood
    dpp_regularizer
    vfe_diagnostics
