Training
========

.. note::

   **WRITEME.** This page is a stub. Cover the two training paths
   (``compile_training_step`` for Adam/SGD vs. ``compile_scipy_objective``
   for L-BFGS-B), MAP vs. pure-ELBO, ``param_groups`` for per-parameter
   learning rates, ``minimize_staged_vfe``, and ``compile_predict``.

ptgp offers two training paths, each matched to a different regime:

- :func:`~ptgp.optim.compile_training_step` — Adam / SGD via PyTensor
  shared variables, for stochastic or minibatch training (SVGP).
  Prediction reads the same shared variables.
- :func:`~ptgp.optim.compile_scipy_objective` — returns a ``(loss, grad)``
  callable for :func:`scipy.optimize.minimize`, for full-batch training
  (Unapproximated, VFE).

Priors regularize training by default: both compilers add
``model.logp(jacobian=True, sum=True)`` to the loss, yielding MAP in the
unconstrained value-var space. Pass ``include_prior=False`` for MLE / pure
ELBO.

Staged & tracked
----------------

- :func:`~ptgp.optim.minimize_staged_vfe` runs the VFE training schedule
  in phases (kernel-only, then jointly with inducing points, etc.).
- :func:`~ptgp.optim.tracked_minimize` wraps :func:`scipy.optimize.minimize`
  with per-iteration diagnostics.
