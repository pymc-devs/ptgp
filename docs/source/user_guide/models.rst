GP Models
=========

.. note::

   **WRITEME.** This page is a stub. Walk through ``Unapproximated``,
   ``VFE``, and ``SVGP`` — when to use each, what they assume about your
   data, and how to set them up. Cross-link to the corresponding objective
   in :mod:`ptgp.objectives`.

ptgp ships three user-facing GP models:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Model
     - Scale
     - Best for
   * - :class:`~ptgp.gp.Unapproximated`
     - ``N < ~2,000``
     - Exact inference, model comparison.
   * - :class:`~ptgp.gp.VFE`
     - ``N < ~50,000``
     - Medium-scale data with inducing points.
   * - :class:`~ptgp.gp.SVGP`
     - ``N`` up to ``~500,000``
     - Large data, non-Gaussian likelihoods, minibatch training.

A fourth model, :class:`~ptgp.inducing_fourier.FourierFeatures1D`, supplies a
structured ``K_uu`` for 1-D Matérn kernels via a Fourier basis and removes
the need to place inducing points by hand.
