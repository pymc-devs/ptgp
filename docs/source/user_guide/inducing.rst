Inducing Points
===============

.. note::

   **WRITEME.** This page is a stub. Cover the three initialization
   strategies (random subsample, k-means, greedy variance), the
   diagnostics namedtuples they return, and when to prefer each.

ptgp provides three init routines for inducing point locations, each
paired with a diagnostics object:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Diagnostics
   * - :func:`~ptgp.inducing.random_subsample_init`
     - :class:`~ptgp.inducing.RandomSubsampleDiagnostics`
   * - :func:`~ptgp.inducing.kmeans_init`
     - :class:`~ptgp.inducing.KMeansDiagnostics`
   * - :func:`~ptgp.inducing.greedy_variance_init`
     - :class:`~ptgp.inducing.GreedyVarianceDiagnostics`

All three return a :class:`~ptgp.inducing.Points` object wrapping a numpy
array, so ``ip.Z`` is directly usable for plotting.
