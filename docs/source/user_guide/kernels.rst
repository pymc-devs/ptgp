Kernels
=======

.. note::

   **WRITEME.** This page is a stub. Cover the kernel catalogue,
   composition via ``+`` and ``*``, ``active_dims``, the ``eta`` scale
   convention, and the ``pt.assume(K, symmetric=True, positive_definite=True)``
   annotation pattern.

ptgp kernels are implemented natively (not reused from PyMC) and operate on
matrix pairs. ``Kernel.__call__(X, Y=None)`` returns a symbolic PyTensor
tensor. When called with ``Y=None`` the kernel produces ``K(X, X)``
annotated as ``symmetric=True, positive_definite=True`` so downstream
rewrites can specialize.

Available families
------------------

- **Stationary**: :class:`~ptgp.kernels.ExpQuad`,
  :class:`~ptgp.kernels.Matern52`, :class:`~ptgp.kernels.Matern32`,
  :class:`~ptgp.kernels.Matern12`.
- **Non-stationary**: :class:`~ptgp.kernels.Gibbs`,
  :class:`~ptgp.kernels.RandomWalk`, :class:`~ptgp.kernels.WarpedInput`.
- **Categorical**: :class:`~ptgp.kernels.Overlap`,
  :class:`~ptgp.kernels.LowRankCategorical` for multi-class or categorical
  inputs.
- **Composition**: :class:`~ptgp.kernels.SumKernel`,
  :class:`~ptgp.kernels.ProductKernel`, produced by the ``+`` and ``*``
  operators on existing kernels.

Scale convention
----------------

Kernels are scaled by ``eta**2`` (e.g. ``eta**2 * ExpQuad(ls=ls)``), so
``eta`` is always squared.
