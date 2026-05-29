Design Overview
===============

.. note::

   **WRITEME.** This page is a stub. The full design notes live in
   ``DESIGN.md`` at the repo root; port the relevant sections here and
   prune anything that's no longer accurate.

ptgp is built on PyTensor's symbolic graph and rewrite system. Kernels,
likelihoods, and GP models return symbolic PyTensor tensors with naive
linear algebra (``pt.linalg.inv(K)``, ``pt.linalg.slogdet(K)``) that
PyTensor's rewrite system lowers to efficient, Cholesky-based code using
declared matrix properties.

Key design choices:

- **PyMC priors only.** ``pm.Model()`` is used as a prior container; ptgp
  never invokes a PyMC sampler.
- **Naive linear algebra in user code.** Don't hand-roll Cholesky chains —
  write the math symbolically and let the rewrite system specialize.
- **Native kernels.** Kernels are implemented in ptgp, not reused from
  PyMC. Long-term goal is for PyMC to depend on ptgp for kernels.
- **Standalone objectives.** Loss functions are free functions, one per
  model, with a uniform ``(model, X, y)`` signature suitable for the
  training compilers.
