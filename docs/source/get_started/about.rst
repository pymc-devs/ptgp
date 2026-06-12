About ptgp
==========

.. note::

   **WRITEME.** This page is a stub. Fill in project motivation, scope, and
   how ptgp relates to the rest of the GP ecosystem (GPJax, GPflow,
   GPyTorch, PyMC's built-in GP).

ptgp is a Gaussian process library built on PyTensor's symbolic graph and
rewrite system. It is aimed at practitioners who need flexible,
well-supported GP modeling on real-world datasets — exact GPs, sparse
variational methods, non-Gaussian likelihoods, and PyMC priors on
hyperparameters.

Researchers benefit from the underlying design: ptgp models return symbolic
PyTensor tensors, so writing GP math directly (``pt.linalg.inv(K)``,
``pt.linalg.slogdet(K)``) lets the compiler pick efficient algorithms based
on declared matrix structure. This makes it straightforward to implement
new GP approximations and custom models.

ptgp distills approaches from existing GP libraries to make them more
accessible, drawing primarily from
`GPJax <https://github.com/JaxGaussianProcesses/GPJax>`_,
`GPflow <https://github.com/GPflow/GPflow>`_, and
`GPyTorch <https://github.com/cornellius-gp/gpytorch>`_.
