ptgp
====

A Gaussian process library for building GP models that solve real-world
problems. Built on PyTensor's symbolic graph and rewrite system, with PyMC
priors and native optimizers.

ptgp ships exact GPs, sparse VFE, SVGP with minibatch training, and Fourier
features; a full kernel library with composition and ``active_dims``;
non-Gaussian likelihoods; and a training toolbox with L-BFGS-B and Adam,
per-parameter learning rates, staged optimization, and inducing-point
initialization.

Quick install
-------------

.. code-block:: bash

    pip install -e .

Requires PyTensor from ``main`` and PyMC ``>= 6.0``. See the
:doc:`installation guide <get_started/install>` for details.

Quick example
-------------

.. code-block:: python

    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt
    import ptgp as pg

    X = np.random.randn(200, 1)
    y = np.sin(X.ravel()) + 0.1 * np.random.randn(200)
    Z = np.linspace(-2, 2, 20)[:, None]

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)

        vp = pg.gp.init_variational_params(M=20)
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.Gaussian(sigma=0.1),
            inducing_variable=pg.inducing.Points(pt.as_tensor(Z)),
            variational_params=vp,
        )

See the :doc:`example gallery <examples/gallery>` for full end-to-end
walkthroughs.

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   get_started/index
   user_guide/index
   kernels/gallery
   examples/gallery
   api
   dev/index
   release/index
