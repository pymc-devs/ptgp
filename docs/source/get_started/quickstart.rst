Quickstart
==========

.. note::

   **WRITEME.** This page is a stub. Walk a new user end-to-end through
   fitting and predicting with one of the GP models. Cross-link to the
   :doc:`/examples/gallery` for the full notebooks.

The snippet below trains an SVGP on a small toy dataset.

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

    X_var = pt.matrix("X", shape=(None, 1))
    y_var = pt.vector("y", shape=(None,))

    step, shared_params, shared_extras = pg.optim.compile_training_step(
        lambda gp, X, y: pg.objectives.elbo(gp, X, y).elbo,
        svgp, X_var, y_var,
        model=model,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
        learning_rate=1e-2,
    )

    for _ in range(500):
        loss = step(X, y)

See :doc:`/examples/gallery` for the full end-to-end notebooks covering
``Unapproximated``, ``VFE``, and ``SVGP``.
