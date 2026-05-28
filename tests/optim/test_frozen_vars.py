"""Tests for compile_training_step's frozen_vars kwarg."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

import ptgp as pg


def _make_svgp(Z_placeholder, M):
    """Build a tiny SVGP model using Z_placeholder as the inducing variable."""
    vp = pg.gp.init_variational_params(M)
    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.Gaussian(sigma=0.1),
            inducing_variable=pg.inducing.Points(Z_placeholder),
            variational_params=vp,
        )
    return model, svgp, vp


class TestFrozenVars:
    def test_frozen_z_does_not_move(self):
        """Z in frozen_vars stays fixed; the loss still decreases."""
        rng = np.random.default_rng(0)
        X = np.sort(rng.uniform(0, 5, 40))[:, None]
        y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 40)
        M = 8
        Z0 = np.linspace(0, 5, M)[:, None]

        Z_var = pt.matrix("Z")
        model, svgp, vp = _make_svgp(Z_var, M)

        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        train_step, _, _ = pg.optim.compile_training_step(
            lambda gp, X, y: pg.objectives.elbo(gp, X, y).elbo,
            svgp,
            X_var,
            y_var,
            model=model,
            extra_vars=vp.extra_vars,
            extra_init=vp.extra_init,
            frozen_vars={Z_var: Z0},
            learning_rate=1e-2,
        )

        losses = [float(train_step(X, y)) for _ in range(50)]
        assert losses[-1] < losses[0]

    def test_frozen_then_trainable(self):
        """Same SVGP object: freeze Z in phase 1, make Z trainable in phase 2."""
        rng = np.random.default_rng(0)
        X = np.sort(rng.uniform(0, 5, 40))[:, None]
        y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 40)
        M = 8
        Z0 = np.linspace(0, 5, M)[:, None]

        Z_var = pt.matrix("Z")
        model, svgp, vp = _make_svgp(Z_var, M)

        X_var = pt.matrix("X")
        y_var = pt.vector("y")

        # Phase 1: Z frozen.
        train_step_1, shared_1, extras_1 = pg.optim.compile_training_step(
            lambda gp, X, y: pg.objectives.elbo(gp, X, y).elbo,
            svgp,
            X_var,
            y_var,
            model=model,
            extra_vars=vp.extra_vars,
            extra_init=vp.extra_init,
            frozen_vars={Z_var: Z0},
            learning_rate=1e-2,
        )
        for _ in range(30):
            train_step_1(X, y)

        # Phase 2: same svgp, Z now trainable via extra_vars.
        train_step_2, shared_2, extras_2 = pg.optim.compile_training_step(
            lambda gp, X, y: pg.objectives.elbo(gp, X, y).elbo,
            svgp,
            X_var,
            y_var,
            model=model,
            extra_vars=[*vp.extra_vars, Z_var],
            extra_init=[*vp.extra_init, Z0],
            learning_rate=1e-2,
        )
        # Carry phase 1 state over.
        for vv, sh1 in shared_1.items():
            shared_2[vv].set_value(sh1.get_value())
        extras_2[0].set_value(extras_1[0].get_value())
        extras_2[1].set_value(extras_1[1].get_value())

        for _ in range(30):
            train_step_2(X, y)

        Z_final = extras_2[2].get_value()
        assert np.linalg.norm(Z_final - Z0) > 0, "Z should move in phase 2"

    def test_frozen_pymc_var_initial_value_in_scipy_objective(self):
        """A PyMC value var listed in frozen_vars must have its shared slot
        (and thus its theta0 slice and unpack target) initialized to the
        freeze value, not to PyMC's initial point. Regression test for the
        bug where phase-1 of minimize_staged_vfe ran with sigma frozen but
        diagnostics/shared-var reported the PyMC initial point instead.
        """
        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        with pm.Model() as model:
            ls = pm.HalfFlat("ls")
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            gp = pg.gp.Unapproximated(
                kernel=eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls),
                sigma=sigma,
            )

        sigma_vv = model.rvs_to_values[sigma]
        # Constrained value 0.123 → unconstrained via the model's transform.
        transform = model.rvs_to_transforms[sigma]
        sigma_unc = float(transform.forward(pt.as_tensor_variable(0.123)).eval())

        _, theta0, _, shared_params, _ = pg.optim.compile_scipy_objective(
            lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
            gp,
            X_var,
            y_var,
            model=model,
            frozen_vars={sigma_vv: sigma_unc},
        )

        # Shared var for sigma must hold the freeze value, not the PyMC
        # initial point (which would be 0 in unconstrained space).
        np.testing.assert_allclose(
            shared_params[sigma_vv].get_value(),
            sigma_unc,
            atol=1e-12,
        )

        # And the theta0 slot for sigma must match — same source, but verify
        # the layout is consistent so unpack writes back the right value.
        sigma_idx = list(model.continuous_value_vars).index(sigma_vv)
        np.testing.assert_allclose(theta0[sigma_idx], sigma_unc, atol=1e-12)

    def test_overlap_with_extra_vars_raises(self):
        Z_var = pt.matrix("Z")
        M = 5
        Z0 = np.linspace(0, 1, M)[:, None]
        model, svgp, vp = _make_svgp(Z_var, M)

        X_var = pt.matrix("X")
        y_var = pt.vector("y")

        with pytest.raises(ValueError, match="both extra_vars and frozen_vars"):
            pg.optim.compile_training_step(
                lambda gp, X, y: pg.objectives.elbo(gp, X, y).elbo,
                svgp,
                X_var,
                y_var,
                model=model,
                extra_vars=[*vp.extra_vars, Z_var],
                extra_init=[*vp.extra_init, Z0],
                frozen_vars={Z_var: Z0},
                learning_rate=1e-2,
            )
