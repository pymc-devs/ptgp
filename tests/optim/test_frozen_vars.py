"""Tests for compile_training_step's frozen_vars kwarg."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pytensor_ml.optim import adam

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


def _pymc_frozen_setup():
    """Data and an SVGP with fixed Z, for freezing a PyMC hyperparameter."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (40, 1))
    y = np.sin(X[:, 0])
    Z0 = np.linspace(-2, 2, 6)[:, None]
    model, svgp, vp = _make_svgp(pt.as_tensor_variable(Z0), 6)
    return X, y, model, svgp, vp


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
            optimizer=adam(1e-2),
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
            optimizer=adam(1e-2),
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
            optimizer=adam(1e-2),
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

    def test_frozen_pymc_var_in_training_step(self):
        """A frozen PyMC value var enters the loss at its freeze value while the rest train."""
        X, y, model, svgp, vp = _pymc_frozen_setup()
        ls_vv = model.rvs_to_values[model["ls"]]
        eta_vv = model.rvs_to_values[model["eta"]]

        def compile_frozen(ls_value):
            return pg.optim.compile_training_step(
                pg.objectives.elbo,
                svgp,
                pt.matrix("X"),
                pt.vector("y"),
                model=model,
                optimizer=adam(1e-2),
                extra_vars=vp.extra_vars,
                extra_init=vp.extra_init,
                frozen_vars={ls_vv: ls_value},
            )

        train_step, shared, _ = compile_frozen(0.5)
        eta0 = shared[eta_vv].get_value().copy()
        losses = [float(train_step(X, y)) for _ in range(20)]
        assert losses[-1] < losses[0]
        assert not np.allclose(shared[eta_vv].get_value(), eta0)

        other_step, _, _ = compile_frozen(-1.0)
        assert not np.isclose(float(other_step(X, y)), losses[0])

    def test_frozen_pymc_var_with_param_groups(self):
        """A frozen value var left out of every param group compiles and trains."""
        X, y, model, svgp, vp = _pymc_frozen_setup()
        ls_vv = model.rvs_to_values[model["ls"]]
        eta_vv = model.rvs_to_values[model["eta"]]

        train_step, shared, _ = pg.optim.compile_training_step(
            pg.objectives.elbo,
            svgp,
            pt.matrix("X"),
            pt.vector("y"),
            model=model,
            optimizer={"hyper": adam(1e-2), "variational": adam(1e-2, namespace="variational")},
            param_groups={"hyper": [eta_vv], "variational": vp.extra_vars},
            extra_vars=vp.extra_vars,
            extra_init=vp.extra_init,
            frozen_vars={ls_vv: 0.5},
        )
        eta0 = shared[eta_vv].get_value().copy()
        losses = [float(train_step(X, y)) for _ in range(20)]
        assert losses[-1] < losses[0]
        assert not np.allclose(shared[eta_vv].get_value(), eta0)

    def test_frozen_var_in_param_group_raises(self):
        X, y, model, svgp, vp = _pymc_frozen_setup()
        ls_vv = model.rvs_to_values[model["ls"]]
        eta_vv = model.rvs_to_values[model["eta"]]

        with pytest.raises(ValueError, match="contains frozen variable"):
            pg.optim.compile_training_step(
                pg.objectives.elbo,
                svgp,
                pt.matrix("X"),
                pt.vector("y"),
                model=model,
                optimizer={"hyper": adam(1e-2), "variational": adam(1e-2, namespace="variational")},
                param_groups={"hyper": [ls_vv, eta_vv], "variational": vp.extra_vars},
                extra_vars=vp.extra_vars,
                extra_init=vp.extra_init,
                frozen_vars={ls_vv: 0.5},
            )

    def test_frozen_rv_key_raises(self):
        """Freezing the RV instead of its value var raises rather than silently training it."""
        X, y, model, svgp, vp = _pymc_frozen_setup()

        with pytest.raises(ValueError, match="are PyMC random variables"):
            pg.optim.compile_training_step(
                pg.objectives.elbo,
                svgp,
                pt.matrix("X"),
                pt.vector("y"),
                model=model,
                extra_vars=vp.extra_vars,
                extra_init=vp.extra_init,
                frozen_vars={model["ls"]: 0.5},
            )

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
                optimizer=adam(1e-2),
            )
