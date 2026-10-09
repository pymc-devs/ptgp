"""Tests for per-group optimizers in compile_training_step.

Schedules and optimizer rules themselves are pytensor-ml's to test.
These cover only ptgp's group-resolution plumbing.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pytensor_ml.optim import adam, clip_by_global_norm, exponential_schedule, sgd
from pytensor_ml.params import StepCounter

import ptgp as pg


@pytest.fixture
def gp_setup():
    rng = np.random.default_rng(0)
    X = np.sort(rng.uniform(0, 5, 40))[:, None]
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 40)

    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        sigma = pm.Exponential("sigma", lam=1.0)
        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        gp = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)

    X_var = pt.matrix("X")
    y_var = pt.vector("y")
    return X, y, model, gp, X_var, y_var


def _mll(gp, X, y):
    return pg.objectives.marginal_log_likelihood(gp, X, y).mll


class TestGroupValidation:
    def test_dict_without_groups_raises(self, gp_setup):
        _, _, model, gp, X_var, y_var = gp_setup
        with pytest.raises(ValueError, match="param_groups was not given"):
            pg.optim.compile_training_step(
                _mll,
                gp,
                X_var,
                y_var,
                model=model,
                optimizer={"all": adam(1e-2)},
            )

    def test_groups_without_dict_raises(self, gp_setup):
        _, _, model, gp, X_var, y_var = gp_setup
        ls_vv = model.rvs_to_values[model["ls"]]
        with pytest.raises(ValueError, match="single transform"):
            pg.optim.compile_training_step(
                _mll,
                gp,
                X_var,
                y_var,
                model=model,
                optimizer=adam(1e-2),
                param_groups={"all": [ls_vv]},
            )

    def test_mismatched_keys_raises(self, gp_setup):
        _, _, model, gp, X_var, y_var = gp_setup
        ls_vv = model.rvs_to_values[model["ls"]]
        with pytest.raises(ValueError, match="do not match"):
            pg.optim.compile_training_step(
                _mll,
                gp,
                X_var,
                y_var,
                model=model,
                optimizer={"a": adam(1e-2)},
                param_groups={"b": [ls_vv]},
            )

    def test_unknown_var_in_group_raises(self, gp_setup):
        _, _, model, gp, X_var, y_var = gp_setup
        stranger = pt.vector("stranger")
        with pytest.raises(ValueError, match="unknown variable"):
            pg.optim.compile_training_step(
                _mll,
                gp,
                X_var,
                y_var,
                model=model,
                optimizer={"all": adam(1e-2)},
                param_groups={"all": [stranger]},
            )

    def test_param_in_multiple_groups_raises(self, gp_setup):
        _, _, model, gp, X_var, y_var = gp_setup
        ls_vv = model.rvs_to_values[model["ls"]]
        eta_vv = model.rvs_to_values[model["eta"]]
        sigma_vv = model.rvs_to_values[model["sigma"]]
        with pytest.raises(ValueError, match="multiple groups"):
            pg.optim.compile_training_step(
                _mll,
                gp,
                X_var,
                y_var,
                model=model,
                optimizer={"a": adam(1e-2), "b": adam(1e-2, namespace="b")},
                param_groups={"a": [ls_vv, eta_vv, sigma_vv], "b": [ls_vv]},
            )

    def test_uncovered_param_raises(self, gp_setup):
        _, _, model, gp, X_var, y_var = gp_setup
        ls_vv = model.rvs_to_values[model["ls"]]
        with pytest.raises(ValueError, match="No update reaches"):
            pg.optim.compile_training_step(
                _mll,
                gp,
                X_var,
                y_var,
                model=model,
                optimizer={"kernel": adam(1e-2)},
                param_groups={"kernel": [ls_vv]},
            )

    def test_gradients_only_group_raises(self, gp_setup):
        """A clip alone produces gradients, not steps. Compiling it would train uphill."""
        _, _, model, gp, X_var, y_var = gp_setup
        ls_vv = model.rvs_to_values[model["ls"]]
        eta_vv = model.rvs_to_values[model["eta"]]
        sigma_vv = model.rvs_to_values[model["sigma"]]

        with pytest.raises(ValueError, match="returned gradients rather than steps"):
            pg.optim.compile_training_step(
                _mll,
                gp,
                X_var,
                y_var,
                model=model,
                optimizer={"kernel": adam(1e-2), "noise": clip_by_global_norm(1.0)},
                param_groups={"kernel": [ls_vv, eta_vv], "noise": [sigma_vv]},
            )


class TestGroupTraining:
    def test_per_group_optimizers_train(self, gp_setup):
        X, y, model, gp, X_var, y_var = gp_setup
        ls_vv = model.rvs_to_values[model["ls"]]
        eta_vv = model.rvs_to_values[model["eta"]]
        sigma_vv = model.rvs_to_values[model["sigma"]]

        noise_schedule = exponential_schedule(1e-2, total_steps=100, final_learning_rate=1e-3)
        train_step, shared_params, _ = pg.optim.compile_training_step(
            _mll,
            gp,
            X_var,
            y_var,
            model=model,
            optimizer={
                "kernel": adam(1e-2, namespace="kernel"),
                "noise": adam(noise_schedule, namespace="noise"),
            },
            param_groups={"kernel": [ls_vv, eta_vv], "noise": [sigma_vv]},
        )

        losses = [float(train_step(X, y)) for _ in range(100)]
        assert losses[-1] < losses[0]

        clocks = {v.name: v for v in train_step.get_shared() if isinstance(v, StepCounter)}
        assert sorted(clocks) == ["kernel/step_count", "noise/step_count"]
        assert all(int(clock.get_value()) == 100 for clock in clocks.values())

    def test_default_namespace_collision_raises(self, gp_setup):
        _, _, model, gp, X_var, y_var = gp_setup
        ls_vv = model.rvs_to_values[model["ls"]]
        eta_vv = model.rvs_to_values[model["eta"]]
        sigma_vv = model.rvs_to_values[model["sigma"]]

        with pytest.raises(ValueError, match="adam/step_count"):
            pg.optim.compile_training_step(
                _mll,
                gp,
                X_var,
                y_var,
                model=model,
                optimizer={"kernel": adam(1e-2), "noise": adam(1e-3)},
                param_groups={"kernel": [ls_vv, eta_vv], "noise": [sigma_vv]},
            )

    def test_zero_rate_group_stays_put(self, gp_setup):
        X, y, model, gp, X_var, y_var = gp_setup
        ls_vv = model.rvs_to_values[model["ls"]]
        eta_vv = model.rvs_to_values[model["eta"]]
        sigma_vv = model.rvs_to_values[model["sigma"]]

        train_step, shared_params, _ = pg.optim.compile_training_step(
            _mll,
            gp,
            X_var,
            y_var,
            model=model,
            optimizer={"kernel": adam(1e-2), "noise": sgd(learning_rate=0.0)},
            param_groups={"kernel": [ls_vv, eta_vv], "noise": [sigma_vv]},
        )

        sigma_before = shared_params[sigma_vv].get_value().copy()
        ls_before = shared_params[ls_vv].get_value().copy()
        for _ in range(20):
            train_step(X, y)

        np.testing.assert_array_equal(shared_params[sigma_vv].get_value(), sigma_before)
        assert not np.allclose(shared_params[ls_vv].get_value(), ls_before)

    def test_extra_vars_in_group(self):
        """Variational params given as extra_vars resolve into their own group."""
        rng = np.random.default_rng(0)
        X = np.sort(rng.uniform(0, 5, 40))[:, None]
        y = np.sin(X[:, 0]) + rng.normal(0, 0.1, 40)
        M = 8
        Z_init = np.linspace(0, 5, M)[:, None]
        vp = pg.gp.init_variational_params(M)

        with pm.Model() as model:
            ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
            eta = pm.Exponential("eta", lam=1.0)
            kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
            svgp = pg.gp.SVGP(
                kernel=kernel,
                likelihood=pg.likelihoods.Gaussian(sigma=0.1),
                inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z_init)),
                variational_params=vp,
            )

        ls_vv = model.rvs_to_values[model["ls"]]
        eta_vv = model.rvs_to_values[model["eta"]]

        train_step, shared_params, shared_extras = pg.optim.compile_training_step(
            lambda gp, X_, y_: pg.objectives.elbo(gp, X_, y_).elbo,
            svgp,
            pt.matrix("X"),
            pt.vector("y"),
            model=model,
            extra_vars=vp.extra_vars,
            extra_init=vp.extra_init,
            optimizer={
                "hyper": adam(1e-2, namespace="hyper"),
                "variational": adam(5e-2, namespace="variational"),
            },
            param_groups={"hyper": [ls_vv, eta_vv], "variational": list(vp.extra_vars)},
        )

        q_mu_before = shared_extras[0].get_value().copy()
        losses = [float(train_step(X, y)) for _ in range(20)]

        assert losses[-1] < losses[0]
        assert not np.allclose(shared_extras[0].get_value(), q_mu_before)
