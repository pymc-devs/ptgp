"""Tests for LR schedules and per-group learning rates."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

import ptgp as pg

from ptgp.optim import schedules
from ptgp.optim.optimizers import adam


def _eval_at(schedule, t_value):
    t = pt.scalar("t", dtype="float64")
    fn = pytensor.function([t], schedule(t), on_unused_input="ignore")
    return float(fn(float(t_value)))


class TestSchedules:
    def test_constant(self):
        s = schedules.constant(0.05)
        assert _eval_at(s, 0) == pytest.approx(0.05)
        assert _eval_at(s, 10_000) == pytest.approx(0.05)

    def test_exponential_decay_halves_over_decay_steps(self):
        s = schedules.exponential_decay(1.0, decay_rate=0.5, decay_steps=100)
        assert _eval_at(s, 0) == pytest.approx(1.0)
        assert _eval_at(s, 100) == pytest.approx(0.5)
        assert _eval_at(s, 200) == pytest.approx(0.25)

    def test_cosine_endpoints(self):
        s = schedules.cosine(base=1.0, T_max=1000, lr_min=0.1)
        assert _eval_at(s, 0) == pytest.approx(1.0)
        assert _eval_at(s, 500) == pytest.approx(0.55, rel=1e-6)
        assert _eval_at(s, 1000) == pytest.approx(0.1, abs=1e-12)
        # clamps after T_max
        assert _eval_at(s, 2000) == pytest.approx(0.1, abs=1e-12)


class TestAdamPerGroup:
    def test_dict_without_groups_raises(self):
        x = pytensor.shared(np.zeros(3))
        loss = (x**2).sum()
        with pytest.raises(ValueError, match="param_groups is required"):
            adam(loss, [x], learning_rate={"a": 1e-2})

    def test_mismatched_keys_raises(self):
        x = pytensor.shared(np.zeros(3))
        loss = (x**2).sum()
        with pytest.raises(ValueError, match="do not match"):
            adam(
                loss,
                [x],
                learning_rate={"a": 1e-2},
                param_groups={"b": [x]},
            )

    def test_param_in_multiple_groups_raises(self):
        x = pytensor.shared(np.zeros(3))
        loss = (x**2).sum()
        with pytest.raises(ValueError, match="multiple groups"):
            adam(
                loss,
                [x],
                learning_rate={"a": 1e-2, "b": 1e-3},
                param_groups={"a": [x], "b": [x]},
            )

    def test_missing_param_raises(self):
        x = pytensor.shared(np.zeros(3))
        y = pytensor.shared(np.zeros(3))
        loss = (x**2).sum() + (y**2).sum()
        with pytest.raises(ValueError, match="not in any group"):
            adam(
                loss,
                [x, y],
                learning_rate={"a": 1e-2},
                param_groups={"a": [x]},
            )

    def test_per_group_lr_applied(self):
        """A group with lr=0 should not move its parameter."""
        x = pytensor.shared(np.ones(3))
        y = pytensor.shared(np.ones(3))
        loss = (x**2).sum() + (y**2).sum()
        updates = adam(
            loss,
            [x, y],
            learning_rate={"frozen": 0.0, "train": 1e-2},
            param_groups={"frozen": [x], "train": [y]},
        )
        step = pytensor.function([], [], updates=updates)
        for _ in range(20):
            step()
        np.testing.assert_array_equal(x.get_value(), np.ones(3))
        assert np.all(y.get_value() < 1.0)

    def test_schedule_lr_decays(self):
        """LR schedule shrinks steps over time."""
        x = pytensor.shared(np.array([10.0]))
        loss = (x**2).sum()
        updates = adam(
            loss,
            [x],
            learning_rate=schedules.exponential_decay(1e-1, decay_rate=0.5, decay_steps=5),
        )
        step = pytensor.function([], [], updates=updates)
        history = []
        prev = x.get_value()[0]
        for _ in range(30):
            step()
            curr = x.get_value()[0]
            history.append(abs(prev - curr))
            prev = curr
        # Later step sizes should be smaller than early ones
        assert np.mean(history[-5:]) < np.mean(history[:5])


class TestCompileTrainingStepGroups:
    def test_per_group_lr_via_compile(self):
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

        ls_vv = model.rvs_to_values[ls]
        eta_vv = model.rvs_to_values[eta]
        sigma_vv = model.rvs_to_values[sigma]

        train_step, shared_params, _ = pg.optim.compile_training_step(
            lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
            gp,
            X_var,
            y_var,
            model=model,
            param_groups={
                "kernel": [ls_vv, eta_vv],
                "noise": [sigma_vv],
            },
            learning_rate={
                "kernel": 1e-2,
                "noise": schedules.exponential_decay(1e-2, 0.9, 50),
            },
        )

        losses = [float(train_step(X, y)) for _ in range(100)]
        assert losses[-1] < losses[0]

    def test_unknown_var_in_group_raises(self):
        rng = np.random.default_rng(0)
        X = np.sort(rng.uniform(0, 5, 20))[:, None]
        np.sin(X[:, 0]) + rng.normal(0, 0.1, 20)

        with pm.Model() as model:
            ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.Exponential("sigma", lam=1.0)
            kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
            gp = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)

        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        stranger = pt.vector("stranger")

        with pytest.raises(ValueError, match="unknown variable"):
            pg.optim.compile_training_step(
                lambda gp, X, y: pg.objectives.marginal_log_likelihood(gp, X, y).mll,
                gp,
                X_var,
                y_var,
                model=model,
                param_groups={"all": [stranger]},
                learning_rate={"all": 1e-2},
            )
