import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

import ptgp as pg

from ptgp.idata import to_idata


def _train_vfe(*, with_observed=True, with_data_container=True, n_data=20, M=4, maxiter=5):
    """Train a tiny VFE for a few scipy iterations.

    Toggle ``with_observed`` / ``with_data_container`` to exercise paths where
    the model has no observed RV or no ``pm.Data`` container.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(n_data, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(n_data)
    Z_init = np.linspace(-3, 3, M).reshape(-1, 1)

    coords = {"obs": np.arange(n_data), "feat": np.arange(1)} if with_data_container else None
    with pm.Model(coords=coords) as model:
        if with_data_container:
            pm.Data("X_data", X, dims=("obs", "feat"))
        if with_observed:
            pm.Normal(
                "y_obs",
                mu=0.0,
                sigma=1.0,
                observed=y,
                dims="obs" if with_data_container else None,
            )
        eta = pm.HalfNormal("eta", sigma=1.0)
        ls = pm.LogNormal("ls", mu=0.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        kernel = eta**2 * pg.kernels.ExpQuad(input_dim=1, ls=ls)
        Z_var = pt.matrix("Z", shape=(M, 1))
        vfe = pg.gp.VFE(kernel=kernel, sigma=sigma, inducing_variable=pg.inducing.Points(Z_var))

    X_var = pt.matrix("X_arg", shape=(None, 1))
    y_var = pt.vector("y_arg", shape=(None,))
    fun, theta0, unpack, sp, se = pg.optim.compile_scipy_objective(
        lambda gp, X, y: pg.objectives.collapsed_elbo(gp, X, y).elbo,
        vfe,
        X_var,
        y_var,
        model=model,
        extra_vars=[Z_var],
        extra_init=[Z_init],
    )
    diag_fn = pg.optim.compile_scipy_diagnostics(
        pg.objectives.vfe_diagnostics,
        vfe,
        X_var,
        y_var,
        model=model,
        extra_vars=[Z_var],
        extra_init=[Z_init],
    )
    result, history = pg.optim.tracked_minimize(
        fun,
        theta0,
        args=(X, y),
        diag_fn=diag_fn,
        options={"maxiter": maxiter},
    )
    unpack(result.x)
    return model, sp, se, result, history


def test_all_groups_present():
    model, sp, se, result, history = _train_vfe()
    idata = to_idata(sp, se, result=result, history=history, model=model)
    for group in (
        "posterior",
        "unconstrained_posterior",
        "optimizer_result",
        "observed_data",
        "constant_data",
    ):
        assert group in idata.children


def test_transforms_applied_to_constrained_posterior():
    """``posterior`` uses backward transform; ``unconstrained_posterior`` does not."""
    model, sp, se, *_ = _train_vfe()
    idata = to_idata(sp, se, model=model)
    for name in ("eta", "ls", "sigma"):
        constrained = float(idata.posterior[name].squeeze())
        unconstrained = float(idata.unconstrained_posterior[f"{name}_log__"].squeeze())
        assert np.isclose(constrained, np.exp(unconstrained))


def test_extras_appear_in_both_posterior_groups():
    """Non-PyMC extras have no transform; identical values appear in both groups."""
    model, sp, se, *_ = _train_vfe()
    idata = to_idata(sp, se, model=model)
    assert "Z" in idata.posterior.data_vars
    assert "Z" in idata.unconstrained_posterior.data_vars
    np.testing.assert_array_equal(
        idata.posterior["Z"].values, idata.unconstrained_posterior["Z"].values
    )


def test_optimizer_result_combines_scalars_and_trajectory():
    """Result scalars and history fields coexist in one Dataset."""
    model, sp, se, result, history = _train_vfe()
    idata = to_idata(sp, se, result=result, history=history, model=model)
    opt = idata.optimizer_result
    assert float(opt.fun) == pytest.approx(float(result.fun))
    assert bool(opt.success) == bool(result.success)
    assert int(opt.nit) == int(result.nit)
    # VFEDiagnostics fields land on the iteration dim.
    assert opt.sizes["iteration"] == len(history)
    assert opt.elbo.dims == ("iteration",)
    assert opt.trace_penalty.dims == ("iteration",)


def test_no_result_no_history_omits_optimizer_result():
    model, sp, se, *_ = _train_vfe()
    idata = to_idata(sp, se, model=model)
    assert "optimizer_result" not in idata.children


def test_history_without_result_keeps_only_trajectory():
    model, sp, se, _, history = _train_vfe()
    idata = to_idata(sp, se, history=history, model=model)
    opt = idata.optimizer_result
    assert "fun" not in opt.data_vars
    assert opt.sizes["iteration"] == len(history)


def test_result_without_history_keeps_only_scalars():
    model, sp, se, result, _ = _train_vfe()
    idata = to_idata(sp, se, result=result, model=model)
    opt = idata.optimizer_result
    assert "fun" in opt.data_vars
    assert "iteration" not in opt.sizes


def test_no_observed_data_omits_group():
    model, sp, se, *_ = _train_vfe(with_observed=False)
    idata = to_idata(sp, se, model=model)
    assert "observed_data" not in idata.children


def test_no_constant_data_omits_group():
    model, sp, se, *_ = _train_vfe(with_data_container=False)
    idata = to_idata(sp, se, model=model)
    assert "constant_data" not in idata.children


def test_named_dims_flow_through_to_groups():
    """Coords registered on the model propagate to observed_data and constant_data."""
    model, sp, se, *_ = _train_vfe()
    idata = to_idata(sp, se, model=model)
    assert "obs" in idata.observed_data.coords
    assert "obs" in idata.constant_data.coords
    assert idata.observed_data.y_obs.dims == ("obs",)


def test_phase_labels_attached_as_coord():
    model, sp, se, result, history = _train_vfe()
    labels = [f"phase{(i % 2) + 1}" for i in range(len(history))]
    idata = to_idata(sp, se, result=result, history=history, phase_labels=labels, model=model)
    assert "phase" in idata.optimizer_result.coords
    np.testing.assert_array_equal(idata.optimizer_result.phase.values, np.asarray(labels))


def test_phase_labels_length_mismatch_raises():
    model, sp, se, result, history = _train_vfe()
    with pytest.raises(ValueError, match="phase_labels"):
        to_idata(
            sp,
            se,
            result=result,
            history=history,
            phase_labels=["phase1"] * (len(history) - 1),
            model=model,
        )


def test_shared_extras_must_have_name():
    model, sp, *_ = _train_vfe()
    bad = pytensor.shared(np.zeros((3, 1)))
    with pytest.raises(ValueError, match=".name set"):
        to_idata(sp, [bad], model=model)


def test_shared_extras_duplicate_names_raise():
    model, sp, *_ = _train_vfe()
    a = pytensor.shared(np.zeros((3, 1)), name="dup")
    b = pytensor.shared(np.ones((3, 1)), name="dup")
    with pytest.raises(ValueError, match="duplicate"):
        to_idata(sp, [a, b], model=model)


def test_modelcontext_default():
    """No model= argument uses the active model context."""
    model, sp, se, *_ = _train_vfe()
    with model:
        idata = to_idata(sp, se)
    assert "posterior" in idata.children
    assert "eta" in idata.posterior.data_vars
