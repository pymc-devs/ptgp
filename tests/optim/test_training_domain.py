"""Tests for the auto-installed VFF domain check on every compile entry point."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

import ptgp as pg

from ptgp.gp.svgp import init_variational_params
from ptgp.inducing_fourier import FourierFeatures1D
from ptgp.objectives import elbo
from ptgp.optim.training import (
    compile_predict,
    compile_scipy_objective,
    compile_training_step,
)


def _build_vff_svgp(num_frequencies=3):
    f = FourierFeatures1D(a=0.0, b=1.0, num_frequencies=num_frequencies)
    vp = init_variational_params(f.num_inducing)
    with pm.Model() as model:
        ls = pm.Exponential("ls", lam=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        kernel = eta**2 * pg.kernels.Matern32(input_dim=1, ls=ls)
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.Gaussian(sigma=0.1),
            inducing_variable=f,
            variational_params=vp,
            whiten=True,
        )
    return model, svgp, vp


def _build_vff_svgp_matern52_extrapolating(num_frequencies=3):
    f = FourierFeatures1D(
        a=0.0,
        b=1.0,
        num_frequencies=num_frequencies,
        allow_extrapolation=True,
    )
    vp = init_variational_params(f.num_inducing)
    with pm.Model() as model:
        ls = pm.Exponential("ls", lam=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.Gaussian(sigma=0.1),
            inducing_variable=f,
            variational_params=vp,
            whiten=True,
        )
    return model, svgp, vp


def _elbo_scalar(gp, X, y):
    return elbo(gp, X, y).elbo


def test_compile_training_step_preserves_tuple_arity():
    model, svgp, vp = _build_vff_svgp()
    X_var = pt.matrix("X")
    y_var = pt.vector("y")
    out = compile_training_step(
        _elbo_scalar,
        svgp,
        X_var,
        y_var,
        model=model,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
        learning_rate=1e-2,
    )
    assert isinstance(out, tuple) and len(out) == 3


def test_compile_training_step_domain_check_fires():
    model, svgp, vp = _build_vff_svgp()
    X_var = pt.matrix("X")
    y_var = pt.vector("y")
    step, *_ = compile_training_step(
        _elbo_scalar,
        svgp,
        X_var,
        y_var,
        model=model,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
        learning_rate=1e-2,
    )
    X_bad = np.array([[5.0], [10.0]])
    y = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="domain"):
        step(X_bad, y)


def test_compile_predict_domain_check():
    model, svgp, vp = _build_vff_svgp()
    X_var = pt.matrix("X")
    y_var = pt.vector("y")
    _, shared_params, shared_extras = compile_training_step(
        _elbo_scalar,
        svgp,
        X_var,
        y_var,
        model=model,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
        learning_rate=1e-2,
    )
    X_new_var = pt.matrix("X_new")
    pred = compile_predict(
        svgp,
        X_new_var,
        model,
        shared_params,
        extra_vars=vp.extra_vars,
        shared_extras=shared_extras,
    )
    with pytest.raises(ValueError, match="domain"):
        pred(np.array([[5.0]]))


def test_compile_predict_rejects_matern52_extrapolation_even_when_opted_out():
    model, svgp, vp = _build_vff_svgp_matern52_extrapolating()
    X_var = pt.matrix("X")
    y_var = pt.vector("y")
    _, shared_params, shared_extras = compile_training_step(
        _elbo_scalar,
        svgp,
        X_var,
        y_var,
        model=model,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
        learning_rate=1e-2,
    )
    X_new_var = pt.matrix("X_new")
    pred = compile_predict(
        svgp,
        X_new_var,
        model,
        shared_params,
        extra_vars=vp.extra_vars,
        shared_extras=shared_extras,
    )
    with pytest.raises(ValueError, match=r"Matern52.*outside"):
        pred(np.array([[5.0]]))


def test_compile_scipy_preserves_5_tuple():
    model, svgp, vp = _build_vff_svgp()
    X_var = pt.matrix("X")
    y_var = pt.vector("y")
    out = compile_scipy_objective(
        _elbo_scalar,
        svgp,
        X_var,
        y_var,
        model=model,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
    )
    assert isinstance(out, tuple) and len(out) == 5


def test_compile_scipy_validates_X_not_theta():
    model, svgp, vp = _build_vff_svgp()
    X_var = pt.matrix("X")
    y_var = pt.vector("y")
    fun, theta0, *_ = compile_scipy_objective(
        _elbo_scalar,
        svgp,
        X_var,
        y_var,
        model=model,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
    )
    X_in = np.array([[0.5]])
    y = np.array([0.0])
    fun(theta0, X_in, y)
    X_bad = np.array([[5.0]])
    with pytest.raises(ValueError, match="domain"):
        fun(theta0, X_bad, y)
