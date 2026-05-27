"""One-shot ``fit`` and ``predict`` for the common case.

Drop down to :func:`compile_scipy_objective` / :func:`compile_predict`
for staged training, custom placeholders, or per-group learning rates.
"""

from typing import Any, NamedTuple

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.optimize

from ptgp.objectives import collapsed_elbo, elbo, marginal_log_likelihood
from ptgp.optim.training import (
    compile_predict,
    compile_scipy_objective,
    get_trained_params,
)


class FitResult(NamedTuple):
    result: Any
    params: dict
    shared_params: dict
    shared_extras: tuple
    model: Any


def _default_objective(gp_model):
    from ptgp.gp import SVGP, VFE, Unapproximated

    if isinstance(gp_model, Unapproximated):
        return marginal_log_likelihood
    if isinstance(gp_model, VFE):
        return collapsed_elbo
    if isinstance(gp_model, SVGP):
        return elbo
    raise ValueError(
        f"No default objective for {type(gp_model).__name__}; pass objective= explicitly."
    )


def _as_2d(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    return X


def fit(
    gp_model,
    X,
    y,
    *,
    model=None,
    objective=None,
    method="L-BFGS-B",
    init="prior_median",
    init_rng=None,
    compile_kwargs=None,
    **scipy_kwargs,
):
    """One-shot fit for ``Unapproximated`` / ``VFE`` / ``SVGP``.

    Parameters
    ----------
    gp_model : Unapproximated, VFE, or SVGP
    X : ndarray, shape (N,) or (N, D)
    y : ndarray, shape (N,)
    model : pm.Model, optional
        Defaults to the enclosing ``with pm.Model()`` context.
    objective : callable, optional
        Defaults to ``marginal_log_likelihood`` / ``collapsed_elbo`` /
        ``elbo`` for ``Unapproximated`` / ``VFE`` / ``SVGP``.
    method : str
        Forwarded to :func:`scipy.optimize.minimize`.
    init, init_rng, compile_kwargs :
        Forwarded to :func:`compile_scipy_objective`.
    **scipy_kwargs
        Forwarded to :func:`scipy.optimize.minimize`.

    Returns
    -------
    FitResult
    """
    model = pm.modelcontext(model)
    if objective is None:
        objective = _default_objective(gp_model)

    X = _as_2d(X)
    y = np.asarray(y, dtype=np.float64)
    D = X.shape[1]

    X_var = pt.matrix("X", shape=(None, D))
    y_var = pt.vector("y", shape=(None,))

    fun, theta0, unpack, shared_params, shared_extras = compile_scipy_objective(
        objective,
        gp_model,
        X_var,
        y_var,
        model=model,
        init=init,
        init_rng=init_rng,
        compile_kwargs=compile_kwargs,
    )

    result = scipy.optimize.minimize(
        fun, theta0, args=(X, y), jac=True, method=method, **scipy_kwargs
    )
    unpack(result.x)

    return FitResult(
        result=result,
        params=get_trained_params(model, shared_params),
        shared_params=shared_params,
        shared_extras=tuple(shared_extras),
        model=model,
    )


def predict(
    gp_model,
    X_new,
    fit_result,
    *,
    X_train=None,
    y_train=None,
    incl_lik=False,
    compile_kwargs=None,
):
    """Posterior mean and variance at ``X_new``.

    ``X_train`` / ``y_train`` are required for ``Unapproximated`` and
    ``VFE``; ignored for ``SVGP``.
    """
    from ptgp.gp import SVGP

    X_new = _as_2d(X_new)
    D = X_new.shape[1]
    X_new_var = pt.matrix("X_new", shape=(None, D))

    if isinstance(gp_model, SVGP):
        X_train_arg = y_train_arg = None
    else:
        if X_train is None or y_train is None:
            raise ValueError(
                f"{type(gp_model).__name__}.predict requires X_train and y_train; "
                "the conditional posterior needs the training data."
            )
        X_train_arg = _as_2d(X_train)
        y_train_arg = np.asarray(y_train, dtype=np.float64)

    pred = compile_predict(
        gp_model,
        X_new_var,
        fit_result.model,
        fit_result.shared_params,
        shared_extras=fit_result.shared_extras,
        X_train=X_train_arg,
        y_train=y_train_arg,
        incl_lik=incl_lik,
        compile_kwargs=compile_kwargs,
    )
    return pred(X_new)
