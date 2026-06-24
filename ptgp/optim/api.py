from typing import Any, NamedTuple

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.optimize

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
    objective = getattr(gp_model, "default_objective", None)
    if objective is None:
        raise ValueError(
            f"{type(gp_model).__name__} has no default_objective; pass objective= explicitly."
        )
    return objective


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
    """Estimation method for GP models.

    Estimation uses reasonable defaults based on the provided approximation. For
    fine-grained control, use the low-level API.

    Parameters
    ----------
    gp_model : Unapproximated, VFE, or SVGP
        The PTGP model whose hyperparameters are PyMC RVs. Must expose a
        ``default_objective`` attribute or be passed ``objective=``.
    X : ndarray, shape (N,) or (N, D)
        Training inputs. 1D inputs are reshaped to ``(N, 1)``.
    y : ndarray, shape (N,) or (N, 1)
        Training targets. Column vectors are squeezed to 1D for the caller.
    model : pm.Model, optional
        Defaults to the enclosing ``with pm.Model()`` context.
    objective : callable, optional
        ``(gp_model, X_var, y_var) -> scalar | namedtuple``. Defaults to
        ``gp_model.default_objective``.
    method : str
        Optimization method passed to :func:`scipy.optimize.minimize`. Must
        accept a Jacobian; ``fit`` always supplies one (``jac=True``).
        Default ``"L-BFGS-B"``.
    init : str
        Strategy for the initial parameter vector. One of:

        - ``"prior_median"`` (default): median of each prior via
          ``pm.icdf(rv, 0.5)``, with a 500-sample fallback when icdf is
          unimplemented and a per-RV fallback to PyMC's initial point for
          improper priors.
        - ``"prior_draw"``: one draw from each prior. Stochastic; pin with
          ``init_rng`` for reproducibility.
        - ``"unconstrained_zero"``: PyMC's ``model.initial_point()`` (0 in
          unconstrained space unless ``initval=`` was set on the RV).
    init_rng : int or numpy.random.Generator, optional
        Seed for the sampling-based portions of ``"prior_median"`` and
        ``"prior_draw"``. No effect under ``"unconstrained_zero"``.
    compile_kwargs : dict, optional
        Forwarded as ``**compile_kwargs`` to ``pytensor.function`` when
        compiling the loss+grad. Use to set ``mode`` (e.g. ``"NUMBA"``,
        ``"JAX"``) or other compile-time options.
    **scipy_kwargs
        Additional keyword arguments forwarded to
        :func:`scipy.optimize.minimize` (e.g. ``tol``, ``options``,
        ``bounds``). Do not pass ``jac``; it is set internally.

    Returns
    -------
    FitResult
        Namedtuple with the scipy result, trained parameters in constrained
        space, shared variables for prediction handoff, and the PyMC model.
    """
    model = pm.modelcontext(model)
    if objective is None:
        objective = _default_objective(gp_model)

    X = _as_2d(X)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"y must be 1D or a column vector; got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y disagree on number of observations: "
            f"X.shape[0]={X.shape[0]} vs y.shape[0]={y.shape[0]}."
        )
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

    ``X_train`` / ``y_train`` are required when
    ``gp_model.predict_needs_data`` is True (``Unapproximated``, ``VFE``)
    and ignored otherwise (``SVGP``).

    Parameters
    ----------
    gp_model : Unapproximated, VFE, or SVGP
        The same PTGP model used to produce ``fit_result``.
    X_new : ndarray, shape (M,) or (M, D)
        Prediction inputs. 1D arrays are reshaped to ``(M, 1)``.
    fit_result : FitResult
        Output of :func:`fit`.
    X_train, y_train : ndarray, optional
        Required when ``gp_model.predict_needs_data`` is True.
    incl_lik : bool
        If True, add likelihood noise to the predictive variance.
    compile_kwargs : dict, optional
        Forwarded as ``**compile_kwargs`` to ``pytensor.function`` when
        compiling the predict graph.

    Returns
    -------
    mean : ndarray, shape (M,)
        Posterior mean at each row of ``X_new``.
    var : ndarray, shape (M,)
        Posterior marginal variance at each row of ``X_new``.
    """
    X_new = _as_2d(X_new)
    D = X_new.shape[1]
    X_new_var = pt.matrix("X_new", shape=(None, D))

    if getattr(gp_model, "predict_needs_data", True):
        if X_train is None or y_train is None:
            raise ValueError(
                f"{type(gp_model).__name__}.predict requires X_train and y_train; "
                "the conditional posterior needs the training data."
            )
        X_train_arg = _as_2d(X_train)
        y_train_arg = np.asarray(y_train, dtype=np.float64)
    else:
        X_train_arg = y_train_arg = None

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
