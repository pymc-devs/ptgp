"""Diagnostic utilities for PTGP models."""

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_initial_params(model, init="prior_median", rng=None, n_median_samples=500):
    """Return constrained-space initial values for all free RVs in a PyMC model.

    Uses the same initialization strategies as ``compile_scipy_objective``.
    Use it to build proxy kernels with concrete float values to pass to
    ``greedy_variance_init``.

    Parameters
    ----------
    model : pm.Model
    init : str
        ``"prior_median"`` (default), ``"prior_draw"``, or
        ``"unconstrained_zero"``. See :func:`ptgp.optim.training._make_initial_point`
        for the per-RV fallback behaviour on improper priors.
    rng : int or numpy Generator, optional
    n_median_samples : int
        Number of draws used to estimate medians. Only used when
        ``init="prior_median"``.

    Returns
    -------
    dict
        ``{rv_name: constrained_value}`` for each free RV.  Scalar RVs are
        returned as plain Python floats; array RVs as numpy arrays.
    """
    import pymc as pm

    from ptgp.optim.training import _make_initial_point

    model = pm.modelcontext(model)
    ip = _make_initial_point(model, init=init, rng=rng, n_median_samples=n_median_samples)

    result = {}
    for rv in model.free_RVs:
        vv = model.rvs_to_values[rv]
        transform = model.rvs_to_transforms.get(rv)
        unconstrained = np.asarray(ip[vv.name], dtype=np.float64)
        if transform is not None:
            constrained = np.asarray(transform.backward(unconstrained).eval(), dtype=np.float64)
        else:
            constrained = unconstrained
        result[rv.name] = float(constrained) if constrained.ndim == 0 else constrained

    return result


_LARGE_GRAD_WARN = 1e4


def check_init(
    fun,
    theta0,
    X,
    y,
    model=None,
    extra_vars=None,
    extra_init=None,
    top_k=10,
):
    """Evaluate the compiled objective at theta0 and report whether the result is finite.

    Prints a one-line summary for the loss and grad, then lists the top ``top_k``
    largest ``|grad|`` components annotated with their variable names (when
    ``model`` is supplied).

    Parameters
    ----------
    fun : callable
        ``(theta, X, y) -> (loss, grad)`` as returned by
        ``compile_scipy_objective``.
    theta0 : ndarray
        Flat initial parameter vector (the ``theta0`` returned by
        ``compile_scipy_objective``).
    X : ndarray
        Training inputs.
    y : ndarray
        Training targets.
    model : pm.Model, optional
        PyMC model used in ``compile_scipy_objective``.  When given, PyMC
        value vars are labelled by name in the top-K table.
    extra_vars : list of TensorVariable, optional
        Extra symbolic variables passed to ``compile_scipy_objective``.
    extra_init : list of ndarray, optional
        Initial values for ``extra_vars``.  Required for labelling when
        ``extra_vars`` is provided.
    top_k : int
        Number of largest ``|grad|`` components to print.  Default 10.

    Returns
    -------
    bool
        True if both loss and all grad components are finite.
    """
    loss, grad = fun(theta0, X, y)
    loss = float(loss)
    grad = np.asarray(grad, dtype=np.float64)

    loss_ok = bool(np.isfinite(loss))
    grad_ok = bool(np.all(np.isfinite(grad)))
    max_g = float(np.abs(grad).max()) if grad.size > 0 else 0.0

    status = "OK" if loss_ok else "NaN/Inf -- BAD"
    logger.info(f"loss at init : {loss:.6g}  ({status})")
    logger.info(f"grad finite  : {grad_ok}  (max |g| = {max_g:.3g})")

    if loss_ok and grad_ok and max_g > _LARGE_GRAD_WARN:
        logger.warning(
            f"max |grad| = {max_g:.2e} exceeds {_LARGE_GRAD_WARN:.0e}"
            " -- may indicate a pathological initialization"
        )

    labels = _build_index_labels(theta0.size, model, extra_vars, extra_init)

    k = min(top_k, grad.size)
    top_idx = np.argsort(np.abs(grad))[::-1][:k]
    logger.info(f"\ntop-{k} |grad| components:")
    for i in top_idx:
        label = labels[i] if labels is not None else str(i)
        logger.info(f"  [{i:6d}]  {np.abs(grad[i]):.4g}   {label}")

    return loss_ok and grad_ok


def _build_index_labels(n_theta, model, extra_vars, extra_init):
    """Return a list of string labels, one per element of theta.

    Returns None if neither ``model`` nor labelled extras are available.
    """
    try:
        import pymc as pm
    except ImportError:
        return None

    if model is None and extra_vars is None:
        return None

    labels = []

    if model is not None:
        model = pm.modelcontext(model)
        ip = model.initial_point()
        for vv in model.continuous_value_vars:
            val = np.asarray(ip[vv.name])
            name = vv.name
            if val.size == 1:
                labels.append(name)
            else:
                for flat_i in range(val.size):
                    idx = np.unravel_index(flat_i, val.shape)
                    idx_str = ",".join(str(j) for j in idx)
                    labels.append(f"{name}[{idx_str}]")

    if extra_vars is not None and extra_init is not None:
        for var, init_val in zip(extra_vars, extra_init):
            init_val = np.asarray(init_val)
            name = getattr(var, "name", None) or repr(var)
            if init_val.size == 1:
                labels.append(name)
            else:
                for flat_i in range(init_val.size):
                    idx = np.unravel_index(flat_i, init_val.shape)
                    idx_str = ",".join(str(j) for j in idx)
                    labels.append(f"{name}[{idx_str}]")
    elif extra_vars is not None:
        # Names but no shapes -- fill remaining with var name + index
        n_remaining = n_theta - len(labels)
        if n_remaining > 0:
            for var in extra_vars:
                name = getattr(var, "name", None) or repr(var)
                labels.append(f"{name}[?]")

    if len(labels) != n_theta:
        # Shape mismatch -- fall back to plain indices rather than mislabelling.
        return None

    return labels


_FIT_PARAM_PREFIX = "p__"
_FIT_EXTRA_PREFIX = "e__"
_FIT_META_KEY = "__meta__"


def save_fit(path, shared_params, shared_extras=(), meta=None):
    """Save trained PTGP shared-variable values to a compressed ``.npz`` file.

    Stores the unconstrained values held by every entry of ``shared_params``
    and ``shared_extras`` under name-keyed slots, plus an optional JSON
    ``meta`` blob. Reload with :func:`load_fit` after rebuilding the model.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination ``.npz`` path. The ``.npz`` suffix is added by numpy if
        absent.
    shared_params : dict
        ``{value_var: shared_var}`` from ``compile_scipy_objective`` /
        ``compile_training_step`` / ``minimize_staged_vfe``.
    shared_extras : sequence of pytensor shared variables, optional
        From the same training call (e.g. ``Z``, ``q_mu``, ``q_sqrt``).
    meta : dict, optional
        Arbitrary JSON-serialisable metadata stored alongside the values.
        Recommended entries: ``as_of`` date, target column name, feature
        column list, model commit hash, fit timestamp. Read back as a dict
        by :func:`load_fit`.

    Raises
    ------
    ValueError
        If two shared variables share the same name within either bucket
        (would silently overwrite on save).

    Examples
    --------
    Saving after a VFE fit::

        result, history, labels, unpack, sp, se = minimize_staged_vfe(
            objective_fn,
            gp_model,
            X_var,
            y_var,
            X,
            y,
            model,
            sigma_init=0.1,
            Z_var=Z_var,
            Z_init=Z_init,
        )
        unpack(result.x)

        from ptgp.utils import save_fit

        save_fit(
            "fit_threefactor_2026-04-22.npz",
            sp,
            se,
            meta={
                "as_of": "2026-04-22",
                "target": "oas_sofr",
                "features": ["maturity_years", "leverage", "vol_1", "market_cap"],
                "ptgp_commit": "abc1234",
                "model": "threefactor_vfe",
            },
        )
    """
    blob = {}
    seen = set()
    for vv, sv in shared_params.items():
        name = vv.name
        if name in seen:
            raise ValueError(f"Duplicate shared_params name {name!r}; cannot save unambiguously.")
        seen.add(name)
        blob[_FIT_PARAM_PREFIX + name] = np.asarray(sv.get_value())

    seen = set()
    for sv in shared_extras:
        name = sv.name
        if name is None:
            raise ValueError(
                "Every entry of shared_extras must have a .name set; "
                "found a shared variable with name=None."
            )
        if name in seen:
            raise ValueError(f"Duplicate shared_extras name {name!r}; cannot save unambiguously.")
        seen.add(name)
        blob[_FIT_EXTRA_PREFIX + name] = np.asarray(sv.get_value())

    if meta is not None:
        blob[_FIT_META_KEY] = np.asarray(json.dumps(meta))

    np.savez_compressed(path, **blob)


def load_fit(path, shared_params, shared_extras=(), strict=True):
    """Load values saved by :func:`save_fit` into freshly built shared vars.

    Rebuild the PyMC model, GP model, and call ``compile_scipy_objective``
    (or ``compile_training_step`` / ``minimize_staged_vfe``) the same way
    you did for the original fit; that gives you a new ``shared_params``
    and ``shared_extras`` initialised from the prior. Pass them here and
    every shared variable is overwritten with its saved value, ready for
    :func:`compile_predict`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a file written by :func:`save_fit`.
    shared_params : dict
        Same layout as the dict that was saved.
    shared_extras : sequence of pytensor shared variables, optional
        Same as the sequence that was saved (matched by ``.name``, not order).
    strict : bool
        If True (default), raise when the file's name set does not exactly
        match the provided shared variables (missing or extra names) or when
        any shape disagrees. If False, missing names are skipped with a
        warning and extras in the file are ignored.

    Returns
    -------
    dict or None
        The ``meta`` dict that was passed to :func:`save_fit`, or ``None``
        if no metadata was stored.

    Raises
    ------
    ValueError
        Under ``strict=True``, if any saved name is absent from the
        provided shared variables (or vice versa), or if shapes mismatch.

    Examples
    --------
    Reload and predict, no retraining::

        # Rebuild model graph identically (same PyMC code, same gp_model,
        # same compile_scipy_objective call):
        with pm.Model() as model:
            ...  # priors
            gp_model = VFE(...)
        fun, theta0, unpack, sp, se = compile_scipy_objective(
            objective_fn,
            gp_model,
            X_var,
            y_var,
            model=model,
            extra_vars=[Z_var],
            extra_init=[Z_init],
        )

        from ptgp.utils import load_fit

        meta = load_fit("fit_threefactor_2026-04-22.npz", sp, se)
        print(meta["as_of"], meta["features"])

        predict_fn = compile_predict(
            gp_model,
            X_new_var,
            model,
            sp,
            extra_vars=[Z_var],
            shared_extras=se,
        )
        mean, var = predict_fn(X_new)
    """
    blob = np.load(path, allow_pickle=False)
    file_keys = set(blob.files)

    param_keys = {
        k[len(_FIT_PARAM_PREFIX) :]: k for k in file_keys if k.startswith(_FIT_PARAM_PREFIX)
    }
    extra_keys = {
        k[len(_FIT_EXTRA_PREFIX) :]: k for k in file_keys if k.startswith(_FIT_EXTRA_PREFIX)
    }

    shared_params_by_name = {vv.name: sv for vv, sv in shared_params.items()}
    shared_extras_by_name = {sv.name: sv for sv in shared_extras}

    if strict:
        missing_p = set(param_keys) - set(shared_params_by_name)
        extra_p = set(shared_params_by_name) - set(param_keys)
        missing_e = set(extra_keys) - set(shared_extras_by_name)
        extra_e = set(shared_extras_by_name) - set(extra_keys)
        problems = []
        if missing_p:
            problems.append(f"params in file but not in shared_params: {sorted(missing_p)}")
        if extra_p:
            problems.append(f"shared_params not present in file: {sorted(extra_p)}")
        if missing_e:
            problems.append(f"extras in file but not in shared_extras: {sorted(missing_e)}")
        if extra_e:
            problems.append(f"shared_extras not present in file: {sorted(extra_e)}")
        if problems:
            raise ValueError("load_fit name mismatch (strict=True):\n  " + "\n  ".join(problems))

    for name, key in param_keys.items():
        sv = shared_params_by_name.get(name)
        if sv is None:
            logger.warning(f"[load_fit] skipping param {name!r}: not in shared_params")
            continue
        val = blob[key]
        cur = sv.get_value()
        if val.shape != cur.shape:
            raise ValueError(
                f"Shape mismatch for param {name!r}: file {val.shape} vs current {cur.shape}"
            )
        sv.set_value(np.asarray(val))

    for name, key in extra_keys.items():
        sv = shared_extras_by_name.get(name)
        if sv is None:
            logger.warning(f"[load_fit] skipping extra {name!r}: not in shared_extras")
            continue
        val = blob[key]
        cur = sv.get_value()
        if val.shape != cur.shape:
            raise ValueError(
                f"Shape mismatch for extra {name!r}: file {val.shape} vs current {cur.shape}"
            )
        sv.set_value(np.asarray(val))

    if _FIT_META_KEY in file_keys:
        return json.loads(str(blob[_FIT_META_KEY]))
    return None
