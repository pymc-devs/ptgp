import numpy as np
import pymc as pm
import xarray as xr

from arviz_base import dict_to_dataset, from_dict
from pymc.backends.arviz import (
    coords_and_dims_for_inferencedata,
    find_constants,
    find_observations,
)


def to_idata(
    shared_params,
    shared_extras=(),
    *,
    result=None,
    history=None,
    phase_labels=None,
    model=None,
):
    """Package a ptgp training run as an ``xarray.DataTree``.

    Groups produced (empty groups are omitted):

    - ``point_estimate`` — every optimized value in constrained space, on its
      own dims with no ``chain``/``draw`` axes (nothing was sampled). Includes
      PyMC RVs (``sigma``, ``eta``, …) *and* non-PyMC extras (``Z``, ``q_mu``,
      ``q_sqrt_flat``, …).
    - ``unconstrained_point_estimate`` — the same values in value-var
      (transformed) space.
    - ``optimizer_result`` — scalar fields from the scipy ``OptimizeResult``
      (``fun``, ``success``, ``status``, ``message``, ``nit``, ``nfev``,
      ``njev``) plus per-iteration trajectory DataArrays from ``history``
      (``elbo``, ``trace_penalty``, …) on an ``iteration`` dimension. There is
      no ``sample_stats``: nothing was sampled.
    - ``observed_data`` / ``constant_data`` — pulled from the PyMC model.

    Parameters
    ----------
    shared_params : dict
        ``{value_var: shared_var}`` returned by ``compile_scipy_objective`` /
        ``compile_training_step`` / ``minimize_staged_vfe``.
    shared_extras : sequence of pytensor shared variables, optional
        Trained extras (``Z``, ``q_mu``, ``q_sqrt_flat``, …). Each entry must
        have a ``.name``; values appear in ``point_estimate`` keyed by that name.
        Default ``()``.
    result : scipy.optimize.OptimizeResult, optional
        Final optimization result. Default ``None``.
    history : list of namedtuples, optional
        Per-iteration diagnostics from :func:`ptgp.optim.tracked_minimize`. Each
        namedtuple field becomes its own ``(iteration,)`` DataArray under
        ``optimizer_result``. Default ``None``.
    phase_labels : list of str, optional
        Per-iteration phase labels from :func:`ptgp.optim.minimize_staged_vfe`.
        Attached as a ``phase`` coord on the ``iteration`` dim. Default ``None``.
    model : pm.Model, optional
        Defaults to ``pm.modelcontext(model)``.

    Returns
    -------
    xr.DataTree
    """
    model = pm.modelcontext(model)
    coords, dims = coords_and_dims_for_inferencedata(model)

    constrained, unconstrained = _collect_optimized(model, shared_params, shared_extras)
    idata = from_dict(
        {
            "point_estimate": constrained,
            "unconstrained_point_estimate": unconstrained,
        },
        sample_dims=[],
        coords=coords,
        dims=dims,
        name="ptgp_fit",
    )

    optimizer_ds = _build_optimizer_result(result, history, phase_labels)
    if optimizer_ds.data_vars:
        idata["optimizer_result"] = xr.DataTree(dataset=optimizer_ds)

    observations = find_observations(model)
    if observations:
        idata["observed_data"] = xr.DataTree(
            dataset=dict_to_dataset(
                observations, inference_library=pm, coords=coords, dims=dims, sample_dims=[]
            )
        )
    constants = find_constants(model)
    if constants:
        idata["constant_data"] = xr.DataTree(
            dataset=dict_to_dataset(
                constants, inference_library=pm, coords=coords, dims=dims, sample_dims=[]
            )
        )

    return idata


def _collect_optimized(model, shared_params, shared_extras):
    """Return ``(constrained, unconstrained)`` dicts of every optimized value.

    PyMC RVs contribute to both dicts: the constrained dict uses ``rv.name`` and
    applies the backward transform; the unconstrained dict uses ``vv.name`` and
    keeps the value-var-space value. Non-PyMC extras have no transform — they
    appear in both dicts under ``sv.name`` so callers can find them in either
    group.
    """
    constrained, unconstrained = {}, {}
    for rv in model.free_RVs:
        vv = model.rvs_to_values[rv]
        if vv not in shared_params:
            continue
        unc = np.asarray(shared_params[vv].get_value())
        unconstrained[vv.name] = unc
        transform = model.rvs_to_transforms.get(rv)
        constrained[rv.name] = np.asarray(transform.backward(unc).eval()) if transform else unc

    seen = set()
    for sv in shared_extras:
        if sv.name is None:
            raise ValueError(
                "every shared_extras entry must have a .name set; "
                "construct extras with pt.matrix('Z', ...) / pt.vector('q_mu', ...)."
            )
        if sv.name in seen:
            raise ValueError(f"duplicate shared_extras name {sv.name!r}.")
        seen.add(sv.name)
        val = np.asarray(sv.get_value())
        constrained[sv.name] = val
        unconstrained[sv.name] = val
    return constrained, unconstrained


_RESULT_SCALAR_FIELDS = {
    "fun": float,
    "success": bool,
    "status": int,
    "message": str,
    "nit": int,
    "nfev": int,
    "njev": int,
}


def _build_optimizer_result(result, history, phase_labels):
    """Combine ``OptimizeResult`` scalars and ``history`` trajectories into one Dataset.

    Result fields use the raw scipy names (``fun``, ``nit``, …) so users
    familiar with ``scipy.optimize.OptimizeResult`` recognize them. Per-
    iteration namedtuple fields become 1-D DataArrays on the ``iteration`` dim,
    each named after its namedtuple field (so a VFEDiagnostics history
    populates ``elbo``, ``trace_penalty``, …).

    If a history field name collides with a result-scalar name, the trajectory
    wins — it's strictly more informative than the terminal value.
    """
    data_vars = {}
    coords = {}

    if result is not None:
        for attr, caster in _RESULT_SCALAR_FIELDS.items():
            if not hasattr(result, attr):
                continue
            val = getattr(result, attr)
            if val is None:
                continue
            if attr == "fun" and not np.isfinite(val):
                val = float("nan")
            else:
                val = caster(val)
            data_vars[attr] = xr.DataArray(val, dims=[])

    if history:
        n_iter = len(history)
        for f in type(history[0])._fields:
            col = np.asarray([getattr(h, f) for h in history], dtype=np.float64)
            data_vars[f] = xr.DataArray(col, dims=["iteration"])
        coords["iteration"] = np.arange(n_iter)
        if phase_labels is not None:
            if len(phase_labels) != n_iter:
                raise ValueError(
                    f"phase_labels has length {len(phase_labels)} but history has "
                    f"{n_iter} entries."
                )
            coords["phase"] = ("iteration", np.asarray(phase_labels))

    return xr.Dataset(data_vars, coords=coords)
