---
name: large_grad_at_init
severity: medium
applies_to: [VFE]
symptoms:
  - check_init reports |grad| > 1e4 on at least one parameter
  - L-BFGS-B's first few steps overshoot or oscillate
related_pitfalls: [non_finite_at_init, bad_priors, lbfgsb_abnormal]
---

## Detection

```python
diag = pg.utils.check_init(fun, theta0, model=model, ...)
huge = np.abs(diag["grad"]).max() > 1e4
```

The threshold `1e4` matches `_LARGE_GRAD_WARN` in `ptgp/utils.py:51`.

## Diagnosis

A large gradient at the initial point typically means the loss
surface is steep there, *not* that the loss is non-finite. L-BFGS-B
will usually take a tiny step and recover, but at extreme magnitudes:

- The line search may overshoot, jump to a worse region, or
  oscillate.
- The Hessian approximation gets a very rough seed and the first few
  iterations are wasted.

Underlying causes:

1. The initial kernel and Z combination is far from a sane optimum
   in one specific direction (e.g. a single lengthscale).
2. Prior log-density at `theta0` is large in magnitude — e.g. tight
   prior far from `theta0`. This is added to the loss when
   `include_prior=True`.

## Fix

1. Check **which parameter** has the huge gradient — `check_init`
   reports per-parameter values. If one specific lengthscale, fix
   its prior or `initval`.
2. Switch `init="prior_median"` (the default) if you've been using
   `"unconstrained_zero"`.
3. If priors are the cause, set `initval=` on the affected RV at a
   value closer to where the optimiser will end up. PyMC respects
   `initval` even under `init="prior_median"` *when prior_median
   falls back to PyMC's initial_point* — but a tighter prior is
   usually the right fix, not `initval` tuning.
4. As a last resort, pass `scipy_options={"maxiter": ..., "ftol":
   ...}` and let L-BFGS-B work through it — the initial-grad warning
   is just a warning, not a failure.

## See also

- `ptgp/utils.py:51` — `_LARGE_GRAD_WARN` constant.
- [non_finite_at_init](non_finite_at_init.md) — the harder failure
  mode.
- [bad_priors](bad_priors.md) — common upstream cause.
