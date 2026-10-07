---
name: large_grad_at_init
severity: medium
applies_to: [all]
symptoms:
  - check_init reports |grad| > 1e4 on at least one parameter
  - L-BFGS-B's first few steps overshoot or oscillate
related_pitfalls: [non_finite_at_init, bad_priors, lbfgsb_abnormal]
---

## Detection

```python
pg.utils.check_init(fun, theta0, X, y, model=model)  # logs a warning if huge
_, grad = fun(theta0, X, y)
huge = np.abs(grad).max() > 1e4
```

The threshold `1e4` matches `_LARGE_GRAD_WARN` in `ptgp/utils.py`, which
`check_init` uses for its warning.

## Diagnosis

A large gradient at the initial point typically means the loss
surface is steep there, *not* that the loss is non-finite. L-BFGS-B
will usually take a tiny step and recover, but at extreme magnitudes:

- The line search may overshoot, jump to a worse region, or
  oscillate.
- The Hessian approximation gets a very rough seed and the first few
  iterations are wasted.

Underlying causes:

1. The initial hyperparameters (and Z, for inducing-point models) are
   far from a sane optimum in one specific direction (e.g. a single
   lengthscale).
2. Prior log-density at `theta0` is large in magnitude — e.g. tight
   prior far from `theta0`. This is added to the loss when
   `include_prior=True`.

## Fix

1. Check **which parameter** has the huge gradient: `check_init` logs
   the `top_k` largest `|grad|` components with parameter labels. If it
   is one specific lengthscale, fix its prior or `initval`.
2. Switch `init="prior_median"` (the default) if you've been using
   `"unconstrained_zero"`.
3. If priors are the cause, set `initval=` on the affected RV at a
   value closer to where the optimiser will end up. `initval` always
   takes priority under `init="prior_median"` and `"prior_draw"`. A
   tighter prior is usually the right fix, though, not `initval` tuning.
4. As a last resort, raise the optimizer budget and let L-BFGS-B work
   through it: `options={"maxiter": ..., "ftol": ...}` for `pg.fit` and
   `tracked_minimize`, or the `phase*_maxiter` arguments of
   `minimize_staged_vfe`. The initial-grad warning is just a warning,
   not a failure.

## See also

- `ptgp/utils.py:_LARGE_GRAD_WARN`: the warning threshold.
- [non_finite_at_init](non_finite_at_init.md) — the harder failure
  mode.
- [bad_priors](bad_priors.md) — common upstream cause.
