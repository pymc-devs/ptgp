---
name: excess_fit_per_n_negative
severity: high
applies_to: [Unapproximated, VFE]
symptoms:
  - history[-1].excess_fit_per_n <= 0
  - the model fits no better than a constant mean with Gaussian noise at the residual variance
related_pitfalls: [sigma_inflation, M_too_small, inducing_layout_poor, lengthscale_runaway, bad_priors]
---

## Detection

```python
history[-1].excess_fit_per_n <= 0
```

`excess_fit_per_n = fit_per_n + 0.5 * log(2π * Var(y - m(X))) + 0.5`
(see `vfe_diagnostics` in `ptgp/objectives.py`;
`unapproximated_diagnostics` reports the same field for the exact GP,
computed from `mll`). Strictly: positive
means the model beats a constant-mean Gaussian at the residual variance;
≤ 0 means it does not. Because the reference is the residual variance
rather than `sigma**2`, it is invariant to the scale of `y`.

## Diagnosis

This is the *strongest* signal that the model has produced a degenerate fit,
because it doesn't depend on absolute ELBO scale or arbitrary
thresholds — `excess_fit_per_n` is comparable across runs and
across datasets.

Cause is always one of:

1. **`sigma` inflated** to swallow the variance. See
   [sigma_inflation](sigma_inflation.md).
2. **Lengthscale runaway** — kernel went near-constant. See
   [lengthscale_runaway](lengthscale_runaway.md).
3. **For VFE, M too small or Z layout wrong**: Q is so far from Kff
   that the model is effectively constant + noise. See
   [M_too_small](M_too_small.md),
   [inducing_layout_poor](inducing_layout_poor.md).
4. **Bad priors** that made (1) or (2) attractive to the optimiser.
   See [bad_priors](bad_priors.md).

## Fix

There is no direct fix — work backward to find the actual
upstream cause:

1. Read `history[-1].sigma`. Is it > 0.5 * std(y)? Open
   [sigma_inflation](sigma_inflation.md).
2. Read trained kernel params via `get_trained_params`. Is a
   lengthscale at the prior boundary? Open
   [lengthscale_runaway](lengthscale_runaway.md).
3. For VFE, read `history[-1].nystrom_residual` (already per point).
   Is it large compared with `history[-1].sigma**2`? Open
   [M_too_small](M_too_small.md) or
   [inducing_layout_poor](inducing_layout_poor.md).
4. None of the above? Audit priors. See
   [bad_priors](bad_priors.md).

## See also

- [reference/interpretation.md](../reference/interpretation.md) —
  `fit_per_n`, `excess_fit_per_n` definitions.
- `ptgp/objectives.py:vfe_diagnostics`: exact formula.
