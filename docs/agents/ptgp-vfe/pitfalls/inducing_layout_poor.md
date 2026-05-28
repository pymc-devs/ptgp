---
name: inducing_layout_poor
severity: medium
applies_to: [VFE]
symptoms:
  - nystrom_residual stays large at convergence
  - trace_curve has flattened by chosen M (knee passed)
  - d_final has scattered hot spots — particular data points are poorly covered
related_pitfalls: [M_too_small, sigma_inflation, lengthscale_runaway]
---

## Detection

```python
nys_large = history[-1].nystrom_residual > 0.05 * history[0].nystrom_residual
frac_unexpl_at_M = diag.trace_curve[-1] / diag.total_variance
curve_flat = frac_unexpl_at_M < 0.01

# d_final has hot spots: a small fraction of points has much larger residual variance
d = diag.d_final
hot_spots = (d > 10 * np.median(d)).sum() > 0

layout_poor = nys_large and curve_flat and hot_spots
```

Visually: `scripts/check_inducing.py` — middle panel is **below** the
1% threshold (so `M` is fine), but the right panel (`d_final`
scatter) shows points whose residual variance is much higher than
the rest.

## Diagnosis

The trace curve shows that `M` is sufficient *for the kernel used in
the greedy step* — but the kernel that was used (typically a
prior-median proxy) doesn't match the kernel after training. After
training, the lengthscale (or other kernel parameters) shifted, and
the original Z layout is no longer near-optimal.

Distinguishing from [M_too_small](M_too_small.md): the trace curve
*has* flattened. Distinguishing from
[lengthscale_runaway](lengthscale_runaway.md): if the lengthscale is
within prior bounds, the layout — not the kernel — is the issue.

## Fix

The "Tier B → re-init Z and retrain" loop:

1. After the current fit, build a numerical kernel using the
   *trained* hyperparameters (use `pg.optim.get_trained_params`
   plus a substitution into the kernel).
2. Re-run `greedy_variance_init(X, M, trained_kernel)` to get a
   layout matched to the learned scale.
3. Refit. The second pass should converge faster and to a better
   point.

If the new trace curve also hasn't flattened at the new `M`, escalate
to [M_too_small](M_too_small.md).

## See also

- [reference/choosing_M.md](../reference/choosing_M.md) — the
  trace-curve / d_final dichotomy is described under "When to revisit
  M".
- [M_too_small](M_too_small.md) — alternative cause.
- [reference/workflow.md](../reference/workflow.md) — re-init loop in
  Tier B.
