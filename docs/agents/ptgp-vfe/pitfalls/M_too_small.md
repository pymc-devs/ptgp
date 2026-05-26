---
name: M_too_small
severity: medium
applies_to: [VFE]
symptoms:
  - nystrom_residual stays large at convergence
  - GreedyVarianceDiagnostics.trace_curve has not flattened by chosen M
  - d_final is uniformly elevated (not just hot spots)
related_pitfalls: [inducing_layout_poor, sigma_inflation, excess_fit_per_n_negative]
---

## Detection

Two-source: needs both `history[-1]` and a `GreedyVarianceDiagnostics`
run at `M_max > current_M`:

```python
nys_large = history[-1].nystrom_residual > 0.05 * history[0].nystrom_residual
frac_unexpl_at_M = diag_max.trace_curve[current_M - 1] / diag_max.total_variance
curve_not_flat   = frac_unexpl_at_M > 0.05  # > 5% — knee not reached

m_too_small = nys_large and curve_not_flat
```

Visually: `scripts/check_inducing.py` — middle panel
(fraction-unexplained vs M) is still well above the 1% threshold line
at your chosen M.

## Diagnosis

`M` is below the rank needed to approximate `Kff` to ELBO-tightness.
No Z layout, however clever, can compensate. Distinguishing this
from [inducing_layout_poor](inducing_layout_poor.md) is the whole
point of running greedy at `M_max`:

| Signal | M_too_small | inducing_layout_poor |
|---|---|---|
| `history[-1].nystrom_residual` | large | large |
| `trace_curve / total_variance` at chosen M | still > ~5% | < ~1% |
| `d_final` distribution | uniformly elevated | a few hot spots |
| Fix | increase M | re-run greedy with the trained kernel |

## Fix

1. Look at the trace curve to pick a new `M` from the knee — see
   [reference/choosing_M.md](../reference/choosing_M.md).
2. Re-run greedy at the new `M`, refit the model.
3. If the trace curve never flattens (even at `M_max = N`), the
   kernel is wrong for the data — short lengthscale relative to data
   spread, missing structure. Tier E modelling change.

## See also

- [reference/choosing_M.md](../reference/choosing_M.md) — the
  selection rule.
- [inducing_layout_poor](inducing_layout_poor.md) — the alternative
  cause of large `nystrom_residual`.
- [reference/interpretation.md](../reference/interpretation.md) —
  `nystrom_residual`, `trace_curve` semantics.
