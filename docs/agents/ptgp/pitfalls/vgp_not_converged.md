---
name: vgp_not_converged
severity: medium
applies_to: [VGP]
symptoms:
  - max |g_lambda| (or |g_nu|) is not small relative to the data scale
  - alpha differs from score, or lam from info, at the end of the fit
  - fit.result.success is False, or L-BFGS-B hit maxiter
related_pitfalls: [vgp_points_discounted, slow_convergence, lbfgsb_abnormal]
---

## Detection

```python
d = pg.gp.get_vgp_point_diagnostics(vgp, fit, X, y)
scale = np.max(np.abs(d.score)) + 1e-12
not_converged = np.max(np.abs(d.g_nu)) > 1e-3 * scale or np.max(np.abs(d.g_lambda)) > 1e-3
```

Do not judge convergence by `alpha - score` or `lam - info`, or by
`d.converged` alone.

## Diagnosis

At the optimum `alpha = score` and `lam = info`, but the gradients multiply
these differences by `K` and by the Hadamard square `S o S`
(Opper & Archambeau 2009, Eq. 11-12). When either matrix is ill-conditioned,
which is the common case for inputs dense relative to the lengthscale, the
ELBO barely depends on some directions of `alpha - score` and `lam - info`, so
those differences can stay O(1) at a converged fit. `g_nu` and `g_lambda`
measure convergence in the coordinates the optimizer sees.

- `g_nu` small, `g_lambda` not: the mean has converged and the variances are
  still moving. This is the typical slow mode of the `(alpha, lambda)`
  parameterization, which is non-concave in `lambda` (Khan, Mohamed & Murphy
  2012).
- `d.converged` False with small gradients: L-BFGS-B stopped on a line-search
  or relative-reduction criterion. The fit is usually fine.

## Fix

1. **Run longer.** Hundreds to thousands of iterations are normal for VGP:
   raise `options={"maxiter": ...}` in `pg.fit` and re-check `g_nu` and
   `g_lambda`.
2. **Warm start** from the previous fit's `alpha` and `lam` via
   `init_vgp_params(N, alpha_init=..., lambda_init=...)` when refitting.

## See also

- [reference/vgp_interpretation.md](../reference/vgp_interpretation.md):
  field semantics.
- [slow_convergence](slow_convergence.md), [lbfgsb_abnormal](lbfgsb_abnormal.md).
