---
name: lbfgsb_abnormal
severity: high
applies_to: [all]
symptoms:
  - scipy.optimize.minimize returns result.status == 2
  - result.message contains "ABNORMAL_TERMINATION_IN_LNSRCH" or similar
  - history may stop short of convergence
related_pitfalls: [non_finite_at_init, large_grad_at_init, kuu_ill_conditioned, slow_convergence]
---

## Detection

```python
result.status == 2
# or
"abnormal" in str(result.message).lower()
```

`status == 2` from L-BFGS-B is "ABNORMAL_TERMINATION_IN_LNSRCH" — the
line search couldn't find a decrease direction.

## Diagnosis

L-BFGS-B's line search couldn't make progress. Common causes,
roughly in priority order:

1. **Gradient or loss became non-finite** along the search direction.
   Often caused by an extreme parameter combination making
   `K + sigma^2 I` numerically non-PSD, or, for inducing-point methods,
   by Kuu becoming singular as Z moves. See
   [kuu_ill_conditioned](kuu_ill_conditioned.md).
2. **Gradient was already large or ill-scaled at the starting
   theta** of this iteration; line search fails to make headway.
   See [large_grad_at_init](large_grad_at_init.md).
3. **The Hessian approximation is broken**, typically because the
   objective has a saddle or near-flat region the optimiser keeps
   bouncing through.
4. **Numerical noise in the gradient**: for very tight tolerances,
   PyTensor's gradient may be too noisy for L-BFGS-B's defaults.

## Fix

1. **Re-evaluate at the abort point.** Build `diag_fn =
   compile_scipy_diagnostics(<diagnostics>, ...)` with the same arguments
   as the objective (`vfe_diagnostics` for VFE,
   `unapproximated_diagnostics` for the exact GP), then call
   `diag_fn(result.x, X, y)` and `fun(result.x, X, y)` to look for
   non-finite values or huge gradients. Localises the cause.
2. **Loosen scipy tolerances**:

   ```python
   options={"ftol": 1e-7, "gtol": 1e-5}  # pg.fit / tracked_minimize kwargs
   ```

   Tighter than scipy's defaults for L-BFGS-B is rarely useful in
   practice for these objectives.
3. **Optimise fewer parameters at once.** A failed line search is often
   a sign that joint optimisation is trying to do too much. Freeze Z
   (`Points(Z0)` with a concrete array) if it is trainable. For VFE,
   `minimize_staged_vfe` (see [api.md](../reference/api.md)) trains in
   phases, each with fewer parameters to coordinate.
4. **If status==2 at the very first step**: check init —
   [non_finite_at_init](non_finite_at_init.md) or
   [large_grad_at_init](large_grad_at_init.md) is the upstream
   cause.

## See also

- [slow_convergence](slow_convergence.md) — the milder
  non-convergence mode (status==1, "MAXITER").
