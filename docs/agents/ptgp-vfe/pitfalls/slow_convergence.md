---
name: slow_convergence
severity: low
applies_to: [VFE]
symptoms:
  - L-BFGS-B exhausts maxiter
  - history is monotone but ELBO is still rising
  - result.success == False with status code "MAXITER" / 1
related_pitfalls: [lbfgsb_abnormal, kuu_ill_conditioned]
---

## Detection

```python
result.status == 1  # scipy L-BFGS-B "MAXITER"
or (not result.success and "iteration" in str(result.message).lower())
```

Plus: history is monotone-decreasing in loss with no obvious
plateau.

## Diagnosis

Either the optimiser is making slow but real progress (the bound is
just hard to optimise on this problem — common with large `M` or
high-dimensional input) or the gradient is ill-scaled and L-BFGS-B's
line search is taking many tiny steps. Distinguish by:

- **Per-step ELBO improvement**: divide
  `(history[-1].elbo - history[-50].elbo)` by the step count. If it's
  > 1e-3 in absolute terms, real progress; bump maxiter. If it's
  < 1e-6, the gradient is effectively dead.
- **Gradient norm at the last theta**: large + monotone progress is
  fine; large + stalled is a conditioning problem.

## Fix

1. **Increase scipy maxiter** via `scipy_options={"maxiter": 1000}`
   — first cheap thing to try.
2. **Tighten priors** to remove flat directions in unconstrained
   space. `HalfFlat` priors give scipy nothing to lean on.
3. **Lower M** if Cholesky cost per step is the bottleneck (you'd
   feel this as wall-clock per iter, not iter count).
4. If the gradient is dead but loss isn't moving, you're at a
   saddle / boundary. Try [staged VFE](../reference/workflow.md#tier-d--staged-vfe)
   or `init_rng` to perturb the starting point.

## See also

- [reference/workflow.md](../reference/workflow.md) — Tier C / Tier D
  decision.
- [lbfgsb_abnormal](lbfgsb_abnormal.md) — the harder failure mode.
