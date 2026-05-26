---
name: lengthscale_runaway
severity: high
applies_to: [VFE]
symptoms:
  - a lengthscale shrinks toward 0 or grows toward infinity during training
  - nystrom_residual rises during training
  - the kernel becomes near-constant or near-degenerate
related_pitfalls: [bad_priors, sigma_inflation, M_too_small]
---

## Detection

After training (or watching the trained_params trajectory if you have
a per-iteration log of constrained-space params):

```python
params = pg.optim.get_trained_params(model, shared_params)
ls = float(params["ls"])
data_extent = float(np.max(X) - np.min(X))

shrunk = ls < 0.001 * data_extent
exploded = ls > 100 * data_extent
runaway = shrunk or exploded
```

Indirect indicator from `VFEDiagnostics`:

```python
nys_traj = np.array([d.nystrom_residual for d in history])
rising = nys_traj[-1] > 2 * nys_traj[len(nys_traj) // 2]
```

(Lengthscale shrinking ⇒ Q is a worse approximation to Kff ⇒
`nystrom_residual` rises.)

## Diagnosis

Two distinct directions, same name:

1. **Shrinking**: lengthscale → 0 makes `K(X, X) → I` (or
   near-diagonal with the chosen amplitude). The model overfits
   per-point noise; `nystrom_residual` rises because Z can no longer
   capture per-point covariance.
2. **Growing**: lengthscale → ∞ makes `K(X, X) → 1*1.T` (rank-1
   constant). The model effectively becomes a constant + noise; this
   is upstream of [sigma_inflation](sigma_inflation.md).

Both are usually driven by [bad_priors](bad_priors.md) — flat or
weakly-informative priors that don't push back against the
optimiser's preference for one of the degenerate kernels.

## Fix

1. **Tighten the lengthscale prior** to a scale matched to the data:

   ```python
   # for data with characteristic spread `s`:
   ls = pm.InverseGamma("ls", alpha=3, beta=s)  # mode ≈ s/(α+1)
   ```

   `InverseGamma` is the standard choice — it has zero density at
   both 0 and infinity, structurally preventing both runaway
   directions.
2. Use prior-predictive simulation (draw from the prior, evaluate
   the kernel on a synthetic grid) to confirm the prior gives sane
   sample functions.
3. If runaway happens *during* training despite a sensible prior,
   escalate to Tier D ([staged VFE](../reference/workflow.md#tier-d--staged-vfe))
   so the kernel is fit early under frozen sigma — usually settles
   the lengthscale before sigma can drift.

## See also

- [bad_priors](bad_priors.md) — usually the upstream cause.
- [sigma_inflation](sigma_inflation.md) — common downstream symptom
  when lengthscale grows.
- [reference/interpretation.md](../reference/interpretation.md) —
  `nystrom_residual` rising is the visible signal.
