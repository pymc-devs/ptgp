---
name: sigma_collapse
severity: high
applies_to: [VFE]
symptoms:
  - sigma drifts toward 0 during training
  - ELBO appears to improve while predictive fit gets worse
  - trace_penalty also drops, but only because sigma shrinks the denominator
related_pitfalls: [sigma_inflation, eta_collapse, lbfgsb_abnormal]
---

## Detection

```python
sigma_traj = np.array([d.sigma for d in history])
elbo_traj  = np.array([d.elbo  for d in history])
nys_traj   = np.array([d.nystrom_residual for d in history])

collapsing = (
    sigma_traj[-1] < 0.1 * sigma_traj[0]
    and sigma_traj[-1] < 1e-3
    and elbo_traj[-1] > elbo_traj[0]
    and nys_traj[-1] > nys_traj[0]
)
```

Visually: in the 6-panel `plot_history.py` PNG, the `sigma` panel
plummets while `elbo` rises. The `nystrom_residual` panel does **not**
fall — Q has not actually become a better approximation to Kff.

## Diagnosis

The Titsias collapsed ELBO contains
`trace_penalty = -0.5 / sigma^2 * sum(Kff_diag - Q_diag)`. When Z is
poorly placed and the optimiser can't move it (Tier B with frozen Z)
or moves it slowly, shrinking sigma is a cheap way to inflate `1/sigma^2`'s
denominator and make the penalty appear small. ELBO rises but the
underlying approximation has not improved.

What rules this out: the optimiser is not actually overfitting; it's
exploiting an algebraic loophole. `excess_fit_per_n` typically does
*not* improve commensurately, because the noise floor `0.5 log(2π σ²)`
is also moving.

## Fix

1. **Escalate to Tier D** ([staged VFE](../reference/workflow.md#tier-d--staged-vfe)).
   `minimize_staged_vfe` was built for exactly this: it freezes sigma
   in phase 1 while Z and the kernel hyperparameters move.
2. If you're already in Tier D and seeing collapse during phase 2b
   (sigma free again), the issue is that Z still hasn't moved enough.
   Increase `phase2_cycles` or raise `phase2_maxiter_Z`.
3. If escalating doesn't help, the underlying problem is more
   fundamental — usually `M_too_small` or
   [bad_priors](bad_priors.md) on the kernel. Open those.

Do **not** "fix" sigma collapse by clamping sigma with a tight prior
without also addressing Z layout. That just hides the symptom.

## See also

- [reference/workflow.md](../reference/workflow.md) — Tier C → Tier D
  escalation.
- [reference/interpretation.md](../reference/interpretation.md) —
  `sigma`, `trace_penalty`, `excess_fit_per_n` field semantics.
- `ptgp/optim/training.py:728` — `minimize_staged_vfe` docstring,
  which describes this failure mode in its first paragraph.
