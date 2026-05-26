---
name: sigma_inflation
severity: high
applies_to: [VFE]
symptoms:
  - sigma grows during training while ELBO plateaus
  - sigma approaches std(y) — the model is regressing toward the mean
  - excess_fit_per_n drifts toward 0 or negative
related_pitfalls: [sigma_collapse, lengthscale_runaway, M_too_small, bad_priors]
---

## Detection

```python
sigma_traj = np.array([d.sigma for d in history])
elbo_traj  = np.array([d.elbo  for d in history])
exc_traj   = np.array([d.excess_fit_per_n for d in history])
y_std = float(np.std(y))

inflating = (
    sigma_traj[-1] > 0.5 * y_std
    and (elbo_traj[-1] - elbo_traj[len(elbo_traj) // 2]) < 1e-3
    and exc_traj[-1] < exc_traj[0]
)
```

Visually: the `sigma` panel rises while `elbo` flattens; the
`excess_fit_per_n` panel falls.

## Diagnosis

The optimiser is giving up on fitting the data and instead treating
the GP as nearly a flat prior with large noise. Two common upstream
causes:

1. **Z layout is wrong** and the trace penalty is hard to satisfy
   except by inflating sigma (which weakens its `1/sigma^2`
   coefficient). Symmetric to `sigma_collapse`: same algebraic
   loophole, opposite direction. Why this direction rather than the
   other depends on how the rest of the gradient pulls sigma — often
   Tier C with weak hyperparameter priors lands here.
2. **Lengthscale runaway** — a lengthscale grew unbounded, making
   the kernel near-constant; the only fit signal left is `sigma`.
   Open [lengthscale_runaway](lengthscale_runaway.md).

## Fix

1. Confirm by reading the kernel hyperparameters at convergence via
   `pg.optim.get_trained_params(model, shared_params)`. If a
   lengthscale is at the prior boundary (or absurdly large), fix
   that first.
2. **Tighten priors on noise**, not on lengthscale —
   `pm.HalfNormal("sigma", sigma=...)` with `sigma` matched to the
   empirical residual std of a baseline mean predictor. Avoid
   `HalfFlat`.
3. Escalate to Tier D ([staged VFE](../reference/workflow.md#tier-d--staged-vfe))
   so sigma is frozen during initial Z movement, then released.
4. If escalation still inflates, M is probably too small or Z layout
   is wrong — open [M_too_small](M_too_small.md) or
   [inducing_layout_poor](inducing_layout_poor.md).

## See also

- [reference/interpretation.md](../reference/interpretation.md) —
  `sigma`, `excess_fit_per_n` field semantics.
- [sigma_collapse](sigma_collapse.md) — the mirror failure.
- [lengthscale_runaway](lengthscale_runaway.md) — common upstream
  cause.
