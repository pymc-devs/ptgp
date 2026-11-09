---
name: sigma_inflation
severity: high
applies_to: [all]
symptoms:
  - sigma grows during training while the objective plateaus
  - sigma approaches std(y) — the model is regressing toward the mean
  - excess_fit_per_n drifts toward 0 or negative
related_pitfalls: [eta_collapse, sigma_collapse, lengthscale_runaway, M_too_small, bad_priors]
---

## Detection

```python
sigma_traj = np.array([d.sigma for d in history])
obj_traj   = np.array([d[0] for d in history])  # elbo (VFE) or mll (exact GP)
exc_traj   = np.array([d.excess_fit_per_n for d in history])
y_std = float(np.std(y))

inflating = (
    sigma_traj[-1] > 0.5 * y_std
    and (obj_traj[-1] - obj_traj[len(obj_traj) // 2]) < 1e-3
    and exc_traj[-1] < exc_traj[0]
)
```

Visually: the `sigma` panel rises while the objective flattens; the
`excess_fit_per_n` panel falls.

## Diagnosis

The optimiser is giving up on fitting the data and instead treating
the GP as nearly a flat prior with large noise. Common upstream
causes:

1. **(VFE) Z layout is wrong** and the trace penalty is hard to satisfy
   except by inflating sigma (which weakens its `1/sigma^2`
   coefficient). This is the collapsed bound's one cheap escape: a
   larger sigma always shrinks the penalty, while a smaller sigma only
   grows it. Training Z jointly (`Points(Z_var, Z_init=Z0)`) with weak
   hyperparameter priors often lands here.
   [sigma_collapse](sigma_collapse.md) is not the mirror image of this;
   it needs dense Z and is an exact-GP overfitting problem.
2. **Lengthscale runaway** — a lengthscale grew unbounded, making
   the kernel near-constant; the only fit signal left is `sigma`.
   Open [lengthscale_runaway](lengthscale_runaway.md).
3. **Eta collapse** — the kernel amplitude went to 0, so sigma absorbs
   all the variance. Open [eta_collapse](eta_collapse.md).

## Fix

1. Confirm by reading the kernel hyperparameters at convergence via
   `pg.optim.get_trained_params(model, shared_params)`. If a
   lengthscale is at the prior boundary (or absurdly large), fix
   that first.
2. **Tighten priors on noise**, not on lengthscale —
   `pm.HalfNormal("sigma", sigma=...)` with `sigma` matched to the
   empirical residual std of a baseline mean predictor. Avoid
   `HalfFlat`.
3. For VFE, train with `minimize_staged_vfe` (see
   [reference/api.md](../reference/api.md)) so sigma is frozen during
   initial Z movement, then released.
4. If staged VFE training still inflates, M is probably too small or the Z
   layout is wrong: open [M_too_small](M_too_small.md) or
   [inducing_layout_poor](inducing_layout_poor.md).

## See also

- [reference/interpretation.md](../reference/interpretation.md) —
  `sigma`, `excess_fit_per_n` field semantics.
- [sigma_collapse](sigma_collapse.md): the opposite symptom, with a different cause.
- [lengthscale_runaway](lengthscale_runaway.md) — common upstream
  cause.
