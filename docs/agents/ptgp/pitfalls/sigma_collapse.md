---
name: sigma_collapse
severity: medium
applies_to: [all]
symptoms:
  - sigma drifts toward 0 during training
  - train fit keeps improving while held-out predictions get worse
  - nystrom_residual is already near 0 (Z covers the data densely)
related_pitfalls: [sigma_inflation, lengthscale_runaway, bad_priors]
---

## Detection

```python
sigma_traj = np.array([d.sigma for d in history])
nys_traj   = np.array([d.nystrom_residual for d in history])
exc_traj   = np.array([d.excess_fit_per_n for d in history])
resid_std  = float(np.std(y - y.mean()))  # or the residual std of a baseline mean model

collapsing = (
    sigma_traj[-1] < 0.1 * sigma_traj[0]
    and sigma_traj[-1] < 0.05 * resid_std
    and nys_traj[-1] < 0.01 * sigma_traj[-1] ** 2  # trace penalty no longer resists
    and exc_traj[-1] > exc_traj[0]
)
```

Confirm with held-out data: a collapsing fit interpolates the training
points, so held-out error rises while `excess_fit_per_n` keeps rising.

## Diagnosis

Sigma collapse is a general GP failure, not a VFE one. With weak priors,
the lengthscale shrinks until the kernel interpolates the noise, and sigma
is no longer needed. An exact GP can do this on its own.

Sparse VFE normally resists it better than the exact GP. With sparse `Z`,
`Q` is low rank and cannot interpolate the noise, and the trace penalty
`-0.5 / sigma^2 * sum(Kff_diag - Q_diag)` grows as sigma shrinks. VFE loses
that protection when `Q ≈ Kff` at the resolution of the noise, which needs
dense inducing points (`M` close to `N`, or trained `Z` sitting on the
data) together with a short lengthscale. VFE then behaves like the exact
GP, overfitting included. `Q ≈ Kff` from a long lengthscale alone is
harmless, because a kernel that smooth cannot interpolate noise.

The trace-penalty loophole runs the other way: inflating sigma quiets the
penalty. See [sigma_inflation](sigma_inflation.md).

A case that looks like collapse but is not a failure: **noise-free or
nearly noise-free data** (deterministic simulators, interpolated tables,
repeated rows whose `y` agree). A small sigma is the right answer.

## Fix

1. **Fit `pg.gp.Unapproximated` on a subsample** (N <= ~1000) to check
   the model itself, separate from the approximation. If the exact GP
   also collapses, the problem is the model, not VFE.
2. **Keep the lengthscale away from 0** with a prior whose density
   vanishes there, e.g. `pm.InverseGamma`, set from the input spacing. See
   [bad_priors](bad_priors.md) and
   [lengthscale_runaway](lengthscale_runaway.md).
3. **Keep sigma away from 0** if the data is known to be noisy, with a
   prior that vanishes at 0, e.g. `pm.InverseGamma` or `pm.LogNormal`
   centred on the expected noise level, instead of `HalfNormal` or
   `HalfFlat`.
4. **Use sparser inducing points** if `M` is close to `N` or trained `Z`
   has moved onto the data: re-initialize `Z` with `greedy_variance_init`
   at a smaller `M` and freeze it. Freezing an already dense `Z` does not
   help. Dense inducing points give up the regularisation VFE would
   otherwise provide.

Staged VFE does not address this. `minimize_staged_vfe` freezes sigma
early to stop *inflation*, and releases it in later phases, where a
collapse driven by the kernel and priors would still happen.

## See also

- [reference/interpretation.md](../reference/interpretation.md):
  `sigma`, `nystrom_residual`, `excess_fit_per_n` field semantics.
- [sigma_inflation](sigma_inflation.md): the trace-penalty loophole.
