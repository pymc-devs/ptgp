---
name: eta_collapse
severity: high
applies_to: [VFE]
symptoms:
  - kernel amplitude eta drops toward 0 during training
  - the kernel becomes near-zero everywhere; predictions revert to mean ± sigma
  - often paired with sigma_collapse
related_pitfalls: [sigma_collapse, sigma_inflation, bad_priors, excess_fit_per_n_negative]
---

## Detection

```python
params = pg.optim.get_trained_params(model, shared_params)
eta = float(params["eta"])
collapsed = eta < 1e-4
```

Indirect: predictions tend toward `y_mean` regardless of `X`; the GP
has no signal.

## Diagnosis

Kernels are scaled `eta**2 * k_base(...)`. If `eta` drops toward 0,
the kernel becomes near-zero, and the model is essentially
`y = mean + Gaussian(0, sigma^2)` — a constant predictor. ELBO can
appear fine (the mean predictor explains as much variance as it
can, sigma absorbs the rest), but the GP isn't fitting any
structure.

Often accompanies [sigma_collapse](sigma_collapse.md) because both
push the model toward a degenerate fit; rarely accompanies
[sigma_inflation](sigma_inflation.md), which is the same underlying
"give up" but in the opposite direction.

## Fix

1. **Tighten the `eta` prior** away from 0:

   ```python
   eta = pm.HalfNormal("eta", sigma=np.std(y))   # bulk away from 0
   # or
   eta = pm.Exponential("eta", lam=1 / np.std(y))
   ```

   For models with structured kernels (`eta1`, `eta2`, ...), each
   needs its own scale-matched prior.
2. **Audit the `eta` × `ls` × `sigma` interaction** — if the
   lengthscale ran away (see
   [lengthscale_runaway](lengthscale_runaway.md)), `eta` collapse
   often follows. Fix lengthscale priors first.
3. Escalate to Tier D ([staged VFE](../reference/workflow.md#tier-d--staged-vfe))
   so the kernel hyperparameters fit early under frozen sigma.
4. Confirm at convergence by examining sample functions from the
   *trained* kernel: they should have non-trivial range at the data
   scale.

## See also

- [sigma_collapse](sigma_collapse.md) — the typical co-failure.
- [bad_priors](bad_priors.md) — upstream cause when the eta prior is
  too tight at 0.
- [reference/api.md](../reference/api.md) — `eta` is the convention
  for kernel amplitude in ptgp; kernels are scaled `eta**2 * ...`.
