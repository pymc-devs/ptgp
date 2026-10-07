---
name: eta_collapse
severity: high
applies_to: [all]
symptoms:
  - kernel amplitude eta drops toward 0 during training
  - the kernel becomes near-zero everywhere; predictions revert to mean ± sigma
  - often paired with sigma_inflation
related_pitfalls: [sigma_inflation, sigma_collapse, bad_priors, excess_fit_per_n_negative]
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
`y = mean + Gaussian(0, sigma^2)`, a constant predictor. The
objective can look fine (the mean explains as much variance as it can,
sigma absorbs the rest), but the GP isn't fitting any structure.

Usually accompanies [sigma_inflation](sigma_inflation.md): once the
kernel carries no signal, sigma has to absorb all of the residual
variance, so it grows toward the residual std. In `VFEDiagnostics` and
`UnapproximatedDiagnostics` this shows as `frac_signal` near 0 and
`frac_noise` near 1.

## Fix

1. **Use an `eta` prior whose density vanishes at 0**, scaled to the
   data. `HalfNormal` and `Exponential` have their mode at 0, so they
   do not resist collapse:

   ```python
   eta = pm.Gamma("eta", alpha=2.0, beta=2.0 / np.std(y))  # mode ~ std(y) / 2
   # or
   eta = pm.LogNormal("eta", mu=np.log(np.std(y)), sigma=1.0)
   ```

   For models with structured kernels (`eta1`, `eta2`, ...), each
   needs its own scale-matched prior.
2. **Audit the `eta` × `ls` × `sigma` interaction** — if the
   lengthscale ran away (see
   [lengthscale_runaway](lengthscale_runaway.md)), `eta` collapse
   often follows. Fix lengthscale priors first.
3. For VFE, train with `minimize_staged_vfe` (see
   [reference/api.md](../reference/api.md)) so the kernel
   hyperparameters fit early while sigma is frozen.
4. Confirm at convergence by examining sample functions from the
   *trained* kernel: they should have non-trivial range at the data
   scale.

## See also

- [sigma_inflation](sigma_inflation.md): the typical co-failure.
- [bad_priors](bad_priors.md) — upstream cause when the eta prior is
  too tight at 0.
- [reference/api.md](../reference/api.md) — `eta` is the convention
  for kernel amplitude in ptgp; kernels are scaled `eta**2 * ...`.
