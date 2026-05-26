---
name: bad_priors
severity: high
applies_to: [VFE]
symptoms:
  - Kuu ill-conditioned at prior-median hyperparameters
  - non-finite or huge gradient at init
  - lengthscale or eta runs away during training
related_pitfalls: [kuu_ill_conditioned, lengthscale_runaway, sigma_inflation, non_finite_at_init, large_grad_at_init]
---

## Detection

Pre-training: build a numerical kernel from the prior median and
inspect Kuu at the chosen Z layout.

```python
from ptgp.utils import get_initial_params
from ptgp.inducing import greedy_variance_init

params = get_initial_params(model, init="prior_median")
# Substitute params into the kernel to get a numerical kernel.
# The exact substitution depends on how the kernel is constructed —
# typically rebuild the kernel with concrete floats:
num_kernel = build_numerical_kernel(params)

_, diag = greedy_variance_init(X, M, num_kernel)
print(repr(diag))

bad = (
    diag.kuu_condition_number > 1e8
    or diag.kuu_n_small_eigenvalues > 0
)
```

Also: prior-predictive sample functions on a synthetic grid. If
samples are constant or wildly oscillatory, priors are wrong.

## Diagnosis

"Weak priors" is ambiguous (weakly informative is often fine);
**bad** priors are mis-specified relative to the data scale or
inducing layout. Common species:

- **Lengthscale prior centred at the wrong scale** — too short ⇒
  Kff near-diagonal, Q nowhere near it; too long ⇒ Kff near-rank-1,
  Kuu near-singular.
- **Improper priors** (`HalfFlat`, `Flat`) on parameters that need
  any constraint at all — the optimiser is free to wander to
  degenerate values.
- **Noise prior allowing sigma → 0** with no resistance — feeds
  [sigma_collapse](sigma_collapse.md).
- **Amplitude prior (`eta`) too tight at 0** — feeds
  [eta_collapse](eta_collapse.md).

The notebook lesson: prior quality drives Kuu conditioning. A
sensible-looking prior on each parameter individually can still
combine to put the prior median in a nasty corner of the kernel
space.

## Fix

1. **Audit each prior**. For lengthscale, prefer
   `InverseGamma` matched to the data's characteristic scale (zero
   density at both 0 and infinity). For amplitude `eta`, use
   `Exponential` or `HalfNormal`. For noise, use `HalfNormal`
   matched to the empirical residual std of a baseline mean
   predictor.
2. **Avoid `HalfFlat` / `Flat`** in production. They're useful only
   when you can't write down anything sensible at all, and they
   defeat `init="prior_median"`'s ability to give a useful starting
   point (the per-RV fallback uses PyMC's `initial_point`, which is
   1 constrained for `HalfFlat` — often wrong).
3. **Run the pre-training Kuu check** above. If it fails at the
   prior median, tighten the priors *before* training, not after.
4. Use **prior-predictive simulation** to double-check: draw from
   each prior, build the kernel, sample sample functions on a grid.
   They should look like the data could have plausibly come from
   them.

## See also

- [kuu_ill_conditioned](kuu_ill_conditioned.md) — the most common
  downstream symptom.
- [lengthscale_runaway](lengthscale_runaway.md) — runaway is bad
  priors leaking through training.
- `ptgp/utils.py:get_initial_params` — for building the numerical
  proxy kernel.
