---
name: non_finite_at_init
severity: high
applies_to: [VFE]
symptoms:
  - check_init reports NaN or inf in loss or grad at theta0
  - scipy aborts immediately with a non-finite-value error
  - first history entry has elbo == -inf or NaN
related_pitfalls: [bad_priors, kuu_ill_conditioned, large_grad_at_init]
---

## Detection

```python
diag = pg.utils.check_init(fun, theta0, model=model, ...)
non_finite = (
    not np.isfinite(diag["loss"])
    or not np.all(np.isfinite(diag["grad"]))
)
```

Or directly in history:

```python
not np.isfinite(history[0].elbo)
```

## Diagnosis

The training graph evaluated to non-finite at `theta0`. Common
mechanisms:

1. **`init="prior_median"` produced a problematic kernel** — e.g. a
   prior median lengthscale much larger than data spread, making
   `Kff + sigma^2 I` numerically singular. See
   [bad_priors](bad_priors.md).
2. **Kuu was ill-conditioned at chosen Z**: the initial Cholesky
   underflows or produces non-finite values. See
   [kuu_ill_conditioned](kuu_ill_conditioned.md).
3. **Bug in the kernel**: `eta = 0` (impossible if eta is positive,
   but possible at unconstrained `eta = -inf` due to extreme priors).
4. **Improper prior with no `initval`**: if `init="unconstrained_zero"`
   and a `HalfFlat` parameter is at 0 unconstrained ↔ 1 constrained,
   that's usually fine — but combined with other unsensible defaults
   it can land in a bad spot.

## Fix

1. Run `pg.utils.check_init` with verbose output and read which
   parameter has a non-finite gradient. That localises the cause.
2. Compute Kuu eigenvalues at the initial Z and median-prior kernel:

   ```python
   from ptgp.utils import get_initial_params
   params = get_initial_params(model, init="prior_median")
   # build numerical kernel from params, then:
   _, diag_init = greedy_variance_init(X, M, num_kernel)
   print(repr(diag_init))
   ```

3. If Kuu is the issue, see
   [kuu_ill_conditioned](kuu_ill_conditioned.md). If priors are the
   issue, see [bad_priors](bad_priors.md).
4. As a tactical workaround, try `init="prior_draw"` with a different
   `init_rng` to land at a different starting point — but this hides
   the underlying problem.

## See also

- `ptgp/utils.py:check_init` — the diagnostic function this pitfall
  is named after.
- [bad_priors](bad_priors.md) — most common root cause.
- [large_grad_at_init](large_grad_at_init.md) — the milder sibling
  failure.
