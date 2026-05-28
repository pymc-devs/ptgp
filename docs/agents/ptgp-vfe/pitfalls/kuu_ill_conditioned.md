---
name: kuu_ill_conditioned
severity: high
applies_to: [VFE]
symptoms:
  - GreedyVarianceDiagnostics.kuu_condition_number > 1e8
  - kuu_n_small_eigenvalues > 0
  - non-finite loss or grad at init (downstream symptom)
related_pitfalls: [bad_priors, inducing_collapse, non_finite_at_init, large_grad_at_init]
---

## Detection

```python
diag.kuu_condition_number > 1e8
or diag.kuu_n_small_eigenvalues > 0
or diag.kuu_min_eigenvalue < diag.kuu_eig_threshold
```

`repr(diag)` prints all four eigenvalue fields side-by-side; that's
the fastest visual check. The `scripts/check_inducing.py` CLI prints
verdicts on each.

## Diagnosis

Kuu is `K(Z, Z) + jitter * I`. It's near-singular when:

1. **Two or more Z rows are near-duplicates** — see
   [inducing_collapse](inducing_collapse.md). Direct test: pairwise
   distances on Z; the closest pair is below the kernel's effective
   resolution (~lengthscale).
2. **The kernel's effective rank is below M** — typical of long
   lengthscales, where many Z rows look "the same" to the kernel.
   The Burt-style greedy can mitigate this, but won't if the kernel
   is wrong (see (3) below).
3. **Bad priors picked the kernel that makes Kuu ill-conditioned**.
   Most common: prior median lengthscale much larger than the data
   spread. See [bad_priors](bad_priors.md). The fix here is to audit
   the priors, **not** to pile on more jitter.

The `scripts/check_inducing.py` 3-panel diagnostic shows this: if the
trace curve is *flat from the start*, the kernel is probably wrong
for the data scale.

## Fix

In order of preference (do not skip ahead):

1. **Audit priors**: are they centred at the right scale? Use
   `pg.utils.get_initial_params(model, init="prior_median")` to print
   the median values. Do they match what `np.std(X, axis=0)` and
   `np.std(y)` suggest? If not, fix the priors first. See
   [bad_priors](bad_priors.md).
2. **Check for duplicate Z**: if `kuu_n_small_eigenvalues > 0` and
   `kuu_condition_number` is very large, look at pairwise distances
   on Z. See [inducing_collapse](inducing_collapse.md).
3. **Lower M** if `M ≳ N/4` — VFE is no longer in its design regime.
4. **Increase jitter** as a last resort, only if you've ruled out
   1–3. The default `_DEFAULT_JITTER = 1e-6` matches GPflow / GPJax /
   PyMC. You can override per-call (see `objectives.py` /
   `conditionals.py`); larger jitter biases the bound.

## See also

- [reference/interpretation.md](../reference/interpretation.md) —
  Kuu eigenvalue fields are on `GreedyVarianceDiagnostics`, not on
  `VFEDiagnostics`.
- [bad_priors](bad_priors.md) — most common upstream cause.
- [inducing_collapse](inducing_collapse.md) — duplicate Z rows.
- `ptgp/inducing.py` (`greedy_variance_init`) — produces the Kuu
  eigenvalue stats.
