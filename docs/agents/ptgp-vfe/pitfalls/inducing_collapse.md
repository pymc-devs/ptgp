---
name: inducing_collapse
severity: high
applies_to: [VFE]
symptoms:
  - two or more Z rows are duplicates or near-duplicates
  - GreedyVarianceDiagnostics.kuu_n_small_eigenvalues > 0
  - GreedyVarianceDiagnostics.kuu_condition_number very large
related_pitfalls: [kuu_ill_conditioned, categorical_inducing_dim]
---

## Detection

After training (or at any point with a current Z):

```python
from scipy.spatial.distance import pdist
d = pdist(Z)
collapsed = (d < 1e-6).any()
```

Or use the eigenvalue indicator:

```python
diag.kuu_n_small_eigenvalues > 0  # at default kuu_eig_threshold = 1e-4
```

`kmeans_init` already deduplicates (`inducing.py:101-117`); collapse
under VFE happens when **gradient-trained** Z drifts two rows
together.

## Diagnosis

Two Z rows that are nearly equal make Kuu's null space grow. The
collapsed bound's gradient w.r.t. Z is well-defined when Kuu is
PSD-and-invertible, but as two rows merge, the gradient becomes
ill-conditioned and the optimiser can keep pushing them together —
the partial derivatives at the singular point still point "merge"
because the bound is symmetric in a redundant pair.

Adding the [DPP regulariser](../reference/api.md#objectives--ptgpobjectivespy)
`+ alpha * dpp_regularizer(vfe)` to the objective penalises this.
The repulsive term `log det K(Z, Z)` goes to `-inf` as any two
points collapse.

## Fix

1. **Add the DPP regulariser** if Z is gradient-trained:
   ```python
   def objective(vfe, X, y):
       return collapsed_elbo(vfe, X, y).elbo + 0.1 * dpp_regularizer(vfe)
   ```
   Tune `alpha` (start at 0.1, increase if collapse persists).
   This makes the objective a *regularised* objective, not a strict
   ELBO — that's the trade.
2. **Use Tier B** (frozen Z) — the simplest fix is to not train Z at
   all. With a good greedy init, this is often sufficient.
3. If collapse happens *during greedy* (rare), it's a kernel issue —
   see [kuu_ill_conditioned](kuu_ill_conditioned.md).
4. Categorical inducing dims that snap to the same category produce
   collapse-like duplicates; see
   [categorical_inducing_dim](categorical_inducing_dim.md).

## See also

- `ptgp/objectives.py:dpp_regularizer` — the repulsion term.
- `ptgp/inducing.py:101-117` — `kmeans_init`'s built-in dedup.
- [kuu_ill_conditioned](kuu_ill_conditioned.md) — downstream
  numerical failure.
