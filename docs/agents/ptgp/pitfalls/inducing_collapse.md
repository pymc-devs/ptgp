---
name: inducing_collapse
severity: high
applies_to: [VFE, SVGP]
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

`diag` here is any diagnostic carrying the `kuu_*` fields:
`GreedyVarianceDiagnostics`, or `KernelHealthDiagnostics` from
`compute_inducing_diagnostics(kernel, X, Z)`, which checks any Z,
including a trained SVGP one.

`kmeans_init` already deduplicates (see `kmeans_init` in
`ptgp/inducing.py`); collapse happens when **gradient-trained** Z drifts
two rows together.

## Diagnosis

Two Z rows that are nearly equal make Kuu's null space grow. The
objective's gradient with respect to Z is well defined when Kuu is
positive definite, but as two rows merge, the gradient becomes
ill-conditioned and the optimiser can keep pushing them together: the
partial derivatives at the singular point still point "merge" because
the bound is symmetric in a redundant pair.

Adding the [DPP regulariser](../reference/api.md#objectives--ptgpobjectivespy)
`+ alpha * dpp_regularizer(vfe)` to the objective penalises this.
The repulsive term `log det K(Z, Z)` goes to `-inf` as any two
points collapse.

## Fix

1. **Add the DPP regulariser** if Z is gradient-trained. For VFE:
   ```python
   def objective(vfe, X, y):
       return collapsed_elbo(vfe, X, y).elbo + 0.1 * dpp_regularizer(vfe)
   ```
   Tune `alpha` (start at 0.1, increase if collapse persists).
   This makes the objective a *regularised* objective, not a strict
   ELBO — that's the trade.
2. **Freeze Z** (`Points(Z0)` with a concrete array): the simplest fix
   is to not train Z at all. With a good greedy init, this is often
   sufficient.
3. If collapse happens *during greedy* (rare), it's a kernel issue —
   see [kuu_ill_conditioned](kuu_ill_conditioned.md).
4. Categorical inducing dims that snap to the same category produce
   collapse-like duplicates; see
   [categorical_inducing_dim](categorical_inducing_dim.md).

## See also

- `ptgp/objectives.py:dpp_regularizer` — the repulsion term.
- `ptgp/inducing.py:kmeans_init`: built-in near-duplicate removal.
- [kuu_ill_conditioned](kuu_ill_conditioned.md) — downstream
  numerical failure.
