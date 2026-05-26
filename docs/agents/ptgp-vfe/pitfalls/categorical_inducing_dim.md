---
name: categorical_inducing_dim
severity: medium
applies_to: [VFE]
symptoms:
  - X has integer-coded or one-hot categorical columns
  - Z values land between integer categories after training
  - duplicate Z rows after gradient updates push two together within a category
related_pitfalls: [inducing_collapse, kuu_ill_conditioned]
---

## Detection

If column `j` of `X` is integer-coded with categories
`cats = np.unique(X[:, j])`:

```python
# Z values not in the legal set
illegal = ~np.isin(Z[:, j], cats)
n_illegal = illegal.sum()
```

For one-hot blocks, look for non-{0,1} values and rows whose entries
don't sum to 1.

## Diagnosis

VFE's continuous Z assumption is wrong for categorical / integer-coded
columns. Two underlying causes:

1. **`greedy_variance_init` selected continuous-space rows** that
   happened to be integer-coded but treats the column as continuous —
   no problem at init, but downstream gradient training moves Z off
   the integer grid.
2. **Gradient-based Z optimisation** (Tier C, Tier D phase 2a) moves
   Z continuously, so a category-coded dim drifts to in-between
   values that no real `X` row has.

The collapsed bound's gradient w.r.t. that dim is well-defined but
meaningless — it's interpolating the kernel over a coordinate axis
that has no real data between integer values.

## Fix

See [reference/discrete_inputs.md](../reference/discrete_inputs.md)
for the three working approaches:

1. **Use a categorical kernel** (`Overlap` or `LowRankCategorical`)
   for the discrete dim. Cleanest.
2. **Per-category greedy init + freeze Z** (Tier B). Cheap retrofit.
3. **Snap-to-nearest-category post-hoc** with `cKDTree`. Cheapest
   retrofit, loses optimality, can introduce duplicates — combine
   with a dedup pass.

For approaches 2 and 3, do **not** unfreeze Z's categorical column
during gradient training; otherwise the problem just comes back.

## See also

- [reference/discrete_inputs.md](../reference/discrete_inputs.md) —
  the design recipes.
- [inducing_collapse](inducing_collapse.md) — categorical collapse
  manifests as duplicate rows after snap-to-category.
- `ptgp.kernels.categorical.Overlap`,
  `ptgp.kernels.categorical.LowRankCategorical` — the kernel
  primitives.
