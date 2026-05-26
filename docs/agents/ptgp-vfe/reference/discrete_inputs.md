# Discrete inputs in a VFE model

Most real datasets have a categorical or integer-coded column. VFE's
defaults assume continuous inputs, so dropping a categorical column
into `X` without thinking causes silent failures.

## Why VFE's defaults assume continuous inputs

Two places in VFE handle `X` continuously:

1. **`greedy_variance_init`** runs a pivoted Cholesky of `K(X, X)`
   using the *continuous* kernel. It picks `M` rows of `X` whose
   discrete-column values are whatever happened to be in those rows.
2. **Gradient-based Z optimisation** (Tier C, Tier D phase 2a) moves
   Z in continuous space. The gradient step lands between integer
   categories.

For an integer-coded categorical column, both behaviours are wrong:
greedy may select inducing points with awkward category combinations,
and the gradient pulls Z to between-category values that no real data
point sits at.

## Three working approaches

Ordered cleanest to most retrofit-friendly.

### 1. Use a categorical kernel for the discrete dim

ptgp ships two:

- `ptgp.kernels.categorical.Overlap` — delta kernel: `1` if same
  category, `0` otherwise. ICM-style multitask.
- `ptgp.kernels.categorical.LowRankCategorical` — learned low-rank
  similarity matrix between categories.

Combine via product (multiplicative; e.g. shared smooth function
modulated per category) or sum (additive; per-category offset on top
of a shared smooth):

```python
# Multiplicative (ICM-style)
k = k_cont(x_cont) * Overlap(x_cat)

# Additive
k = k_cont + LowRankCategorical(x_cat)
```

Z lives in the continuous dims only. The categorical dim of Z is
**enumerated**: typically one Z block per category, or a learned
subset. The continuous block is greedy-initialised within each
category's data.

This is the cleanest design and the one to prefer when you can refactor
the kernel.

### 2. Per-category greedy init

When refactoring the kernel isn't an option, run `greedy_variance_init`
**within** each category and stack the results. Allocate
`M_k ≈ M · N_k / N` inducing points per category `k`:

```python
Z_blocks = []
for cat, X_k in groupby_category(X):
    M_k = round(M * len(X_k) / N)
    ip_k, _ = greedy_variance_init(X_k, M_k, kernel)
    Z_blocks.append(ip_k.Z)
Z = np.vstack(Z_blocks)
```

Z's categorical column is filled by category and *never moves*. Freeze
Z (Tier B) so the gradient never tries to interpolate between
categories.

### 3. Snap-to-nearest-category post-hoc

Run continuous greedy as if the categorical column were just another
continuous feature, then project each Z row's discrete dim onto the
nearest legal category — typically with `scipy.spatial.cKDTree` against
the unique category codes:

```python
from scipy.spatial import cKDTree

cats = np.unique(X[:, cat_dim]).reshape(-1, 1)
tree = cKDTree(cats)
_, idx = tree.query(Z[:, [cat_dim]])
Z[:, cat_dim] = cats[idx, 0]
```

Cheap to retrofit but loses the optimality guarantee of greedy and can
introduce duplicates. Combine with a dedup pass (the same logic as in
`kmeans_init`, `inducing.py:101`) to drop near-duplicate Z rows.

## Ordinal columns

If an integer column is genuinely ordinal (a rank, not a label),
continuous treatment is fine — *if* the lengthscale prior allows
resolution at integer spacing. Otherwise treat as categorical.

## Z optimisation under categorical kernels

Don't gradient-train Z's categorical dim. Two options:

- **Freeze Z entirely** (Tier B with `frozen_vars={Z_var: Z_init}`).
- **Split Z** into a trainable continuous block + a frozen categorical
  block. Custom; see how `Z_var` is plumbed in `minimize_staged_vfe` —
  you'd need a similar two-block setup.

## When it goes wrong

Open [categorical_inducing_dim](../pitfalls/categorical_inducing_dim.md)
when:

- Z values land between categories after training.
- Z duplicates after gradient updates push points together.
- `kuu_ill_conditioned` traceable to the categorical block — typically
  too many Z rows in one category.
