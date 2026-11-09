# Discrete inputs with inducing points

Most real datasets have a categorical or integer-coded column. The
inducing-point tools (VFE, SVGP) assume continuous inputs, so dropping a categorical column
into `X` without thinking causes silent failures.

## Why the defaults assume continuous inputs

Two places handle `X` continuously:

1. **`greedy_variance_init`** runs a pivoted Cholesky of `K(X, X)`
   using the *continuous* kernel. It picks `M` rows of `X` whose
   discrete-column values are whatever happened to be in those rows.
2. **Gradient-based Z optimisation** (trainable Z, including phase 2a of
   `minimize_staged_vfe`) moves Z in continuous space. The gradient step lands between integer
   categories.

For an integer-coded categorical column, both behaviours are wrong:
greedy may select inducing points with awkward category combinations,
and the gradient pulls Z to between-category values that no real data
point sits at.

## Three working approaches

Ordered cleanest to most retrofit-friendly.

### 1. Use a categorical kernel for the discrete dim

ptgp ships two:

- `ptgp.kernels.Overlap(input_dim, active_dims=[j])`: Hamming kernel,
  the fraction of active categorical columns that match. With one
  column it is a delta kernel, `1` if same category and `0` otherwise,
  so a product with a continuous kernel gives independent per-category
  functions with shared hyperparameters.
- `ptgp.kernels.LowRankCategorical(input_dim, num_levels, W, kappa,
  active_dims=[j])`: learned similarity `B = W W^T + diag(kappa)`
  between the levels of one column. This is the Coregion / ICM kernel.

Combine via product (multiplicative; e.g. shared smooth function
modulated per category) or sum (additive; per-category offset on top
of a shared smooth):

```python
# X has a continuous column 0 and an integer-coded categorical column 1
k_cont = eta**2 * pg.kernels.Matern52(input_dim=2, ls=ls, active_dims=[0])

# Multiplicative (ICM with LowRankCategorical; independent tasks with Overlap)
k = k_cont * pg.kernels.LowRankCategorical(
    input_dim=2, num_levels=L, W=W, kappa=kappa, active_dims=[1]
)

# Additive: per-category offset on top of a shared smooth function
k = k_cont + pg.kernels.Overlap(input_dim=2, active_dims=[1])
```

Z has all `D` columns, because each kernel slices its own
`active_dims` from `Z` just as it does from `X`. The categorical column
of Z is **enumerated**: typically one Z block per category, or a learned
subset, and greedy selection runs on the continuous columns within each
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
Z (`Points(Z)` with the concrete array) so the gradient never tries to
interpolate between categories.

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
introduce duplicates. Combine with a dedup pass (the same logic as the
near-duplicate removal in `kmeans_init`) to drop near-duplicate Z rows.

## Ordinal columns

If an integer column is genuinely ordinal (a rank, not a label),
continuous treatment is fine — *if* the lengthscale prior allows
resolution at integer spacing. Otherwise treat as categorical.

## Z optimisation under categorical kernels

Don't gradient-train Z's categorical dim. Two options:

- **Freeze Z entirely**: `Points(Z)` with the concrete array, or `frozen_vars={Z_var: Z}`.
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
