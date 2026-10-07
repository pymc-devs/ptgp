# Choosing M (number of inducing points)

`M` is the one knob in an inducing-point model (VFE, SVGP) that the
optimizer cannot tune. You pick it before training, from a one-shot
greedy-variance run.

## Why M matters

The collapsed bound is tight when `nystrom_residual ≈ 0`. That requires
`Q = Kuf.T inv(Kuu) Kuf` to track `Kff` well on the diagonal — which
requires Z to cover the data well. If `M` is too small, no Z layout
covers the data; if `M` is too large, you waste compute (Cholesky of
`M × M` dominates) and Kuu becomes more prone to near-singularity.

There is no magic value. The right `M` depends on data extent, kernel
lengthscale, input dimensionality, and whether the data is clustered.
Use the rule below, not a number lifted from a tutorial.

## Selection rule

1. Build a **numerical proxy kernel**: substitute prior-median values
   for any symbolic hyperparameters (use `pg.utils.get_initial_params`
   to pull the unconstrained-zero or prior-median values).
2. Run greedy at a generous upper bound:

   ```python
   M_max = min(N, 1024)
   _, diag = greedy_variance_init(X, M_max, proxy_kernel)
   ```

3. Plot `diag.trace_curve / diag.total_variance` against `M`, where
   `diag.trace_curve[m]` is the residual after `m` points (so the x-axis
   runs from 0 to `M_max`). This is the **fraction-unexplained curve**.
4. Find the **knee** — the M at which the curve crosses ~1% (and look
   at the slope past it). Any M between the knee and the elbow of the
   slope-change is a defensible choice. Below ~5% is usable; below ~1%
   is a tight bound; below ~0.1% is wasted M.

The `scripts/check_inducing.py` CLI plots this with a 1% dashed
threshold by default.

## Lengthscale interaction

The trace curve depends on the kernel's lengthscale: short
lengthscales need more `M` to cover the same data extent. After
training, if the learned lengthscale shrank substantially relative to
the prior median, **re-run greedy on the trained kernel**:

```python
# After training:
trained_kernel = ...  # kernel evaluated at trained hyperparameters
_, diag2 = greedy_variance_init(X, M, trained_kernel)
```

If the new trace curve hasn't flattened by the M you chose, increase
`M` and retrain. Re-running greedy on the trained kernel and freezing
the new Z is also the fix for a poor layout at the right `M`.

## Floors and ceilings

- **Floor.** `M` must be at least the rank of the signal you care
  about. Rough heuristic: input dimension × number of active basis
  components. For smooth 1-D signals, M of 50–100 is often enough.
  For 10-D modestly-smooth signals, expect 200–1000.
- **Ceiling.** M ≪ N is the whole point. At M ≳ N/4 the sparse model
  buys little over the exact GP (`pg.gp.Unapproximated`), and you pay
  the approximation's slack for nothing.

## When to revisit M

- Open [M_too_small](../pitfalls/M_too_small.md) if `nystrom_residual`
  stays large at convergence **and** the trace curve hasn't flattened
  by your chosen M.
- Open [inducing_layout_poor](../pitfalls/inducing_layout_poor.md) if
  the trace curve **has** flattened but `d_final` has hot spots — the
  problem is layout, not count, and re-running greedy with the
  trained kernel is the fix.
