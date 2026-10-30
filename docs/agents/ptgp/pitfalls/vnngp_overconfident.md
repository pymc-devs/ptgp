---
name: vnngp_overconfident
severity: medium
applies_to: [VNNGP]
symptoms:
  - held-out coverage of predictive intervals well below nominal
  - test NLL much worse than RMSE suggests
  - q_sd_ratio small while screening_ratio is moderate
related_pitfalls: [vnngp_neighbors_uninformative, sigma_collapse]
---

## Detection

```python
q_sd = np.array([d.q_sd_ratio for d in history])
lo, hi = mean - 1.96 * np.sqrt(var), mean + 1.96 * np.sqrt(var)
coverage = np.mean((y_test >= lo) & (y_test <= hi))   # predictions with incl_lik=True
overconfident = coverage < 0.85
```

## Diagnosis

With `block_size=1`, q is mean-field: it ignores posterior correlations
between inducing values. Mean-field variational inference underestimates
marginal variances when the true posterior is correlated (Bishop 2006,
Sec 10.1.2), and with inducing points at every input, neighboring values are
strongly correlated a posteriori. Song & Datta (arXiv:2507.12251, Sec 4) and
Cao et al. (arXiv:2301.13303, Fig 10) both report VNNGP predictive variances
that are too small. Conditioning a test point on only its `k` nearest
inducing points is not the cause; the variance shrinkage comes from the
mean-field q.

## Fix

1. **Use block q:** `VNNGP(..., block_size=16)` or larger. Each block keeps a
   full covariance, so correlations between neighbors in the same block are
   kept. Check `NeighborUpdate.block_coverage` from `recompute_neighbors`:
   the fraction of neighbor pairs sharing a block. Low coverage means most
   correlations still cross block boundaries.
2. **Partition blocks in the trained metric** by passing `scale=` to `VNNGP`,
   so blocks are compact in the distance that matters.
3. **Check the noise level.** VNNGP tends to learn small likelihood noise
   (Wu, Pleiss & Cunningham, arXiv:2202.01694, Table 3), which narrows
   observation intervals; see [sigma_collapse](sigma_collapse.md).

## See also

- [vnngp_neighbors_uninformative](vnngp_neighbors_uninformative.md)
