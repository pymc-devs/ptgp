---
name: vnngp_conditioning
severity: high
applies_to: [VNNGP]
symptoms:
  - min_log_F close to log(jitter)
  - NaN or very large gradients during training
  - screening_ratio near jitter / k_jj
related_pitfalls: [vnngp_neighbors_uninformative, lengthscale_runaway]
---

## Detection

```python
min_log_F = np.array([d.min_log_F for d in history])
ill_conditioned = min_log_F[-1] < np.log(vnngp.jitter) + 3
```

## Diagnosis

`F_j` is the prior variance of `u_j` given its neighbors. It collapses toward
the jitter when a neighbor is nearly at the same location as `z_j`, measured
on the lengthscale. VNNGP removes exact duplicates (inducing points are the
unique rows of `X`), but near-duplicates, such as inputs that differ only by
rounding or measurement noise, remain. A lengthscale that grows far beyond
the point spacing makes all nearby points nearly duplicates. A tiny `F_j`
makes the KL term `E_q[r_j^2] / (2 F_j)` very large and its gradient
unstable.

## Fix

1. **Round or bin inputs** to the resolution that matters, so near-duplicates
   become exact duplicates and merge into one inducing point.
2. **Raise `jitter`** (default 1e-6). GPyTorch uses 1e-3.
3. **Constrain the lengthscale** with a prior whose density vanishes at
   large values relative to the input range.

## See also

- [vnngp_neighbors_uninformative](vnngp_neighbors_uninformative.md): the
  opposite extreme, `screening_ratio` near 1.
