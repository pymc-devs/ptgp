---
name: vnngp_neighbors_uninformative
severity: high
applies_to: [VNNGP]
symptoms:
  - screening_ratio near 1 at convergence
  - VNNGP predictions are much worse than an exact GP or SVGP on a subsample
  - ARD lengthscales differ by orders of magnitude across inputs
related_pitfalls: [vnngp_overconfident, vnngp_conditioning, lengthscale_runaway]
---

## Detection

```python
screening = np.array([d.screening_ratio for d in history])
uninformative = screening[-1] > 0.5
```

`history` is the list of `VNNGPDiagnostics` from calling the function built by
`compile_diagnostics(vnngp_diagnostics, ...)` on a fixed evaluation batch
every few steps.

## Diagnosis

`screening_ratio` is the median of `F_j / k_jj`: the fraction of each inducing
value's prior variance that its `k` neighbors leave unexplained. Near 0, the
neighbors pin `u_j` down and the nearest-neighbor prior is close to the full
GP. Near 1, each `u_j` is nearly independent of its neighbors, so the prior
has lost most of the correlation the kernel describes. Causes:

1. **Neighbors chosen in the wrong metric.** Neighbors start in Euclidean
   distance on the raw inputs. With ARD lengthscales, the nearest points in
   raw space are not the most correlated ones. With one relevant input among
   several, scaled neighbors with `k = 5` beat raw neighbors with `k = 50`
   (Katzfuss, Guinness & Lawrence, arXiv:2005.00386, Sec 4.2).
2. **`k` too small for the point density.** If the lengthscale is shorter than
   the spacing between points, no small neighbor set carries information.
3. **A kernel the Euclidean metric cannot describe** (Gibbs, WarpedInput,
   sums of kernels with different lengthscales, categorical inputs), where
   the most correlated points are not the nearest in any scaled distance.

## Fix

1. **Recompute neighbors in the trained metric** on the schedule from
   `recompute_steps(n_steps)` and once after training:
   `vnngp.recompute_neighbors(model, shared_params, shared_extras=shared_extras, scale=ls)`,
   where `ls` is the PyMC random variable (not its log-transformed value variable).
2. **Use the correlation metric** for kernels from cause 3:
   `vnngp.recompute_neighbors(..., metric="correlation")`. It selects the most
   correlated points under the current kernel (Kang & Katzfuss 2023,
   arXiv:2112.14591, Eq 4) and costs O(M^2) kernel evaluations per call.
3. **Increase `k`.** Cost per step grows as `k^3`; Wu, Pleiss & Cunningham
   (arXiv:2202.01694, Fig 2) use `k` from 32 to 256.
4. **Try `order="maximin"`** for low-dimensional inputs, where it improves on
   random ordering at the same `k` (Guinness 2018, arXiv:1609.05372, Sec 5).

## See also

- [vnngp_conditioning](vnngp_conditioning.md): the opposite extreme,
  `screening_ratio` near `jitter / k_jj`.
- [lengthscale_runaway](lengthscale_runaway.md): a lengthscale shrinking
  below the point spacing produces cause 2.
