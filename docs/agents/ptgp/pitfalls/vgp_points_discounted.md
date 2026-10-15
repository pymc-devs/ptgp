---
name: vgp_points_discounted
severity: low
applies_to: [VGP]
symptoms:
  - info < 0 for some points
  - lam pinned near zero at those points
  - Student-t outliers or badly misclassified Bernoulli points have little effect on q(f)
related_pitfalls: [vgp_not_converged]
---

## Detection

```python
d = pg.gp.get_vgp_point_diagnostics(vgp, fit, X, y)
discounted = np.flatnonzero(d.info < 0)
```

## Diagnosis

`info` is the expected negative second derivative of the log-likelihood at
each point. For log-concave likelihoods (Gaussian, Poisson, NegativeBinomial,
logit Bernoulli) it is positive. For non-log-concave ones it can be negative:

- **StudentT:** residuals beyond about `sigma * sqrt(nu)`, i.e. outliers.
- **Bernoulli with the default probit link:** its label-flip floor,
  `p = 0.998 Phi(f) + 0.001`, makes it non-log-concave. For `y = 1`, `info`
  turns negative for badly misclassified points (about `f < -2.3`), which are
  then treated as likely mislabels.

Opper & Archambeau only require `K^-1 + diag(lam)` to be positive definite,
so negative `lam` is allowed in the paper. ptgp constrains `lam > 0` through a
softplus, so at these points `lam` is pinned near zero and the point is
discounted. The optimum is over a restricted family and can differ from the
unrestricted Gaussian VI optimum. This is usually the intended robustness, not
a failure.

## Fix

1. **Check the flagged rows** for data errors, outliers, or mislabels.
2. **If the points should count fully,** use a log-concave likelihood:
   Gaussian instead of StudentT, or the logit link
   (`Bernoulli(invlink=pt.sigmoid)`) instead of the default probit.

## See also

- [reference/vgp_interpretation.md](../reference/vgp_interpretation.md):
  `score` and `info` by likelihood.
