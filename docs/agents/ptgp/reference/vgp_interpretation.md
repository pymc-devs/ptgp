# Interpreting VGPPointDiagnostics

`pg.gp.get_vgp_point_diagnostics(vgp, fit, X, y)` returns per-point quantities
of a fitted `VGP`, which places q(f) = N(m + K alpha, S) over the latent
function at the N training inputs, with S = (K^-1 + diag(lam))^-1 (Opper &
Archambeau, 2009). Every array field has shape `(N,)` and is aligned to the
training rows. Notation follows the paper, where `nu` is ptgp's `alpha`.

| Field | Meaning |
| --- | --- |
| `alpha`, `lam` | Trained variational parameters. |
| `fmean`, `fvar` | Marginal mean and variance of q(f_i). `fvar` is `diag(S)`. |
| `score` | `E_q[d log p(y_i given f_i) / df_i]`, the paper's `nu_bar` (Eq. 13). |
| `info` | `E_q[-d^2 log p(y_i given f_i) / df_i^2]`, the paper's `lambda_bar` (Eq. 14). |
| `g_nu` | Gradient of the negative ELBO with respect to `alpha`: `K (alpha - score)` (Eq. 11). |
| `g_lambda` | Gradient of the negative ELBO with respect to `lam`: `0.5 (S o S)(lam - info)` (Eq. 12). |
| `converged` | `fit.result.success`. |

Read `score` and `info`, not `alpha` and `lam`. They depend only on the
marginals of q(f), so they are well defined even when `K` is ill-conditioned.
Check convergence with `g_nu` and `g_lambda`; see
[vgp_not_converged](../pitfalls/vgp_not_converged.md).

## Reading score and info by likelihood

`score` is the point's pull on the posterior mean (a pseudo-residual). `info`
is the point's effective precision: how much it tightens q(f_i).

| Likelihood | `score` | `info` |
| --- | --- | --- |
| Gaussian(sigma) | `(y - fmean) / sigma^2` | `1 / sigma^2` for every point. |
| Poisson (log link) | `y - rate`, with `rate = exp(fmean + fvar / 2)` | `rate`: counts in high-rate regions are more informative. |
| NegativeBinomial(alpha), log link | `E[alpha (y - mu) / (alpha + mu)]` with `mu = exp(f)`: the Poisson residual shrunk by `alpha / (alpha + mu)`. | `E[alpha mu (alpha + y) / (alpha + mu)^2]`, always positive. Below the Poisson rate for typical counts, above it for large counts. |
| Bernoulli, logit (`invlink=pt.sigmoid`) | Signed residual `y - p` | `E[p (1 - p)]`: peaks at the decision boundary, near zero for confident points. |
| Bernoulli, probit (default) | Signed residual | The default link has a label-flip floor, `p = 0.998 Phi(f) + 0.001`. For `y = 1`, `info` is near zero for confident correct points (`f > 2`), rises through the boundary to a peak of about 0.78 near `f = -1`, then turns negative for badly misclassified points (about `f < -2.3`). See [vgp_points_discounted](../pitfalls/vgp_points_discounted.md). |
| StudentT(nu, sigma) | Bounded: large residuals are downweighted | Negative for residuals beyond about `sigma * sqrt(nu)`. See [vgp_points_discounted](../pitfalls/vgp_points_discounted.md). |
| CompositeLikelihood | Per point, by the group that governs it | Split the arrays by the composite's group labels (the column of `X` passed as `group`) before comparing across groups. |

## References

- Opper, M. and Archambeau, C. (2009). The variational Gaussian approximation
  revisited. Neural Computation 21(3), 786-792.
- Khan, M. E., Mohamed, S. and Murphy, K. P. (2012). Fast Bayesian inference for
  non-conjugate Gaussian process regression. NeurIPS.
