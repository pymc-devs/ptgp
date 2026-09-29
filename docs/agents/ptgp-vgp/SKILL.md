---
name: ptgp-vgp
description: Diagnose ptgp VGP (Opper-Archambeau full variational GP) fits and read the per-point variational quantities. Use when working with VGP, vgp_elbo, init_vgp_params, get_vgp_point_diagnostics, VGPPointDiagnostics, CompositeLikelihood on VGP, or non-Gaussian likelihoods without inducing points.
---

# ptgp-vgp: VGP fit and data diagnostics

VGP places q(f) = N(m + K alpha, S) over the latent function at the N training
inputs, with S = (K^-1 + diag(lam))^-1 (Opper & Archambeau, 2009). The
per-point quantities from `pg.gp.get_vgp_point_diagnostics(vgp, fit, X, y)` tell
you whether the fit converged and how each observation shapes the posterior.

## Workflow

1. **Convergence.** Look at `g_nu` and `g_lambda`, the negative-ELBO gradients
   with respect to `alpha` and `lam`. Do not rely on `converged`
   (`fit.result.success`) alone, and do not compare `alpha` with `score`
   directly. See [reference/interpretation.md](reference/interpretation.md#convergence).
2. **Per-point reading.** Use `score` and `info`, not `alpha` and `lam`. They
   depend only on the marginals of q(f), so they are well defined even when
   `K` is ill-conditioned. See the likelihood table in
   [reference/interpretation.md](reference/interpretation.md#reading-score-and-info-by-likelihood).
3. **Discounted points.** Points with `info < 0` are being discounted: outliers
   under Student-t, and badly misclassified points under the default probit
   Bernoulli (its 0.001 label-flip floor makes it non-log-concave). Under
   ptgp's parameterization `lam` is pinned near zero there.

## Known limitations

- **Slow convergence.** L-BFGS on the `(alpha, lambda)` parameterization is
  non-concave in `lambda` and converges slowly (Khan, Mohamed & Murphy, 2012).
  Hundreds to thousands of iterations are normal. Raise
  `options={"maxiter": ...}` in `pg.fit` and re-check `g_nu` / `g_lambda`.
- **`lam > 0` is enforced by softplus.** Opper and Archambeau only require
  K^-1 + diag(lam) to be positive definite, so negative `lam` is allowed in the
  paper. For non-log-concave likelihoods (Student-t, default probit
  Bernoulli), ptgp's optimum is over a restricted family and can differ from
  the unrestricted Gaussian-VI optimum.
