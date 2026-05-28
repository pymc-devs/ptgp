# API reference (call-site)

One-liners for the ptgp functions this skill references. Source paths are
absolute within the repo. Read the function docstring for the full
parameter list.

## Objectives — `ptgp/objectives.py`

- **`marginal_log_likelihood(gp, X, y) -> MLLTerms(mll, fit, logdet)`**
  — exact GP marginal log-likelihood. Used in Tier A.
- **`elbo(svgp, X, y, n_data=None) -> ELBOTerms(elbo, var_exp, kl)`** —
  SVGP / non-conjugate ELBO. Not VFE.
- **`collapsed_elbo(vfe, X, y) -> CollapsedELBOTerms(elbo, fit,
  trace_penalty, nystrom_residual)`** — Titsias collapsed ELBO in the
  Bauer/GPflow factored form. Used by Tiers B–D as the optimisation
  target.
- **`vfe_diagnostics(vfe, X, y) -> VFEDiagnostics(elbo, fit,
  trace_penalty, nystrom_residual, sigma, fit_per_n,
  excess_fit_per_n)`** — diagnostic-only namedtuple producer for
  `compile_scipy_diagnostics` + `tracked_minimize`.
- **`fitc_log_marginal_likelihood(vfe, X, y) -> FITCTerms(fitc, fit,
  logdet)`** — FITC approximate log marginal likelihood. Uses the true
  per-point diagonal ``ν_i = Kff_ii - Q_ii + σ²`` instead of the flat
  ``σ²`` of VFE. Not a lower bound; tends to give better-calibrated
  predictive variances. Same Woodbury factorisation as `collapsed_elbo`.
- **`dpp_regularizer(vfe, jitter=1e-6) -> scalar`** — `log det K(Z, Z)`
  repulsive regulariser. Add a positive multiple to `collapsed_elbo` to
  fight `inducing_collapse`.

## Optim — `ptgp/optim/training.py`

- **`compile_scipy_objective(objective_fn, gp, X_var, y_var, model=None,
  extra_vars=None, extra_init=None, frozen_vars=None, include_prior=True,
  init="prior_median", init_rng=None) -> (fun, theta0, unpack, sp,
  se)`** — scipy-compatible loss + grad. `init` defaults to
  `"prior_median"` (improper priors fall back per-RV; see the function
  docstring).
- **`compile_scipy_diagnostics(diagnostic_fn, gp, X_var, y_var, ...) ->
  diag_fn`** — companion that compiles a forward-only pass returning
  every namedtuple field at a given theta. Pair with
  `tracked_minimize`.
- **`tracked_minimize(fun, theta0, args, diag_fn=None,
  print_every=None, **scipy_kwargs) -> (result, history)`** — scipy
  wrapper that calls `diag_fn` per iteration and accumulates the
  namedtuple history. Used in Tier B / Tier C. On `KeyboardInterrupt`,
  returns gracefully with `result.status == 99`,
  `result.message` starting with `"KeyboardInterrupt"`, and
  `result.x` set to the most recent iterate seen by the callback (or
  `theta0` if no iteration completed).
- **`minimize_staged_vfe(objective_fn, gp_model, X_var, y_var, X, y,
  model, sigma_init, Z_var, Z_init, ...) -> (result, history,
  phase_labels, unpack, sp, se)`** — Tier D entry point. Four-phase
  staged schedule preventing sigma collapse. See the docstring at
  `training.py:728` for the per-phase trainable / frozen split. On
  `KeyboardInterrupt` during any sub-phase, halts that phase via
  `tracked_minimize`'s graceful interrupt handler, runs `unpack` on
  the last iterate, and returns immediately. The returned
  `result.status == 99`; `unpack`/`sp`/`se` correspond to the
  *interrupted* phase, so `compile_predict` wires up to the
  partially-trained state.

## Inducing — `ptgp/inducing.py`

- **`greedy_variance_init(X, M, kernel, threshold=0.0, jitter=1e-12,
  rng=None, eig_threshold=1e-4, compile_kwargs=None) -> (Points,
  GreedyVarianceDiagnostics)`** — Burt et al. ConditionalVariance /
  pivoted-Cholesky inducing-point selection.
- **`random_subsample_init(X, M, rng=None, kernel=None, jitter=1e-6,
  eig_threshold=1e-4, compile_kwargs=None) -> (Points,
  RandomSubsampleDiagnostics)`** — pick `M` rows of `X` uniformly at
  random. If `kernel` is given, the diagnostic's `kernel_health` field
  is populated with `KernelHealthDiagnostics`.
- **`kmeans_init(X, M, rng=None, tol=1e-6, kernel=None, jitter=1e-6,
  eig_threshold=1e-4, compile_kwargs=None) -> (Points,
  KMeansDiagnostics)`** — k-means++ centroids with built-in
  near-duplicate removal at `tol`. `kernel=` populates `kernel_health`
  on the diagnostic.
- **`compute_inducing_diagnostics(kernel, X, Z, jitter=1e-6,
  eig_threshold=1e-4, compile_kwargs=None) ->
  KernelHealthDiagnostics`** — kernel-derived health metrics for an
  arbitrary `(kernel, X, Z)`. Same computation that the `kernel=`
  argument on the init routines triggers.
- **`Points(Z)`** — wraps a `(M, D)` array as `InducingVariables`.
- **`GreedyVarianceDiagnostics`** — dataclass with fields
  `trace_curve, d_final, total_variance, kuu_min_eigenvalue,
  kuu_max_eigenvalue, kuu_condition_number, kuu_n_small_eigenvalues,
  kuu_eig_threshold`. `repr()` prints a one-screen summary.
- **`RandomSubsampleDiagnostics`** — dataclass with fields
  `M_requested, M_returned, N_candidates, n_unique,
  pairwise_min_distance, pairwise_mean_distance, kernel_health`
  (`KernelHealthDiagnostics | None`).
- **`KMeansDiagnostics`** — dataclass with fields `M_requested,
  M_returned, n_removed_duplicates, dedup_tol, inertia,
  pairwise_min_distance, pairwise_mean_distance, kernel_health`
  (`KernelHealthDiagnostics | None`).
- **`KernelHealthDiagnostics`** — dataclass with fields `d_final,
  total_variance, nystrom_residual, kuu_min_eigenvalue,
  kuu_max_eigenvalue, kuu_condition_number, kuu_n_small_eigenvalues,
  kuu_eig_threshold`. `repr()` prints a one-screen summary.

## Utils — `ptgp/utils.py`

- **`check_init(fun_or_theta0, ..., model, extra_vars=None,
  extra_init=None) -> dict`** — evaluate loss + grad at theta0, report
  per-parameter values and gradients, flag non-finite or
  large-gradient entries (`_LARGE_GRAD_WARN = 1e4`). Run before scipy
  starts.
- **`get_initial_params(model, init="prior_median", rng=None,
  n_median_samples=500) -> dict`** — constrained-space values for all
  free RVs at the chosen init strategy. Used to build numerical proxy
  kernels for `greedy_variance_init`.
