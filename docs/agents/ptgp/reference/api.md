# API reference (call-site)

One-liners for the ptgp functions this skill references. Source paths are
relative to the repo root. Cite functions by name rather than line number,
and read the function docstring for the full parameter list.

## Objectives — `ptgp/objectives.py`

- **`marginal_log_likelihood(gp, X, y) -> MLLTerms(mll, fit, logdet)`**:
  exact GP marginal log-likelihood.
- **`elbo(svgp, X, y, n_data=None) -> ELBOTerms(elbo, var_exp, kl)`**:
  SVGP / non-conjugate ELBO. Not VFE.
- **`collapsed_elbo(vfe, X, y) -> CollapsedELBOTerms(elbo, fit,
  trace_penalty, nystrom_residual)`**: Titsias collapsed ELBO in the
  Bauer/GPflow factored form. The VFE training target. Here `nystrom_residual` is the unnormalised sum `tr(Kff - Qff)`.
- **`vfe_diagnostics(vfe, X, y) -> VFEDiagnostics(elbo, fit,
  trace_penalty, nystrom_residual, sigma, fit_per_n, excess_fit_per_n,
  frac_mean, frac_signal, frac_noise, var_ratio)`**: diagnostic-only
  namedtuple producer for `compile_scipy_diagnostics` + `tracked_minimize`.
  Here `nystrom_residual` is per point, `tr(Kff - Qff) / N`. Field meanings
  are in [interpretation.md](interpretation.md).
- **`variance_budget(gp, X, y) -> VarianceBudget(mean_var, signal_var,
  noise_var, total_var, frac_mean, frac_signal, frac_noise, var_ratio)`**:
  splits the model-implied variance of `y` into mean, GP signal, and noise
  parts. `vfe_diagnostics` already includes the `frac_*` and `var_ratio`
  fields.
- **`unapproximated_diagnostics(gp, X, y) -> UnapproximatedDiagnostics(mll,
  fit, logdet, sigma, fit_per_n, logdet_per_n, excess_fit_per_n, frac_mean,
  frac_signal, frac_noise, var_ratio)`**: the exact-GP analogue of
  `vfe_diagnostics`, for exact-GP histories.
- **`fitc_log_marginal_likelihood(vfe, X, y) -> FITCTerms(fitc, fit,
  logdet)`**: FITC approximate log marginal likelihood. Uses the true
  per-point diagonal `ν_i = Kff_ii - Q_ii + σ²` instead of the flat `σ²`
  of VFE. Not a lower bound; tends to give better-calibrated predictive
  variances. Same Woodbury factorisation as `collapsed_elbo`.
- **`dpp_regularizer(vfe, jitter=1e-6) -> scalar`**: `log det K(Z, Z)`
  repulsive regulariser. Add a positive multiple to `collapsed_elbo` to
  fight `inducing_collapse`.

The jitter added to `Kuu` inside the training objectives is fixed
(`_DEFAULT_JITTER`); only `dpp_regularizer` and the inducing-init /
diagnostic helpers below take a `jitter=` argument.

## Optim — `ptgp/optim/training.py`

- **`compile_scipy_objective(objective_fn, gp, X_var, y_var, model=None,
  extra_vars=None, extra_init=None, frozen_vars=None, include_prior=True,
  compile_kwargs=None, init="prior_median", init_rng=None) -> (fun,
  theta0, unpack, sp, se)`**: scipy-compatible loss + grad. `objective_fn`
  may return a scalar or a namedtuple (first field is taken as the
  scalar). Pass `extra_vars` and `extra_init` together, or neither: when
  both are omitted they default to `gp.extra_vars` / `gp.extra_init`,
  which for VFE come from the inducing variable (a trainable `Z` when it
  was built as `Points(Z_var, Z_init=Z0)`, nothing otherwise). Pass empty
  tuples to exclude all extras. `init` defaults to `"prior_median"`
  (improper priors fall back per RV; see the function docstring).
- **`compile_scipy_diagnostics(diagnostic_fn, gp, X_var, y_var,
  model=None, extra_vars=None, extra_init=None, frozen_vars=None,
  compile_kwargs=None, init="prior_median", init_rng=None) -> diag_fn`**:
  companion that compiles a forward-only pass returning every namedtuple
  field at a given theta, `diag_fn(theta, X, y)`. Use
  `diagnostic_fn=vfe_diagnostics` for the histories this skill's scripts
  read. Pass the same `extra_vars` / `frozen_vars` as the objective so the
  theta layouts match.
- **`tracked_minimize(fun, theta0, args, diag_fn=None,
  print_every=None, **scipy_kwargs) -> (result, history)`**: scipy
  wrapper that calls `diag_fn` per iteration and accumulates the
  namedtuple history. Progress is written to the `ptgp` logger every
  `print_every` iterations. Optimizer settings go through scipy kwargs,
  e.g. `options={"maxiter": 2000}`. On
  `KeyboardInterrupt`, returns gracefully with `result.status == 99`,
  `result.message` starting with `"KeyboardInterrupt"`, and `result.x`
  set to the most recent iterate seen by the callback (or `theta0` if no
  iteration completed).
- **`minimize_staged_vfe(objective_fn, gp_model, X_var, y_var, X, y,
  model, sigma_init, Z_var, Z_init, phase1_freeze_Z=False,
  phase1_maxiter=200, phase2_cycles=3, phase2_maxiter_Z=100,
  phase2_maxiter_hyper=100, phase3_maxiter=300, print_every=20,
  compile_kwargs=None, scipy_options=None, init="prior_median",
  init_rng=None) -> (result, history, phase_labels, unpack, sp, se)`**:
  Staged VFE trainer: a schedule that keeps sigma from inflating
  while `Z` is still poorly placed, which is what drives inducing-point
  collapse. See the `minimize_staged_vfe` docstring for the per-phase
  trainable / frozen split. Set per-phase budgets with the `phase*_maxiter`
  arguments; a `maxiter` inside `scipy_options` overrides all of them. On
  `KeyboardInterrupt` during any sub-phase, halts that phase via
  `tracked_minimize`'s graceful interrupt handler, runs `unpack` on the
  last iterate, and returns immediately. The returned
  `result.status == 99`; `unpack`/`sp`/`se` correspond to the
  *interrupted* phase, so `compile_predict` wires up to the
  partially-trained state.
- **`compile_training_step(objective_fn, gp, X_var, y_var, model=None,
  optimizer_fn=None, extra_vars=None, extra_init=None, frozen_vars=None,
  param_groups=None, include_prior=True, compile_kwargs=None,
  **optimizer_kwargs) -> (train_step, sp, se)`**: Adam / SGD step for
  minibatch training. Mostly SVGP; listed for completeness.
- **`compile_predict(gp, X_new_var, model, shared_params,
  extra_vars=None, shared_extras=None, X_train=None, y_train=None,
  incl_lik=False, compile_kwargs=None) -> predict_fn`**: prediction
  function reading the trained shared variables. VFE needs `X_train` /
  `y_train`.
- **`get_trained_params(model, shared_params) -> dict`**: trained values
  of every free RV in constrained space, `{rv_name: value}`.
- **`phase_sort_key(label)`**: sort key for `minimize_staged_vfe` phase
  labels, used by the history scripts.

## Inducing — `ptgp/inducing.py`

### Freezing or training inducing points

```python
ind = pg.inducing.Points(Z0)                # concrete array: Z is fixed
ind = pg.inducing.Points(Z_var, Z_init=Z0)  # trainable: picked up via gp.extra_vars
ind = pg.inducing.Points(Z_var)             # symbolic without init: pass
                                            # frozen_vars={Z_var: Z0} to the compile helpers
```

`pg.fit` and the compile helpers train whatever `gp.extra_vars` exposes, so
the first two forms need no extra arguments. Fixed Z from
`greedy_variance_init` is usually about as good as trained Z (Burt et al.
2020) and avoids [inducing_collapse](../pitfalls/inducing_collapse.md) and
[categorical_inducing_dim](../pitfalls/categorical_inducing_dim.md).


- **`greedy_variance_init(X, M, kernel, threshold=0.0, jitter=1e-12,
  rng=None, eig_threshold=1e-4, compile_kwargs=None) -> (Points,
  GreedyVarianceDiagnostics)`**: Burt et al. ConditionalVariance /
  pivoted-Cholesky inducing-point selection. With `threshold > 0` it
  returns the smallest prefix whose residual trace is below it.
- **`random_subsample_init(X, M, rng=None, kernel=None, jitter=1e-6,
  eig_threshold=1e-4, compile_kwargs=None) -> (Points,
  RandomSubsampleDiagnostics)`**: pick `M` rows of `X` uniformly at
  random. If `kernel` is given, the diagnostic's `kernel_health` field
  is populated with `KernelHealthDiagnostics`.
- **`kmeans_init(X, M, rng=None, tol=1e-6, kernel=None, jitter=1e-6,
  eig_threshold=1e-4, compile_kwargs=None) -> (Points,
  KMeansDiagnostics)`**: k-means++ centroids with built-in
  near-duplicate removal at `tol`. `kernel=` populates `kernel_health`
  on the diagnostic.
- **`compute_inducing_diagnostics(kernel, X, Z, jitter=1e-6,
  eig_threshold=1e-4, compile_kwargs=None) ->
  KernelHealthDiagnostics`**: kernel-derived health metrics for an
  arbitrary `(kernel, X, Z)`. Same computation that the `kernel=`
  argument on the init routines triggers.
- **`Points(Z, Z_init=None)`**: wraps inducing locations of shape
  `(M, D)`. An array `Z` is cast to `floatX` and stays fixed. A symbolic
  `Z_var` with `Z_init` becomes trainable: it is exposed through
  `extra_vars` / `extra_init`, so the compile helpers optimize it by
  default. A symbolic `Z_var` without `Z_init` must be supplied through
  `frozen_vars` or explicit `extra_vars` / `extra_init`.
- **`GreedyVarianceDiagnostics`**: dataclass with fields
  `trace_curve` (shape `(M + 1,)`), `d_final` (shape `(N,)`, aligned to the
  rows of `X`), `total_variance, kuu_min_eigenvalue,
  kuu_max_eigenvalue, kuu_condition_number, kuu_n_small_eigenvalues,
  kuu_eig_threshold`. `repr()` prints a one-screen summary.
- **`RandomSubsampleDiagnostics`**: dataclass with fields
  `M_requested, M_returned, N_candidates, n_unique,
  pairwise_min_distance, pairwise_mean_distance, kernel_health`
  (`KernelHealthDiagnostics | None`).
- **`KMeansDiagnostics`**: dataclass with fields `M_requested,
  M_returned, n_removed_duplicates, dedup_tol, inertia,
  pairwise_min_distance, pairwise_mean_distance, kernel_health`
  (`KernelHealthDiagnostics | None`).
- **`KernelHealthDiagnostics`**: dataclass with fields `d_final,
  total_variance, nystrom_residual, kuu_min_eigenvalue,
  kuu_max_eigenvalue, kuu_condition_number, kuu_n_small_eigenvalues,
  kuu_eig_threshold`. Here `nystrom_residual` is the unnormalised sum
  `tr(Kff - Qff)`. `repr()` prints a one-screen summary.

The init and diagnostic helpers above run in float64 whatever `floatX` is.

## Utils — `ptgp/utils.py`

- **`check_init(fun, theta0, X, y, model=None, extra_vars=None,
  extra_init=None, top_k=10) -> bool`**: evaluates loss + grad at `theta0`
  and returns `True` when both are finite. It logs the loss, whether the
  gradient is finite, the max `|grad|` component (with a warning above
  `_LARGE_GRAD_WARN = 1e4`), and the `top_k` largest `|grad|` components
  with parameter labels. Run before scipy starts.
- **`get_initial_params(model, init="prior_median", rng=None,
  n_median_samples=500) -> dict`**: constrained-space values for all
  free RVs at the chosen init strategy. Used to build numerical proxy
  kernels for `greedy_variance_init`.
- **`save_fit(path, shared_params, shared_extras=(), meta=None)`** /
  **`load_fit(path, shared_params, shared_extras=(), strict=True)`**:
  save trained shared values to `.npz` and load them into freshly built
  shared variables (shapes must match).

`ptgp.idata.to_idata(shared_params, shared_extras=(), *, result=None,
history=None, phase_labels=None, model=None)` packages a run, including a
`tracked_minimize` or `minimize_staged_vfe` history, as an
`xarray.DataTree`. It is an alternative to pickling histories.

## Convenience API — `ptgp/optim/api.py`

Also available at the top level as `pg.fit`, `pg.predict`, and
`pg.FitResult`.

- **`fit(gp, X, y, *, model=None, objective=None, method="L-BFGS-B",
  init="prior_median", init_rng=None, compile_kwargs=None,
  **scipy_kwargs) -> FitResult`**: one-shot training. Uses
  `gp.default_objective` (`marginal_log_likelihood` / `collapsed_elbo` /
  `elbo`) when `objective` is omitted, compiles, minimizes, unpacks, and
  returns `FitResult(result, params, shared_params, shared_extras,
  model)`. Optimizer settings go through scipy kwargs, e.g.
  `options={"maxiter": 2000}`. Wraps `compile_scipy_objective` +
  `scipy.optimize.minimize`; drop down for per-iteration diagnostics,
  frozen variables, or staged training.
- **`predict(gp, X_new, fit_result, *, X_train=None, y_train=None,
  incl_lik=False, compile_kwargs=None) -> (mean, var)`**: compile
  + evaluate. `X_train` / `y_train` required for `Unapproximated`
  and `VFE`; ignored for `SVGP`.

## Precision (`floatX`)

Model graphs and compiled functions follow `pytensor.config.floatX`. Data
passed as float64 is downcast automatically, scipy's `theta` vector stays
float64, and the inducing-init helpers always run in float64. Run under
`pytensor.config.change_flags(floatX="float32")` for float32 training.
