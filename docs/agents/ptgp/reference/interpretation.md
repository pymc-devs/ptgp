# Interpreting diagnostic fields

Two unrelated diagnostic types to keep straight:

- **`VFEDiagnostics`** (namedtuple): per-iteration training history
  fields. Lives in `ptgp/objectives.py`. One per scipy callback step.
- **`GreedyVarianceDiagnostics`** (dataclass): one-shot inducing-point
  selection diagnostics. Lives in `ptgp/inducing.py`. One total, returned
  alongside the selected Z.

The most common cross-wiring mistake is reading `kuu_*` fields off
`VFEDiagnostics`. They live on `GreedyVarianceDiagnostics`, and on
`KernelHealthDiagnostics` (reached through `.kernel_health` on the
random-subsample and k-means diagnostics, or from
`compute_inducing_diagnostics`). `VFEDiagnostics` exposes
`nystrom_residual`, which is related (it's `tr(Kff - Q) / N` where
`Q = Kuf.T @ inv(Kuu) @ Kuf`) but not interchangeable with Kuu's
eigenstructure. `CollapsedELBOTerms.nystrom_residual` and
`KernelHealthDiagnostics.nystrom_residual` are the same trace without the
division by N.

---

## `VFEDiagnostics`

Source: `ptgp/objectives.py` (`vfe_diagnostics` factory + namedtuple
definition). For an exact GP, `unapproximated_diagnostics` returns
`UnapproximatedDiagnostics` with the same `sigma`, `fit_per_n`,
`excess_fit_per_n`, `frac_*`, and `var_ratio` fields (read them the same
way), plus `mll`, `logdet`, and `logdet_per_n` in place of the
collapsed-bound terms.

| Field | Definition | Healthy | Suspicious | Pitfall |
|---|---|---|---|---|
| `elbo` | `fit + trace_penalty` (the Titsias collapsed ELBO) | Monotone-rising over training, plateaus at convergence | Non-monotone, plateaus too early, or rising while sigma collapses | [sigma_collapse](../pitfalls/sigma_collapse.md), [slow_convergence](../pitfalls/slow_convergence.md) |
| `fit` | `-0.5 (quad + logdet_cov + N log 2π)` — the Gaussian log-density of `y` under the Nyström-approximated covariance | Rises during training | Drops, or rises only because `trace_penalty` is being silenced | [excess_fit_per_n_negative](../pitfalls/excess_fit_per_n_negative.md) |
| `trace_penalty` | `-0.5 / sigma^2 · sum(Kff_diag - Q_diag)` — penalises the Nyström approximation gap | Goes to ~0 at convergence; magnitude shrinks as Z covers the data | Stays large; or shrinks only because sigma is being inflated | [M_too_small](../pitfalls/M_too_small.md), [inducing_layout_poor](../pitfalls/inducing_layout_poor.md), [sigma_inflation](../pitfalls/sigma_inflation.md) |
| `nystrom_residual` | `(sum(Kff_diag - Q_diag)) / N` — same gap as `trace_penalty` numerator, normalised by N and stripped of `sigma` | Goes to ~0 at convergence | Stays large; rises during training | [M_too_small](../pitfalls/M_too_small.md), [inducing_layout_poor](../pitfalls/inducing_layout_poor.md), [lengthscale_runaway](../pitfalls/lengthscale_runaway.md) |
| `sigma` | Likelihood noise (constrained space); the mean of `sigma` when it is heteroskedastic | Stable, near the empirical residual std of a baseline mean predictor | Drifting toward 0 (collapse) or growing toward `std(y)` (inflation) | [sigma_collapse](../pitfalls/sigma_collapse.md), [sigma_inflation](../pitfalls/sigma_inflation.md) |
| `fit_per_n` | `fit / N`, the per-point data fit. Depends on the scale of `y` | Rises during training | Falls below the baseline `-0.5 log(2π Var(y - m(X))) - 0.5`, i.e. `excess_fit_per_n < 0` | [excess_fit_per_n_negative](../pitfalls/excess_fit_per_n_negative.md) |
| `excess_fit_per_n` | `fit_per_n + 0.5 log(2π Var(y - m(X))) + 0.5`: per-point fit relative to a constant-mean Gaussian at the residual variance. Invariant to the scale of `y` | > 0 and rising: the kernel explains structure beyond the mean | ≤ 0: the model fits no better than the mean function with Gaussian noise at the residual variance | [excess_fit_per_n_negative](../pitfalls/excess_fit_per_n_negative.md) |
| `frac_mean`, `frac_signal`, `frac_noise` | Shares of the model-implied variance of `y` from the mean function, the GP prior signal (`mean(diag(K))`), and the noise. Sum to 1; invariant to the mean and scale of `y` | `frac_signal` well above 0 when the data has structure | `frac_signal` near 0 with `frac_noise` near 1: the kernel explains nothing | [sigma_inflation](../pitfalls/sigma_inflation.md), [eta_collapse](../pitfalls/eta_collapse.md) |
| `var_ratio` | `total_var / Var(y)`: model-implied marginal variance over the empirical variance | Within a few-fold of 1. A stationary GP's prior variance need not match the sample variance of one realisation over a finite window, so values like 0.5 or 2 are normal | Off by an order of magnitude or more: the hyperparameters imply the wrong overall scale (check priors on `eta` and `sigma`) | [bad_priors](../pitfalls/bad_priors.md) |

**Ratio rule.** `|trace_penalty| / |elbo|`:
- < 10% — bound is tight; `M` and Z are sufficient.
- 10–50% — usable but loose; consider more `M` or better Z.
- \> 50% — overconfident-looking but the bound's slack is dominated by
  Nyström error. Re-check Z and M.

**Patterns table.**

| Pattern | Interpretation |
|---|---|
| `trace_penalty` dominates `elbo` at init | Z poorly placed; use greedy init or increase M |
| `nystrom_residual` rises during training | Lengthscale shrinking — Q is a worse approximation of Kff. Overfitting / weak priors |
| `fit` improves while `trace_penalty` worsens | Model trading approximation quality for data fit. OK if ELBO still rising; bad if flat (ridge) |
| `trace_penalty` ≈ 0 at convergence | Bound tight — M sufficient, Z covers data |
| `nystrom_residual` still large at convergence | If `trace_curve` hasn't flattened: increase M. If it has: re-init Z with the trained kernel |

---

## `GreedyVarianceDiagnostics`

Source: `ptgp/inducing.py`. Returned by `greedy_variance_init(X, M, kernel)`
alongside the `Points` of selected Z. One snapshot per call; these fields
do not change during training.

| Field | Definition | Healthy | Suspicious | Pitfall |
|---|---|---|---|---|
| `trace_curve` | shape `(M + 1,)`. `trace_curve[m]` = residual unexplained variance after `m` selections, so `trace_curve[0]` is `total_variance` and `trace_curve[-1]` is the residual of the returned Z | Falls steeply early then flattens | Flat from the start (kernel is too short-lengthscale or M is way too small) | [M_too_small](../pitfalls/M_too_small.md) |
| `d_final` | shape `(N,)`, aligned to the rows of `X`. Per-data-point residual conditional variance after all selected points | Concentrated near 0, with a thin tail | Bimodal or uniformly elevated: points poorly covered | [inducing_layout_poor](../pitfalls/inducing_layout_poor.md) |
| `total_variance` | `tr(Kff)` before any selection | — | — | (used as a denominator for the fraction-unexplained curve) |
| `kuu_min_eigenvalue` | smallest eigenvalue of `K(Z, Z) + jitter * I` | > `kuu_eig_threshold` (default 1e-4) | Below threshold → near-singular Kuu | [kuu_ill_conditioned](../pitfalls/kuu_ill_conditioned.md), [bad_priors](../pitfalls/bad_priors.md), [inducing_collapse](../pitfalls/inducing_collapse.md) |
| `kuu_max_eigenvalue` | largest eigenvalue of Kuu | depends on kernel amplitude | (used for the condition number) | — |
| `kuu_condition_number` | `max / min` eigenvalue ratio | < 1e5 | > 1e8 → numerical trouble; > 1e10 → broken | [kuu_ill_conditioned](../pitfalls/kuu_ill_conditioned.md) |
| `kuu_n_small_eigenvalues` | count of eigenvalues below `kuu_eig_threshold` | 0 | > 0 → near-duplicate inducing points or a kernel mismatch | [kuu_ill_conditioned](../pitfalls/kuu_ill_conditioned.md), [inducing_collapse](../pitfalls/inducing_collapse.md) |
| `kuu_eig_threshold` | the threshold used to count small eigenvalues | — | — | (default 1e-4; raise/lower if you want a stricter / looser test) |

**Reading the diagnostic.** `repr(diag)` prints a one-screen summary
with the variance-explained percent and the four eigenvalue stats —
that's the first thing to look at after greedy selection.

---

## `KernelHealthDiagnostics`

Source: `ptgp/inducing.py`. Returned by `compute_inducing_diagnostics(kernel,
X, Z)`, and attached as `.kernel_health` by `random_subsample_init` and
`kmeans_init` when they are given `kernel=`. Use it to check any `Z`,
including one that did not come from greedy selection.

| Field | Definition |
|---|---|
| `d_final` | shape `(N,)`. Per-data-point residual conditional variance given `Z`; read it like `GreedyVarianceDiagnostics.d_final` |
| `total_variance` | `tr(Kff)` |
| `nystrom_residual` | `tr(Kff - Qff)`, unnormalised. Divide by `total_variance` for the fraction unexplained |
| `kuu_*` | Same five eigenvalue fields as on `GreedyVarianceDiagnostics` |
