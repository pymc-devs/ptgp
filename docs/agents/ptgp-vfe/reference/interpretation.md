# Interpreting VFE diagnostic fields

Two unrelated namedtuples to keep straight:

- **`VFEDiagnostics`** — per-iteration training history fields. Lives in
  `ptgp/objectives.py`. One per scipy callback step.
- **`GreedyVarianceDiagnostics`** — one-shot inducing-point selection
  diagnostics. Lives in `ptgp/inducing.py`. One total, returned alongside
  the selected Z.

The most common cross-wiring mistake is reading `kuu_*` fields off
`VFEDiagnostics`. They live on `GreedyVarianceDiagnostics`. `VFEDiagnostics`
exposes `nystrom_residual`, which is related (it's `tr(Kff - Q) / N` where
`Q = Kuf.T @ inv(Kuu) @ Kuf`) but not interchangeable with Kuu's
eigenstructure.

---

## `VFEDiagnostics`

Source: `ptgp/objectives.py` (`vfe_diagnostics` factory + namedtuple
definition).

| Field | Definition | Healthy | Suspicious | Pitfall |
|---|---|---|---|---|
| `elbo` | `fit + trace_penalty` (the Titsias collapsed ELBO) | Monotone-rising over training, plateaus at convergence | Non-monotone, plateaus too early, or rising while sigma collapses | [sigma_collapse](../pitfalls/sigma_collapse.md), [slow_convergence](../pitfalls/slow_convergence.md) |
| `fit` | `-0.5 (quad + logdet_cov + N log 2π)` — the Gaussian log-density of `y` under the Nyström-approximated covariance | Rises during training | Drops, or rises only because `trace_penalty` is being silenced | [excess_fit_per_n_negative](../pitfalls/excess_fit_per_n_negative.md) |
| `trace_penalty` | `-0.5 / sigma^2 · sum(Kff_diag - Q_diag)` — penalises the Nyström approximation gap | Goes to ~0 at convergence; magnitude shrinks as Z covers the data | Stays large; or shrinks only because sigma is being inflated | [M_too_small](../pitfalls/M_too_small.md), [inducing_layout_poor](../pitfalls/inducing_layout_poor.md), [sigma_inflation](../pitfalls/sigma_inflation.md) |
| `nystrom_residual` | `(sum(Kff_diag - Q_diag)) / N` — same gap as `trace_penalty` numerator, normalised by N and stripped of `sigma` | Goes to ~0 at convergence | Stays large; rises during training | [M_too_small](../pitfalls/M_too_small.md), [inducing_layout_poor](../pitfalls/inducing_layout_poor.md), [lengthscale_runaway](../pitfalls/lengthscale_runaway.md) |
| `sigma` | Likelihood noise (constrained space) | Stable, near the empirical residual std of a baseline mean predictor | Drifting toward 0 (collapse) or growing toward `std(y)` (inflation) | [sigma_collapse](../pitfalls/sigma_collapse.md), [sigma_inflation](../pitfalls/sigma_inflation.md) |
| `fit_per_n` | `fit / N` — scale-invariant data fit | Rises during training | Drops below `-0.5 log(2π σ²)` (the noise-floor fit) | [excess_fit_per_n_negative](../pitfalls/excess_fit_per_n_negative.md) |
| `excess_fit_per_n` | `fit_per_n + 0.5 log(2π σ²)` — fit relative to the noise floor | > 0 (model beats noise); rises during training | ≤ 0 (model fits no better than predicting `y_mean ± sigma`) | [excess_fit_per_n_negative](../pitfalls/excess_fit_per_n_negative.md) |

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
alongside the `Points` of selected Z. One snapshot per call — these fields
do not change during VFE training.

| Field | Definition | Healthy | Suspicious | Pitfall |
|---|---|---|---|---|
| `trace_curve` | shape `(M,)`. `trace_curve[m]` = residual unexplained variance after `m` selections | Falls steeply early then flattens | Flat from the start (kernel is too short-lengthscale or M is way too small) | [M_too_small](../pitfalls/M_too_small.md) |
| `d_final` | shape `(N,)`. Per-data-point residual conditional variance after all M selections | Concentrated near 0, with a thin tail | Bimodal or uniformly elevated — points poorly covered | [inducing_layout_poor](../pitfalls/inducing_layout_poor.md) |
| `total_variance` | `tr(Kff)` before any selection | — | — | (used as a denominator for the fraction-unexplained curve) |
| `kuu_min_eigenvalue` | smallest eigenvalue of `K(Z, Z) + jitter * I` | > `kuu_eig_threshold` (default 1e-4) | Below threshold → near-singular Kuu | [kuu_ill_conditioned](../pitfalls/kuu_ill_conditioned.md), [bad_priors](../pitfalls/bad_priors.md), [inducing_collapse](../pitfalls/inducing_collapse.md) |
| `kuu_max_eigenvalue` | largest eigenvalue of Kuu | depends on kernel amplitude | (used for the condition number) | — |
| `kuu_condition_number` | `max / min` eigenvalue ratio | < 1e5 | > 1e8 → numerical trouble; > 1e10 → broken | [kuu_ill_conditioned](../pitfalls/kuu_ill_conditioned.md) |
| `kuu_n_small_eigenvalues` | count of eigenvalues below `kuu_eig_threshold` | 0 | > 0 → near-duplicate inducing points or a kernel mismatch | [kuu_ill_conditioned](../pitfalls/kuu_ill_conditioned.md), [inducing_collapse](../pitfalls/inducing_collapse.md) |
| `kuu_eig_threshold` | the threshold used to count small eigenvalues | — | — | (default 1e-4; raise/lower if you want a stricter / looser test) |

**Reading the diagnostic.** `repr(diag)` prints a one-screen summary
with the variance-explained percent and the four eigenvalue stats —
that's the first thing to look at after greedy selection.
