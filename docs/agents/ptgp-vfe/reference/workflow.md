# VFE training workflow — five-tier escalation ladder

Tiers are labelled A–E to avoid clashing with `minimize_staged_vfe`'s
internal `phase1` / `phase2{a,b}` / `phase3` numbering. Start at the
lowest tier that fits your problem; only escalate when the lower tier
fails.

---

## Tier A — exact GP smoke test

**When.** A new dataset, a new kernel, or any time you suspect a
*modelling* bug rather than a sparsification artefact.

**What.** Build `pg.gp.Unapproximated`, fit on a small subset of the
data (`N ≤ 1000`), and confirm the marginal log-likelihood and
predictions look sensible. Use `compile_scipy_objective` with
`marginal_log_likelihood` and L-BFGS-B.

**Pass criteria.** Predictions track the data; learned hyperparameters
are not at the prior boundary; loss decreases monotonically; no
non-finite values; condition number of `K + sigma^2 I` stays bounded.

**Why.** If the exact GP is broken, no sparsification recipe will
rescue it. Most "VFE doesn't work" reports turn out to be priors that
don't match the data scale, a kernel that doesn't fit the structure,
or a likelihood mismatch — all visible at Tier A.

**Failure → escalate.** If Tier A fails, the problem is upstream of
VFE. See [bad_priors](../pitfalls/bad_priors.md),
[non_finite_at_init](../pitfalls/non_finite_at_init.md),
[lengthscale_runaway](../pitfalls/lengthscale_runaway.md). Don't
proceed to Tier B until Tier A on a subsample passes.

---

## Choose M

Before Tier B, pick the number of inducing points `M`.

**Rule.** Run `greedy_variance_init(X, M_max, kernel)` once at an
`M_max` larger than you'll need (e.g. `min(N, 1024)`) using a
prior-median proxy kernel. Plot `trace_curve / total_variance` vs `M`
and read off the **knee**, requiring residual variance below ~1%.

**See.** [reference/choosing_M.md](choosing_M.md).

---

## Tier B — joint VFE with frozen Z

**When.** Tier A passes; you've chosen `M` from the trace-curve knee.

**What.** Run `greedy_variance_init` to get a Z layout, then
`compile_scipy_objective(collapsed_elbo, ..., frozen_vars={Z_var: Z0})`
and minimise with L-BFGS-B. Hyperparameters move; Z stays put.

**Notes.**
- If your kernel uses symbolic transform parameters (e.g. a warp
  applied to lengthscales), evaluate them at concrete numerical
  values before passing the kernel to `greedy_variance_init`. The
  pivoted-Cholesky greedy uses concrete `kernel(X, Z)`, so symbolic
  parameters need to be substituted out first. The general principle:
  whatever kernel you're using to *select* Z must be a numerical
  kernel.
- Use `pg.utils.check_init` before scipy starts to catch non-finite
  losses or huge gradients at the initial theta.

**Pass criteria.** L-BFGS-B converges with `result.success == True`;
`history[-1].excess_fit_per_n > 0` (model fits better than the noise
floor); `nystrom_residual` shrinks to near zero or stabilises.

**Failure → escalate to Tier C** if `nystrom_residual` stays large at
convergence (a sign Z needs to move) — see
[inducing_layout_poor](../pitfalls/inducing_layout_poor.md). Failure
→ escalate **further** to Tier D directly if you see
[sigma_collapse](../pitfalls/sigma_collapse.md) or
[sigma_inflation](../pitfalls/sigma_inflation.md).

---

## Tier C — joint VFE with Z trainable

**When.** Tier B converges but `nystrom_residual` stays large and you
suspect the inducing layout is the bottleneck.

**What.** Same compile call as Tier B but pass `Z_var` via
`extra_vars=[Z_var]` (and remove from `frozen_vars`). Z and
hyperparameters now share the optimizer.

**Notes.** The collapsed bound's gradient w.r.t. Z is cheap, so this
is faster than alternating schemes when it works. The risk:
sigma collapse. Watch `history[i].sigma` — if it drifts toward zero
while `nystrom_residual` is large, the optimizer is silencing the
trace penalty by inflating noise rather than improving Z.

**Failure → escalate to Tier D** when:
- [sigma_collapse](../pitfalls/sigma_collapse.md): sigma falls toward
  zero with ELBO seemingly improving.
- [sigma_inflation](../pitfalls/sigma_inflation.md): sigma grows while
  ELBO plateaus.
- [lbfgsb_abnormal](../pitfalls/lbfgsb_abnormal.md): scipy returns
  `result.status == 2`.
- `check_init` flags the initial gradient as non-finite or huge — see
  [non_finite_at_init](../pitfalls/non_finite_at_init.md),
  [large_grad_at_init](../pitfalls/large_grad_at_init.md).

---

## Tier D — staged VFE

**When.** Tier C fails by sigma collapse, sigma inflation, or
non-convergence; or Tier B reveals layout problems but Tier C is
unstable.

**What.** `pg.optim.minimize_staged_vfe` — a four-phase schedule that
prevents the sigma-collapse failure mode by structurally freezing
sigma during early Z movement. See its docstring (`training.py:728`)
for the schedule details and tunable cycles.

**Pass criteria.** `history[-1]` shows
`excess_fit_per_n > 0`, sigma at a sane value (typically near the
empirical residual std), `nystrom_residual` small. The history's
phase labels (`phase1` / `phase2a_cN` / `phase2b_cN` / `phase3`) help
diagnose *where* failures localise: a healthy run shows ELBO improving
through phase 1, oscillating slightly in phase 2 cycles, then settling
in phase 3.

**Failure → escalate to Tier E** when no schedule recovers a sane
sigma or fit — that's a modelling problem, not an optimisation
problem.

---

## Tier E — modelling change

**When.** All staged-training schedules fail.

**Options.**
- **Different kernel**: try `Matern52` if `ExpQuad` is over-smooth, or
  add structure (additive sums, ICM blocks for multi-output).
- **More M**: re-do the choose-M analysis with the trained kernel —
  the knee may have moved out.
- **Tighter / better-centred priors**: see
  [bad_priors](../pitfalls/bad_priors.md). Prior-predictive checks at
  proposed Z layouts catch most failures here.
- **Different likelihood**: if the residuals are heavy-tailed,
  `Gaussian` is wrong.
- **Different inducing inputs**: separate continuous and categorical
  blocks; see [reference/discrete_inputs.md](discrete_inputs.md).

There is no "next tier" — Tier E means stepping back from
optimisation tactics.
