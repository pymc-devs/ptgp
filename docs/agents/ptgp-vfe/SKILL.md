---
name: ptgp-vfe
description: Diagnose and fix VFE (Titsias collapsed-bound) sparse-GP training failures in ptgp. Use when working with VFE, collapsed ELBO, trace penalty, inducing points, minimize_staged_vfe, sigma collapse, or VFEDiagnostics output.
---

# ptgp-vfe — VFE training diagnostic skill

Routes an AI coding assistant to the right context when diagnosing a Titsias
collapsed-bound (VFE) sparse-GP run in ptgp. Three branches:

## reference/

Background and recipes — read first when the user is *setting up* a run.

- [reference/workflow.md](reference/workflow.md) — five-tier escalation
  ladder (smoke test → frozen-Z → trainable-Z → staged → modelling
  change). Start here for "how should I train this?".
- [reference/interpretation.md](reference/interpretation.md) — what each
  field of `VFEDiagnostics` and `GreedyVarianceDiagnostics` means.
  Healthy ranges, suspicious values, which pitfall to open.
- [reference/choosing_M.md](reference/choosing_M.md) — picking the number
  of inducing points before training.
- [reference/discrete_inputs.md](reference/discrete_inputs.md) — handling
  categorical / integer-coded columns in the kernel and inducing layout.
- [reference/api.md](reference/api.md) — concise call-site reference for
  the ptgp functions the skill references.

## pitfalls/

One file per failure mode. Open the relevant slug when symptoms match.
Each file follows the same schema: Detection / Diagnosis / Fix / See also.

| Slug | Headline symptom |
|---|---|
| [sigma_collapse](pitfalls/sigma_collapse.md) | sigma drops toward 0; ELBO appears to improve but fit is degenerate |
| [sigma_inflation](pitfalls/sigma_inflation.md) | sigma grows during training while ELBO plateaus |
| [kuu_ill_conditioned](pitfalls/kuu_ill_conditioned.md) | Kuu condition number > 1e8 or small eigenvalues |
| [excess_fit_per_n_negative](pitfalls/excess_fit_per_n_negative.md) | model fit is worse than the noise floor |
| [M_too_small](pitfalls/M_too_small.md) | nystrom_residual large at convergence; trace curve hasn't flattened |
| [inducing_layout_poor](pitfalls/inducing_layout_poor.md) | nystrom_residual large at convergence; trace curve has flattened but `d_final` has hot spots |
| [slow_convergence](pitfalls/slow_convergence.md) | L-BFGS-B hits maxiter with monotone-decreasing history |
| [inducing_collapse](pitfalls/inducing_collapse.md) | two or more Z rows are duplicates |
| [categorical_inducing_dim](pitfalls/categorical_inducing_dim.md) | Z values land between integer-coded categories |
| [non_finite_at_init](pitfalls/non_finite_at_init.md) | `check_init` reports NaN/inf loss or grad |
| [large_grad_at_init](pitfalls/large_grad_at_init.md) | `check_init` reports grad norm > 1e4 |
| [lengthscale_runaway](pitfalls/lengthscale_runaway.md) | a lengthscale shrinks toward 0 or grows toward infinity |
| [bad_priors](pitfalls/bad_priors.md) | priors mis-centred relative to data scale; drives Kuu issues |
| [eta_collapse](pitfalls/eta_collapse.md) | kernel amplitude eta drops toward 0 (often paired with sigma_collapse) |
| [lbfgsb_abnormal](pitfalls/lbfgsb_abnormal.md) | scipy returns `result.status == 2` (ABNORMAL) |

## scripts/

CLI tools. Each runs from a saved pickle of training output.

- [scripts/check_inducing.py](scripts/check_inducing.py) — pre-fit health
  check on a `GreedyVarianceDiagnostics` pickle. Prints the `repr` plus
  per-`kuu_*` verdicts, writes the canonical 3-panel inducing PNG.
- [scripts/plot_history.py](scripts/plot_history.py) — read `(history,
  phase_labels)` pickle; write the canonical 6-panel history PNG.
- [scripts/detect_collapse.py](scripts/detect_collapse.py) — read same
  pickle; run every pitfall's detection rule; print a sorted Verdict
  table.
