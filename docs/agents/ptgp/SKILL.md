---
name: ptgp
description: Diagnose and fix Gaussian process fitting problems in ptgp (exact GP, VFE, SVGP). Use when a ptgp fit misbehaves (sigma or eta collapsing or inflating, lengthscales running away, non-finite or huge gradients, L-BFGS-B stopping abnormally) or when working with inducing points, VFEDiagnostics, GreedyVarianceDiagnostics, or minimize_staged_vfe.
---

# ptgp: GP fit diagnosis

Each page under `pitfalls/` describes one problem: how to detect it, what
causes it, and how to fix it. Find the symptom below and open its page.
`applies_to` in each page's frontmatter says which models it concerns:
`all`, the inducing-point methods (`VFE`, `SVGP`), or `VFE` only. The
inducing-point pages so far are written mostly from VFE experience.

When a sparse model misbehaves, fitting `pg.gp.Unapproximated` on a
subsample (N ≤ ~1000) separates problems with the model itself (priors,
kernel, likelihood) from problems with the approximation.

## pitfalls/

| Slug | Applies to | Headline symptom |
|---|---|---|
| [bad_priors](pitfalls/bad_priors.md) | all | priors mis-centred relative to the data scale; drives conditioning issues |
| [lengthscale_runaway](pitfalls/lengthscale_runaway.md) | all | a lengthscale shrinks toward 0 or grows toward infinity |
| [eta_collapse](pitfalls/eta_collapse.md) | all | kernel amplitude eta drops toward 0 (often paired with sigma_inflation) |
| [sigma_collapse](pitfalls/sigma_collapse.md) | all | sigma drops toward 0; train fit improves, held-out fit worsens |
| [sigma_inflation](pitfalls/sigma_inflation.md) | VFE | sigma grows during training while the ELBO plateaus |
| [excess_fit_per_n_negative](pitfalls/excess_fit_per_n_negative.md) | Unapproximated, VFE | the fit is no better than a constant mean with Gaussian noise |
| [non_finite_at_init](pitfalls/non_finite_at_init.md) | all | `check_init` reports NaN/inf loss or grad |
| [large_grad_at_init](pitfalls/large_grad_at_init.md) | all | `check_init` warns that the max `|grad|` component is > 1e4 |
| [lbfgsb_abnormal](pitfalls/lbfgsb_abnormal.md) | all | scipy returns `result.status == 2` (ABNORMAL) |
| [slow_convergence](pitfalls/slow_convergence.md) | all | L-BFGS-B hits maxiter with a still-improving history |
| [M_too_small](pitfalls/M_too_small.md) | VFE, SVGP | Nyström residual large at convergence; the greedy trace curve hasn't flattened |
| [inducing_layout_poor](pitfalls/inducing_layout_poor.md) | VFE, SVGP | Nyström residual large at convergence; trace curve flat but `d_final` has hot spots |
| [inducing_collapse](pitfalls/inducing_collapse.md) | VFE, SVGP | two or more Z rows are duplicates |
| [kuu_ill_conditioned](pitfalls/kuu_ill_conditioned.md) | VFE, SVGP | Kuu condition number > 1e8 or small eigenvalues |
| [categorical_inducing_dim](pitfalls/categorical_inducing_dim.md) | VFE, SVGP | Z values land between integer-coded categories |

## reference/

- [reference/api.md](reference/api.md): call-site reference for the ptgp
  functions the pages use, including how to freeze or train inducing
  points and the staged VFE trainer.
- [reference/interpretation.md](reference/interpretation.md): what each
  field of `VFEDiagnostics`, `UnapproximatedDiagnostics`,
  `GreedyVarianceDiagnostics`, and `KernelHealthDiagnostics` means, with
  healthy ranges and the page to open when a value is suspicious.
- [reference/choosing_M.md](reference/choosing_M.md): picking the number
  of inducing points.
- [reference/discrete_inputs.md](reference/discrete_inputs.md): handling
  categorical / integer-coded columns in the kernel and inducing layout.

## scripts/

CLI tools that check symptoms from saved output. The history scripts need
`VFEDiagnostics` entries, i.e. a history recorded with
`diag_fn = compile_scipy_diagnostics(vfe_diagnostics, ...)` (or from
`minimize_staged_vfe`). A history of plain `CollapsedELBOTerms` lacks the
`sigma` and `excess_fit_per_n` fields they read.

- [scripts/check_inducing.py](scripts/check_inducing.py): inducing-point
  health check on a `GreedyVarianceDiagnostics` or
  `KernelHealthDiagnostics` pickle. Prints the `repr` plus per-`kuu_*`
  verdicts and writes a 3-panel PNG.
- [scripts/plot_history.py](scripts/plot_history.py): reads a `(history,
  phase_labels)` pickle and writes a 6-panel history PNG.
- [scripts/detect_collapse.py](scripts/detect_collapse.py): reads the same
  pickle, runs each page's detection rule, and prints which pages match.
