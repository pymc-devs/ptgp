# Plan: water-mains example (watermains.py + two notebooks), revision 3

Working document, updated 2026-07-12 after the panel/time redesign. Supersedes the aggregated
walk-forward design; see git history of this file for prior revisions.

## Goal and audience

- MAIN notebook (`notebooks/watermains.ipynb`): for statisticians / modelers / scientists
  evaluating GPs for their own problem. Demonstrates ptgp's capabilities (a showcase), shows GP
  predictions, gives a high-level account of the decision-theoretic framework, and shows results
  validated with realized outcomes. The multi-model sequence exists to show that model building
  is a process and that ptgp lets you build sequences of GPs nimbly. NOT pedagogical data-science
  how-to material.
- SUPPLEMENT (`notebooks/watermains_supplement.ipynb`): for readers who want to dig into methods.
- Companion module `notebooks/watermains.py`: all mechanical code + unit tests
  (`pytest notebooks/watermains.py`). Existing notebooks `decision_theory_water_mains.ipynb`
  (PRIMARY, working tree) and `water_main_decision_theory.ipynb` (DEV) remain READ-ONLY sources;
  the user renames/deletes them at the end.

## Data layout: pipe-year panel

One row per pipe per year s in [2010, t]. Target = breaks in year s (linked MAIN breaks);
exposure = length_km x 1 year; covariates measured at s (age = s - install). Walk-forward
folds t = 2019..2025: train rows s < t, test rows s = t (the "annual plan" replay). 2026 is
partial and unused. History feature per row = empirical prior break rate over [1985, s)
(breaks / km-years), log1p, standardized with training stats: window-invariant and strictly
before the row's year, so no circularity.

X column layout: 0-3 = age, log_size, lon, lat (standardized); 4 = year (standardized);
5 = material code; 6 = zone code; 7 = log exposure; 8 = history feature; 9 = pipe index.
Kernels only ever touch 0-6; the mean function reads 7 (and 8 for M2/M3); the frailty reads 9.

## The model sequence (FINAL narrative, user-defined 2026-07-14)

Stylized, clear logical steps; the supplement is METHODS DEPTH FOR READERS, NOT A PROCESS
DIARY (no development history, no rejected-variant chronicles unless they teach a method).

- Model 1: covariates-only SVGP. Matern52 ARD over [age, log_size, lon, lat] x
  NormalizedLowRankCategorical(material) x NormalizedLowRankCategorical(zone),
  ExposureOffset mean, Poisson, M=1024 greedy-init trained inducing points.
  ADOPTED 2026-07-14: categorical kernels are normalized by their frequency-weighted mean
  diagonal (scale pinned to 1) so eta is identified and eta^2 is THE mean per-point GP prior
  variance; eta ~ HalfNormal(1.5) (HalfNormal(1) binds once eta carries the full scale).
  Known cost ~1.8 ELPD on fold 2025 at 700 steps (optimization geometry, same model class);
  expected to close at the final-pass longer runs.
- Model 2: + break history AND calendar year as LINEAR MEAN COEFFICIENTS ONLY (single
  coefficient each; kernel unchanged). Introduces the R2D2 prior splitting the variance
  budget over THREE components {GP amplitude, history slab, year slab}, MAP-estimated in
  log-variance coordinates IF FEASIBLE (verify allocation moves and fit is healthy);
  if infeasible, R2D2's introduction shifts to Model 3.
  R2 PRIOR (user-locked 2026-07-14): R2 ~ Beta(1, 99). Mean 0.01 = the realized-view anchor
  from Model 1 (var(fm)/(var(fm)+1/mean(y)) ~ 0.010 across folds); a=1 keeps the density
  monotone (no interior peak, reads weakly informative); median 0.007 also covers the
  amplitude-view anchor eta^2/(eta^2+sigma2) ~ 0.005.
- Model 3: the BEST GP model under the R2D2M2 prior family (required until that path is
  exhausted; user expects it to be superior with changes). Adds frailty as a budget
  component; kernel placement of year/hist and other structure free to optimize.
  HARD TARGET: beat tuned gradient boosting on ELPD.

## Evaluation protocol (final, 2026-07-14)

- Panel starts at 1995 for ALL tests (user-approved 2026-07-15, superseding the 1985 start):
  the break layer nominally extends to 1985 but holds 8 records total in 1985-94 vs 330-580
  per 5-year bin after, so effective coverage begins 1995. Discovered via Model 2's linear
  year term fitting the completeness ramp (+41%/yr weighted slope on the 1985 panel,
  +31.6/yr forecast bias); honest 2010-2024 trend is -3.5%/yr. RECORD_START and PANEL_START
  are both 1995 in watermains.py; the history-rate window is [1995, s).
- 7-fold walk-forward (test 2019-2025) KEPT, but run ONCE at full budget for the frozen
  final models (the untouched benchmark).
- CURRENT LADDER STATE (seq95 caches, M=1024, 700 steps, 2026-07-16): M1 -2308.1 / z2 0.99;
  M2 -2270.0 / AUC-PR 0.070 / z2 1.34 (beats M1 on all 7 folds, +38 pooled; phi stable
  [0.59 GP, 0.22 hist, 0.18 year], R2hat 0.007, beta_year ~ 0 on the honest panel);
  HGB bar -2257.9 / z2 2.32; GLM -2360.4. M3 must find ~12 ELPD points. R2 anchors on the
  1995 panel: amplitude 0.0061 +- 0.0002, realized 0.0130 +- 0.0017 (Beta(1,99) still fits).
- Development comparisons use the SCREENING protocol: single fit on fold 2025 (train
  1985-2024), scored as PAIRED per-row ELPD difference vs a cached incumbent with
  SE = sqrt(n var(row diffs)); |diff| > 2 SE decides; 3-fold confirm (2019/2022/2025) when
  needed. HGB bar must be RE-ESTABLISHED on the 1985 panel.
- HGB benchmark (user-locked 2026-07-14): plain HGB, NO trend-offset or other bolt-ons
  (a Poisson-trend-offset variant screened at +8.7 +- 3.6 on fold 2025 but was rejected as
  not-the-benchmark; note it in the supplement discussion only if it earns its place).
  Hyperparameters from hgb_random_grid (80-config random search, seed 7) per fold, validated
  on the last training year. HGB cannot extrapolate calendar year (frozen last histogram
  bin; 2019-trained model predicts a constant ~76 breaks/yr through 2025 vs realized 43-90),
  so it is a fair one-year-ahead benchmark only; say so where the bar is presented.
- CV caches for the adopted setup: watermains_cv_seq95_{t}.pkl (1995 panel: normalized M1,
  M2 R2D2, retuned HGB, GLM). seq85n (1985 panel, normalized) and seq85 (pre-normalization)
  are superseded; older cache families are historical.

## Decision theory (main notebook), validated on the walk-forward

Decisions are computed per fold from that fold's Model 3 fit; the DISPLAY plan is the final
fold ("the 2025 annual plan"). Rules:
- Replace-vs-defer (R&W eq. 2.32/2.33): threshold C_break * E[N over H=30y] > C_replace, with
  the planning rate holding year-t conditions (no 30-year extrapolation of the year covariate;
  say so honestly).
- Newsvendor annual budget B* at critical fractile C_surge/(C_surge+C_idle) = 0.83.

Validation with REALIZED outcomes (user-approved 2026-07-12):
- (a) Newsvendor audit: realized loss L(N_t, B*_t) summed over 2019-2025, compared against
  planning-to-the-mean, planning-to-last-year's-count, and the oracle B=N_t. Reported in $.
- (b) Replacement audit: per-fold flagged list; next-year hit rate on flagged vs unflagged
  pipes pooled across folds; triangular follow-up (2019 plan observed ~7.5y ... 2025 plan ~1y)
  reporting realized break-cost avoided vs capital committed.
- The fixed-budget capture curve (breaks captured @ $5M/yr per policy) remains as the ranking
  validation.

## Main notebook contents (confirmed by user)

Title/thesis; data section with network+breaks map; condensed feature bullet list; walk-forward
harness (brief); M1 (ptgp code visible) + score row; brief model criticism (phi, repeat-breaker
ratio) motivating M2/M3; M2 + row; M3 (inline VI frailty) + row; risk map (showcase);
decision-theory section (eqs, 2025 plan, flagged-pipes map COLORED BY PLANNING YEAR across
folds); newsvendor calibration/audit figure; capture curve; benchmark section (tuned HGB + GLM
rows + honest commentary); takeaway. Markdown cells: no hard line wrapping.

## Twin-model (structured VI) design, added 2026-07-13

Model 3 twin: SVGP-form (NOT Hensman-form; q(u) integrated analytically via predict_marginal).
Guide blocks: q(u_prod) and q(u_year) Gaussian (the SVGP variational approximations, one per
additive component, each with UNIT-amplitude kernel so amplitudes factor out and rescale
analytically, one Cholesky per component per step); q(eps) mean-field frailty; q(phi) Gaussian
in unconstrained space over ONLY {logit R^2, stick-breaking Dirichlet allocation (5 comps:
product GP, year GP, hist main, year main, frailty), c, beta_hist, beta_year}. POINT-optimized
(user decision): Z, lengthscales, and all inner kernel params (W/kappa). Fixed base draws
(S=16, DADVI-style) make the objective deterministic. ELBO = mean_s inner(phi_s)
- KL(q(phi)||p(phi)) - prior_kl terms; priors via transforms (Beta(2,175) on R^2 through logit,
Dirichlet(1) through stick-breaking with log-det-Jacobian, budget-tied Normals on betas).
Warm start from MAP fits. The SUPPLEMENT must include a detailed math section deriving: the
ELBO identity, the structured family q(f,u,eps)=p(f|u)q(u)q(eps) and why the f-KL vanishes,
variational_expectation as the data term, the phi-block reparameterization, the
amplitude-factoring argument (why point lengthscales save S-1 Choleskies per step), and the
twin/guide correspondence (SVGP q(u) as a structured guide block; delta/diag/full ladder).
The math section must ALSO cover DADVI (Giordano, Ingram, Broderick JMLR 2024): fixed base
draws turn the ELBO into a deterministic SAA surrogate (real convergence criteria, exact
reproducibility, linear-response covariance corrections available later); why this setting is
its best-case regime (the only sampled block is ~8-dim, everything else analytic; SAA error
scales with sampled-block dimension); the hybrid nature here (SAA over the phi block, SGD over
data minibatches); and that fixed shared draws made the init-confound comparison free of MC
noise. Related finding to write up honestly: the (R2, stick-breaking, shared-W)
parameterization is badly conditioned for optimization (allocation crawls under both MAP and
VI; uniform-init twin reached only phi=[0.27,0.20,0.20,0.18,0.15] in 800 steps while informed
init sat at [0.65,0.05,0.10,0.05,0.15]); log per-component-variance coordinates are the
natural fix (same posterior, direct gradients).

User directives 2026-07-13 (late): (1) implement a NATURAL-GRADIENT prototype for the SVGP
variational blocks NOW, as part of finishing Model 3 (the example exists to motivate ptgp
development; do NOT defer to lifetracker). Salimbeni-2018 style: NGD steps on each whitened
q(u) block's natural params (host-side updates from dL/dm and dL/dS), Adam on everything
else. (2) The guide's budget block uses LOG PER-COMPONENT-VARIANCE coordinates (R2D2M2 prior
transformed exactly). (3) PRODUCTION initialization = EQUAL allocation, not tilted; dual-init
fold-2019 run must confirm init-independence first. (4) The two-SVGP-objects pattern is
disliked (likelihood passed to each component is awkward); the prototype composes additive
latent-GP components from raw ptgp primitives (kernel + inducing set + whitened q block) with
ONE likelihood at the top and NO SVGP objects; after Model 3 is finished, open up the SVGP
API along these lines (component blocks + single likelihood).

## Supplement contents

Rate-vs-age-by-material plot; material & zone covariance heatmaps; mat/pz value-count tables +
category-consolidation prose; "integrating the predictive changes the action" scatter; feature
histograms; inducing-point diagnostics; deeper EDA (recurrence, age-at-break, data quirks:
age-0 spike, negative ages, spatial outliers); overdispersion/PPC detail; HGB tuning detail;
sensitivity sweeps (costs, quantiles, horizon); frailty posterior diagnostics. Backlink to main.

## watermains.py

Follow ptgp conventions (ruff line 100, numpydoc on classes/module functions/substantive public
methods; no docstrings needed on __init__/__call__/private methods; no >>> doctests; no
module-level side effects; heavy imports function-local). Panel functions to be PROMOTED from
scratchpad/seq_experiment.py: panel row builder, panel design matrix, generalized SVGP builder
(cont dims parameter), panel fold fitting helpers, tuned-HGB and GLM benchmarks. The aggregated
fold_frame/fit_fold_* path is superseded and should be removed once the notebook is migrated.
Unit tests updated accordingly (leakage guard test for the panel builder).

## Final results (2026-07-13, revision 4 executed) [SUPERSEDED by the 2026-07-18 section: predates the 1997 panel and the finalized model set]

Pooled walk-forward table: M1 -2288.9 / 0.926 / 0.060 / 45.0 / z2 1.04; M2 -2242.6 / 0.928 /
0.077 / 71.0 / 1.07; M3 -2234.0 / 0.929 / 0.078 / 65.0 / 1.06; HGB(tuned, plug-in) -2242.3 /
0.928 / 0.065 / 61.0 / 1.91; GLM -2344.5 / 0.917 / 0.056 / 54.0 / 1.97. M3 beats HGB on every
column (user requirement). Rejected variants (supplement material): M=1024 (~1 ELPD point),
1500 steps (ELPD -2226 but worse ranking/capture at 3x cost), linear-only history mean.
Decision audits: newsvendor realized losses $1.52M (B*) vs $2.21M (mean) vs $3.08M (last-year),
saving $686k / $1.56M; replacement rule 16x next-year lift, 2019 plan pre-empted $3.95M over
its follow-up. 2026 deployment plan: 316 mains, $7.5M. Notebook revision 4 fully executed with
remove-input tags on workflow cells; sigma_f = 0.37 on the full panel.

## NGD twin production result (2026-07-14 overnight, batch 4096, 700 steps, checkpointing) [SUPERSEDED: the finalized M3 (2026-07-18) resolves this via option B-style inference on the step-4 architecture]

Pooled: ELPD -2252.7, ROC 0.926, AP 0.064, captured 54.0, z2 1.77, bias +3.5/yr. WORSE than
m3_main_year (-2230.2 / 0.929 / 0.078 / 65.8 / 1.13 / +1.7) on every column and worse than HGB
on ELPD. The INFERENCE worked: fixed-eval curves show textbook convergence (steep descent,
plateau, best within ~1% of final; gamma ramp caused no instability), and the posterior is
highly reproducible across folds (r2 = 0.0044-0.0046; phi ~ [0.36, 0.20, 0.15, 0.05, 0.24];
sigma-shares stable). Fitted W ~ 0.85 is consistent with the unconstrained models' summed
amplitudes (the earlier "implied R2 = 0.011" used realized latent spread, which includes the
categorical kernel scales that sit OUTSIDE the budget; not a prior-squeeze problem). Leading
diagnosis for the predictive drop: the ARCHITECTURE change, moving year out of the product
kernel into an additive 1-D GP removed year x covariate interactions (same class of signal
that made hist-in-kernel worth +14 ELPD and AP 0.071 -> 0.078). Decision pending (user):
(A) keep m3_main_year as the notebook Model 3, twin = supplement centerpiece;
(B) twin v2 = R2D2M2+NGD inference on the m3_main_year architecture (year back in the product
kernel), ~2.5 h run; (C) proceed to integration as-is.

## Before the final pass (user reminder, 2026-07-14)

1. RERUN ALL GPs WITH MORE ITERATIONS: every model in the final comparison (M1, M2, M3/twin)
   refit with a larger step budget (and matched budgets across models for fair rows) before
   the numbers are frozen into the notebook.
2. MORE THOROUGH HGB HYPERPARAMETER SEARCH: RESOLVED 2026-07-17; per-fold 80-config random
   search (hgb_random_grid, validated on the last training year) is now the standard, baked
   into fit_hgb_panel and the ladder runner. (BayesOpt-via-ptgp remains filed in lifetracker
   as a possible future docs example.)
   Item 1 remains OPEN in part: M4 runs at 2500 steps but M1/M2 are still at 700; consider
   matched step budgets before freezing the final table.

## Current state and desktop migration (2026-07-18)

MODEL SET FINALIZED (five rows; rerun everything on the desktop, no result caches carried):
- M1: SVGP, Matern52(age, size, lon, lat) x NormalizedLowRankCategorical(mat) x NLRC(zone),
  eta ~ HalfNormal(1.5), exposure offset. Normalized cat kernels pin the frequency-weighted
  mean diagonal to 1, so eta is the identified per-point prior scale (0.99-1.08 across folds).
- M2: M1 kernel + linear mean (c, beta_hist, beta_year), R2D2 budget K=3, R2 ~ Beta(1, 99),
  MAP in log-variance coordinates.
- M3 FINAL (called M4 during development): additive latent
  f = c + sqrt(v1) f_base(cont | mat, zone) + sqrt(v2) f_trend(year; ExpQuad)
      + sqrt(v3) f_hist(hist, age | mat) + kappa_t (iid year, v4) + eps_i (frailty, v5),
  R2D2 budget K=5, Beta(1, 99). Inference: twin composition; three whitened unit-amplitude
  GP blocks (M=1024 base / 28-year grid trend / M=256 hist) with full-cov Gaussian q and
  natural-gradient updates (PD backtracking); mean-field frailty and year blocks with
  elementwise NGD; q(log v_1..5, c) = 6-dim diagonal Gaussian with 16 fixed base draws
  (DADVI-style) through the exact R2D2 prior; point/MAP under Adam: all lengthscales,
  W/kappa, both Z sets. Two-phase Adam (1e-2 to 2e-3 at step 700), NGD gamma ramp to 0.02
  then hyperbolic decay after step 400, fixed-eval checkpointing (32768 rows, every 50),
  restore-on-divergence guard. 2500 steps/fold.
- Benchmarks: HGB (per-fold 80-config random search) and Poisson GLM.

PANEL: 1997 start (RECORD_START = PANEL_START = 1997.0). Coverage: 8 records total in
1985-94, one each in 1995 and 1996, 99-125/yr from 1997 on. Both coverage artifacts
(beta_year ramp on the 1985 panel; kappa_1995/96 = -2.4 on the 1995 panel) are documented
material for the notebook's model-criticism thread.

LAPTOP RESULTS TO REPRODUCE (references, not carried): seq97 pooled 7-fold:
M1 -2304.1 / 0.924 / 0.054 / 42.0 / z2 0.99 / +12.9; M2 -2276.3 / 0.926 / 0.060 / 56.1 /
1.13 / +6.1; HGB -2246.5 / 0.928 / 0.066 / 63.0 / 1.56 / +3.8; GLM -2351.4 / 0.916 / 0.054 /
59.0 / 1.60 / +5.8. M3-final fold-2025 screen: ties its MAP variant (-1.3 +- 2.9),
+19.1 +- 7.5 vs M2, +6.0 +- 5.5 vs HGB, z2 0.48, predicted 74 vs 78; posterior budget
phi = [0.19 +- 0.02 base, 0.22 +- 0.07 trend, 0.31 +- 0.04 hist, 0.10 +- 0.04 year-iid,
0.19 +- 0.02 frailty], R2 0.0048 +- 0.0005, sigma_y 0.28 +- 0.06. CV fold 2025 at 2500
steps confirmed: phi = [0.194, 0.217, 0.309, 0.099, 0.181], m4-hgb +7.2 +- 5.4.

HOW TO RERUN (desktop): conda env create -f conda_envs/environment.yaml; pip install -e .
with that env; pytest notebooks/watermains.py -q (11 tests). Then
python notebooks/dev/final_ladder_seq97.py   (M1/M2/HGB/GLM walk-forward; caches
  notebooks/watermains_cv_seq97_{t}.pkl per fold; ~2h laptop-scale)
python notebooks/dev/m4_cv.py                (M3-final 7-fold at 2500 steps; caches
  notebooks/watermains_cv_m4_{t}.pkl; ~5h laptop-scale; must run AFTER the ladder, it
  reads the seq97 caches for paired comparisons; resumable per fold via the caches)
First run re-downloads the Kitchener data into notebooks/watermains_cache.pkl. All pickles
are gitignored; the dev scripts derive paths from their own location.

NEXT (superseded by the 2026-07-20 section below): (1) both CVs on the desktop; (2) freeze the
pooled five-row table; (3) notebook:
refresh hidden cells to the seq97/m4 caches, add the Model 3 section, the kappa_t
anomaly-detection figure, then the decision-theory sections; (4) supplement; (5) M3 tweaks
only if the full CV results warrant them.

## Desktop CV complete (2026-07-20)

Both walk-forward CVs reran on the Ubuntu desktop with ALL THREE GP MODELS AT 3000 STEPS
(user-set, superseding the 700-step M1/M2 and 2500-step M3 budgets). The dev scripts now run
folds parallel-by-fold via a WM_FOLDS subset selector (2 folds x 6 MKL threads on the 12-core
desktop); ladder ~2.4h, M3 ~1.7h wall-clock.

FROZEN POOLED 7-fold (1997 panel, M=1024, 3000 steps): ELPD / ROC / AP / cap@5M / z2 / bias
M1  -2310.1 / 0.925 / 0.053 / 49.0 / 0.86 / 16.4
M2  -2260.1 / 0.928 / 0.070 / 53.0 / 1.07 /  9.1
HGB -2244.6 / 0.927 / 0.068 / 77.0 / 2.08 /  3.3
GLM -2351.4 / 0.916 / 0.054 / 59.0 / 1.62 /  5.8
M3  -2232.4 / 0.929 / 0.074 / 54.8 / 1.34 /  9.2
M3 beats tuned HGB on ELPD (+12.2 +- 9.4 pooled, ~1.3 SE, inside 2 SE) and on ROC/AP/z2; the
hard target is met, though not by a >2 SE ELPD margin. m3-m2 +27.7 +- 8.6. GLM reproduced the
laptop to the decimal (-2351.4), confirming the pipeline is faithful across machines. M3
posterior budget phi ~ [0.20 base, 0.23 trend, 0.33 hist, 0.07 year, 0.16 frailty],
sigma_y ~ 0.25, R2 ~ 0.005.

BETA_YEAR SIGN CHANGE (supersedes "beta_year ~ 0 on the honest panel" in the earlier sections):
at 3000 steps M2's year coefficient is CONSISTENTLY NEGATIVE, beta_year ~ -0.09 (range -0.04 to
-0.14 across folds), not the near-zero value the 700-step fits showed. The longer optimization
lets the single linear year term pick up a mild declining calendar drift, coherent with the
honest 2010-2024 downtrend. M2 allocation 61/20/18 (GP/hist/year), R2hat ~ 0.008,
beta_hist ~ 0.21. Interpret this when the Model 3 / model-criticism narrative is written.

NOTEBOOK: watermains.ipynb rewired to the seq97 caches (run_model reads them, no refit) and
executed top-to-bottom through Model 2 (0 errors); the M1/M2/HGB/GLM tables and the M2
commentary are refreshed to the 3000-step numbers. Still to author: the Model 3 section, the
kappa_t anomaly figure, and the decision-theory sections. Minor: cell-18 R2 anchor figures
(realized ~1.4%, amplitude ~0.7%) not yet synced.

HANDOFF: the 14 seq97/m4 caches are committed on this branch (targeted .gitignore exception) so a
second machine continues without rerunning the multi-hour CV. Drop them before merging to main
(regenerable; the docs notebook ships with baked-in outputs). The raw data cache
(watermains_cache.pkl) is NOT committed; first run re-downloads the Kitchener data.

NEXT: (1) author the Model 3 + decision-theory sections against the m4 caches; (2) interpret the
negative beta_year; (3) sync the cell-18 figures; (4) supplement; (5) drop the handoff caches
before merge.

DEFERRED, OUT OF SCOPE for this example (filed in lifetracker projects/ptgp-future.md):
HSGP component blocks; the Titsias 2025 tighter SVGP/VFE bounds; the component-VI block
protocol / API work; Vecchia; CG/SLQ solver backends.

## Verification

1. `pytest notebooks/watermains.py` green; ruff clean; import cheap.
2. Main notebook executes top-to-bottom on the ptgp kernel with fold caches
   (watermains_cv_seq_*.pkl, gitignored); committed WITH outputs (docs execution is OFF).
3. Sequential table reproduces the experiment: each model row improves on the last; benchmarks
   present with plug-in annotation.
4. Decision audits report realized dollars; newsvendor B* never short in observed years unless
   the data say otherwise.
5. Both notebooks in the docs gallery; relative links resolve.
