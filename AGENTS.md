# AGENTS.md

Guidance for AI coding assistants (Claude Code, Codex, Cursor, Aider, …)
working in this repository. Follows the [AGENTS.md](https://agents.md/)
convention.

## Mental model

PTGP is a Gaussian process library whose central bet is: **write naive
linear algebra, let PyTensor's rewrite system pick efficient algorithms
based on declared matrix structure**. Three pieces hold this up:

1. **Naive linalg in source.** Always write `pt.linalg.inv(K)`,
   `pt.linalg.slogdet(K)`, `pt.linalg.solve(K, b)` — never hand-roll a
   Cholesky + solve_triangular chain. The rewrite system picks the
   algorithm.
2. **Assumption annotations.** When a kernel evaluates `K(X, X)`, wrap
   the result with `pta.assume(K, symmetric=True, positive_definite=True)`
   (where `pta` is `pytensor.assumptions`) so the rewrite system can prove
   it's PSD and lower it to Cholesky. Cross-covariance `K(X, Y)` is left
   **unannotated**. This happens automatically in `Kernel.__call__` —
   don't duplicate it in kernel subclasses.
3. **Cubic-op floor.** `tests/test_cubic_floor.py` asserts each model's
   joint (loss + all grads) graph compiles to the *minimum* count of
   O(N³) factorizations: 1 for `Unapproximated`, 2 for `VFE`, 1 for
   `SVGP`. This test is the canary for the whole rewrite story — run it
   after any change to `ptgp/rewrites.py`, `ptgp/objectives.py`, or kernel
   evaluation. When it fails, `scripts/joint_graph_analysis.py` prints the
   full per-model op breakdown for diagnosis.

`ptgp/rewrites.py` and `ptgp/linalg/rewrites.py` register assumption rules
and structural rewrites into PyTensor's global registries at import time
(both are imported for side-effects in `ptgp/__init__.py` —
**don't remove those imports**). `ptgp/rewrites.py` also monkey-patches
`pta.assume` to accept `positive=True` for scalar positivity.

**PyMC is a prior container only.** `pm.Model()` holds priors on
hyperparameters; `model.logp()` is added to the objective for MAP
training. **Never call `pm.sample()`, `pm.find_MAP()`, or any PyMC
inference routine** — PTGP's training paths in `ptgp/optim/` are the
inference layer.

## Running things

All commands assume the project's `ptgp` conda env is active (Python
3.14, PyTensor installed from `pymc-devs/pytensor@main`). The env path
varies by machine — if no `ptgp` env is found, ask the user where their
env lives rather than guessing.

```bash
# tests (pyproject sets -v and --doctest-modules, so docstrings run too)
python -m pytest tests/
python -m pytest tests/test_vfe.py                          # single file
python -m pytest tests/test_vfe.py::test_name               # single test
python -m pytest -k vfe                                     # by keyword
python -m pytest tests/test_cubic_floor.py                  # the rewrite-floor canary

# lint / format (also enforced by pre-commit; line-length 100)
ruff check ptgp/ tests/
ruff format ptgp/ tests/

# diagnostic scripts — not in the test suite, run by hand when needed
python scripts/joint_graph_analysis.py        # op-count tables per model, with/without ptgp rewrites
python scripts/inplace_audit.py               # which linalg ops are inplace and why others aren't

# execute the introduction notebook in place
python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/introduction/introduction.ipynb --ExecutePreprocessor.kernel_name=ptgp

# install the bundled agent-skill docs into a Claude Code skill dir (optional)
python scripts/install_claude_skills.py --project .   # ./.claude/skills/
python scripts/install_claude_skills.py --user        # ~/.claude/skills/
```

The reference usage example for all three models is
[`notebooks/introduction/introduction.ipynb`](notebooks/introduction/introduction.ipynb).

## Where things live

- **Models** — `ptgp/gp/`. `Unapproximated`, `VFE`, `SVGP`. Each exposes
  `predict_marginal` / `predict_joint`; `SVGP` adds `predict_f_samples`
  and `prior_kl`. `VariationalParams` + `init_variational_params(M)` build
  the symbolic `q_mu`, `q_sqrt`, plus the underlying trainable
  `extra_vars` / `extra_init` needed by the compile helpers.
- **Objectives** — `ptgp/objectives.py`. Standalone functions, **not**
  methods on models, each returning a named tuple whose first field is
  the scalar loss: `marginal_log_likelihood` → `MLLTerms`, `elbo` →
  `ELBOTerms`, `collapsed_elbo` → `CollapsedELBOTerms`,
  `fitc_log_marginal_likelihood` → `FITCTerms`, plus `vfe_diagnostics`
  and `dpp_regularizer`.
- **Kernels** — `ptgp/kernels/`. `Kernel` base class; subclasses
  implement `_eval(X, Y)` and (optionally) `diag(X)`. Stationary
  (`ExpQuad`, `Matern12/32/52`), nonstationary (`Gibbs`, `RandomWalk`,
  `WarpedInput`), categorical (`Overlap`, `LowRankCategorical`),
  combination (`SumKernel`, `ProductKernel`, via `+` / `*`). `active_dims`
  selects input columns.
- **Likelihoods** — `ptgp/likelihoods/`. `Gaussian`, `Bernoulli`,
  `Poisson`, `NegativeBinomial`, `StudentT`. Non-Gaussian variants
  implement `variational_expectation` for SVGP.
- **Inducing variables** — `ptgp/inducing.py` (`Points`,
  `random_subsample_init`, `kmeans_init`, `greedy_variance_init`),
  `ptgp/inducing_fourier.py` (`FourierFeatures1D` — 1D Matern VFF with
  `M = 2 * num_frequencies + 1`). Init routines return rich diagnostics
  (`GreedyVarianceDiagnostics`, `KernelHealthDiagnostics`, …).
- **Optim** — `ptgp/optim/`. Two training paths and a prediction
  compiler:
  - `compile_training_step` — Adam/SGD via `pytensor.shared` for SVGP /
    minibatch. Auto-discovers PyMC free RVs as trainable; pass
    `extra_vars` (e.g. `q_mu`, `q_sqrt`, trainable `Z`), optional
    `frozen_vars` for staged training, `param_groups` for per-group
    learning rates. Returns `(train_step, shared_params, shared_extras)`.
  - `compile_scipy_objective` — L-BFGS-B path. Pair with
    `tracked_minimize` (one phase, per-iteration diagnostics) or
    `minimize_staged_vfe` (multi-phase VFE with per-phase diagnostic
    capture). `compile_scipy_diagnostics` builds a sibling callable that
    returns `VFEDiagnostics` for the same parameter vector.
  - `compile_predict` — reads the same `shared_params` produced by
    training, so prediction picks up the trained values automatically
    (no model reconstruction).
- **Conditionals / KL** — `ptgp/conditionals.py`, `ptgp/kl.py`. Whitened
  vs unwhitened SVGP posterior pieces; `gauss_kl` / `gauss_kl_structured`.
- **Linalg** — `ptgp/linalg/`. `LinearOperator` abstraction (matvec /
  solve / logdet ops) + structural rewrites. Powers the Fourier-features
  path.
- **Utils** — `ptgp/utils.py`. `check_init` (NaN/inf + grad-norm sanity
  check at initialisation — call this before training), `get_initial_params`
  (prior-median / prior-draw / unconstrained-zero), `save_fit` / `load_fit`.
- **Rewrites** — `ptgp/rewrites.py` + `ptgp/linalg/rewrites.py`. The
  cubic-floor enablers; see the mental-model section.

`comparison_libraries/` is **gitignored** — it's a local-only directory
where some users keep checked-out reference repos (GPJax, GPflow,
GPyTorch, CoLA, linear_operator, PyTensor, PyMC). GPJax is the primary
reference for correctness; GPflow has the cleanest VFE/SVGP architecture.
If `comparison_libraries/` isn't present on this machine, fetch from the
upstream repos when reference code is needed.

## Conventions

Things that aren't self-evident from reading the code:

- **Action-first names.** `minimize_staged_vfe`, not
  `staged_vfe_minimize`. `compile_scipy_objective`, not
  `objective_scipy_compile`.
- **Kernel scale = `eta`, lengthscale = `ls`.** Kernels are written
  `eta**2 * ExpQuad(ls=ls)` — `eta` is **always squared**. Multiple of
  either get numeric suffixes: `eta1`, `ls2`.
- **Shape-annotate every symbolic var.** Pass `shape=(...)` whenever a
  dimension is known at construction time. Use `None` only for axes that
  genuinely vary across calls. Standard letters:
  - `N` — data points (almost always `None`)
  - `M` — inducing points (concrete int)
  - `D` — input dimension (concrete int)
  - `K` — number of outputs (concrete int)

  ```python
  X    = pt.matrix("X",    shape=(None, D))
  Z    = pt.matrix("Z",    shape=(M, D))
  q_mu = pt.vector("q_mu", shape=(M,))
  ```

  Shape info lets rewrites specialize on known relationships (e.g. gating
  on `M < N`) and catches mismatches at compile time. Even single-axis
  annotations (`shape=(None, 1)`) are worth it.
- **Internal symbolic var names start with `_`.** Anything named inside
  library code (e.g. `pt.matrix("_X")` in `greedy_variance_init`) gets
  the underscore prefix so it's distinguishable from user-named vars in
  error messages and graph dumps.
- **Name vars in complex examples.** When symbolic vars become dict keys
  (`param_groups`, multi-phase training), pass `name="..."` — unnamed
  vars render as opaque addresses in error text via
  `var.name or repr(var)`.
- **Error messages: two short sentences.** First names the problem,
  second names the reason or fix. Example: `"Variables appear in both
  extra_vars and frozen_vars: [...]. They cannot be both trainable and
  frozen."`
- **Tests use float64; GPJax uses float32.** For cross-library
  correctness comparisons, use `atol=1e-5`.
- **Notebooks** are stored as `.ipynb` and edited directly — no Jupytext
  pairing or markdown mirror.

## Stage and scope

Prototype. Keep code and tests brief; the goal is a working prototype,
not completeness. Don't add features, abstractions, or backwards-compat
shims beyond what the task requires.

## Skill docs (`docs/agents/`)

The repo ships a backend-agnostic agent-skill doc at
`docs/agents/ptgp-vfe/` covering VFE training diagnosis (pitfalls,
escalation workflow, interpretation of `VFEDiagnostics` /
`GreedyVarianceDiagnostics`). `scripts/install_claude_skills.py` is a
Claude-specific convenience that symlinks it into a Claude Code skill
directory (`~/.claude/skills/` or a project-local `.claude/skills/`);
other AI tools can read `docs/agents/` directly.

**When you change any of the following, check whether the skill needs
updating:**

| File                  | Skill files to scan                                            |
| --------------------- | -------------------------------------------------------------- |
| `ptgp/objectives.py`  | `reference/api.md`, `reference/interpretation.md`              |
| `ptgp/optim/training.py` | `reference/api.md`, `pitfalls/lbfgsb_abnormal.md`, `pitfalls/slow_convergence.md` |
| `ptgp/inducing.py`    | `reference/interpretation.md`, `pitfalls/inducing_*.md`        |
| `ptgp/utils.py`       | `pitfalls/non_finite_at_init.md`, `pitfalls/large_grad_at_init.md` |

The skill's pitfall pages reference fields on `VFEDiagnostics`,
`CollapsedELBOTerms`, and `GreedyVarianceDiagnostics` by name — any
change to those namedtuples needs a sweep through the skill.

## Commits

- One-sentence subject for small changes. Add a multi-line body for
  multi-file refactors, non-obvious design decisions, or behavior changes
  that need context.
- After finishing a feature or plan, **prompt the user** about staging
  and committing. Don't auto-commit.
- Never add a `Co-Authored-By: …` trailer or any AI-attribution footer.
