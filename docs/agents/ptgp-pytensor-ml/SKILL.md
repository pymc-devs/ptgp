---
name: ptgp-pytensor-ml
description: Use pytensor-ml optimizers correctly in ptgp's gradient training path. Use when working with compile_training_step, optimizer transforms, chains, learning-rate schedules, per-group optimizers, param_groups, clipping, apply_if_finite, or reduce_on_plateau.
---

# ptgp-pytensor-ml -- optimizer transforms in ptgp

ptgp's gradient training path (`compile_training_step` in
`ptgp/optim/training.py`) takes its optimizer from pytensor-ml: the
`optimizer=` argument is a configured `pytensor_ml.optim` transform, and
compilation goes through `pytensor_ml.optim.compile_train`. The scipy
L-BFGS-B path (`compile_scipy_objective`, `tracked_minimize`,
`minimize_staged_vfe`, `fit`) does not touch pytensor-ml. pytensor-ml ships an
`lbfgs` rule, but it is a stochastic per-step rule, not a replacement for
scipy's L-BFGS-B with line search. When a task says "L-BFGS", it means the
scipy path.

Compile training steps through `compile_training_step`, never by handing a
rule's updates dict to raw `pytensor.function`. Rules carry training clocks
(step counters read by schedules and by adam's bias correction), and
`compile_train` advances any clock that something reads but no rule writes.
Under raw `pytensor.function` such a clock stays frozen at step zero, which
under a warmup schedule is a learning rate of exactly zero, and nothing
raises.

## reference/

- [reference/api.md](reference/api.md) -- the core API: rules, transforms,
  `chain`, schedules, guards, policies, with exact signatures and the
  Updates/Gradients/Steps contract. Open when writing or reviewing an
  `optimizer=` argument.
- [reference/recipes.md](reference/recipes.md) -- copy-paste recipes for the
  common ptgp training setups: SVGP minibatch, per-group rates, guarded
  training, warmup, plateau decay, staged phases.
- [reference/gp_training.md](reference/gp_training.md) -- which optimizer and
  learning rate for which GP training problem, with the reasoning and the
  sources. Open when choosing or debugging a training configuration.
- [reference/sharp_edges.md](reference/sharp_edges.md) -- the pytensor-ml
  behaviors that silently differ from torch/optax habits, cut down to the
  ones a GP training run can hit. Open when a run trains wrong without
  erroring, or when porting a config from another framework.
