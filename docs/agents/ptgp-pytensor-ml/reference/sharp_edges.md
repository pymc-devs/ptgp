# Sharp edges

Distilled from pytensor-ml's sharp-edges page, keeping only what a GP
training run in ptgp can hit. Each of these trains wrong without erroring,
or errors somewhere far from the cause.

## Raw pytensor.function silently freezes training clocks

A rule's updates dict carries step-counter clocks that schedules and adam's
bias correction read. `compile_train` (which `compile_training_step` uses)
advances every clock the step reads, including one baked into a rate graph
that no rule writes. Plain `pytensor.function` threads none of them, so every
schedule stays pinned at step zero, and under a warmup that is a rate of
exactly zero with parameters that never move. If a custom compile is
unavoidable, use `pytensor_ml.pytensorf.function`, never the plain one.

## Chain position decides what a transform does

```python
chain(clip_by_global_norm(1.0), adam(1e-3))   # bounds the gradient
chain(adam(1e-3), clip_by_global_norm(1.0))   # bounds the step
```

Only the first stops an exploding gradient. Adam normalizes its step to
roughly the learning rate whatever the gradient was, so a clip behind it
almost never fires and the spike still lands in the moment estimates. GP
gradient spikes come from ill-conditioned `Kuu` or a lengthscale near a
cliff, so clip ahead of the rule.

## Clipping cannot rescue a non-finite step

One infinite gradient coordinate makes the global norm infinite, the clip
scale becomes `max_norm / inf = 0`, and the poisoned coordinate becomes
`inf * 0 = NaN`. Every healthy parameter loses its step and one parameter is
destroyed, silently. GP losses go non-finite readily (sigma near zero, a
Cholesky leaving its domain), so wrap the chain in `apply_if_finite`, which
skips the step and keeps the parameters. Clipping bounds a batch that is
merely large. The guard survives one that overflows. They compose and are
not substitutes.

## A chain needs exactly one rule

A chain of only gradient transforms (a lone clip, a lone `trace`) returns
`Gradients`, and compiling that would move parameters uphill. Both ptgp paths
refuse it: `compile_train` raises for a single transform and
`_grouped_updates` raises per group ("returned gradients rather than
steps"). The fix is a missing rule, not a reordering. Two rules in one chain
also raise, since the second would descend along the first one's step.

## Optimizer state is matched by name

Two same-kind rules in one training step derive identical state names, most
visibly two default-namespace adams colliding on `adam/step_count`. In
ptgp's per-group dict this is caught at compile time by
`require_unique_state_names`, and the fix is always a distinct `namespace`
per group, never suppression. The same applies to two `trace` transforms in
one chain.

## A schedule's third argument is an endpoint, not a factor

Optax's `alpha` is a fraction of the initial rate. Here the third positional
argument is the rate the schedule arrives at. `cosine_schedule(3e-4, 10_000,
0.1)` ramps up to 0.1 and nothing warns, because an endpoint above the
start is a legitimate warmup. Schedules also clamp at their endpoint after
`total_steps` rather than decaying forever, so pick `total_steps` to cover
the run, not the half-life.

The first step trains at `schedule(0)` while adam's bias correction uses
`t = 1`.

## reduce_on_plateau counts steps, not epochs

The policy is graph nodes evaluated once per call, and the graph has no
epoch concept, so `patience=10` waits ten steps. A torch config transcribed
directly cuts the rate `steps_per_epoch` times too aggressively, on
per-batch noise, and the defaults compound it: `min_scale=0.0` lets noise
cut the rate to the floor and `cooldown=0` restarts the counter immediately.
Recover epoch cadence with `accumulation_size=steps_per_epoch`, and note it
multiplies the wait. An epoch is arithmetic on steps:
`n_samples // X_batch.shape[0]` is accepted symbolically by every count.

## Optimizer state is per compile

A transform invocation allocates fresh state every call, so two calls to
`compile_training_step` never share momentum, even with the same transform
object. Each staged phase starts with cold moments and a clock at zero,
which is usually right after a freeze boundary. Sharing state across two
compiled functions is pytensor-ml's compile-from-one-updates-dict pattern
and is not exposed through `compile_training_step`.

## Backend notes

Rates are cast to `pytensor.config.floatX`. ptgp runs float64 throughout, so
this matters only if a config sets floatX to float32. Under
`compile_kwargs={"mode": "JAX"}` a shared random generator does not advance
between calls. Training losses in ptgp are deterministic, so training is
unaffected, but do not route `predict_f_samples` through a JAX-compiled
function that is expected to produce fresh draws per call.
