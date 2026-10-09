# Core API

Everything lives in `pytensor_ml.optim`. One type covers the whole system: a
*transform* is a callable `(loss_or_grads_or_updates, parameters) -> Updates`.
Rules, clips, scales, guards, and policies all share this signature, which is
why they compose in any order through `chain`.

## The Updates contract

An `Updates` dict maps each shared variable to its next value. Every
transform reads what it needs as `updates[parameter] - parameter` and writes a
new value back. What that difference means depends on position:

- `Gradients` (what a loss or gradient list seeds): the difference is the
  gradient `g` itself, stored as `p + g`.
- `Steps` (what every rule returns): the difference is the move the rule
  decided on.

A rule refuses `Steps` input (it would negate another rule's step and ascend),
and `compile_train` refuses a `Gradients` result (nothing turned gradients
into a descent step). Exactly one rule per chain.

## ptgp entry point

```python
compile_training_step(objective_fn, gp_model, X_var, y_var, model=...,
                      optimizer=...,        # Transform, or dict[str, Transform]
                      param_groups=...,     # dict[str, list[var]], required iff optimizer is a dict
                      extra_vars=..., extra_init=..., frozen_vars=...,
                      include_prior=True, compile_kwargs=None)
```

`optimizer` defaults to `adam(1e-2)`. With a dict, keys must match
`param_groups` keys exactly, groups must partition the trainable variables
(PyMC value vars plus `extra_vars` entries), and each group's transform must
contain a rule. To freeze a variable use `frozen_vars=`, not an omitted group.

## Rules

Every `learning_rate` accepts a float, a shared scalar (steerable via
`set_value` without recompiling), any scalar graph, or a schedule. Every rule
takes a keyword-only `namespace` that prefixes its state names.

```python
sgd(learning_rate=0.01, momentum=0.0, nesterov=False)
adam(learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, amsgrad=False)
adamw(learning_rate=1e-3, weight_decay=0.01, ..., mask=None)
nadam(learning_rate=2e-3, ...)
adamax(learning_rate=2e-3, ...)
rmsprop(learning_rate=0.01, rho=0.9, momentum=0.0, epsilon=1e-8, centered=False)
rprop(learning_rate=0.01, eta_minus=0.5, eta_plus=1.2, step_min=1e-6, step_max=50.0)
adagrad(learning_rate=0.01, epsilon=1e-8)
adadelta(learning_rate=1.0, rho=0.9, epsilon=1e-8)
lbfgs(learning_rate=1.0, memory_size=10, scale_init_precond=True)
```

`rprop` takes a plain float only (the rate initializes per-parameter step
sizes). Each alias has a matching `*_updates` function
(`adam_updates(loss, params, ...)`) that returns the updates dict directly,
for compiling two functions that share one optimizer state.

## Transforms

```python
chain(*transforms)                 # left to right, each reads the previous output
scale(factor)                      # multiply the step (or gradient) by factor
scale_by_schedule(schedule)        # same, factor read off a training clock
trace(decay=0.9, nesterov=False)   # momentum buffer; heavy-ball ahead of a rule
add_weight_decay(weight_decay=0.01, mask=None)   # coupled L2 ahead of a rule, AdamW behind it
clip_by_global_norm(max_norm=1.0)
clip_by_value(min_value=-1.0, max_value=1.0)
```

Position decides meaning. Ahead of the rule a clip bounds the gradient,
behind it the step. See sharp_edges.md for why only the first stops a spike.

## Guards and policies

```python
apply_if_finite(rule, max_consecutive_skips=5)   # skip the step when anything is non-finite
skip_if(rule, condition)                          # general form; conditions below
nonfinite()                                       # SkipCondition: any non-finite gradient
large_step(max_norm)                              # SkipCondition: step norm exceeds max_norm
reduce_on_plateau(rule, scale, factor=0.1, patience=10, cooldown=0,
                  rtol=1e-4, atol=0.0, min_scale=0.0, accumulation_size=1)
```

`reduce_on_plateau` wraps a rule and cuts a shared `scale` (from
`scalar_state("plateau/scale", fill_value=1.0)`) by `factor` when the loss
stops improving. The rule's learning rate must read that same variable
(`adam(learning_rate=scale * 1e-2)`), or the policy raises at compile because
the cuts would reach nothing. `patience` counts steps, not epochs. After
`max_consecutive_skips` consecutive skips a guard raises instead of skipping
again.

## Schedules

A schedule is `step_count -> rate`, passed as a rule's `learning_rate`. The
third positional argument is always the final rate the schedule arrives at,
never an optax-style fraction of the initial rate.

```python
constant_schedule(lr)
linear_schedule(lr, total_steps, final_learning_rate=0.0, transition_begin=0)
cosine_schedule(lr, total_steps, final_learning_rate=0.0)
exponential_schedule(lr, total_steps, final_learning_rate, transition_begin=0)
polynomial_schedule(lr, total_steps, final_learning_rate=0.0, transition_begin=0, power=1.0)
linear_onecycle_schedule(peak_value, total_steps, pct_start=0.3, pct_final=0.85,
                         div_factor=25.0, final_div_factor=1e4)
step_decay(lr, decay_every=..., decay_factor=0.1, min_learning_rate=0.0, transition_begin=0)
join_schedules(schedules, boundaries)   # boundaries are the step counts where the next phase starts
```

Schedules hold their final value after `total_steps`. Every count
(`total_steps`, `transition_begin`, `patience`, `accumulation_size`,
`boundaries`, clip bounds) accepts a symbolic value, so an epoch under
minibatching is `n_samples // X_batch.shape[0]` written into the count.

## State helpers

```python
scalar_state(name, fill_value=0.0, dtype=None)   # named shared scalar, e.g. a steerable rate
state_for(parameter, slot, fill_value=0.0)       # optimizer-state buffer shaped like parameter
```

`state_for` requires the parameter to have a name. ptgp names every shared
variable after its PyMC value var or `extra_vars` entry, so this only bites
when an `extra_vars` entry was created without `name=`.
