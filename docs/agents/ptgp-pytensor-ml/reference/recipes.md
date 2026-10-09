# Recipes

All recipes assume the standard ptgp setup: a `pm.Model()` holding priors, a
GP model, and symbolic `X_var` / `y_var`. Only the `optimizer=` and
`param_groups=` arguments change.

## SVGP minibatch training with a decaying rate

The default for stochastic training. The schedule runs over the planned step
budget and lands at a tenth of the starting rate.

```python
from pytensor_ml.optim import adam, exponential_schedule

n_steps = 2000
train_step, shared_params, shared_extras = pg.optim.compile_training_step(
    lambda gp, X, y: pg.objectives.elbo(gp, X, y, n_data=N).elbo,
    svgp, X_var, y_var, model=model,
    extra_vars=svgp.extra_vars,
    extra_init=svgp.extra_init,
    optimizer=adam(exponential_schedule(1e-2, total_steps=n_steps, final_learning_rate=1e-3)),
)
```

## Fast variational parameters, slow hyperparameters

The two-rate split from the natural-gradient literature, approximated with
two adams. Distinct namespaces are required, or the compile raises on the
`adam/step_count` name collision.

```python
ls_vv = model.rvs_to_values[model["ls"]]
eta_vv = model.rvs_to_values[model["eta"]]
sigma_vv = model.rvs_to_values[model["sigma"]]

train_step, shared_params, shared_extras = pg.optim.compile_training_step(
    objective, svgp, X_var, y_var, model=model,
    extra_vars=svgp.extra_vars,
    extra_init=svgp.extra_init,
    optimizer={"hyper": adam(1e-2, namespace="hyper"),
               "variational": adam(5e-2, namespace="variational")},
    param_groups={"hyper": [ls_vv, eta_vv, sigma_vv],
                  "variational": list(svgp.extra_vars)},
)
```

## Guarded training that survives a bad batch

`clip_by_global_norm` ahead of the rule bounds a large gradient before it
reaches the moment estimates. `apply_if_finite` around the whole chain skips
the step entirely when a gradient overflows, and raises after five
consecutive skips instead of looping forever. Use both, because they handle
different failures.

```python
from pytensor_ml.optim import adam, apply_if_finite, chain, clip_by_global_norm

optimizer = apply_if_finite(chain(clip_by_global_norm(10.0), adam(1e-2)))
```

Pick `max_norm` from `check_init`'s reported gradient norm at initialization,
roughly an order of magnitude above the healthy value, so the clip fires on
spikes and not on ordinary steps.

## Warmup

There is no warmup helper. A schedule whose endpoint is above its start is
one, and `join_schedules` attaches the decay phase. Boundaries are the step
counts where the next schedule takes over.

```python
from pytensor_ml.optim import adam, cosine_schedule, join_schedules, linear_schedule

warmup = linear_schedule(1e-4, total_steps=100, final_learning_rate=1e-2)
decay = cosine_schedule(1e-2, total_steps=1900, final_learning_rate=1e-3)
optimizer = adam(join_schedules([warmup, decay], boundaries=[100]))
```

## Cut the rate on a plateau, judged per epoch

`patience` counts steps. To decide on epoch-mean loss rather than per-batch
noise, set `accumulation_size` to the steps per epoch, and remember it
multiplies the wait (`patience=10` with a window of 20 cuts after 200 steps
of no improvement). Set `min_scale` so noise cannot cut the rate to zero.

The rule's learning rate must be built from the same shared `scale` variable
the policy cuts, or `reduce_on_plateau` raises at compile ("no parameter's
step reads it").

```python
from pytensor_ml.optim import adam, reduce_on_plateau, scalar_state

steps_per_epoch = N // batch_size
scale = scalar_state("plateau/scale", fill_value=1.0)
optimizer = reduce_on_plateau(adam(learning_rate=scale * 1e-2),
                              scale,
                              factor=0.3,
                              patience=10 * steps_per_epoch,
                              accumulation_size=steps_per_epoch,
                              cooldown=2 * steps_per_epoch,
                              min_scale=0.01)
```

## Steer the rate between phases without recompiling

A shared scalar rate is read at every step, so Python can change it mid-run.
This replaces recompilation when the only thing a phase changes is the rate.

```python
from pytensor_ml.optim import adam, scalar_state

rate = scalar_state("rate", fill_value=1e-2)
train_step, *_ = pg.optim.compile_training_step(..., optimizer=adam(rate))

for _ in range(1000):
    train_step(X, y)
rate.set_value(np.asarray(1e-3))
for _ in range(1000):
    train_step(X, y)
```

## Staged phases that freeze and release variables

Freezing changes the graph, so each phase is its own compile, with
`frozen_vars` pinning what that phase holds still. Optimizer state does not
carry across compiles: each phase starts with cold moments and a clock at
zero, which is usually right after a freeze boundary. Carry the parameter
state over by copying shared values, as `tests/optim/test_frozen_vars.py`
does.

```python
step1, shared1, extras1 = pg.optim.compile_training_step(
    objective, vfe, X_var, y_var, model=model,
    frozen_vars={Z_var: Z_init},
    optimizer=adam(1e-2),
)
# ... train phase 1, then rebuild with Z trainable and phase-1 values as init
```

For the full staged-VFE problem (sigma collapse, inducing point placement),
use `minimize_staged_vfe` on the scipy path instead and see the ptgp-vfe
skill.
