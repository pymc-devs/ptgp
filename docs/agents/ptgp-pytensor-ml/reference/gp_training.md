# Optimizers and learning rates for GP training

GP hyperparameter optimization is not deep learning. The trainable set is
tens of scalars, not millions of weights, the objective is deterministic
except under minibatching, and every positive parameter is trained in
unconstrained log space through PyMC's transforms.

## Which path for which problem

Full-batch deterministic objectives (exact MLL, collapsed VFE) belong on the
scipy L-BFGS-B path. Quasi-Newton methods with line search exploit the
smooth, low-dimensional surface and converge in tens of iterations where adam
takes thousands. GPy's default optimizer is L-BFGS-B and GPflow's `Scipy`
optimizer wraps the same method for its non-stochastic models. In ptgp this
is `fit` / `compile_scipy_objective` / `minimize_staged_vfe`.

The pytensor-ml gradient path earns its place when the objective is
stochastic (SVGP minibatching, where line search is meaningless) or when
training must interleave with Python-side control the scipy callback cannot
express. Minibatch ELBO training for GPs is the construction of Hensman,
Fusi, and Lawrence (2013), "Gaussian Processes for Big Data".

## Learning rates in log space

Adam's step size is approximately its learning rate per coordinate,
independent of the gradient's magnitude. For a parameter living in log space,
a rate of 1e-2 means roughly a 1 percent multiplicative change per step, so
2000 steps can move a lengthscale by a factor of e^20 if the gradient keeps
pointing one way. The deep-learning default of 1e-3 is an order of magnitude
too timid for a dozen log-scale hyperparameters. It costs thousands of extra
steps and hides misspecification behind "still converging". ptgp's default of
`adam(1e-2)` reflects this. GPyTorch's exact-GP tutorial goes further and
uses adam at 0.1, which works there because the parameter count is tiny and
each step re-evaluates the full-batch loss.

Start at 1e-2. Go up to 5e-2 when the loss curve is smooth and progress is
slow. Come down to 1e-3 only when the loss oscillates and the oscillation
survives a larger batch.

## Variational parameters want a faster rate than hyperparameters

For fixed hyperparameters and a Gaussian likelihood, the optimal `q_mu` /
`q_sqrt` have a closed form, and the natural-gradient step with unit step
size jumps straight to it. Salimbeni, Eleftheriadis, and Hensman (2018),
"Natural Gradients in Practice", show that natural gradient descent on the
variational parameters plus adam on the hyperparameters beats adam on
everything, and GPflow's SVGP examples pair `NaturalGradient(gamma=0.1)` with
adam on that advice. pytensor-ml has no natural-gradient rule, so the
available approximation is the two-group recipe: adam on the variational
group at 3x to 10x the hyperparameter rate. The whitened parametrization in
`ptgp/conditionals.py` is what keeps the faster rate stable, since it
decorrelates `q_mu` from the kernel hyperparameters.

## Parameter-specific behavior

- Noise sigma learns fastest and is the usual failure point: it inflates
  to silence the VFE trace penalty or collapses toward zero under a flexible
  kernel. A lower rate on a "noise" group, or `frozen_vars` during an early
  phase, is the gradient-path analog of `minimize_staged_vfe`'s phase 1. The
  ptgp-vfe skill's sigma_collapse and sigma_inflation pitfalls carry the
  diagnostics.
- Inducing points Z get weak, local gradients. Initialization
  (`kmeans_init`, `greedy_variance_init`) matters more than the optimizer,
  and letting Z move before the hyperparameters settle lets it chase a
  transient lengthscale. Freeze Z first or give it a slower group.
- Lengthscales under a MAP objective are already regularized by their
  PyMC prior. Do not add `adamw` or `add_weight_decay`, because decay in
  unconstrained space is an extra zero-centered prior nobody declared, and
  `include_prior=True` is the regularizer this library intends.

## Schedules

A decaying rate buys late-stage stability under minibatch noise. The default
shape is exponential or cosine from 1e-2 down to about a tenth of that over
the planned budget, which matches what the SVGP notebook in
`notebooks/approximations/` ships. Adam's gradient normalization makes
training insensitive to the exact shape, so do not spend tuning effort there.

Warmup is rarely needed: there is no batch-norm warm start and the parameter
count is small. The exception is a large gradient norm at initialization,
which `check_init` reports. Fix the initialization or the priors first, and
reach for a 100-step linear warmup only when the large norm is genuinely
transient.

`reduce_on_plateau` is the alternative to a fixed budget when run length is
unknown. Judge on epoch means (`accumulation_size`), set `min_scale` above
zero, and remember it cannot undo a cut. A rate driven down by a noisy
stretch stays down.

## Minibatching

The scaled ELBO (`elbo(..., n_data=N)`) is unbiased at any batch size, and
gradient noise shrinks as 1/sqrt(batch size). Batches of 256 to 1024 are the
usual range. If loss oscillation persists after halving the rate, double the
batch before touching anything else, since that cleanly separates noise from
instability.
