"""Optimizers for PyTensor shared variables.

Each optimizer takes a loss expression and a list of shared variables,
and returns an OrderedDict mapping each shared variable to its update
expression. Pass this dict as ``updates`` to ``pytensor.function``.

Ported from ``pymc.variational.updates`` (originally from Lasagne).
"""

from collections import OrderedDict

import numpy as np
import pytensor
import pytensor.tensor as pt

from ptgp.optim.schedules import constant


def _get_grads(loss, params):
    for p in params:
        if not isinstance(p, pytensor.compile.SharedVariable):
            raise ValueError("params must contain shared variables only")
    return pytensor.grad(loss, params)


def _resolve_lr(learning_rate, param_groups, params, t):
    """Resolve ``learning_rate`` into a per-parameter symbolic LR list.

    ``learning_rate`` may be a scalar, a schedule callable, or a dict
    mapping group name to scalar/callable. When a dict is passed,
    ``param_groups`` must map the same names to shared-variable lists
    whose union equals ``params``.
    """
    if isinstance(learning_rate, dict):
        if param_groups is None:
            raise ValueError("param_groups is required when learning_rate is a dict")
        if set(learning_rate) != set(param_groups):
            raise ValueError(
                f"learning_rate keys {sorted(learning_rate)} do not match "
                f"param_groups keys {sorted(param_groups)}"
            )
        group_of = {}
        for name, group in param_groups.items():
            for p in group:
                if id(p) in group_of:
                    raise ValueError(f"parameter {p.name or repr(p)} appears in multiple groups")
                group_of[id(p)] = name
        missing = [p for p in params if id(p) not in group_of]
        if missing:
            names = [p.name or repr(p) for p in missing]
            raise ValueError(f"parameters not in any group: {names}")

        group_lr = {name: _as_schedule(lr)(t) for name, lr in learning_rate.items()}
        return [group_lr[group_of[id(p)]] for p in params]

    lr = _as_schedule(learning_rate)(t)
    return [lr] * len(params)


def _as_schedule(lr):
    """Promote a scalar to ``constant(lr)``; pass callables through."""
    if callable(lr):
        return lr
    return constant(lr)


def sgd(loss, params, learning_rate=1e-2, param_groups=None):
    """Stochastic gradient descent.

    Parameters
    ----------
    learning_rate : float, callable, or dict
        Scalar, a schedule from :mod:`ptgp.optim.schedules`, or a dict
        keyed by group name (matching ``param_groups``) whose values are
        scalars or schedules.
    param_groups : dict[str, list[SharedVariable]], optional
        Required when ``learning_rate`` is a dict. Must partition ``params``.
    """
    grads = _get_grads(loss, params)
    t_prev = pytensor.shared(np.float64(0.0))
    t = t_prev + 1
    lrs = _resolve_lr(learning_rate, param_groups, params, t)

    updates = OrderedDict()
    for param, grad, lr in zip(params, grads, lrs):
        updates[param] = param - (lr * grad).astype(param.dtype)
    updates[t_prev] = t
    return updates


def adam(
    loss,
    params,
    learning_rate=1e-2,
    param_groups=None,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    """Adam optimizer (Kingma & Ba, 2014).

    Parameters
    ----------
    learning_rate : float, callable, or dict
        Scalar, a schedule from :mod:`ptgp.optim.schedules`, or a dict
        keyed by group name (matching ``param_groups``) whose values are
        scalars or schedules.
    param_groups : dict[str, list[SharedVariable]], optional
        Required when ``learning_rate`` is a dict. Must partition ``params``.
    """
    grads = _get_grads(loss, params)
    t_prev = pytensor.shared(np.float64(0.0))
    updates = OrderedDict()

    one = pt.constant(1)
    t = t_prev + 1
    lrs = _resolve_lr(learning_rate, param_groups, params, t)
    bias_correction = pt.sqrt(one - beta2**t) / (one - beta1**t)

    for param, g_t, lr in zip(params, grads, lrs):
        value = param.get_value(borrow=True)
        m_prev = pytensor.shared(
            np.zeros(value.shape, dtype=value.dtype),
            shape=param.type.shape,
        )
        v_prev = pytensor.shared(
            np.zeros(value.shape, dtype=value.dtype),
            shape=param.type.shape,
        )

        m_t = beta1 * m_prev + (one - beta1) * g_t
        v_t = beta2 * v_prev + (one - beta2) * g_t**2
        step = (lr * bias_correction * m_t / (pt.sqrt(v_t) + epsilon)).astype(param.dtype)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates
