from collections.abc import Callable
from typing import NamedTuple

import pytensor.tensor as pt


class PredictSpec(NamedTuple):
    """What a model needs to compile its marginal predictions.

    Returned by a model's ``predict_spec`` and consumed by
    :func:`ptgp.optim.compile_predict`, which compiles ``inputs -> outputs``
    and runs ``prepare`` on the caller's arguments before each call. Models
    with extra compiled inputs (e.g. precomputed neighbor indices) or
    host-side input checks express them here, so ``compile_predict`` needs no
    per-model code.

    Attributes
    ----------
    inputs : list of TensorVariable
        Symbolic inputs of the compiled function, in call order.
    outputs : tuple of TensorVariable
        ``(mean, var)``.
    prepare : callable
        Maps the caller's arguments to the compiled function's arguments.
    """

    inputs: list
    outputs: tuple
    prepare: Callable


def identity_prepare(*args):
    """Pass the caller's arguments through unchanged."""
    return args


def training_data_predict_spec(gp_model, X_new, incl_lik, X_train, y_train):
    """``PredictSpec`` for models whose posterior conditions on the training data.

    ``X_train`` and ``y_train`` are embedded in the graph as constants, so the
    compiled function takes ``X_new`` only.
    """
    if X_train is None or y_train is None:
        raise ValueError(
            f"{type(gp_model).__name__} prediction requires X_train and y_train. "
            "The posterior conditions on the training data."
        )
    mean, var = gp_model.predict_marginal(
        X_new,
        pt.as_tensor_variable(X_train),
        pt.as_tensor_variable(y_train),
        incl_lik=incl_lik,
    )
    return PredictSpec([X_new], (mean, var), identity_prepare)
