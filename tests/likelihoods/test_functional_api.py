"""The likelihood works in a purely functional way: a likelihood node carries
its Op, design-matrix input, and parameters, so it can be used without the
stateful holder at all. Everything dispatches through ``owner.op``."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pytensor.graph.fg import FunctionGraph

from ptgp.likelihoods import (
    Gaussian,
    op_of,
    param,
    predict_mean_and_var,
    variational_expectation,
    at,
)
from ptgp.likelihoods.gaussian import GaussianOp
from pytensor.graph.replace import clone_replace


def _eval(*tensors):
    f = pytensor.function([], list(tensors) if len(tensors) > 1 else tensors[0])
    return f()


def test_likelihood_survives_clone_and_reroots_on_the_clone():
    X = pt.matrix("X", shape=(None, 1))
    lik = Gaussian(0.1 + 0.05 * X[:, 0] ** 2, x=X)
    assert isinstance(lik.owner.op, GaussianOp)
    X_new = pt.matrix("X_new", shape=(None, 1))
    X_new_test = np.array([[0.0], [1.0], [2.0]])
    mu = pt.as_tensor_variable(np.zeros(3))
    var = pt.as_tensor_variable(np.ones(3))

    _, orig_var = lik.at(X_new).predict_mean_and_var(mu, var)
    orig_eval = orig_var.eval({X_new: X_new_test})

    [cloned_lik] = clone_replace([lik])
    assert cloned_lik is not lik
    assert isinstance(cloned_lik.owner.op, GaussianOp)
    _, cloned_var = predict_mean_and_var(at(cloned_lik, X_new), mu, var)
    cloned_eval = cloned_var.eval({X_new: X_new_test})

    expected = 1.0 + (0.1 + 0.05 * X_new_test[:, 0] ** 2) ** 2  # var + sigma(X_new)**2
    np.testing.assert_allclose(orig_eval, expected)
    np.testing.assert_allclose(cloned_eval, expected)


def test_node_built_without_holder_is_fully_usable():
    """Build a likelihood node directly from the Op — no holder — and drive it
    through the functional API: param access, re-rooting, and prediction all
    come from ``owner.op``."""
    X = pt.matrix("X", shape=(None, 1))
    sigma = 0.1 + 0.05 * X[:, 0] ** 2
    lik = GaussianOp(has_data=True)(X, sigma)  # a node, via the Op directly

    assert isinstance(op_of(lik), GaussianOp)
    assert param(lik, "sigma") is sigma  # the parameter is the node's input

    new = np.array([[0.0], [1.0], [2.0]])
    rerooted = at(lik, pt.as_tensor_variable(new))
    np.testing.assert_allclose(_eval(param(rerooted, "sigma")), 0.1 + 0.05 * new[:, 0] ** 2)


def test_functional_matches_holder():
    """The free functions and the holder methods are the same computation."""
    X = pt.matrix("X", shape=(None, 1))
    mu = pt.as_tensor_variable(np.array([0.0, 1.0]))
    var = pt.as_tensor_variable(np.array([0.5, 0.2]))
    new = pt.as_tensor_variable(np.array([[0.5], [1.5]]))

    lik = Gaussian(0.1 + 0.05 * X[:, 0] ** 2, x=X)  # a LikelihoodVariable

    hm, hv = lik.at(new).predict_mean_and_var(mu, var)  # method API
    fm, fv = predict_mean_and_var(at(lik, new), mu, var)  # functional API
    np.testing.assert_allclose(_eval(hm), _eval(fm))
    np.testing.assert_allclose(_eval(hv), _eval(fv))


def test_functional_variational_expectation_preserves_rv_identity():
    """The functional path is RV-safe: an RV hyperparameter in the parameter
    survives re-rooting, so it can be evaluated inside a model context."""
    with pm.Model():
        alpha = pm.HalfNormal("alpha")
        X = pt.matrix("X", shape=(None, 1))
        lik = GaussianOp(has_data=True)(X, alpha + 0.05 * X[:, 0] ** 2)
        new = pt.as_tensor_variable(np.array([[0.0], [2.0]]))
        ve = variational_expectation(
            at(lik, new),
            pt.as_tensor_variable(np.array([0.0, 0.0])),
            pt.as_tensor_variable(np.array([0.0, 0.0])),
            pt.as_tensor_variable(np.array([1.0, 1.0])),
        )
    # alpha is still a free input of the re-rooted expectation graph
    val = pytensor.function([alpha], ve)(0.3)
    assert np.isfinite(val).all()


def test_design_matrix_handle_survives_rewrite():
    """Regression — the "don't lose the handle" case: when the design matrix is an
    intermediate node (X = log(x)) and a parameter exps it (sigma = exp(X)), an
    algebraic rewrite would cancel exp(log(x)) -> x and sever our handle to X. The
    design matrix is routed through an opaque gp_data barrier, so canonicalization
    cannot see through it: X survives the rewrite and re-rooting still reaches it.
    """
    from pytensor.graph.rewriting.utils import rewrite_graph
    from pytensor.graph.traversal import ancestors

    x = pt.matrix("x", shape=(None, 1))
    X = pt.log(x)  # design matrix is an intermediate node, not a leaf
    lik = Gaussian(pt.exp(X[:, 0]), x=X)  # sigma = exp(log(x)) — would cancel without a barrier
    sigma = lik.owner.inputs[1]  # the parameter, routed through the gp_data barrier

    rewritten = rewrite_graph(sigma, include=("canonicalize",))
    # The barrier blocked the exp(log(x)) -> x cancellation, so X (=log(x)) survives.
    assert X in set(ancestors([rewritten]))

    # And re-rooting the (held) likelihood still reaches the design matrix.
    X_new = pt.matrix("X_new", shape=(None, 1))
    got = lik.at(X_new).sigma.eval({X_new: np.array([[2.0], [3.0]])})
    np.testing.assert_allclose(got, np.exp([2.0, 3.0]))
