import cloudpickle
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats

from pymc.model.fgraph import clone_model
from pytensor.assumptions import POSITIVE_DEFINITE, SYMMETRIC
from pytensor.assumptions.core import check_assumption
from pytensor.gradient import pushforward
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.graph.traversal import ancestors
from pytensor.scalar.basic import Add

from ptgp.kernels import (
    ExpQuad,
    Gibbs,
    Linear,
    LowRankCategorical,
    Matern12,
    Matern32,
    Matern52,
    Overlap,
    RandomWalk,
    WarpedInput,
)
from ptgp.kernels.base import Kernel, KernelOp, row_placeholders
from ptgp.kernels.nonstationary import GibbsOp
from ptgp.kernels.stationary import Matern52Op

RNG = np.random.default_rng(0)
ROWS = np.column_stack(
    [np.abs(RNG.normal(size=(9, 2))) + 0.1, RNG.integers(low=0, high=3, size=9).astype(float)]
)
OTHER_ROWS = np.column_stack(
    [np.abs(RNG.normal(size=(5, 2))) + 0.1, RNG.integers(low=0, high=3, size=5).astype(float)]
)
LS = pt.scalar("ls")
ETA = pt.scalar("eta")
W = pt.matrix("W", shape=(3, 2))
VALUES = {LS: np.asarray(0.8), ETA: np.asarray(1.3), W: RNG.normal(size=(3, 2))}
CONTINUOUS = [0, 1]

KERNELS = {
    "matern52": Matern52(input_dim=3, ls=LS, active_dims=CONTINUOUS),
    "matern32": Matern32(input_dim=3, ls=LS, active_dims=CONTINUOUS),
    "matern12": Matern12(input_dim=3, ls=LS, active_dims=CONTINUOUS),
    "expquad_scaled": ETA**2 * ExpQuad(input_dim=3, ls=LS, active_dims=CONTINUOUS),
    "linear": Linear(input_dim=3, c=ETA, active_dims=CONTINUOUS),
    "random_walk": RandomWalk(input_dim=3, active_dims=[0]),
    "overlap": Overlap(input_dim=3, active_dims=[2]),
    "low_rank": LowRankCategorical(
        input_dim=3, num_levels=3, W=W, kappa=pt.ones(3), active_dims=[2]
    ),
    "gibbs": Gibbs(lambda inputs: pt.exp(LS * inputs[:, 0]), input_dim=3, active_dims=[0]),
    "warped": WarpedInput(
        input_dim=3,
        kernel_func=Matern52(input_dim=2, ls=LS),
        warp_func=lambda inputs: ETA * inputs,
        active_dims=CONTINUOUS,
    ),
    "sum": ETA**2 * Matern52(input_dim=3, ls=LS, active_dims=CONTINUOUS)
    + ExpQuad(input_dim=3, ls=2.0, active_dims=CONTINUOUS),
    "product_categorical": Matern32(input_dim=3, ls=LS, active_dims=CONTINUOUS)
    * Overlap(input_dim=3, active_dims=[2]),
}


def _run(output, givens=None):
    givens = givens or {}
    return pytensor.function(list(givens), output, on_unused_input="ignore")(*givens.values())


def _rows():
    return pt.matrix("rows", shape=(None, 3)), pt.matrix("other_rows", shape=(None, 3))


def _kernel_ops(graph):
    return [
        variable.owner.op
        for variable in ancestors([graph])
        if variable.owner is not None and isinstance(variable.owner.op, KernelOp)
    ]


def test_base_kernel_is_a_typed_op_with_named_inputs():
    rows, other_rows = _rows()
    K = KERNELS["matern52"](rows, other_rows)

    assert isinstance(K.owner.op, Matern52Op)
    assert K.owner.inputs[0] is rows
    assert K.owner.inputs[1] is other_rows
    assert K.owner.inputs[2] is LS


def test_composition_is_graph_arithmetic_on_typed_ops():
    rows, _ = _rows()
    K = (ETA**2 * Matern52(input_dim=3, ls=LS) + ExpQuad(input_dim=3, ls=2.0))(rows, rows)

    assert isinstance(K.owner.op.scalar_op, Add)
    assert sorted(type(op).__name__ for op in _kernel_ops(K)) == ["ExpQuadOp", "Matern52Op"]


def test_a_kernel_stores_only_its_graph():
    assert set(vars(KERNELS["matern52"])) == {"input_dim", "graph"}


@pytest.mark.parametrize("kernel", KERNELS.values(), ids=KERNELS.keys())
def test_diag_matches_gram(kernel):
    rows, _ = _rows()
    givens = {rows: ROWS, **VALUES}

    np.testing.assert_allclose(
        _run(kernel.diag(rows), givens), np.diag(_run(kernel(rows), givens)), atol=1e-7
    )


def test_row_dependent_expressions_stay_in_the_outer_graph():
    rows, _ = _rows()
    K = KERNELS["gibbs"](rows, rows)

    assert isinstance(K.owner.op, GibbsOp)
    assert LS in ancestors(K.owner.inputs[2:])


def test_same_rows_prove_symmetric_positive_definite():
    rows, other_rows = _rows()
    kernel = KERNELS["matern52"]

    K = kernel(rows)
    fgraph = FunctionGraph(outputs=[K], clone=False)
    assert check_assumption(fgraph, K, SYMMETRIC) is True
    assert check_assumption(fgraph, K, POSITIVE_DEFINITE) is True

    cross = kernel(rows, other_rows)
    cross_fgraph = FunctionGraph(outputs=[cross], clone=False)
    assert check_assumption(cross_fgraph, cross, POSITIVE_DEFINITE) is not True


def test_gradient_with_same_row_node_in_both_slots():
    """K(Z, Z) passes one node twice, and its gradient must not count either slot twice."""
    Z = pt.matrix("Z", shape=(4, 3))
    Z_value = ROWS[:4]
    weights = np.random.default_rng(1).normal(size=(4, 4))

    objective = pt.sum(KERNELS["matern52"](Z) * pt.as_tensor_variable(weights))
    gradient = _run(pytensor.grad(objective, Z), {Z: Z_value, LS: VALUES[LS]})

    step = 1e-6
    numeric = np.zeros_like(Z_value)
    for position in np.ndindex(Z_value.shape):
        bump = np.zeros_like(Z_value)
        bump[position] = step
        above = _run(objective, {Z: Z_value + bump, LS: VALUES[LS]})
        below = _run(objective, {Z: Z_value - bump, LS: VALUES[LS]})
        numeric[position] = (above - below) / (2 * step)

    np.testing.assert_allclose(gradient, numeric, atol=1e-6)


@pytest.mark.parametrize(
    "name", ["matern52", "matern12", "gibbs", "warped", "sum", "product_categorical"]
)
def test_parameter_gradient_matches_finite_difference(name):
    rows, other_rows = _rows()
    objective = pt.sum(KERNELS[name](rows, other_rows))
    givens = {rows: ROWS, other_rows: OTHER_ROWS, ETA: VALUES[ETA], LS: VALUES[LS]}

    gradient = _run(pytensor.grad(objective, LS), givens)

    step = 1e-6
    above = _run(objective, {**givens, LS: VALUES[LS] + step})
    below = _run(objective, {**givens, LS: VALUES[LS] - step})
    np.testing.assert_allclose(gradient, (above - below) / (2 * step), atol=1e-6)


def test_gradient_through_integer_cast_rows():
    """Rows reach ``LowRankCategorical`` through an integer cast, so they are disconnected."""
    rows, _ = _rows()
    objective = pt.sum(KERNELS["low_rank"](rows))
    W_value = VALUES[W]

    gradient = _run(pytensor.grad(objective, W), {rows: ROWS, W: W_value})

    step = 1e-6
    bump = np.zeros_like(W_value)
    bump[1, 0] = step
    above = _run(objective, {rows: ROWS, W: W_value + bump})
    below = _run(objective, {rows: ROWS, W: W_value - bump})
    np.testing.assert_allclose(gradient[1, 0], (above - below) / (2 * step), atol=1e-6)


def test_pushforward_matches_finite_difference():
    rows, other_rows = _rows()
    kernel = KERNELS["matern52"]
    tangent = pt.scalar("tangent")
    wrapped = kernel(rows, other_rows)

    derivative = pushforward(wrapped, [LS], [tangent], use_op_pushforward=True)
    givens = {rows: ROWS, other_rows: OTHER_ROWS, LS: VALUES[LS], tangent: np.asarray(1.0)}

    step = 1e-6
    above = _run(wrapped, {rows: ROWS, other_rows: OTHER_ROWS, LS: VALUES[LS] + step})
    below = _run(wrapped, {rows: ROWS, other_rows: OTHER_ROWS, LS: VALUES[LS] - step})

    np.testing.assert_allclose(_run(derivative, givens), (above - below) / (2 * step), atol=1e-7)


def test_extra_column_raises():
    kernel = KERNELS["matern52"]
    rows = pytensor.shared(ROWS, shape=(None, None))
    function = pytensor.function([LS], kernel(rows))

    rows.set_value(np.column_stack([ROWS, ROWS[:, 0]]))
    with pytest.raises(AssertionError):
        function(0.8)


def test_graph_replace_of_rows():
    rows, _ = _rows()
    K = KERNELS["sum"](rows)
    givens = {ETA: VALUES[ETA], LS: VALUES[LS]}

    new_rows = pt.matrix("new_rows", shape=(None, 3))
    replaced = graph_replace(K, {rows: new_rows})

    np.testing.assert_allclose(
        _run(replaced, {new_rows: OTHER_ROWS, **givens}),
        _run(K, {rows: OTHER_ROWS, **givens}),
        atol=1e-12,
    )


def test_vectorize_over_rows():
    rows, _ = _rows()
    K = KERNELS["sum"](rows)
    givens = {ETA: VALUES[ETA], LS: VALUES[LS]}

    batched_rows = pt.tensor3("batched_rows", shape=(None, None, 3))
    batched = vectorize_graph(K, {rows: batched_rows})

    batch = np.stack([ROWS, ROWS + 1.0])
    expected = [_run(K, {rows: batch[index], **givens}) for index in range(2)]
    np.testing.assert_allclose(_run(batched, {batched_rows: batch, **givens}), expected, atol=1e-12)


def _model_with_deterministic():
    with pm.Model() as model:
        log_ls = pm.Normal("log_ls", mu=0.0, sigma=1.0)
        ls = pm.Deterministic("ls", pt.exp(log_ls))
        eta = pm.Exponential("eta", scale=1.0)
        kernel = eta**2 * Matern52(input_dim=2, ls=ls)

        rows = pm.Data("rows", ROWS[:, :2])
        pm.Potential("trace", pt.sum(pt.diagonal(kernel(rows) @ kernel(rows))))
    return model


def test_conditioning_reaches_the_kernel():
    """`pm.do` on a Deterministic must reach the Op, with no model handed to the kernel."""
    conditioned = pm.do(_model_with_deterministic(), {"ls": 0.7, "eta": 1.2})
    expected_kernel = 1.2**2 * Matern52(input_dim=2, ls=pt.as_tensor_variable(0.7))
    K = expected_kernel(ROWS[:, :2]).eval()

    np.testing.assert_allclose(_run(conditioned.potentials[0]), np.sum(K * K), atol=1e-9)


def test_set_data_after_conditioning_reaches_the_kernel():
    conditioned = pm.do(_model_with_deterministic(), {"ls": 0.7, "eta": 1.2})
    expected_kernel = 1.2**2 * Matern52(input_dim=2, ls=pt.as_tensor_variable(0.7))
    function = pytensor.function([], conditioned.potentials[0])

    new_rows = OTHER_ROWS[:, :2]
    with conditioned:
        pm.set_data({"rows": new_rows})

    K = expected_kernel(new_rows).eval()
    np.testing.assert_allclose(function(), np.sum(K * K), atol=1e-9)


def test_conditioning_a_valued_variable_needs_no_potential():
    rows = ROWS[:, :2]
    observed_y = np.sin(rows[:, 0])
    with pm.Model() as model:
        ls = pm.Deterministic("ls", pt.exp(pm.Normal("log_ls")))
        eta = pm.Exponential("eta", scale=1.0)
        kernel = eta**2 * Matern52(input_dim=2, ls=ls)
        covariance = kernel(pm.Data("rows", rows)) + 0.25 * pt.eye(9)
        pm.MvNormal("y", mu=np.zeros(9), cov=covariance, observed=observed_y)

    conditioned = pm.do(pm.observe(model, {"eta": 1.2}), {"ls": 0.7})
    logp = conditioned.compile_logp(vars=[conditioned["y"]])

    K = (1.2**2 * Matern52(input_dim=2, ls=pt.as_tensor_variable(0.7)))(rows).eval()
    expected = scipy.stats.multivariate_normal(np.zeros(9), K + 0.25 * np.eye(9)).logpdf(observed_y)
    assert not model.potentials
    np.testing.assert_allclose(logp({"log_ls": 0.0}), expected, atol=1e-9)


def test_constant_rooted_random_variable_is_an_input():
    with pm.Model():
        ls = pm.DiracDelta("ls", 1.5)
        rows = pm.Data("rows", ROWS[:, :1])
        K = Matern52(input_dim=1, ls=ls)(rows, rows)

    assert ls in K.owner.inputs


def test_lowered_diag_keeps_model_variables():
    rows = ROWS[:, :1]
    with pm.Model() as model:
        c = pm.Normal("c")
        diagonal = Linear(input_dim=1, c=c).diag(pm.Data("rows", rows))

    assert c in ancestors([diagonal])

    [diagonal_at_value] = model.replace_rvs_by_values([diagonal])
    value = _run(diagonal_at_value, {model.rvs_to_values[c]: np.asarray(0.3)})
    np.testing.assert_allclose(value, (rows[:, 0] - 0.3) ** 2)


def test_observing_a_parameter_reaches_the_kernel():
    observed = pm.observe(_model_with_deterministic(), {"eta": 1.2})
    expected_kernel = 1.2**2 * Matern52(input_dim=2, ls=pt.as_tensor_variable(0.7))
    K = expected_kernel(ROWS[:, :2]).eval()

    [potential] = observed.replace_rvs_by_values(observed.potentials)
    value = observed.rvs_to_values[observed["log_ls"]]

    np.testing.assert_allclose(
        _run(potential, {value: np.log(np.asarray(0.7))}), np.sum(K * K), atol=1e-9
    )


@pytest.mark.parametrize(
    "transform",
    [clone_model, lambda m: cloudpickle.loads(cloudpickle.dumps(m))],
    ids=["clone", "pickle"],
)
def test_model_with_kernels_survives(transform):
    model = _model_with_deterministic()
    point = model.initial_point()

    np.testing.assert_allclose(
        transform(model).compile_logp()(point), model.compile_logp()(point), atol=1e-10
    )


@pytest.mark.parametrize("dtype", ["int64", "float32"])
def test_rows_of_any_numeric_dtype_are_accepted(dtype):
    rows = np.arange(4.0)[:, None]
    kernel = Matern52(input_dim=1, ls=1.0)

    np.testing.assert_allclose(kernel(rows.astype(dtype)).eval(), kernel(rows).eval(), atol=1e-12)


def test_overlap_accepts_integer_level_codes():
    levels = np.array([[0], [1], [0]])

    np.testing.assert_allclose(Overlap(input_dim=1)(levels).eval(), (levels == levels.T) * 1.0)


@pytest.mark.parametrize(
    ("built", "called"),
    [("float32", "float64"), ("float64", "float32")],
    ids=["built_float32", "built_float64"],
)
def test_kernel_outlives_a_change_of_floatx(built, called):
    rows = np.arange(4.0)[:, None]

    with pytensor.config.change_flags(floatX=built):
        kernel = Matern52(input_dim=1, ls=1.0)
    expected = Matern52(input_dim=1, ls=1.0)(rows).eval()

    with pytensor.config.change_flags(floatX=called):
        K = kernel(rows.astype(called))

    np.testing.assert_allclose(K.eval(), expected, atol=1e-6)


@pytest.mark.parametrize(
    "kernel_name", ["matern52", "sum", "warped"], ids=["base", "composite", "warped"]
)
def test_rows_are_found_structurally_not_by_name(kernel_name, monkeypatch):
    left, right = row_placeholders(3)
    monkeypatch.setattr(left, "name", None)
    monkeypatch.setattr(right, "name", "renamed")

    found_left, found_right = KERNELS[kernel_name].rows

    assert found_left is left
    assert found_right is right


def _pickled(kernel_type, **kwargs):
    return cloudpickle.loads(cloudpickle.dumps(kernel_type(**kwargs)))


def _built_under_float32(kernel_type, **kwargs):
    with pytensor.config.change_flags(floatX="float32"):
        return kernel_type(**kwargs)


@pytest.mark.parametrize(
    "inner_builder", [_pickled, _built_under_float32], ids=["unpickled", "other_floatx"]
)
def test_warp_applies_to_a_kernel_on_other_row_placeholders(inner_builder):
    rows = np.arange(4.0)[:, None]
    inner = inner_builder(Matern52, input_dim=1, ls=1.0)

    warped = WarpedInput(input_dim=1, kernel_func=inner, warp_func=lambda x: 2.0 * x)

    expected = Matern52(input_dim=1, ls=1.0)(2.0 * rows).eval()
    np.testing.assert_allclose(warped(rows).eval(), expected, atol=1e-6)


@pytest.mark.parametrize(
    "inner_builder", [_pickled, _built_under_float32], ids=["unpickled", "other_floatx"]
)
@pytest.mark.parametrize("flip", [False, True], ids=["left_operand", "right_operand"])
def test_combination_rebases_a_kernel_on_other_row_placeholders(inner_builder, flip):
    rows = np.arange(4.0)[:, None]
    other = inner_builder(Matern52, input_dim=1, ls=2.0)
    kernel = Matern52(input_dim=1, ls=1.0)

    combined = other + kernel if flip else kernel + other

    first, second = Matern52(input_dim=1, ls=1.0), Matern52(input_dim=1, ls=2.0)
    expected = first(rows).eval() + second(rows).eval()
    np.testing.assert_allclose(combined(rows).eval(), expected, atol=1e-6)


def test_rows_of_a_graph_without_a_kernel_op_raises():
    kernel = Kernel(input_dim=1, graph=pt.ones((2, 2)))

    with pytest.raises(ValueError, match="no KernelOp"):
        kernel.rows


def test_rows_of_a_warp_of_a_warp():
    inner = WarpedInput(
        input_dim=2, kernel_func=Matern52(input_dim=2, ls=1.0), warp_func=lambda x: 2.0 * x
    )
    outer = WarpedInput(input_dim=2, kernel_func=inner, warp_func=pt.exp)

    left, right = row_placeholders(2)
    assert outer.rows == (left, right)

    rows = ROWS[:, :2]
    expected = Matern52(input_dim=2, ls=1.0)(2.0 * np.exp(rows)).eval()
    np.testing.assert_allclose(outer(rows).eval(), expected, atol=1e-12)


def test_warp_parameter_gradient_matches_finite_difference():
    rows, _ = _rows()
    objective = pt.sum(KERNELS["warped"](rows))
    givens = {rows: ROWS, ETA: VALUES[ETA], LS: VALUES[LS]}

    gradient = _run(pytensor.grad(objective, ETA), givens)

    step = 1e-6
    above = _run(objective, {**givens, ETA: VALUES[ETA] + step})
    below = _run(objective, {**givens, ETA: VALUES[ETA] - step})
    np.testing.assert_allclose(gradient, (above - below) / (2 * step), atol=1e-6)


def test_conditioning_reaches_a_warp_parameter():
    rows = ROWS[:, :2]
    with pm.Model() as model:
        scale = pm.Normal("scale")
        kernel = WarpedInput(
            input_dim=2,
            kernel_func=Matern52(input_dim=2, ls=1.0),
            warp_func=lambda inputs: scale * inputs,
        )
        pm.Potential("total", pt.sum(kernel(pm.Data("rows", rows))))

    conditioned = pm.do(model, {"scale": 2.0})
    expected = Matern52(input_dim=2, ls=1.0)(2.0 * rows).eval()

    np.testing.assert_allclose(_run(conditioned.potentials[0]), expected.sum(), atol=1e-9)
