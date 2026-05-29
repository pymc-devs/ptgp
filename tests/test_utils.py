"""Tests for ptgp.utils.check_init."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

import ptgp as pg

from ptgp.utils import _build_index_labels, check_init


def _make_simple_model_and_objective():
    """1-D GP on synthetic data; returns (fun, theta0, X, y, model)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 5, size=(30, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(30)

    X_var = pt.matrix("X", shape=X.shape)
    y_var = pt.vector("y", shape=y.shape)

    Z_init = pg.inducing.kmeans_init(X, 8, rng=0)[0].Z

    with pm.Model() as model:
        eta = pm.Exponential("eta", scale=1.0)
        sigma = pm.Exponential("sigma", scale=0.5)
        ls = pm.Exponential("ls", scale=1.0)
        kernel = eta**2 * pg.kernels.ExpQuad(input_dim=1, ls=ls)
        gp = pg.gp.VFE(
            kernel=kernel,
            mean=pg.mean.Zero(),
            sigma=sigma,
            inducing_variable=pg.inducing.Points(Z_init),
        )

    fun, theta0, *_ = pg.optim.compile_scipy_objective(
        lambda gp, X, y: pg.objectives.collapsed_elbo(gp, X, y).elbo, gp, X_var, y_var, model=model
    )
    return fun, theta0, X, y, model


class TestCheckInit:
    def test_returns_true_on_valid_init(self):
        fun, theta0, X, y, model = _make_simple_model_and_objective()
        result = check_init(fun, theta0, X, y, model=model)
        assert result is True

    def test_returns_false_on_nan_loss(self):
        """Inject a NaN-producing function and confirm check_init catches it."""

        def bad_fun(theta, X, y):
            return np.nan, np.zeros_like(theta)

        theta0 = np.ones(3)
        result = check_init(bad_fun, theta0, np.zeros((5, 1)), np.zeros(5))
        assert result is False

    def test_returns_false_on_nan_grad(self):
        def bad_fun(theta, X, y):
            g = np.zeros_like(theta)
            g[0] = np.nan
            return 1.0, g

        theta0 = np.ones(3)
        result = check_init(bad_fun, theta0, np.zeros((5, 1)), np.zeros(5))
        assert result is False

    def test_prints_loss_line(self, capsys):
        fun, theta0, X, y, model = _make_simple_model_and_objective()
        check_init(fun, theta0, X, y, model=model)
        out = capsys.readouterr().out
        assert "loss at init" in out
        assert "grad finite" in out

    def test_prints_topk_lines(self, capsys):
        fun, theta0, X, y, model = _make_simple_model_and_objective()
        top_k = 2
        check_init(fun, theta0, X, y, model=model, top_k=top_k)
        out = capsys.readouterr().out
        assert f"top-{top_k}" in out
        # Expect exactly top_k labelled rows in the table
        table_lines = [line for line in out.splitlines() if line.strip().startswith("[")]
        assert len(table_lines) == top_k

    def test_labels_contain_param_names(self, capsys):
        fun, theta0, X, y, model = _make_simple_model_and_objective()
        check_init(fun, theta0, X, y, model=model, top_k=10)
        out = capsys.readouterr().out
        # The model has eta, sigma, ls -- their transformed names should appear
        # somewhere in the output (exact transform suffix depends on PyMC version)
        assert any(name in out for name in ("eta", "sigma", "ls"))

    def test_no_model_uses_indices(self, capsys):
        """Without a model, the table should still print (using numeric indices)."""

        def ok_fun(theta, X, y):
            return -5.0, np.arange(len(theta), dtype=float)

        theta0 = np.zeros(5)
        result = check_init(ok_fun, theta0, np.zeros((2, 1)), np.zeros(2), top_k=3)
        out = capsys.readouterr().out
        assert result is True
        assert "top-3" in out

    def test_top_k_capped_at_theta_size(self, capsys):
        """Requesting more top-K than parameters should not error."""
        fun, theta0, X, y, model = _make_simple_model_and_objective()
        check_init(fun, theta0, X, y, model=model, top_k=10_000)
        out = capsys.readouterr().out
        table_lines = [line for line in out.splitlines() if line.strip().startswith("[")]
        assert len(table_lines) == theta0.size


class TestBuildIndexLabels:
    def test_returns_none_without_model(self):
        result = _build_index_labels(5, model=None, extra_vars=None, extra_init=None)
        assert result is None

    def test_scalar_params_no_brackets(self):
        with pm.Model() as model:
            pm.Exponential("eta", scale=1.0)
            pm.Exponential("sigma", scale=1.0)
        ip = model.initial_point()
        n = sum(np.asarray(v).size for v in ip.values())
        labels = _build_index_labels(n, model=model, extra_vars=None, extra_init=None)
        assert labels is not None
        assert len(labels) == n
        # Scalar params should not have bracket suffixes
        assert all("[" not in label for label in labels)

    def test_vector_param_has_index_suffix(self):
        with pm.Model() as model:
            pm.Exponential("ls", scale=1.0, shape=3)
        ip = model.initial_point()
        n = sum(np.asarray(v).size for v in ip.values())
        labels = _build_index_labels(n, model=model, extra_vars=None, extra_init=None)
        assert labels is not None
        bracketed = [label for label in labels if "[" in label]
        assert len(bracketed) == 3

    def test_extra_vars_labelled(self):
        import pytensor.tensor as pt

        with pm.Model() as model:
            pm.Exponential("eta", scale=1.0)
        Z_var = pt.matrix("Z_logit")
        Z_init = np.zeros((4, 2))
        ip = model.initial_point()
        n = sum(np.asarray(v).size for v in ip.values()) + Z_init.size
        labels = _build_index_labels(n, model=model, extra_vars=[Z_var], extra_init=[Z_init])
        assert labels is not None
        assert len(labels) == n
        z_labels = [label for label in labels if "Z_logit" in label]
        assert len(z_labels) == Z_init.size
