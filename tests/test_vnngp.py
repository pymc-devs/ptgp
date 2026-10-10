import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

import ptgp as pg

from ptgp.gp import SVGP, VNNGP, VariationalParams
from ptgp.inducing import Points
from ptgp.kernels import ExpQuad
from ptgp.likelihoods import Gaussian
from ptgp.objectives import vnngp_diagnostics, vnngp_elbo

M, D = 24, 2


def _inputs(seed=0):
    return np.random.default_rng(seed).uniform(size=(M, D))


def _random_q(vnngp, seed=1):
    rng = np.random.default_rng(seed)
    mu0, flat0 = vnngp.extra_init
    return mu0 + rng.normal(size=mu0.shape), flat0 + 0.3 * rng.normal(size=flat0.shape)


def _q_moments(vnngp, q_values, idx):
    means, S = vnngp.variational_params.local_moments(pt.as_tensor_variable(idx))
    fn = pytensor.function(list(vnngp.extra_vars), [means, S])
    return fn(*q_values)


@pytest.mark.parametrize("block_size", [1, 4])
def test_kl_is_exact_with_all_predecessors(block_size):
    """With k = M - 1 the nearest-neighbor prior is the full GP prior, so the summed
    per-point KL equals the dense Gaussian KL against Kmm."""
    X = _inputs()
    kernel = ExpQuad(input_dim=D, ls=0.15)  # cond(Kmm) ~ 3e4 for these points
    vnngp = VNNGP(kernel, X, Gaussian(0.1), k=M - 1, block_size=block_size, jitter=1e-10)
    q = _random_q(vnngp)
    kl_idx = pt.lvector("kl_idx")
    kl = pt.sum(vnngp.prior_kl_terms(kl_idx).kl)
    kl_val = pytensor.function([kl_idx, *vnngp.extra_vars], kl)(np.arange(M), *q)

    m, S = _q_moments(vnngp, q, np.arange(M)[None, :])
    m, S = m[0], S[0]
    Kmm = pytensor.function([], kernel(pt.as_tensor_variable(vnngp.Z)))()
    Kinv_S = np.linalg.solve(Kmm, S)
    ref = 0.5 * (
        np.trace(Kinv_S)
        + m @ np.linalg.solve(Kmm, m)
        - M
        + np.linalg.slogdet(Kmm)[1]
        - np.linalg.slogdet(S)[1]
    )
    np.testing.assert_allclose(kl_val, ref, rtol=1e-6)


def test_prediction_matches_svgp_on_neighbors():
    """A test point's VNNGP prediction is the SVGP conditional with Z = its k
    nearest inducing points and q restricted to them."""
    X = _inputs()
    kernel = ExpQuad(input_dim=D, ls=0.4)
    k = 5
    vnngp = VNNGP(kernel, X, Gaussian(0.1), k=k, block_size=4)
    q = _random_q(vnngp)
    X_new = np.random.default_rng(2).uniform(size=(6, D))
    nn = pg.neighbors.query_neighbors(X_new, k, Z=vnngp.Z)

    X_t = pt.matrix("X_new", shape=(None, D))
    nn_t = pt.matrix("nn", shape=(None, k), dtype="int64")
    mean, var = vnngp.predict_marginal(X_t, nn_t)
    mean, var = pytensor.function([X_t, nn_t, *vnngp.extra_vars], [mean, var])(X_new, nn, *q)

    m_all, S_all = _q_moments(vnngp, q, nn)
    for i in range(len(X_new)):
        vp = VariationalParams(
            q_mu=pt.as_tensor_variable(m_all[i]),
            q_sqrt=pt.as_tensor_variable(np.linalg.cholesky(S_all[i])),
        )
        svgp = SVGP(
            kernel=kernel,
            likelihood=Gaussian(0.1),
            inducing_variable=Points(pt.as_tensor_variable(vnngp.Z[nn[i]])),
            variational_params=vp,
            whiten=False,
        )
        ref_m, ref_v = (
            t.eval() for t in svgp.predict_marginal(pt.as_tensor_variable(X_new[i : i + 1]))
        )
        np.testing.assert_allclose(mean[i], ref_m[0], atol=1e-6)
        np.testing.assert_allclose(var[i], ref_v[0], atol=1e-6)


def test_train_recompute_predict_heteroskedastic():
    rng = np.random.default_rng(0)
    N = 120
    X = rng.uniform(size=(N, D))
    y = np.sin(4 * X[:, 0]) + 0.1 * rng.normal(size=N)
    X_var = pt.matrix("X", shape=(None, D))
    y_var = pt.vector("y")
    row_idx = pt.lvector("row_idx")
    kl_idx = pt.lvector("kl_idx")
    with pm.Model() as model:
        ls = pm.LogNormal("ls", 0.0, 0.5, shape=D)
        sigma = 0.1 + 0.05 * X_var[:, 0]
        vnngp = VNNGP(ExpQuad(input_dim=D, ls=ls), X, Gaussian(sigma, x=X_var), k=8, block_size=4)

    step, shared_params, shared_extras = pg.optim.compile_training_step(
        vnngp_elbo, vnngp, X_var, y_var, model, extra_inputs=[row_idx, kl_idx]
    )
    diag = pg.optim.compile_diagnostics(
        vnngp_diagnostics,
        vnngp,
        [X_var, y_var, row_idx, kl_idx],
        model,
        shared_params,
        shared_extras=shared_extras,
    )
    losses = []
    for _ in range(60):
        rows = rng.choice(N, 32, replace=False)
        kls = rng.choice(vnngp.num_inducing, 32, replace=False)
        losses.append(step(X[rows], y[rows], rows, kls))
    assert np.mean(losses[-10:]) < np.mean(losses[:10])

    all_rows, all_z = np.arange(N), np.arange(vnngp.num_inducing)
    d = diag(X, y, all_rows, all_z)
    assert all(np.isfinite(v) for v in d)

    update = vnngp.recompute_neighbors(model, shared_params, shared_extras=shared_extras, scale=ls)
    assert 0.0 <= update.frac_changed <= 1.0 and 0.0 < update.block_coverage <= 1.0
    ls_shared = shared_params[model.rvs_to_values[ls]]
    np.testing.assert_allclose(vnngp._predict_metric["scale"], np.exp(ls_shared.get_value()))
    with pytest.raises(AssertionError, match="same number of rows"):
        step(X[:4], y[:4], np.arange(5), all_z[:4])

    X_new = rng.uniform(size=(10, D))
    X_new_var = pt.matrix("X_new", shape=(None, D))
    kw = dict(shared_extras=shared_extras)
    m, v = pg.optim.compile_predict(vnngp, X_new_var, model, shared_params, **kw)(X_new)
    ym, yv = pg.optim.compile_predict(vnngp, X_new_var, model, shared_params, incl_lik=True, **kw)(
        X_new
    )
    np.testing.assert_allclose(yv - v, (0.1 + 0.05 * X_new[:, 0]) ** 2, atol=1e-10)

    ls_shared.set_value(ls_shared.get_value() + 0.5)
    predict = pg.optim.compile_predict(vnngp, X_new_var, model, shared_params, **kw)
    with pytest.raises(ValueError, match="recompute_neighbors"):
        predict(X_new)
