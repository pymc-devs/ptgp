"""Each GP model's loss + gradients compile and evaluate under PyTensor's MLX backend.

MLX only runs float32 on the GPU, and its ``arange`` needs concrete bounds, so
the inducing matrix carries a static ``M`` (as the shape conventions require).
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp.gp import SVGP, VFE, Unapproximated, init_variational_params
from ptgp.inducing import Points
from ptgp.kernels import ExpQuad
from ptgp.likelihoods import Gaussian
from ptgp.mean import Zero
from ptgp.objectives import collapsed_elbo, elbo, marginal_log_likelihood

pytest.importorskip("mlx")

M = 8


@pytest.fixture(autouse=True)
def _float32():
    old = pytensor.config.floatX
    pytensor.config.floatX = "float32"
    yield
    pytensor.config.floatX = old


def _data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (50, 1)).astype("float32")
    y = (np.sin(X[:, 0]) + 0.1 * rng.normal(size=50)).astype("float32")
    Z = np.linspace(-3, 3, M, dtype="float32")[:, None]
    return X, y, Z


def _symbols():
    X = pt.matrix("X", shape=(None, 1))
    y = pt.vector("y")
    sigma = pt.scalar("sigma")
    ls = pt.scalar("ls")
    Z = pt.matrix("Z", shape=(M, 1))
    return X, y, sigma, ls, Z


def _assert_mlx_matches_default(loss, inputs, wrt, values):
    outs = [loss, *pt.grad(loss, wrt)]
    ref = pytensor.function(inputs, outs)(*values)
    got = pytensor.function(inputs, outs, mode="MLX")(*values)
    for r, g in zip(ref, got, strict=True):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-3, atol=1e-3)


def test_unapproximated_runs_under_mlx():
    X, y, sigma, ls, _ = _symbols()
    Xv, yv, _ = _data()
    gp = Unapproximated(kernel=ExpQuad(input_dim=1, ls=ls), mean=Zero(), sigma=sigma)
    loss = -marginal_log_likelihood(gp, X, y).mll
    _assert_mlx_matches_default(loss, [X, y, sigma, ls], [sigma, ls], [Xv, yv, 0.3, 1.0])


def test_vfe_runs_under_mlx():
    X, y, sigma, ls, Z = _symbols()
    Xv, yv, Zv = _data()
    vfe = VFE(
        kernel=ExpQuad(input_dim=1, ls=ls), mean=Zero(), sigma=sigma, inducing_variable=Points(Z)
    )
    loss = -collapsed_elbo(vfe, X, y).elbo
    _assert_mlx_matches_default(loss, [X, y, sigma, ls, Z], [sigma, ls, Z], [Xv, yv, 0.3, 1.0, Zv])


def test_svgp_runs_under_mlx():
    X, y, sigma, ls, Z = _symbols()
    Xv, yv, Zv = _data()
    vp = init_variational_params(M)
    svgp = SVGP(
        kernel=ExpQuad(input_dim=1, ls=ls),
        mean=Zero(),
        likelihood=Gaussian(sigma),
        inducing_variable=Points(Z),
        variational_params=vp,
    )
    loss = -elbo(svgp, X, y).elbo
    q_sqrt_flat = vp.extra_vars[1]
    _assert_mlx_matches_default(
        loss,
        [X, y, sigma, ls, Z, vp.q_mu, q_sqrt_flat],
        [sigma, ls, Z, vp.q_mu, q_sqrt_flat],
        [Xv, yv, 0.3, 1.0, Zv, np.zeros(M, "float32"), np.zeros(M * (M + 1) // 2, "float32")],
    )
