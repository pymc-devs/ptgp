"""Regression tests pinning the cubic-factorization count of each GP model's joint graph.

These tests assert that PTGP's rewrite system compiles the joint graph
(loss + all gradients) down to the *minimum* number of O(N³) factorizations
(Cholesky / Solve / Det / SLogDet / MatrixInverse / LUFactor) — the
cubic-op floor. If a regression in either ptgp's own rewrites or the
upstream PyTensor pipeline causes the floor to be missed, these tests
fail loudly.

The floor counts are:

- ``Unapproximated`` — 1 (one Cholesky on ``K + σ²I``).
- ``VFE``           — 2 (Cholesky on ``Kuu`` and on ``D = σ²·Kuu + Kuf @ Kuf.T``).
- ``SVGP``          — 1 (Cholesky on ``Kuu``).
- ``VNNGP``         — 1 batched Cholesky on the ``(batch, k, k)`` neighbor
  covariances; nothing touches an ``M x M`` or ``N x N`` matrix.

When a test fails, first check whether the joint graph genuinely has
extra cubic ops (regression) or whether the upstream PyTensor compile
pipeline has restructured something benignly. ``scripts/joint_graph_analysis.py``
prints the full op breakdown and is the right tool for diagnosis.
"""

import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.tensor.linalg.decomposition.cholesky import Cholesky
from pytensor.tensor.linalg.decomposition.lu import LUFactor
from pytensor.tensor.linalg.inverse import MatrixInverse
from pytensor.tensor.linalg.solvers.general import Solve
from pytensor.tensor.linalg.summary import Det, SLogDet

from ptgp.gp import SVGP, VFE, VNNGP, Unapproximated, init_variational_params
from ptgp.inducing import Points
from ptgp.kernels import ExpQuad
from ptgp.likelihoods import Gaussian
from ptgp.mean import Zero
from ptgp.objectives import collapsed_elbo, elbo, marginal_log_likelihood, vnngp_elbo

CUBIC_OPS = (Cholesky, MatrixInverse, Solve, Det, SLogDet, LUFactor)


def _count_cubic_ops(fn) -> int:
    """Count Apply nodes in the compiled graph whose op is an O(N³) factorization."""
    n = 0
    for node in fn.maker.fgraph.apply_nodes:
        # Blockwise, and the BlockwiseWithCoreShape wrapper the Numba backend uses, expose core_op
        core = getattr(node.op, "core_op", node.op)
        if isinstance(core, CUBIC_OPS):
            n += 1
    return n


def test_unapproximated_joint_graph_at_cubic_floor():
    """Unapproximated GP joint graph compiles to exactly 1 cubic factorization."""
    X = pt.dmatrix("X")
    y = pt.dvector("y")
    sigma = pt.dscalar("sigma")
    ls = pt.dscalar("ls")
    gp = Unapproximated(kernel=ExpQuad(input_dim=1, ls=ls), mean=Zero(), sigma=sigma)
    loss = marginal_log_likelihood(gp, X, y).mll
    g_sigma, g_ls = pt.grad(loss, [sigma, ls])
    fn = pytensor.function([X, y, sigma, ls], [loss, g_sigma, g_ls])
    n = _count_cubic_ops(fn)
    assert n == 1, (
        f"Unapproximated joint graph should have 1 cubic factorization "
        f"(Cholesky on K + σ²I); got {n}. Run scripts/joint_graph_analysis.py "
        f"for the full op breakdown."
    )


def test_vfe_joint_graph_at_cubic_floor():
    """VFE joint graph compiles to exactly 2 cubic factorizations."""
    X = pt.dmatrix("X")
    y = pt.dvector("y")
    sigma = pt.dscalar("sigma")
    ls = pt.dscalar("ls")
    Z = pt.dmatrix("Z")
    vfe = VFE(
        kernel=ExpQuad(input_dim=1, ls=ls),
        mean=Zero(),
        sigma=sigma,
        inducing_variable=Points(Z),
    )
    loss = -collapsed_elbo(vfe, X, y).elbo
    g_sigma, g_ls, g_Z = pt.grad(loss, [sigma, ls, Z])
    fn = pytensor.function([X, y, sigma, ls, Z], [loss, g_sigma, g_ls, g_Z])
    n = _count_cubic_ops(fn)
    assert n == 2, (
        f"VFE joint graph should have 2 cubic factorizations "
        f"(Cholesky on Kuu and on D = σ²·Kuu + Kuf @ Kuf.T); got {n}. "
        f"Run scripts/joint_graph_analysis.py for the full op breakdown."
    )


def test_svgp_joint_graph_at_cubic_floor():
    """SVGP joint graph compiles to exactly 1 cubic factorization."""
    X = pt.dmatrix("X")
    y = pt.dvector("y")
    sigma = pt.dscalar("sigma")
    ls = pt.dscalar("ls")
    Z = pt.dmatrix("Z")
    vp = init_variational_params(8)
    svgp = SVGP(
        kernel=ExpQuad(input_dim=1, ls=ls),
        mean=Zero(),
        likelihood=Gaussian(sigma),
        inducing_variable=Points(Z),
        variational_params=vp,
    )
    loss = -elbo(svgp, X, y).elbo
    g_sigma, g_ls, g_Z, g_qmu, g_qsq = pt.grad(loss, [sigma, ls, Z, vp.q_mu, vp.extra_vars[1]])
    fn = pytensor.function(
        [X, y, sigma, ls, Z, vp.q_mu, vp.extra_vars[1]],
        [loss, g_sigma, g_ls, g_Z, g_qmu, g_qsq],
    )
    n = _count_cubic_ops(fn)
    assert n == 1, (
        f"SVGP joint graph should have 1 cubic factorization "
        f"(Cholesky on Kuu); got {n}. Run scripts/joint_graph_analysis.py "
        f"for the full op breakdown."
    )


@pytest.mark.parametrize("block_size", [1, 4])
def test_vnngp_joint_graph_at_cubic_floor(block_size):
    """VNNGP joint graph has one batched Cholesky, on the (batch, k, k) neighbor covariances."""
    import numpy as np

    k = 6
    X_data = np.random.default_rng(0).uniform(size=(40, 2))
    X = pt.dmatrix("X")
    y = pt.dvector("y")
    row_idx = pt.lvector("row_idx")
    kl_idx = pt.lvector("kl_idx")
    sigma = pt.dscalar("sigma")
    ls = pt.dscalar("ls")
    vnngp = VNNGP(ExpQuad(input_dim=2, ls=ls), X_data, Gaussian(sigma), k=k, block_size=block_size)
    loss = -vnngp_elbo(vnngp, X, y, row_idx, kl_idx).elbo
    wrt = [sigma, ls, *vnngp.extra_vars]
    fn = pytensor.function([X, y, row_idx, kl_idx, *wrt], [loss, *pt.grad(loss, wrt)])
    cubic = [
        node
        for node in fn.maker.fgraph.apply_nodes
        if isinstance(getattr(node.op, "core_op", node.op), CUBIC_OPS)
    ]
    assert len(cubic) == 1, (
        f"VNNGP joint graph should have 1 cubic factorization; got {len(cubic)}. "
        f"Run scripts/joint_graph_analysis.py for the full op breakdown."
    )
    assert cubic[0].inputs[0].type.shape[-2:] == (k, k)
