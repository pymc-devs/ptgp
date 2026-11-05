"""VFE sparse GP tests against GPJax reference implementation."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp.gp.unapproximated import Unapproximated
from ptgp.gp.vfe import VFE
from ptgp.inducing import InducingVariables, Points
from ptgp.kernels.stationary import ExpQuad, Matern32


class _NoZInducing(InducingVariables):
    @property
    def num_inducing(self):
        return 5


def test_vfe_rejects_inducing_without_Z():
    with pytest.raises(TypeError, match=r"\.Z attribute"):
        VFE(kernel=Matern32(input_dim=1, ls=0.5), inducing_variable=_NoZInducing())


def test_vfe_predict_matches_exact_gp_when_Z_equals_X():
    """With Z = X, VFE prediction matches the exact GP, even when Kuu is ill-conditioned.

    The tolerance absorbs the Kuu jitter, which shifts the mean by about 3e-4 here.
    """
    rng = np.random.default_rng(0)
    X_np = np.sort(rng.uniform(0, 10, 40))[:, None]
    y_np = np.sin(X_np[:, 0]) + 0.1 * rng.normal(size=40)
    Xs_np = np.linspace(-1, 11, 25)[:, None]

    X, Xs, y = pt.dmatrix("X"), pt.dmatrix("Xs"), pt.dvector("y")
    kernel = ExpQuad(input_dim=1, ls=5.0)
    vfe = VFE(kernel=kernel, sigma=0.1, inducing_variable=Points(X))
    exact = Unapproximated(kernel=kernel, sigma=0.1)

    fn = pytensor.function(
        [Xs, X, y], [*vfe.predict_marginal(Xs, X, y), *exact.predict_marginal(Xs, X, y)]
    )
    mean_vfe, var_vfe, mean_exact, var_exact = fn(Xs_np, X_np, y_np)
    np.testing.assert_allclose(mean_vfe, mean_exact, atol=1e-3)
    np.testing.assert_allclose(var_vfe, var_exact, rtol=1e-3)
