import numpy as np
import pytensor.tensor as pt
import pytest

from ptgp.kernels.stationary import ExpQuad, Matern12, Matern32, Matern52


@pytest.mark.parametrize("kernel_cls", [ExpQuad, Matern12, Matern32, Matern52])
def test_stationary_diag_matches_pt_diag(kernel_cls):
    k = kernel_cls(input_dim=1, ls=1.0)
    X = pt.as_tensor(np.linspace(0, 1, 7)[:, None])
    np.testing.assert_allclose(k.diag(X).eval(), np.diag(k(X).eval()), atol=1e-12)


def test_stationary_diag_is_ones():
    k = Matern32(input_dim=1, ls=0.5)
    X = pt.as_tensor(np.linspace(0, 1, 50)[:, None])
    np.testing.assert_allclose(k.diag(X).eval(), np.ones(50), atol=1e-12)


def test_product_diag_with_scalar():
    k = 2.0 * Matern32(input_dim=1, ls=1.0)
    X = pt.as_tensor(np.linspace(0, 1, 5)[:, None])
    np.testing.assert_allclose(k.diag(X).eval(), 2.0 * np.ones(5), atol=1e-12)


def test_sum_diag():
    k = Matern32(input_dim=1, ls=1.0) + ExpQuad(input_dim=1, ls=2.0)
    X = pt.as_tensor(np.linspace(0, 1, 5)[:, None])
    np.testing.assert_allclose(k.diag(X).eval(), np.diag(k(X).eval()), atol=1e-12)
