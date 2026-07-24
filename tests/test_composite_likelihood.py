"""CompositeLikelihood tests: per-subset dispatch and end-to-end use with VGP."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

import ptgp as pg


def test_variational_expectation_equals_per_subset():
    """The combinator equals each sub-likelihood computed on its slice."""
    rng = np.random.default_rng(0)
    N = 10
    mu = rng.standard_normal(N)
    var = rng.uniform(0.1, 1.0, N)
    y = rng.standard_normal(N)
    idx0 = np.array([0, 2, 4, 6, 8])  # interleaved to exercise non-contiguous subsets
    idx1 = np.array([1, 3, 5, 7, 9])

    lik0 = pg.likelihoods.StudentT(nu=4.0, sigma=0.5)
    lik1 = pg.likelihoods.Gaussian(0.3)
    composite = pg.likelihoods.CompositeLikelihood([lik0, lik1], [idx0, idx1])

    yv, muv, varv = pt.vector("y"), pt.vector("mu"), pt.vector("var")
    comp_fn = pytensor.function([yv, muv, varv], composite.variational_expectation(yv, muv, varv))
    e0_fn = pytensor.function([yv, muv, varv], lik0.variational_expectation(yv, muv, varv))
    e1_fn = pytensor.function([yv, muv, varv], lik1.variational_expectation(yv, muv, varv))

    out = comp_fn(y, mu, var)
    expected = np.zeros(N)
    expected[idx0] = e0_fn(y[idx0], mu[idx0], var[idx0])
    expected[idx1] = e1_fn(y[idx1], mu[idx1], var[idx1])
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_predict_mean_and_var_dispatch():
    """predict_mean_and_var is scattered per subset."""
    rng = np.random.default_rng(1)
    N = 6
    mu = rng.standard_normal(N)
    var = rng.uniform(0.1, 1.0, N)
    idx0 = np.array([0, 1, 2])
    idx1 = np.array([3, 4, 5])

    lik0 = pg.likelihoods.Poisson()
    lik1 = pg.likelihoods.Gaussian(0.3)
    composite = pg.likelihoods.CompositeLikelihood([lik0, lik1], [idx0, idx1])

    muv, varv = pt.vector("mu"), pt.vector("var")
    comp_fn = pytensor.function([muv, varv], list(composite.predict_mean_and_var(muv, varv)))
    m0_fn = pytensor.function([muv, varv], list(lik0.predict_mean_and_var(muv, varv)))
    m1_fn = pytensor.function([muv, varv], list(lik1.predict_mean_and_var(muv, varv)))

    sm, sv = comp_fn(mu, var)
    em, ev = np.zeros(N), np.zeros(N)
    m0, v0 = m0_fn(mu[idx0], var[idx0])
    m1, v1 = m1_fn(mu[idx1], var[idx1])
    em[idx0], ev[idx0] = m0, v0
    em[idx1], ev[idx1] = m1, v1
    np.testing.assert_allclose(sm, em, atol=1e-10)
    np.testing.assert_allclose(sv, ev, atol=1e-10)


def test_heteroskedastic_sigma_subset_aligned():
    """A vector-sigma sub-likelihood works when sigma is aligned to its subset."""
    rng = np.random.default_rng(2)
    N = 8
    mu = rng.standard_normal(N)
    var = rng.uniform(0.1, 1.0, N)
    y = rng.standard_normal(N)
    idx0 = np.arange(0, 4)
    idx1 = np.arange(4, 8)

    sigma0 = rng.uniform(0.2, 0.6, idx0.size)  # length matches the subset
    lik0 = pg.likelihoods.Gaussian(pt.as_tensor_variable(sigma0))
    lik1 = pg.likelihoods.Gaussian(0.3)
    composite = pg.likelihoods.CompositeLikelihood([lik0, lik1], [idx0, idx1])

    yv, muv, varv = pt.vector("y"), pt.vector("mu"), pt.vector("var")
    comp_fn = pytensor.function([yv, muv, varv], composite.variational_expectation(yv, muv, varv))
    e0_fn = pytensor.function([yv, muv, varv], lik0.variational_expectation(yv, muv, varv))
    e1_fn = pytensor.function([yv, muv, varv], lik1.variational_expectation(yv, muv, varv))

    out = comp_fn(y, mu, var)
    expected = np.zeros(N)
    expected[idx0] = e0_fn(y[idx0], mu[idx0], var[idx0])
    expected[idx1] = e1_fn(y[idx1], mu[idx1], var[idx1])
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_rejects_non_partition():
    """Index arrays that do not partition range(N) are rejected."""
    lik = pg.likelihoods.Gaussian(0.3)
    with pytest.raises(ValueError):
        pg.likelihoods.CompositeLikelihood([lik, lik], [np.array([0, 1]), np.array([1, 2])])


def test_vgp_with_composite_likelihood_trains():
    """VGP with a composite likelihood (Gaussian + Student-t) trains."""
    rng = np.random.default_rng(3)
    N = 40
    X = np.sort(rng.uniform(-3, 3, N))[:, None]
    y = np.sin(X.ravel()) + 0.15 * rng.standard_normal(N)
    idx0 = np.arange(0, 20)
    idx1 = np.arange(20, N)
    y[idx1[:3]] += 4.0  # outliers in the Student-t subset

    lik = pg.likelihoods.CompositeLikelihood(
        [pg.likelihoods.Gaussian(0.2), pg.likelihoods.StudentT(nu=3.0, sigma=0.2)],
        [idx0, idx1],
    )
    vp = pg.gp.init_vgp_params(N)
    with pm.Model() as model:
        ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
        eta = pm.Exponential("eta", lam=1.0)
        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        vgp = pg.gp.VGP(kernel=kernel, likelihood=lik, variational_params=vp)

    X_var, y_var = pt.matrix("X"), pt.vector("y")
    train_step, _, _ = pg.optim.compile_training_step(
        lambda gp, X, y: pg.objectives.vgp_elbo(gp, X, y).elbo,
        vgp,
        X_var,
        y_var,
        model=model,
        extra_vars=vp.extra_vars,
        extra_init=vp.extra_init,
        learning_rate=1e-2,
    )
    losses = [float(train_step(X, y)) for _ in range(300)]
    assert losses[-1] < losses[0], "VGP + CompositeLikelihood loss should decrease"
