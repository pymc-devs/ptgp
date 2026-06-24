"""VGP (Opper-Archambeau full variational GP) tests.

The decisive test is the exact-GP recovery: with a Gaussian likelihood, the VGP
at its closed-form optimum reproduces the exact ``Unapproximated`` GP
(predictions and ELBO), which validates the ELBO, KL, training marginals, and
predict math all at once.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import scipy.optimize

import ptgp as pg

from ptgp.mean import Constant, Zero
from ptgp.optim.training import compile_scipy_objective


def _eval_kernel(kernel, X):
    """Evaluate a kernel Gram matrix numerically."""
    Xv = pt.matrix("X")
    return pytensor.function([Xv], kernel(Xv))(X)


def _oa_optimal_params(K, y, m, sigma):
    """Closed-form OA params for a Gaussian likelihood.

    The exact posterior over f has precision ``K^{-1} + sigma^{-2} I``, so
    ``lambda = 1/sigma**2`` and ``alpha = (K + sigma**2 I)^{-1} (y - m)``.
    """
    N = K.shape[0]
    alpha = np.linalg.solve(K + sigma**2 * np.eye(N), y - m)
    lam = (1.0 / sigma**2) * np.ones(N)
    return alpha, lam


def _count_data(rng, n=80):
    """1D Poisson data with a smoothly varying log-rate."""
    X = np.sort(rng.uniform(-3, 3, n))[:, None]
    rate = np.exp(0.5 * np.sin(X[:, 0]) + 0.3)
    y = rng.poisson(rate).astype(np.float64)
    return X, y


def _binary_data(rng, n=120):
    """1D Bernoulli data with the decision boundary near x=0."""
    from scipy.special import erf

    X = np.sort(rng.uniform(-3, 3, n))[:, None]
    p = 0.5 * (1.0 + erf(X[:, 0] / np.sqrt(2.0)))
    y = (rng.uniform(0, 1, n) < p).astype(np.float64)
    return X, y


class TestVGPRecoversExactGP:
    """Gaussian VGP at the closed-form optimum equals the exact GP."""

    def _run(self, mean_c):
        rng = np.random.default_rng(0)
        N = 14
        X = np.sort(rng.uniform(-2, 2, N))[:, None]
        y = np.sin(X.ravel()) + 0.1 * rng.standard_normal(N)
        Xs = np.linspace(-2.5, 2.5, 9)[:, None]
        sigma, ls, eta = 0.3, 0.7, 1.2

        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        mean = Zero() if mean_c is None else Constant(float(mean_c))
        K = _eval_kernel(kernel, X)
        m = np.zeros(N) if mean_c is None else mean_c * np.ones(N)
        alpha, lam = _oa_optimal_params(K, y, m, sigma)

        vp = pg.gp.init_vgp_params(N, alpha_init=alpha, lambda_init=lam)
        vgp = pg.gp.VGP(
            kernel=kernel,
            mean=mean,
            likelihood=pg.likelihoods.Gaussian(sigma),
            variational_params=vp,
        )
        exact = pg.gp.Unapproximated(kernel=kernel, mean=mean, sigma=sigma)

        # ELBO at the optimum equals the exact marginal log-likelihood.
        Xv, yv = pt.matrix("X"), pt.vector("y")
        elbo_fn = pytensor.function(
            [Xv, yv, *vp.extra_vars], pg.objectives.vgp_elbo(vgp, Xv, yv).elbo
        )
        mll_fn = pytensor.function(
            [Xv, yv], pg.objectives.marginal_log_likelihood(exact, Xv, yv).mll
        )
        elbo_val = float(elbo_fn(X, y, *vp.extra_init))
        mll_val = float(mll_fn(X, y))
        np.testing.assert_allclose(elbo_val, mll_val, atol=1e-6)

        # Predictions match the exact GP.
        Xn, Xt, yt = pt.matrix("Xn"), pt.matrix("Xt"), pt.vector("yt")
        vm, vv = vgp.predict_marginal(Xn, Xt)
        vgp_pred = pytensor.function([Xn, Xt, *vp.extra_vars], [vm, vv])
        em, ev = exact.predict_marginal(Xn, Xt, yt)
        exact_pred = pytensor.function([Xn, Xt, yt], [em, ev])

        vmean, vvar = vgp_pred(Xs, X, *vp.extra_init)
        emean, evar = exact_pred(Xs, X, y)
        np.testing.assert_allclose(vmean, emean, atol=1e-6)
        np.testing.assert_allclose(vvar, evar, atol=1e-6)

    def test_zero_mean(self):
        """VGP recovers the exact GP with a zero mean function."""
        self._run(mean_c=None)

    def test_nonzero_constant_mean(self):
        """VGP recovers the exact GP with a non-zero constant mean (guards centering)."""
        self._run(mean_c=1.5)


class TestVGPElboBound:
    """The VGP ELBO is a lower bound on the exact marginal likelihood."""

    def test_elbo_leq_mll(self):
        """At arbitrary feasible variational params, ELBO <= MLL."""
        rng = np.random.default_rng(1)
        N = 12
        X = np.sort(rng.uniform(0, 5, N))[:, None]
        y = np.sin(X.ravel()) + 0.1 * rng.standard_normal(N)
        sigma = 0.4
        kernel = pg.kernels.Matern52(input_dim=1, ls=1.0)

        # Feasible but non-optimal params.
        vp = pg.gp.init_vgp_params(
            N, alpha_init=0.1 * rng.standard_normal(N), lambda_init=2.0 * np.ones(N)
        )
        vgp = pg.gp.VGP(
            kernel=kernel, likelihood=pg.likelihoods.Gaussian(sigma), variational_params=vp
        )
        exact = pg.gp.Unapproximated(kernel=kernel, sigma=sigma)

        Xv, yv = pt.matrix("X"), pt.vector("y")
        elbo_val = float(
            pytensor.function([Xv, yv, *vp.extra_vars], pg.objectives.vgp_elbo(vgp, Xv, yv).elbo)(
                X, y, *vp.extra_init
            )
        )
        mll_val = float(
            pytensor.function([Xv, yv], pg.objectives.marginal_log_likelihood(exact, Xv, yv).mll)(
                X, y
            )
        )
        assert elbo_val <= mll_val + 1e-6


class TestVGPNonGaussianSmoke:
    """VGP trains end-to-end with non-Gaussian likelihoods."""

    def _smoke(self, X, y, likelihood, n_steps=300, lr=1e-2):
        N = X.shape[0]
        vp = pg.gp.init_vgp_params(N)
        with pm.Model() as model:
            ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
            eta = pm.Exponential("eta", lam=1.0)
            kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
            vgp = pg.gp.VGP(kernel=kernel, likelihood=likelihood, variational_params=vp)

        X_var, y_var = pt.matrix("X"), pt.vector("y")
        train_step, _, _ = pg.optim.compile_training_step(
            lambda gp, X, y: pg.objectives.vgp_elbo(gp, X, y).elbo,
            vgp,
            X_var,
            y_var,
            model=model,
            extra_vars=vp.extra_vars,
            extra_init=vp.extra_init,
            learning_rate=lr,
        )
        losses = [float(train_step(X, y)) for _ in range(n_steps)]
        assert losses[-1] < losses[0], "VGP loss should decrease"

    def test_poisson(self):
        """VGP + Poisson loss decreases."""
        rng = np.random.default_rng(2)
        X, y = _count_data(rng, n=80)
        self._smoke(X, y, pg.likelihoods.Poisson())

    def test_bernoulli(self):
        """VGP + Bernoulli loss decreases."""
        rng = np.random.default_rng(3)
        X, y = _binary_data(rng, n=100)
        self._smoke(X, y, pg.likelihoods.Bernoulli())


class TestVGPGradient:
    """Finite-difference gradient check of the VGP ELBO objective."""

    def test_grad_matches_finite_difference(self):
        """compile_scipy_objective's analytic gradient matches central differences."""
        rng = np.random.default_rng(4)
        N = 8
        X = np.sort(rng.uniform(-2, 2, N))[:, None]
        y = np.sin(X.ravel()) + 0.1 * rng.standard_normal(N)

        vp = pg.gp.init_vgp_params(N)
        with pm.Model() as model:
            ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
            eta = pm.Exponential("eta", lam=1.0)
            kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
            vgp = pg.gp.VGP(
                kernel=kernel, likelihood=pg.likelihoods.Poisson(), variational_params=vp
            )

        X_var, y_var = pt.matrix("X"), pt.vector("y")
        fun, theta0, _, _, _ = compile_scipy_objective(
            pg.objectives.vgp_elbo, vgp, X_var, y_var, model=model
        )

        def f(theta):
            return fun(theta, X, y)[0]

        def g(theta):
            return fun(theta, X, y)[1]

        err = scipy.optimize.check_grad(f, g, theta0, epsilon=1e-6)
        assert err < 1e-3, f"gradient mismatch {err:.2e}"
