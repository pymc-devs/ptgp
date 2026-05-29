"""SVGP tests for likelihoods that GPJax does not implement.

These tests are the structural equivalent of the GPJax cross-checks in
``tests/test_svgp.py``: evaluate the whitened-SVGP ELBO in PTGP and in
an independent reference at a fixed configuration, and require them to
match at atol=1e-5. The difference is the reference — GPJax doesn't ship
StudentT or NegativeBinomial likelihoods, so the reference ELBO here is
assembled from numpy + scipy (whitened predict by hand, scipy.integrate.quad
for the variational expectation, closed-form whitened KL).

If we ever want to drop GPJax as a test dependency, the Bernoulli and
Poisson tests in ``test_svgp.py`` could be converted to this same pattern
— the reference machinery below is likelihood-agnostic and only needs a
numpy log-prob function.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pymc as pm  # noqa: E402
import pytensor  # noqa: E402
import pytensor.tensor as pt  # noqa: E402
import scipy.linalg  # noqa: E402

from scipy import integrate  # noqa: E402
from scipy.special import gammaln  # noqa: E402

import ptgp as pg  # noqa: E402

ATOL = 1e-5


def _matern52_numpy(X1, X2, ls, eta):
    """Matern52 kernel in numpy: k(r) = eta^2 (1 + sqrt(5)r + 5r^2/3) exp(-sqrt(5)r)."""
    sqd = np.sum(X1**2, axis=-1)[:, None] + np.sum(X2**2, axis=-1)[None, :] - 2.0 * X1 @ X2.T
    r = np.sqrt(np.maximum(sqd, 0.0)) / ls
    s5 = np.sqrt(5.0)
    return eta**2 * (1.0 + s5 * r + 5.0 * r**2 / 3.0) * np.exp(-s5 * r)


def _whitened_predict(X, Z, ls, eta, q_mu, q_sqrt):
    """Whitened-SVGP marginal predictive mean and variance at X."""
    Kzz = _matern52_numpy(Z, Z, ls, eta)
    Kzx = _matern52_numpy(Z, X, ls, eta)
    K_diag = np.full(X.shape[0], eta**2)  # Matern52 diag is eta^2
    Lz = scipy.linalg.cholesky(Kzz, lower=True)
    A = scipy.linalg.solve_triangular(Lz, Kzx, lower=True)  # (M, N)
    mu_f = A.T @ q_mu
    var_f = K_diag - np.sum(A**2, axis=0) + np.sum((A.T @ q_sqrt) ** 2, axis=1)
    return mu_f, var_f


def _whitened_kl(q_mu, q_sqrt):
    """Closed-form KL[N(q_mu, q_sqrt q_sqrt^T) || N(0, I)]."""
    M = q_mu.size
    logdet_S = 2.0 * np.sum(np.log(np.abs(np.diag(q_sqrt))))
    return 0.5 * (np.sum(q_sqrt**2) + q_mu @ q_mu - M - logdet_S)


def _variational_expectation_quad(log_prob_fn, y, mu_f, var_f):
    """Sum of per-point E_{q(f_n)}[log p(y_n|f_n)] via adaptive quadrature.

    Integrates in the standard-normal z-scale: f = mu + sqrt(var) * z. The
    [-30, 30] z-range is well past the numerical support of exp(-z^2/2).
    """
    total = 0.0
    for yn, m, v in zip(y, mu_f, var_f):
        sd = np.sqrt(v)

        def integrand(z):
            return log_prob_fn(m + sd * z, yn) * np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)

        val, _ = integrate.quad(integrand, -30.0, 30.0)
        total += val
    return total


def _reference_elbo(X, y, Z, q_mu, q_sqrt, ls, eta, log_prob_fn):
    """Whitened-SVGP ELBO assembled from numpy + scipy."""
    mu_f, var_f = _whitened_predict(X, Z, ls, eta, q_mu, q_sqrt)
    ve = _variational_expectation_quad(log_prob_fn, y, mu_f, var_f)
    kl = _whitened_kl(q_mu, q_sqrt)
    return ve - kl


def _fixed_config(rng, N=40, M=8, x_range=(-2.0, 2.0)):
    """Shared fixed configuration for ELBO-match tests."""
    X = np.sort(rng.uniform(x_range[0], x_range[1], N))[:, None]
    Z = np.linspace(x_range[0], x_range[1], M)[:, None]
    q_mu = rng.normal(0, 0.3, M)
    L = np.tril(rng.normal(0, 0.2, (M, M)))
    L[np.arange(M), np.arange(M)] = np.abs(L[np.arange(M), np.arange(M)]) + 0.5
    return X, Z, q_mu, L


def _studentt_logprob(nu, sigma):
    """Return a numpy log_prob(f, y) for StudentT(y; f, sigma, nu)."""

    def logp(f, y):
        z = (y - f) / sigma
        return (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi * sigma**2)
            - 0.5 * (nu + 1.0) * np.log1p(z**2 / nu)
        )

    return logp


def _studentt_data(rng, n=80, nu=5.0, sigma=0.3):
    """1D StudentT data with smooth latent and heavy-tailed noise."""
    X = np.sort(rng.uniform(-2, 2, n))[:, None]
    f_true = np.sin(X[:, 0])
    y = f_true + sigma * rng.standard_t(nu, n)
    return X, y


class TestSVGPStudentTSmoke:
    """End-to-end sanity: SVGP + StudentT trains and recovers the latent
    signal under heavy-tailed noise. Exercises the quadrature variational
    expectation branch with a continuous, real-valued likelihood — a
    different regime from Bernoulli (bounded, binary) and Poisson (closed
    form). A failure here points at the quadrature wiring for real-valued
    likelihoods or at the StudentT log-prob itself.
    """

    def test_loss_decreases_and_signal_correlates(self):
        """Loss decreases over 400 Adam steps; predicted mean correlates
        with true latent at Pearson r > 0.8."""
        rng = np.random.default_rng(4)
        X, y = _studentt_data(rng, n=120)
        M = 12

        Z_init = np.linspace(-2, 2, M)[:, None]
        vp = pg.gp.init_variational_params(M)

        with pm.Model() as model:
            ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
            eta = pm.Exponential("eta", lam=1.0)
            sigma = pm.Exponential("sigma", lam=1.0)
            kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
            svgp = pg.gp.SVGP(
                kernel=kernel,
                likelihood=pg.likelihoods.StudentT(nu=5.0, sigma=sigma),
                inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z_init)),
                variational_params=vp,
            )

        X_var = pt.matrix("X")
        y_var = pt.vector("y")

        train_step, shared_params, shared_extras = pg.optim.compile_training_step(
            lambda gp, X, y: pg.objectives.elbo(gp, X, y).elbo,
            svgp,
            X_var,
            y_var,
            model=model,
            extra_vars=vp.extra_vars,
            extra_init=vp.extra_init,
            learning_rate=5e-2,
        )

        losses = [float(train_step(X, y)) for _ in range(400)]
        assert losses[-1] < losses[0], "SVGP+StudentT loss should decrease"

        X_new_var = pt.matrix("X_new")
        predict_fn = pg.optim.compile_predict(
            svgp,
            X_new_var,
            model,
            shared_params,
            extra_vars=vp.extra_vars,
            shared_extras=shared_extras,
            incl_lik=False,
        )
        f_mean, _ = predict_fn(X)
        f_true = np.sin(X[:, 0])
        r = float(np.corrcoef(f_mean, f_true)[0, 1])
        assert r > 0.8, f"signal correlation {r:.2f} too low"


class TestSVGPStudentTElboMatchesReference:
    """Evaluate the whitened-SVGP ELBO in PTGP and in the scipy reference
    at a fixed StudentT configuration and require the two scalars to match
    at atol=1e-5. Pins PTGP's Gauss-Hermite variational expectation for
    StudentT against adaptive quadrature, independent of any optimizer.
    """

    def _ptgp_elbo(self, X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val, nu_val, sigma_val):
        """Evaluate PTGP whitened-SVGP ELBO at the fixed configuration."""
        ls = pt.scalar("ls")
        eta = pt.scalar("eta")
        sigma = pt.scalar("sigma")
        M = q_mu_val.shape[0]
        vp = pg.gp.init_variational_params(M, q_mu_init=q_mu_val, q_sqrt_init=q_sqrt_val)
        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.StudentT(nu=nu_val, sigma=sigma),
            inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z)),
            variational_params=vp,
            whiten=True,
        )
        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        elbo_expr = pg.objectives.elbo(svgp, X_var, y_var).elbo
        fn = pytensor.function([X_var, y_var, *vp.extra_vars, ls, eta, sigma], elbo_expr)
        return float(fn(X, y, *vp.extra_init, ls_val, eta_val, sigma_val))

    def test_elbo_match(self):
        rng = np.random.default_rng(5)
        X, Z, q_mu_val, q_sqrt_val = _fixed_config(rng)
        y = np.sin(X[:, 0]) + 0.3 * rng.standard_t(5.0, X.shape[0])
        ls_val, eta_val, nu_val, sigma_val = 1.3, 0.9, 5.0, 0.3

        e_ptgp = self._ptgp_elbo(X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val, nu_val, sigma_val)
        e_ref = _reference_elbo(
            X,
            y,
            Z,
            q_mu_val,
            q_sqrt_val,
            ls_val,
            eta_val,
            _studentt_logprob(nu_val, sigma_val),
        )

        np.testing.assert_allclose(e_ptgp, e_ref, atol=ATOL)


def _negbinom_logprob(alpha):
    """Return a numpy log_prob(f, y) for NegativeBinomial(y; exp(f), alpha)."""

    def logp(f, y):
        mu = np.exp(f)
        return (
            gammaln(y + alpha)
            - gammaln(alpha)
            - gammaln(y + 1.0)
            + alpha * np.log(alpha / (alpha + mu))
            + y * np.log(mu / (alpha + mu))
        )

    return logp


def _negbinom_data(rng, n=80, alpha=2.0):
    """1D overdispersed count data: log-linear mean with NB noise."""
    X = np.sort(rng.uniform(-2, 2, n))[:, None]
    mu = np.exp(0.5 * X[:, 0] + 0.3)
    p = alpha / (alpha + mu)
    y = rng.negative_binomial(alpha, p).astype(np.float64)
    return X, y


class TestSVGPNegBinomSmoke:
    """End-to-end sanity: SVGP + NegativeBinomial trains on overdispersed
    counts and the predicted rate tracks the true log-linear mean.
    Complements the Poisson smoke test (which hits the closed-form log-link
    VE) by exercising the quadrature branch for count data.
    """

    def test_loss_decreases_and_rate_correlates(self):
        """Loss decreases over 400 Adam steps; predicted mean correlates
        with true mean at Pearson r > 0.8."""
        rng = np.random.default_rng(6)
        X, y = _negbinom_data(rng, n=120)
        M = 12

        Z_init = np.linspace(-2, 2, M)[:, None]
        vp = pg.gp.init_variational_params(M)

        with pm.Model() as model:
            ls = pm.InverseGamma("ls", alpha=2.0, beta=1.0)
            eta = pm.Exponential("eta", lam=1.0)
            alpha = pm.Exponential("alpha", lam=0.5)
            kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
            svgp = pg.gp.SVGP(
                kernel=kernel,
                likelihood=pg.likelihoods.NegativeBinomial(alpha=alpha),
                inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z_init)),
                variational_params=vp,
            )

        X_var = pt.matrix("X")
        y_var = pt.vector("y")

        train_step, shared_params, shared_extras = pg.optim.compile_training_step(
            lambda gp, X, y: pg.objectives.elbo(gp, X, y).elbo,
            svgp,
            X_var,
            y_var,
            model=model,
            extra_vars=vp.extra_vars,
            extra_init=vp.extra_init,
            learning_rate=5e-2,
        )

        losses = [float(train_step(X, y)) for _ in range(400)]
        assert losses[-1] < losses[0], "SVGP+NegativeBinomial loss should decrease"

        X_new_var = pt.matrix("X_new")
        predict_fn = pg.optim.compile_predict(
            svgp,
            X_new_var,
            model,
            shared_params,
            extra_vars=vp.extra_vars,
            shared_extras=shared_extras,
            incl_lik=True,
        )
        rate_pred, _ = predict_fn(X)
        rate_true = np.exp(0.5 * X[:, 0] + 0.3)
        r = float(np.corrcoef(rate_pred, rate_true)[0, 1])
        assert r > 0.8, f"rate correlation {r:.2f} too low"


class TestSVGPNegBinomElboMatchesReference:
    """Evaluate the whitened-SVGP ELBO in PTGP and in the scipy reference
    at a fixed NegativeBinomial configuration and require the two scalars
    to match at atol=1e-5. Pins PTGP's Gauss-Hermite variational expectation
    for NegativeBinomial against adaptive quadrature.
    """

    def _ptgp_elbo(self, X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val, alpha_val):
        """Evaluate PTGP whitened-SVGP ELBO at the fixed configuration."""
        ls = pt.scalar("ls")
        eta = pt.scalar("eta")
        alpha = pt.scalar("alpha")
        M = q_mu_val.shape[0]
        vp = pg.gp.init_variational_params(M, q_mu_init=q_mu_val, q_sqrt_init=q_sqrt_val)
        kernel = eta**2 * pg.kernels.Matern52(input_dim=1, ls=ls)
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.NegativeBinomial(alpha=alpha),
            inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z)),
            variational_params=vp,
            whiten=True,
        )
        X_var = pt.matrix("X")
        y_var = pt.vector("y")
        elbo_expr = pg.objectives.elbo(svgp, X_var, y_var).elbo
        fn = pytensor.function([X_var, y_var, *vp.extra_vars, ls, eta, alpha], elbo_expr)
        return float(fn(X, y, *vp.extra_init, ls_val, eta_val, alpha_val))

    def test_elbo_match(self):
        rng = np.random.default_rng(7)
        X, Z, q_mu_val, q_sqrt_val = _fixed_config(rng)
        alpha_val = 2.0
        mu_true = np.exp(0.5 * X[:, 0] + 0.3)
        p = alpha_val / (alpha_val + mu_true)
        y = rng.negative_binomial(alpha_val, p).astype(np.float64)
        ls_val, eta_val = 1.3, 0.9

        e_ptgp = self._ptgp_elbo(X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val, alpha_val)
        e_ref = _reference_elbo(
            X,
            y,
            Z,
            q_mu_val,
            q_sqrt_val,
            ls_val,
            eta_val,
            _negbinom_logprob(alpha_val),
        )

        np.testing.assert_allclose(e_ptgp, e_ref, atol=ATOL)
