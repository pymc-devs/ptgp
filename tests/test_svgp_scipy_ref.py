"""SVGP tests for likelihoods cross-checked against an independent reference.

Evaluate the whitened-SVGP ELBO in PTGP and in a numpy+scipy reference at a
fixed configuration, and require them to match at atol=1e-5. The reference
machinery lives in ``tests/_svgp_ref.py``; it is likelihood-agnostic and only
needs a numpy log-prob function, so StudentT and NegativeBinomial (which GPJax
does not ship) are checked the same way as Bernoulli and Poisson.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from scipy.special import gammaln

import ptgp as pg

from tests._svgp_ref import fixed_config, reference_elbo

ATOL = 1e-5


# ---- StudentT -------------------------------------------------------------


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
        X, Z, q_mu_val, q_sqrt_val = fixed_config(rng)
        y = np.sin(X[:, 0]) + 0.3 * rng.standard_t(5.0, X.shape[0])
        ls_val, eta_val, nu_val, sigma_val = 1.3, 0.9, 5.0, 0.3

        e_ptgp = self._ptgp_elbo(X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val, nu_val, sigma_val)
        e_ref = reference_elbo(
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


# ---- NegativeBinomial -----------------------------------------------------


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
        X, Z, q_mu_val, q_sqrt_val = fixed_config(rng)
        alpha_val = 2.0
        mu_true = np.exp(0.5 * X[:, 0] + 0.3)
        p = alpha_val / (alpha_val + mu_true)
        y = rng.negative_binomial(alpha_val, p).astype(np.float64)
        ls_val, eta_val = 1.3, 0.9

        e_ptgp = self._ptgp_elbo(X, y, Z, q_mu_val, q_sqrt_val, ls_val, eta_val, alpha_val)
        e_ref = reference_elbo(
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
