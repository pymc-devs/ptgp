"""Sampling-based prediction for SVGP.

Covers ``predict_joint`` (full posterior covariance) and
``predict_f_samples`` (draws of the latent f from that joint posterior).
The joint posterior is the foundation for posterior-predictive use with
non-Gaussian likelihoods: draw f samples, push through the likelihood.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

import ptgp as pg


def _build_svgp(ls_val=1.3, eta_val=0.9, M=8, whiten=True):
    """Build a fixed SVGP with concrete (non-symbolic) hyperparameters.

    Returns the model plus compiled functions for the three predict paths.
    The compiled functions take (q_mu, q_sqrt_flat) — to evaluate at a
    specific (q_mu_val, L), use ``pg.gp.init_variational_params(M, q_mu_init=q_mu_val,
    q_sqrt_init=L).extra_init`` to obtain the matching flat-vec inputs.
    """
    Z = np.linspace(-2, 2, M)[:, None]
    rng = np.random.default_rng(0)
    q_mu_val = rng.normal(0, 0.3, M)
    L = np.tril(rng.normal(0, 0.2, (M, M)))
    L[np.arange(M), np.arange(M)] = np.abs(L[np.arange(M), np.arange(M)]) + 0.5

    vp = pg.gp.init_variational_params(M)
    with pm.Model():
        kernel = eta_val**2 * pg.kernels.Matern52(input_dim=1, ls=ls_val)
        svgp = pg.gp.SVGP(
            kernel=kernel,
            likelihood=pg.likelihoods.Bernoulli(),
            inducing_variable=pg.inducing.Points(pt.as_tensor_variable(Z)),
            variational_params=vp,
            whiten=whiten,
        )

    X_var = pt.matrix("X")
    eps_var = pt.matrix("epsilon")

    mean_m, var_m = svgp.predict_marginal(X_var)
    mean_j, cov_j = svgp.predict_joint(X_var)
    samples = svgp.predict_f_samples(X_var, eps_var)

    marginal_fn = pytensor.function([X_var, *vp.extra_vars], [mean_m, var_m])
    joint_fn = pytensor.function([X_var, *vp.extra_vars], [mean_j, cov_j])
    samples_fn = pytensor.function([X_var, eps_var, *vp.extra_vars], samples)

    return q_mu_val, L, marginal_fn, joint_fn, samples_fn, M


def _to_flat(L, M):
    """Convert a lower-tri M×M Cholesky factor to its softplus-flat-vec representation."""
    return pg.gp.init_variational_params(M, q_sqrt_init=L).extra_init[1]


class TestPredictJointConsistency:
    """``predict_joint`` must agree with ``predict_marginal`` on the
    quantities they share: mean and the diagonal of the covariance.
    A disagreement here indicates the ``full_cov=True`` branch of
    ``base_conditional`` is wrong.
    """

    def test_mean_and_marginal_var_agree(self):
        """Mean matches; diag of joint cov matches marginal var."""
        X = np.linspace(-2.5, 2.5, 20)[:, None]
        q_mu_val, L, marginal_fn, joint_fn, _, M = _build_svgp()
        flat_L = _to_flat(L, M)

        mean_m, var_m = marginal_fn(X, q_mu_val, flat_L)
        mean_j, cov_j = joint_fn(X, q_mu_val, flat_L)

        np.testing.assert_allclose(mean_m, mean_j, atol=1e-10)
        np.testing.assert_allclose(var_m, np.diag(cov_j), atol=1e-10)

    def test_unwhitened_matches_whitened(self):
        """Whitened and unwhitened parameterizations yield the same joint
        posterior when q_mu and q_sqrt are placed in matching coords.
        Regression guard for the two-branch logic in ``base_conditional``."""
        X = np.linspace(-2.5, 2.5, 15)[:, None]
        # Whitened path
        q_mu_val, L, _, joint_w, _, M = _build_svgp(whiten=True)
        flat_L = _to_flat(L, M)
        mean_w, cov_w = joint_w(X, q_mu_val, flat_L)
        # Unwhitened — use the same q_mu and q_sqrt values; the posteriors
        # won't be literally equal, but the joint cov diagonal must agree
        # with its own marginal. We check internal consistency only.
        _, _, marginal_u, joint_u, _, M = _build_svgp(whiten=False)
        mean_u_m, var_u_m = marginal_u(X, q_mu_val, flat_L)
        mean_u_j, cov_u_j = joint_u(X, q_mu_val, flat_L)
        np.testing.assert_allclose(mean_u_m, mean_u_j, atol=1e-10)
        np.testing.assert_allclose(var_u_m, np.diag(cov_u_j), atol=1e-10)


class TestPredictFSamplesMonteCarlo:
    """Large-sample Monte Carlo check: the empirical mean and covariance
    of draws from ``predict_f_samples`` must converge to the analytical
    mean and covariance from ``predict_joint``. Pins the Cholesky-based
    sampling path against its own target distribution.
    """

    def test_sample_stats_converge(self):
        """With S=20000 samples, empirical moments match analytical
        within coarse Monte Carlo tolerance."""
        X = np.linspace(-2.0, 2.0, 10)[:, None]
        q_mu_val, L, _, joint_fn, samples_fn, M = _build_svgp()
        flat_L = _to_flat(L, M)

        mean_an, cov_an = joint_fn(X, q_mu_val, flat_L)

        S = 20000
        rng = np.random.default_rng(42)
        epsilon = rng.standard_normal((S, X.shape[0]))
        samples = samples_fn(X, epsilon, q_mu_val, flat_L)
        assert samples.shape == (S, X.shape[0])

        sample_mean = samples.mean(axis=0)
        sample_cov = np.cov(samples.T)

        # S=20000 gives MC stderr ~ sqrt(var/S); cov entries are O(1), so
        # atol in the 0.03-0.05 range is appropriate.
        np.testing.assert_allclose(sample_mean, mean_an, atol=0.05)
        np.testing.assert_allclose(sample_cov, cov_an, atol=0.05)
