from collections import namedtuple

import pytensor.tensor as pt

NNKLTerms = namedtuple("NNKLTerms", ["kl", "logdet", "trace", "mahal"])


def gauss_kl(q_mu, q_sqrt, K=None):
    """KL divergence KL[q || p] between multivariate Gaussians.

    q(x) = N(q_mu, q_sqrt @ q_sqrt.T)
    p(x) = N(0, K)   if K is provided (unwhitened)
    p(x) = N(0, I)    if K is None (whitened)

    Parameters
    ----------
    q_mu : tensor, shape (M,)
        Variational mean.
    q_sqrt : tensor, shape (M, M)
        Lower-triangular Cholesky factor of variational covariance.
    K : tensor, shape (M, M), optional
        Prior covariance. If None, prior is N(0, I).

    Returns
    -------
    scalar
        KL divergence.
    """
    M = q_mu.shape[0].astype(q_mu.dtype)
    q_cov = q_sqrt @ q_sqrt.T

    if K is None:
        # Whitened: KL[N(q_mu, q_cov) || N(0, I)]
        # = 0.5 * (tr(q_cov) + q_mu.T @ q_mu - M - log|q_cov|)
        trace = pt.trace(q_cov)
        mahal = q_mu @ q_mu
        sign, logdet = pt.linalg.slogdet(q_cov)
        return 0.5 * (trace + mahal - M - logdet)
    else:
        # Unwhitened: KL[N(q_mu, q_cov) || N(0, K)]
        # = 0.5 * (tr(K^{-1} q_cov) + q_mu.T @ K^{-1} @ q_mu - M - log|q_cov| + log|K|)
        K_inv = pt.linalg.inv(K)
        trace = pt.trace(K_inv @ q_cov)
        mahal = q_mu @ K_inv @ q_mu
        sign_q, logdet_q = pt.linalg.slogdet(q_cov)
        sign_K, logdet_K = pt.linalg.slogdet(K)
        return 0.5 * (trace + mahal - M - logdet_q + logdet_K)


def gauss_kl_structured(q_mu, q_sqrt, K_solve, K_logdet):
    """Unwhitened KL with structured prior.

    K_solve(rhs) returns K^{-1} @ rhs for rhs of shape (M, K).
    K_logdet is a scalar tensor with log|K|.
    Vector q_mu is promoted internally; caller must pass an (M,) tensor.
    """
    M = q_mu.shape[0].astype(q_mu.dtype)
    Kinv_qsqrt = K_solve(q_sqrt)
    trace = pt.sum(Kinv_qsqrt * q_sqrt)
    Kinv_qmu = K_solve(q_mu[:, None])[:, 0]
    mahal = q_mu @ Kinv_qmu
    _, logdet_q = pt.linalg.slogdet(q_sqrt @ q_sqrt.T)
    return 0.5 * (trace + mahal - M - logdet_q + K_logdet)


def nn_kl_terms(b, F, mean_local, cov_local, prior_mean_local, log_sd_cond):
    """Per-point KL terms of a Gaussian q against a nearest-neighbor (Vecchia) prior.

    The prior factorizes as ``prod_j N(u_j | mu_j + b_j^T (u_n - mu_n), F_j)``.
    With ``a_j = [1, -b_j]`` and ``r_j = a_j^T (u_local - mu_local)``,

        KL_j = -1/2 - log_sd_cond_j + 1/2 log F_j + E_q[r_j^2] / (2 F_j)
        E_q[r_j^2] = (a_j^T (m_local - mu_local))^2 + a_j^T S_local a_j

    Summed over all points this is the exact KL of q against the prior, so a
    uniform minibatch of terms scaled by ``M / batch_size`` is unbiased. For
    a mean-field q this is Eq 28 of Wu, Pleiss & Cunningham (2022,
    arXiv:2202.01694) and GPyTorch's ``NNVariationalStrategy._stochastic_kl_helper``.

    Parameters
    ----------
    b : tensor, shape (B, K)
        Neighbor weights, zero on padded slots.
    F : tensor, shape (B,)
        Conditional variances.
    mean_local : tensor, shape (B, K + 1)
        q means of ``[u_j, u_n(j)]``.
    cov_local : tensor, shape (B, K + 1, K + 1)
        q covariance of ``[u_j, u_n(j)]``.
    prior_mean_local : tensor, shape (B, K + 1)
        Prior means of ``[u_j, u_n(j)]``.
    log_sd_cond : tensor, shape (B,)
        Log of the diagonal entry of q's Cholesky factor at ``j``, so that the
        terms sum to the entropy of q.

    Returns
    -------
    NNKLTerms
        ``kl`` (B,) and its parts ``logdet``, ``trace``, ``mahal`` (B,), with
        ``kl = logdet + trace + mahal``.
    """
    a = pt.concatenate([pt.ones_like(F)[:, None], -b], axis=1)
    resid = pt.sum(a * (mean_local - prior_mean_local), axis=1)
    quad = pt.sum(a * pt.sum(cov_local * a[:, None, :], axis=2), axis=1)
    logdet = 0.5 * pt.log(F) - log_sd_cond
    trace = 0.5 * quad / F - 0.5
    mahal = 0.5 * resid**2 / F
    return NNKLTerms(kl=logdet + trace + mahal, logdet=logdet, trace=trace, mahal=mahal)
