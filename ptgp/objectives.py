from collections import namedtuple

import numpy as np
import pytensor.tensor as pt

from ptgp import assume

MLLTerms = namedtuple("MLLTerms", ["mll", "fit", "logdet"])
ELBOTerms = namedtuple("ELBOTerms", ["elbo", "var_exp", "kl"])
CollapsedELBOTerms = namedtuple(
    "CollapsedELBOTerms", ["elbo", "fit", "trace_penalty", "nystrom_residual"]
)
FITCTerms = namedtuple("FITCTerms", ["fitc", "fit", "logdet"])


# Diagonal jitter added to Kuu before Cholesky / inversion, to keep it PSD
# under floating-point noise. Matches GPflow / GPJax / PyMC defaults of 1e-6.
_DEFAULT_JITTER = 1e-6


def marginal_log_likelihood(gp, X, y):
    """Exact GP log marginal likelihood.

    log p(y|X, theta) = log N(y; m(X), K(X,X) + sigma^2 I)

    Parameters
    ----------
    gp : GP
        Exact GP model with kernel, mean function, and likelihood.
    X : tensor, shape (N, D)
    y : tensor, shape (N,)

    Returns
    -------
    scalar
        Log marginal likelihood.
    """
    mu = gp.mean(X)
    K = gp.kernel(X) + gp.likelihood.sigma**2 * pt.eye(X.shape[0])

    diff = y - mu
    sign, logdet_K = pt.linalg.slogdet(K)
    K_inv = pt.linalg.inv(K)
    N = X.shape[0]

    fit = -0.5 * (diff @ K_inv @ diff + N * pt.log(2.0 * pt.pi))
    logdet = -0.5 * logdet_K
    return MLLTerms(mll=fit + logdet, fit=fit, logdet=logdet)


def elbo(svgp, X, y, n_data=None):
    """SVGP evidence lower bound.

    ELBO = E_{q(f)}[log p(y|f)] - KL[q(u) || p(u)]

    Scaled by n_data / batch_size for minibatch training.

    Parameters
    ----------
    svgp : SVGP
        Stochastic variational GP model.
    X : tensor, shape (batch_size, D)
    y : tensor, shape (batch_size,)
    n_data : int, optional
        Total number of data points. If None, no scaling is applied.

    Returns
    -------
    scalar
        ELBO value.
    """
    fmean, fvar = svgp.predict_marginal(X)

    if n_data is not None:
        batch_size = X.shape[0]
        scale = n_data / batch_size
    else:
        scale = 1.0

    var_exp = scale * pt.sum(svgp.likelihood.variational_expectation(y, fmean, fvar))
    kl = svgp.prior_kl()
    return ELBOTerms(elbo=var_exp - kl, var_exp=var_exp, kl=kl)


def collapsed_elbo(vfe, X, y):
    """VFE/SGPR collapsed ELBO (Titsias' bound), unified for scalar and callable sigma.

    When ``vfe.likelihood.sigma`` is a scalar tensor the formulation reduces exactly
    to the classic homoskedastic bound.  When it is a callable ``X -> σ_vec`` the
    full heteroskedastic Woodbury factorisation is used:

        B = A / σ[None, :]   (M × N, each column divided by σᵢ)
        inner = I + B Bᵀ     (eigenvalues ≥ 1, well-conditioned)

    The two paths are mathematically equivalent for constant σ (verified by
    substitution: quad, logdet_cov, and trace_penalty all coincide).

    Parameters
    ----------
    vfe : VFE
        VFE sparse GP model.
    X : tensor, shape (N, D)
    y : tensor, shape (N,)

    Returns
    -------
    CollapsedELBOTerms
    """
    N = X.shape[0]
    Z = vfe.inducing_variable.Z
    M = Z.shape[0]

    mu = vfe.mean(X)
    Kff_diag = vfe.kernel.diag(X)
    Kuf = vfe.kernel(Z, X)  # M × N
    Kuu = vfe.kernel(Z)  # M × M
    Kuu = assume(
        Kuu + _DEFAULT_JITTER * pt.eye(M, dtype=Kuu.dtype),
        positive_definite=True,
        symmetric=True,
    )

    Lu = pt.linalg.cholesky(Kuu)
    A = pt.linalg.solve_triangular(Lu, Kuf, lower=True)  # M × N
    Q_diag = pt.sum(A * A, axis=0)  # N

    sigma_raw = vfe.likelihood.sigma
    sigma_vec = sigma_raw(X) if callable(sigma_raw) else sigma_raw * pt.ones(N, dtype=Kuu.dtype)
    sigma2_vec = sigma_vec**2

    diff = y - mu
    w = diff / sigma_vec  # noise-whitened residuals, N
    B = A / sigma_vec[None, :]  # M × N, column-rescaled
    inner = pt.eye(M, dtype=Kuu.dtype) + B @ B.T  # eigenvalues ≥ 1
    inner = assume(inner, positive_definite=True, symmetric=True)

    Bw = B @ w
    quad = pt.dot(w, w) - Bw @ pt.linalg.inv(inner) @ Bw

    _, logdet_inner = pt.linalg.slogdet(inner)
    logdet_cov = pt.sum(pt.log(sigma2_vec)) + logdet_inner

    fit = -0.5 * (quad + logdet_cov + N * pt.log(2.0 * pt.pi))
    nystrom_residual = pt.sum(Kff_diag - Q_diag)
    trace_penalty = -0.5 * pt.dot(Kff_diag - Q_diag, 1.0 / sigma2_vec)
    return CollapsedELBOTerms(
        elbo=fit + trace_penalty,
        fit=fit,
        trace_penalty=trace_penalty,
        nystrom_residual=nystrom_residual,
    )


def fitc_log_marginal_likelihood(vfe, X, y):
    """FITC (Fully Independent Training Conditional) approximate log marginal likelihood.

    Unlike ``collapsed_elbo``, FITC is not a lower bound — it approximates the
    log marginal likelihood using the true per-point diagonal rather than the
    Nystrom diagonal throughout. The FITC covariance is::

        K_fitc = Q + diag(ν),   ν_i = Kff_ii - Q_ii + σ²

    where ``Q = Kuf.T @ inv(Kuu) @ Kuf``. Each ``ν_i ≥ σ² > 0``, so ``K_fitc``
    is always positive definite. The per-point correction makes the marginal
    variance of each ``f_i`` exact (not just its Nystrom approximation).

    Factorisation
    -------------
    Let ``Lu = chol(Kuu)`` and ``A = Lu^{-1} Kuf`` (M × N). Then
    ``Q_ii = sum(A[:, i]**2)``, ``ν_i = Kff_ii - Q_ii + σ²``, and by the
    Woodbury identity and matrix determinant lemma::

        K_fitc^{-1} = diag(ν⁻¹) - diag(ν⁻¹) A^T B^{-1} A diag(ν⁻¹)
        log|K_fitc| = Σ log(ν_i) + log|B|

    where ``B = I + A diag(ν⁻¹) A^T`` (M × M) has eigenvalues ≥ 1 and is
    therefore well-conditioned regardless of σ² or the kernel scale.

    Parameters
    ----------
    vfe : VFE
        VFE sparse GP model. FITC uses the same inducing-variable structure as VFE.
    X : tensor, shape (N, D)
    y : tensor, shape (N,)

    Returns
    -------
    FITCTerms
        ``fitc`` — FITC approximate log marginal likelihood (fit + logdet).
        ``fit`` — quadratic term: ``-0.5 * (y^T K_fitc^{-1} y + N log 2π)``.
        ``logdet`` — log-determinant term: ``-0.5 log|K_fitc|``.
    """
    sigma2 = vfe.likelihood.sigma**2
    N = X.shape[0]
    Z = vfe.inducing_variable.Z
    M = Z.shape[0]

    mu = vfe.mean(X)
    Kff_diag = vfe.kernel.diag(X)
    Kuf = vfe.kernel(Z, X)  # M × N
    Kuu = vfe.kernel(Z)  # M × M
    Kuu = assume(
        Kuu + _DEFAULT_JITTER * pt.eye(M, dtype=Kuu.dtype),
        positive_definite=True,
        symmetric=True,
    )

    Lu = pt.linalg.cholesky(Kuu)
    A = pt.linalg.solve_triangular(Lu, Kuf, lower=True)  # M × N
    Q_diag = pt.sum(A * A, axis=0)  # N

    # Per-point FITC variance: true marginal minus Nystrom approx plus noise.
    # Guaranteed ≥ σ² > 0 because Kff_ii ≥ Q_ii (Kff - Q is PSD).
    nu = Kff_diag - Q_diag + sigma2  # N

    diff = y - mu
    beta = diff / nu  # N
    alpha = A @ beta  # M

    # B has eigenvalues ≥ 1 (A diag(ν⁻¹) A^T is PSD), so it is well-conditioned.
    B = pt.eye(M, dtype=Kuu.dtype) + (A / nu[None, :]) @ A.T
    B = assume(B, positive_definite=True, symmetric=True)

    quad = pt.sum(diff * beta) - alpha @ pt.linalg.inv(B) @ alpha

    _, logdet_B = pt.linalg.slogdet(B)
    logdet_Kfitc = pt.sum(pt.log(nu)) + logdet_B

    fit = -0.5 * (quad + N * pt.log(2.0 * pt.pi))
    logdet = -0.5 * logdet_Kfitc
    return FITCTerms(fitc=fit + logdet, fit=fit, logdet=logdet)


def dpp_regularizer(vfe, jitter=_DEFAULT_JITTER):
    """Determinantal Point Process repulsive regularizer for inducing points.

    Returns ``log det K(Z, Z)``, which is large when the inducing points are
    spread out (diverse) and goes to ``-inf`` as any two points collapse
    together. Adding a positive multiple of this to ``collapsed_elbo`` makes
    the effective ``logdet_Kuu`` coefficient larger than the 0.5 that comes
    from the Woodbury derivation, increasing repulsion between Z points.

    Note: adding this term makes the objective a *regularized* objective, not
    a valid evidence lower bound. Use it when numerical stability of Kuu
    matters more than a tight bound -- for example, when jointly optimizing Z
    with the hyperparameters.

    Parameters
    ----------
    vfe : VFE
        VFE sparse GP model.
    jitter : float, optional
        Diagonal jitter added to K(Z, Z) before computing the log-determinant.
        Should match the jitter used in ``collapsed_elbo``.

    Returns
    -------
    scalar
        ``log det (K(Z, Z) + jitter * I)``.

    Examples
    --------
    Make the total ``logdet_Kuu`` coefficient 1.0 instead of 0.5::

        def objective(vfe, X, y):
            return collapsed_elbo(vfe, X, y).elbo + 0.5 * dpp_regularizer(vfe)

    Tune the strength via a variable::

        strength = 1.0


        def objective(vfe, X, y):
            return collapsed_elbo(vfe, X, y).elbo + strength * dpp_regularizer(vfe)
    """
    Z = vfe.inducing_variable.Z
    M = Z.shape[0]
    Kuu = vfe.kernel(Z)
    Kuu = assume(
        Kuu + jitter * pt.eye(M, dtype=Kuu.dtype),
        positive_definite=True,
        symmetric=True,
    )
    _, logdet_Kuu = pt.linalg.slogdet(Kuu)
    return logdet_Kuu


VFEDiagnostics = namedtuple(
    "VFEDiagnostics",
    ["elbo", "fit", "trace_penalty", "nystrom_residual", "sigma", "fit_per_n", "excess_fit_per_n"],
)


def vfe_diagnostics(vfe, X, y):
    """Collapsed ELBO terms plus sigma and two normalised fit metrics.

    Returns a ``VFEDiagnostics`` namedtuple of symbolic TensorVariables,
    suitable for use with :func:`ptgp.optim.compile_scipy_diagnostics`.

    Fields
    ------
    elbo, fit, trace_penalty
        Direct from :func:`collapsed_elbo`.
    nystrom_residual
        ``tr(Kff - Qff) / N`` — per-point Nyström approximation error.
    sigma
        Likelihood noise (constrained space).
    fit_per_n
        ``fit / N`` — scale-invariant data fit.
    excess_fit_per_n
        ``fit_per_n + 0.5 * log(2π σ²)`` — how much better than noise floor.
        Goes to zero when the model fits at the noise level only.
    """
    terms = collapsed_elbo(vfe, X, y)
    N = X.shape[0]
    sigma_raw = vfe.likelihood.sigma
    sigma_vec = sigma_raw(X) if callable(sigma_raw) else sigma_raw * pt.ones(N)
    sigma_mean = pt.mean(sigma_vec)  # scalar; mean of a constant vector = that constant
    fit_per_n = terms.fit / N
    excess_fit_per_n = fit_per_n + 0.5 * pt.log(2.0 * np.pi * sigma_mean**2)
    return VFEDiagnostics(
        elbo=terms.elbo,
        fit=terms.fit,
        trace_penalty=terms.trace_penalty,
        nystrom_residual=terms.nystrom_residual / N,
        sigma=sigma_mean,
        fit_per_n=fit_per_n,
        excess_fit_per_n=excess_fit_per_n,
    )
