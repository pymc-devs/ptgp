import dataclasses

import numpy as np
import pytensor.assumptions as pta
import pytensor.tensor as pt

from ptgp.conditionals import conditional_unwhitened, conditional_whitened
from ptgp.kl import gauss_kl, gauss_kl_structured
from ptgp.mean import Zero


def _softplus_lower_triangular(flat, M):
    """Build M×M lower-triangular matrix from a flat vector, with softplus on the diagonal.

    The flat vector has length M·(M+1)/2 in row-major lower-triangular layout
    (matching ``np.tril_indices(M)``). Diagonal entries are passed through
    ``pt.softplus`` so the result is a lower-triangular Cholesky factor with
    strictly positive diagonal at every optimizer step.
    """
    rows, cols = np.tril_indices(M)
    L = pt.set_subtensor(pt.zeros((M, M))[rows, cols], flat)
    diag_idx = np.arange(M)
    L = pt.set_subtensor(L[diag_idx, diag_idx], pt.softplus(L[diag_idx, diag_idx]))
    return L


def _matrix_to_softplus_flat_init(L_init, M):
    """Inverse of ``_softplus_lower_triangular`` at init time.

    Given an M×M lower-triangular Cholesky factor with strictly positive
    diagonal, return the flat vector ``flat`` such that
    ``_softplus_lower_triangular(flat, M)`` evaluates to ``L_init``.
    """
    if L_init.shape != (M, M):
        raise ValueError(f"L_init must have shape ({M}, {M}); got {L_init.shape}.")
    diag_vals = np.diag(L_init)
    if np.any(diag_vals <= 0):
        raise ValueError(
            "L_init must have strictly positive diagonal (it represents a "
            "Cholesky factor under softplus parameterization)."
        )
    rows, cols = np.tril_indices(M)
    flat = L_init[rows, cols].astype(np.float64)
    diag_positions = np.cumsum(np.arange(1, M + 1)) - 1
    flat[diag_positions] = np.log(np.expm1(diag_vals.astype(np.float64)))
    return flat


@dataclasses.dataclass
class VariationalParams:
    """Symbolic variational parameters for SVGP.

    Attributes
    ----------
    q_mu : TensorVariable
        Symbolic variational mean, shape ``(M,)``. Pass to ``SVGP`` via
        ``variational_params=...``.
    q_sqrt : TensorVariable
        Symbolic Cholesky factor of the variational covariance, shape ``(M, M)``,
        annotated ``lower_triangular=True``. Pass to ``SVGP`` via
        ``variational_params=...``.
    extra_vars : list of TensorVariable
        Trainable leaf variables underlying ``q_mu`` and ``q_sqrt``. Pass to
        ``compile_training_step`` / ``compile_scipy_objective`` as ``extra_vars``.
    extra_init : list of ndarray
        Initial values for ``extra_vars``, in matching order. Pass to
        ``compile_training_step`` / ``compile_scipy_objective`` as ``extra_init``.
    """

    q_mu: object
    q_sqrt: object
    extra_vars: list = dataclasses.field(default_factory=list)
    extra_init: list = dataclasses.field(default_factory=list)


def init_variational_params(M, q_mu_init=None, q_sqrt_init=None):
    """Build symbolic variational parameters with GPJax-style parameterization.

    ``q_sqrt`` is stored as a flat vector of length ``M·(M+1)/2`` and
    materialised by filling a lower-triangular matrix and applying
    ``pt.softplus`` to the diagonal — guarantees a true Cholesky factor
    (lower-triangular with strictly positive diagonal) at every optimizer step.
    ``q_mu`` is stored as a length-``M`` flat vector.

    Parameters
    ----------
    M : int
        Number of inducing points.
    q_mu_init : ndarray of shape (M,), optional
        Initial variational mean. Defaults to zeros.
    q_sqrt_init : ndarray of shape (M, M), optional
        Initial Cholesky factor. Must be lower-triangular with positive diagonal.
        Defaults to the identity matrix.

    Returns
    -------
    VariationalParams

    Examples
    --------
    >>> vp = init_variational_params(M=8)
    >>> svgp = SVGP(kernel=..., likelihood=..., inducing_variable=..., variational_params=vp)
    >>> train_step, _, _ = compile_training_step(
    ...     elbo,
    ...     svgp,
    ...     X,
    ...     y,
    ...     model=model,
    ...     extra_vars=vp.extra_vars,
    ...     extra_init=vp.extra_init,
    ... )
    """
    if q_mu_init is None:
        q_mu_init = np.zeros(M, dtype=np.float64)
    else:
        q_mu_init = np.asarray(q_mu_init, dtype=np.float64)
        if q_mu_init.shape != (M,):
            raise ValueError(f"q_mu_init must have shape ({M},); got {q_mu_init.shape}.")
    if q_sqrt_init is None:
        q_sqrt_init = np.eye(M, dtype=np.float64)
    else:
        q_sqrt_init = np.asarray(q_sqrt_init, dtype=np.float64)

    n_lower = M * (M + 1) // 2
    q_mu = pt.vector("q_mu", shape=(M,), dtype="float64")
    q_sqrt_flat = pt.vector("q_sqrt_flat", shape=(n_lower,), dtype="float64")
    q_sqrt = pta.assume(
        _softplus_lower_triangular(q_sqrt_flat, M),
        lower_triangular=True,
    )
    flat_init = _matrix_to_softplus_flat_init(q_sqrt_init, M)
    return VariationalParams(
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        extra_vars=[q_mu, q_sqrt_flat],
        extra_init=[q_mu_init, flat_init],
    )


class SVGP:
    """Stochastic Variational Gaussian Process.

    Parameters
    ----------
    kernel : Kernel
        Covariance function.
    mean : callable, optional
        Mean function (default: Zero()).
    likelihood : Likelihood
        Observation likelihood.
    inducing_variable : InducingVariables
        Inducing point locations (or structured inducing variables).
    variational_params : VariationalParams
        Symbolic variational parameters and their backing trainable
        leaves. Construct via :func:`init_variational_params` (recommended)
        or build a :class:`VariationalParams` directly for custom
        parameterizations. ``vp.extra_vars`` / ``vp.extra_init`` should be
        passed to ``compile_training_step``.
    whiten : bool
        If True, use whitened variational parameterization (default True).
    """

    def __init__(
        self,
        kernel,
        mean=None,
        likelihood=None,
        inducing_variable=None,
        variational_params=None,
        whiten=True,
    ):
        if variational_params is None:
            raise ValueError(
                "SVGP requires variational_params. Construct via "
                "ptgp.gp.init_variational_params(M) and pass as "
                "variational_params=..., or build a VariationalParams directly."
            )
        self.kernel = kernel
        self.mean = mean if mean is not None else Zero()
        self.likelihood = likelihood
        self.inducing_variable = inducing_variable
        self.whiten = whiten
        self.variational_params = variational_params
        self.q_mu = variational_params.q_mu
        self.q_sqrt = variational_params.q_sqrt

    def predict_marginal(self, X, incl_lik=False):
        """Posterior marginal mean and variance at each point in X.

        Returns the per-point posterior — correlations between test
        points are discarded. Use ``predict_joint`` for the full (N, N)
        covariance or ``predict_f_samples`` to draw smooth function samples.

        Parameters
        ----------
        X : tensor, shape (N, D)
        incl_lik : bool
            If True, push through the likelihood's predictive mean/var
            (observation-space uncertainty).

        Returns
        -------
        mean : tensor, shape (N,)
        var : tensor, shape (N,)
        """
        ind, kernel = self.inducing_variable, self.kernel
        Kmn = ind.K_uf(kernel, X)
        Knn_diag = kernel.diag(X)
        if self.whiten:
            A_w = ind.Kuu_sqrt_solve(kernel, Kmn)
            fmean, fvar = conditional_whitened(A_w, Knn_diag, self.q_mu, self.q_sqrt)
        else:
            A = ind.Kuu_solve(kernel, Kmn)
            fmean, fvar = conditional_unwhitened(A, Kmn, Knn_diag, self.q_mu, self.q_sqrt)
        fmean = fmean + self.mean(X)
        if incl_lik:
            return self.likelihood.predict_mean_and_var(fmean, fvar)
        return fmean, fvar

    def predict_joint(self, X):
        """Posterior joint mean and full covariance of the latent f at X.

        The diagonal of the returned covariance equals the variance from
        ``predict_marginal``; the off-diagonals capture correlations
        between test points, which are needed to draw smooth samples.

        Parameters
        ----------
        X : tensor, shape (N, D)

        Returns
        -------
        mean : tensor, shape (N,)
        cov : tensor, shape (N, N)
        """
        ind, kernel = self.inducing_variable, self.kernel
        Kmn = ind.K_uf(kernel, X)
        Knn = kernel(X)
        if self.whiten:
            A_w = ind.Kuu_sqrt_solve(kernel, Kmn)
            fmean, fcov = conditional_whitened(A_w, Knn, self.q_mu, self.q_sqrt, full_cov=True)
        else:
            A = ind.Kuu_solve(kernel, Kmn)
            fmean, fcov = conditional_unwhitened(A, Kmn, Knn, self.q_mu, self.q_sqrt, full_cov=True)
        fmean = fmean + self.mean(X)
        return fmean, fcov

    def predict_f_samples(self, X, epsilon, jitter=1e-6):
        """Draw samples of the latent f at X from the joint posterior.

        Samples are produced via a Cholesky transform of caller-supplied
        iid-standard-normal noise. The caller owns the RNG — the function
        itself is deterministic.

        Parameters
        ----------
        X : tensor, shape (N, D)
        epsilon : tensor, shape (S, N)
            iid N(0, 1) draws. ``S`` is the number of samples.
        jitter : float
            Added to the diagonal of the posterior covariance before
            Cholesky, for numerical stability (default 1e-6).

        Returns
        -------
        samples : tensor, shape (S, N)
        """
        fmean, fcov = self.predict_joint(X)
        N = X.shape[0]
        L = pt.linalg.cholesky(fcov + jitter * pt.eye(N))
        return fmean[None, :] + epsilon @ L.T

    def prior_kl(self):
        """KL divergence KL[q(u) || p(u)]."""
        if self.whiten:
            return gauss_kl(self.q_mu, self.q_sqrt, K=None)
        ind, kernel = self.inducing_variable, self.kernel
        return gauss_kl_structured(
            self.q_mu,
            self.q_sqrt,
            K_solve=lambda rhs: ind.Kuu_solve(kernel, rhs),
            K_logdet=ind.Kuu_logdet(kernel),
        )
