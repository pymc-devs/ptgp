import dataclasses

import numpy as np
import pytensor.assumptions as pta
import pytensor.tensor as pt

from ptgp.mean import Zero
from ptgp.objectives import vgp_elbo


@dataclasses.dataclass(frozen=True)
class VGPParams:
    """Symbolic Opper-Archambeau variational parameters for VGP.

    Attributes
    ----------
    alpha : TensorVariable
        Mean weights, shape ``(N,)``. The posterior mean is ``mean(X) + K alpha``.
    lam : TensorVariable
        Posterior precision diagonal, shape ``(N,)``, strictly positive
        (materialised as ``softplus(lambda_raw)``).
    extra_vars : tuple of TensorVariable
        Trainable leaves underlying ``alpha`` and ``lam``.
    extra_init : tuple of ndarray
        Initial values for ``extra_vars``, in matching order.
    """

    alpha: object
    lam: object
    extra_vars: tuple = ()
    extra_init: tuple = ()


def init_vgp_params(N, alpha_init=None, lambda_init=None):
    """Build symbolic Opper-Archambeau variational parameters sized to N.

    The posterior precision diagonal ``lam`` is stored as ``lambda_raw`` and
    materialised via ``pt.softplus`` so it is strictly positive at every
    optimizer step (matching the softplus convention in
    :func:`ptgp.gp.init_variational_params`). ``alpha`` is a free length-``N``
    vector.

    Parameters
    ----------
    N : int
        Number of training points (the variational posterior is over the latent
        function at these inputs).
    alpha_init : ndarray of shape (N,), optional
        Initial mean weights. Defaults to zeros.
    lambda_init : ndarray of shape (N,), optional
        Initial precision diagonal, strictly positive. Defaults to ones.

    Returns
    -------
    VGPParams

    Examples
    --------
    >>> vp = init_vgp_params(N=10)
    >>> vgp = VGP(kernel=..., likelihood=..., variational_params=vp)
    """
    if alpha_init is None:
        alpha_init = np.zeros(N, dtype=np.float64)
    else:
        alpha_init = np.asarray(alpha_init, dtype=np.float64)
        if alpha_init.shape != (N,):
            raise ValueError(f"alpha_init must have shape ({N},); got {alpha_init.shape}.")

    if lambda_init is None:
        lambda_init = np.ones(N, dtype=np.float64)
    else:
        lambda_init = np.asarray(lambda_init, dtype=np.float64)
        if lambda_init.shape != (N,):
            raise ValueError(f"lambda_init must have shape ({N},); got {lambda_init.shape}.")
    if np.any(lambda_init <= 0):
        raise ValueError("lambda_init must be strictly positive (it is a precision diagonal).")

    alpha = pt.vector("alpha", shape=(N,), dtype="float64")
    lambda_raw = pt.vector("lambda_raw", shape=(N,), dtype="float64")
    lam = pt.softplus(lambda_raw)
    # Inverse softplus: lambda_raw such that softplus(lambda_raw) == lambda_init.
    lambda_raw_init = np.log(np.expm1(lambda_init))
    return VGPParams(
        alpha=alpha,
        lam=lam,
        extra_vars=(alpha, lambda_raw),
        extra_init=(alpha_init, lambda_raw_init),
    )


class VGP:
    """Full variational Gaussian Process (Opper-Archambeau).

    The variational posterior q(f) = N(mean(X) + K alpha, S) is placed directly
    over the latent function at the N training inputs, with
    S = (K^{-1} + diag(lambda))^{-1}. There are no inducing points; the free
    parameters are ``alpha`` (N,) and the precision diagonal ``lambda`` (N,),
    so the posterior costs O(N) parameters (Opper & Archambeau, 2009).

    All computation routes through the factor A = I + G K G,
    ``G = diag(sqrt(lambda))``, whose eigenvalues are >= 1, so K itself is never
    factorised or inverted and no jitter is required.

    Parameters
    ----------
    kernel : Kernel
        Covariance function.
    mean : callable, optional
        Mean function (default: ``Zero()``).
    likelihood : Likelihood
        Observation likelihood (Gaussian or any non-Gaussian likelihood).
    variational_params : VGPParams
        Symbolic variational parameters sized to N. Construct via
        :func:`init_vgp_params` (recommended). ``vp.extra_vars`` /
        ``vp.extra_init`` are surfaced for the optimizer.
    """

    default_objective = staticmethod(vgp_elbo)
    predict_needs_data = True

    def __init__(self, kernel, mean=None, likelihood=None, variational_params=None):
        """Store the kernel, mean, likelihood, and variational parameters."""
        if variational_params is None:
            raise ValueError(
                "VGP requires variational_params. Construct via "
                "ptgp.gp.init_vgp_params(N) and pass as variational_params=..., "
                "where N is the number of training points."
            )
        self.kernel = kernel
        self.mean = mean if mean is not None else Zero()
        self.likelihood = likelihood
        self.variational_params = variational_params
        self.alpha = variational_params.alpha
        self.lam = variational_params.lam

    @property
    def extra_vars(self):
        """Trainable variational leaves (alpha, lambda_raw)."""
        return tuple(self.variational_params.extra_vars)

    @property
    def extra_init(self):
        """Initial values for the trainable variational leaves."""
        return tuple(self.variational_params.extra_init)

    def _build_A(self, X):
        """Return ``(K, g, L)`` where K = kernel(X), g = sqrt(lambda), and L is
        the Cholesky factor of A = I + G K G (G = diag(g)).

        A is symmetric positive definite (eigenvalues >= 1) for any PSD K, so no
        jitter is needed and K is never factorised on its own.
        """
        K = self.kernel(X)
        n = K.shape[0]
        g = pt.sqrt(self.lam)
        A = pt.eye(n, dtype=K.dtype) + g[:, None] * K * g[None, :]
        A = pta.assume(A, positive_definite=True, symmetric=True)
        L = pt.linalg.cholesky(A)
        return K, g, L

    def _train_marginals(self, X):
        """Marginal mean and variance of q(f) at the training inputs.

        ``fmean = mean(X) + K alpha`` and ``fvar = diag(S)`` where
        ``diag(S) = diag(K) - colsum((L^{-1} G K)**2)``.

        Returns
        -------
        fmean : tensor, shape (N,)
        fvar : tensor, shape (N,)
        """
        K, g, L = self._build_A(X)
        fmean = self.mean(X) + K @ self.alpha
        B = pt.linalg.solve_triangular(L, g[:, None] * K, lower=True)
        fvar = pt.diagonal(K) - pt.sum(B**2, axis=0)
        return fmean, fvar

    def prior_kl(self, X):
        """KL[q(f) || p(f)] in closed form.

        ``KL = 0.5 * (tr(A^{-1}) + alpha^T K alpha - N + log|A|)``, using
        ``tr(K^{-1} S) = tr(A^{-1})`` and ``log|K| - log|S| = log|A|``.

        Returns
        -------
        scalar
        """
        K, g, L = self._build_A(X)
        n = K.shape[0]
        logdet_A = 2.0 * pt.sum(pt.log(pt.diagonal(L)))
        Linv = pt.linalg.solve_triangular(L, pt.eye(n, dtype=K.dtype), lower=True)
        tr_Ainv = pt.sum(Linv**2)
        quad = self.alpha @ (K @ self.alpha)
        return 0.5 * (tr_Ainv + quad - n + logdet_A)

    def predict_marginal(self, X_new, X_train, y_train=None, incl_lik=False):
        """Posterior marginal mean and variance at each point in X_new.

        Returns the per-point posterior; correlations between test points are
        discarded. ``y_train`` is accepted for API compatibility with
        :func:`ptgp.optim.predict` but ignored: the posterior is carried by
        ``alpha``/``lambda``, not recomputed from the targets.

        Parameters
        ----------
        X_new : tensor, shape (N*, D)
        X_train : tensor, shape (N, D)
        y_train : tensor, optional
            Ignored.
        incl_lik : bool
            If True, push through the likelihood's predictive mean/var.

        Returns
        -------
        mean : tensor, shape (N*,)
        var : tensor, shape (N*,)
        """
        K, g, L = self._build_A(X_train)
        Kfnew = self.kernel(X_train, X_new)  # (N, N*)
        fmean = self.mean(X_new) + Kfnew.T @ self.alpha
        C = pt.linalg.solve_triangular(L, g[:, None] * Kfnew, lower=True)
        fvar = self.kernel.diag(X_new) - pt.sum(C**2, axis=0)
        if incl_lik:
            return self.likelihood.predict_mean_and_var(fmean, fvar)
        return fmean, fvar
