import pytensor.assumptions as pta
import pytensor.tensor as pt

# Diagonal jitter added to Kmm before Cholesky / inversion, to keep it PSD
# under floating-point noise. Matches GPflow / GPJax / PyMC defaults of 1e-6.
_DEFAULT_JITTER = 1e-6


def conditional_whitened(A_white, Knn, f, q_sqrt=None, full_cov=False):
    """Posterior conditional in whitened parameterisation.

    A_white : (M, N), satisfies A_white = R^{-1} @ Kmn where R @ R.T = Kmm.
    Knn     : (N,) prior diagonal if ``full_cov=False``, else (N, N) prior covariance.
    f       : (M,) variational mean (whitened: prior on v is N(0, I)).
    q_sqrt  : (M, K), optional.
    full_cov: if True, return full (N, N) covariance; else (N,) marginals.
    """
    fmean = A_white.T @ f
    if full_cov:
        fvar = Knn - A_white.T @ A_white
    else:
        fvar = Knn - pt.sum(A_white**2, axis=0)
    if q_sqrt is not None:
        B = A_white.T @ q_sqrt
        if full_cov:
            fvar = fvar + B @ B.T
        else:
            fvar = fvar + pt.sum(B**2, axis=1)
    return fmean, fvar


def conditional_unwhitened(A, Kmn, Knn, f, q_sqrt=None, full_cov=False):
    """Posterior conditional in unwhitened parameterisation.

    A    : (M, N), satisfies A = Kmm^{-1} @ Kmn.
    Kmn  : (M, N) cross-covariance.
    Knn  : (N,) prior diagonal if ``full_cov=False``, else (N, N) prior covariance.
    f    : (M,) variational mean (prior on u is N(0, Kmm)).
    full_cov: if True, return full (N, N) covariance; else (N,) marginals.
    """
    fmean = A.T @ f
    if full_cov:
        fvar = Knn - A.T @ Kmn
    else:
        fvar = Knn - pt.sum(A * Kmn, axis=0)
    if q_sqrt is not None:
        B = A.T @ q_sqrt
        if full_cov:
            fvar = fvar + B @ B.T
        else:
            fvar = fvar + pt.sum(B**2, axis=1)
    return fmean, fvar


def base_conditional(Kmn, Kmm, Knn, f, q_sqrt=None, white=False, full_cov=False):
    """Back-compat wrapper. Materialises Kmm; use the helpers above directly
    when you have a structured Kuu_solve / Kuu_sqrt_solve."""
    # Add jitter to keep Kmm PSD under float noise; matches GPflow / PyMC default.
    # Re-annotate after the addition: PyTensor canonicalizes ``Kmm + c·I`` into a
    # ``set_subtensor`` on the diagonal, which our PSD-inference rules don't see
    # through. The mathematical identity (PSD + c·I PSD ⇒ PSD) is sound.
    Kmm = pta.assume(
        Kmm + _DEFAULT_JITTER * pt.eye(Kmm.shape[-1], dtype=Kmm.dtype),
        positive_definite=True,
        symmetric=True,
    )
    if white:
        L = pt.linalg.cholesky(Kmm)
        A_white = pt.linalg.solve(L, Kmn)
        return conditional_whitened(A_white, Knn, f, q_sqrt, full_cov=full_cov)
    A = pt.linalg.inv(Kmm) @ Kmn
    return conditional_unwhitened(A, Kmn, Knn, f, q_sqrt, full_cov=full_cov)


def nn_conditional(kernel, X, Z_nbr, mask, jitter=_DEFAULT_JITTER):
    """Weights and residual variance of each point given its masked neighbors.

    For one point ``x`` with neighbor locations ``Z_n``, returns
    ``b = K_nn^{-1} k_nx`` and ``F = k_xx - k_nx^T b``, the Vecchia
    conditional ``u_x | u_n ~ N(b^T u_n, F)`` (Datta et al. 2016,
    arXiv:1406.7343, Sec 2; Wu, Pleiss & Cunningham 2022, arXiv:2202.01694,
    Eq 17-18). Padded neighbor slots (``mask == 0``) get identity rows and
    columns in ``K_nn`` and zero cross-covariance, so their weights are
    exactly 0. The graph is written for one point and lifted over the batch
    with ``vectorize_graph``, so the ``K x K`` solves become a batched Cholesky.

    Parameters
    ----------
    kernel : Kernel
    X : tensor, shape (B, D)
        Points to condition.
    Z_nbr : tensor, shape (B, K, D)
        Neighbor locations of each point.
    mask : tensor, shape (B, K)
        1 for real neighbors, 0 for padding.
    jitter : float
        Added to the diagonal of ``K_nn`` and to ``F``.

    Returns
    -------
    b : tensor, shape (B, K)
    F : tensor, shape (B,)
    """
    from pytensor.graph.replace import vectorize_graph

    K = Z_nbr.type.shape[-2]
    D = Z_nbr.type.shape[-1]
    x = pt.vector("_x", shape=(D,), dtype=X.dtype)
    zn = pt.matrix("_zn", shape=(K, D), dtype=Z_nbr.dtype)
    m = pt.vector("_m", shape=(K,), dtype=X.dtype)

    eye = pt.eye(K, dtype=X.dtype)
    Knn = kernel(zn) * pt.outer(m, m) + eye * (1.0 - m)[None, :] + jitter * eye
    Knn = pta.assume(Knn, symmetric=True, positive_definite=True)
    knx = kernel(zn, x[None, :])[:, 0] * m
    b = pt.linalg.solve(Knn, knx)
    F = kernel.diag(x[None, :])[0] - knx @ b + jitter

    b_batch, F_batch = vectorize_graph([b, F], {x: X, zn: Z_nbr, m: pt.cast(mask, X.dtype)})
    return b_batch, F_batch
