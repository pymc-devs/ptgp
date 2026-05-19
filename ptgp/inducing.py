"""Inducing variables and initialization strategies.

The `*_init` functions return an :class:`Points` wrapping a plain
numpy array, so ``ip.Z`` is directly usable for plotting.
"""

from dataclasses import dataclass

import numpy as np
import pytensor
import pytensor.tensor as pt
import scipy.cluster.vq

from ptgp.kernels.base import Kernel


class InducingVariables:
    """Base class for inducing variables.

    Subclasses must implement ``num_inducing``, ``K_uu(kernel)``, and
    ``K_uf(kernel, X)``. The default ``Kuu_solve`` / ``Kuu_sqrt_solve`` /
    ``Kuu_logdet`` methods materialise ``K_uu`` and call dense linalg;
    structured subclasses override them with cheaper variants.
    """

    @property
    def num_inducing(self):
        raise NotImplementedError

    def K_uu(self, kernel):
        raise NotImplementedError

    def K_uf(self, kernel, X):
        raise NotImplementedError

    def Kuu_solve(self, kernel, rhs):
        return pt.linalg.solve(self.K_uu(kernel), rhs)

    def Kuu_sqrt_solve(self, kernel, rhs):
        L = pt.linalg.cholesky(self.K_uu(kernel))
        return pt.linalg.solve(L, rhs)

    def Kuu_logdet(self, kernel):
        _, ld = pt.linalg.slogdet(self.K_uu(kernel))
        return ld


class Points(InducingVariables):
    """Standard real-space inducing points.

    Parameters
    ----------
    Z : tensor or PyMC random variable, shape (M, D)
        Inducing point locations.
    """

    def __init__(self, Z):
        self.Z = Z

    @property
    def num_inducing(self):
        return self.Z.shape[0]

    def K_uu(self, kernel):
        return kernel(self.Z)

    def K_uf(self, kernel, X):
        return kernel(self.Z, X)


@dataclass
class KernelHealthDiagnostics:
    """Kernel-derived health metrics for a (kernel, X, Z) triple.

    Independent of how ``Z`` was produced. Used as the optional
    ``kernel_health`` field on :class:`RandomSubsampleDiagnostics` and
    :class:`KMeansDiagnostics`, and as the standalone return type of
    :func:`compute_inducing_diagnostics`.

    Attributes
    ----------
    d_final : ndarray, shape (N,)
        Per-data-point residual conditional variance ``Kff_diag - Q_diag``.
        Large values identify points poorly covered by ``Z``.
    total_variance : float
        ``tr(Kff)`` — the kernel diagonal sum on ``X`` before conditioning.
    nystrom_residual : float
        ``sum(d_final) = tr(Kff - Q)`` — total unexplained variance.
    kuu_min_eigenvalue : float
        Smallest eigenvalue of ``K(Z, Z) + jitter * I``.
    kuu_max_eigenvalue : float
        Largest eigenvalue of ``K(Z, Z) + jitter * I``.
    kuu_condition_number : float
        ``max_eig / min_eig``. Below ~1e5 is healthy.
    kuu_n_small_eigenvalues : int
        Number of eigenvalues below ``kuu_eig_threshold``.
    kuu_eig_threshold : float
        Threshold used to count small eigenvalues.
    """

    d_final: np.ndarray
    total_variance: float
    nystrom_residual: float
    kuu_min_eigenvalue: float
    kuu_max_eigenvalue: float
    kuu_condition_number: float
    kuu_n_small_eigenvalues: int
    kuu_eig_threshold: float

    def __repr__(self):
        pct = 100.0 * (1.0 - self.nystrom_residual / self.total_variance)
        return "\n".join(
            [
                f"variance explained: {pct:.1f}%",
                f"nystrom_residual  : {self.nystrom_residual:.4g}",
                f"min eigenvalue    : {self.kuu_min_eigenvalue:.3g}",
                f"max eigenvalue    : {self.kuu_max_eigenvalue:.3g}",
                f"condition number  : {self.kuu_condition_number:.3g}",
                f"eigs < {self.kuu_eig_threshold:.0e}     : {self.kuu_n_small_eigenvalues}",
            ]
        )


@dataclass
class RandomSubsampleDiagnostics:
    """Diagnostics returned by :func:`random_subsample_init`.

    Attributes
    ----------
    M_requested : int
        The ``M`` argument.
    M_returned : int
        Rows in the returned ``Z``. Always equals ``M_requested`` for this
        routine — kept for API symmetry with the other init routines.
    N_candidates : int
        Rows in the input ``X``.
    n_unique : int
        Number of unique rows in the returned ``Z``. Equals ``M_returned``
        unless ``X`` itself contained duplicate rows that were both
        sampled.
    pairwise_min_distance : float
        Minimum off-diagonal Euclidean distance between selected points.
        ``nan`` if ``M_returned < 2``.
    pairwise_mean_distance : float
        Mean off-diagonal Euclidean distance between selected points.
        ``nan`` if ``M_returned < 2``.
    kernel_health : KernelHealthDiagnostics or None
        Populated only when a ``kernel`` was passed to
        :func:`random_subsample_init`. ``None`` otherwise.
    """

    M_requested: int
    M_returned: int
    N_candidates: int
    n_unique: int
    pairwise_min_distance: float
    pairwise_mean_distance: float
    kernel_health: KernelHealthDiagnostics | None = None

    def __repr__(self):
        density = 100.0 * self.M_returned / self.N_candidates
        lines = [
            f"M requested        : {self.M_requested}",
            f"M returned         : {self.M_returned}",
            f"N candidates       : {self.N_candidates}",
            f"selection density  : {density:.2f}%",
            f"unique rows        : {self.n_unique}",
            f"pairwise min dist  : {self.pairwise_min_distance:.3g}",
            f"pairwise mean dist : {self.pairwise_mean_distance:.3g}",
        ]
        if self.kernel_health is not None:
            kh = self.kernel_health
            lines.append(
                f"kernel health      : cond={kh.kuu_condition_number:.3g}  "
                f"eigs<{kh.kuu_eig_threshold:.0e}={kh.kuu_n_small_eigenvalues}  "
                f"nystrom={kh.nystrom_residual:.3g}"
            )
        return "\n".join(lines)


@dataclass
class KMeansDiagnostics:
    """Diagnostics returned by :func:`kmeans_init`.

    Attributes
    ----------
    M_requested : int
        Clusters requested. The k-means algorithm runs at this size before
        any near-duplicate removal.
    M_returned : int
        Centroids returned after near-duplicate removal. ``<= M_requested``.
    n_removed_duplicates : int
        ``M_requested - M_returned``. Non-zero values mean k-means
        produced near-duplicate centroids that were deduplicated.
    dedup_tol : float
        The Euclidean-distance threshold used for deduplication.
    inertia : float
        ``sum_i ||X[i] - centroid[label[i]]||^2`` — k-means' standard
        within-cluster sum of squares. Smaller is tighter.
    pairwise_min_distance : float
        Minimum Euclidean distance between returned centroids. ``nan`` if
        ``M_returned < 2``. Should be ``> dedup_tol``.
    pairwise_mean_distance : float
        Mean off-diagonal Euclidean distance between returned centroids.
    kernel_health : KernelHealthDiagnostics or None
        Populated only when a ``kernel`` was passed to
        :func:`kmeans_init`. ``None`` otherwise.
    """

    M_requested: int
    M_returned: int
    n_removed_duplicates: int
    dedup_tol: float
    inertia: float
    pairwise_min_distance: float
    pairwise_mean_distance: float
    kernel_health: KernelHealthDiagnostics | None = None

    def __repr__(self):
        lines = [
            f"M requested        : {self.M_requested}",
            f"M returned         : {self.M_returned}",
            f"removed duplicates : {self.n_removed_duplicates}  (tol={self.dedup_tol:.0e})",
            f"inertia            : {self.inertia:.4g}",
            f"pairwise min dist  : {self.pairwise_min_distance:.3g}",
            f"pairwise mean dist : {self.pairwise_mean_distance:.3g}",
        ]
        if self.kernel_health is not None:
            kh = self.kernel_health
            lines.append(
                f"kernel health      : cond={kh.kuu_condition_number:.3g}  "
                f"eigs<{kh.kuu_eig_threshold:.0e}={kh.kuu_n_small_eigenvalues}  "
                f"nystrom={kh.nystrom_residual:.3g}"
            )
        return "\n".join(lines)


def _pairwise_distance_stats(Z):
    """Return ``(min, mean)`` of the upper-triangular Euclidean distance matrix.

    O(M^2) memory and compute. ``nan, nan`` if ``M < 2``.
    """
    M = Z.shape[0]
    if M < 2:
        return float("nan"), float("nan")
    d = np.sqrt(np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1))
    iu = np.triu_indices(M, k=1)
    return float(d[iu].min()), float(d[iu].mean())


def _compute_kernel_health(kernel, X, Z, jitter=1e-6, eig_threshold=1e-4, compile_kwargs=None):
    """Compute :class:`KernelHealthDiagnostics` for an arbitrary (kernel, X, Z).

    Used by :func:`compute_inducing_diagnostics` and the optional
    ``kernel`` path of :func:`random_subsample_init` / :func:`kmeans_init`.
    """
    if not isinstance(kernel, Kernel):
        raise TypeError("kernel must be a ptgp Kernel")
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    M = Z.shape[0]
    D = X.shape[1]

    X_sym = pt.matrix("_X", shape=(None, D), dtype="float64")
    Y_sym = pt.matrix("_Y", shape=(None, D), dtype="float64")
    ck = compile_kwargs or {}
    k_cross_fn = pytensor.function([X_sym, Y_sym], kernel(X_sym, Y_sym), **ck)
    k_diag_fn = pytensor.function([X_sym], pt.diag(kernel(X_sym)), **ck)

    Kff_diag = k_diag_fn(X)
    Kuu = k_cross_fn(Z, Z) + jitter * np.eye(M)
    Kuf = k_cross_fn(Z, X)

    L = np.linalg.cholesky(Kuu)
    A = np.linalg.solve(L, Kuf)
    Q_diag = np.sum(A * A, axis=0)
    d_final = np.maximum(Kff_diag - Q_diag, 0.0)

    return KernelHealthDiagnostics(
        d_final=d_final,
        total_variance=float(Kff_diag.sum()),
        nystrom_residual=float(d_final.sum()),
        **_compute_kuu_eig_stats(Kuu, eig_threshold),
    )


def random_subsample_init(
    X, M, rng=None, kernel=None, jitter=1e-6, eig_threshold=1e-4, compile_kwargs=None
):
    """Select ``M`` inducing points uniformly at random from ``X``.

    Parameters
    ----------
    X : array-like, shape (N, D)
        Candidate locations.
    M : int
        Number of inducing points.
    rng : int or numpy Generator, optional
        Seed or generator for reproducibility.
    kernel : Kernel, optional
        If given, also compute :class:`KernelHealthDiagnostics`
        (Kuu eigenvalues, Nyström residual, per-point coverage) and
        attach it to the returned diagnostic's ``kernel_health`` field.
        Otherwise ``kernel_health`` is ``None``.
    jitter : float, optional
        Diagonal jitter for ``K(Z, Z)`` when ``kernel`` is given.
        Default 1e-6.
    eig_threshold : float, optional
        Threshold below which Kuu eigenvalues are counted as "small".
        Default 1e-4.
    compile_kwargs : dict, optional
        Forwarded to ``pytensor.function`` for the kernel evaluation.

    Returns
    -------
    points : Points
        Wrapping an ``(M, D)`` numpy array.
    diagnostics : RandomSubsampleDiagnostics
        Spread / coverage summary, optionally with ``kernel_health``.
        ``repr(diagnostics)`` prints a one-screen summary.
    """
    X = np.asarray(X)
    N = X.shape[0]
    if M > N:
        raise ValueError(f"M={M} exceeds number of candidate points N={N}")
    rng = np.random.default_rng(rng)
    idx = rng.choice(N, size=M, replace=False)
    Z = X[idx]
    pmin, pmean = _pairwise_distance_stats(Z)
    kernel_health = (
        _compute_kernel_health(
            kernel, X, Z, jitter=jitter, eig_threshold=eig_threshold, compile_kwargs=compile_kwargs
        )
        if kernel is not None
        else None
    )
    diag = RandomSubsampleDiagnostics(
        M_requested=int(M),
        M_returned=int(Z.shape[0]),
        N_candidates=int(N),
        n_unique=int(np.unique(Z, axis=0).shape[0]),
        pairwise_min_distance=pmin,
        pairwise_mean_distance=pmean,
        kernel_health=kernel_health,
    )
    return Points(Z), diag


def kmeans_init(
    X, M, rng=None, tol=1e-6, kernel=None, jitter=1e-6, eig_threshold=1e-4, compile_kwargs=None
):
    """k-means++ centroids of ``X`` as inducing points.

    Uses :func:`scipy.cluster.vq.kmeans2` with ``minit="++"``.  After
    clustering, any centroids whose pairwise Euclidean distance is below
    ``tol`` are deduplicated (greedy: the first of each near-duplicate group
    is kept).  A summary is printed if any are removed.

    Parameters
    ----------
    X : array-like, shape (N, D)
    M : int
        Number of clusters / inducing points requested.
    rng : int or numpy Generator, optional
    tol : float, optional
        Euclidean-distance threshold below which two centroids are considered
        duplicates.  Default ``1e-6``.
    kernel : Kernel, optional
        If given, also compute :class:`KernelHealthDiagnostics`
        (Kuu eigenvalues, Nyström residual, per-point coverage) and
        attach it to the returned diagnostic's ``kernel_health`` field.
        Otherwise ``kernel_health`` is ``None``.
    jitter : float, optional
        Diagonal jitter for ``K(Z, Z)`` when ``kernel`` is given.
        Default 1e-6.
    eig_threshold : float, optional
        Threshold below which Kuu eigenvalues are counted as "small".
        Default 1e-4.
    compile_kwargs : dict, optional
        Forwarded to ``pytensor.function`` for the kernel evaluation.

    Returns
    -------
    points : Points
        Wrapping an ``(M', D)`` numpy array of centroids, with ``M' <= M``.
    diagnostics : KMeansDiagnostics
        Cluster-fit summary, optionally with ``kernel_health``.
        ``repr(diagnostics)`` prints a one-screen summary.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if M > N:
        raise ValueError(f"M={M} exceeds number of candidate points N={N}")
    seed = int(np.random.default_rng(rng).integers(0, 2**31 - 1))
    centroids, labels = scipy.cluster.vq.kmeans2(X, M, minit="++", seed=seed)

    # Inertia: within-cluster sum of squared distances. Computed on the
    # full M centroid set (before dedup) since labels are assigned to those.
    inertia = float(np.sum((X - centroids[labels]) ** 2))

    # Greedy deduplication: keep centroid i if it is > tol from all earlier
    # kept centroids.  O(M^2) but M is typically small (100-500).
    dists = np.sqrt(np.sum((centroids[:, None] - centroids[None, :]) ** 2, axis=-1))
    keep = np.ones(M, dtype=bool)
    for i in range(M):
        if not keep[i]:
            continue
        near = dists[i, i + 1 :] < tol
        keep[i + 1 :][near] = False

    n_removed = int((~keep).sum())
    if n_removed > 0:
        print(
            f"kmeans_init: removed {n_removed} near-duplicate centroid(s) "
            f"(tol={tol:.0e}); returning {keep.sum()} of {M} requested."
        )

    Z = centroids[keep]
    pmin, pmean = _pairwise_distance_stats(Z)
    kernel_health = (
        _compute_kernel_health(
            kernel, X, Z, jitter=jitter, eig_threshold=eig_threshold, compile_kwargs=compile_kwargs
        )
        if kernel is not None
        else None
    )
    diag = KMeansDiagnostics(
        M_requested=int(M),
        M_returned=int(Z.shape[0]),
        n_removed_duplicates=n_removed,
        dedup_tol=float(tol),
        inertia=inertia,
        pairwise_min_distance=pmin,
        pairwise_mean_distance=pmean,
        kernel_health=kernel_health,
    )
    return Points(Z), diag


@dataclass
class GreedyVarianceDiagnostics:
    """Diagnostics returned by :func:`greedy_variance_init`.

    Attributes
    ----------
    trace_curve : ndarray, shape (M,)
        Remaining unexplained variance after conditioning on each successive
        inducing point. ``trace_curve[0]`` is the total kernel diagonal sum
        before any conditioning; ``trace_curve[m]`` is the residual after
        conditioning on ``m`` points. Divide by ``total_variance`` to get the
        fraction of variance still unexplained.
    d_final : ndarray, shape (N,)
        Per-data-point residual conditional variance after all M points are
        selected. Large values identify data points poorly covered by the
        current inducing set.
    total_variance : float
        Total kernel diagonal sum before conditioning (``trace_curve[0]``).
    kuu_min_eigenvalue : float
        Smallest eigenvalue of ``K(Z, Z) + jitter * I``.
    kuu_max_eigenvalue : float
        Largest eigenvalue of ``K(Z, Z) + jitter * I``.
    kuu_condition_number : float
        ``max_eig / min_eig``. Values below ~1e5 are numerically healthy.
        Large values indicate near-duplicate inducing points or a kernel
        lengthscale too short relative to the inducing-point spacing.
    kuu_n_small_eigenvalues : int
        Number of eigenvalues below ``kuu_eig_threshold``. Non-zero values
        mean the inducing-point covariance is near-singular.
    kuu_eig_threshold : float
        Threshold used to count small eigenvalues (default 1e-4).
    """

    trace_curve: np.ndarray
    d_final: np.ndarray
    total_variance: float
    kuu_min_eigenvalue: float
    kuu_max_eigenvalue: float
    kuu_condition_number: float
    kuu_n_small_eigenvalues: int
    kuu_eig_threshold: float

    def __repr__(self):
        M = len(self.trace_curve)
        pct = 100.0 * (1.0 - self.trace_curve[-1] / self.total_variance)
        lines = [
            f"M                 : {M}",
            f"variance explained: {pct:.1f}%",
            f"min eigenvalue    : {self.kuu_min_eigenvalue:.3g}",
            f"max eigenvalue    : {self.kuu_max_eigenvalue:.3g}",
            f"condition number  : {self.kuu_condition_number:.3g}",
            f"eigs < {self.kuu_eig_threshold:.0e}     : {self.kuu_n_small_eigenvalues}",
        ]
        return "\n".join(lines)


def greedy_variance_init(
    X, M, kernel, threshold=0.0, jitter=1e-12, rng=None, eig_threshold=1e-4, compile_kwargs=None
):
    """Greedy conditional-variance (pivoted-Cholesky) selection.

    Implements the "ConditionalVariance" initialization of Burt et al. (2020),
    *Convergence of Sparse Variational Inference in GP Regression*. At each
    step, the next inducing point is the row of ``X`` with largest remaining
    conditional variance given the already-selected points — equivalent to
    running a partial pivoted Cholesky decomposition of ``K(X, X)`` with the
    standard max-diagonal pivot rule. Selected points are a **subset of X**;
    this is discrete subset selection, not continuous optimization.

    Adapted from markvdw/RobustGP (Apache-2.0). Time O(N·M^2), memory O(N·M).

    Recommended workflow
    --------------------
    Burt et al. show that with a good greedy initialization, ``Z`` typically
    does **not** need to be gradient-optimized during training — for most
    problems the frozen subset is within noise of jointly-optimized ``Z`` at a
    tiny fraction of the compute. The standard recipe:

    1. Initialize ``Z`` with ``greedy_variance_init(X, M, kernel)`` using
       initial kernel hyperparameters.
    2. Freeze ``Z``. Train the kernel/likelihood hyperparameters (and, for
       SVGP, the variational parameters).
    3. *Optional.* Re-initialize ``Z`` with the learned hyperparameters and
       retrain briefly. Usually a small improvement.

    For VFE/SGPR (Titsias collapsed bound) ``Z`` is sometimes still optimized
    because gradients are cheap; for SVGP, frozen greedy ``Z`` is the norm.

    Parameters
    ----------
    X : array-like, shape (N, D)
    M : int
        Maximum number of inducing points. Fewer may be returned if the
        approximation converges.
    kernel : Kernel
        PTGP kernel, compiled internally via ``pytensor.function``.
    threshold : float, optional
        Stop early if the trace of the residual ``K - Q`` drops below this.
        Default 0 (run the full ``M`` iterations).
    jitter : float, optional
        Small diagonal jitter for numerical stability.
    rng : int or numpy Generator, optional
    eig_threshold : float, optional
        Eigenvalues of ``K(Z, Z) + jitter * I`` below this value are counted
        in ``kuu_n_small_eigenvalues``. Default ``1e-4``.
    compile_kwargs : dict, optional
        Forwarded as ``**compile_kwargs`` to ``pytensor.function`` when
        compiling the kernel evaluations. Use to set ``mode``
        (e.g. ``"NUMBA"``, ``"JAX"``). Same pattern as ``pm.sample``'s
        ``compile_kwargs``.

    Returns
    -------
    points : Points
        Wrapping an ``(M', D)`` numpy array with ``M' <= M``.
    diagnostics : GreedyVarianceDiagnostics
        Dataclass with fields ``trace_curve``, ``d_final``, ``total_variance``,
        ``kuu_min_eigenvalue``, ``kuu_max_eigenvalue``, ``kuu_condition_number``,
        ``kuu_n_small_eigenvalues``, and ``kuu_eig_threshold``.
        ``repr(diagnostics)`` prints a one-screen summary.
    """
    if not isinstance(kernel, Kernel):
        raise TypeError("kernel must be a ptgp Kernel")
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if M > N:
        raise ValueError(f"M={M} exceeds number of candidate points N={N}")
    rng = np.random.default_rng(rng)

    D = X.shape[1]
    X_sym = pt.matrix("_X", shape=(None, D), dtype="float64")
    Y_sym = pt.matrix("_Y", shape=(None, D), dtype="float64")
    ck = compile_kwargs or {}
    k_cross_fn = pytensor.function([X_sym, Y_sym], kernel(X_sym, Y_sym), **ck)
    k_diag_fn = pytensor.function([X_sym], pt.diag(kernel(X_sym)), **ck)

    perm = rng.permutation(N)
    Xp = X[perm]

    d = k_diag_fn(Xp) + jitter
    total_variance = float(d.sum())
    indices = np.zeros(M, dtype=int)
    indices[0] = int(np.argmax(d))

    if M == 1:
        Z1 = Xp[indices]
        Kuu1 = k_cross_fn(Z1, Z1) + jitter * np.eye(1)
        diag = GreedyVarianceDiagnostics(
            trace_curve=np.array([total_variance]),
            d_final=d.copy(),
            total_variance=total_variance,
            **_compute_kuu_eig_stats(Kuu1, eig_threshold),
        )
        return Points(Z1), diag

    C = np.zeros((M - 1, N))
    final_m = M
    trace_curve = np.empty(M)
    trace_curve[0] = total_variance

    for m in range(M - 1):
        j = int(indices[m])
        dj = np.sqrt(d[j])
        cj = C[:m, j]

        Kj = k_cross_fn(Xp, Xp[j : j + 1]).ravel()
        Kj[j] += jitter

        e = (Kj - C[:m].T @ cj) / dj
        C[m, :] = e

        d = np.maximum(d - e**2, 0.0)
        trace_curve[m + 1] = float(d.sum())

        indices[m + 1] = int(np.argmax(d))

        if d.sum() < threshold:
            final_m = m + 2
            break

    Z_selected = Xp[indices[:final_m]]
    Kuu = k_cross_fn(Z_selected, Z_selected) + jitter * np.eye(final_m)
    eig_stats = _compute_kuu_eig_stats(Kuu, eig_threshold)

    diag = GreedyVarianceDiagnostics(
        trace_curve=trace_curve[:final_m].copy(),
        d_final=d.copy(),
        total_variance=total_variance,
        **eig_stats,
    )
    return Points(Z_selected), diag


def _compute_kuu_eig_stats(Kuu, eig_threshold=1e-4):
    """Eigenvalue summary for a Kuu matrix.

    Returns a dict with the five ``kuu_*`` fields used by
    :class:`GreedyVarianceDiagnostics`.
    """
    eigs = np.linalg.eigvalsh(Kuu)
    return {
        "kuu_min_eigenvalue": float(eigs[0]),
        "kuu_max_eigenvalue": float(eigs[-1]),
        "kuu_condition_number": float(eigs[-1] / eigs[0]),
        "kuu_n_small_eigenvalues": int(np.sum(eigs < eig_threshold)),
        "kuu_eig_threshold": eig_threshold,
    }


def compute_inducing_diagnostics(
    kernel, X, Z, jitter=1e-6, eig_threshold=1e-4, compile_kwargs=None
):
    """Evaluate inducing-point health at a *given* (kernel, X, Z).

    Computes the same kernel-derived metrics that
    :func:`greedy_variance_init` reports (Kuu eigenvalues, per-point
    residual conditional variance, total variance), but for a ``Z`` that
    you already have — from k-means, manual placement, post-training, or
    elsewhere. No greedy selection is performed; no ``trace_curve``
    history is available.

    This is a thin wrapper around :func:`_compute_kernel_health`; the
    same diagnostic is also exposed by passing ``kernel=`` to
    :func:`random_subsample_init` and :func:`kmeans_init`.

    Parameters
    ----------
    kernel : Kernel
        PTGP kernel, compiled internally via ``pytensor.function``.
    X : array-like, shape (N, D)
        Data locations.
    Z : array-like, shape (M, D)
        Inducing locations.
    jitter : float, optional
        Diagonal jitter added to ``K(Z, Z)`` before the eigenvalue and
        Cholesky computations. Default 1e-6.
    eig_threshold : float, optional
        Threshold below which Kuu eigenvalues are counted as "small".
    compile_kwargs : dict, optional
        Forwarded as ``**compile_kwargs`` to ``pytensor.function``.

    Returns
    -------
    KernelHealthDiagnostics
        ``d_final``, ``total_variance``, ``nystrom_residual``, and the
        ``kuu_*`` eigenvalue stats. ``repr()`` prints a one-screen
        summary.
    """
    return _compute_kernel_health(
        kernel,
        X,
        Z,
        jitter=jitter,
        eig_threshold=eig_threshold,
        compile_kwargs=compile_kwargs,
    )
