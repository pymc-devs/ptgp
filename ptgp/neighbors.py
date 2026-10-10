"""Host-side ordering, neighbor search, and block partitioning for nearest-neighbor GPs.

Everything here is NumPy / SciPy and runs outside the PyTensor graph. Two
metrics are supported:

- Euclidean on ``Z / scale``, searched with :class:`scipy.spatial.cKDTree`.
  With ``scale`` set to the ARD lengthscales this is the scaled Vecchia metric
  of Katzfuss, Guinness & Lawrence (arXiv:2005.00386, Eq 1, 3).
- Correlation distance ``sqrt(1 - |rho|)`` of Kang & Katzfuss (Statistics and
  Computing 2023, arXiv:2112.14591, Eq 4), searched by chunked brute force.
  Callers supply ``corr_rows(idx) -> |rho(Z[idx], Z)|``. For a stationary ARD
  kernel it selects the same neighbors as Euclidean on ``Z / ls`` (their
  Prop 1).

Orderings are integer arrays where ``order[r]`` is the point at rank ``r``.
Neighbor arrays are indexed by point, not by rank: ``nn_idx[j]`` holds the
points that ``j`` conditions on, all earlier than ``j`` in the ordering.
"""

import numpy as np

from scipy.spatial import cKDTree

_CHUNK = 1024


def order_random(M, seed=None):
    """Random ordering of ``M`` points.

    Random ordering is the default in Wu, Pleiss & Cunningham (ICML 2022,
    arXiv:2202.01694, Sec 3.2) and is sharper than coordinate ordering
    (Guinness 2018, arXiv:1609.05372, Sec 5).

    Parameters
    ----------
    M : int
        Number of points.
    seed : int or numpy.random.Generator, optional
        Random seed.

    Returns
    -------
    ndarray of int, shape (M,)
    """
    return np.random.default_rng(seed).permutation(M)


def order_maximin(dist_fn, M, first=0):
    """Exact maximin ordering by farthest-point sampling.

    Each next point is the one farthest from all points already ordered
    (Guinness 2018, arXiv:1609.05372, Sec 2; reverse form in Schafer,
    Katzfuss & Owhadi 2021, arXiv:2004.14455, Eq 3.1-3.2). Costs ``M`` calls
    to ``dist_fn``, so O(M^2) distance evaluations and O(M) memory.

    Parameters
    ----------
    dist_fn : callable
        ``dist_fn(i) -> ndarray, shape (M,)``, distances from point ``i`` to
        every point.
    M : int
        Number of points.
    first : int
        Point placed at rank 0.

    Returns
    -------
    ndarray of int, shape (M,)
    """
    order = np.empty(M, dtype=np.int64)
    order[0] = first
    mind = np.asarray(dist_fn(first), dtype=np.float64).copy()
    mind[first] = -np.inf
    for r in range(1, M):
        j = int(np.argmax(mind))
        order[r] = j
        np.minimum(mind, dist_fn(j), out=mind)
        mind[j] = -np.inf
    return order


def euclidean_dist_fn(Z, scale=None):
    """``dist_fn`` for :func:`order_maximin` under Euclidean distance on ``Z / scale``."""
    Zs = _scaled(Z, scale)
    return lambda i: np.sqrt(np.sum((Zs - Zs[i]) ** 2, axis=1))


def correlation_dist_fn(corr_rows):
    """``dist_fn`` for :func:`order_maximin` under correlation distance ``sqrt(1 - |rho|)``."""
    return lambda i: np.sqrt(np.maximum(1.0 - corr_rows(np.array([i]))[0], 0.0))


def find_neighbors(order, k, Z=None, scale=None, corr_rows=None):
    """Up to ``k`` nearest preceding points of every point in an ordering.

    Euclidean search follows Guinness 2018 (arXiv:1609.05372, Sec 4.1): query
    ``2k`` nearest candidates among all points, keep those earlier in the
    ordering, and re-query points left short with twice as many candidates.

    Parameters
    ----------
    order : ndarray of int, shape (M,)
        ``order[r]`` is the point at rank ``r``.
    k : int
        Maximum number of neighbors.
    Z : ndarray, shape (M, D), optional
        Point locations, for the Euclidean metric.
    scale : float or ndarray, shape (D,), optional
        Divides ``Z`` before measuring distance (e.g. ARD lengthscales).
    corr_rows : callable, optional
        ``corr_rows(idx) -> ndarray, shape (len(idx), M)`` of ``|rho|``. When
        given, the correlation metric is used and ``Z`` is ignored.

    Returns
    -------
    nn_idx : ndarray of int, shape (M, k)
        Neighbors of each point, nearest first. Padded slots hold the point
        itself.
    nn_mask : ndarray of bool, shape (M, k)
        False for padded slots. Point at rank ``r`` has ``min(r, k)`` neighbors.
    """
    order = np.asarray(order, dtype=np.int64)
    M = order.size
    rank = np.empty(M, dtype=np.int64)
    rank[order] = np.arange(M)
    nn_idx = np.full((M, k), -1, dtype=np.int64)

    if corr_rows is not None:
        _correlation_predecessors(nn_idx, order, rank, k, corr_rows)
    else:
        _euclidean_predecessors(nn_idx, order, rank, k, _scaled(Z, scale))

    nn_mask = nn_idx >= 0
    nn_idx = np.where(nn_mask, nn_idx, np.arange(M)[:, None])
    return nn_idx, nn_mask


def _euclidean_predecessors(nn_idx, order, rank, k, Zs):
    M = order.size
    head = order[: min(k, M)]
    for r, j in enumerate(head[1:], start=1):
        preds = order[:r]
        d = np.sum((Zs[preds] - Zs[j]) ** 2, axis=1)
        nn_idx[j, :r] = preds[np.argsort(d, kind="stable")]

    tree = cKDTree(Zs)
    todo = order[k:]
    n_query = 2 * k
    while todo.size:
        q = min(n_query, M)
        _, cand = tree.query(Zs[todo], q)
        cand = cand.reshape(todo.size, q)
        is_pred = rank[cand] < rank[todo][:, None]
        done = is_pred.sum(axis=1) >= k
        pos = np.argsort(~is_pred[done], axis=1, kind="stable")[:, :k]
        nn_idx[todo[done]] = np.take_along_axis(cand[done], pos, axis=1)
        todo = todo[~done]
        n_query *= 2


def _correlation_predecessors(nn_idx, order, rank, k, corr_rows):
    M = order.size
    for start in range(0, M, _CHUNK):
        rows = order[start : start + _CHUNK]
        d = 1.0 - corr_rows(rows)
        d[rank[None, :] >= rank[rows][:, None]] = np.inf
        n_pred = np.minimum(rank[rows], k)
        kk = min(k, M - 1)
        part = (
            np.argpartition(d, kk - 1, axis=1)[:, :kk] if kk > 0 else np.zeros((rows.size, 0), int)
        )
        part_d = np.take_along_axis(d, part, axis=1)
        sorted_pos = np.argsort(part_d, axis=1, kind="stable")
        best = np.take_along_axis(part, sorted_pos, axis=1)
        for i, j in enumerate(rows):
            nn_idx[j, : n_pred[i]] = best[i, : n_pred[i]]


def query_neighbors(X_new, k, Z=None, scale=None, corr_rows=None, tree=None):
    """The ``k`` nearest points of ``Z`` to each row of ``X_new``.

    Used for prediction, where a test point conditions on its nearest
    inducing points among all of ``Z``.

    Parameters
    ----------
    X_new : ndarray, shape (N, D)
        Query points.
    k : int
        Number of neighbors.
    Z : ndarray, shape (M, D), optional
        Point locations, for the Euclidean metric.
    scale : float or ndarray, shape (D,), optional
        Divides inputs before measuring distance.
    corr_rows : callable, optional
        ``corr_rows(X_chunk) -> ndarray, shape (len(X_chunk), M)`` of
        ``|rho(X_chunk, Z)|``. When given, the correlation metric is used.
    tree : scipy.spatial.cKDTree, optional
        Prebuilt tree on ``Z / scale``, reused across calls.

    Returns
    -------
    ndarray of int, shape (N, k)
        Nearest first.
    """
    X_new = np.asarray(X_new)
    if corr_rows is None:
        if tree is None:
            tree = cKDTree(_scaled(Z, scale))
        _, idx = tree.query(_scaled(X_new, scale), k)
        return idx.reshape(X_new.shape[0], k).astype(np.int64)

    out = np.empty((X_new.shape[0], k), dtype=np.int64)
    for start in range(0, X_new.shape[0], _CHUNK):
        d = 1.0 - corr_rows(X_new[start : start + _CHUNK])
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        sorted_pos = np.argsort(np.take_along_axis(d, part, axis=1), axis=1, kind="stable")
        out[start : start + _CHUNK] = np.take_along_axis(part, sorted_pos, axis=1)
    return out


def partition_blocks(Z, block_size, scale=None):
    """Partition points into spatially compact blocks by recursive median splits.

    Each split cuts along the widest dimension of ``Z / scale`` at a multiple
    of ``block_size``, so every block is full except at most one.

    Parameters
    ----------
    Z : ndarray, shape (M, D)
        Point locations.
    block_size : int
        Maximum points per block.
    scale : float or ndarray, shape (D,), optional
        Divides ``Z`` before splitting.

    Returns
    -------
    ndarray of int, shape (n_blocks, block_size)
        Point ids per block, padded with -1.
    """
    Zs = _scaled(Z, scale)
    blocks = []

    def split(ids):
        n_blocks = -(-ids.size // block_size)
        if n_blocks <= 1:
            blocks.append(ids)
            return
        sub = Zs[ids]
        dim = int(np.argmax(sub.max(axis=0) - sub.min(axis=0)))
        n_left = block_size * (n_blocks // 2)
        part = np.argpartition(sub[:, dim], n_left - 1)
        split(ids[part[:n_left]])
        split(ids[part[n_left:]])

    split(np.arange(Zs.shape[0]))
    out = np.full((len(blocks), block_size), -1, dtype=np.int64)
    for b, ids in enumerate(blocks):
        out[b, : ids.size] = np.sort(ids)
    return out


def _scaled(Z, scale):
    Z = np.asarray(Z, dtype=np.float64)
    if scale is None:
        return Z
    return Z / np.asarray(scale, dtype=np.float64)
