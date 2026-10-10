import numpy as np
import pytest

from ptgp.neighbors import (
    correlation_dist_fn,
    euclidean_dist_fn,
    find_neighbors,
    order_maximin,
    order_random,
    partition_blocks,
    query_neighbors,
)

M, D, K = 150, 2, 6


def _points(seed=0):
    return np.random.default_rng(seed).uniform(size=(M, D))


def _brute_predecessors(Zs, order, k):
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    out = []
    for j in range(len(order)):
        preds = np.flatnonzero(rank < rank[j])
        d = np.sum((Zs[preds] - Zs[j]) ** 2, axis=1)
        out.append(set(preds[np.argsort(d)][:k].tolist()))
    return out


@pytest.mark.parametrize("ordering", ["random", "maximin"])
def test_euclidean_predecessors_match_brute_force(ordering):
    Z = _points()
    scale = np.array([0.5, 2.0])
    order = (
        order_random(M, seed=1)
        if ordering == "random"
        else order_maximin(euclidean_dist_fn(Z, scale), M)
    )
    nn_idx, nn_mask = find_neighbors(order, K, Z=Z, scale=scale)
    ref = _brute_predecessors(Z / scale, order, K)
    for j in range(M):
        assert set(nn_idx[j][nn_mask[j]].tolist()) == ref[j]
    rank = np.empty(M, int)
    rank[order] = np.arange(M)
    assert np.array_equal(nn_mask.sum(axis=1), np.minimum(rank, K))


def test_correlation_metric_matches_scaled_euclidean_for_ard():
    """Kang & Katzfuss (2023) Prop 1: ARD ExpQuad correlation neighbors are Euclidean on Z / ls."""
    Z = _points()
    ls = np.array([0.3, 1.5])

    def corr_rows(idx):
        d2 = np.sum(((Z[idx][:, None, :] - Z[None, :, :]) / ls) ** 2, axis=-1)
        return np.exp(-0.5 * d2)

    order = order_random(M, seed=2)
    nn_c, mask_c = find_neighbors(order, K, corr_rows=corr_rows)
    nn_e, mask_e = find_neighbors(order, K, Z=Z, scale=ls)
    assert np.array_equal(mask_c, mask_e)
    assert np.array_equal(np.where(mask_c, nn_c, -1), np.where(mask_e, nn_e, -1))

    first = order_maximin(correlation_dist_fn(corr_rows), M)
    assert np.array_equal(first, order_maximin(euclidean_dist_fn(Z, ls), M))


def test_maximin_picks_farthest_point():
    Z = _points()
    order = order_maximin(euclidean_dist_fn(Z), M)
    assert np.array_equal(np.sort(order), np.arange(M))
    for r in range(1, 20):
        prev = Z[order[:r]]
        mind = np.min(np.sum((Z[:, None, :] - prev[None]) ** 2, axis=-1), axis=1)
        mind[order[:r]] = -np.inf
        assert np.isclose(mind[order[r]], mind.max())


def test_query_neighbors_match_brute_force():
    Z = _points()
    X_new = np.random.default_rng(3).uniform(size=(20, D))
    idx = query_neighbors(X_new, K, Z=Z)
    d = np.sum((X_new[:, None, :] - Z[None]) ** 2, axis=-1)
    np.testing.assert_array_equal(idx, np.argsort(d, axis=1)[:, :K])


def test_partition_blocks_covers_points_once():
    Z = _points()
    blocks = partition_blocks(Z, 16)
    ids = blocks[blocks >= 0]
    assert np.array_equal(np.sort(ids), np.arange(M))
    assert np.sum((blocks >= 0).sum(axis=1) < 16) <= 1
