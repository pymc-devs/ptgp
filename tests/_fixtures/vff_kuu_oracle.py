"""Numpy port of the VFF Kuu oracle for Matern{12,32,52}.

Ported from st--/VFF (https://github.com/st--/VFF), which is itself derived from
the GPflow VFF reference by Hensman, Durrande, and Solin (2017). Original
licence: Apache-2.0.

This oracle computes the dense (M, M) Kuu matrix for **unit-variance** Matern
kernels in **block ordering**:

    rows 0..K    : cosine basis at omegas[0..K]   (omega_0 = 0 included)
    rows K+1..2K : sine basis at omegas[1..K]     (omega_0 row dropped)

so M = (K + 1) + K = 2*K + 1 with ms = arange(K + 1).

The structured form is Kuu = diag(d) + U @ U.T. This module returns the dense
matrix only; tests reconstruct the structured form internally.
"""

from __future__ import annotations

import numpy as np


def _omegas(a, b, ms):
    return 2.0 * np.pi * np.asarray(ms, dtype=float) / (b - a)


def oracle_kuf_no_edges(a, b, ms, X):
    """Dense ``Kuf`` (M, N) for unit-variance Matern in block ordering, X in [a, b].

    Mirrors ``make_Kuf_no_edges`` from st--/VFF; the basis is the same for all
    Matern{12,32,52} on the interior of the domain. Out-of-domain handling is
    delegated to the domain wrapper at compile time.
    """
    omegas = _omegas(a, b, ms)
    x = np.asarray(X, dtype=float).ravel() - a
    cos_block = np.cos(omegas[:, None] * x[None, :])  # (K+1, N)
    omegas_sin = omegas[omegas != 0]
    sin_block = np.sin(omegas_sin[:, None] * x[None, :])  # (K, N)
    return np.vstack([cos_block, sin_block])


def oracle_kuu_matern12(a, b, ms, ls):
    """Dense Kuu for unit-variance Matern12 in block ordering."""
    omegas = _omegas(a, b, ms)
    lamb = 1.0 / ls
    two_or_four = np.where(omegas == 0, 2.0, 4.0)
    d_cos = (b - a) * (lamb**2 + omegas**2) / lamb / two_or_four
    v_cos = np.ones_like(d_cos)

    omegas_sin = omegas[omegas != 0]
    d_sin = (b - a) * (lamb**2 + omegas_sin**2) / lamb / 4.0

    # Block-diag (Rank1Mat(d_cos, v_cos), DiagMat(d_sin))
    Kcos = np.diag(d_cos) + np.outer(v_cos, v_cos)
    Ksin = np.diag(d_sin)
    M = d_cos.shape[0] + d_sin.shape[0]
    Kuu = np.zeros((M, M))
    Kuu[: d_cos.shape[0], : d_cos.shape[0]] = Kcos
    Kuu[d_cos.shape[0] :, d_cos.shape[0] :] = Ksin
    return Kuu


def oracle_kuu_matern32(a, b, ms, ls):
    """Dense Kuu for unit-variance Matern32 in block ordering."""
    omegas = _omegas(a, b, ms)
    lamb = np.sqrt(3.0) / ls
    four_or_eight = np.where(omegas == 0, 4.0, 8.0)
    d_cos = (b - a) * (lamb**2 + omegas**2) ** 2 / lamb**3 / four_or_eight
    v_cos = np.ones_like(d_cos)

    omegas_sin = omegas[omegas != 0]
    d_sin = (b - a) * (lamb**2 + omegas_sin**2) ** 2 / lamb**3 / 8.0
    v_sin = omegas_sin / lamb

    Kcos = np.diag(d_cos) + np.outer(v_cos, v_cos)
    Ksin = np.diag(d_sin) + np.outer(v_sin, v_sin)
    M = d_cos.shape[0] + d_sin.shape[0]
    Kuu = np.zeros((M, M))
    Kuu[: d_cos.shape[0], : d_cos.shape[0]] = Kcos
    Kuu[d_cos.shape[0] :, d_cos.shape[0] :] = Ksin
    return Kuu


def oracle_kuu_matern52(a, b, ms, ls):
    """Dense Kuu for unit-variance Matern52 in block ordering."""
    omegas = _omegas(a, b, ms)
    lamb = np.sqrt(5.0) / ls
    sixteen_or_32 = np.where(omegas == 0, 16.0, 32.0)
    v1 = (3.0 * (omegas / lamb) ** 2 - 1.0) / np.sqrt(8.0)
    v2 = np.ones_like(v1)
    W_cos = np.stack([v1, v2], axis=1)  # (K+1, 2)
    d_cos = 3.0 * (b - a) / sixteen_or_32 / lamb**5 * (lamb**2 + omegas**2) ** 3

    omegas_sin = omegas[omegas != 0]
    v_sin = np.sqrt(3.0) * omegas_sin / lamb
    d_sin = 3.0 * (b - a) / 32.0 / lamb**5 * (lamb**2 + omegas_sin**2) ** 3

    Kcos = np.diag(d_cos) + W_cos @ W_cos.T
    Ksin = np.diag(d_sin) + np.outer(v_sin, v_sin)
    M = d_cos.shape[0] + d_sin.shape[0]
    Kuu = np.zeros((M, M))
    Kuu[: d_cos.shape[0], : d_cos.shape[0]] = Kcos
    Kuu[d_cos.shape[0] :, d_cos.shape[0] :] = Ksin
    return Kuu
