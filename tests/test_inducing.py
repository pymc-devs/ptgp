"""Tests for inducing point initialization strategies."""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from ptgp import assume
from ptgp.inducing import (
    KernelHealthDiagnostics,
    KMeansDiagnostics,
    Points,
    RandomSubsampleDiagnostics,
    compute_inducing_diagnostics,
    greedy_variance_init,
    kmeans_init,
    random_subsample_init,
)
from ptgp.kernels import ExpQuad


def _eval_kernel(kernel, X, Y=None):
    X_pt = pt.as_tensor_variable(X)
    Y_pt = pt.as_tensor_variable(Y) if Y is not None else None
    return pytensor.function([], kernel(X_pt, Y_pt))()


def _nystrom_residual_trace(X, Z, kernel, jitter=1e-9):
    """tr(K(X,X) - K(X,Z) K(Z,Z)^{-1} K(Z,X))."""
    Kxx = _eval_kernel(kernel, X)
    Kxz = _eval_kernel(kernel, X, Z)
    Kzz = _eval_kernel(kernel, Z) + jitter * np.eye(len(Z))
    Q = Kxz @ np.linalg.solve(Kzz, Kxz.T)
    return float(np.trace(Kxx - Q))


@pytest.fixture
def gaussian_blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[-3.0, -3.0], [3.0, -3.0], [0.0, 3.0]])
    X = np.vstack([c + 0.3 * rng.standard_normal((50, 2)) for c in centers])
    return X, centers


class TestRandomSubsample:
    def test_returns_inducing_points(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        ip, diag = random_subsample_init(X, 10, rng=0)
        assert isinstance(ip, Points)
        assert isinstance(ip.Z, np.ndarray)
        assert ip.Z.shape == (10, 2)
        assert isinstance(diag, RandomSubsampleDiagnostics)

    def test_rows_are_in_x(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        ip, _ = random_subsample_init(X, 10, rng=0)
        for z in ip.Z:
            assert np.any(np.all(X == z, axis=1))

    def test_no_duplicates(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        ip, _ = random_subsample_init(X, 30, rng=0)
        assert len(np.unique(ip.Z, axis=0)) == 30

    def test_reproducible(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        ip1, _ = random_subsample_init(X, 10, rng=42)
        ip2, _ = random_subsample_init(X, 10, rng=42)
        np.testing.assert_array_equal(ip1.Z, ip2.Z)

    def test_rejects_m_too_large(self):
        X = np.random.default_rng(0).standard_normal((10, 2))
        with pytest.raises(ValueError, match="exceeds"):
            random_subsample_init(X, 20)

    def test_diagnostics_fields(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        _, diag = random_subsample_init(X, 10, rng=0)
        assert diag.M_requested == 10
        assert diag.M_returned == 10
        assert diag.N_candidates == 50
        assert diag.n_unique == 10
        assert diag.pairwise_min_distance > 0
        assert diag.pairwise_mean_distance > diag.pairwise_min_distance
        assert diag.kernel_health is None
        # repr must produce non-empty text
        assert "M requested" in repr(diag)

    def test_kernel_health_attached(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        kernel = ExpQuad(input_dim=2, ls=1.0)
        _, diag = random_subsample_init(X, 10, rng=0, kernel=kernel)
        assert isinstance(diag.kernel_health, KernelHealthDiagnostics)
        kh = diag.kernel_health
        assert kh.d_final.shape == (50,)
        assert kh.total_variance > 0
        assert kh.nystrom_residual >= 0
        assert kh.kuu_min_eigenvalue > 0
        assert kh.kuu_condition_number >= 1
        # one-line kernel-health summary appears in the parent repr
        assert "kernel health" in repr(diag)


class TestKmeans:
    def test_returns_inducing_points(self, gaussian_blobs):
        X, _ = gaussian_blobs
        ip, diag = kmeans_init(X, 3, rng=0)
        assert isinstance(ip, Points)
        assert isinstance(ip.Z, np.ndarray)
        assert ip.Z.shape == (3, 2)
        assert isinstance(diag, KMeansDiagnostics)

    def test_recovers_cluster_centers(self, gaussian_blobs):
        """On well-separated blobs, centroids should lie near the true centers."""
        X, centers = gaussian_blobs
        ip, _ = kmeans_init(X, 3, rng=0)
        dists = np.linalg.norm(ip.Z[:, None, :] - centers[None, :, :], axis=-1)
        assert np.all(dists.min(axis=0) < 0.5)

    def test_rejects_m_too_large(self):
        X = np.random.default_rng(0).standard_normal((5, 2))
        with pytest.raises(ValueError, match="exceeds"):
            kmeans_init(X, 20)

    def test_diagnostics_fields(self, gaussian_blobs):
        X, _ = gaussian_blobs
        _, diag = kmeans_init(X, 3, rng=0)
        assert diag.M_requested == 3
        assert diag.M_returned == 3
        assert diag.n_removed_duplicates == 0
        assert diag.dedup_tol == pytest.approx(1e-6)
        assert diag.inertia > 0
        assert diag.pairwise_min_distance > 0
        assert diag.pairwise_min_distance > diag.dedup_tol
        assert "M requested" in repr(diag)

    def test_diagnostics_dedup(self, gaussian_blobs):
        """Forcing M much larger than the natural cluster count triggers dedup."""
        X, _ = gaussian_blobs  # 3 well-separated blobs, 50 points each
        # Asking for 30 clusters from 3 blobs with a tol larger than the
        # natural intra-blob spacing forces near-duplicates that get
        # deduplicated.
        _, diag = kmeans_init(X, 30, rng=0, tol=0.5)
        assert diag.M_requested == 30
        assert diag.M_returned <= 30
        assert diag.n_removed_duplicates == diag.M_requested - diag.M_returned
        # Some dedup should happen at this M and tol on tight blobs.
        assert diag.n_removed_duplicates > 0

    def test_kernel_health_attached(self, gaussian_blobs):
        X, _ = gaussian_blobs
        kernel = ExpQuad(input_dim=2, ls=1.0)
        _, diag = kmeans_init(X, 3, rng=0, kernel=kernel)
        assert isinstance(diag.kernel_health, KernelHealthDiagnostics)
        kh = diag.kernel_health
        assert kh.d_final.shape == (X.shape[0],)
        assert kh.kuu_min_eigenvalue > 0
        assert "kernel health" in repr(diag)


class TestComputeInducingDiagnostics:
    def test_returns_kernel_health(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 2))
        Z = X[:10]
        kernel = ExpQuad(input_dim=2, ls=1.0)
        diag = compute_inducing_diagnostics(kernel, X, Z)
        assert isinstance(diag, KernelHealthDiagnostics)
        assert diag.d_final.shape == (50,)
        assert diag.total_variance > 0
        assert diag.nystrom_residual >= 0
        assert diag.kuu_condition_number >= 1
        assert "variance explained" in repr(diag)


class TestGreedyVariance:
    def test_returns_inducing_points(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        kernel = ExpQuad(input_dim=2, ls=1.0)
        ip, diag = greedy_variance_init(X, 10, kernel, rng=0)
        assert isinstance(ip, Points)
        assert isinstance(ip.Z, np.ndarray)
        assert ip.Z.shape == (10, 2)
        assert hasattr(diag, "trace_curve")
        assert hasattr(diag, "d_final")
        assert hasattr(diag, "total_variance")
        assert hasattr(diag, "kuu_min_eigenvalue")
        assert hasattr(diag, "kuu_condition_number")

    def test_rows_are_in_x(self):
        X = np.random.default_rng(0).standard_normal((50, 2))
        kernel = ExpQuad(input_dim=2, ls=1.0)
        ip, _ = greedy_variance_init(X, 15, kernel, rng=0)
        for z in ip.Z:
            assert np.any(np.all(np.isclose(X, z), axis=1))

    def test_beats_random_on_average(self):
        """Greedy space-filling selection yields lower Nystrom residual than random."""
        X = np.linspace(0.0, 10.0, 100)[:, None]
        kernel = ExpQuad(input_dim=1, ls=1.0)
        M = 10

        ip, _ = greedy_variance_init(X, M, kernel, rng=0)
        greedy_err = _nystrom_residual_trace(X, ip.Z, kernel)
        random_errs = [
            _nystrom_residual_trace(X, random_subsample_init(X, M, rng=s)[0].Z, kernel)
            for s in range(10)
        ]
        assert greedy_err < np.mean(random_errs)

    def test_threshold_terminates_early(self):
        """With a large threshold, fewer than M points should be returned."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 2))
        kernel = ExpQuad(input_dim=2, ls=1.0)
        ip, _ = greedy_variance_init(X, 25, kernel, threshold=1e6, rng=0)
        assert ip.Z.shape[0] < 25

    def test_rejects_non_kernel(self):
        X = np.random.default_rng(0).standard_normal((20, 2))
        with pytest.raises(TypeError, match="ptgp Kernel"):
            greedy_variance_init(X, 5, kernel=lambda X, Y=None: X @ X.T)

    def test_rejects_m_too_large(self):
        X = np.random.default_rng(0).standard_normal((5, 2))
        kernel = ExpQuad(input_dim=2, ls=1.0)
        with pytest.raises(ValueError, match="exceeds"):
            greedy_variance_init(X, 20, kernel)


class TestIntegrationWithSVGP:
    """Sanity check: output feeds into SVGP without hand-wrapping in pt.as_tensor_variable."""

    def test_numpy_z_flows_into_kernel(self):
        from ptgp.gp import SVGP, VariationalParams
        from ptgp.likelihoods import Gaussian
        from ptgp.objectives import elbo

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 1))
        y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(40)
        kernel = ExpQuad(input_dim=1, ls=1.0)

        ip, _ = greedy_variance_init(X, 5, kernel, rng=0)
        vp = VariationalParams(
            q_mu=pt.zeros(5),
            q_sqrt=assume(pt.eye(5), lower_triangular=True),
        )
        svgp = SVGP(
            kernel=kernel,
            likelihood=Gaussian(sigma=0.1),
            inducing_variable=ip,
            variational_params=vp,
        )
        val = pytensor.function(
            [], elbo(svgp, pt.as_tensor_variable(X), pt.as_tensor_variable(y)).elbo
        )()
        assert np.isfinite(val)
