from collections import namedtuple

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.traversal import ancestors
from scipy.spatial import cKDTree

from ptgp.conditionals import _DEFAULT_JITTER, nn_conditional
from ptgp.gp.base import PredictSpec
from ptgp.kl import nn_kl_terms
from ptgp.mean import Zero
from ptgp.neighbors import (
    _scaled,
    correlation_dist_fn,
    euclidean_dist_fn,
    find_neighbors,
    order_maximin,
    order_random,
    partition_blocks,
    query_neighbors,
)
from ptgp.objectives import vnngp_elbo

NeighborUpdate = namedtuple("NeighborUpdate", ["frac_changed", "block_coverage"])


class BlockVariationalParams:
    """Block-diagonal Gaussian ``q(u)`` over the inducing values of a VNNGP.

    Points are grouped into blocks of ``block_size``; each block has a full
    covariance through a lower-triangular factor with softplus diagonal, and
    covariances across blocks are zero. ``block_size=1`` is the mean-field
    family of Wu, Pleiss & Cunningham (2022, arXiv:2202.01694, Eq 10-11).

    Attributes
    ----------
    q_mu : TensorVariable, shape (M,)
        Variational means.
    q_L : TensorVariable, shape (n_blocks, block_size, block_size)
        Per-block Cholesky factors.
    blocks : ndarray of int, shape (n_blocks, block_size)
        Point ids per block, padded with -1.
    extra_vars, extra_init : tuple
        Trainable leaves and their initial values.
    """

    def __init__(self, blocks, init_sd=1e-2):
        blocks = np.asarray(blocks, dtype=np.int64)
        n_blocks, B = blocks.shape
        real = blocks >= 0
        M = int(real.sum())
        floatX = pytensor.config.floatX

        self.blocks = blocks
        self.block_size = B
        self.block_of = np.empty(M, dtype=np.int64)
        self.pos_of = np.empty(M, dtype=np.int64)
        b_idx, p_idx = np.nonzero(real)
        self.block_of[blocks[real]] = b_idx
        self.pos_of[blocks[real]] = p_idx

        rows, cols = np.tril_indices(B)
        n_tri = rows.size
        self.q_mu = pt.vector("q_mu", shape=(M,), dtype=floatX)
        q_L_flat = pt.matrix("q_L_flat", shape=(n_blocks, n_tri), dtype=floatX)
        L = pt.set_subtensor(pt.zeros((n_blocks, B, B), dtype=floatX)[:, rows, cols], q_L_flat)
        diag = np.arange(B)
        self.q_L = pt.set_subtensor(L[:, diag, diag], pt.softplus(L[:, diag, diag]))

        sd = np.broadcast_to(np.asarray(init_sd, dtype=np.float64), (M,))
        flat_init = np.zeros((n_blocks, n_tri))
        diag_pos = np.cumsum(np.arange(1, B + 1)) - 1
        sd_blocks = np.ones((n_blocks, B))
        sd_blocks[real] = sd[blocks[real]]
        flat_init[:, diag_pos] = np.log(np.expm1(sd_blocks))

        self.extra_vars = (self.q_mu, q_L_flat)
        self.extra_init = (np.zeros(M, dtype=floatX), flat_init.astype(floatX))

    def local_moments(self, idx):
        """q means ``(b, P)`` and covariance ``(b, P, P)`` of the points in ``idx``."""
        blk = pt.as_tensor_variable(self.block_of)[idx]
        pos = pt.as_tensor_variable(self.pos_of)[idx]
        rows = self.q_L[blk, pos]
        same_block = pt.eq(blk[..., :, None], blk[..., None, :])
        S = (rows @ pt.swapaxes(rows, -1, -2)) * same_block
        return self.q_mu[idx], S

    def log_diag(self, idx):
        """Log of each point's diagonal entry in its block's Cholesky factor."""
        blk = pt.as_tensor_variable(self.block_of)[idx]
        pos = pt.as_tensor_variable(self.pos_of)[idx]
        return pt.log(self.q_L[blk, pos, pos])


class VNNGP:
    """Variational Nearest Neighbor Gaussian Process.

    Inducing points are the unique rows of the training inputs. Each inducing
    value conditions on its ``k`` nearest preceding points in an ordering
    (Wu, Pleiss & Cunningham 2022, arXiv:2202.01694), so both the data term
    and the KL term of the ELBO can be minibatched at ``O(k^3)`` per point.

    Parameters
    ----------
    kernel : Kernel
        Covariance function.
    X : ndarray, shape (N, D)
        Training inputs. ``Z`` is their unique rows; ``data_to_z`` maps each
        row of ``X`` to its row of ``Z``.
    likelihood : LikelihoodVariable
        Observation likelihood.
    k : int
        Number of neighbors.
    mean : callable, optional
        Mean function (default: ``Zero()``).
    block_size : int
        Block size of the variational covariance; 1 is mean-field.
    init_sd : float or ndarray, shape (M,)
        Initial variational standard deviation. A small multiple of the prior
        standard deviation works well, e.g. ``1e-2 * eta``.
    order : {"random", "maximin"} or ndarray
        Ordering of ``Z``.
    scale : float or ndarray, shape (D,), optional
        Divides ``Z`` for the initial Euclidean neighbor search and for block
        partitioning. Use :meth:`recompute_neighbors` to update the metric
        from trained hyperparameters.
    jitter : float
        Diagonal jitter for the neighbor solves.
    seed : int, optional
        Seed for the random ordering.
    """

    default_objective = staticmethod(vnngp_elbo)

    def __init__(
        self,
        kernel,
        X,
        likelihood,
        k,
        mean=None,
        block_size=1,
        init_sd=1e-2,
        order="random",
        scale=None,
        jitter=_DEFAULT_JITTER,
        seed=None,
    ):
        X = np.asarray(X, dtype=pytensor.config.floatX)
        Z, data_to_z = np.unique(X, axis=0, return_inverse=True)
        M = Z.shape[0]
        if k >= M:
            raise ValueError(
                f"k={k} must be smaller than the number of unique inputs ({M}). "
                "Reduce k or add data."
            )
        self.kernel = kernel
        self.likelihood = likelihood
        self.mean = mean if mean is not None else Zero()
        self.k = int(k)
        self.jitter = jitter
        self.Z = Z
        self.data_to_z = data_to_z.reshape(-1).astype(np.int64)
        self.n_data = X.shape[0]
        self.num_inducing = M
        self._seed = seed

        blocks = (
            np.arange(M)[:, None] if block_size == 1 else partition_blocks(Z, block_size, scale)
        )
        self.variational_params = BlockVariationalParams(blocks, init_sd=init_sd)

        self.order = self._make_order(order, euclidean_dist_fn(Z, scale))
        nn_idx, nn_mask = find_neighbors(self.order, self.k, Z=Z, scale=scale)
        self.nn_idx = pytensor.shared(nn_idx, name="_nn_idx", shape=nn_idx.shape)
        self.nn_mask = pytensor.shared(
            nn_mask.astype(pytensor.config.floatX), name="_nn_mask", shape=nn_mask.shape
        )
        self._predict_metric = {
            "tree": cKDTree(_scaled(Z, scale)),
            "scale": scale,
            "corr_rows": None,
            "snapshot": {},
        }

    @property
    def extra_vars(self):
        return self.variational_params.extra_vars

    @property
    def extra_init(self):
        return self.variational_params.extra_init

    def _make_order(self, order, dist_fn):
        M = self.num_inducing
        if isinstance(order, str):
            if order == "random":
                return order_random(M, self._seed)
            if order == "maximin":
                center = int(np.argmin(np.sum((self.Z - self.Z.mean(axis=0)) ** 2, axis=1)))
                return order_maximin(dist_fn, M, first=center)
            raise ValueError(
                f"order must be 'random', 'maximin', or an index array; got {order!r}. "
                "Pass one of these."
            )
        order = np.asarray(order, dtype=np.int64)
        if not np.array_equal(np.sort(order), np.arange(M)):
            raise ValueError(
                f"order must be a permutation of range({M}). "
                "Inducing points are the unique rows of X."
            )
        return order

    def _Z(self):
        return pt.as_tensor_variable(self.Z)

    def _prior_mean_at(self, idx):
        Zi = self._Z()[idx]
        flat = Zi.reshape((-1, Zi.shape[-1]))
        return self.mean(flat).reshape(idx.shape)

    def data_marginals(self, row_idx):
        """q mean and variance of the latent f at training rows ``row_idx``."""
        z = pt.as_tensor_variable(self.data_to_z)[row_idx]
        means, S = self.variational_params.local_moments(z[:, None])
        return means[:, 0], S[:, 0, 0]

    def prior_conditional(self, kl_idx):
        """Neighbor weights ``b`` (B, k) and conditional variances ``F`` (B,) of ``u[kl_idx]``."""
        nbr = pt.specify_shape(self.nn_idx[kl_idx], (None, self.k))
        Zc = self._Z()
        return nn_conditional(self.kernel, Zc[kl_idx], Zc[nbr], self.nn_mask[kl_idx], self.jitter)

    def prior_kl_terms(self, kl_idx):
        """Per-point KL terms of ``q(u)`` against the nearest-neighbor prior at ``kl_idx``."""
        b, F = self.prior_conditional(kl_idx)
        local = pt.concatenate([kl_idx[:, None], self.nn_idx[kl_idx]], axis=1)
        means, S = self.variational_params.local_moments(local)
        return nn_kl_terms(
            b,
            F,
            means,
            S,
            self._prior_mean_at(local),
            self.variational_params.log_diag(kl_idx),
        )

    def predict_marginal(self, X_new, nn_idx, incl_lik=False):
        """Posterior marginal mean and variance at ``X_new`` given its neighbor indices.

        Each test point conditions on its ``k`` nearest inducing points, the
        LF-ind scheme of Katzfuss et al. (2020, arXiv:1805.03309).

        Parameters
        ----------
        X_new : tensor, shape (N*, D)
        nn_idx : tensor of int, shape (N*, k)
            Nearest inducing points of each row, from :meth:`predict_spec`'s
            ``prepare`` or :func:`ptgp.neighbors.query_neighbors`.
        incl_lik : bool
            If True, push through the likelihood's predictive mean/var.

        Returns
        -------
        mean : tensor, shape (N*,)
        var : tensor, shape (N*,)
        """
        nn_idx = pt.specify_shape(nn_idx, (None, self.k))
        mask = pt.ones(nn_idx.shape, dtype=X_new.dtype)
        b, F = nn_conditional(self.kernel, X_new, self._Z()[nn_idx], mask, self.jitter)
        means, S = self.variational_params.local_moments(nn_idx)
        resid = means - self._prior_mean_at(nn_idx)
        fmean = self.mean(X_new) + pt.sum(b * resid, axis=1)
        fvar = F + pt.sum(b * pt.sum(S * b[:, None, :], axis=2), axis=1)
        if incl_lik:
            return self.likelihood.at(X_new).predict_mean_and_var(fmean, fvar)
        return fmean, fvar

    def predict_spec(self, X_new, incl_lik=False, X_train=None, y_train=None):
        """Prediction spec for :func:`ptgp.optim.compile_predict`.

        The compiled function takes ``(X_new, nn_idx)``; ``prepare`` finds the
        neighbors host side in the metric of the last :meth:`recompute_neighbors`
        call. ``X_train`` and ``y_train`` are not used.
        """
        nn_new = pt.matrix("_nn_idx_new", shape=(None, self.k), dtype="int64")
        mean, var = self.predict_marginal(X_new, nn_new, incl_lik=incl_lik)
        return PredictSpec([X_new, nn_new], (mean, var), self._prepare_prediction)

    def _prepare_prediction(self, X_new):
        metric = self._predict_metric
        for sv, value in metric["snapshot"].items():
            if not np.array_equal(sv.get_value(), value):
                raise ValueError(
                    "Neighbors were computed under different hyperparameters. "
                    "Call recompute_neighbors before predicting."
                )
        X_new = np.asarray(X_new)
        nn = query_neighbors(
            X_new,
            self.k,
            Z=self.Z,
            scale=metric["scale"],
            corr_rows=metric["corr_rows"],
            tree=metric["tree"],
        )
        return X_new, nn

    def recompute_neighbors(
        self,
        model,
        shared_params,
        extra_vars=None,
        shared_extras=None,
        metric="euclidean",
        scale=None,
        order=None,
    ):
        """Recompute the ordering and neighbor sets under trained hyperparameters.

        Gradients ignore how neighbors depend on hyperparameters; the sets are
        refreshed between optimizer steps on a schedule such as
        :func:`recompute_steps` (Katzfuss, Guinness & Lawrence, arXiv:2005.00386,
        Sec 3.2; Kang & Katzfuss 2023, arXiv:2112.14591, Sec 3.3). ``q(u)`` is
        indexed by point and blocks are fixed, so it carries over unchanged;
        the ELBO jumps because the prior changes. Recompute once more after
        training: predictions use this metric, and ``compile_predict``
        functions refuse to run if the hyperparameters it depends on change.

        Parameters
        ----------
        model : pm.Model
        shared_params : dict
            ``{value_var: shared_var}`` from training.
        extra_vars, shared_extras : sequence, optional
            As in :func:`ptgp.optim.compile_predict`.
        metric : {"euclidean", "correlation"}
            ``"euclidean"`` uses ``Z / scale``; ``"correlation"`` uses
            ``sqrt(1 - |rho|)`` from the kernel (Kang & Katzfuss 2023, Eq 4).
        scale : float, ndarray, or tensor, optional
            Euclidean scale; a symbolic expression such as the ARD ``ls`` is
            evaluated at the trained values.
        order : {"random", "maximin"} or ndarray, optional
            New ordering. Defaults to the current one.

        Returns
        -------
        NeighborUpdate
            ``frac_changed``: fraction of points whose neighbor set changed.
            ``block_coverage``: fraction of neighbor pairs sharing a q block.
        """
        from ptgp.optim.training import _replace_graph

        if extra_vars is None:
            extra_vars = self.extra_vars

        def evaluate(outputs, inputs=()):
            replaced = _replace_graph(outputs, model, shared_params, extra_vars, shared_extras)
            fn = pytensor.function(list(inputs), replaced, on_unused_input="ignore")
            shared = {
                v
                for v in ancestors(replaced)
                if isinstance(v, SharedVariable) and v not in (self.nn_idx, self.nn_mask)
            }
            return fn, shared

        corr_rows_z = corr_rows_x = None
        if metric == "correlation":
            A = pt.matrix("_A", shape=(None, self.Z.shape[1]), dtype=self.Z.dtype)
            Zc = self._Z()
            denom = pt.sqrt(self.kernel.diag(A)[:, None] * self.kernel.diag(Zc)[None, :])
            corr = pt.abs(self.kernel(A, Zc)) / denom
            (corr_fn, deps) = evaluate([corr], [A])
            corr_rows_x = lambda Xc: corr_fn(Xc)[0]  # noqa: E731
            corr_rows_z = lambda idx: corr_fn(self.Z[idx])[0]  # noqa: E731
            dist_fn = correlation_dist_fn(corr_rows_z)
            scale_value = None
        elif metric == "euclidean":
            if isinstance(scale, pt.TensorVariable):
                scale_fn, deps = evaluate([scale])
                scale_value = np.asarray(scale_fn()[0])
            else:
                deps, scale_value = set(), scale
            dist_fn = euclidean_dist_fn(self.Z, scale_value)
        else:
            raise ValueError(
                f"metric must be 'euclidean' or 'correlation'; got {metric!r}. "
                "Pass one of these."
            )

        if order is not None:
            self.order = self._make_order(order, dist_fn)
        old_idx, old_mask = self.nn_idx.get_value(), self.nn_mask.get_value() > 0
        nn_idx, nn_mask = find_neighbors(
            self.order, self.k, Z=self.Z, scale=scale_value, corr_rows=corr_rows_z
        )
        self.nn_idx.set_value(nn_idx)
        self.nn_mask.set_value(nn_mask.astype(self.nn_mask.dtype))

        self._predict_metric = {
            "tree": None if corr_rows_x else cKDTree(_scaled(self.Z, scale_value)),
            "scale": scale_value,
            "corr_rows": corr_rows_x,
            "snapshot": {sv: np.array(sv.get_value(), copy=True) for sv in deps},
        }

        old = np.where(old_mask, old_idx, -1)
        new = np.where(nn_mask, nn_idx, -1)
        changed = np.mean(np.any(np.sort(old, axis=1) != np.sort(new, axis=1), axis=1))
        blk = self.variational_params.block_of
        n_pairs = nn_mask.sum()
        same = (blk[nn_idx] == blk[:, None]) & nn_mask
        coverage = float(same.sum() / n_pairs) if n_pairs else 1.0
        return NeighborUpdate(frac_changed=float(changed), block_coverage=coverage)


def recompute_steps(n_steps, start=2):
    """Iterations ``start, 2 * start, 4 * start, ...`` below ``n_steps`` at which to recompute neighbors.

    The geometric schedule of Katzfuss, Guinness & Lawrence (arXiv:2005.00386,
    Sec 3.2) and GPBoost. Recompute once more after the last step.
    """
    steps = []
    s = start
    while s < n_steps:
        steps.append(s)
        s *= 2
    return steps
