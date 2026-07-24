import numpy as np
import pytensor.tensor as pt

from ptgp.likelihoods.base import Likelihood


class CompositeLikelihood(Likelihood):
    """Apply a different likelihood to different observations.

    The data is partitioned into disjoint subsets, each governed by its own
    sub-likelihood. Because the ELBO data term is a sum over independent points,
    the combined ``variational_expectation`` is just the per-subset expectations
    scattered back into a length-N vector, and similarly for prediction. This
    composes directly with the VGP (and SVGP) objectives.

    Parameters
    ----------
    likelihoods : list of Likelihood
        One sub-likelihood per subset.
    indices : list of array-like of int
        Integer index arrays, one per sub-likelihood, that partition
        ``range(N)`` (disjoint, exhaustive, non-empty). ``indices[k]`` selects
        the observations governed by ``likelihoods[k]``.

    Notes
    -----
    A sub-likelihood reads its own parameters internally. If a sub-likelihood
    carries a per-point parameter (for example a vector ``sigma`` on a
    heteroskedastic :class:`~ptgp.likelihoods.Gaussian`), that parameter must
    already be aligned to its subset, since the combinator slices only ``y``,
    ``mu``, and ``var``, not the sub-likelihood's internal tensors.
    """

    def __init__(self, likelihoods, indices):
        """Validate that ``indices`` partition ``range(N)`` and store the parts."""
        if len(likelihoods) != len(indices):
            raise ValueError(
                f"likelihoods and indices must have equal length; got "
                f"{len(likelihoods)} and {len(indices)}."
            )
        idx_arrays = [np.asarray(idx, dtype=np.intp) for idx in indices]
        for a in idx_arrays:
            if a.ndim != 1 or a.size == 0:
                raise ValueError("each entry of indices must be a non-empty 1D integer array.")
        concat = np.concatenate(idx_arrays)
        N = concat.size
        if not np.array_equal(np.sort(concat), np.arange(N)):
            raise ValueError(
                "indices must partition range(N): disjoint, exhaustive, and covering 0..N-1."
            )
        self.likelihoods = list(likelihoods)
        self._indices = idx_arrays
        self.num_data = N

    def variational_expectation(self, y, mu, var):
        """Per-point E_{q(f)}[log p(y|f)], dispatched by subset.

        Returns a length-N vector; the VGP/SVGP objective sums it.
        """
        out = pt.zeros_like(mu)
        for idx, lik in zip(self._indices, self.likelihoods):
            out = pt.set_subtensor(out[idx], lik.variational_expectation(y[idx], mu[idx], var[idx]))
        return out

    def predict_mean_and_var(self, mu, var):
        """Predictive mean and variance, dispatched by subset."""
        mean_out = pt.zeros_like(mu)
        var_out = pt.zeros_like(var)
        for idx, lik in zip(self._indices, self.likelihoods):
            m_i, v_i = lik.predict_mean_and_var(mu[idx], var[idx])
            mean_out = pt.set_subtensor(mean_out[idx], m_i)
            var_out = pt.set_subtensor(var_out[idx], v_i)
        return mean_out, var_out

    def predict_log_density(self, y, mu, var):
        """Predictive log-density at test points, dispatched by subset."""
        out = pt.zeros_like(mu)
        for idx, lik in zip(self._indices, self.likelihoods):
            out = pt.set_subtensor(out[idx], lik.predict_log_density(y[idx], mu[idx], var[idx]))
        return out
