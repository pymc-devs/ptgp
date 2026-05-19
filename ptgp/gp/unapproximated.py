import pytensor.tensor as pt

from ptgp.likelihoods import Gaussian
from ptgp.mean import Zero


class Unapproximated:
    """Exact (unapproximated) Gaussian process.

    The observation model is Gaussian; parameterize the noise via ``sigma``.

    Parameters
    ----------
    kernel : Kernel
        Covariance function.
    mean : callable, optional
        Mean function (default: ``Zero()``).
    sigma : tensor or PyMC random variable
        Observation noise standard deviation.
    """

    def __init__(self, kernel, mean=None, sigma=None):
        """Store the kernel and mean; build a Gaussian likelihood from sigma."""
        self.kernel = kernel
        self.mean = mean if mean is not None else Zero()
        self.likelihood = Gaussian(sigma)

    def predict_marginal(self, X_new, X_train, y_train, incl_lik=False):
        """Posterior marginal mean and variance at each point in X_new.

        Returns the per-point posterior; correlations between test points
        are discarded.

        Parameters
        ----------
        X_new : tensor, shape (N*, D)
        X_train : tensor, shape (N, D)
        y_train : tensor, shape (N,)
        incl_lik : bool
            If True, include likelihood noise in the predictions.

        Returns
        -------
        mean : tensor, shape (N*,)
        var : tensor, shape (N*,)
        """
        Knn = self.kernel(X_train)
        Knn_noisy = Knn + self.likelihood.sigma**2 * pt.eye(X_train.shape[0])
        Kns = self.kernel(X_train, X_new)  # (N, N*)
        Kss_diag = self.kernel.diag(X_new)

        Knn_inv = pt.linalg.inv(Knn_noisy)

        mu_train = self.mean(X_train)
        fmean = self.mean(X_new) + Kns.T @ Knn_inv @ (y_train - mu_train)
        fvar = Kss_diag - pt.sum(Kns * (Knn_inv @ Kns), axis=0)

        if incl_lik:
            return self.likelihood.predict_mean_and_var(fmean, fvar)
        return fmean, fvar
