import pytensor.tensor as pt

from ptgp.likelihoods import Gaussian
from ptgp.mean import Zero
from ptgp.objectives import collapsed_elbo


class VFE:
    """Variational Free Energy (SGPR) sparse Gaussian Process.

    Uses Titsias' collapsed bound — inducing variables are analytically
    integrated out. The observation model is Gaussian; parameterize the noise
    via ``sigma``.

    Parameters
    ----------
    kernel : Kernel
        Covariance function.
    mean : callable, optional
        Mean function (default: ``Zero()``).
    sigma : tensor or PyMC random variable
        Observation noise standard deviation.
    inducing_variable : InducingVariables
        Inducing point locations.
    """

    default_objective = staticmethod(collapsed_elbo)
    predict_needs_data = True

    def __init__(self, kernel, mean=None, sigma=None, inducing_variable=None):
        """Store the kernel, mean, and inducing variable; build a Gaussian likelihood from sigma."""
        if not hasattr(inducing_variable, "Z"):
            raise TypeError(
                f"VFE requires inducing variables with a .Z attribute "
                f"(got {type(inducing_variable).__name__}). "
                f"Use SVGP for structured inducing variables like FourierFeatures1D."
            )
        self.kernel = kernel
        self.mean = mean if mean is not None else Zero()
        self.likelihood = Gaussian(sigma)
        self.inducing_variable = inducing_variable

    @property
    def extra_vars(self):
        return tuple(self.inducing_variable.extra_vars)

    @property
    def extra_init(self):
        return tuple(self.inducing_variable.extra_init)

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
        Z = self.inducing_variable.Z
        sigma = self.likelihood.sigma
        sigma2_vec = sigma**2 * pt.ones(X_train.shape[0])  # (N,); scalar broadcasts

        Kuu = self.kernel(Z)  # (M, M)
        Kuf = self.kernel(Z, X_train)  # (M, N)
        Kus = self.kernel(Z, X_new)  # (M, N*)
        Kss_diag = self.kernel.diag(X_new)

        diff = y_train - self.mean(X_train)
        Kuf_laminv = Kuf / sigma2_vec[None, :]  # each column ÷ σᵢ²
        Sigma = Kuu + Kuf_laminv @ Kuf.T
        Sigma_inv = pt.linalg.inv(Sigma)
        alpha = Sigma_inv @ Kuf_laminv @ diff

        fmean = self.mean(X_new) + Kus.T @ alpha
        Kuu_inv = pt.linalg.inv(Kuu)
        fvar = Kss_diag - pt.sum(Kus * ((Kuu_inv - Sigma_inv) @ Kus), axis=0)

        if incl_lik:
            sigma_new = self.likelihood.sigma_at(X_train, X_new)
            return fmean, fvar + sigma_new**2
        return fmean, fvar
