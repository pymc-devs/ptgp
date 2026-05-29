import pytensor.tensor as pt

from ptgp.likelihoods.base import Likelihood, LikelihoodOp, _param_property


class StudentTOp(LikelihoodOp):
    """Student-T likelihood Op. Expectations via Gauss-Hermite quadrature."""

    def _log_prob(self, f, y, nu, sigma):
        z = (y - f) / sigma
        return (
            pt.gammaln((nu + 1.0) / 2.0)
            - pt.gammaln(nu / 2.0)
            - 0.5 * pt.log(nu * pt.pi * sigma**2)
            - 0.5 * (nu + 1.0) * pt.log1p(z**2 / nu)
        )

    def _conditional_mean(self, f, nu, sigma):
        return f

    def _conditional_variance(self, f, nu, sigma):
        return pt.ones_like(f) * sigma**2 * nu / (nu - 2.0)


class StudentT(Likelihood):
    """Student-T likelihood p(y|f) = StudentT(y; f, sigma, nu).

    Variational expectation via Gauss-Hermite quadrature.

    Parameters
    ----------
    nu : tensor or PyMC random variable
        Degrees of freedom.
    sigma : tensor or PyMC random variable
        Scale parameter.
    n_points : int
        Number of Gauss-Hermite quadrature points (default 20).
    x : tensor, optional
        The design matrix ``nu``/``sigma`` were built against, for
        heteroskedastic parameters re-rooted onto the test inputs at predict.
    """

    op_cls = StudentTOp
    param_names = ("nu", "sigma")
    nu = _param_property("nu")
    sigma = _param_property("sigma")

    def __init__(self, nu, sigma, n_points=20, x=None):
        super().__init__(x=x, n_points=n_points, nu=nu, sigma=sigma)
