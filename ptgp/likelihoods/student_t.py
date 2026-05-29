import pytensor.tensor as pt

from ptgp.likelihoods.base import LikelihoodOp, to_inputs


class StudentTOp(LikelihoodOp):
    """Student-T likelihood Op. Expectations via Gauss-Hermite quadrature."""

    param_names = ("nu", "sigma")

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


def StudentT(nu, sigma, n_points=20, x=None):
    """Build a Student-T likelihood p(y|f) with scale sigma and degrees of freedom nu.

    Returns a :class:`~ptgp.likelihoods.base.LikelihoodVariable`. Variational
    expectation via Gauss-Hermite quadrature.

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
        heteroskedastic parameters re-rooted onto test inputs via ``.at``.
    """
    op = StudentTOp(n_points=n_points, has_data=x is not None)
    return op(*to_inputs([nu, sigma], x))
