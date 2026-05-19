"""Regenerate the Points-SVGP unwhitened numeric baseline.

Run before any planned change to SVGP / KL / conditionals numerics.
"""

import pickle

import numpy as np
import pytensor.tensor as pt

from ptgp.gp.svgp import SVGP
from ptgp.inducing import Points
from ptgp.kernels.stationary import Matern32
from ptgp.likelihoods.gaussian import Gaussian
from ptgp.objectives import elbo


def main():
    rng = np.random.default_rng(0)
    N, M = 200, 20
    X = np.sort(rng.uniform(0, 1, N))[:, None]
    y = np.sin(2 * np.pi * X[:, 0]) + 0.1 * rng.standard_normal(N)
    Z = np.linspace(0.05, 0.95, M)[:, None]

    k = 1.0 * Matern32(input_dim=1, ls=0.2)
    svgp = SVGP(
        kernel=k,
        likelihood=Gaussian(sigma=0.1),
        inducing_variable=Points(Z),
        whiten=False,
    )
    elbo_val = elbo(svgp, pt.as_tensor(X), pt.as_tensor(y), n_data=N).eval()
    fmean, fvar = [t.eval() for t in svgp.predict(pt.as_tensor(X))]
    kl = svgp.prior_kl().eval()

    out = {
        "elbo": float(elbo_val),
        "fmean": fmean,
        "fvar": fvar,
        "kl": float(kl),
        "X": X,
        "y": y,
        "Z": Z,
    }
    with open("tests/_fixtures/svgp_points_unwhitened_baseline.pkl", "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
