"""VFE sparse GP tests against GPJax reference implementation."""

import pytest

from ptgp.gp.vfe import VFE
from ptgp.inducing import InducingVariables
from ptgp.kernels.stationary import Matern32


class _NoZInducing(InducingVariables):
    @property
    def num_inducing(self):
        return 5


def test_vfe_rejects_inducing_without_Z():
    with pytest.raises(TypeError, match=r"\.Z attribute"):
        VFE(kernel=Matern32(input_dim=1, ls=0.5), inducing_variable=_NoZInducing())
