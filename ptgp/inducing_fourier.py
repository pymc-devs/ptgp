import numpy as np
import pytensor.tensor as pt

from ptgp.inducing import InducingVariables
from ptgp.kernels.base import Kernel

# Basis ordering for the Fourier features: BLOCK ordering, matching the
# st--/VFF reference (Hensman, Durrande, Solin 2017). With K = num_frequencies
# and omegas = 2*pi*[0..K]/(b-a), the basis is laid out as
#     phi_0..phi_K       = cos(omega_k * (x - a)) for k = 0..K
#     phi_{K+1}..phi_{2K} = sin(omega_k * (x - a)) for k = 1..K
# The omega_0 sine row is dropped (sin(0) = 0). Total M = 2K + 1.
# This ordering must be matched by Kuf (Task 13) and any downstream consumer.


_R_BY_KERNEL = {"Matern12": 2, "Matern32": 4, "Matern52": 6}


def _maybe_wrap_with_domain_check(step_fn, gp_model, *, input_index):
    """Wrap ``step_fn`` so its positional arg at ``input_index`` is validated as ``X``.

    Returns ``step_fn`` unchanged when the model's inducing variable does not
    expose ``_domain_check`` (i.e. for stock SVGP / VFE).
    """
    ind = getattr(gp_model, "inducing_variable", None)
    if ind is None or not hasattr(ind, "_domain_check"):
        return step_fn
    kernel = gp_model.kernel

    def wrapped(*args, **kwargs):
        if len(args) > input_index:
            ind._domain_check(args[input_index], kernel)
        return step_fn(*args, **kwargs)

    wrapped.__wrapped__ = step_fn
    return wrapped


class FourierFeatures1D(InducingVariables):
    """1D Variational Fourier Features for Matern-1/2, 3/2, 5/2.

    Parameters
    ----------
    a, b : float
        Domain endpoints with a < b.
    num_frequencies : int
        Number of Fourier frequencies. ``num_inducing = 2 * num_frequencies + 1``.
    allow_extrapolation : bool, default False
        Suppress the runtime domain check installed at compile time.
    """

    def __init__(self, a, b, num_frequencies, *, allow_extrapolation=False):
        if not (float(a) < float(b)):
            raise ValueError(f"FourierFeatures1D requires a < b, got a={a}, b={b}.")
        if (float(b) - float(a)) < 1e-12:
            raise ValueError(f"Domain width b-a={b - a} is below numerical floor.")
        if int(num_frequencies) < 1:
            raise ValueError(f"num_frequencies must be >= 1, got {num_frequencies}.")
        self.a = float(a)
        self.b = float(b)
        self.num_frequencies = int(num_frequencies)
        self.allow_extrapolation = bool(allow_extrapolation)

    @property
    def num_inducing(self):
        return 2 * self.num_frequencies + 1

    @classmethod
    def from_data(cls, X, num_frequencies, buffer=0.1, *, allow_extrapolation=False):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError(
                f"FourierFeatures1D.from_data requires X of shape (N, 1), got {X.shape}."
            )
        lo, hi = float(X.min()), float(X.max())
        width = hi - lo
        return cls(
            a=lo - buffer * width,
            b=hi + buffer * width,
            num_frequencies=num_frequencies,
            allow_extrapolation=allow_extrapolation,
        )

    def _resolve_scaled_matern(self, kernel):
        """Recursively unwrap scalar products around a single Matern leaf.

        FourierFeatures1D follows the raw VFF feature convention
        ``u = P_phi(f)`` used by Hensman, Durrande, and Solin. Scalar kernel
        variance affects ``Kuu`` as ``Kuu / scale`` and does not scale ``Kuf``.
        """
        from ptgp.kernels.combination import ProductKernel, SumKernel
        from ptgp.kernels.stationary import Matern12, Matern32, Matern52

        supported = (Matern12, Matern32, Matern52)

        def walk(node):
            if isinstance(node, SumKernel):
                raise NotImplementedError(
                    "FourierFeatures1D does not implement additive VFF. "
                    "Use a single scalar-scaled 1D Matern kernel."
                )
            if isinstance(node, ProductKernel):
                s1, b1 = walk(node.k1)
                s2, b2 = walk(node.k2)
                if b1 is not None and b2 is not None:
                    raise NotImplementedError(
                        "FourierFeatures1D does not implement separable/product VFF "
                        f"({type(b1).__name__} * {type(b2).__name__}). "
                        "Use a single scalar-scaled 1D Matern kernel."
                    )
                return s1 * s2, (b1 if b1 is not None else b2)
            if isinstance(node, Kernel):
                if not isinstance(node, supported):
                    raise NotImplementedError(
                        f"FourierFeatures1D supports Matern12/32/52 only; "
                        f"got {type(node).__name__}."
                    )
                return pt.as_tensor(1.0), node
            return pt.as_tensor(node), None

        scale, base = walk(kernel)
        if base is None:
            raise NotImplementedError(
                f"FourierFeatures1D requires a supported Matern leaf; got {kernel!r}."
            )
        return scale, base

    def _structured_Kuu_base(self, base_kernel):
        """Closed-form ``(d, U)`` such that ``Kuu(base) = diag(d) + U @ U.T``.

        Operates on a **unit-variance** Matern{12,32,52} ``base_kernel``; any
        amplitude prefactor is expected to have been peeled off by
        :meth:`_resolve_scaled_matern`. The scaled wrapper divides this
        unit-variance structure by the amplitude scale, matching the authors'
        raw feature convention. Uses block ordering (cos block then sin block,
        omega_0 sine row dropped). See Hensman et al. 2017 App. A and the
        st--/VFF reference.

        Returns
        -------
        d : pytensor tensor of shape (M,)
        U : pytensor tensor of shape (M, R), with R = 1, 2, 3 for
            Matern12, Matern32, Matern52 respectively.
        """
        from ptgp.kernels.stationary import Matern12, Matern32, Matern52

        a, b = self.a, self.b
        K = self.num_frequencies
        omegas = 2.0 * np.pi * np.arange(K + 1) / (b - a)  # length K+1
        omegas_sin = omegas[1:]  # drop omega_0 = 0
        ls = base_kernel.ls
        width = b - a
        n_cos = K + 1
        n_sin = K

        ones_cos = pt.as_tensor(np.ones(n_cos))
        zeros_cos_1 = pt.as_tensor(np.zeros((n_cos, 1)))
        zeros_sin_1 = pt.as_tensor(np.zeros((n_sin, 1)))

        if isinstance(base_kernel, Matern12):
            lamb = 1.0 / ls
            two_or_four = np.where(omegas == 0, 2.0, 4.0)
            d_cos = width * (lamb**2 + omegas**2) / lamb / two_or_four
            d_sin = width * (lamb**2 + omegas_sin**2) / lamb / 4.0

            d = pt.concatenate([pt.as_tensor(d_cos), pt.as_tensor(d_sin)], axis=0)
            # U has rank 1: cos block all ones, sin block zeros.
            U_cos = pt.reshape(ones_cos, (n_cos, 1))
            U_sin = pt.as_tensor(np.zeros((n_sin, 1)))
            U = pt.concatenate([U_cos, U_sin], axis=0)
            return d, U

        if isinstance(base_kernel, Matern32):
            lamb = np.sqrt(3.0) / ls
            four_or_eight = np.where(omegas == 0, 4.0, 8.0)
            d_cos = width * (lamb**2 + omegas**2) ** 2 / lamb**3 / four_or_eight
            d_sin = width * (lamb**2 + omegas_sin**2) ** 2 / lamb**3 / 8.0
            v_sin = omegas_sin / lamb

            d = pt.concatenate([pt.as_tensor(d_cos), pt.as_tensor(d_sin)], axis=0)
            # U has rank 2: column 0 covers cos block, column 1 covers sin block.
            U_cos_col = pt.concatenate([pt.reshape(ones_cos, (n_cos, 1)), zeros_cos_1], axis=1)
            U_sin_col = pt.concatenate([zeros_sin_1, pt.reshape(v_sin, (n_sin, 1))], axis=1)
            U = pt.concatenate([U_cos_col, U_sin_col], axis=0)
            return d, U

        if isinstance(base_kernel, Matern52):
            lamb = np.sqrt(5.0) / ls
            sixteen_or_32 = np.where(omegas == 0, 16.0, 32.0)
            v1 = (3.0 * (omegas / lamb) ** 2 - 1.0) / np.sqrt(8.0)
            # W_cos shape (n_cos, 2): col 0 = v1, col 1 = ones
            W_cos = pt.concatenate(
                [pt.reshape(v1, (n_cos, 1)), pt.reshape(ones_cos, (n_cos, 1))],
                axis=1,
            )
            d_cos = 3.0 * width / sixteen_or_32 / lamb**5 * (lamb**2 + omegas**2) ** 3

            v_sin = np.sqrt(3.0) * omegas_sin / lamb
            d_sin = 3.0 * width / 32.0 / lamb**5 * (lamb**2 + omegas_sin**2) ** 3

            d = pt.concatenate([pt.as_tensor(d_cos), pt.as_tensor(d_sin)], axis=0)
            # U has rank 3: cos block fills first 2 columns, sin block fills col 2.
            U_cos = pt.concatenate([W_cos, zeros_cos_1], axis=1)
            U_sin = pt.concatenate(
                [pt.as_tensor(np.zeros((n_sin, 2))), pt.reshape(v_sin, (n_sin, 1))],
                axis=1,
            )
            U = pt.concatenate([U_cos, U_sin], axis=0)
            return d, U

        raise NotImplementedError(
            f"_structured_Kuu_base supports Matern12/32/52 only; got {type(base_kernel).__name__}."
        )

    def _structured_Kuu(self, kernel):
        """Return ``(d, U)`` for ``Kuu(kernel) = diag(d) + U @ U.T``.

        Scalar kernel amplitude follows the st--/VFF convention:
        ``Kuu(scale * base) = Kuu(base) / scale`` while ``Kuf`` is unscaled.
        Validates ``num_inducing > R`` for the unwrapped Matern kernel.
        """
        scale, base = self._resolve_scaled_matern(kernel)
        R = _R_BY_KERNEL[type(base).__name__]
        if self.num_inducing <= R:
            min_freq = (R + 1) // 2
            raise ValueError(
                f"FourierFeatures1D requires num_inducing > R "
                f"(got M={self.num_inducing}, R={R} for {type(base).__name__}). "
                f"Use num_frequencies >= {min_freq} for {type(base).__name__}."
            )
        d_base, U_base = self._structured_Kuu_base(base)
        sqrt_scale = pt.sqrt(scale)
        return d_base / scale, U_base / sqrt_scale

    def K_uu(self, kernel):
        """Dense ``Kuu`` via ``diag(d) + U @ U.T`` for compatibility with the base API."""
        d, U = self._structured_Kuu(kernel)
        K = pt.diag(d) + U @ U.T
        return pt.specify_assumptions(K, symmetric=True, positive_definite=True)

    def _domain_check(self, X_numeric, kernel):
        """Raise when numeric inputs use unsupported out-of-domain VFF behavior."""
        from ptgp.kernels.stationary import Matern52

        _, base = self._resolve_scaled_matern(kernel)
        if len(base.active_dims) != 1:
            raise ValueError(
                f"FourierFeatures1D requires a 1D kernel (len(active_dims)==1); "
                f"got active_dims={list(base.active_dims)}."
            )
        col_idx = int(base.active_dims[0])
        X = np.asarray(X_numeric)
        col = X[:, col_idx]
        lo, hi = float(col.min()), float(col.max())
        outside = lo < self.a or hi > self.b
        if not outside:
            return
        if isinstance(base, Matern52):
            raise ValueError(
                f"FourierFeatures1D does not support Matern52 extrapolation outside "
                f"domain [{self.a}, {self.b}] (got [{lo}, {hi}]). Reconstruct with "
                f"wider (a, b) or use FourierFeatures1D.from_data(X, ...)."
            )
        if not self.allow_extrapolation:
            raise ValueError(
                f"X[:, {col_idx}] (column {col_idx}) has entries outside FourierFeatures1D "
                f"domain [{self.a}, {self.b}] (got [{lo}, {hi}]). Either re-construct with "
                f"wider (a, b), use FourierFeatures1D.from_data(X, ...), or pass "
                f"allow_extrapolation=True."
            )

    def _get_active_column(self, kernel, X):
        _, base = self._resolve_scaled_matern(kernel)
        if len(base.active_dims) != 1:
            raise ValueError(
                f"FourierFeatures1D requires a 1D kernel (len(active_dims)==1); "
                f"got active_dims={list(base.active_dims)}."
            )
        col = int(base.active_dims[0])
        return X[:, col]

    def _K_uf_base(self, base_kernel, x):
        """Fourier feature covariance for unit-variance Matern. Returns ``(M, N)``.

        Block ordering: rows ``[cos(omega_k*(x-a)) for k=0..K, sin(omega_k*(x-a)) for k=1..K]``.
        For Matern12/32, out-of-domain rows use the edge covariances from the
        authors' implementation. Matern52 edge covariances are not implemented
        in the reference and are rejected by the numeric domain wrapper.
        """
        from ptgp.kernels.stationary import Matern12, Matern32

        a, b = self.a, self.b
        K = self.num_frequencies
        omegas = 2.0 * np.pi * np.arange(K + 1) / (b - a)  # length K+1
        omegas_sin = omegas[1:]
        shifted = x - a
        # cos block: shape (K+1, N)
        cos_block = pt.cos(pt.as_tensor(omegas)[:, None] * shifted[None, :])
        # sin block: shape (K, N)
        sin_block = pt.sin(pt.as_tensor(omegas_sin)[:, None] * shifted[None, :])
        lt_a = x < a
        gt_b = x > b

        if isinstance(base_kernel, Matern12):
            left_cos = pt.exp(-pt.abs(x - a) / base_kernel.ls)
            right_cos = pt.exp(-pt.abs(x - b) / base_kernel.ls)
            cos_block = pt.switch(lt_a[None, :], left_cos[None, :], cos_block)
            cos_block = pt.switch(gt_b[None, :], right_cos[None, :], cos_block)
            sin_block = pt.switch(
                (lt_a | gt_b)[None, :],
                pt.zeros_like(sin_block),
                sin_block,
            )
            return pt.concatenate([cos_block, sin_block], axis=0)

        if isinstance(base_kernel, Matern32):
            sqrt3 = np.sqrt(3.0)
            arg_left = sqrt3 * pt.abs(x - a) / base_kernel.ls
            arg_right = sqrt3 * pt.abs(x - b) / base_kernel.ls
            left_exp = pt.exp(-arg_left)
            right_exp = pt.exp(-arg_right)
            left_cos = (1.0 + arg_left) * left_exp
            right_cos = (1.0 + arg_right) * right_exp
            cos_block = pt.switch(lt_a[None, :], left_cos[None, :], cos_block)
            cos_block = pt.switch(gt_b[None, :], right_cos[None, :], cos_block)

            omega_sin_tensor = pt.as_tensor(omegas_sin)[:, None]
            left_sin = (x - a)[None, :] * left_exp[None, :] * omega_sin_tensor
            right_sin = (x - b)[None, :] * right_exp[None, :] * omega_sin_tensor
            sin_block = pt.switch(lt_a[None, :], left_sin, sin_block)
            sin_block = pt.switch(gt_b[None, :], right_sin, sin_block)
            return pt.concatenate([cos_block, sin_block], axis=0)

        return pt.concatenate([cos_block, sin_block], axis=0)

    def K_uf(self, kernel, X):
        _, base = self._resolve_scaled_matern(kernel)
        x = self._get_active_column(kernel, X)
        return self._K_uf_base(base, x)

    def Kuu_solve(self, kernel, rhs):
        """``Kuu^{-1} @ rhs`` via the Woodbury identity. ``rhs`` must be ``(M, K)``."""
        d, U = self._structured_Kuu(kernel)
        Dinv_rhs = rhs / d[:, None]
        Dinv_U = U / d[:, None]
        R = U.shape[1]
        C = pt.eye(R) + U.T @ Dinv_U
        correction = Dinv_U @ pt.linalg.solve(C, U.T @ Dinv_rhs)
        return Dinv_rhs - correction

    def Kuu_logdet(self, kernel):
        """``log det Kuu`` via the matrix-determinant lemma."""
        d, U = self._structured_Kuu(kernel)
        R = U.shape[1]
        Dinv_U = U / d[:, None]
        _, ld_small = pt.linalg.slogdet(pt.eye(R) + U.T @ Dinv_U)
        return pt.sum(pt.log(d)) + ld_small

    def Kuu_sqrt_solve(self, kernel, rhs):
        """Apply ``R^{-1}`` where ``R @ R.T = Kuu = diag(d) + U @ U.T``. ``rhs`` is ``(M, K)``.

        Uses ``delta = -1 / (sqrt(1 + lam) * (1 + sqrt(1 + lam)))`` — algebraically
        equivalent to ``1/sqrt(1+lam) - 1`` divided by ``lam`` but with no division by
        ``lam`` or ``sqrt(lam)``, so it stays finite as ``lam -> 0``.
        """
        d, U = self._structured_Kuu(kernel)
        sqrt_d = pt.sqrt(d)
        G = U.T @ (U / d[:, None])
        lam, Q = pt.linalg.eigh(G)
        sqrt1p = pt.sqrt(1.0 + lam)
        delta = -1.0 / (sqrt1p * (1.0 + sqrt1p))

        t = rhs / sqrt_d[:, None]
        UT_Dinv_rhs = U.T @ (rhs / d[:, None])
        z = Q.T @ UT_Dinv_rhs
        z_scaled = delta[:, None] * z
        y = Q @ z_scaled
        correction = (U @ y) / sqrt_d[:, None]
        return t + correction
