import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytensor
import pytensor.tensor as pt

mpl.rcParams["font.family"] = "serif"

_BG_COLOR = "#eff3f7"
_GRID_COLOR = "black"
_GRID_LW = 0.3
_GRID_LS = "-"
_SAMPLE_PALETTE = ["#e8543f", "#f3a712", "#9b4dca", "#1f4e79", "#2a9d8f"]
_LINE_LW = 2.0
_LINE_ALPHA = 0.8
_PANEL_W = 3.5
_PANEL_H = 2.6
_DPI = 144

_OBS_X = np.array([[-1.6], [-0.4], [1.8]])
_OBS_Y = np.array([0.65, -0.35, -1.05])
_OBS_SIGMA = 0.3
_OBS_COLOR = "#d11149"
_MARKER_EDGE_COLOR = "black"
_POPSICLE_LW = 8.0
_POPSICLE_ALPHA = 0.6
_MARKER_SIZE = 7.0
_MARKER_EDGE_LW = 1.5

_JITTER = 1e-6


def _compile_k(kernel, X_dtype="float64"):
    _X = pt.matrix("_X_plot", shape=(None, 1), dtype=X_dtype)
    return pytensor.function([_X], kernel(_X))


def _compile_kxy(kernel, X_dtype="float64"):
    _A = pt.matrix("_X_plot_a", shape=(None, 1), dtype=X_dtype)
    _B = pt.matrix("_X_plot_b", shape=(None, 1), dtype=X_dtype)
    return pytensor.function([_A, _B], kernel(_A, _B))


def _style_ax(ax):
    ax.set_facecolor(_BG_COLOR)
    ax.set_axisbelow(True)
    ax.grid(True, color=_GRID_COLOR, lw=_GRID_LW, ls=_GRID_LS, zorder=0)
    ax.tick_params(left=False, bottom=False, labelleft=True, labelbottom=True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def _make_axes(n_panels):
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(_PANEL_W * n_panels, _PANEL_H),
        dpi=_DPI,
        constrained_layout=True,
    )
    return fig, np.atleast_1d(axes)


def plot_kernel_decay(
    kernels_and_labels,
    max_distance=5.0,
    n_x=200,
):
    """Plot :math:`k(0, x)` against distance for a sequence of kernels.

    All curves share a single axes so the user can read off how covariance
    falls off relative to each other for the chosen parameter sweep. Inputs
    are 1-D and the kernel is evaluated at ``x = 0`` vs. a grid in
    ``[0, max_distance]``.

    Parameters
    ----------
    kernels_and_labels : list of (str, Kernel)
        Pairs of label and kernel instance. Each pair becomes one overlaid
        curve.
    max_distance : float
        Upper bound of the distance grid. Default 5.
    n_x : int
        Number of grid points. Default 200.

    Returns
    -------
    None
        Plots into a fresh figure on the current matplotlib backend; in
        notebooks the figure renders inline.
    """
    fig, ax = plt.subplots(
        figsize=(_PANEL_W * 2, _PANEL_H),
        dpi=_DPI,
        constrained_layout=True,
    )
    grid = np.linspace(0.0, max_distance, n_x)[:, None]
    origin = np.zeros((1, 1))
    for i, (label, kernel) in enumerate(kernels_and_labels):
        K = _compile_kxy(kernel)(origin, grid)
        ax.plot(
            grid[:, 0],
            K[0, :],
            lw=_LINE_LW,
            alpha=_LINE_ALPHA,
            color=_SAMPLE_PALETTE[i % len(_SAMPLE_PALETTE)],
            label=label,
        )
    ax.set_xlim(0, max_distance)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$\Vert x - x' \Vert$")
    ax.set_ylabel(r"$k(x, x')$")
    ax.legend(loc="upper right", frameon=False)
    ax.set_facecolor(_BG_COLOR)
    ax.set_axisbelow(True)
    ax.grid(True, color=_GRID_COLOR, lw=_GRID_LW, ls=_GRID_LS, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_prior_samples(
    kernels_and_labels,
    n_samples=4,
    x_range=(-3.0, 3.0),
    y_range=(-3.0, 3.0),
    n_x=200,
    seed=1,
):
    """Plot GP prior samples for a sequence of kernels as side-by-side panels.

    Parameters
    ----------
    kernels_and_labels : list of (str, Kernel)
        Pairs of label and kernel instance. Each pair becomes one panel.
    n_samples : int
        Number of sample functions drawn per panel. Default 4.
    x_range : tuple of float
        ``(low, high)`` range for the evaluation grid. Default ``(-3, 3)``.
    y_range : tuple of float
        ``(low, high)`` fixed y-axis limits. Default ``(-3, 3)``.
    n_x : int
        Number of grid points per panel. Default 200.
    seed : int
        RNG seed for sample reproducibility. Default 1.

    Returns
    -------
    None
        Plots into a fresh figure on the current matplotlib backend; in
        notebooks the figure renders inline.
    """
    fig, axes = _make_axes(len(kernels_and_labels))
    X_np = np.linspace(x_range[0], x_range[1], n_x)[:, None]
    rng = np.random.default_rng(seed)
    for i_ax, (ax, (label, kernel)) in enumerate(zip(axes, kernels_and_labels)):
        K = _compile_k(kernel)(X_np)
        K = 0.5 * (K + K.T) + _JITTER * np.eye(n_x)
        L = np.linalg.cholesky(K)
        samples = L @ rng.standard_normal((n_x, n_samples))
        for i in range(n_samples):
            ax.plot(
                X_np[:, 0],
                samples[:, i],
                lw=_LINE_LW,
                alpha=_LINE_ALPHA,
                color=_SAMPLE_PALETTE[i % len(_SAMPLE_PALETTE)],
            )
        ax.set_title(label, fontsize=11)
        _style_ax(ax)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range, auto=False)
        if i_ax > 0:
            ax.tick_params(labelleft=False)


def plot_conditional(
    kernels_and_labels,
    n_draws=3,
    x_range=(-3.0, 3.0),
    y_range=(-3.0, 3.0),
    n_x=200,
    obs_x=None,
    obs_y=None,
    sigma=_OBS_SIGMA,
    seed=1,
):
    """Plot GP posterior conditional draws given a few noisy observations.

    Draws ``n_draws`` posterior samples per kernel and overlays the
    observations as hollow markers with thick low-alpha popsicle error bars
    indicating the observation noise scale.

    Parameters
    ----------
    kernels_and_labels : list of (str, Kernel)
        Pairs of label and kernel instance. Each pair becomes one panel.
    n_draws : int
        Number of posterior sample functions drawn per panel. Default 3.
    x_range : tuple of float
        ``(low, high)`` range for the evaluation grid. Default ``(-3, 3)``.
    y_range : tuple of float
        ``(low, high)`` fixed y-axis limits. Default ``(-3, 3)``.
    n_x : int
        Number of grid points per panel. Default 200.
    obs_x : ndarray, optional
        Observation locations, shape ``(N_obs, 1)``. Defaults to three points
        spanning the canonical range.
    obs_y : ndarray, optional
        Observed values at ``obs_x``, shape ``(N_obs,)``. Defaults match the
        canonical ``obs_x``.
    sigma : float
        Observation noise standard deviation. Default 0.3.
    seed : int
        RNG seed for sample reproducibility. Default 1.

    Returns
    -------
    None
        Plots into a fresh figure on the current matplotlib backend; in
        notebooks the figure renders inline.
    """
    obs_x = _OBS_X if obs_x is None else obs_x
    obs_y = _OBS_Y if obs_y is None else obs_y
    fig, axes = _make_axes(len(kernels_and_labels))
    X_np = np.linspace(x_range[0], x_range[1], n_x)[:, None]
    rng = np.random.default_rng(seed)
    for i_ax, (ax, (label, kernel)) in enumerate(zip(axes, kernels_and_labels)):
        fn_k = _compile_k(kernel)
        fn_kxy = _compile_kxy(kernel)
        K_oo = fn_k(obs_x) + (sigma**2 + _JITTER) * np.eye(len(obs_x))
        K_go = fn_kxy(X_np, obs_x)
        K_gg = fn_k(X_np)
        L = np.linalg.cholesky(K_oo)
        alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, obs_y))
        mu = K_go @ alpha_vec
        v = np.linalg.solve(L, K_go.T)
        cov = K_gg - v.T @ v
        cov = 0.5 * (cov + cov.T) + _JITTER * np.eye(n_x)
        L_post = np.linalg.cholesky(cov)
        draws = mu[:, None] + L_post @ rng.standard_normal((n_x, n_draws))
        for i in range(n_draws):
            ax.plot(
                X_np[:, 0],
                draws[:, i],
                lw=_LINE_LW,
                alpha=_LINE_ALPHA,
                color=_SAMPLE_PALETTE[i % len(_SAMPLE_PALETTE)],
                zorder=2,
            )
        container = ax.errorbar(
            obs_x[:, 0],
            obs_y,
            yerr=sigma,
            fmt="none",
            ecolor=_OBS_COLOR,
            elinewidth=_POPSICLE_LW,
            capsize=0,
            zorder=4,
        )
        for bar_line in container[2]:
            bar_line.set_capstyle("round")
            bar_line.set_alpha(_POPSICLE_ALPHA)
        ax.plot(
            obs_x[:, 0],
            obs_y,
            "o",
            mfc="none",
            mec=_MARKER_EDGE_COLOR,
            mew=_MARKER_EDGE_LW,
            ms=_MARKER_SIZE,
            zorder=5,
        )
        ax.set_title(label, fontsize=11)
        _style_ax(ax)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range, auto=False)
        if i_ax > 0:
            ax.tick_params(labelleft=False)
