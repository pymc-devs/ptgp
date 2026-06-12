"""
Sphinx plugin that builds a covariance gallery for ptgp kernels.

For each kernel in ``KERNEL_RECIPES``, renders a single cover image (GP prior
samples for stationary kernels, ``K(X, X)`` heatmap for categorical ones) and
emits a grid-card landing page at ``docs/source/kernels/gallery.rst``.

This is the MVP: one panel per cover, no per-kernel pages, no live execution.
Phase 2 adds per-kernel detail pages with math, parameter tables, and tabbed
parameter sweeps. Phase 3 adds Distill-style interactivity.

Inspired by preliz's distribution gallery
(https://github.com/arviz-devs/preliz/blob/main/docs/get_cover_gallery.py).
"""

import logging

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytensor
import pytensor.tensor as pt

from ptgp.kernels import (
    ExpQuad,
    Linear,
    Matern12,
    Matern32,
    Matern52,
    RandomWalk,
)

# Use Sphinx's logger when running inside Sphinx; fall back to stdlib logging
# so the cover script can be run standalone (smoke tests, regeneration).
try:
    import sphinx.util.logging as _sphinx_logging

    logger = _sphinx_logging.getLogger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "serif"


@dataclass
class CoverRecipe:
    """How to instantiate and render one cover for a kernel."""

    name: str
    display_name: str
    builder: Callable
    mode: str  # "samples", "conditional", or "heatmap"
    # Per-recipe observation override for "conditional" mode. None falls back
    # to the module-level _OBS_X / _OBS_Y. RandomWalk needs positive inputs,
    # so it provides its own.
    obs_x: np.ndarray | None = None
    obs_y: np.ndarray | None = None


# -- Builders ----------------------------------------------------------------
# Each builder returns (kernel_instance, X_np) ready to feed into the
# rendering pipeline. Kernels parameterised by floats only — no PyMC RVs in
# the gallery, since the cover is a single canonical instance per kernel.


def _line(low=-3.0, high=3.0, n=200):
    return np.linspace(low, high, n)[:, None]


def _build_expquad():
    return ExpQuad(input_dim=1, ls=1.0), _line()


def _build_matern52():
    return Matern52(input_dim=1, ls=1.0), _line()


def _build_matern32():
    return Matern32(input_dim=1, ls=1.0), _line()


def _build_matern12():
    return Matern12(input_dim=1, ls=1.0), _line()


def _build_linear():
    return Linear(input_dim=1, c=0.0), _line()


def _build_random_walk():
    # Wiener process requires positive inputs.
    return RandomWalk(input_dim=1), _line(low=0.05, high=5.0, n=200)


# Positive-domain obs for RandomWalk: two close-ish times then one farther.
_RW_OBS_X = np.array([[0.6], [1.6], [4.0]])
_RW_OBS_Y = np.array([0.45, 0.55, -1.10])


KERNEL_RECIPES: list[CoverRecipe] = [
    CoverRecipe("ExpQuad", "Exponentiated Quadratic", _build_expquad, "conditional"),
    CoverRecipe("Matern52", "Matérn 5/2", _build_matern52, "conditional"),
    CoverRecipe("Matern32", "Matérn 3/2", _build_matern32, "conditional"),
    CoverRecipe("Matern12", "Matérn 1/2", _build_matern12, "conditional"),
    CoverRecipe("Linear", "Linear", _build_linear, "conditional"),
    CoverRecipe(
        "RandomWalk",
        "Random Walk",
        _build_random_walk,
        "conditional",
        obs_x=_RW_OBS_X,
        obs_y=_RW_OBS_Y,
    ),
]

# Public kernels intentionally excluded from the gallery. Listed here so the
# pre-commit ``check-kernel-gallery`` hook stops nagging about them, but new
# kernels added to ``ptgp.kernels`` will still trigger the check until they
# land in ``KERNEL_RECIPES`` or are added here.
KERNEL_GALLERY_BLACKLIST: set[str] = {
    "Kernel",  # abstract base class
    "SumKernel",  # composition, not a "kernel" on its own
    "ProductKernel",  # composition
    "Gibbs",  # requires a user-supplied lengthscale function
    "WarpedInput",  # requires a user-supplied warp function
    "Overlap",  # categorical, needs a different render mode
    "LowRankCategorical",  # categorical
}


# -- Rendering ---------------------------------------------------------------

_N_SAMPLES = 3
_JITTER = 1e-6
_FIG_SIZE = (3.5, 2.3)
_DPI = 144
_RNG_SEED = 1

# Cover styling: pale off-blue panel + thin dashed black grid + warm sample
# palette. Line thickness + alpha cribbed from McElreath's
# plot_predictive_covariance (alpha=0.5, linewidth=5-ish) so a small number
# of bold draws layer instead of competing.
_BG_COLOR = "#eff3f7"
_GRID_COLOR = "black"
_GRID_LW = 0.3
_GRID_LS = "-"
_SAMPLE_PALETTE = ["#e8543f", "#f3a712", "#9b4dca", "#1f4e79", "#2a9d8f"]
_LINE_LW = 2.0
_LINE_ALPHA = 0.8

# Conditional ("McElreath") render: a few noisy observations with thick
# popsicle errorbars and GP posterior draws threaded through them.
_OBS_X = np.array([[-1.6], [-0.4], [1.8]])
_OBS_Y = np.array([0.65, -0.35, -1.05])
_OBS_SIGMA = 0.3
_N_COND_DRAWS = 3
# Popsicle in vivid red (the canonical McElreath "data" color); marker
# outline in black so the dot pops against any palette color underneath.
_OBS_COLOR = "#d11149"
_MARKER_EDGE_COLOR = "black"
_POPSICLE_LW = 8.0
_POPSICLE_ALPHA = 0.6
_MARKER_SIZE = 7.0
_MARKER_EDGE_LW = 1.5


def _compile_k(kernel, X_dtype):
    X_sym = pt.matrix("_X_cover", shape=(None, 1), dtype=X_dtype)
    K_sym = kernel(X_sym)
    return pytensor.function([X_sym], K_sym)


def _compile_kxy(kernel, X_dtype):
    X_sym = pt.matrix("_X_a", shape=(None, 1), dtype=X_dtype)
    Y_sym = pt.matrix("_X_b", shape=(None, 1), dtype=X_dtype)
    return pytensor.function([X_sym, Y_sym], kernel(X_sym, Y_sym))


def _render_samples(kernel, X_np, ax):
    fn = _compile_k(kernel, X_dtype=str(X_np.dtype))
    K = fn(X_np)
    K = 0.5 * (K + K.T)
    K += _JITTER * np.eye(K.shape[0])
    L = np.linalg.cholesky(K)
    rng = np.random.default_rng(_RNG_SEED)
    samples = L @ rng.standard_normal((K.shape[0], _N_SAMPLES))
    for i in range(_N_SAMPLES):
        ax.plot(
            X_np[:, 0],
            samples[:, i],
            lw=_LINE_LW,
            alpha=_LINE_ALPHA,
            color=_SAMPLE_PALETTE[i % len(_SAMPLE_PALETTE)],
        )


def _render_heatmap(kernel, X_np, ax):
    fn = _compile_k(kernel, X_dtype=str(X_np.dtype))
    K = fn(X_np)
    ax.imshow(K, cmap="viridis", aspect="auto", interpolation="nearest")


def _render_conditional(kernel, X_np, ax, obs_x=None, obs_y=None):
    """McElreath-style: hollow-circle observations + popsicle noise bars + thin
    GP posterior draws threaded through them.
    """
    obs_x = _OBS_X if obs_x is None else obs_x
    obs_y = _OBS_Y if obs_y is None else obs_y
    X_obs = obs_x.astype(X_np.dtype)
    y_obs = obs_y

    fn_oo = _compile_k(kernel, X_dtype=str(X_obs.dtype))
    fn_xy = _compile_kxy(kernel, X_dtype=str(X_np.dtype))
    fn_gg = _compile_k(kernel, X_dtype=str(X_np.dtype))

    K_oo = fn_oo(X_obs) + (_OBS_SIGMA**2 + _JITTER) * np.eye(len(X_obs))
    K_go = fn_xy(X_np, X_obs)
    K_gg = fn_gg(X_np)

    L = np.linalg.cholesky(K_oo)
    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
    mu = K_go @ alpha_vec
    v = np.linalg.solve(L, K_go.T)
    cov = K_gg - v.T @ v
    cov = 0.5 * (cov + cov.T) + _JITTER * np.eye(len(X_np))
    L_post = np.linalg.cholesky(cov)

    rng = np.random.default_rng(_RNG_SEED)
    samples = mu[:, None] + L_post @ rng.standard_normal((len(X_np), _N_COND_DRAWS))

    for i in range(samples.shape[1]):
        ax.plot(
            X_np[:, 0],
            samples[:, i],
            color=_SAMPLE_PALETTE[i % len(_SAMPLE_PALETTE)],
            lw=_LINE_LW,
            alpha=_LINE_ALPHA,
            zorder=2,
        )

    container = ax.errorbar(
        X_obs[:, 0],
        y_obs,
        yerr=_OBS_SIGMA,
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
        X_obs[:, 0],
        y_obs,
        "o",
        mfc="none",
        mec=_MARKER_EDGE_COLOR,
        mew=_MARKER_EDGE_LW,
        ms=_MARKER_SIZE,
        zorder=5,
    )


def _render_cover(recipe: CoverRecipe, out_path: Path):
    kernel, X_np = recipe.builder()
    fig, ax = plt.subplots(figsize=_FIG_SIZE, dpi=_DPI)
    paneled = recipe.mode in ("samples", "conditional")
    if paneled:
        ax.set_facecolor(_BG_COLOR)
        ax.set_axisbelow(True)
    if recipe.mode == "samples":
        _render_samples(kernel, X_np, ax)
    elif recipe.mode == "heatmap":
        _render_heatmap(kernel, X_np, ax)
    elif recipe.mode == "conditional":
        _render_conditional(
            kernel,
            X_np,
            ax,
            obs_x=recipe.obs_x,
            obs_y=recipe.obs_y,
        )
    else:
        raise ValueError(f"Unknown render mode for {recipe.name}: {recipe.mode}")
    if paneled:
        # Grid lines draw at tick locations, so we keep the ticks but hide
        # the tick marks and labels — the grid is the only thing visible.
        ax.grid(True, color=_GRID_COLOR, lw=_GRID_LW, ls=_GRID_LS, zorder=0)
        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, facecolor="white")
    plt.close(fig)


# -- Gallery page ------------------------------------------------------------

_GALLERY_TITLE = """
Covariance Gallery
==================

A visual reference for the kernels shipped in :mod:`ptgp.kernels`. Each cover
shows GP prior samples drawn with the kernel (or the ``K(X, X)`` matrix as a
heatmap for categorical kernels), giving an immediate sense of the kind of
functions the covariance produces.
"""

_TOCTREE_HEAD = """
.. toctree::
   :hidden:

"""

_GRID_HEAD = """
.. grid:: 1 2 3 3
   :gutter: 2 2 3 3

"""

_CARD_TEMPLATE_LINKED = """
   .. grid-item-card::
      :text-align: center
      :shadow: none
      :class-card: example-gallery
      :link: gallery/{slug}
      :link-type: doc

      .. image:: img/{file_name}.png
         :alt: {display_name}

      +++
      {display_name}
"""

_CARD_TEMPLATE_PLAIN = """
   .. grid-item-card::
      :text-align: center
      :shadow: none
      :class-card: example-gallery

      .. image:: img/{file_name}.png
         :alt: {display_name}

      +++
      {display_name}
"""


def main(app):
    logger.info("Starting ptgp covariance gallery generation.")

    src_dir = Path(app.builder.srcdir)
    kernels_dir = src_dir / "kernels"
    img_dir = kernels_dir / "img"
    kernels_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[tuple[str, str, str, bool]] = []
    for recipe in KERNEL_RECIPES:
        try:
            _render_cover(recipe, img_dir / f"{recipe.name}.png")
        except Exception as exc:  # keep build going if one recipe is broken
            logger.warning(
                f"Failed to render cover for {recipe.name}: {exc}",
                type="kernel_gallery",
            )
            continue
        slug = recipe.name.lower()
        page_exists = (kernels_dir / "gallery" / f"{slug}.md").exists()
        rendered.append((recipe.name, recipe.display_name, slug, page_exists))

    # Assemble: title, hidden toctree (only for pages that exist), grid header,
    # cards (linked when the page exists, plain otherwise).
    lines = [_GALLERY_TITLE, _TOCTREE_HEAD]
    lines.extend(f"   gallery/{slug}\n" for _, _, slug, exists in rendered if exists)
    lines.append(_GRID_HEAD)
    for file_name, display_name, slug, page_exists in rendered:
        template = _CARD_TEMPLATE_LINKED if page_exists else _CARD_TEMPLATE_PLAIN
        lines.append(
            template.format(
                file_name=file_name,
                display_name=display_name,
                slug=slug,
            )
        )

    gallery_rst = kernels_dir / "gallery.rst"
    gallery_rst.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote covariance gallery to {gallery_rst.relative_to(src_dir)}")


def setup(app):
    app.connect("builder-inited", main)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
