Working on the docs
===================

This page covers how the documentation system is wired and the workflows
needed to extend it — adding a new kernel page, a new notebook example, or
a citation.

Building locally
----------------

The docs build from ``docs/`` using the ``ptgp-docs`` conda environment:

.. code-block:: bash

    conda env update -f conda_envs/environment-docs.yaml
    cd docs
    make show         # build + open the rendered HTML in the default browser
    make livehtml     # auto-rebuild + auto-refresh on every save (sphinx-autobuild)
    make clean        # wipe build/ and all generated source (gallery, thumbnails, autosummary stubs)

Read the Docs builds the same way; ``.readthedocs.yaml`` points at the same
conda env and ``docs/source/conf.py``.

Layout
------

Source content lives under ``docs/source/``:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Path
      - Purpose
    * - ``index.rst``
      - Landing page + top-level toctree.
    * - ``api.rst`` + ``api/*.rst``
      - Autosummary entry points; one file per public submodule.
    * - ``get_started/``
      - Install + quickstart + about. Hand-written narrative.
    * - ``user_guide/``
      - Conceptual pages (models, kernels, training, design).
    * - ``kernels/gallery.rst``
      - Covariance gallery landing page. **Generated** at build time.
    * - ``kernels/gallery/*.md``
      - Per-kernel detail pages. Hand-written MyST markdown.
    * - ``examples/gallery.rst``
      - Notebook gallery landing page. **Generated** at build time.
    * - ``examples/<category>/*.ipynb``
      - Notebook copies staged from ``notebooks/``. **Generated**.
    * - ``dev/``
      - This page and other contributor docs.
    * - ``release/``
      - Release notes.
    * - ``references.bib``
      - BibTeX entries; cited via ``{cite:t}`` or ``{cite:p}``.
    * - ``_templates/autosummary/``
      - Sphinx autosummary class template (per-method subpages).

Build-time-generated paths are gitignored via ``docs/.gitignore``; never
commit anything under ``source/kernels/img/``, ``source/_thumbnails/``,
``source/examples/<category>/``, ``source/api/**/generated/``, or
``source/kernels/gallery.rst`` / ``source/examples/gallery.rst``.

The two custom Sphinx extensions live at ``docs/sphinxext/``:

* ``generate_gallery.py`` — discovers notebooks under ``notebooks/``, copies
  them into ``docs/source/examples/<category>/``, extracts thumbnails, and
  emits ``examples/gallery.rst``.
* ``generate_kernel_gallery.py`` — renders one cover image per entry in
  ``KERNEL_RECIPES``, emits ``kernels/gallery.rst`` with grid cards. Cards
  link to per-kernel pages when a matching ``kernels/gallery/<slug>.md``
  exists on disk.

Adding a new kernel to the covariance gallery
---------------------------------------------

When a new kernel lands in ``ptgp.kernels``, the
``check-kernel-gallery`` pre-commit hook fails until you either add a cover
recipe or blacklist the kernel. To add a cover recipe:

1. **Add a builder and recipe** in
   ``docs/sphinxext/generate_kernel_gallery.py``:

   .. code-block:: python

       def _build_mynewkernel():
           return MyNewKernel(input_dim=1, ls=1.0), _line()

       KERNEL_RECIPES.append(
           CoverRecipe("MyNewKernel", "My New Kernel", _build_mynewkernel, "conditional"),
       )

   ``mode`` is one of ``"samples"``, ``"conditional"``, or ``"heatmap"``.
   For non-stationary kernels with constrained domains (like ``RandomWalk``)
   pass ``obs_x=`` / ``obs_y=`` so the conditional-render observations land
   inside the valid input range.

2. **Write the per-kernel page** at
   ``docs/source/kernels/gallery/mynewkernel.md`` — the slug must be the
   lowercase recipe name. Use ``docs/source/kernels/gallery/expquad.md`` as
   the template. The standard sections are:

   * MyST + Jupytext frontmatter.
   * H1 title and a row of tag chip links.
   * Two paragraphs of prose describing what the kernel is and when to use
     it.
   * ``## Key properties and parameters`` — table of domain, stationarity,
     sample smoothness, variance; one-sentence constructor link to the
     autosummary page.
   * ``### Covariance function`` — math equation followed by a tab-set with
     **Kernel decay**, **Prior samples**, **Posterior given 3 observations**,
     **Code** tabs. The first three tabs use the plot helpers in
     ``ptgp.plotting``; sync them on ``:sync: ls`` so clicking a
     lengthscale selection persists across tabs.
   * ``{seealso}`` admonition cross-linking related kernels.
   * One prose paragraph at the end with ``{cite:t}`` citations and a
     ``{bibliography}`` directive filtered by ``docname``.

3. **The gallery extension auto-detects the page**. Cover-card link wiring is
   automatic — no edits needed elsewhere.

To **blacklist** a kernel instead (composition kernels, kernels that take a
user-supplied callable, anything that doesn't have a canonical instance to
visualize), add its name to ``KERNEL_GALLERY_BLACKLIST`` in the same module
with a one-line comment explaining why.

Adding a notebook example
-------------------------

Drop the ``.ipynb`` under ``notebooks/``. The ``generate_gallery``
extension auto-discovers it on the next build, extracts the last image
output as a thumbnail, and emits a grid card.

To group notebooks into named categories, create subdirectories under
``notebooks/`` (e.g. ``notebooks/introductory/foo.ipynb``). The subdir name
becomes the category id; pretty titles are looked up in ``CATEGORY_TITLES``
inside ``generate_gallery.py`` and fall back to title-casing the folder
name.

Notebooks must be tracked by git to appear in the gallery (so untracked
work-in-progress notebooks under ``notebooks/`` don't pollute the build).

Adding a citation
-----------------

Append the BibTeX entry to ``docs/source/references.bib``, then cite from
prose with either textual or parenthetical form:

.. code-block:: markdown

    The kernel ridge regression view is discussed by {cite:t}`somekey-2024`.
    This approach goes back decades {cite:p}`smith-1985`.

Each page that uses citations should also include a per-page bibliography:

.. code-block:: markdown

    ```{bibliography}
    :filter: docname in docnames
    ```

The ``:filter: docname in docnames`` clause restricts the rendered
bibliography to entries cited on the current page. Sphinx will emit
"duplicate citation" warnings when the same key is rendered from multiple
pages — these are tolerated; we prefer per-page bibliographies for
readability.

Plot helpers
------------

Helpers used by the per-kernel pages live in :mod:`ptgp.plotting`:

* :func:`ptgp.plotting.plot_kernel_decay` — overlaid curves of
  :math:`k(0, x)` against distance.
* :func:`ptgp.plotting.plot_prior_samples` — side-by-side panels of GP
  prior samples for a list of kernel instances.
* :func:`ptgp.plotting.plot_conditional` — side-by-side panels of GP
  posterior draws given a few noisy observations, rendered McElreath-style
  (hollow markers with popsicle error bars).

These are public API — users can call them in their own notebooks to
reproduce the gallery style. When adding a new visualization that more than
one kernel page will use, extend ``ptgp.plotting`` rather than putting the
helper in a docs-only module.

Pre-commit checks
-----------------

* ``check-kernel-gallery`` — fails when a kernel exported from
  ``ptgp.kernels`` is missing from both ``KERNEL_RECIPES`` and
  ``KERNEL_GALLERY_BLACKLIST``. AST-only; no docs deps required.
* ``ruff`` / ``ruff-format`` — apply to ``docs/sphinxext/`` and
  ``ptgp.plotting``.
* ``no-commit-to-branch`` — work on a feature branch, never commit
  directly to ``main``.
