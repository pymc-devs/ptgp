import os
import sys

from pathlib import Path

root_dir = Path("../..").resolve()
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "docs" / "sphinxext"))

import ptgp  # noqa: E402

# -- Project information -----------------------------------------------------
project = "ptgp"
copyright = "2025, Bill Engels"
author = "Bill Engels"
language = "en"
html_baseurl = "https://ptgp.readthedocs.io"

# -- Version handling --------------------------------------------------------
# Mirrors the gEconpy / pytensor pattern so RTD version selector labels match
# what users see in the package.
version = ptgp.__version__
on_readthedocs = os.environ.get("READTHEDOCS", None)
rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
if on_readthedocs:
    if rtd_version.lower() == "stable":
        version = ptgp.__version__.split("+")[0]
    elif rtd_version.lower() == "latest":
        version = "dev"
    else:
        version = rtd_version
else:
    rtd_version = "local"
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_codeautolink",
    "generate_gallery",
    "generate_kernel_gallery",
]

# Use the document path as prefix for autosectionlabel anchors so the same
# section title in two files doesn't collide.
autosectionlabel_prefix_document = True

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "*/autosummary/*.rst",
    "Thumbs.db",
    ".DS_Store",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

master_doc = "index"

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_typehints = "none"
autoclass_content = "class"
# Class method pages live under api/.../classmethods/ — keep them out of the
# global toctree so they don't pollute the sidebar.
remove_from_toctrees = ["**/classmethods/*"]

numpydoc_show_class_members = False
numpydoc_xref_param_type = True

# Teach numpydoc about project-specific section headers so they render as
# proper sections instead of emitting "Unknown section" warnings.
from numpydoc.docscrape import NumpyDocString  # noqa: E402

for _section in ("Recommended Workflow", "Factorisation", "Fields", "Interrupts"):
    NumpyDocString.sections.setdefault(_section, [])
del NumpyDocString
numpydoc_xref_ignore = {
    "of",
    "or",
    "optional",
    "default",
    "numeric",
    "type",
    "scalar",
    "instance",
    "array",
    "array_like",
    "1D",
    "2D",
    "3D",
    "nD",
    "M",
    "N",
    "D",
    "K",
}

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "ptgp"
html_short_title = "ptgp"
html_last_updated_fmt = ""

sitemap_url_scheme = f"{{lang}}{rtd_version}/{{link}}"

html_theme_options = {
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "navbar_start": ["navbar-logo"],
    "show_prev_next": True,
    "icon_links": [
        {
            "url": "https://github.com/bwengals/ptgp",
            "icon": "fa-brands fa-github",
            "name": "GitHub",
            "type": "fontawesome",
        },
    ],
}

github_version = version if "." in rtd_version else "main"
html_context = {
    "github_url": "https://github.com",
    "github_user": "bwengals",
    "github_repo": "ptgp",
    "github_version": github_version,
    "doc_path": "docs/source",
    "default_mode": "dark",
}

html_sidebars = {"**": ["sidebar-nav-bs.html", "searchbox.html"]}
html_static_path = ["_static"]

# -- MyST / MyST-NB config ---------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "substitution",
]
myst_dmath_double_inline = True

# Notebooks ship pre-rendered. Re-executing them on RTD would require pinning
# every numerical dep and would slow the build significantly; flip this to
# "auto" or "force" later if the gallery becomes the source of truth.
nb_execution_mode = "off"

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "pytensor": ("https://pytensor.readthedocs.io/en/latest/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "arviz": ("https://python.arviz.org/en/latest/", None),
    "myst": ("https://myst-parser.readthedocs.io/en/latest/", None),
    "myst-nb": ("https://myst-nb.readthedocs.io/en/latest/", None),
}
