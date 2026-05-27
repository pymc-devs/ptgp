"""
Sphinx plugin that builds an example gallery from ``notebooks/``.

Sources notebooks from the top-level ``notebooks/`` directory at the repo root
and emits a grid-card gallery page at ``docs/source/examples/gallery.rst``.

Categorization rule:
  - Notebooks placed in subdirectories (``notebooks/<category>/foo.ipynb``)
    are grouped under that subdirectory's title (looked up in
    ``CATEGORY_TITLES``, with the folder name title-cased as fallback).
  - Notebooks at the top level (``notebooks/foo.ipynb``) fall under
    "Examples".

For each notebook the script copies the file into
``docs/source/examples/<category>/<name>.ipynb`` so MyST-NB picks it up, and
extracts the last image output as a thumbnail under
``docs/source/_thumbnails/<category>/<name>.png``. A user-supplied thumbnail
at that path is left alone.

Adapted from gEconpy, which adapted it from PyMC / seaborn / mpld3.
"""

import base64
import json
import shutil
import subprocess

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sphinx

from matplotlib import image

logger = sphinx.util.logging.getLogger(__name__)

# Repo root: docs/sphinxext/generate_gallery.py -> repo
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"

# Pretty titles for known subfolders. Anything not listed is title-cased.
CATEGORY_TITLES = {
    "examples": "Examples",
    "introductory": "Introductory",
    "advanced": "Advanced",
    "case_study": "Case Studies",
}

DEFAULT_IMG_LOC = None

HEAD = """
Example Gallery
===============

.. toctree::
   :hidden:

"""

SECTION_TEMPLATE = """
.. _gallery-{section_id}:

{section_title}
{underlines}

.. grid:: 1 2 3 3
   :gutter: 4

"""

ITEM_TEMPLATE = """
   .. grid-item-card:: :doc:`{doc_name}`
      :img-top: {image}
      :link: {doc_reference}
      :link-type: {link_type}
      :shadow: none
"""


def is_tracked_by_git(filepath):
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(filepath)],
            capture_output=True,
            check=False,
            cwd=REPO_ROOT,
        )
    except FileNotFoundError:
        return True
    else:
        return result.returncode == 0


def create_thumbnail(infile, width=275, height=275, cx=0.5, cy=0.5, border=4):
    im = image.imread(infile)
    rows, cols = im.shape[:2]
    size = min(rows, cols)
    if size == cols:
        xslice = slice(0, size)
        ymin = min(max(0, int(cy * rows - size // 2)), rows - size)
        yslice = slice(ymin, ymin + size)
    else:
        yslice = slice(0, size)
        xmin = min(max(0, int(cx * cols - size // 2)), cols - size)
        xslice = slice(xmin, xmin + size)
    thumb = im[yslice, xslice]
    thumb[:border, :, :3] = thumb[-border:, :, :3] = 0
    thumb[:, :border, :3] = thumb[:, -border:, :3] = 0

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], aspect="auto", frameon=False, xticks=[], yticks=[])
    ax.imshow(thumb, aspect="auto", resample=True, interpolation="bilinear")
    fig.savefig(infile, dpi=dpi)
    plt.close(fig)
    return fig


class NotebookGenerator:
    """Extract a thumbnail and stage a notebook for inclusion in the gallery."""

    def __init__(self, src_nb: Path, category: str, examples_dir: Path, thumbnails_dir: Path):
        self.src_nb = src_nb
        self.stripped_name = src_nb.stem
        self.category = category
        self.staged_nb = examples_dir / category / f"{self.stripped_name}.ipynb"
        self.png_path = thumbnails_dir / category / f"{self.stripped_name}.png"

        with src_nb.open(encoding="utf-8") as fid:
            self.json_source = json.load(fid)

    def stage_notebook(self):
        self.staged_nb.parent.mkdir(parents=True, exist_ok=True)
        # Always re-copy: notebooks at the source can change between builds.
        shutil.copyfile(self.src_nb, self.staged_nb)

    def extract_preview_pic(self):
        pic = None
        for cell in self.json_source["cells"]:
            for output in cell.get("outputs", []):
                if "image/png" in output.get("data", []):
                    pic = output["data"]["image/png"]
        if pic is not None:
            return base64.b64decode(pic)
        return None

    def gen_previews(self):
        self.png_path.parent.mkdir(parents=True, exist_ok=True)
        if self.png_path.exists():
            logger.info(
                f"Custom thumbnail already exists for {self.src_nb.name}, skipping extraction",
                type="thumbnail_extractor",
            )
            return

        preview = self.extract_preview_pic()
        if preview is not None:
            with self.png_path.open("wb") as buff:
                buff.write(preview)
            create_thumbnail(self.png_path)
        else:
            if DEFAULT_IMG_LOC is not None:
                shutil.copy(DEFAULT_IMG_LOC, self.png_path)
                create_thumbnail(self.png_path)
            else:
                logger.warning(
                    f"No image found in {self.src_nb.name} and no default thumbnail set",
                    type="thumbnail_extractor",
                )


def discover_notebooks():
    """Group notebooks by category.

    Returns dict mapping ``category -> list[Path]`` where category is the
    immediate subfolder of ``notebooks/`` (or ``"examples"`` for top-level
    notebooks).
    """
    if not NOTEBOOKS_ROOT.exists():
        return {}

    grouped: dict[str, list[Path]] = {}
    for path in sorted(NOTEBOOKS_ROOT.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        rel = path.relative_to(NOTEBOOKS_ROOT)
        category = rel.parts[0] if len(rel.parts) > 1 else "examples"
        grouped.setdefault(category, []).append(path)
    return grouped


def main(app):
    logger.info("Starting ptgp example gallery generation.")

    src_dir = Path(app.builder.srcdir)
    examples_dir = src_dir / "examples"
    thumbnails_dir = src_dir / "_thumbnails"
    examples_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    grouped = discover_notebooks()

    if not grouped:
        logger.warning(
            "No notebooks found under notebooks/; writing empty gallery.",
            type="thumbnail_extractor",
        )

    file_lines = [HEAD]

    for category in sorted(grouped):
        nb_paths = grouped[category]
        title = CATEGORY_TITLES.get(category, category.replace("_", " ").title())
        file_lines.append(
            SECTION_TEMPLATE.format(
                section_title=title,
                section_id=category,
                underlines="-" * len(title),
            )
        )

        for nb_path in nb_paths:
            if not is_tracked_by_git(nb_path):
                logger.info(
                    f"Skipping {nb_path.name}, not tracked by git",
                    type="thumbnail_extractor",
                )
                continue

            nbg = NotebookGenerator(
                src_nb=nb_path,
                category=category,
                examples_dir=examples_dir,
                thumbnails_dir=thumbnails_dir,
            )
            nbg.stage_notebook()
            nbg.gen_previews()

            doc_name = f"{category}/{nbg.stripped_name}"
            # Path is relative to docs/source/ — the leading slash makes
            # Sphinx resolve it from the source root, matching gEconpy's
            # convention so users can drop in custom thumbnails too.
            img_path = f"/_thumbnails/{category}/{nbg.stripped_name}.png"
            file_lines.append(
                ITEM_TEMPLATE.format(
                    doc_name=doc_name,
                    image=img_path,
                    doc_reference=doc_name,
                    link_type="doc",
                )
            )

    gallery_rst = examples_dir / "gallery.rst"
    gallery_rst.write_text("\n".join(file_lines), encoding="utf-8")
    logger.info(f"Wrote gallery to {gallery_rst.relative_to(src_dir)}")


def setup(app):
    app.connect("builder-inited", main)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
