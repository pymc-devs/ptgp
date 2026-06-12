"""Fail if any public kernel is neither in the gallery nor blacklisted.

Run by the ``check-kernel-gallery`` pre-commit hook whenever ``ptgp.kernels``
or the gallery module changes. To resolve a failure, either add the kernel to
``KERNEL_RECIPES`` in ``docs/sphinxext/generate_kernel_gallery.py`` or list
it in ``KERNEL_GALLERY_BLACKLIST`` in the same file.

Uses stdlib-only AST parsing so the hook runs in pre-commit's minimal
``language: system`` env (no numpy, pytensor, or ptgp install required).
"""

import ast
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS_INIT = ROOT / "ptgp" / "kernels" / "__init__.py"
GALLERY = ROOT / "docs" / "sphinxext" / "generate_kernel_gallery.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_assign(tree: ast.Module, name: str) -> ast.expr | None:
    """Return the RHS of the module-level assignment ``name = ...``."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return node.value
    return None


def _string_elts(expr: ast.expr | None, source: Path, name: str) -> set[str]:
    if expr is None or not isinstance(expr, ast.List | ast.Set | ast.Tuple):
        raise RuntimeError(f"could not find list/set literal {name} in {source}")
    out: set[str] = set()
    for el in expr.elts:
        if not isinstance(el, ast.Constant) or not isinstance(el.value, str):
            raise RuntimeError(f"non-string entry in {name} in {source}: {ast.dump(el)}")
        out.add(el.value)
    return out


def _recipe_names(tree: ast.Module) -> set[str]:
    """Collect the first positional arg of every ``CoverRecipe(...)`` call."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "CoverRecipe"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return names


def main() -> int:
    kernels_tree = _parse(KERNELS_INIT)
    gallery_tree = _parse(GALLERY)

    public = _string_elts(_find_assign(kernels_tree, "__all__"), KERNELS_INIT, "__all__")
    blacklist = _string_elts(
        _find_assign(gallery_tree, "KERNEL_GALLERY_BLACKLIST"),
        GALLERY,
        "KERNEL_GALLERY_BLACKLIST",
    )
    recipes = _recipe_names(gallery_tree)

    missing = sorted(public - recipes - blacklist)
    if not missing:
        return 0

    print(
        "Public kernels missing from the covariance gallery:\n"
        + "\n".join(f"  - {name}" for name in missing)
        + "\n\nAdd each to KERNEL_RECIPES in docs/sphinxext/generate_kernel_gallery.py"
        " (with a cover-rendering recipe) or to KERNEL_GALLERY_BLACKLIST in the same"
        " file if it should be excluded.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
