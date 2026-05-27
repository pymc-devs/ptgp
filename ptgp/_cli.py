"""ptgp CLI: ``ptgp <subcommand> [args]``.

Currently provides ``install-skills`` to symlink bundled Claude Code skills
into ``~/.claude/skills/`` (or a project-local ``.claude/skills/``).
"""

import argparse
import importlib.resources
import sys

from pathlib import Path


def _resolve_skills_dir():
    """Locate the bundled `_skills` directory.

    Strategy:
    1. ``importlib.resources.files("ptgp._skills")`` — wheel installs land here.
    2. Walk up from ``ptgp/__file__`` looking for ``.claude/skills/`` —
       editable installs from a source checkout.
    """
    try:
        ref = importlib.resources.files("ptgp._skills")
        path = Path(str(ref))
        if path.is_dir():
            return path
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass

    import ptgp

    cur = Path(ptgp.__file__).resolve().parent
    for parent in [cur, *cur.parents]:
        candidate = parent / ".claude" / "skills"
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate bundled skills. Looked for ptgp._skills (wheel "
        "install) and .claude/skills/ next to ptgp (editable install)."
    )


def _install_one(src: Path, dst: Path, force: bool) -> str:
    """Symlink ``src`` -> ``dst``. Returns a one-line status."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            return f"skip   {dst}  (exists; use --force to override)"
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            import shutil

            shutil.rmtree(dst)
    dst.symlink_to(src)
    return f"link   {dst}  ->  {src}"


def install_skills(args: argparse.Namespace) -> int:
    """``ptgp install-skills`` entry point."""
    skills_dir = _resolve_skills_dir()
    if args.project is not None:
        target_root = Path(args.project) / ".claude" / "skills"
    else:
        target_root = Path.home() / ".claude" / "skills"

    skills = sorted(p for p in skills_dir.iterdir() if p.is_dir())
    if not skills:
        print(f"No skills found in {skills_dir}")
        return 1

    for src in skills:
        dst = target_root / src.name
        print(_install_one(src.resolve(), dst, force=args.force))
    return 0


def main(argv=None) -> int:
    """``ptgp`` console-script entry point."""
    parser = argparse.ArgumentParser(prog="ptgp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_install = sub.add_parser(
        "install-skills",
        help="Symlink bundled Claude Code skills into ~/.claude/skills/ "
        "(or a project's .claude/skills/).",
    )
    target = p_install.add_mutually_exclusive_group()
    target.add_argument(
        "--user", action="store_true", help="Install into ~/.claude/skills/ (default)"
    )
    target.add_argument(
        "--project",
        metavar="DIR",
        help="Install into <DIR>/.claude/skills/ instead of ~/.claude/skills/",
    )
    p_install.add_argument("--force", action="store_true", help="Overwrite existing entries")
    p_install.set_defaults(func=install_skills)

    args = parser.parse_args(argv)
    rc: int = args.func(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
