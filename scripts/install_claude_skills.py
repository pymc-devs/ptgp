"""Install the bundled agent-skill docs into a Claude Code skill directory.

The canonical, backend-agnostic skill content lives in ``docs/agents/``.
This script symlinks each skill directory into ``~/.claude/skills/`` (or
a project-local ``.claude/skills/``) so Claude Code's auto-discovery
picks it up. Other AI tools can read ``docs/agents/`` directly without
running this.

Run from a clone of the ptgp repo:

    python scripts/install_claude_skills.py --user             # ~/.claude/skills/
    python scripts/install_claude_skills.py --project .        # ./.claude/skills/
    python scripts/install_claude_skills.py --project /path    # /path/.claude/skills/
"""

import argparse
import sys

from pathlib import Path


def _resolve_skills_dir() -> Path:
    """Locate ``docs/agents/`` by walking up from this script."""
    cur = Path(__file__).resolve().parent
    for parent in [cur, *cur.parents]:
        candidate = parent / "docs" / "agents"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate docs/agents/ above {Path(__file__).resolve()}. "
        "Run this script from a clone of the ptgp repo."
    )


def _install_one(src: Path, dst: Path, force: bool) -> str:
    """Symlink ``src`` -> ``dst``. Return a one-line status."""
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="install_claude_skills",
        description="Symlink docs/agents/ skills into a Claude Code skill directory.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--user", action="store_true", help="Install into ~/.claude/skills/ (default)"
    )
    target.add_argument(
        "--project",
        metavar="DIR",
        help="Install into <DIR>/.claude/skills/ instead of ~/.claude/skills/",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing entries")
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
