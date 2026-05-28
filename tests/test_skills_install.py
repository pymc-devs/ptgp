"""Smoke test for ``scripts/install_claude_skills.py``."""

import importlib.util

from pathlib import Path


def _load_install_script():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "scripts" / "install_claude_skills.py"
    spec = importlib.util.spec_from_file_location("install_claude_skills", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_install_claude_skills_project(tmp_path):
    module = _load_install_script()
    rc = module.main(["--project", str(tmp_path)])
    assert rc == 0

    target = tmp_path / ".claude" / "skills" / "ptgp-vfe" / "SKILL.md"
    assert target.exists(), f"{target} missing"
    assert target.read_text().startswith("---\nname: ptgp-vfe\n")

    skill_link = target.parent
    assert skill_link.is_symlink()
    resolved = skill_link.resolve()
    assert (
        "ptgp-vfe" in resolved.parts and "agents" in resolved.parts
    ), f"unexpected resolution: {resolved}"
