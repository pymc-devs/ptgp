"""Smoke test for ``ptgp install-skills``."""

from ptgp._cli import main as cli_main


def test_install_skills_project(tmp_path):
    rc = cli_main(["install-skills", "--project", str(tmp_path)])
    assert rc == 0

    target = tmp_path / ".claude" / "skills" / "ptgp-vfe" / "SKILL.md"
    assert target.exists(), f"{target} missing"
    assert target.read_text().startswith("---\nname: ptgp-vfe\n")

    skill_link = target.parent
    assert skill_link.is_symlink()
    resolved = skill_link.resolve()
    # Resolves either inside the installed wheel or inside the repo.
    parts = resolved.parts
    assert "ptgp-vfe" in parts and (
        "_skills" in parts or ".claude" in parts
    ), f"unexpected resolution: {resolved}"
