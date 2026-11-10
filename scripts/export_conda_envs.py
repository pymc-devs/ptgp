import re
import subprocess

from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent
CONDA_ENVS = REPO / "conda_envs"


class CondaExport(NamedTuple):
    pixi_environment: str
    conda_name: str
    output: Path
    editable_install: bool


# CI and Read the Docs install ptgp themselves, so only the dev environment carries the editable
# install.
EXPORTS = [
    CondaExport(
        pixi_environment="default",
        conda_name="ptgp",
        output=CONDA_ENVS / "environment.yaml",
        editable_install=True,
    ),
    CondaExport(
        pixi_environment="test",
        conda_name="ptgp-test",
        output=CONDA_ENVS / "environment-test.yaml",
        editable_install=False,
    ),
    CondaExport(
        pixi_environment="docs",
        conda_name="ptgp-docs",
        output=CONDA_ENVS / "environment-docs.yaml",
        editable_install=False,
    ),
]

# conda runs pip with the environment file's directory as its working directory
# (conda/env/installers/pip.py), so pixi's `-e .` would name conda_envs/ instead of the repository
# root.
PIXI_EDITABLE = "-e ."
REPO_EDITABLE = "-e .."

CONDA_ITEM = "- "
PIP_ITEM = "  - "

# Pinned so the files do not depend on the machine running the hook.
EXPORT_PLATFORM = "linux-64"

PACKAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def normalized_name(spec: str) -> str:
    """Return the PEP 503 normalized package name at the start of a conda or pip requirement."""
    match = PACKAGE_NAME.match(spec)
    if match is None:
        raise ValueError(f"No package name in requirement {spec!r}")
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def export_command(export: CondaExport) -> list[str]:
    return [
        "pixi",
        "workspace",
        "export",
        "conda-environment",
        "-e",
        export.pixi_environment,
        "-n",
        export.conda_name,
        "-p",
        EXPORT_PLATFORM,
    ]


def rewrite_pip_section(exported: str, editable_install: bool) -> str:
    """Repoint or drop the editable install, and drop pip duplicates of conda packages."""
    lines = exported.rstrip("\n").splitlines()

    # pixi repeats [project].dependencies in the pip section, and conda installs that section with
    # `pip install -U`, which could replace a conda-forge build with a PyPI wheel.
    conda_packages = {
        normalized_name(line.removeprefix(CONDA_ITEM))
        for line in lines
        if line.startswith(CONDA_ITEM)
    }

    if f"{PIP_ITEM}{PIXI_EDITABLE}" not in lines:
        raise ValueError(
            f"pixi's export no longer contains the editable install line {PIXI_EDITABLE!r}"
        )

    pip_requirements = []
    other_lines = []
    for line in lines:
        requirement = line.removeprefix(PIP_ITEM)
        if not line.startswith(PIP_ITEM):
            other_lines.append(line)
        elif requirement == PIXI_EDITABLE:
            if editable_install:
                pip_requirements.append(f"{PIP_ITEM}{REPO_EDITABLE}")
        elif normalized_name(requirement) not in conda_packages:
            pip_requirements.append(line)

    if not pip_requirements:
        other_lines.remove(f"{CONDA_ITEM}pip:")

    return "\n".join(other_lines + pip_requirements) + "\n"


def main() -> None:
    for export in EXPORTS:
        exported = subprocess.run(
            export_command(export), cwd=REPO, check=True, stdout=subprocess.PIPE, text=True
        ).stdout
        export.output.write_text(
            rewrite_pip_section(exported, editable_install=export.editable_install),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
