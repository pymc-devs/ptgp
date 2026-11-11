import importlib.util

from pathlib import Path

import pytest


def _load_export_script():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "scripts" / "export_conda_envs.py"
    spec = importlib.util.spec_from_file_location("export_conda_envs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PIXI_EXPORT = """\
name: ptgp
channels:
- conda-forge
dependencies:
- python >=3.12
- pytensor-ml >=0.4.0,<0.5.0
- pip
- pip:
  - ruff==0.5.5
  - -e .
  - pytensor_ml>=0.4.0, <0.5.0
"""


def test_editable_install_points_at_repo_root_and_conda_duplicates_are_dropped():
    module = _load_export_script()
    rewritten = module.rewrite_pip_section(PIXI_EXPORT, editable_install=True)
    assert rewritten.endswith("- pip\n- pip:\n  - ruff==0.5.5\n  - -e ..\n")


def test_empty_pip_section_is_removed():
    module = _load_export_script()
    export_without_ruff = PIXI_EXPORT.replace("  - ruff==0.5.5\n", "")
    rewritten = module.rewrite_pip_section(export_without_ruff, editable_install=False)
    assert rewritten.endswith("- pytensor-ml >=0.4.0,<0.5.0\n- pip\n")


def test_missing_editable_line_raises():
    module = _load_export_script()
    export_without_editable = PIXI_EXPORT.replace("  - -e .\n", "")
    with pytest.raises(ValueError, match="editable install line"):
        module.rewrite_pip_section(export_without_editable, editable_install=True)
