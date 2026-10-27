"""Guards that node install paths stay aligned after the optional-extras split."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKFILE = REPO_ROOT / "Taskfile.yml"
PYPROJECT = REPO_ROOT / "pyproject.toml"
AUDITOR_STEPS = REPO_ROOT / "ops" / "validator_ops" / "autoupdate_auditor_steps.sh"
VALIDATOR_STEPS = REPO_ROOT / "ops" / "validator_ops" / "autoupdate_validator_steps.sh"

INSTALL_EXTRAS_RE = re.compile(
    r"""pip\s+install\s+-e\s+["']\.\[(?P<extras>[^\]"']+)\]["']"""
)
UNINSTALL_CYSCALE_RE = re.compile(r"pip\s+uninstall\b[^\n]*\bcyscale\b")
PIP_INSTALL_RE = re.compile(r"pip\s+install\b")
OPTIONAL_DEPS_SECTION_RE = re.compile(
    r"\[project\.optional-dependencies\]\n(.*?)(?=\n\[|\Z)",
    flags=re.DOTALL,
)
EXTRA_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*=", flags=re.MULTILINE)


def _install_task_body(taskfile_text: str) -> str:
    match = re.search(
        r"^  install:\n(?:(?:    |\t).*\n)+",
        taskfile_text,
        flags=re.MULTILINE,
    )
    assert match is not None, "Taskfile.yml is missing an install task"
    return match.group(0)


def _optional_dependency_names(pyproject_text: str) -> set[str]:
    section = OPTIONAL_DEPS_SECTION_RE.search(pyproject_text)
    assert section is not None, "pyproject.toml is missing [project.optional-dependencies]"
    return set(EXTRA_NAME_RE.findall(section.group(1)))


def test_install_task_extras_exist_in_pyproject():
    taskfile_text = TASKFILE.read_text()
    install_body = _install_task_body(taskfile_text)
    extras_match = INSTALL_EXTRAS_RE.search(install_body)
    assert extras_match is not None, "install task must pip install -e .[extras]"

    named_extras = [part.strip() for part in extras_match.group("extras").split(",")]
    assert named_extras, "install task extras list must not be empty"

    declared = _optional_dependency_names(PYPROJECT.read_text())
    missing = [name for name in named_extras if name not in declared]
    assert not missing, f"install task references undeclared extras: {missing}"


def test_install_paths_do_not_uninstall_cyscale():
    for path in (TASKFILE, AUDITOR_STEPS, VALIDATOR_STEPS):
        text = path.read_text()
        assert UNINSTALL_CYSCALE_RE.search(text) is None, f"{path} must not uninstall cyscale"


def test_autoupdate_scripts_delegate_to_task_install():
    for path in (AUDITOR_STEPS, VALIDATOR_STEPS):
        text = path.read_text()
        assert "task install" in text, f"{path} must call task install"
        assert PIP_INSTALL_RE.search(text) is None, f"{path} must not contain raw pip install"
