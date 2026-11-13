import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from validator.tournament import github_validation


TOKEN = "ghp_secretsecretsecret"
REPO = "https://github.com/org/repo"
VALIDATOR_ROOT = Path(github_validation.__file__).resolve().parents[2]


async def test_obfuscation_output_is_logged_without_the_token(monkeypatch):
    stdout = f"Cloning https://x-access-token:{TOKEN}@github.com/org/repo.git ... clean"
    monkeypatch.setattr(
        github_validation.subprocess,
        "run",
        MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")),
    )
    logger = MagicMock()
    monkeypatch.setattr(github_validation, "logger", logger)

    assert await github_validation.validate_repo_obfuscation(REPO, "c" * 40, TOKEN)

    logged = " ".join(str(call) for call in logger.info.call_args_list)
    assert TOKEN not in logged
    assert "clean" in logged


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


@pytest.fixture
def license_repo(tmp_path: Path) -> tuple[str, str, str]:
    """A repo whose first commit has no LICENSE/NOTICE and whose second (the default branch head) adds them."""
    repo = tmp_path / "miner"
    repo.mkdir()
    _git(repo, "init", "--quiet", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "train.py").write_text("print('hi')\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "no licence")
    unlicensed = _git(repo, "rev-parse", "HEAD")
    shutil.copy(VALIDATOR_ROOT / "LICENSE.md", repo / "LICENSE.md")
    shutil.copy(VALIDATOR_ROOT / "NOTICE", repo / "NOTICE")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "licence")
    licensed = _git(repo, "rev-parse", "HEAD")
    return str(repo), unlicensed, licensed


async def test_license_checked_at_the_pinned_commit_not_the_default_branch(license_repo):
    repo, unlicensed, licensed = license_repo
    assert await github_validation.validate_repo_license(repo, None, licensed)
    assert not await github_validation.validate_repo_license(repo, None, unlicensed)


async def test_license_without_a_commit_checks_the_default_branch(license_repo):
    repo, _, _ = license_repo
    assert await github_validation.validate_repo_license(repo, None)


async def test_license_fails_for_a_commit_missing_from_the_repo(license_repo):
    repo, _, _ = license_repo
    assert not await github_validation.validate_repo_license(repo, None, "0" * 40)
