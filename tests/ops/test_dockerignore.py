"""The repo-root .dockerignore must keep secrets out of every image.

All images build with the repo root as context and several `COPY . .`, so a real
checkout's `.vali.env` (or a backup of it) would be baked into the layers. Matching
goes through docker-py's `exclude_paths`, a port of the daemon's own
.dockerignore matcher, so root anchoring and `**` behave as `docker build` does.
"""

import subprocess
from pathlib import Path

import pytest
from docker.utils.build import exclude_paths


REPO_ROOT = Path(__file__).resolve().parents[2]

SECRET_PATHS = [
    ".env",
    ".env.local",
    ".env.trainer-logs",
    ".vali.env",
    ".trainer.env",
    ".miner.env",
    ".vali.env.bak",
    ".vali.env.bak.20261027",
    ".vali.env.bak.1730000000",
    ".trainer.env.backup",
    "prod.env",
    "ops/.vali.env",
    "validator/.env",
    "deep/nested/dir/.trainer.env.bak.1",
    "certs/server.pem",
    "tls.key",
    "id_rsa",
    "id_ed25519.pub",
    ".ssh/id_ed25519",
    ".git-credentials",
    ".netrc",
    ".pgpass",
    ".aws/credentials",
    ".docker/config.json",
    ".bittensor/wallets/default/hotkeys/default",
    "wallets/default/coldkey",
    "validator_keys.encrypted",
]

BUILD_INPUTS = [
    "pyproject.toml",
    "README.md",
    "core/constants/credentials.py",
    "core/whitelisted_env_models.json",
    "core/training_templates/base_environment.yml",
    "ops/docker/validator-env.dockerfile",
    "ops/docker/model-prep-env.dockerfile",
    "ops/docker/requirements/quant-evaluator.txt",
    "ops/docker/scripts/image_eval_entrypoint.sh",
    "validator/core/config.py",
    "trainer/model_prep/env_stats.py",
]


def _patterns() -> list[str]:
    lines = (REPO_ROOT / ".dockerignore").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _included(root: Path, rel_paths: list[str]) -> set[str]:
    for rel in rel_paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    return exclude_paths(str(root), _patterns())


@pytest.mark.parametrize("rel_path", SECRET_PATHS)
def test_secret_is_excluded_from_build_context(tmp_path, rel_path):
    included = _included(tmp_path, [rel_path, "core/keep.py"])
    assert "core/keep.py" in included
    assert rel_path not in included


@pytest.mark.parametrize("rel_path", BUILD_INPUTS)
def test_build_input_stays_in_build_context(tmp_path, rel_path):
    assert rel_path in _included(tmp_path, [rel_path])


def test_no_tracked_file_is_excluded_by_the_secret_patterns(tmp_path):
    try:
        tracked = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files"], check=True, capture_output=True, text=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    # Only the secrets block: the earlier entries deliberately drop tracked dirs such as data/.
    patterns = _patterns()
    secret_patterns = patterns[patterns.index("**/.env*") :]
    for rel in tracked:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    excluded = set(tracked) - exclude_paths(str(tmp_path), secret_patterns)
    assert excluded == set()
