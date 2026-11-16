"""The cache-cleanup image runs `trainer/containers/cache_cleanup.py` on a bare python image with only the directories
its Dockerfile copies; the script's imports must resolve there with the standard library alone."""

import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "ops/docker/cache-cleanup.dockerfile"


def _copied_dirs() -> list[str]:
    return re.findall(r"^COPY (\S+)/ \1/$", DOCKERFILE.read_text(), flags=re.MULTILINE)


def test_the_image_copies_trainer_and_core():
    assert {"trainer", "core"} <= set(_copied_dirs())


def test_cache_cleanup_imports_with_only_the_copied_dirs_and_no_site_packages(tmp_path):
    for name in _copied_dirs():
        shutil.copytree(REPO_ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", "import sys; sys.path.insert(0, '.'); import trainer.containers.cache_cleanup"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
