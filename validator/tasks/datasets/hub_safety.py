"""
Guard for loading caller-selected Hugging Face datasets.

datasets 2.x imports a repository's Python loading script when remote code is trusted, and it
re-imports a previously cached dynamic module even when it is not. The validator therefore only
loads code-free Hub repositories, pinned to the commit that was checked here.
"""

import os

from huggingface_hub import HfApi
from huggingface_hub.utils import validate_repo_id

from core.logging import get_logger


logger = get_logger(__name__)


class ScriptedDatasetError(ValueError):
    """The dataset reference would make the loader execute Python from the repository or host."""


def resolve_code_free_hub_dataset(dataset_id: str) -> str:
    """
    Check that `dataset_id` names a Hub dataset repository with no Python files and return the
    commit sha that was checked. Pass that sha as `revision` to every datasets loader call so a
    later push to the repository cannot swap in a loading script.

    Raises ScriptedDatasetError for local paths, malformed ids, and repositories containing `.py`
    files. Hub/network errors propagate unchanged.
    """
    dataset_id = dataset_id.strip()
    if os.path.exists(dataset_id):
        raise ScriptedDatasetError(f"Dataset {dataset_id!r} must be a Hugging Face Hub repository id, not a local path.")
    try:
        validate_repo_id(dataset_id)
    except ValueError as ex:
        raise ScriptedDatasetError(f"Dataset {dataset_id!r} is not a valid Hugging Face Hub repository id.") from ex

    info = HfApi().dataset_info(dataset_id)
    scripts = [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith(".py")]
    if scripts:
        logger.warning(f"Rejected Hugging Face dataset {dataset_id!r} at {info.sha}: contains Python files {scripts[:5]}")
        raise ScriptedDatasetError(
            f"Dataset {dataset_id!r} contains a Python loading script, which is not supported. "
            "Use a dataset stored as Parquet, JSON/JSONL, CSV or Arrow files."
        )
    if not info.sha:
        raise ScriptedDatasetError(f"Could not resolve a commit for dataset {dataset_id!r}.")
    return info.sha
