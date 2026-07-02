"""Audited-remote-code pinning for custom-arch continuous-SFT lineages (e.g. quasar).

Shared by the eval container (validator.evaluation.common) and the model-prep container
(trainer.model_prep.entrypoint): both load miner-controlled repos under trust_remote_code=True, so
both must force the modeling *.py to our audited seed mirror instead of running the miner's code.
Lives in core/ because that's the only package both container images copy in.
"""

import glob
import json
import os
import shutil
import sys
import tempfile

from huggingface_hub import snapshot_download

import core.constants as core_cst
from core.logging import get_logger


logger = get_logger(__name__)

_AUDITED_CODE_DIRS: dict[str, str] = {}


def continuous_sft_trust_remote_code() -> bool:
    """trust_remote_code=True only for custom-arch continuous-SFT lineages (audited-mirror env set)."""
    return bool(os.environ.get(core_cst.CONTINUOUS_SFT_REMOTE_CODE_REPO_ENV))


def _audited_code_dir(audited_repo: str, local_files_only: bool = False) -> str:
    """Download (once per process) the audited custom-arch code: *.py + config.json (for auto_map).

    Only config + code (never weights) is fetched; local_files_only keeps an offline eval offline.
    """
    if audited_repo not in _AUDITED_CODE_DIRS:
        _AUDITED_CODE_DIRS[audited_repo] = snapshot_download(
            audited_repo,
            allow_patterns=["*.py", "config.json"],
            token=None if local_files_only else os.environ.get("HUGGINGFACE_TOKEN"),
            local_files_only=local_files_only,
        )
    return _AUDITED_CODE_DIRS[audited_repo]


def pin_trusted_remote_code(
    model_name_or_path: str, local_files_only: bool = False, expected_base_model: str | None = None
) -> str:
    """Return a local model dir whose custom-arch *.py are forced to our audited copies.

    RCE guard: submission/carried-base repos are miner-controlled, so under trust_remote_code=True
    their *.py would run arbitrary code in the container (which holds HF + S3 creds). We drop the
    miner's *.py, copy in the audited modeling files + config from the pinned seed mirror, and reset
    config.json's auto_map to them. Weights/tokenizer are symlinked. No-op if no audited repo is set.

    For a LoRA adapter, peft resolves the base from adapter_config.json (also miner-controlled) and
    loads it with trust_remote_code=True — so the miner's base_model_name_or_path could redirect the
    base load to an arbitrary repo's code. Pass expected_base_model to force that field to our
    audited/expected base (itself pinned), closing that path.
    """
    audited_repo = os.environ.get(core_cst.CONTINUOUS_SFT_REMOTE_CODE_REPO_ENV)
    if not audited_repo:
        return model_name_or_path

    if os.path.isdir(model_name_or_path):
        model_dir = model_name_or_path
    else:
        model_dir = snapshot_download(
            model_name_or_path,
            ignore_patterns=["*.py"],  # never even fetch miner code
            token=None if local_files_only else os.environ.get("HUGGINGFACE_TOKEN"),
            local_files_only=local_files_only,
        )

    audited_dir = _audited_code_dir(audited_repo, local_files_only)
    audited_auto_map = {}
    audited_cfg = os.path.join(audited_dir, "config.json")
    if os.path.exists(audited_cfg):
        with open(audited_cfg) as f:
            audited_auto_map = json.load(f).get("auto_map", {})

    work = tempfile.mkdtemp(prefix="pinned_remote_code_")
    for name in os.listdir(model_dir):
        if name.endswith(".py"):
            continue  # drop every miner-supplied module
        src = os.path.join(model_dir, name)
        if os.path.isdir(src):
            continue
        dst = os.path.join(work, name)
        if name == "config.json":
            with open(src) as f:
                cfg = json.load(f)
            # Force the loader to our audited modeling code and never let the miner's auto_map
            # redirect it — an empty audited auto_map means drop any miner auto_map entirely.
            if audited_auto_map:
                cfg["auto_map"] = audited_auto_map
            else:
                cfg.pop("auto_map", None)
            with open(dst, "w") as f:
                json.dump(cfg, f)
        elif name == "adapter_config.json" and expected_base_model is not None:
            with open(src) as f:
                acfg = json.load(f)
            # Pin the peft base to our audited/expected repo, not the miner-declared one.
            acfg["base_model_name_or_path"] = pin_trusted_remote_code(expected_base_model, local_files_only)
            with open(dst, "w") as f:
                json.dump(acfg, f)
        else:
            os.symlink(os.path.realpath(src), dst)  # weights/tokenizer: link, not copy
    for py in glob.glob(os.path.join(audited_dir, "*.py")):
        shutil.copy2(py, os.path.join(work, os.path.basename(py)))
    # Custom-arch modeling files (e.g. quasar) import their sibling config by ABSOLUTE name
    # (`from configuration_qwen3_5 import ...`), which transformers' remote-code loader only
    # auto-resolves for RELATIVE imports. Put the pinned dir on sys.path so the audited sibling
    # modules import cleanly (and check_imports doesn't flag them as missing packages).
    if work not in sys.path:
        sys.path.insert(0, work)
    logger.info(f"Pinned remote code for {model_name_or_path} to audited {audited_repo}")
    return work
