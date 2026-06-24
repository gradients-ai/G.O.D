"""Reconstruct a continuation miner's base model for PvP evaluation.

A continuation miner trains on the foundation with their previous-round adapter
merged in, so their uploaded adapter is relative to that merged base. This rebuilds
that base by mirroring the trainer's merge-on-download: from the top adapter, walk
`adapter_config.base_model_name_or_path` down to the foundation, then merge every
adapter bottom-to-top. Walking (rather than assuming a single flattened hop) keeps
eval identical to the trainer even when an adapter's base pointer was not flattened
to the foundation at upload — the same parity, including the 10-hop cap, that
trainer_downloader._detect_and_merge_lora gives the training side.
"""

import json
import os

from huggingface_hub import hf_hub_download

from validator.evaluation.eval_environment import _download_lora_with_retry
from validator.evaluation.eval_environment import _download_model_with_retry
from validator.evaluation.eval_environment import _merge_base_and_lora
from validator.utils.logging import get_logger


logger = get_logger(__name__)

# Matches trainer_downloader._detect_and_merge_lora's guard, so both sides truncate
# an over-deep lineage at the same depth and still agree.
MAX_CHAIN_DEPTH = 10


def _declared_base(repo: str) -> str | None:
    """The base a repo's adapter_config declares, or None if it isn't an adapter."""
    try:
        config_path = hf_hub_download(repo, "adapter_config.json", token=os.getenv("HUGGINGFACE_TOKEN"))
    except Exception:
        return None  # no adapter_config -> repo is a foundation model
    with open(config_path) as f:
        return json.load(f).get("base_model_name_or_path") or None


def _resolve_chain(top_adapter: str, fallback_foundation: str) -> tuple[str, list[str]]:
    """Walk from top_adapter to the foundation, returning (foundation, adapters).

    `adapters` is ordered bottom-to-top (merge order); `foundation` is the first
    non-adapter repo reached. Mirrors the trainer's walk exactly.
    """
    base = _declared_base(top_adapter)
    if base is None:
        return fallback_foundation, [top_adapter]

    intermediate: list[str] = []  # adapters between top_adapter's base and the foundation
    foundation = base
    for _ in range(MAX_CHAIN_DEPTH):
        parent = _declared_base(foundation)
        if parent is None:
            break  # foundation is a real model
        intermediate.append(foundation)
        foundation = parent

    return foundation, list(reversed(intermediate)) + [top_adapter]


def materialize_base_model(
    foundation_repo: str, base_chain: list[str], label: str = "", device: str | None = None
) -> str:
    """Return a local path to the base a continuation miner trained on.

    Empty chain returns the foundation repo id unchanged (SGLang downloads it).
    Otherwise the lineage is resolved from the chain's top adapter and merged
    bottom-to-top. `label` keeps per-model scratch dirs distinct (two models are
    prepared before either SGLang server starts, so a shared path would clobber the
    first model's base).
    """
    if not base_chain:
        return foundation_repo

    foundation, adapters = _resolve_chain(base_chain[0], foundation_repo)
    logger.info("Reconstructing base for %s: foundation=%s adapters=%s", base_chain[0], foundation, adapters)

    base_path = _download_model_with_retry(foundation)
    for idx, adapter_repo in enumerate(adapters):
        lora_dir = f"/tmp/base_chain_{label}_lora_{idx}"
        _download_lora_with_retry(adapter_repo, lora_dir)
        output_dir = f"/tmp/base_chain_{label}_merged_{idx}"
        base_path = _merge_base_and_lora(base_path, lora_dir, output_dir=output_dir, device=device)
        logger.info("Merged base-chain adapter %s -> %s", adapter_repo, base_path)
    return base_path
