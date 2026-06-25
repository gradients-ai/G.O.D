"""Reconstruct a continuation miner's base model for PvP and env evaluation."""

import json
import os

from huggingface_hub import hf_hub_download

from core.logging import get_logger
from validator.evaluation.evaluators.environment import _download_lora_with_retry
from validator.evaluation.evaluators.environment import _download_model_with_retry
from validator.evaluation.evaluators.environment import _merge_base_and_lora


logger = get_logger(__name__)
MAX_CHAIN_DEPTH = 10


def _declared_base(repo: str) -> str | None:
    """Return a repo's adapter_config base, or None if it is not an adapter."""
    try:
        config_path = hf_hub_download(repo, "adapter_config.json", token=os.getenv("HUGGINGFACE_TOKEN"))
    except Exception:
        return None
    with open(config_path) as f:
        return json.load(f).get("base_model_name_or_path") or None


def _resolve_chain(top_adapter: str, fallback_foundation: str) -> tuple[str, list[str]]:
    """Walk adapter_config lineage and return foundation plus bottom-to-top adapters."""
    base = _declared_base(top_adapter)
    if base is None:
        return fallback_foundation, [top_adapter]

    intermediate: list[str] = []
    foundation = base
    for _ in range(MAX_CHAIN_DEPTH):
        parent = _declared_base(foundation)
        if parent is None:
            break
        intermediate.append(foundation)
        foundation = parent

    return foundation, list(reversed(intermediate)) + [top_adapter]


def materialize_base_model(
    foundation_repo: str,
    base_chain: list[str],
    label: str = "",
    device: str | None = None,
) -> str:
    """Return a local path to the base a continuation miner trained on."""
    if not base_chain:
        if _declared_base(foundation_repo) is None:
            return foundation_repo
        # foundation_repo is itself a LoRA (e.g. previous_winner task base).
        # SGLang can't load a LoRA as a base — walk and merge it.
        base_chain = [foundation_repo]

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
