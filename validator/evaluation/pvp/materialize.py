"""Reconstruct a continuation miner's base model for PvP evaluation.

A continuation miner trains on the foundation with their previous-round adapter
merged in, so their uploaded adapter is relative to that merged base. This rebuilds
it — the previous adapter(s) merged onto the base each one declares — reusing the
env-eval merge primitives so both eval paths share one merge implementation.
"""

import json
import os

from validator.evaluation.eval_environment import _download_lora_with_retry
from validator.evaluation.eval_environment import _download_model_with_retry
from validator.evaluation.eval_environment import _merge_base_and_lora
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def _declared_base(lora_dir: str, fallback: str) -> str:
    """The base an adapter was trained on, per its adapter_config (else fallback)."""
    config_path = os.path.join(lora_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            base = json.load(f).get("base_model_name_or_path")
        if base:
            return base
    return fallback


def materialize_base_model(
    foundation_repo: str, base_chain: list[str], label: str = "", device: str | None = None
) -> str:
    """Return a local path to the base a continuation miner trained on.

    Empty chain returns the foundation repo id unchanged (SGLang downloads it).
    Otherwise each adapter is merged onto the base it declares — matching what the
    trainer merged onto — so eval and train reconstruct an identical base. `label`
    keeps per-model scratch dirs distinct (two models are prepared before either
    SGLang server starts, so a shared path would clobber the first model's base).
    """
    if not base_chain:
        return foundation_repo

    base_path: str | None = None
    for idx, adapter_repo in enumerate(base_chain):
        lora_dir = f"/tmp/base_chain_{label}_lora_{idx}"
        _download_lora_with_retry(adapter_repo, lora_dir)
        if base_path is None:
            base_path = _download_model_with_retry(_declared_base(lora_dir, foundation_repo))
        output_dir = f"/tmp/base_chain_{label}_merged_{idx}"
        base_path = _merge_base_and_lora(base_path, lora_dir, output_dir=output_dir, device=device)
        logger.info("Merged base-chain adapter %s -> %s", adapter_repo, base_path)
    assert base_path is not None  # base_chain is non-empty, so the loop ran
    return base_path
