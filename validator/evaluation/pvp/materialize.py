"""Reconstruct a continuation miner's true base model for PvP evaluation.

A round >= 2 (CONTINUATION) miner trains a new LoRA *on top of* the foundation with
their previous-round adapter already merged in — call that merged base M1. The new
adapter's weights are therefore relative to M1, but `hf_upload.patch_model_metadata`
flattens the uploaded `adapter_config.base_model_name_or_path` to the bare
foundation. Serving foundation + new-adapter (the default PvP path) consequently
drops the previous-round delta, evaluating a model the miner never trained.

This module rebuilds the base the trainer actually used: foundation + the
previous-round adapter chain, merged bottom-to-top. In practice the chain holds a
single repo (`starting_model_repo`), because the same upload flattening defeats the
trainer's own chain walk, so round N merges only R_{N-1} onto the foundation. The
loop handles longer chains for forward-compatibility.

It reuses the env-eval merge primitives (runtime-installs peft/accelerate, merges a
LoRA into a base on GPU), so PvP and individual env eval share one merge path.
"""

from validator.evaluation.eval_environment import _download_lora_with_retry
from validator.evaluation.eval_environment import _download_model_with_retry
from validator.evaluation.eval_environment import _merge_base_and_lora
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def materialize_base_model(foundation_repo: str, base_chain: list[str]) -> str:
    """Return a local path to foundation_repo with base_chain adapters merged in.

    With an empty chain the foundation repo id is returned unchanged (SGLang
    downloads it itself, the existing behaviour). Otherwise the foundation is
    downloaded and each adapter in `base_chain` is merged in order, the output of
    one merge feeding the next, yielding the full pre-current-adapter base.
    """
    if not base_chain:
        return foundation_repo

    base_path = _download_model_with_retry(foundation_repo)
    for idx, adapter_repo in enumerate(base_chain):
        lora_dir = f"/tmp/base_chain_lora_{idx}"
        _download_lora_with_retry(adapter_repo, lora_dir)
        output_dir = f"/tmp/base_chain_merged_{idx}"
        base_path = _merge_base_and_lora(base_path, lora_dir, output_dir=output_dir)
        logger.info(
            "Materialized continuation base step %d: merged adapter %s -> %s",
            idx,
            adapter_repo,
            base_path,
        )
    return base_path
