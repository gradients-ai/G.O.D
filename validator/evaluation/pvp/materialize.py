"""Reconstruct a continuation miner's base model for PvP evaluation.

A continuation miner trains on the foundation with their previous-round adapter
merged in, so their uploaded adapter is relative to that merged base. This rebuilds
it — foundation + the previous adapter(s), merged in order — reusing the env-eval
merge primitives so both eval paths share one merge implementation.
"""

from validator.evaluation.eval_environment import _download_lora_with_retry
from validator.evaluation.eval_environment import _download_model_with_retry
from validator.evaluation.eval_environment import _merge_base_and_lora
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def materialize_base_model(foundation_repo: str, base_chain: list[str]) -> str:
    """Return a local path to foundation_repo with base_chain adapters merged in.

    Empty chain returns the foundation repo id unchanged (SGLang downloads it).
    """
    if not base_chain:
        return foundation_repo

    base_path = _download_model_with_retry(foundation_repo)
    for idx, adapter_repo in enumerate(base_chain):
        lora_dir = f"/tmp/base_chain_lora_{idx}"
        _download_lora_with_retry(adapter_repo, lora_dir)
        output_dir = f"/tmp/base_chain_merged_{idx}"
        base_path = _merge_base_and_lora(base_path, lora_dir, output_dir=output_dir)
        logger.info("Merged base-chain adapter %s -> %s", adapter_repo, base_path)
    return base_path
