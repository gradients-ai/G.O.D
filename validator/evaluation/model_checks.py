from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from huggingface_hub import HfApi

from core.logging import get_logger
from validator.infrastructure.retries import retry_on_5xx


if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM


logger = get_logger(__name__)
hf_api = HfApi()


def model_is_a_finetune(
    original_repo: str,
    finetuned_model: AutoModelForCausalLM,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> bool:
    from transformers import AutoConfig

    if trust_remote_code:
        # Lazy import: common pulls axolotl/peft at module load, absent in orchestration
        # (scoring/tasks.py imports model_checks), so keep model_checks importable without those deps.
        from validator.evaluation.common import pin_trusted_remote_code

        original_repo = pin_trusted_remote_code(original_repo, local_files_only)
        local_files_only = False  # pinned dir loads via the online path

    max_retries = 3
    base_delay = 2

    # For local files, try to load config directly from snapshot
    if local_files_only:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        cache_path = os.path.join(cache_dir, "hub", f"models--{original_repo.replace('/', '--')}")

        if os.path.exists(cache_path):
            snapshots_dir = os.path.join(cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = sorted(os.listdir(snapshots_dir))

                for snapshot in snapshots:
                    snapshot_path = os.path.join(snapshots_dir, snapshot)
                    if ".no_exist" in snapshot_path:
                        continue
                    config_path = os.path.join(snapshot_path, "config.json")

                    if os.path.exists(config_path) and os.path.getsize(config_path) > 0:
                        logger.info(f"Loading original model config from snapshot: {snapshot}")
                        try:
                            original_config = AutoConfig.from_pretrained(
                                snapshot_path, local_files_only=True, trust_remote_code=trust_remote_code
                            )
                            logger.info("Successfully loaded config from snapshot")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load config from snapshot {snapshot}: {e}")
                            continue
                else:
                    logger.error(f"No valid config found in snapshots for {original_repo}")
                    return False
            else:
                logger.error(f"No snapshots directory found for {original_repo}")
                return False
        else:
            logger.error(f"No cache found for {original_repo}")
            return False
    else:
        # Standard online loading with retries
        for attempt in range(max_retries):
            try:
                kwargs = {"token": os.environ.get("HUGGINGFACE_TOKEN"), "trust_remote_code": trust_remote_code}

                original_config = AutoConfig.from_pretrained(original_repo, **kwargs)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

                error_msg = str(e).lower()
                if any(
                    pattern in error_msg for pattern in ["connection", "timeout", "5xx", "too many requests", "couldn't connect"]
                ):
                    delay = base_delay * (2**attempt)
                    logger.info(
                        f"HuggingFace connection issue (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    raise e
    finetuned_config = finetuned_model.config

    try:
        architecture_classes_match = finetuned_config.architectures == original_config.architectures
    except Exception as e:
        logger.debug(f"There is an issue with checking the architecture classes {e}")
        architecture_classes_match = False

    attrs_to_compare = [
        "architectures",
        "hidden_size",
        "n_layer",
        "intermediate_size",
        "head_dim",
        "hidden_act",
        "model_type",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ]
    architecture_same = True
    for attr in attrs_to_compare:
        if getattr(original_config, attr, None) is not None:
            if not hasattr(finetuned_config, attr):
                architecture_same = False
                break
            if getattr(original_config, attr) != getattr(finetuned_config, attr):
                architecture_same = False
                break

    logger.info(f"Architecture same: {architecture_same}, Architecture classes match: {architecture_classes_match}")
    return architecture_same and architecture_classes_match


@retry_on_5xx()
def check_for_lora(model_id: str, local_files_only: bool = False) -> bool:
    """
    Check if a Hugging Face model has LoRA adapters by looking for adapter_config.json.

    Args:
        model_id (str): The Hugging Face model ID (e.g., 'username/model-name') or path
        local_files_only (bool): If True, only check local files without making API calls

    Returns:
        bool: True if it's a LoRA adapter, False otherwise
    """
    LORA_CONFIG_FILE = "adapter_config.json"
    try:
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            repo_path = os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
            if os.path.exists(repo_path):
                for root, dirs, files in os.walk(repo_path):
                    if ".no_exist" in root:
                        continue
                    if LORA_CONFIG_FILE in files:
                        config_path = os.path.join(root, LORA_CONFIG_FILE)
                        if os.path.getsize(config_path) > 0:
                            return True
            return False
        else:
            return LORA_CONFIG_FILE in hf_api.list_repo_files(model_id)
    except Exception as e:
        logger.error(f"Error checking for LoRA adapters: {e}")
        return False


@retry_on_5xx()
def check_lora_has_added_tokens(model_id: str, local_files_only: bool = False) -> bool:
    """
    Check if a LoRA repo includes added_tokens.json.

    This is used to decide whether we need to merge LoRA into base model
    before launching SGLang.
    """
    ADDED_TOKENS_FILE = "added_tokens.json"
    try:
        if local_files_only:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            repo_path = os.path.join(cache_dir, "models--" + model_id.replace("/", "--"))
            if os.path.exists(repo_path):
                for root, dirs, files in os.walk(repo_path):
                    if ".no_exist" in root:
                        continue
                    if ADDED_TOKENS_FILE in files:
                        token_file = os.path.join(root, ADDED_TOKENS_FILE)
                        if os.path.getsize(token_file) > 0:
                            return True
            return False
        return ADDED_TOKENS_FILE in hf_api.list_repo_files(model_id)
    except Exception as e:
        logger.error(f"Error checking for added_tokens.json in LoRA repo: {e}")
        return False
