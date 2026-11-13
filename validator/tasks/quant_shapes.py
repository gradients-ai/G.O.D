"""Which base models a quant task may use: dense, no remote code, and shapes every launch format can pack.

The check reads the validator's own download of the model's config.json, never miner-supplied metadata.
Parameter bounds are not checked here; they stay in the model-pool query. A chat-data quant task also
needs a tokenizer with its own chat template (filter_has_chat_template).
"""

import asyncio
import json
from typing import AsyncGenerator

from huggingface_hub import hf_hub_download

from core.constants.credentials import HUGGINGFACE_TOKEN
from core.constants.quant import QUANT_MODEL_CONFIG_FILENAME
from core.logging import get_logger
from core.tokenizer_utils import read_chat_template


logger = get_logger(__name__)

# Dense architectures vLLM loads natively (no trust_remote_code).
QUANT_SUPPORTED_MODEL_TYPES = (
    "llama",
    "qwen2",
    "qwen3",
    "mistral",
    "gemma2",
    "gemma3_text",
    "phi3",
    "olmo2",
    "granite",
    "smollm3",
)
# Any of these in config.json marks a mixture-of-experts model, which is not supported.
MOE_CONFIG_KEYS = ("num_local_experts", "num_experts", "n_routed_experts", "moe_intermediate_size", "num_experts_per_tok")
_REQUIRED_SHAPE_KEYS = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads")


def quant_compatible(config: dict, group_size: int = 128) -> str | None:
    """None when the model fits every launch format, else why not.

    Group 128 is the largest launch group size, so divisibility by 128 covers every launch format.
    """
    if "auto_map" in config:
        return "requires remote code (auto_map)"
    model_type = config.get("model_type")
    if model_type not in QUANT_SUPPORTED_MODEL_TYPES:
        return f"model_type {model_type!r} is not supported"
    moe_keys = [key for key in MOE_CONFIG_KEYS if key in config]
    if moe_keys:
        return f"mixture-of-experts config keys {moe_keys}"
    missing = [key for key in _REQUIRED_SHAPE_KEYS if key not in config]
    if missing:
        return f"missing shape keys {missing}"

    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    dims = {
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "num_key_value_heads * head_dim": config["num_key_value_heads"] * head_dim,
    }
    for name, value in dims.items():
        if value % group_size:
            return f"{name}={value} is not divisible by group size {group_size}"
    return None


async def _download_config(model_id: str, token: str | None) -> dict:
    path = await asyncio.to_thread(hf_hub_download, model_id, QUANT_MODEL_CONFIG_FILENAME, token=token)
    with open(path) as f:
        return json.load(f)


async def filter_quant_compatible(
    models: AsyncGenerator[str, None], token: str | None = HUGGINGFACE_TOKEN
) -> AsyncGenerator[str, None]:
    """Yield only the model ids whose config.json passes quant_compatible; skip (and log) the rest.

    token defaults to the validator's HUGGINGFACE_TOKEN because huggingface_hub reads HF_TOKEN, not
    HUGGINGFACE_TOKEN: without it the gated families (meta-llama, gemma) fail the config download and
    drop out of the pool.
    """
    async for model_id in models:
        try:
            config = await _download_config(model_id, token)
        except Exception as e:
            logger.warning(f"Skipping {model_id} for quant: could not read {QUANT_MODEL_CONFIG_FILENAME}: {e}")
            continue
        reason = quant_compatible(config)
        if reason:
            logger.info(f"Skipping {model_id} for quant: {reason}")
            continue
        yield model_id


async def filter_has_chat_template(
    models: AsyncGenerator[str, None], token: str | None = HUGGINGFACE_TOKEN
) -> AsyncGenerator[str, None]:
    """Yield only the model ids whose tokenizer ships a chat template; skip (and log) the rest.

    Chat-data quant tasks label chat rows with `tokenizer_default`, the base tokenizer's own template, as the continuous-SFT
    boss task does, and resolve_chat_template raises when there is none, failing prep. The template is read the
    way the LoRA-merge paths read it (read_chat_template: chat_template.jinja, else a string chat_template in
    tokenizer_config.json). A tokenizer that keeps only named templates in tokenizer_config.json reads as none
    and is skipped, which only narrows the pool; an unreadable tokenizer is skipped too.
    """
    async for model_id in models:
        if await asyncio.to_thread(read_chat_template, model_id, token):
            yield model_id
        else:
            logger.info(f"Skipping {model_id} for a chat quant task: its tokenizer has no chat template")
