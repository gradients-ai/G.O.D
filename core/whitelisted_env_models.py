import json
from pathlib import Path


_WHITELIST_PATH = Path(__file__).parent / "whitelisted_env_models.json"

# Base models miners may train for the environment tournament. Ordered list
# (a seeded random.choice picks from it), so keep it a list, not a set. Each
# entry must have a tool-calling chat template and a mapped SGLang tool-call
# parser (see core/pvp/sglang_parsers.py) or it forfeits every turn.
SUPPORTED_ENV_MODELS: list[str] = json.loads(_WHITELIST_PATH.read_text())

ENV_MODEL_SIZE_B: dict[str, float] = {
    "unsloth/Llama-3.2-3B-Instruct": 3.0,
    "unsloth/Meta-Llama-3.1-8B-Instruct": 8.0,
    "Qwen/Qwen3-4B-Instruct-2507": 4.0,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen/Qwen2.5-3B-Instruct": 3.0,
    "Qwen/Qwen2.5-7B-Instruct": 7.0,
    "Qwen/Qwen3.5-0.8B": 0.8,
    "Qwen/Qwen3.5-2B": 2.0,
    # The 4B language backbone plus multimodal components has a roughly 5B footprint.
    "Qwen/Qwen3.5-4B": 5.0,
    "ibm-granite/granite-4.1-3b": 3.0,
    "allenai/Olmo-3-7B-Instruct": 7.0,
    "allenai/Olmo-Hybrid-Instruct-SFT-7B": 7.0,
    "LiquidAI/LFM2.5-2.6B": 2.6,
    "mistralai/Ministral-3-3B-Base-2512": 4.0,
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": 4.0,
    # E2B is effective size; the complete multimodal checkpoint is about 5.1B.
    "google/gemma-4-E2B": 5.1,
}
