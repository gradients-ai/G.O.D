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
}
