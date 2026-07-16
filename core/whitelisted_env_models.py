import json
from pathlib import Path


_WHITELIST_PATH = Path(__file__).parent / "whitelisted_env_models.json"

# Base models miners may train for the environment tournament. Ordered list
# (a seeded random.choice picks from it), so keep it a list, not a set. Each
# entry must have a tool-calling chat template and a mapped SGLang tool-call
# parser (see core/pvp/sglang_parsers.py) or it forfeits every turn.
SUPPORTED_ENV_MODELS: list[str] = json.loads(_WHITELIST_PATH.read_text())

# R1 runs on a single H100, so restrict its base model pool to models no larger than 4B.
_R1_ENV_MODEL_ALLOWLIST = {
    "unsloth/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
}
R1_SUPPORTED_ENV_MODELS: list[str] = [model for model in SUPPORTED_ENV_MODELS if model in _R1_ENV_MODEL_ALLOWLIST]
