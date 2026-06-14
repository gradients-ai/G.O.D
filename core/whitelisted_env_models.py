import json
from pathlib import Path


_WHITELIST_PATH = Path(__file__).parent / "whitelisted_env_models.json"

# Base models miners may train for the environment tournament. Ordered list
# (a seeded random.choice picks from it), so keep it a list, not a set. Each
# entry must have a tool-calling chat template and a mapped SGLang tool-call
# parser (see core/pvp/sglang_parsers.py) or it forfeits every turn.
SUPPORTED_ENV_MODELS: list[str] = json.loads(_WHITELIST_PATH.read_text())
