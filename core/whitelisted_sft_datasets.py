import json
from pathlib import Path

_WHITELIST_PATH = Path(__file__).parent / "whitelisted_sft_datasets.json"

WHITELISTED_SFT_DATASETS: set[str] = set(json.loads(_WHITELIST_PATH.read_text()))

MAX_REQUESTED_DATASETS = 2


def validate_requested_datasets(requested_datasets: list[str] | None) -> list[str]:
    if not requested_datasets:
        return []

    valid = [ds for ds in requested_datasets if ds in WHITELISTED_SFT_DATASETS]
    return valid[:MAX_REQUESTED_DATASETS]
