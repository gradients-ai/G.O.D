import hashlib
import json
import os
from pathlib import Path


def get_anonymous_model_dir(model_id: str) -> str:
    """Convert a HF model ID to an anonymous salted hash for cache directory naming.

    The salt comes from MODEL_HASH_SALT env var (set on validators, never exposed to miners).
    Without the salt, miners cannot reverse the hash to discover the model identity.
    """
    salt = os.environ.get("MODEL_HASH_SALT", "")
    return hashlib.sha256((salt + model_id).encode()).hexdigest()[:16]


def scrub_model_identity(model_dir: str) -> None:
    """Remove model identity fields from config files in a downloaded model directory."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        if "_name_or_path" in config:
            del config["_name_or_path"]
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Scrubbed _name_or_path from {config_path}")
    except Exception as e:
        print(f"Warning: Failed to scrub model identity from {config_path}: {e}")
