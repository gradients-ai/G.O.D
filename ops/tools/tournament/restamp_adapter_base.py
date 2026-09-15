#!/usr/bin/env python3
"""Rewrite base_model_name_or_path in a LoRA adapter's adapter_config.json.

Why this exists
---------------
When continuous-SFT carries a LoRA winner as next week's base, the trainer downloader
merges that adapter into full weights before training. The uploader then stamps the
new submission's adapter_config with the foundation (walking past the seed via
``_resolve_base_model``), so eval reconstructs the wrong base. The trained weights
are correct; only the label is wrong.

This tool rewrites that single field so the chain describes the repo the miner
actually trained against. Must run before the next final round's model-prep
downloads the repo.

Usage
-----
    # Dry run (default): print current vs proposed base; write nothing.
    python ops/tools/tournament/restamp_adapter_base.py \\
        --repo gradients-io-tournaments/tournament-... \\
        --base-model gradients-io-tournaments/tournament-...

    # Actually upload the rewritten adapter_config.json.
    python ops/tools/tournament/restamp_adapter_base.py \\
        --repo gradients-io-tournaments/tournament-... \\
        --base-model gradients-io-tournaments/tournament-... \\
        --apply

Reads HUGGINGFACE_TOKEN from .vali.env / .env / the environment.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

from core.constants.paths import LORA_ADAPTER_CONFIG_FILE


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_environment() -> None:
    for candidate in (".vali.env", ".env"):
        env_path = REPO_ROOT / candidate
        if env_path.exists():
            load_dotenv(env_path, override=False)


def _require_hf_token() -> str:
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "HUGGINGFACE_TOKEN not found. Run from the G.O.D root with a `.vali.env` "
            "(or `.env`) containing HUGGINGFACE_TOKEN=..."
        )
    return token


def restamp_adapter_base(repo_id: str, base_model: str, *, apply: bool, token: str) -> None:
    """Rewrite adapter_config.json base_model_name_or_path for ``repo_id``."""
    try:
        config_path = hf_hub_download(repo_id, LORA_ADAPTER_CONFIG_FILE, token=token)
    except Exception as e:
        raise SystemExit(f"Failed to download {LORA_ADAPTER_CONFIG_FILE} from {repo_id}: {e}") from e

    with open(config_path) as f:
        config = json.load(f)

    current = config.get("base_model_name_or_path")
    logger.info(f"repo={repo_id}")
    logger.info(f"  current base_model_name_or_path={current!r}")
    logger.info(f"  proposed base_model_name_or_path={base_model!r}")

    if current == base_model:
        logger.info("Already stamped correctly; nothing to do")
        return

    if not apply:
        logger.info("Dry run only; pass --apply to upload the rewritten config")
        return

    config["base_model_name_or_path"] = base_model
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(config, tmp, indent=2)
        tmp.write("\n")
        tmp_path = tmp.name

    try:
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=LORA_ADAPTER_CONFIG_FILE,
            repo_id=repo_id,
            commit_message=f"Re-stamp base_model_name_or_path -> {base_model}",
        )
    finally:
        os.unlink(tmp_path)

    logger.info(f"Uploaded {LORA_ADAPTER_CONFIG_FILE} to {repo_id}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite base_model_name_or_path in a LoRA adapter_config.json on Hugging Face."
    )
    parser.add_argument("--repo", required=True, help="HF repo id of the adapter to re-stamp")
    parser.add_argument(
        "--base-model",
        required=True,
        help="HF repo id that should be recorded as base_model_name_or_path",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Upload the rewritten config (default is dry-run)",
    )
    args = parser.parse_args(argv)

    _load_environment()
    token = _require_hf_token()
    restamp_adapter_base(args.repo, args.base_model, apply=args.apply, token=token)


if __name__ == "__main__":
    main(sys.argv[1:])
