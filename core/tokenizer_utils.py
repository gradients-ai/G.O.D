"""Shared chat-template + tokenizer helpers for the LoRA-merge sites.

A merge rebuilds the merged tokenizer from either the adapter or the base, and whichever is picked
can lack the adapter's chat template (adapters often ship it only as a standalone chat_template.jinja),
so it gets silently dropped. read_chat_template + ensure_chat_template carry it; sanitize_tokenizer_config
normalizes a v5-saved tokenizer dir so v4 consumers can still load it.
"""

import json
import os

from huggingface_hub import hf_hub_download

from core.constants.paths import CHAT_TEMPLATE_FILE


TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


def read_chat_template(source: str, hf_token: str | None = None) -> str | None:
    """Chat template from a local dir or HF repo id (standalone jinja first, then inline config)."""
    if os.path.isdir(source):
        return _read_from_dir(source)
    return _read_from_hub(source, hf_token)


def _read_from_dir(source_dir: str) -> str | None:
    jinja_path = os.path.join(source_dir, CHAT_TEMPLATE_FILE)
    if os.path.exists(jinja_path):
        with open(jinja_path) as f:
            template = f.read().strip()
        if template:
            return template
    config_path = os.path.join(source_dir, TOKENIZER_CONFIG_FILE)
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                template = json.load(f).get("chat_template")
        except (ValueError, OSError):
            return None
        if isinstance(template, str) and template.strip():
            return template
    return None


def _read_from_hub(repo_id: str, hf_token: str | None) -> str | None:
    try:
        jinja_path = hf_hub_download(repo_id, CHAT_TEMPLATE_FILE, token=hf_token)
        with open(jinja_path) as f:
            template = f.read().strip()
        if template:
            return template
    except Exception:
        pass
    try:
        config_path = hf_hub_download(repo_id, TOKENIZER_CONFIG_FILE, token=hf_token)
        with open(config_path) as f:
            template = json.load(f).get("chat_template")
        if isinstance(template, str) and template.strip():
            return template
    except Exception:
        pass
    return None


def ensure_chat_template(tokenizer, *candidates: str | dict | None) -> None:
    """Graft the first non-empty candidate onto tokenizer if it has no chat template of its own.

    A candidate may be a template string or a dict of named templates (multi-template tokenizers);
    both are passed through unchanged.
    """
    if tokenizer.chat_template:
        return
    for candidate in candidates:
        if candidate:
            tokenizer.chat_template = candidate
            return


def sanitize_tokenizer_config(out_dir: str) -> None:
    """Undo v5 tokenizer-serialization quirks in place so v4 consumers can load the dir.

    - tokenizer_class="TokenizersBackend" (v5 backend marker, not a loadable class) -> the concrete
      fast class if tokenizer.json is present, else dropped for autodetection.
    - extra_special_tokens list -> dict (v4 calls .keys() on it).
    - chat template folded from the standalone jinja back inline (kept in both), for pre-4.47 readers.

    Rewrites tokenizer_config.json in place, so call it only on a dir you just wrote — never a
    pin_trusted_remote_code work dir, whose tokenizer files are symlinks into the immutable seed.
    """
    config_path = os.path.join(out_dir, TOKENIZER_CONFIG_FILE)
    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        config = json.load(f)
    changed = False

    if config.get("tokenizer_class") == "TokenizersBackend":
        if os.path.exists(os.path.join(out_dir, "tokenizer.json")):
            config["tokenizer_class"] = "PreTrainedTokenizerFast"
        else:
            del config["tokenizer_class"]
        changed = True

    extra_special_tokens = config.get("extra_special_tokens")
    if isinstance(extra_special_tokens, list):
        config["extra_special_tokens"] = {token: token for token in extra_special_tokens if isinstance(token, str)}
        changed = True

    if not config.get("chat_template"):
        folded = _read_from_dir(out_dir)
        if folded:
            config["chat_template"] = folded
            changed = True

    if changed:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
