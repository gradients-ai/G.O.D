"""Unit tests for core.tokenizer_utils — the single home for chat-template carry + tokenizer
normalization shared across the four LoRA-merge sites (trainer downloader, model-prep entrypoint,
env evaluator, intercode evaluator).

Pure filesystem/dict logic — no torch/transformers/network — so these run on any box.
"""

import json
import os

from core.constants.paths import CHAT_TEMPLATE_FILE
from core.tokenizer_utils import TOKENIZER_CONFIG_FILE
from core.tokenizer_utils import ensure_chat_template
from core.tokenizer_utils import read_chat_template
from core.tokenizer_utils import sanitize_tokenizer_config


TEMPLATE = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"
OTHER_TEMPLATE = "{{ messages[0]['content'] }}"


def _write(dir_path, name, content):
    with open(os.path.join(dir_path, name), "w") as f:
        f.write(content)


def _write_config(dir_path, config):
    with open(os.path.join(dir_path, TOKENIZER_CONFIG_FILE), "w") as f:
        json.dump(config, f)


class _FakeTokenizer:
    """Minimal stand-in: .chat_template is a plain settable attribute, like PreTrainedTokenizerBase."""

    def __init__(self, chat_template=None):
        self.chat_template = chat_template


# ── read_chat_template ────────────────────────────────────────────────────────


def test_read_prefers_standalone_jinja(tmp_path):
    _write(tmp_path, CHAT_TEMPLATE_FILE, TEMPLATE)
    _write_config(tmp_path, {"chat_template": OTHER_TEMPLATE})
    assert read_chat_template(str(tmp_path)) == TEMPLATE


def test_read_falls_back_to_inline_config(tmp_path):
    _write_config(tmp_path, {"chat_template": OTHER_TEMPLATE})
    assert read_chat_template(str(tmp_path)) == OTHER_TEMPLATE


def test_read_empty_jinja_falls_through_to_inline(tmp_path):
    # A jinja file that exists but is blank must not shadow a real inline template.
    _write(tmp_path, CHAT_TEMPLATE_FILE, "   \n")
    _write_config(tmp_path, {"chat_template": OTHER_TEMPLATE})
    assert read_chat_template(str(tmp_path)) == OTHER_TEMPLATE


def test_read_returns_none_when_absent(tmp_path):
    _write_config(tmp_path, {"model_max_length": 2048})
    assert read_chat_template(str(tmp_path)) is None


def test_read_returns_none_on_empty_dir(tmp_path):
    assert read_chat_template(str(tmp_path)) is None


def test_read_survives_malformed_config(tmp_path):
    # A miner adapter with invalid tokenizer_config.json must not crash the merge (AutoTokenizer
    # load is already guarded upstream and falls back to base); recovering no template is fine.
    _write(tmp_path, TOKENIZER_CONFIG_FILE, "{not valid json")
    assert read_chat_template(str(tmp_path)) is None


def test_read_ignores_non_string_inline_template(tmp_path):
    _write_config(tmp_path, {"chat_template": {"default": OTHER_TEMPLATE}})
    assert read_chat_template(str(tmp_path)) is None


# ── ensure_chat_template ──────────────────────────────────────────────────────


def test_ensure_noop_when_target_has_template():
    tok = _FakeTokenizer(chat_template=OTHER_TEMPLATE)
    ensure_chat_template(tok, TEMPLATE, "base_template")
    assert tok.chat_template == OTHER_TEMPLATE  # target's own template wins


def test_ensure_grafts_first_candidate_when_missing():
    tok = _FakeTokenizer(chat_template=None)
    ensure_chat_template(tok, TEMPLATE, "base_template")
    assert tok.chat_template == TEMPLATE


def test_ensure_skips_empty_candidates():
    tok = _FakeTokenizer(chat_template=None)
    ensure_chat_template(tok, None, "", TEMPLATE)
    assert tok.chat_template == TEMPLATE


def test_ensure_falls_back_to_base_when_adapter_none():
    tok = _FakeTokenizer(chat_template=None)
    ensure_chat_template(tok, None, "base_template")
    assert tok.chat_template == "base_template"


def test_ensure_leaves_none_when_no_candidates():
    tok = _FakeTokenizer(chat_template=None)
    ensure_chat_template(tok, None, None)
    assert tok.chat_template is None


# ── sanitize_tokenizer_config ─────────────────────────────────────────────────


def test_sanitize_noop_without_config(tmp_path):
    sanitize_tokenizer_config(str(tmp_path))  # must not raise
    assert not os.path.exists(os.path.join(tmp_path, TOKENIZER_CONFIG_FILE))


def test_sanitize_rewrites_tokenizers_backend_with_tokenizer_json(tmp_path):
    _write_config(tmp_path, {"tokenizer_class": "TokenizersBackend"})
    _write(tmp_path, "tokenizer.json", "{}")
    sanitize_tokenizer_config(str(tmp_path))
    config = json.load(open(os.path.join(tmp_path, TOKENIZER_CONFIG_FILE)))
    assert config["tokenizer_class"] == "PreTrainedTokenizerFast"


def test_sanitize_drops_tokenizers_backend_without_tokenizer_json(tmp_path):
    _write_config(tmp_path, {"tokenizer_class": "TokenizersBackend"})
    sanitize_tokenizer_config(str(tmp_path))
    config = json.load(open(os.path.join(tmp_path, TOKENIZER_CONFIG_FILE)))
    assert "tokenizer_class" not in config


def test_sanitize_normalizes_extra_special_tokens_list(tmp_path):
    _write_config(tmp_path, {"extra_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    sanitize_tokenizer_config(str(tmp_path))
    config = json.load(open(os.path.join(tmp_path, TOKENIZER_CONFIG_FILE)))
    assert config["extra_special_tokens"] == {"<|im_start|>": "<|im_start|>", "<|im_end|>": "<|im_end|>"}


def test_sanitize_folds_jinja_into_inline(tmp_path):
    # v5 writes the template only as a standalone file and pops the inline key; fold it back so a
    # pre-4.47 consumer reading only tokenizer_config.json still finds it.
    _write(tmp_path, CHAT_TEMPLATE_FILE, TEMPLATE)
    _write_config(tmp_path, {"model_max_length": 2048})
    sanitize_tokenizer_config(str(tmp_path))
    config = json.load(open(os.path.join(tmp_path, TOKENIZER_CONFIG_FILE)))
    assert config["chat_template"] == TEMPLATE
    # the standalone file is kept too, so jinja-preferring loaders still see the same template
    assert os.path.exists(os.path.join(tmp_path, CHAT_TEMPLATE_FILE))


def test_sanitize_does_not_clobber_existing_inline_template(tmp_path):
    _write(tmp_path, CHAT_TEMPLATE_FILE, TEMPLATE)
    _write_config(tmp_path, {"chat_template": OTHER_TEMPLATE})
    sanitize_tokenizer_config(str(tmp_path))
    config = json.load(open(os.path.join(tmp_path, TOKENIZER_CONFIG_FILE)))
    assert config["chat_template"] == OTHER_TEMPLATE


def test_sanitize_handles_all_quirks_together(tmp_path):
    _write(tmp_path, "tokenizer.json", "{}")
    _write(tmp_path, CHAT_TEMPLATE_FILE, TEMPLATE)
    _write_config(
        tmp_path,
        {"tokenizer_class": "TokenizersBackend", "extra_special_tokens": ["<|im_start|>"]},
    )
    sanitize_tokenizer_config(str(tmp_path))
    config = json.load(open(os.path.join(tmp_path, TOKENIZER_CONFIG_FILE)))
    assert config["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert config["extra_special_tokens"] == {"<|im_start|>": "<|im_start|>"}
    assert config["chat_template"] == TEMPLATE
