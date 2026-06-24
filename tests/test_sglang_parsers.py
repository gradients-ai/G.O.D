import json

import pytest

from core.pvp.sglang_parsers import TOOL_CALL_PARSER_ENV
from core.pvp.sglang_parsers import _parser_for_template
from core.pvp.sglang_parsers import _parser_from_chat_template
from core.pvp.sglang_parsers import tool_call_parser_for


@pytest.fixture(autouse=True)
def _clear_template_cache():
    _parser_from_chat_template.cache_clear()
    yield
    _parser_from_chat_template.cache_clear()


def test_family_substring_in_model_id():
    assert tool_call_parser_for("Qwen/Qwen2.5-0.5B-Instruct") == "qwen25"
    assert tool_call_parser_for("meta-llama/Llama-3.1-8B-Instruct") == "llama3"
    assert tool_call_parser_for("NousResearch/Hermes-3-Llama-3.1-8B") == "hermes"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv(TOOL_CALL_PARSER_ENV, "qwen25")
    assert tool_call_parser_for("/cache/models/a3f9c2e1b4d8f7a0") == "qwen25"


def test_opaque_id_without_weights_is_unmapped():
    assert tool_call_parser_for("gradients-io/augmented-a3f9c2e1b4d8f7a0") is None


def test_model_type_fallback_for_anonymized_local_dir(tmp_path):
    """Anonymized cache dirs carry no family in the path, but config.json keeps
    model_type — continuation-round miner repos must still resolve a parser."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    assert tool_call_parser_for(str(tmp_path)) == "qwen25"


def test_local_dir_with_unknown_model_type_is_unmapped(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt_bigcode"}))
    assert tool_call_parser_for(str(tmp_path)) is None


def test_local_dir_with_malformed_config_is_unmapped(tmp_path):
    (tmp_path / "config.json").write_text("not json")
    assert tool_call_parser_for(str(tmp_path)) is None


# --- chat-template resolution ---------------------------------------------


def test_parser_for_template_markers():
    assert _parser_for_template("...[TOOL_CALLS]...") == "mistral"
    assert _parser_for_template("...<|python_tag|>...") == "llama3"
    assert _parser_for_template("...<tool_call>{json}</tool_call>...") == "hermes"
    # qwen3-coder's <function= body wins over the shared <tool_call> wrapper.
    assert _parser_for_template("...<tool_call>\n<function=foo>...") == "qwen3_coder"
    assert _parser_for_template("no tool markers here") is None


def test_chat_template_in_tokenizer_config(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "render <tool_call>{...}</tool_call> please"})
    )
    assert tool_call_parser_for(str(tmp_path)) == "hermes"


def test_standalone_chat_template_jinja_preferred(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "[TOOL_CALLS]"}))
    (tmp_path / "chat_template.jinja").write_text("emits <|python_tag|>")
    # The standalone .jinja is the source of truth when both are present.
    assert tool_call_parser_for(str(tmp_path)) == "llama3"


def test_chat_template_as_named_list(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": [
                    {"name": "default", "template": "plain chat, no tools"},
                    {"name": "tool_use", "template": "uses <tool_call> wrapper"},
                ]
            }
        )
    )
    assert tool_call_parser_for(str(tmp_path)) == "hermes"


def test_chat_template_beats_misleading_model_type(tmp_path):
    """The hermes-on-llama regression: model_type reports the architecture
    (llama), but the chat template speaks hermes tool-call format. Template
    must win, otherwise the model forfeits every turn behind a llama3 parser."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "hermes finetune uses <tool_call>...</tool_call>"})
    )
    assert tool_call_parser_for(str(tmp_path)) == "hermes"


def test_named_repo_skips_template_io(monkeypatch):
    """A clearly-named repo resolves by substring without touching the chat
    template (keeps the cheap path cheap and avoids network for the common case)."""
    import core.pvp.sglang_parsers as mod

    def _boom(_model_id):
        raise AssertionError("chat-template resolver should not run for a named repo")

    monkeypatch.setattr(mod, "_parser_from_chat_template", _boom)
    assert tool_call_parser_for("NousResearch/Hermes-3-Llama-3.1-8B") == "hermes"


def test_local_dir_without_template_or_config_is_unmapped(tmp_path):
    assert tool_call_parser_for(str(tmp_path)) is None
