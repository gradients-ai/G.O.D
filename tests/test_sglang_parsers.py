import json

from core.pvp.sglang_parsers import TOOL_CALL_PARSER_ENV
from core.pvp.sglang_parsers import tool_call_parser_for


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
