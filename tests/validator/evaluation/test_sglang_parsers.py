import json
import re
from pathlib import Path

import pytest
from jinja2 import Environment

from core.pvp.sglang_parsers import CHAT_TEMPLATE_ENV
from core.pvp.sglang_parsers import MODEL_ARGS_ENV
from core.pvp.sglang_parsers import TOOL_CALL_PARSER_ENV
from core.pvp.sglang_parsers import requires_lora_merge_for_serving
from core.pvp.sglang_parsers import sglang_model_args_for
from core.pvp.sglang_parsers import tool_call_parser_for
from core.pvp.sglang_parsers import tool_chat_template_for


def test_family_substring_in_model_id():
    assert tool_call_parser_for("Qwen/Qwen2.5-0.5B-Instruct") == "qwen"
    assert tool_call_parser_for("meta-llama/Llama-3.1-8B-Instruct") == "llama3"
    assert tool_call_parser_for("NousResearch/Hermes-3-Llama-3.1-8B") == "hermes"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv(TOOL_CALL_PARSER_ENV, "qwen25")
    assert tool_call_parser_for("/cache/models/a3f9c2e1b4d8f7a0") == "qwen25"


def test_opaque_id_without_weights_is_unmapped():
    assert tool_call_parser_for("gradients-io/augmented-a3f9c2e1b4d8f7a0") is None


def test_model_type_fallback_for_anonymized_local_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    assert tool_call_parser_for(str(tmp_path)) == "qwen"


def test_local_dir_with_unknown_model_type_is_unmapped(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt_bigcode"}))
    assert tool_call_parser_for(str(tmp_path)) is None


def test_local_dir_with_malformed_config_is_unmapped(tmp_path):
    (tmp_path / "config.json").write_text("not json")
    assert tool_call_parser_for(str(tmp_path)) is None


@pytest.mark.parametrize(
    ("model_id", "parser"),
    [
        ("Qwen/Qwen3.5-0.8B", "qwen3_coder"),
        ("Qwen/Qwen3.5-4B", "qwen3_coder"),
        ("ibm-granite/granite-4.1-3b", "qwen"),
        ("allenai/Olmo-3-7B-Instruct", "olmo"),
        ("allenai/Olmo-Hybrid-Instruct-SFT-7B", "olmo"),
        ("LiquidAI/LFM2.5-2.6B", "lfm2"),
        ("mistralai/Ministral-3-3B-Base-2512", "qwen"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "qwen3_coder"),
        ("google/gemma-4-E2B", "qwen"),
    ],
)
def test_round_one_model_parser_mapping(model_id, parser):
    assert tool_call_parser_for(model_id) == parser


@pytest.mark.parametrize(
    ("model_type", "parser"),
    [
        ("qwen3_5", "qwen3_coder"),
        ("granite", "qwen"),
        ("olmo3", "olmo"),
        ("olmo_hybrid", "olmo"),
        ("lfm2", "lfm2"),
        ("mistral3", "mistral"),
        ("nemotron_h", "qwen3_coder"),
        ("gemma4", "gemma4"),
    ],
)
def test_round_one_model_type_fallback(tmp_path, model_type, parser):
    model_dir = tmp_path / model_type
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    assert tool_call_parser_for(str(model_dir)) == parser


@pytest.mark.parametrize(
    "model_id",
    ["google/gemma-4-E2B", "mistralai/Ministral-3-3B-Base-2512"],
)
def test_base_models_use_checked_in_tool_chat_template(model_id):
    template_path = tool_chat_template_for(model_id)
    assert template_path is not None
    assert Path(template_path).is_file()
    assert "<tool_call>" in Path(template_path).read_text()


@pytest.mark.parametrize(
    "model_id",
    ["google/gemma-4-E2B", "mistralai/Ministral-3-3B-Base-2512"],
)
def test_base_model_template_renders_qwen_tool_call_contract(model_id):
    template_path = tool_chat_template_for(model_id)
    assert template_path is not None
    template = Environment().from_string(Path(template_path).read_text())
    rendered = template.render(
        bos_token="",
        eos_token="",
        add_generation_prompt=False,
        tools=[{"type": "function", "function": {"name": "play", "parameters": {"type": "object"}}}],
        messages=[
            {"role": "user", "content": "move"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "play", "arguments": {"card": "ace"}}}],
            },
        ],
    )

    matches = re.findall(r"<tool_call>\n(.*?)\n</tool_call>", rendered, re.DOTALL)
    assert matches
    assert json.loads(matches[-1]) == {"name": "play", "arguments": {"card": "ace"}}


def test_chat_template_and_model_args_overrides_win(monkeypatch):
    monkeypatch.setenv(CHAT_TEMPLATE_ENV, "/tmp/custom.jinja")
    monkeypatch.setenv(MODEL_ARGS_ENV, "--model-impl custom")
    assert tool_chat_template_for("google/gemma-4-E2B") == "/tmp/custom.jinja"
    assert sglang_model_args_for("allenai/Olmo-Hybrid-Instruct-SFT-7B") == "--model-impl custom"


def test_olmo_hybrid_has_no_sglang_backend_override(monkeypatch):
    monkeypatch.delenv(MODEL_ARGS_ENV, raising=False)
    assert sglang_model_args_for("allenai/Olmo-Hybrid-Instruct-SFT-7B") == ""


@pytest.mark.parametrize(
    "model_id",
    [
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "ibm-granite/granite-4.1-3b",
        "allenai/Olmo-3-7B-Instruct",
        "allenai/Olmo-Hybrid-Instruct-SFT-7B",
        "LiquidAI/LFM2.5-2.6B",
        "mistralai/Ministral-3-3B-Base-2512",
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "google/gemma-4-E2B",
    ],
)
def test_families_without_matching_native_lora_path_are_merged(model_id):
    assert requires_lora_merge_for_serving(model_id)


def test_existing_native_lora_family_remains_enabled():
    assert not requires_lora_merge_for_serving("Qwen/Qwen2.5-3B-Instruct")
