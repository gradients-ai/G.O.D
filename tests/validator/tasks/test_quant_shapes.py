"""Quant model-pool filters per track (freeform, fixed) and the seeded fixed-track format draw."""

import inspect
import json
import random

import pytest

import core.constants.credentials as credentials
from core.models.quant_models import LAUNCH_QUANT_FORMATS
from core.models.task_models import TaskType
from validator.tasks import quant_shapes
from validator.tasks.quant_shapes import filter_has_chat_template
from validator.tasks.quant_shapes import filter_quant_compatible
from validator.tasks.quant_shapes import fixed_compatible
from validator.tasks.quant_shapes import freeform_compatible
from validator.tasks.quant_shapes import quant_compatible
from validator.tasks.synthetics.scheduler import draw_quant_format


def _qwen2_1_5b(**overrides) -> dict:
    config = {
        "model_type": "qwen2",
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "head_dim": 128,
    }
    config.update(overrides)
    return config


def test_dense_qwen2_config_is_compatible():
    assert quant_compatible(_qwen2_1_5b(), TaskType.QUANTFIXEDTASK) is None
    assert quant_compatible(_qwen2_1_5b(), TaskType.QUANTFREEFORMTASK) is None


def test_head_dim_defaults_to_hidden_over_heads():
    config = _qwen2_1_5b()
    del config["head_dim"]
    assert quant_compatible(config, TaskType.QUANTFIXEDTASK) is None  # 1536 / 12 = 128, 2 kv heads -> 256


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"auto_map": {"AutoModel": "modeling_x.X"}}, "remote code"),
        ({"num_local_experts": 8}, "mixture-of-experts"),
        ({"model_type": "mamba"}, "model_type"),
        ({"intermediate_size": 8964}, "intermediate_size"),
        ({"num_key_value_heads": 3, "head_dim": 64}, "num_key_value_heads * head_dim=192"),
    ],
)
def test_fixed_incompatible_configs_give_a_reason(overrides, reason_fragment):
    reason = quant_compatible(_qwen2_1_5b(**overrides), TaskType.QUANTFIXEDTASK)
    assert reason is not None and reason_fragment in reason


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({"auto_map": {"AutoModel": "modeling_x.X"}}, "remote code"),
        ({"num_local_experts": 8}, "mixture-of-experts"),
        ({"model_type": "mamba"}, "model_type"),
    ],
)
def test_freeform_rejects_remote_code_moe_and_unsupported_types(overrides, reason_fragment):
    reason = freeform_compatible(_qwen2_1_5b(**overrides))
    assert reason is not None and reason_fragment in reason


@pytest.mark.parametrize(
    "overrides",
    [{"intermediate_size": 8964}, {"num_key_value_heads": 3, "head_dim": 64}, {"hidden_size": 1000}],
)
def test_non_divisible_shapes_are_freeform_compatible_but_not_fixed(overrides):
    """The freeform path assumes no shapes, so only the fixed rule checks divisibility by the launch group size."""
    config = _qwen2_1_5b(**overrides)
    assert freeform_compatible(config) is None
    assert quant_compatible(config, TaskType.QUANTFREEFORMTASK) is None
    assert fixed_compatible(config) is not None


def test_freeform_needs_no_shape_keys():
    assert freeform_compatible({"model_type": "llama"}) is None
    assert "missing" in fixed_compatible({"model_type": "llama"})


def test_quant_compatible_rejects_a_non_quant_track():
    with pytest.raises(ValueError):
        quant_compatible(_qwen2_1_5b(), TaskType.INSTRUCTTEXTTASK)


def test_missing_shape_key_gives_a_reason():
    config = _qwen2_1_5b()
    del config["num_key_value_heads"]
    assert "missing" in quant_compatible(config, TaskType.QUANTFIXEDTASK)


async def test_filter_yields_only_compatible_models_and_skips_download_errors(monkeypatch, tmp_path):
    configs = {
        "org/dense": _qwen2_1_5b(),
        "org/moe": _qwen2_1_5b(num_experts=64),
        "org/also-dense": _qwen2_1_5b(model_type="llama"),
        "org/odd-shape": _qwen2_1_5b(intermediate_size=8964),
    }

    def fake_download(model_id, filename, token=None):
        if model_id == "org/gone":
            raise OSError("404")
        path = tmp_path / f"{model_id.replace('/', '--')}-{filename}"
        path.write_text(json.dumps(configs[model_id]))
        return str(path)

    monkeypatch.setattr(quant_shapes, "hf_hub_download", fake_download)

    async def models():
        for model_id in ("org/dense", "org/moe", "org/gone", "org/odd-shape", "org/also-dense"):
            yield model_id

    fixed = [model_id async for model_id in filter_quant_compatible(models(), TaskType.QUANTFIXEDTASK)]
    assert fixed == ["org/dense", "org/also-dense"]
    freeform = [model_id async for model_id in filter_quant_compatible(models(), TaskType.QUANTFREEFORMTASK)]
    assert freeform == ["org/dense", "org/odd-shape", "org/also-dense"]


async def test_filter_requires_a_quant_track():
    async def models():
        yield "org/dense"

    with pytest.raises(ValueError):
        [model_id async for model_id in filter_quant_compatible(models(), TaskType.INSTRUCTTEXTTASK)]
    assert inspect.signature(filter_quant_compatible).parameters["track"].default is inspect.Parameter.empty


def test_filter_defaults_to_the_validator_hf_token():
    assert inspect.signature(filter_quant_compatible).parameters["token"].default is credentials.HUGGINGFACE_TOKEN


async def test_filter_accepts_an_explicit_none_token(monkeypatch, tmp_path):
    tokens = []

    def fake_download(model_id, filename, token=None):
        tokens.append(token)
        path = tmp_path / "config.json"
        path.write_text(json.dumps(_qwen2_1_5b()))
        return str(path)

    async def one_model():
        yield "org/gated"

    monkeypatch.setattr(quant_shapes, "hf_hub_download", fake_download)

    kept = [model_id async for model_id in filter_quant_compatible(one_model(), TaskType.QUANTFIXEDTASK, token=None)]
    assert kept == ["org/gated"]
    assert tokens == [None]


async def test_chat_filter_yields_only_tokenizers_with_a_chat_template(monkeypatch):
    templates = {"org/instruct": "{{ messages }}", "org/base-no-template": None, "org/unreadable": None}
    calls = []

    def fake_read_chat_template(model_id, token=None):
        calls.append((model_id, token))
        return templates[model_id]

    monkeypatch.setattr(quant_shapes, "read_chat_template", fake_read_chat_template)

    async def models():
        for model_id in templates:
            yield model_id

    assert [model_id async for model_id in filter_has_chat_template(models(), token="hf_x")] == ["org/instruct"]
    assert calls == [(model_id, "hf_x") for model_id in templates]


async def test_chat_filter_reads_the_template_from_the_hub_tokenizer_files(monkeypatch, tmp_path):
    """No template anywhere (a base tokenizer) is skipped; a standalone chat_template.jinja or an inline string counts."""
    files = {
        ("org/jinja", "chat_template.jinja"): "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}",
        ("org/inline", "tokenizer_config.json"): json.dumps({"chat_template": "{{ messages }}"}),
        ("org/base", "tokenizer_config.json"): json.dumps({"bos_token": "<s>"}),
    }

    def fake_download(model_id, filename, token=None):
        if (model_id, filename) not in files:
            raise OSError(f"404 {filename}")
        path = tmp_path / f"{model_id.replace('/', '--')}-{filename}"
        path.write_text(files[(model_id, filename)])
        return str(path)

    monkeypatch.setattr("core.tokenizer_utils.hf_hub_download", fake_download)

    async def models():
        for model_id in ("org/jinja", "org/base", "org/inline"):
            yield model_id

    assert [model_id async for model_id in filter_has_chat_template(models())] == ["org/jinja", "org/inline"]


def test_chat_filter_defaults_to_the_validator_hf_token():
    assert inspect.signature(filter_has_chat_template).parameters["token"].default is credentials.HUGGINGFACE_TOKEN


def test_format_draw_is_seeded_and_from_the_launch_set():
    first = draw_quant_format(random.Random(1))
    assert first == draw_quant_format(random.Random(1))
    assert first in LAUNCH_QUANT_FORMATS
