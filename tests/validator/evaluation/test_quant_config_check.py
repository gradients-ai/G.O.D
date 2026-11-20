import copy
import json
from pathlib import Path

import pytest

from core.models.quant_models import QuantFixedSpec
from core.models.quant_models import QuantFormat
from validator.evaluation.quant_config_check import CONFIG_KEYS_FREE_TO_DIFFER
from validator.evaluation.quant_config_check import config_drift_failure
from validator.evaluation.quant_config_check import quant_config_failure


CONFIG_FIXTURES = Path(__file__).parent / "fixtures" / "quant_configs"
W3A16 = QuantFormat.W3A16_ASYM_G128.spec()
W4A16 = QuantFormat.W4A16_SYM_G128.spec()
W2A16 = QuantFormat.W2A16_ASYM_G64.spec()
PER_TOKEN_DYNAMIC = {"strategy": "token", "dynamic": True, "symmetric": True, "num_bits": 8}


def _config(spec: QuantFixedSpec, activation_type: str | None = None) -> dict:
    """config.json as llm-compressor writes it for spec (pack-quantized, lm_head ignored)."""
    if activation_type is None and spec.activation.value != "bf16":
        activation_type = "float" if spec.activation.value == "fp8" else "int"
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "quantization_status": "compressed",
            "global_compression_ratio": None,
            "ignore": ["lm_head"],
            "kv_cache_scheme": None,
            "sparsity_config": {},
            "transform_config": {},
            "version": "0.19.0",
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": spec.weight_bits,
                        "type": "int",
                        "strategy": "group",
                        "group_size": spec.group_size,
                        "symmetric": spec.symmetric,
                        "actorder": None,
                        "dynamic": False,
                        "observer": "memoryless_minmax",
                        "scale_dtype": "torch.bfloat16",
                    },
                    "input_activations": None
                    if activation_type is None
                    else {**PER_TOKEN_DYNAMIC, "type": activation_type, "observer": None},
                    "output_activations": None,
                }
            },
        },
    }


def _with(config: dict, path: str, value) -> dict:
    """Copy of config with the dotted path under quantization_config set to value."""
    config = copy.deepcopy(config)
    *parents, leaf = path.split(".")
    node = config["quantization_config"]
    for key in parents:
        node = node[key]
    node[leaf] = value
    return config


@pytest.mark.parametrize("quant_format", list(QuantFormat), ids=lambda f: f.value)
def test_accepts_honest_format_config(quant_format):
    assert quant_config_failure(_config(quant_format.spec()), quant_format.spec()) is None


def test_accepts_group_without_its_own_format():
    config = copy.deepcopy(_config(W3A16))
    del config["quantization_config"]["config_groups"]["group_0"]["format"]

    assert quant_config_failure(config, W3A16) is None


def test_w6a16_posing_as_w3a16_is_rejected():
    w6 = W3A16.model_copy(update={"weight_bits": 6})

    assert quant_config_failure(_config(w6), W3A16) == "config_groups.group_0.weights.num_bits is 6, expected 3"


def test_w2a16_g64_posing_as_w3a16_is_rejected():
    w2_g64 = W3A16.model_copy(update={"weight_bits": 2, "group_size": 64})

    assert "num_bits is 2" in quant_config_failure(_config(w2_g64), W3A16)


def test_accepts_honest_w2a16_asym_g64_config():
    assert quant_config_failure(_config(W2A16), W2A16) is None


@pytest.mark.parametrize(
    ("update", "failure"),
    [
        ({"symmetric": True}, "symmetric is True"),
        ({"group_size": 128}, "group_size is 128, expected 64"),
        ({"weight_bits": 3}, "num_bits is 3, expected 2"),
        ({"weight_bits": 4, "symmetric": True}, "num_bits is 4, expected 2"),
    ],
    ids=["sym", "g128", "w3", "w4-sym"],
)
def test_w2a16_asym_g64_rejects_other_variants(update, failure):
    other = W2A16.model_copy(update=update)

    assert failure in quant_config_failure(_config(other), W2A16)


def test_w2a16_asym_g64_config_is_rejected_under_w3a16():
    assert "num_bits is 2" in quant_config_failure(_config(W2A16), W3A16)


def test_group_size_alone_is_checked():
    g64 = W3A16.model_copy(update={"group_size": 64})

    assert quant_config_failure(_config(g64), W3A16) == "config_groups.group_0.weights.group_size is 64, expected 128"


def test_symmetry_is_checked():
    symmetric = W3A16.model_copy(update={"symmetric": True})

    assert "symmetric is True" in quant_config_failure(_config(symmetric), W3A16)


def test_w4a8_int8_posing_as_w4a16_is_rejected():
    assert quant_config_failure(_config(W4A16, activation_type="int"), W4A16) == (
        "config_groups.group_0.input_activations must be null for W4A16-sym-g128"
    )


def test_w4a8_int8_posing_as_w4a8_fp8_is_rejected():
    spec = QuantFormat.W4A8_FP8_G128.spec()

    failure = quant_config_failure(_config(spec, activation_type="int"), spec)

    assert failure == "config_groups.group_0.input_activations.type is 'int', expected 'float'"


def test_w4a8_without_activation_quant_is_rejected():
    spec = QuantFormat.W4A8_FP8_G128.spec()

    failure = quant_config_failure(_with(_config(spec), "config_groups.group_0.input_activations", None), spec)

    assert "input_activations is missing" in failure


def test_static_activations_are_rejected():
    spec = QuantFormat.W4A8_FP8_G128.spec()

    failure = quant_config_failure(_with(_config(spec), "config_groups.group_0.input_activations.dynamic", False), spec)

    assert failure == "config_groups.group_0.input_activations.dynamic is False, expected True"


@pytest.mark.parametrize(
    ("path", "value", "failure"),
    [
        ("format", "int-quantized", "quantization_config.format is 'int-quantized', expected 'pack-quantized'"),
        ("config_groups.group_0.format", "int-quantized", "group_0.format is 'int-quantized'"),
        ("quant_method", "bitsandbytes", "quant_method is 'bitsandbytes'"),
        ("ignore", ["lm_head", "re:.*mlp.*"], "ignore is ['lm_head', 're:.*mlp.*'], expected ['lm_head']"),
        ("ignore", [], "ignore is []"),
        ("config_groups.group_0.targets", ["re:.*self_attn.*"], "targets is ['re:.*self_attn.*'], expected ['Linear']"),
        ("kv_cache_scheme", {"num_bits": 8, "type": "float"}, "kv_cache_scheme is {"),
        ("sparsity_config", {"format": "sparse-24-bitmask"}, "sparsity_config must be empty"),
        ("transform_config", {"config_groups": {"R1": {}}}, "transform_config must be empty"),
        ("config_groups.group_0.weights.type", "float", "weights.type is 'float', expected 'int'"),
        ("config_groups.group_0.weights.strategy", "channel", "weights.strategy is 'channel'"),
        ("config_groups.group_0.weights.actorder", "group", "weights.actorder is 'group', expected None"),
        ("config_groups.group_0.output_activations", {"num_bits": 8}, "output_activations is {"),
        ("config_groups.group_0.weights", None, "group_0.weights is missing"),
    ],
    ids=[
        "int-quantized",
        "group-int-quantized",
        "not-compressed-tensors",
        "bf16-mlps-ignored",
        "lm-head-quantised",
        "bf16-mlps-untargeted",
        "kv-cache-scheme",
        "sparsity",
        "transforms",
        "float-weights",
        "channel-strategy",
        "actorder",
        "output-activations",
        "no-weights",
    ],
)
def test_config_surprises_are_rejected(path, value, failure):
    assert failure in quant_config_failure(_with(_config(W3A16), path, value), W3A16)


def test_bf16_mlps_as_a_second_group_are_rejected():
    config = _config(W3A16)
    groups = config["quantization_config"]["config_groups"]
    groups["group_1"] = {**groups["group_0"], "targets": ["re:.*mlp.*"]}

    assert quant_config_failure(config, W3A16) == "quantization_config.config_groups must hold exactly one group"


def test_unquantised_checkpoint_is_rejected():
    assert quant_config_failure({"architectures": ["Qwen2ForCausalLM"]}, W3A16) == "config.json has no quantization_config"


@pytest.mark.parametrize("config", [[], "x", None], ids=["list", "str", "null"])
def test_non_object_config_is_rejected(config):
    assert quant_config_failure(config, W3A16) == "config.json is not a JSON object"


@pytest.mark.parametrize("scale_dtype", ["torch.bfloat16", None], ids=["bf16", "unstated"])
def test_bf16_or_unstated_scales_are_accepted(scale_dtype):
    config = _with(_config(W3A16), "config_groups.group_0.weights.scale_dtype", scale_dtype)

    assert quant_config_failure(config, W3A16) is None


def test_fp32_scales_are_rejected():
    config = _with(_config(W3A16), "config_groups.group_0.weights.scale_dtype", "torch.float32")

    assert quant_config_failure(config, W3A16) == (
        "config_groups.group_0.weights.scale_dtype is 'torch.float32', expected 'torch.bfloat16'"
    )


def _fixture(name: str) -> dict:
    return json.loads((CONFIG_FIXTURES / f"{name}.json").read_text())


@pytest.mark.parametrize("name", ["olmo2_1b", "gemma3_1b"])
def test_an_honest_transformers_5_save_resolves_to_the_input_models_config(name):
    configs = _fixture(name)
    raw_changes = {
        key
        for key in set(configs["input_model_raw"]) | set(configs["honest_save_raw"])
        if configs["input_model_raw"].get(key) != configs["honest_save_raw"].get(key)
    }

    # The file on disk changes beyond the allowlist (dtype float32 -> bfloat16; Gemma's rope fields re-spelled), so
    # the comparison has to be on resolved configs, where only allowlisted keys differ.
    assert raw_changes - CONFIG_KEYS_FREE_TO_DIFFER
    assert config_drift_failure(configs["honest_save"], configs["input_model"]) is None


def _gemma_resolved(**overrides) -> tuple[dict, dict]:
    configs = _fixture("gemma3_1b")
    submitted = copy.deepcopy(configs["honest_save"])
    submitted.update(overrides)
    return submitted, configs["input_model"]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"rms_norm_eps": 1e-05}, "config.rms_norm_eps is 1e-05, the input model has 1e-06"),
        ({"max_position_embeddings": 4096}, "config.max_position_embeddings is 4096, the input model has 32768"),
        ({"tie_word_embeddings": False}, "config.tie_word_embeddings is False, the input model has True"),
        ({"vocab_size": 262272}, "config.vocab_size is 262272, the input model has 262144"),
        ({"final_logit_softcapping": 30.0}, "config.final_logit_softcapping is 30.0, the input model has None"),
        (
            {"architectures": ["Gemma2ForCausalLM"]},
            "config.architectures is ['Gemma2ForCausalLM'], the input model has ['Gemma3ForCausalLM']",
        ),
        ({"logit_scale": 0.9}, "config.logit_scale is set (0.9) but the input model has no logit_scale"),
    ],
    ids=["rms_norm_eps", "max_position_embeddings", "tie_word_embeddings", "vocab_size", "softcap", "architectures", "new-key"],
)
def test_a_changed_model_field_is_drift(overrides, reason):
    assert config_drift_failure(*_gemma_resolved(**overrides)) == reason


def test_a_changed_nested_rope_theta_is_drift():
    submitted, base = _gemma_resolved()
    submitted["rope_parameters"]["full_attention"]["rope_theta"] = 2000000.0

    assert config_drift_failure(submitted, base) == (
        "config.rope_parameters.full_attention.rope_theta is 2000000.0, the input model has 1000000"
    )


def test_added_rope_scaling_is_drift():
    submitted, base = _gemma_resolved()
    submitted["rope_parameters"]["full_attention"].update(rope_type="linear", factor=4.0)

    assert config_drift_failure(submitted, base) == (
        "config.rope_parameters.full_attention.factor is set (4.0) but the input model has no factor"
    )


def test_a_dropped_field_is_drift():
    submitted, base = _gemma_resolved()
    del submitted["query_pre_attn_scalar"]

    assert config_drift_failure(submitted, base) == "config.query_pre_attn_scalar is missing; the input model has 256"


@pytest.mark.parametrize(
    "overrides",
    [
        {"transformers_version": "4.57.1"},
        {"_name_or_path": "/cache/models/0123456789abcdef"},
        {"dtype": "float16"},
        {"torch_dtype": "float32"},
        {"quantization_config": {"quant_method": "bitsandbytes"}},
    ],
    ids=["transformers_version", "name_or_path", "dtype", "torch_dtype", "quantization_config"],
)
def test_allowlisted_fields_may_differ(overrides):
    assert config_drift_failure(*_gemma_resolved(**overrides)) is None


def test_long_values_are_shortened_in_the_reason():
    submitted, base = _gemma_resolved()
    submitted["layer_types"] = ["full_attention"] * len(base["layer_types"])

    reason = config_drift_failure(submitted, base)

    assert reason.startswith("config.layer_types is ['full_attention', ")
    assert "..." in reason and len(reason) < 250
