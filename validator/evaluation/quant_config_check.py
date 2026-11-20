"""Fixed-track check of a checkpoint's config.json quantization_config, run before vLLM loads it.

vLLM loads whatever the config describes, and the kernel it picks does not pin the format (W6A16, W3A16
and W2A16-asym-g64 all route to Humming; W4A8-int8 routes to Machete like W4A16), so the config
must spell out exactly the task's QuantFixedSpec:

- compressed-tensors, pack-quantized (int-quantized checkpoints do not load in vLLM 0.31).
- One config group targeting Linear, so every linear layer is quantised the same way.
- Weights: int, group strategy, the spec's bits, group size and symmetry, no activation reordering,
  bf16 scales (or none stated: older compressed-tensors omits scale_dtype and stores bf16).
- Input activations: none for A16; symmetric dynamic per-token 8-bit float/int for A8fp8/A8int8.
- Only lm_head ignored; no KV-cache scheme (the KV cache stays bf16), sparsity or transforms.

Everything else in the config must be the input model's (config_drift_failure). vLLM builds the model from the
submission's own config, so rms_norm_eps, rope_theta/rope_scaling, max_position_embeddings, tie_word_embeddings,
vocab or architecture fields would otherwise be free knobs that lower loss without compressing anything. The two
configs are compared as vLLM resolves them (quant_vllm.resolved_model_config), not as raw JSON: transformers 5
rewrites a v4 config on save (rope_theta and rope_scaling into rope_parameters, derived layer_types, head_dim,
null pad_token_id), so a raw diff would fail honest saves. Resolved, a faithful save
equals its input except for CONFIG_KEYS_FREE_TO_DIFFER.

Pure Python so it runs on CPU against parsed config dicts.
"""

from core.models.quant_models import QuantActivationFormat
from core.models.quant_models import QuantFixedSpec


COMPRESSED_TENSORS_METHOD = "compressed-tensors"
PACK_QUANTIZED_FORMAT = "pack-quantized"
QUANTISED_TARGETS = ["Linear"]
IGNORED_MODULES = ["lm_head"]
ACTIVATION_TYPES = {QuantActivationFormat.FP8: "float", QuantActivationFormat.INT8: "int"}
# compressed-tensors writes scale_dtype as str(torch dtype); None is older versions that omit it.
ACCEPTED_SCALE_DTYPES = (None, "torch.bfloat16")
# Resolved-config keys a submission may change, at any depth. None reaches the scored model:
# - quantization_config: checked against the spec by quant_config_failure instead.
# - transformers_version: the writing library's version; to_dict() stamps the reader's anyway.
# - _name_or_path: the trainer scrubs it from the miner's copy (scrub_model_identity) and the resolver sets it to
#   the load path, so it differs for every honest submission.
# - dtype / torch_dtype (the v4 spelling): vLLM is always built with dtype=bfloat16, and save_pretrained writes the
#   in-memory dtype.
CONFIG_KEYS_FREE_TO_DIFFER = frozenset({"quantization_config", "transformers_version", "_name_or_path", "dtype", "torch_dtype"})
DRIFT_VALUE_MAX_CHARS = 80


def _mismatch(where: str, actual: dict, expected: dict) -> str | None:
    for key, value in expected.items():
        if actual.get(key) != value:
            return f"{where}.{key} is {actual.get(key)!r}, expected {value!r}"
    return None


def _expected_weights(spec: QuantFixedSpec) -> dict:
    return {
        "type": "int",
        "strategy": "group",
        "num_bits": spec.weight_bits,
        "group_size": spec.group_size,
        "symmetric": spec.symmetric,
        "actorder": None,
    }


def _weights_failure(where: str, weights: dict, spec: QuantFixedSpec) -> str | None:
    failure = _mismatch(where, weights, _expected_weights(spec))
    if failure is None and weights.get("scale_dtype") not in ACCEPTED_SCALE_DTYPES:
        return f"{where}.scale_dtype is {weights['scale_dtype']!r}, expected 'torch.bfloat16'"
    return failure


def _input_activations_failure(where: str, activations: dict | None, spec: QuantFixedSpec) -> str | None:
    if spec.activation == QuantActivationFormat.BF16:
        return None if activations is None else f"{where}.input_activations must be null for {spec.name}"
    if not isinstance(activations, dict):
        return f"{where}.input_activations is missing; {spec.name} quantises activations"
    expected = {
        "type": ACTIVATION_TYPES[spec.activation],
        "num_bits": 8,
        "strategy": "token",
        "dynamic": True,
        "symmetric": True,
    }
    return _mismatch(f"{where}.input_activations", activations, expected)


def _config_group_failure(name: str, group: dict, spec: QuantFixedSpec) -> str | None:
    where = f"config_groups.{name}"
    failure = _mismatch(where, group, {"targets": QUANTISED_TARGETS, "output_activations": None})
    if failure is not None:
        return failure
    if group.get("format") not in (None, PACK_QUANTIZED_FORMAT):
        return f"{where}.format is {group['format']!r}, expected {PACK_QUANTIZED_FORMAT!r}"
    weights = group.get("weights")
    if not isinstance(weights, dict):
        return f"{where}.weights is missing"
    return _weights_failure(f"{where}.weights", weights, spec) or _input_activations_failure(
        where, group.get("input_activations"), spec
    )


def quant_config_failure(config: object, spec: QuantFixedSpec) -> str | None:
    """Why parsed config.json does not describe a spec checkpoint, or None when it does."""
    if not isinstance(config, dict):
        return "config.json is not a JSON object"
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        return "config.json has no quantization_config"
    expected = {
        "quant_method": COMPRESSED_TENSORS_METHOD,
        "format": PACK_QUANTIZED_FORMAT,
        "ignore": IGNORED_MODULES,
        "kv_cache_scheme": None,
    }
    failure = _mismatch("quantization_config", quant_config, expected)
    if failure is not None:
        return failure
    for extra in ("sparsity_config", "transform_config"):
        if quant_config.get(extra):
            return f"quantization_config.{extra} must be empty"
    groups = quant_config.get("config_groups")
    if not isinstance(groups, dict) or len(groups) != 1:
        return "quantization_config.config_groups must hold exactly one group"
    ((name, group),) = groups.items()
    if not isinstance(group, dict):
        return f"config_groups.{name} is not an object"
    return _config_group_failure(name, group, spec)


def _shown(value: object) -> str:
    text = repr(value)
    return text if len(text) <= DRIFT_VALUE_MAX_CHARS else text[: DRIFT_VALUE_MAX_CHARS - 3] + "..."


def config_drift_failure(submitted: dict, base: dict, where: str = "config") -> str | None:
    """The first resolved-config field where the submission differs from the input model, or None.

    Both dicts are full resolved configs (defaults filled in), so a key missing on one side is a difference too:
    a submission cannot drop a field to fall back on a class default the input model overrides.
    """
    for key in sorted(set(submitted) | set(base)):
        if key in CONFIG_KEYS_FREE_TO_DIFFER:
            continue
        path = f"{where}.{key}"
        if key not in base:
            return f"{path} is set ({_shown(submitted[key])}) but the input model has no {key}"
        if key not in submitted:
            return f"{path} is missing; the input model has {_shown(base[key])}"
        value, expected = submitted[key], base[key]
        if isinstance(value, dict) and isinstance(expected, dict):
            failure = config_drift_failure(value, expected, path)
            if failure is not None:
                return failure
        elif value != expected:
            return f"{path} is {_shown(value)}, the input model has {_shown(expected)}"
    return None
