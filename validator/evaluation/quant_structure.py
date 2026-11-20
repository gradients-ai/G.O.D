"""Fixed-track structural pre-check: a submission's tensor set must equal what the base model implies.

The expected set (name, dtype, shape per tensor) is computed from the base model's safetensors header and
config.json under the task's QuantFixedSpec; the actual set is read from the submission's shard headers.
Both come from hub metadata only (never weights), so the check runs before any GPU is spent. A smaller
group, fp32 scales, an extra bf16 layer, g_idx or activation scales all change the set and fail here.
Config semantics (format, one group, no kv_cache_scheme) stay with the evaluator's config check.

Stdlib + huggingface_hub + pydantic only; no torch.
"""

import asyncio
import json
import math

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import NotASafetensorsRepoError
from huggingface_hub.errors import SafetensorsParsingError
from pydantic import BaseModel

from core.constants.credentials import HUGGINGFACE_TOKEN
from core.models.quant_models import QuantFixedSpec


QUANT_EMBEDDING_MARKERS = ("embed_tokens",)
LM_HEAD_WEIGHT = "lm_head.weight"
WEIGHT_SUFFIX = ".weight"
CONFIG_FILENAME = "config.json"
# Everything not quantised is stored bf16 whatever the base dtype; other dtypes fail the structural check.
UNQUANTISED_DTYPE = "BF16"
PACK_WORD_BITS = 32
STRUCTURAL_FAILURE_MAX_CHARS = 350
MAX_REPORTED_NAMES = 5


class StructuralCheckUnavailable(Exception):
    """Hub metadata could not be read: infrastructure, retried, never a scored fail."""


class StructuralMismatch(Exception):
    """The repo's content cannot pass the check (e.g. no safetensors weights): a scored fail."""


class TensorSpec(BaseModel):
    dtype: str  # safetensors dtype name: BF16, I32, I64, F32, ...
    shape: list[int]


class TensorSet(BaseModel):
    """tensors must all be present; optional ones may be absent but, when present, must match their spec."""

    tensors: dict[str, TensorSpec]
    optional: dict[str, TensorSpec] = {}


def fetch_tensor_set(repo: str, token: str | None = HUGGINGFACE_TOKEN) -> TensorSet:
    """Every tensor declared by the repo's safetensors shard headers (headers only, never weights)."""
    try:
        metadata = HfApi(token=token).get_safetensors_metadata(repo)
    except NotASafetensorsRepoError as e:
        raise StructuralMismatch("no safetensors weights") from e
    except SafetensorsParsingError as e:
        raise StructuralMismatch("unreadable safetensors header") from e
    except Exception as e:
        raise StructuralCheckUnavailable(f"could not read safetensors metadata for {repo}: {e}") from e
    return TensorSet(
        tensors={
            name: TensorSpec(dtype=info.dtype, shape=list(info.shape))
            for file_metadata in metadata.files_metadata.values()
            for name, info in file_metadata.tensors.items()
        }
    )


def fetch_model_config(repo: str, token: str | None = HUGGINGFACE_TOKEN) -> dict:
    try:
        path = hf_hub_download(repo, CONFIG_FILENAME, token=token)
    except Exception as e:
        raise StructuralCheckUnavailable(f"could not download {CONFIG_FILENAME} for {repo}: {e}") from e
    with open(path) as f:
        return json.load(f)


def packed_layout(out_features: int, in_features: int, spec: QuantFixedSpec) -> dict[str, TensorSpec]:
    """Tensors replacing one quantised Linear weight [out, in], keyed by suffix after the module prefix.

    The dense pack-quantized layout (cross-element int32 packing, no padding bits): what the evaluator's
    vLLM 0.31.0 allocates for a compressed-tensors checkpoint (CompressedTensorsWNA16 / W4A8Fp8
    create_weights) and what compressed-tensors >= 0.18 writes. Checkpoints must be pack-quantized; 3-bit is a
    contiguous bitstream. The reader is the contract, not the installed packer: the image's compressed-tensors
    is 0.17.0 (vLLM 0.31.0 pins it), whose own packer pads 3-bit to 10 values per word, but vLLM never calls it
    on load. ops/docker/scripts/check_quant_packed_layout.py asserts this function against vLLM's allocations at
    image build time. Checked against the checkpoint headers in tests/validator/evaluation/fixtures/quant_headers.

    - weight_packed: I32 [out, ceil(in * bits / 32)], packed along the input dimension.
    - weight_scale: BF16 [out, in // group_size].
    - weight_shape: I64 [2], the unpacked [out, in].
    - asymmetric only, weight_zero_point: I32 [ceil(out * bits / 32), in // group_size], packed along dim 0.

    Never present: weight (unpacked), weight_g_idx (act_order is False), input_scale / input_zero_point
    (activations are dynamic per token) and any k/v scale (bf16 KV cache).
    """
    bits = spec.weight_bits
    groups = in_features // spec.group_size
    layout = {
        "weight_packed": TensorSpec(dtype="I32", shape=[out_features, math.ceil(in_features * bits / PACK_WORD_BITS)]),
        "weight_scale": TensorSpec(dtype="BF16", shape=[out_features, groups]),
        "weight_shape": TensorSpec(dtype="I64", shape=[2]),
    }
    if not spec.symmetric:
        layout["weight_zero_point"] = TensorSpec(dtype="I32", shape=[math.ceil(out_features * bits / PACK_WORD_BITS), groups])
    return layout


def _is_embedding(name: str) -> bool:
    return any(marker in name for marker in QUANT_EMBEDDING_MARKERS)


def _is_quantised_weight(name: str, tensor: TensorSpec) -> bool:
    return len(tensor.shape) == 2 and name.endswith(WEIGHT_SUFFIX) and not _is_embedding(name) and name != LM_HEAD_WEIGHT


def _check_base(base: TensorSet, base_config: dict, spec: QuantFixedSpec) -> bool:
    """Our assumptions about the base model; a ValueError is our bug or an unsupported base, never a miner fail.

    Returns whether the base is tied. A tied base (tie_word_embeddings true) may still store a redundant
    lm_head.weight: vLLM's loaders skip it (the head aliases the embedding), so real bases carry one.
    """
    embeddings = [tensor for name, tensor in base.tensors.items() if _is_embedding(name)]
    if len(embeddings) != 1 or embeddings[0].shape[0] != base_config["vocab_size"]:
        raise ValueError(f"expected one embedding with {base_config['vocab_size']} rows, got {embeddings}")
    tied = base_config.get("tie_word_embeddings")
    if tied is None:
        # No key: the default belongs to the model class (True for Gemma, False for Llama), so the base's own
        # tensors decide.
        tied = LM_HEAD_WEIGHT not in base.tensors
    if not tied and LM_HEAD_WEIGHT not in base.tensors:
        raise ValueError(f"{LM_HEAD_WEIGHT} missing although tie_word_embeddings={tied}")
    indivisible = [
        name for name, tensor in base.tensors.items() if _is_quantised_weight(name, tensor) and tensor.shape[1] % spec.group_size
    ]
    if indivisible:
        raise ValueError(f"in_features not a multiple of group size {spec.group_size}: {indivisible[:MAX_REPORTED_NAMES]}")
    return bool(tied)


def expected_fixed_tensor_set(base: TensorSet, base_config: dict, spec: QuantFixedSpec) -> TensorSet:
    """Untied: lm_head.weight is required (BF16, base shape). Tied: a stored base head is optional, as an inert
    BF16 copy at the embedding's shape; a tied base without one expects none."""
    tied = _check_base(base, base_config, spec)
    expected: dict[str, TensorSpec] = {}
    optional: dict[str, TensorSpec] = {}
    for name, tensor in base.tensors.items():
        if tied and name == LM_HEAD_WEIGHT:
            embedding = next(t for n, t in base.tensors.items() if _is_embedding(n))
            optional[name] = TensorSpec(dtype=UNQUANTISED_DTYPE, shape=embedding.shape)
            continue
        if not _is_quantised_weight(name, tensor):
            expected[name] = TensorSpec(dtype=UNQUANTISED_DTYPE, shape=tensor.shape)
            continue
        prefix = name[: -len(WEIGHT_SUFFIX)]
        out_features, in_features = tensor.shape
        for suffix, packed in packed_layout(out_features, in_features, spec).items():
            expected[f"{prefix}.{suffix}"] = packed
    return TensorSet(tensors=expected, optional=optional)


def _listed(names: list[str]) -> str:
    shown = ", ".join(names[:MAX_REPORTED_NAMES])
    more = len(names) - MAX_REPORTED_NAMES
    return f"[{shown}{f', +{more} more' if more > 0 else ''}]"


def _described(tensor: TensorSpec) -> str:
    return f"{tensor.dtype}{tensor.shape}"


def check_fixed_submission(actual: TensorSet, expected: TensorSet) -> str | None:
    """None when {name: (dtype, shape)} matches exactly (optional tensors may be absent); otherwise why, capped."""
    allowed = {**expected.optional, **expected.tensors}
    missing = sorted(expected.tensors.keys() - actual.tensors.keys())
    extra = sorted(actual.tensors.keys() - allowed.keys())
    mismatched = sorted(name for name in allowed.keys() & actual.tensors.keys() if allowed[name] != actual.tensors[name])
    if not (missing or extra or mismatched):
        return None
    parts = []
    if missing:
        parts.append(f"missing {_listed(missing)}")
    if extra:
        parts.append(f"extra {_listed(extra)}")
    if mismatched:
        first = mismatched[0]
        parts.append(f"first mismatch {first} expected {_described(allowed[first])} got {_described(actual.tensors[first])}")
    return f"structural: {', '.join(parts)}"[:STRUCTURAL_FAILURE_MAX_CHARS]


async def expected_tensor_set_for(base_repo: str, spec: QuantFixedSpec, token: str | None = HUGGINGFACE_TOKEN) -> TensorSet:
    """The tensor set every entrant of a task must upload; computed once per task from the input it was given."""
    try:
        base = await asyncio.to_thread(fetch_tensor_set, base_repo, token)
    except StructuralMismatch as e:
        raise ValueError(f"unsupported base model {base_repo}: {e}") from e
    base_config = await asyncio.to_thread(fetch_model_config, base_repo, token)
    return expected_fixed_tensor_set(base, base_config, spec)


async def fixed_submission_failure(repo: str, expected: TensorSet, token: str | None = HUGGINGFACE_TOKEN) -> str | None:
    """Scored-fail reason for a fixed-track repo, or None when it passes. Raises StructuralCheckUnavailable."""
    try:
        actual = await asyncio.to_thread(fetch_tensor_set, repo, token)
    except StructuralMismatch as e:
        return f"structural: {e}"
    return check_fixed_submission(actual, expected)
