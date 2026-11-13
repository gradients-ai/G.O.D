"""Quant evaluator image build check: vLLM allocates exactly the packed tensors quant_structure expects.

The structural pre-check (validator/evaluation/quant_structure.packed_layout) accepts the dense pack-quantized
layout, ceil(in * bits / 32) int32 words per row. vLLM 0.31.0 allocates its compressed-tensors parameters with
its own shape arithmetic and never calls the installed compressed-tensors packer on load, so the packer version
does not decide what loads; vLLM's allocation does, and this pins it. compressed-tensors itself is pinned to
whatever vLLM pins (0.17.0 for vLLM 0.31.0): forcing a newer one would run vLLM against an untested release.

CPU only, no process group: kernel selection (the one GPU-dependent step of create_weights) and the tensor-parallel
rank lookup are stubbed out; the shapes come from vLLM's own create_weights.
"""

from importlib.metadata import requires
from importlib.metadata import version

import torch
import vllm.model_executor.parameter as vllm_parameter
from packaging.requirements import Requirement
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w4a8_fp8
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_wNa16

from core.models.quant_models import QuantActivationFormat
from core.models.quant_models import QuantFixedSpec
from core.models.quant_models import QuantFormat
from validator.evaluation.quant_structure import TensorSpec
from validator.evaluation.quant_structure import packed_layout


COMPRESSED_TENSORS = "compressed-tensors"
# (out_features, in_features): Qwen2.5-0.5B's MLP shapes, plus an out dim whose 3-bit zero points need a partial
# word (1000 * 3 / 32 = 93.75), where the dense and the padded (10 per word) layouts disagree. 384 and 4864 divide
# by 64, so the g64 format is covered too (1000 * 2 / 32 = 62.5 is a partial word for its zero points).
SHAPES = ((896, 4864), (4864, 896), (1000, 384))
SAFETENSORS_DTYPES = {torch.int32: "I32", torch.int64: "I64", torch.bfloat16: "BF16"}


class _NoKernel:
    def __init__(self, *args, **kwargs):
        pass


def check_compressed_tensors_pin() -> str:
    """The installed compressed-tensors is exactly the release vLLM pins."""
    pins = [Requirement(r) for r in requires("vllm") or [] if Requirement(r).name == COMPRESSED_TENSORS]
    assert len(pins) == 1, f"vllm declares {len(pins)} {COMPRESSED_TENSORS} requirements"
    installed = version(COMPRESSED_TENSORS)
    assert pins[0].specifier.contains(installed), f"{COMPRESSED_TENSORS} {installed} is not vllm's {pins[0]}"
    assert str(pins[0].specifier).startswith("=="), f"vllm no longer pins {COMPRESSED_TENSORS} exactly: {pins[0]}"
    return installed


def _scheme(spec: QuantFixedSpec):
    if spec.activation == QuantActivationFormat.BF16:
        module, cls = compressed_tensors_wNa16, compressed_tensors_wNa16.CompressedTensorsWNA16
    elif spec.activation == QuantActivationFormat.FP8:
        module, cls = compressed_tensors_w4a8_fp8, compressed_tensors_w4a8_fp8.CompressedTensorsW4A8Fp8
    else:
        raise AssertionError(f"no scheme mapped for {spec.name}")
    module.choose_mp_linear_kernel = lambda _config: _NoKernel
    return cls(strategy="group", num_bits=spec.weight_bits, group_size=spec.group_size, symmetric=spec.symmetric)


def _single_rank() -> None:
    """One unsharded rank: the parameters read their tp rank/size at construction."""
    vllm_parameter.get_tensor_model_parallel_rank = lambda: 0
    vllm_parameter.get_tensor_model_parallel_world_size = lambda: 1


def vllm_layout(spec: QuantFixedSpec, out_features: int, in_features: int) -> dict[str, TensorSpec]:
    _single_rank()
    layer = torch.nn.Module()
    _scheme(spec).create_weights(
        layer,
        output_size=out_features,
        input_size=in_features,
        output_partition_sizes=[out_features],
        input_size_per_partition=in_features,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *_args, **_kwargs: None,
    )
    return {
        name: TensorSpec(dtype=SAFETENSORS_DTYPES[param.dtype], shape=list(param.shape))
        for name, param in layer.named_parameters()
    }


def check_packed_layouts() -> None:
    for quant_format in tuple(QuantFormat):
        spec = quant_format.spec()
        for out_features, in_features in SHAPES:
            expected = packed_layout(out_features, in_features, spec)
            actual = vllm_layout(spec, out_features, in_features)
            assert actual == expected, (
                f"{spec.name} [{out_features}, {in_features}]: vLLM allocates {actual}, quant_structure expects {expected}"
            )


if __name__ == "__main__":
    ct_version = check_compressed_tensors_pin()
    check_packed_layouts()
    print(
        f"packed layout ok: vllm {version('vllm')}, {COMPRESSED_TENSORS} {ct_version}, "
        f"{len(QuantFormat)} formats (every QuantFormat, old tasks included)"
    )
