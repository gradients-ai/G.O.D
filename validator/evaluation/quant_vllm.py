"""vLLM adapter for the quant scorer. Imported only inside the quant eval image.

The loss math is in quant_loss.py; this module loads a model in vLLM, reads prompt logprobs, and
checks what every linear layer was actually loaded as against the fixed-track spec.

Only the weights and config.json of the loaded repo reach the engine (config.json is held to the input model's
by quant_config_check). Everything else is pinned: the tokenizer is the input model's (the test set was tokenised
with it in quant prep, and prompts go in as token ids, so a miner's tokenizer files or chat template are never
read), the config is parsed in HF format only (a params.json plus consolidated*.safetensors would otherwise switch
vLLM to the Mistral parser and loader, past both config.json and the structural check's shards), weights are
safetensors only, and the repo's generation_config.json supplies no sampling defaults.
"""

import json
import os
import tempfile
from pathlib import Path

import validator.evaluation.constants as vcst


# vLLM reads these at import time, so they are set here rather than trusted to the image.
os.environ.update(vcst.QUANT_VLLM_ENV)

import humming.tune as humming_tune
from pydantic import BaseModel
from vllm import LLM
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.model_executor.layers.linear import LinearBase
from vllm.transformers_utils.config import get_config

from core.models.quant_models import QuantFixedSpec
from core.models.quant_models import QuantFormat
from validator.evaluation.quant_loss import QuantKernelMismatch
from validator.evaluation.quant_loss import assert_expected_kernels


CONFIG_FILENAME = "config.json"
# Offending layers named in a QuantKernelMismatch; the reason is truncated downstream anyway.
_MAX_REPORTED_OFFENDERS = 3


class VllmPromptLogprobEngine:
    def __init__(self, model_path: str, max_prompt_len: int, tokenizer: str):
        """tokenizer is the task's input model, whose tokenizer produced the test set's token ids."""
        # One generated token rides on every prompt, so the context holds max_prompt_len + 1.
        # max_num_batched_tokens >= the longest prompt keeps each prompt to one prefill chunk
        # under V1 chunked prefill.
        max_len = max_prompt_len + 1
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer,
            tokenizer_mode=vcst.QUANT_VLLM_TOKENIZER_MODE,
            config_format=vcst.QUANT_VLLM_CONFIG_FORMAT,
            load_format=vcst.QUANT_VLLM_LOAD_FORMAT,
            generation_config=vcst.QUANT_VLLM_GENERATION_CONFIG,
            dtype="bfloat16",
            max_num_seqs=vcst.QUANT_VLLM_MAX_NUM_SEQS,
            enable_prefix_caching=False,
            max_model_len=max_len,
            max_num_batched_tokens=max_len,
            seed=vcst.QUANT_VLLM_SEED,
            logprobs_mode=vcst.QUANT_VLLM_LOGPROBS_MODE,
            trust_remote_code=False,
        )
        self._sampling_params = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)

    def prompt_logprobs(self, token_ids: list[int]) -> list[float | None]:
        (output,) = self.llm.generate([TokensPrompt(prompt_token_ids=token_ids)], self._sampling_params, use_tqdm=False)
        return [
            None if position is None else position[token_id].logprob
            for token_id, position in zip(token_ids, output.prompt_logprobs)
        ]


def resolved_model_config(config: dict) -> dict:
    """A parsed config.json as vLLM resolves it for the engine (HF format, no remote code), as a plain dict.

    Resolved from a temporary local dir holding only that file, so nothing is fetched and a failure is the file's
    own. This is the view quant_config_check.config_drift_failure compares, so a field spelled differently on disk
    (rope_theta vs rope_parameters, torch_dtype vs dtype) compares by meaning.
    """
    with tempfile.TemporaryDirectory(prefix="quant-config-") as model_dir:
        (Path(model_dir) / CONFIG_FILENAME).write_text(json.dumps(config))
        return get_config(model_dir, trust_remote_code=False, config_format=vcst.QUANT_VLLM_CONFIG_FORMAT).to_dict()


def disable_humming_stream_k() -> None:
    """Make Humming (the W3A16 / W2A16 kernel) deterministic. Call before building any LLM.

    Humming's SM90 heuristics turn stream-K on, and stream-K reduces partial sums with bf16
    atomicAdd, so the same checkpoint scores differently run to run.
    With it off, scores are bit-identical across processes. This
    patches a private function of humming-kernels 0.1.16, so a missing target is a broken
    image (an evaluation error), never the miner's fault.
    """
    heuristics_config = getattr(humming_tune, "_get_heuristics_config", None)
    if heuristics_config is None:
        raise RuntimeError("humming.tune._get_heuristics_config is missing; the stream-K patch targets humming-kernels 0.1.16")

    def without_stream_k(*args, **kwargs):
        # One config dict for a fixed shape_m, else a list of (.., .., config) entries.
        config = heuristics_config(*args, **kwargs)
        for entry in config if isinstance(config, list) else [config]:
            (entry[2] if isinstance(entry, (list, tuple)) else entry)["use_stream_k"] = False
        return config

    # 0.1.16 lru_caches _get_heuristics_config, so this also edits the cached dicts; harmless, as
    # every caller now sees use_stream_k=False. get_heuristics_config looks the name up through
    # module globals at call time, so the patch holds whenever vllm was imported.
    humming_tune._get_heuristics_config = without_stream_k


class QuantLinearLayer(BaseModel):
    """What vLLM loaded one linear layer as. scheme is None when the layer was left unquantised."""

    quant_method: str
    scheme: str | None = None
    kernel: str | None = None
    weight_bits: int | None = None
    group_size: int | None = None
    symmetric: bool | None = None


def _describe_linear(module: LinearBase) -> QuantLinearLayer:
    quant_method = type(module.quant_method).__name__
    scheme = getattr(module, "scheme", None)
    if scheme is None:
        return QuantLinearLayer(quant_method=quant_method)
    # getattr: the scheme class is the checkpoint's choice, so a missing attribute is a mismatch.
    kernel = getattr(scheme, "kernel", None)
    quant_type = getattr(scheme, "quant_type", None)
    return QuantLinearLayer(
        quant_method=quant_method,
        scheme=type(scheme).__name__,
        kernel=None if kernel is None else type(kernel).__name__,
        weight_bits=getattr(quant_type, "size_bits", None),
        group_size=getattr(scheme, "group_size", None),
        symmetric=getattr(scheme, "symmetric", None),
    )


def _linear_layers(model) -> dict[str, QuantLinearLayer]:
    # lm_head is a ParallelLMHead (an embedding, not a LinearBase), so it is never inspected. Nor are
    # FusedMoE experts, so this check covers dense bases only (every launch base is dense).
    return {name: _describe_linear(module) for name, module in model.named_modules() if isinstance(module, LinearBase)}


def collect_linear_layers(llm: LLM) -> dict[str, QuantLinearLayer]:
    # The engine runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so there is one worker.
    (layers,) = llm.apply_model(_linear_layers)
    return layers


def kernel_classes(layers: dict[str, QuantLinearLayer]) -> dict[str, str]:
    """Kernel class per quantised layer, keyed by module name."""
    return {name: layer.kernel for name, layer in layers.items() if layer.kernel is not None}


def _layer_mismatch(layer: QuantLinearLayer, spec: QuantFixedSpec) -> str | None:
    if layer.scheme is None:
        return f"unquantised ({layer.quant_method})"
    activation = vcst.QUANT_SCHEME_ACTIVATIONS.get(layer.scheme)
    loaded = (layer.weight_bits, layer.group_size, layer.symmetric, activation)
    if loaded == (spec.weight_bits, spec.group_size, spec.symmetric, spec.activation):
        return None
    return (
        f"{layer.scheme} W{layer.weight_bits} g{layer.group_size} symmetric={layer.symmetric} "
        f"activation={activation.value if activation else 'unknown'}"
    )


def assert_layers_match(layers: dict[str, QuantLinearLayer], spec: QuantFixedSpec, expected_kernels: frozenset[str]) -> None:
    """Every linear layer must be quantised exactly as spec says and run one of expected_kernels."""
    offenders = [(name, reason) for name, layer in layers.items() if (reason := _layer_mismatch(layer, spec)) is not None]
    if offenders:
        raise QuantKernelMismatch(
            f"{len(offenders)} of {len(layers)} linear layers are not {spec.name}: {offenders[:_MAX_REPORTED_OFFENDERS]}"
        )
    assert_expected_kernels(kernel_classes(layers), expected_kernels)


def checked_kernel_classes(llm: LLM, spec: QuantFixedSpec | None, expected_kernels: frozenset[str] | None) -> dict[str, str]:
    """Kernel class per quantised layer. On the fixed track (spec given) raises QuantKernelMismatch
    unless every linear layer matches spec and expected_kernels; the freeform track only reports."""
    layers = collect_linear_layers(llm)
    if spec is not None:
        assert_layers_match(layers, spec, expected_kernels)
    return kernel_classes(layers)


def expected_kernels_for(spec: QuantFixedSpec) -> frozenset[str]:
    for quant_format in QuantFormat:
        if quant_format.spec() == spec:
            return vcst.QUANT_EXPECTED_KERNEL_CLASSES[quant_format]
    raise ValueError(f"quant spec {spec.name} matches no quant format")
