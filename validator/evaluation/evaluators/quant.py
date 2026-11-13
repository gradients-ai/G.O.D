"""Quant-task evaluator: scores each repo's test loss from vLLM prompt logprobs.

Env: ORIGINAL_MODEL, MODELS (comma list), DATASET or DATASET_URL (tokenised JSONL, see quant_loss),
QUANT_SPEC (QuantFixedSpec JSON, fixed track only), QUANT_FORMAT (QuantFormat value, optional) and
DECODED_MODEL_URLS (freeform only: the decoded bf16 model per repo, scored instead of the HF artifact),
QUANT_REFERENCE_MODEL (the task's bf16 reference) and QUANT_CACHED_REFERENCES (JSON {reference_key: loss}).

A fixed-track checkpoint whose config.json does not describe the spec or, resolved as vLLM resolves it, differs from
the input model's config anywhere outside quantization_config (quant_config_check.config_drift_failure), a model
that fails to load,
a linear layer that is unquantised or loaded in another format or kernel, and a non-finite loss are
scored fails (eval_loss None plus quant_failure), not evaluation errors; so is a decoded model that
fails to download or does not match its index.

Before any model is loaded, the container reads its GPU (quant_gpu.py, in a subprocess) and exits
non-zero unless its device name and SM count match one of QUANT_EVAL_ALLOWED_SKUS (H100 SXM or PCIe):
scoring on another GPU is an evaluation error, never a scored fail. QUANT_EVAL_ALLOW_ANY_GPU=1 (test
boxes only) admits any GPU with a loud warning. Every result records the GPU and SKU it was scored on.
QUANT_REFERENCE_ONLY=1 (set by the dispatcher for a bare bf16 reference measurement) widens the list to
QUANT_REFERENCE_ALLOWED_SKUS (adds 80 GB A100s), and only when MODELS is exactly QUANT_REFERENCE_MODEL with no
QUANT_SPEC and no DECODED_MODEL_URLS; otherwise the container exits non-zero (QuantReferenceOnlyMisuse).

Every quant loss is compared as a multiple of the bf16 reference scored on the same SKU. When this
container's SKU is not in QUANT_CACHED_REFERENCES, it scores QUANT_REFERENCE_MODEL first, on the same GPU
and test set, and writes it under QUANT_REFERENCE_RESULT_KEY next to the repo's result; a cached SKU costs
nothing. A reference that is also one of the repos is scored once.

This module and single_quant.py run inside the quant evaluator image, which carries vLLM but not the
validator stack: they stay free of torch, axolotl/peft (validator.evaluation.common) and fiber
(core.logging, whose pydantic pin conflicts with vLLM's), and log through the standard library. The
per-repo logic here is testable on CPU; single_quant.py wires in the vLLM engine.
"""

import json
import logging
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import TypeAdapter

import core.constants as core_cst
import validator.evaluation.constants as vcst
from core.models.quant_models import QuantEvalGpu
from core.models.quant_models import QuantEvalSku
from core.models.quant_models import QuantFixedSpec
from core.models.quant_models import QuantFormat
from validator.evaluation.errors import evaluation_error_result
from validator.evaluation.errors import is_infra_error
from validator.evaluation.quant_artifacts import DecodedModelCorrupt
from validator.evaluation.quant_artifacts import materialise_decoded_model
from validator.evaluation.quant_artifacts import parse_decoded_model_urls
from validator.evaluation.quant_config_check import config_drift_failure
from validator.evaluation.quant_config_check import quant_config_failure
from validator.evaluation.quant_loss import PromptLogprobEngine
from validator.evaluation.quant_loss import QuantEvalExample
from validator.evaluation.quant_loss import QuantKernelMismatch
from validator.evaluation.quant_loss import QuantScoringError
from validator.evaluation.quant_loss import is_scored_fail
from validator.evaluation.quant_loss import score_examples
from validator.tasks.datasets.constants import CONTAINER_EVAL_RESULTS_PATH


logger = logging.getLogger(__name__)

MODEL_PARAMS_COUNT_KEY = "model_params_count"
_CACHED_REFERENCES = TypeAdapter(dict[str, float])


class QuantEvaluationArgs(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    dataset: str
    original_model: str
    repo: str
    quant_spec: QuantFixedSpec | None = None
    quant_format: QuantFormat | None = None
    # Local decoded model dir (freeform); vLLM loads it instead of repo, results stay keyed by repo.
    model_path: str | None = None
    eval_gpu: QuantEvalGpu | None = None
    # Results-file key the result is saved under; the repo itself unless this is the reference run.
    result_key: str | None = None

    @property
    def saved_as(self) -> str:
        return self.result_key or self.repo


class QuantEvalSuccess(BaseModel):
    is_finetune: bool = True
    eval_loss: float
    per_example_losses: list[float]
    eval_set_fingerprint: str
    kernel_classes: dict[str, str]
    eval_gpu: QuantEvalGpu | None = None


class QuantEvalFailure(BaseModel):
    is_finetune: bool = True
    eval_loss: None = None
    quant_failure: str
    eval_gpu: QuantEvalGpu | None = None


class QuantGpuMismatch(RuntimeError):
    """The container landed on a GPU that is none of the allowed quant eval SKUs."""


class QuantReferenceOnlyMisuse(RuntimeError):
    """QUANT_REFERENCE_ONLY was set on a run that is not a bare bf16 reference measurement."""


def _scored_fail(repo: str, reason: str, eval_gpu: QuantEvalGpu | None = None) -> dict:
    logger.warning(f"Quant scored fail for {repo}: {reason}")
    return QuantEvalFailure(
        quant_failure=reason[: vcst.QUANT_FAILURE_REASON_MAX_CHARS], eval_gpu=eval_gpu
    ).model_dump()


def probe_gpu() -> QuantEvalGpu:
    """The container's GPU, read by quant_gpu.py in a subprocess so this module stays free of torch."""
    probe = subprocess.run(
        ["python", "-m", "validator.evaluation.quant_gpu"], check=True, capture_output=True, text=True
    )
    return QuantEvalGpu.model_validate_json(probe.stdout.strip().splitlines()[-1])


def _allowed_skus_text(skus: tuple[QuantEvalSku, ...]) -> str:
    return ", ".join(f"{sku.label} ({sku.device_name}, {sku.sm_count} SMs)" for sku in skus)


def check_gpu(
    gpu: QuantEvalGpu, allowed: tuple[QuantEvalSku, ...] = vcst.QUANT_EVAL_ALLOWED_SKUS, allow_any: bool = False
) -> QuantEvalGpu:
    """The GPU with its matched SKU label; QuantGpuMismatch when it matches none and allow_any is off.

    Both the device name and the SM count must match, so a MIG slice or a renamed part fails closed.
    """
    for sku in allowed:
        if gpu.name == sku.device_name and gpu.sm_count == sku.sm_count:
            return gpu.model_copy(update={"sku": sku.label, "override": False})
    if not allow_any:
        raise QuantGpuMismatch(
            f"quant evals run on {_allowed_skus_text(allowed)}; this container has {gpu.name} with {gpu.sm_count} SMs"
        )
    logger.warning(
        f"!!! {core_cst.QUANT_EVAL_ALLOW_ANY_GPU_ENV} OVERRIDE: scoring on {gpu.name} with {gpu.sm_count} SMs, "
        f"which is none of {_allowed_skus_text(allowed)}. These losses are not comparable with prod; "
        "never set this on a validator deployment. !!!"
    )
    return gpu.model_copy(update={"sku": None, "override": True})


def load_hub_config(repo: str) -> object:
    """The repo's parsed config.json. ValueError (a scored fail for a submission) when it is over QUANT_CONFIG_MAX_BYTES:
    the size is checked before the file is read, so a huge one never reaches json.loads or the config resolver."""
    path = Path(hf_hub_download(repo, "config.json", token=os.environ.get("HUGGINGFACE_TOKEN")))
    size = path.stat().st_size
    if size > vcst.QUANT_CONFIG_MAX_BYTES:
        raise ValueError(f"config.json is {size} bytes, over the {vcst.QUANT_CONFIG_MAX_BYTES} byte cap")
    return json.loads(path.read_text())


class FixedConfigCheck(BaseModel):
    """What the fixed-track config check reads besides the submission: the input model and how to resolve a config.

    resolve takes a parsed config.json and returns it as vLLM resolves it, as a plain dict, without touching the
    network (quant_vllm.resolved_model_config in the quant image).
    """

    base_model: str
    resolve: Callable[[dict], dict]


def _config_failure(
    repo: str, spec: QuantFixedSpec, config_loader: Callable[[str], object], check: FixedConfigCheck
) -> str | None:
    try:
        config = config_loader(repo)
    except LocalEntryNotFoundError:
        # The hub was unreachable and nothing is cached: an outage, not the miner's fault.
        raise
    except (OSError, ValueError) as e:
        if is_infra_error(e):
            # A timeout, a transport drop, a 5xx or a 429 reading the file says nothing about the submission.
            raise
        return f"config.json unreadable: {e}"
    reason = quant_config_failure(config, spec)
    if reason is not None:
        return f"quantization_config is not {spec.name}: {reason}"
    # The input model is ours: failing to read or resolve its config is an evaluation error, so nothing is caught.
    # It resolves first with the same function, so a broken resolver fails here rather than scoring everyone.
    base = check.resolve(config_loader(check.base_model))
    try:
        submitted = check.resolve(config)
    except Exception as e:  # resolving is local, so any failure is the submitted file's content
        return f"config.json does not load: {e}"
    reason = config_drift_failure(submitted, base)
    return None if reason is None else f"config differs from the input model: {reason}"


def evaluate_repo(
    repo: str,
    examples: list[QuantEvalExample],
    engine_factory: Callable[[str], PromptLogprobEngine],
    layer_check: Callable[[PromptLogprobEngine], dict[str, str]],
    quant_spec: QuantFixedSpec | None = None,
    config_loader: Callable[[str], object] = load_hub_config,
    config_check: FixedConfigCheck | None = None,
    model_path: str | None = None,
    eval_gpu: QuantEvalGpu | None = None,
) -> dict:
    """Score one repo, loading model_path when given; eval_gpu is recorded in the result.

    layer_check returns the kernel class per quantised layer and raises QuantKernelMismatch when the
    loaded layers break the fixed-track spec. quant_spec is None on the freeform track, which skips
    the config checks (the decoded model is loaded under the input model's own config); on the fixed
    track config_check is required.
    """
    if quant_spec is not None:
        if config_check is None:
            raise ValueError("the fixed-track config check needs the input model (config_check)")
        try:
            reason = _config_failure(repo, quant_spec, config_loader, config_check)
        except LocalEntryNotFoundError:
            raise
        except Exception as e:
            if is_infra_error(e):
                logger.error(f"Could not read the config of {repo}: {e}", exc_info=True)
                return evaluation_error_result(e)
            raise
        if reason is not None:
            return _scored_fail(repo, reason, eval_gpu)

    try:
        engine = engine_factory(model_path or repo)
    except Exception as e:
        logger.error(f"vLLM failed to load {repo} from {model_path or repo}: {e}", exc_info=True)
        if is_infra_error(e):
            # A hub drop, a failed tokenizer fetch or a transport timeout says nothing about the model: re-queue it.
            return evaluation_error_result(e)
        return _scored_fail(repo, f"model failed to load in vLLM: {e}", eval_gpu)

    try:
        kernel_classes = layer_check(engine)
        logger.info(
            f"{repo} (loaded from {model_path or repo}): {len(kernel_classes)} quantised layers, "
            f"kernels {sorted(set(kernel_classes.values()))}"
        )
        score = score_examples(engine, examples)
    except (QuantKernelMismatch, QuantScoringError) as e:
        return _scored_fail(repo, str(e), eval_gpu)

    reason = is_scored_fail(score)
    if reason is not None:
        return _scored_fail(repo, reason, eval_gpu)
    logger.info(f"{repo}: eval_loss {score.eval_loss:.6f} over {score.n_examples} examples")
    return QuantEvalSuccess(
        eval_loss=score.eval_loss,
        per_example_losses=score.per_example_losses,
        eval_set_fingerprint=score.eval_set_fingerprint,
        kernel_classes=kernel_classes,
        eval_gpu=eval_gpu,
    ).model_dump()


def _download_dataset(dataset_url: str) -> str:
    dataset = os.path.join(tempfile.gettempdir(), os.path.basename(dataset_url.split("?")[0]) or "dataset.jsonl")
    urllib.request.urlretrieve(dataset_url, dataset)
    logger.info(f"Downloaded dataset from DATASET_URL to {dataset}")
    return dataset


def load_results() -> dict:
    """The shared results file every repo's subprocess adds to: repo -> result, plus the base size."""
    results_path = Path(CONTAINER_EVAL_RESULTS_PATH)
    return json.loads(results_path.read_text()) if results_path.exists() else {}


def save_results(results: dict) -> None:
    results_path = Path(CONTAINER_EVAL_RESULTS_PATH)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))


def record_base_model_params(original_model: str) -> None:
    """Record the base model's parameter count from its hub safetensors metadata, without loading it."""
    results = load_results()
    if MODEL_PARAMS_COUNT_KEY in results:
        return
    safetensors = HfApi().model_info(original_model).safetensors
    if safetensors is None:
        logger.warning(f"{original_model} has no safetensors metadata; model_params_count left unset")
        return
    results[MODEL_PARAMS_COUNT_KEY] = safetensors.total
    save_results(results)


def _record_scored_fail(repo: str, reason: str, eval_gpu: QuantEvalGpu) -> None:
    """Write a scored fail into the shared results file (same shape single_quant saves)."""
    results = load_results()
    results[repo] = _scored_fail(repo, reason, eval_gpu)
    save_results(results)


def _record_evaluation_error(repo: str, error: BaseException) -> None:
    """Write an evaluation error (not the miner's fault; the validator re-queues it) into the shared results file."""
    results = load_results()
    results[repo] = evaluation_error_result(error)
    save_results(results)


def _materialise_for_repo(
    original_model: str, repo: str, urls: dict[str, str], eval_gpu: QuantEvalGpu
) -> str | None:
    """Local decoded model dir for repo, or None after recording why it could not be built.

    Shards that do not match the decoded model's index are the artifact's fault: a scored fail. Failing to download
    them from our object storage (an expired or unreachable presigned url) is ours: an evaluation error.
    """
    dest = Path(tempfile.mkdtemp(prefix="decoded-"))
    try:
        return str(materialise_decoded_model(original_model, urls, dest, os.environ.get("HUGGINGFACE_TOKEN")))
    except DecodedModelCorrupt as e:
        _record_scored_fail(repo, f"decoded model: {e}", eval_gpu)
        return None
    except urllib.error.URLError as e:
        _record_evaluation_error(repo, e)
        return None


def _allow_any_gpu() -> bool:
    """Only the literal "1" turns the override on, so a stray or empty value fails closed."""
    return os.environ.get(core_cst.QUANT_EVAL_ALLOW_ANY_GPU_ENV, "").strip() == "1"


def _reference_only() -> bool:
    """Only the literal "1" turns reference-only on, so a stray or empty value keeps the entry SKU list."""
    return os.environ.get(core_cst.QUANT_REFERENCE_ONLY_ENV, "").strip() == "1"


def allowed_skus_for_run(
    reference_only: bool,
    repos: list[str],
    reference_model: str | None,
    quant_spec: QuantFixedSpec | None,
    decoded_model_urls: dict,
) -> tuple[QuantEvalSku, ...]:
    """The SKUs this container may run on: entries always get QUANT_EVAL_ALLOWED_SKUS.

    A reference-only run (one bf16 model that is its own reference, nothing quantised or decoded) may also use
    QUANT_REFERENCE_ALLOWED_SKUS; reference-only on anything else raises, so the flag can never move an entry's score
    onto another GPU class.
    """
    if not reference_only:
        return vcst.QUANT_EVAL_ALLOWED_SKUS
    if reference_model is None or repos != [reference_model] or quant_spec is not None or decoded_model_urls:
        raise QuantReferenceOnlyMisuse(
            f"{core_cst.QUANT_REFERENCE_ONLY_ENV} is only for a bare bf16 reference: got MODELS={repos}, "
            f"reference={reference_model}, quant_spec={'set' if quant_spec else 'none'}, "
            f"decoded={'set' if decoded_model_urls else 'none'}"
        )
    return vcst.QUANT_REFERENCE_ALLOWED_SKUS


def _checked_gpu(allowed: tuple[QuantEvalSku, ...] = vcst.QUANT_EVAL_ALLOWED_SKUS) -> QuantEvalGpu:
    """Read the GPU and raise QuantGpuMismatch unless it is one of allowed (or the override is on)."""
    gpu = check_gpu(probe_gpu(), allowed=allowed, allow_any=_allow_any_gpu())
    logger.info(f"Quant eval GPU: {gpu.name} with {gpu.sm_count} SMs (SKU {gpu.sku or 'unlisted, override'})")
    return gpu


def parse_cached_references(env_value: str | None) -> dict[str, float]:
    """reference_key -> cached bf16 reference loss; empty when the env var is unset."""
    return _CACHED_REFERENCES.validate_json(env_value) if env_value else {}


def _run_single(evaluation_args: QuantEvaluationArgs) -> None:
    try:
        # One subprocess per model so each vLLM engine releases its GPU memory on exit.
        subprocess.run(
            ["python", "-m", "validator.evaluation.evaluators.single_quant", evaluation_args.model_dump_json()],
            check=True,
        )
        logger.info(f"Subprocess completed for {evaluation_args.repo}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running subprocess for {evaluation_args.repo}: {e}")


def _reference_needed(reference_model: str | None, cached: dict[str, float], eval_gpu: QuantEvalGpu) -> bool:
    if not reference_model:
        return False
    if eval_gpu.reference_key in cached:
        logger.info(f"bf16 reference for {eval_gpu.reference_key} is cached ({cached[eval_gpu.reference_key]:.6f})")
        return False
    return True


def _copy_reference_result(reference_model: str) -> None:
    """The reference was one of the repos: its own result is the reference, never scored twice."""
    results = load_results()
    if reference_model in results:
        results[vcst.QUANT_REFERENCE_RESULT_KEY] = results[reference_model]
        save_results(results)


def main():
    logger.info("=== QUANT EVALUATION SCRIPT STARTING ===")
    dataset = os.environ.get("DATASET")
    dataset_url = os.environ.get("DATASET_URL")
    original_model = os.environ.get("ORIGINAL_MODEL")
    models_str = os.environ.get("MODELS", "")
    quant_spec_json = os.environ.get(core_cst.QUANT_SPEC_ENV)
    quant_format_str = os.environ.get(core_cst.QUANT_FORMAT_ENV)

    if not dataset and dataset_url:
        dataset = _download_dataset(dataset_url)

    repos = [m.strip() for m in models_str.split(",") if m.strip()]
    if not all([dataset, original_model, repos]):
        logger.error("Missing required environment variables.")
        exit(1)

    quant_spec = QuantFixedSpec.model_validate_json(quant_spec_json) if quant_spec_json else None
    quant_format = QuantFormat(quant_format_str) if quant_format_str else None
    decoded_model_urls = parse_decoded_model_urls(os.environ.get(core_cst.DECODED_MODEL_URLS_ENV))
    reference_model = os.environ.get(core_cst.QUANT_REFERENCE_MODEL_ENV, "").strip() or None
    cached_references = parse_cached_references(os.environ.get(core_cst.QUANT_CACHED_REFERENCES_ENV))
    # Uncaught on a mismatch or a misused reference-only flag: the container exits non-zero before any model is loaded.
    allowed = allowed_skus_for_run(_reference_only(), repos, reference_model, quant_spec, decoded_model_urls)
    eval_gpu = _checked_gpu(allowed)

    measure_reference = _reference_needed(reference_model, cached_references, eval_gpu)
    if measure_reference and reference_model not in repos:
        logger.info(f"Scoring the bf16 reference {reference_model} on {eval_gpu.reference_key} before the submissions")
        # bf16 references are never config- or kernel-checked, so no spec or format.
        _run_single(
            QuantEvaluationArgs(
                dataset=dataset,
                original_model=original_model,
                repo=reference_model,
                eval_gpu=eval_gpu,
                result_key=vcst.QUANT_REFERENCE_RESULT_KEY,
            )
        )

    for repo in repos:
        model_path = None
        if repo in decoded_model_urls:
            model_path = _materialise_for_repo(original_model, repo, decoded_model_urls[repo], eval_gpu)
            if model_path is None:
                continue
        evaluation_args = QuantEvaluationArgs(
            dataset=dataset,
            original_model=original_model,
            repo=repo,
            quant_spec=quant_spec,
            quant_format=quant_format,
            model_path=model_path,
            eval_gpu=eval_gpu,
        )
        _run_single(evaluation_args)

    if measure_reference and reference_model in repos:
        _copy_reference_result(reference_model)

    logger.info("=== QUANT EVALUATION SCRIPT COMPLETED ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
