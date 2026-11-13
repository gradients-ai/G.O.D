import asyncio
import json
import random
from urllib.parse import urlparse
from uuid import UUID

import core.constants.docker as docker_cst
import validator.evaluation.constants as vcst
from core.logging import get_logger
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import GrpoDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.image_models import ImageModelType
from core.models.image_models import ImageTrainingMode
from core.models.payload_models import DockerEvaluationResults
from core.models.payload_models import EvaluationResultText
from core.models.quant_models import QuantFixedSpec
from core.models.quant_models import QuantFormat
from core.models.quant_models import QuantReferenceScore
from validator.db.database import PSQLDB
from validator.evaluation.basilica_deployments import create_basilica_eval_runner_source
from validator.evaluation.db_utils import load_eval_pair_state_for_models
from validator.evaluation.errors import EvaluationInfraError
from validator.evaluation.errors import RepoEvaluationFailed
from validator.evaluation.errors import as_infra_error
from validator.evaluation.quant_artifacts import decoded_model_prefix
from validator.evaluation.remote import run_remote_eval_repos
from validator.evaluation.result_processing import is_evaluation_error_result
from validator.evaluation.result_processing import normalize_rewards_and_compute_loss
from validator.evaluation.result_processing import process_evaluation_results
from validator.infrastructure.minio_client import async_minio_client
from validator.tasks.datasets.constants import CONTAINER_EVAL_RESULTS_PATH


logger = get_logger(__name__)


def _effective_remote_file_format(dataset: str, file_format: FileFormat) -> FileFormat:
    """Map an S3 transport URL to the downloaded file's parser format."""
    if file_format != FileFormat.S3:
        return file_format
    path = urlparse(dataset).path.lower()
    if path.endswith((".json", ".jsonl")):
        return FileFormat.JSON
    if path.endswith(".csv"):
        return FileFormat.CSV
    return file_format


async def _db_read_with_retry(coro_factory, op_name: str):
    last_exc = None
    for attempt in range(1, vcst.EVAL_DB_RETRY_ATTEMPTS + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            delay = vcst.EVAL_DB_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            jitter = random.uniform(0.0, 0.3)
            if attempt < vcst.EVAL_DB_RETRY_ATTEMPTS:
                logger.warning(
                    f"DB read op '{op_name}' failed attempt {attempt}/{vcst.EVAL_DB_RETRY_ATTEMPTS}: {exc}; "
                    f"retrying in {delay + jitter:.2f}s"
                )
                await asyncio.sleep(delay + jitter)
            else:
                logger.error(f"DB read op '{op_name}' failed after {vcst.EVAL_DB_RETRY_ATTEMPTS} attempts: {exc}")
    raise last_exc


def _collect_repo_evaluation_results(models: list[str], repo_results: dict[str, dict | str]) -> dict[str, dict | str | int]:
    evaluation_results: dict[str, dict | str | int] = {}
    model_params_count = 0

    for repo in models:
        raw_result = repo_results.get(repo)
        if not isinstance(raw_result, dict):
            evaluation_results[repo] = str(raw_result)
            continue

        if raw_result.get("model_params_count") and model_params_count == 0:
            model_params_count = raw_result["model_params_count"]

        if repo in raw_result:
            evaluation_results[repo] = raw_result[repo]
            continue

        candidate_keys = [key for key in raw_result.keys() if key != "model_params_count"]
        if len(candidate_keys) == 1:
            evaluation_results[repo] = raw_result[candidate_keys[0]]
        else:
            evaluation_results[repo] = f"Evaluation failed: missing result key for repo {repo}"

    if model_params_count:
        evaluation_results["model_params_count"] = model_params_count

    return evaluation_results


def _collect_quant_repo_evaluation_results(
    models: list[str], repo_results: dict[str, dict | str | Exception]
) -> dict[str, dict | str | int | Exception]:
    """Each repo's entry from its deployment's results file.

    A deployment-level failure (no results file: an exception or a failure string from the backend) and a results
    file without the repo's entry are evaluation errors, never the repo's verdict: EvaluationInfraError. A per-repo
    string inside the results file is the evaluator's verdict and is kept for process_evaluation_results.
    """
    evaluation_results: dict[str, dict | str | int | Exception] = {}
    model_params_count = 0

    for repo in models:
        raw_result = repo_results.get(repo)
        if isinstance(raw_result, Exception):
            evaluation_results[repo] = as_infra_error(raw_result, f"Deployment for {repo} failed")
            continue
        if not isinstance(raw_result, dict):
            evaluation_results[repo] = EvaluationInfraError(f"Deployment for {repo} returned no results: {raw_result}")
            continue

        if raw_result.get("model_params_count") and model_params_count == 0:
            model_params_count = raw_result["model_params_count"]

        if repo in raw_result:
            evaluation_results[repo] = raw_result[repo]
            continue

        candidate_keys = [key for key in raw_result.keys() if key != "model_params_count"]
        if len(candidate_keys) == 1:
            evaluation_results[repo] = raw_result[candidate_keys[0]]
        else:
            evaluation_results[repo] = EvaluationInfraError(
                f"Evaluation results have no entry for {repo} (the evaluator subprocess died before writing it)"
            )

    if model_params_count:
        evaluation_results["model_params_count"] = model_params_count

    return evaluation_results


async def run_evaluation_basilica_text(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType,
    file_format: FileFormat,
    num_gpus: int,
    eval_seed: int | None = None,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
    local_logging: bool | None = False,
    use_kl: bool = False,
    kl_coef: float | None = None,
    continuous_sft_tokenizer_repo: str | None = None,
    emit_per_example_losses: bool = False,
) -> DockerEvaluationResults:
    deployment_ids_by_repo = {}
    db_deployment_ids_by_repo, repo_to_hotkey = await _db_read_with_retry(
        lambda: load_eval_pair_state_for_models(task_id, psql_db, models),
        "load_eval_pair_state_for_models",
    )
    for repo, dep_info in db_deployment_ids_by_repo.items():
        deployment_ids_by_repo.setdefault(repo, dep_info)
    task_type = type(dataset_type).__name__
    basilica_image = docker_cst.VALIDATOR_DOCKER_IMAGE
    if isinstance(dataset_type, (InstructTextDatasetType, ChatTemplateDatasetType)):
        command = ["python", "-m", "validator.evaluation.evaluators.instruct_text"]
    elif isinstance(dataset_type, DpoDatasetType):
        command = ["python", "-m", "validator.evaluation.evaluators.dpo"]
    elif isinstance(dataset_type, GrpoDatasetType):
        return await run_evaluation_basilica_grpo(
            dataset, models, original_model, dataset_type, file_format, num_gpus,
            task_id=task_id,
            psql_db=psql_db,
            deployment_ids_by_repo=deployment_ids_by_repo,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")
    if not dataset.startswith("http://") and not dataset.startswith("https://"):
        raise ValueError(
            "Basilica text eval expects dataset to be an S3/HTTP URL. "
            "Use validator.evaluation.local_evaluation.run_evaluation_docker_text for local file paths."
        )
    dataset_type_str = dataset_type.model_dump_json()
    source = create_basilica_eval_runner_source(command, CONTAINER_EVAL_RESULTS_PATH)

    effective_file_format = _effective_remote_file_format(dataset, file_format)
    base_env = {
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": effective_file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        **vcst.HF_CONTAINER_ENV,
    }
    if use_kl:
        base_env[docker_cst.USE_KL_ENV] = "1"
        if kl_coef is not None:
            base_env[docker_cst.KL_COEF_ENV] = str(kl_coef)
    if continuous_sft_tokenizer_repo:
        base_env[docker_cst.CONTINUOUS_SFT_TOKENIZER_REPO_ENV] = continuous_sft_tokenizer_repo
    if emit_per_example_losses:
        base_env[docker_cst.EMIT_PER_EXAMPLE_LOSSES_ENV] = "1"

    logger.debug(f"Running Basilica {task_type} evaluation (per-repo deployments) for models: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_env)
        repo_env["MODELS"] = repo
        repo_env["DATASET_URL"] = dataset
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await run_remote_eval_repos(
        repos=models,
        model_name=original_model,
        task_type=task_type,
        image=basilica_image,
        source=source,
        build_env_for_repo=build_env_for_repo,
        gpu_count=max(1, num_gpus),
        gpu_models=vcst.BASILICA_GPU_MODELS,
        min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
        storage=False,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo=deployment_ids_str,
        local_logging=local_logging,
        persist_deployment_ids=True,
        reserve_deployment_id=True,
    )

    evaluation_results = _collect_repo_evaluation_results(models, repo_results)
    return process_evaluation_results(evaluation_results, is_image=False)


async def run_evaluation_basilica_quant(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: InstructTextDatasetType | ChatTemplateDatasetType,
    file_format: FileFormat,
    num_gpus: int,
    quant_spec: QuantFixedSpec | None,
    quant_format: QuantFormat | None,
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    local_logging: bool | None = False,
    decoded_model_urls: dict[str, dict[str, str]] | None = None,
    reference_models: dict[str, str] | None = None,
    cached_references: dict[str, dict[str, float]] | None = None,
    reference_only: bool = False,
) -> DockerEvaluationResults:
    """Score quant submissions through vLLM on an allowed H100 SKU. quant_spec is None on the freeform track.

    reference_only (the bf16 reference pre-pass, where each model is its own reference and nothing is quantised or
    decoded) also allows the A100 SKUs of QUANT_REFERENCE_ALLOWED_SKUS; the evaluator refuses it on any other run.

    decoded_model_urls (freeform only) maps repo -> {filename: presigned url} of its decoded model;
    each deployment's env carries only its own repo's urls.

    reference_models maps repo -> the bf16 reference model its loss is normalised by (its own model on an
    own-model task, the common one otherwise) and cached_references repo -> {reference_key: loss} already
    measured for that reference: a deployment whose SKU is missing scores its reference too, and it comes back
    as that repo's result.quant_reference.
    """
    deployment_ids_by_repo, repo_to_hotkey = await _db_read_with_retry(
        lambda: load_eval_pair_state_for_models(task_id, psql_db, models),
        "load_eval_pair_state_for_models",
    )
    command = ["python", "-m", "validator.evaluation.evaluators.quant"]
    base_env = {
        "ORIGINAL_MODEL": original_model,
        "DATASET_URL": dataset,
        # DATASET_TYPE and FILE_FORMAT mirror the text eval env; the quant evaluator reads only the
        # tokenised JSONL today.
        "DATASET_TYPE": dataset_type.model_dump_json(),
        "FILE_FORMAT": _effective_remote_file_format(dataset, file_format).value,
        # No GPU override: the evaluator accepts QUANT_EVAL_ALLOWED_SKUS (QUANT_REFERENCE_ALLOWED_SKUS on a bare reference).
        **vcst.HF_CONTAINER_ENV,
    }
    if reference_only:
        base_env[docker_cst.QUANT_REFERENCE_ONLY_ENV] = "1"
    if quant_spec is not None:
        base_env[docker_cst.QUANT_SPEC_ENV] = quant_spec.model_dump_json()
    if quant_format is not None:
        base_env[docker_cst.QUANT_FORMAT_ENV] = quant_format.value
    reference_models = reference_models or {}
    cached_references = cached_references or {}

    def build_env_for_repo(repo: str) -> dict[str, str]:
        env = {**base_env, "MODELS": repo}
        if repo in reference_models:
            env["ORIGINAL_MODEL"] = reference_models[repo]
            env[docker_cst.QUANT_REFERENCE_MODEL_ENV] = reference_models[repo]
            env[docker_cst.QUANT_CACHED_REFERENCES_ENV] = json.dumps(cached_references.get(repo, {}))
        if decoded_model_urls and repo in decoded_model_urls:
            env[docker_cst.DECODED_MODEL_URLS_ENV] = json.dumps({repo: decoded_model_urls[repo]})
        return env

    logger.debug(f"Running Basilica quant evaluation (per-repo deployments) for models: {models}")
    repo_results = await run_remote_eval_repos(
        repos=models,
        model_name=original_model,
        task_type=type(dataset_type).__name__,
        image=docker_cst.VALIDATOR_DOCKER_IMAGE_QUANT,
        source=create_basilica_eval_runner_source(command, CONTAINER_EVAL_RESULTS_PATH),
        build_env_for_repo=build_env_for_repo,
        gpu_count=max(1, num_gpus),
        gpu_models=vcst.QUANT_REFERENCE_GPU_MODELS if reference_only else vcst.QUANT_EVAL_GPU_MODELS,
        runpod_instance_types=(
            vcst.QUANT_REFERENCE_RUNPOD_INSTANCE_TYPES if reference_only else vcst.QUANT_EVAL_RUNPOD_INSTANCE_TYPES
        ),
        min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
        storage=False,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo={r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)},
        local_logging=local_logging,
        persist_deployment_ids=True,
        reserve_deployment_id=True,
        requeue_infra_errors=True,
    )

    references = {repo: _pop_quant_reference(repo, raw, reference_models.get(repo)) for repo, raw in repo_results.items()}
    evaluation_results = _collect_quant_repo_evaluation_results(models, repo_results)
    processed = process_evaluation_results(evaluation_results, is_image=False)
    for repo, result in processed.results.items():
        # The quant evaluator turns every fault of the submission into a quant_failure result, so an error string for a
        # repo is an exception that escaped scoring (hub unreachable, engine setup): an evaluation error, not a scored fail.
        if isinstance(result, RepoEvaluationFailed):
            processed.results[repo] = EvaluationInfraError(f"Quant evaluator errored on {repo}: {result}")
    _log_quant_eval_gpus(task_id, processed)
    for repo, reference in references.items():
        result = processed.results.get(repo)
        if reference is not None and isinstance(result, EvaluationResultText):
            processed.results[repo] = result.model_copy(update={"quant_reference": reference})
    return processed


def _pop_quant_reference(repo: str, raw_result: dict | str, reference_model: str | None) -> QuantReferenceScore | None:
    """Take the same-deployment bf16 reference out of a repo's raw results; None when none was scored.

    A reference that did not score is logged, never fatal: the submission's own result stands, and finalize
    re-evaluates a row whose SKU still has no reference.
    """
    if not isinstance(raw_result, dict) or vcst.QUANT_REFERENCE_RESULT_KEY not in raw_result:
        return None
    raw_reference = raw_result.pop(vcst.QUANT_REFERENCE_RESULT_KEY)
    if reference_model is None:
        return None
    if not isinstance(raw_reference, dict) or is_evaluation_error_result(raw_reference):
        logger.error(f"bf16 reference {reference_model} errored in the deployment for {repo}: {raw_reference}")
        return None
    try:
        reference = EvaluationResultText.model_validate(raw_reference)
    except ValueError as e:
        logger.error(f"bf16 reference {reference_model} returned an unreadable result for {repo}: {e}")
        return None
    if reference.quant_failure or not reference.per_example_losses or not reference.eval_set_fingerprint:
        reason = reference.quant_failure or "no per-example losses or fingerprint"
        logger.error(f"bf16 reference {reference_model} did not score in the deployment for {repo}: {reason}")
        return None
    if reference.eval_gpu is None:
        logger.error(f"bf16 reference {reference_model} for {repo} carries no eval_gpu; it cannot be keyed by SKU")
        return None
    return QuantReferenceScore(
        model=reference_model,
        eval_loss=reference.eval_loss,
        per_example_losses=reference.per_example_losses,
        eval_set_fingerprint=reference.eval_set_fingerprint,
        eval_gpu=reference.eval_gpu,
    )


def _log_quant_eval_gpus(task_id: UUID | None, processed: DockerEvaluationResults) -> None:
    """One line per repo naming the SKU it was scored on; an override or a missing SKU is an error line."""
    for repo, result in processed.results.items():
        if not isinstance(result, EvaluationResultText) or result.eval_gpu is None:
            continue
        gpu = result.eval_gpu
        message = f"Quant eval of {repo} for task {task_id} ran on {gpu.name} ({gpu.sm_count} SMs, SKU {gpu.sku})"
        if gpu.override or gpu.sku is None:
            logger.error(f"{message}: GPU OVERRIDE, not an allowed quant eval SKU; this result is not comparable")
        else:
            logger.info(message)


async def _decoded_model_file_urls(bucket: str, prefix: str) -> dict[str, str] | None:
    """{filename: presigned url} under prefix, or None while its index is absent (upload incomplete)."""
    listing = await async_minio_client.list_objects(bucket, prefix=f"{prefix}/", recursive=False)
    names = [obj.object_name for obj in await asyncio.to_thread(list, listing)]
    if f"{prefix}/{docker_cst.DECODED_MODEL_INDEX_NAME}" not in names:
        return None
    return {name.removeprefix(f"{prefix}/"): await async_minio_client.get_presigned_url(bucket, name) for name in names}


async def decoded_model_urls_for(task_id: UUID, hotkey_to_repo: dict[str, str], bucket: str) -> dict[str, dict[str, str]]:
    """repo -> decoded-model file urls for every hotkey whose decode job finished uploading."""
    urls_by_repo = {}
    for hotkey, repo in hotkey_to_repo.items():
        file_urls = await _decoded_model_file_urls(bucket, decoded_model_prefix(task_id, hotkey))
        if file_urls is None:
            logger.warning(f"No decoded model index for {hotkey} on task {task_id}; {repo} will not be scored")
            continue
        urls_by_repo[repo] = file_urls
    return urls_by_repo


async def run_evaluation_basilica_grpo(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: GrpoDatasetType,
    file_format: FileFormat,
    num_gpus: int,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
    deployment_ids_by_repo: dict[str, str | dict[str, str]] | None = None,
) -> DockerEvaluationResults:
    deployment_ids_by_repo = deployment_ids_by_repo or {}
    db_deployment_ids_by_repo, repo_to_hotkey = await _db_read_with_retry(
        lambda: load_eval_pair_state_for_models(task_id, psql_db, models),
        "load_eval_pair_state_for_models",
    )
    for repo, dep_info in db_deployment_ids_by_repo.items():
        deployment_ids_by_repo.setdefault(repo, dep_info)
    """
    Run GRPO evaluation on Basilica with separate deployments per repo.
    """
    command = ["python", "-m", "validator.evaluation.evaluators.grpo"]
    if not dataset.startswith("http://") and not dataset.startswith("https://"):
        raise ValueError(
            "Basilica GRPO eval expects dataset to be an S3/HTTP URL. "
            "Use validator.evaluation.local_evaluation.run_evaluation_docker_grpo for local file paths."
        )
    dataset_type_str = dataset_type.model_dump_json()
    source = create_basilica_eval_runner_source(command, CONTAINER_EVAL_RESULTS_PATH)

    effective_file_format = _effective_remote_file_format(dataset, file_format)
    base_environment = {
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": effective_file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        **vcst.HF_CONTAINER_ENV,
    }

    logger.debug(f"Starting Basilica GRPO evaluation for {len(models)} repos: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_environment)
        repo_env["MODELS"] = repo
        repo_env["DATASET_URL"] = dataset
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await run_remote_eval_repos(
        repos=models,
        model_name=original_model,
        task_type="grpo",
        image=docker_cst.VALIDATOR_DOCKER_IMAGE,
        source=source,
        build_env_for_repo=build_env_for_repo,
        gpu_count=max(1, num_gpus),
        gpu_models=vcst.BASILICA_GPU_MODELS,
        min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
        storage=False,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo=deployment_ids_str,
        # Track every pod at reserve time (see image path) so a deploy that never reaches readiness
        # can't leak as an untracked orphan.
        reserve_deployment_id=True,
    )

    evaluation_results = _collect_repo_evaluation_results(models, repo_results)
    evaluation_results = normalize_rewards_and_compute_loss(evaluation_results)
    logger.debug(f"Grpo evaluation results post normalization: {evaluation_results}")
    return process_evaluation_results(evaluation_results, is_image=False)


async def run_evaluation_basilica_image(
    test_split_url: str,
    original_model_repo: str,
    models: list[str],
    model_type: ImageModelType,
    training_mode: ImageTrainingMode,
    num_gpus: int,
    trigger_word: str | None = None,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
) -> DockerEvaluationResults:
    deployment_ids_by_repo = {}
    db_deployment_ids_by_repo, repo_to_hotkey = await _db_read_with_retry(
        lambda: load_eval_pair_state_for_models(task_id, psql_db, models),
        "load_eval_pair_state_for_models",
    )
    for repo, dep_info in db_deployment_ids_by_repo.items():
        deployment_ids_by_repo.setdefault(repo, dep_info)
    if not test_split_url.startswith("http://") and not test_split_url.startswith("https://"):
        raise ValueError("Basilica image eval expects TEST_SPLIT_URL to be an S3/HTTP URL.")
    if model_type == ImageModelType.SDXL:
        raise ValueError("SDXL is not supported by image prediction L2 evaluation")
    command = ["/app/start.sh"]
    source = create_basilica_eval_runner_source(
        command,
        CONTAINER_EVAL_RESULTS_PATH,
    )

    base_env = {
        "ORIGINAL_MODEL_REPO": original_model_repo,
        "MODEL_TYPE": model_type.value,
        "IMAGE_TRAINING_MODE": training_mode.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "PYTHONUNBUFFERED": "1",
        **vcst.HF_CONTAINER_ENV_IMAGE,
    }
    if trigger_word:
        base_env["TRIGGER_WORD"] = trigger_word

    logger.debug(f"Starting Basilica image evaluation for {len(models)} repos: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_env)
        repo_env["MODELS"] = repo
        repo_env["TEST_SPLIT_URL"] = test_split_url
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    toolkit_eval_models = {
        ImageModelType.FLUX2_KLEIN_4B,
        ImageModelType.QWEN_IMAGE_EDIT_2511,
        ImageModelType.HIDREAM_O1,
        ImageModelType.MOSS_IMAGE_BASE,
    }
    if model_type in toolkit_eval_models:
        evaluator_image = docker_cst.VALIDATOR_DOCKER_IMAGE_DIFFUSION_TOOLKIT
    else:
        evaluator_image = docker_cst.VALIDATOR_DOCKER_IMAGE_DIFFUSION

    repo_results = await run_remote_eval_repos(
        repos=models,
        model_name=original_model_repo,
        task_type="image",
        image=evaluator_image,
        source=source,
        build_env_for_repo=build_env_for_repo,
        gpu_count=max(1, num_gpus),
        gpu_models=vcst.BASILICA_GPU_MODELS,
        min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
        storage=False,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo=deployment_ids_str,
        # Stamp deployment_id on the eval row at reserve time (pre-deploy). Image (FLUX) deploys
        # bounce on GPU capacity and often never reach readiness, so the readiness-callback persist
        # never fires — leaving a live-but-untracked pod that only the slow orphan grace reaps.
        # Reserving under the deployment name keeps every image pod tracked from birth.
        reserve_deployment_id=True,
    )

    evaluation_results = _collect_repo_evaluation_results(models, repo_results)
    return process_evaluation_results(evaluation_results, is_image=True)
