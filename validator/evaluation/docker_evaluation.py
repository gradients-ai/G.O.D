import asyncio
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
from core.models.payload_models import DockerEvaluationResults
from validator.db.database import PSQLDB
from validator.evaluation.basilica_deployments import create_basilica_eval_runner_source
from validator.evaluation.db_utils import load_eval_pair_state_for_models
from validator.evaluation.remote import run_remote_eval_repos
from validator.evaluation.result_processing import normalize_rewards_and_compute_loss
from validator.evaluation.result_processing import process_evaluation_results
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
    num_gpus: int,
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
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "PYTHONUNBUFFERED": "1",
        **vcst.HF_CONTAINER_ENV_IMAGE,
    }

    logger.debug(f"Starting Basilica image evaluation for {len(models)} repos: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_env)
        repo_env["MODELS"] = repo
        repo_env["TEST_SPLIT_URL"] = test_split_url
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await run_remote_eval_repos(
        repos=models,
        model_name=original_model_repo,
        task_type="image",
        image=docker_cst.VALIDATOR_DOCKER_IMAGE_DIFFUSION,
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
