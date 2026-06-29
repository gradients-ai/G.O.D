import asyncio
import json
import random
import uuid
from uuid import UUID

import core.constants.docker as docker_cst
import core.constants.environments as env_cst
import validator.evaluation.constants as vcst
from core.logging import get_environment_logger
from core.logging import get_logger
from core.logging import update_environment_logger_labels
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import EnvironmentDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import GrpoDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.image_models import ImageModelType
from core.models.payload_models import DockerEvaluationResults
from core.models.task_models import TaskType
from validator.db.database import PSQLDB
from validator.db.sql import tasks as tasks_sql
from validator.db.sql import tournaments as tournament_sql
from validator.evaluation.basilica import EvaluationCapacityUnavailable
from validator.evaluation.basilica import EvaluationRetryableError
from validator.evaluation.basilica import _BasilicaEvalContext
from validator.evaluation.basilica import _db_call_with_retry
from validator.evaluation.basilica import _delete_eval_deployment
from validator.evaluation.basilica import _deploy_with_readiness_timeout
from validator.evaluation.basilica import _fetch_attempt_logs
from validator.evaluation.basilica import _get_healthy_existing_basilica_deployment
from validator.evaluation.basilica import _poll_eval_deployment
from validator.evaluation.basilica import _release_reserved_gpus
from validator.evaluation.basilica import run_basilica_eval_repos
from validator.evaluation.basilica_deployments import create_basilica_eval_runner_source
from validator.evaluation.db_utils import load_eval_pair_state_for_models
from validator.evaluation.evaluation_logging import _log_eval_step
from validator.evaluation.pvp.models import PvPEvalConfig
from validator.evaluation.pvp.models import PvPEvalResults
from validator.evaluation.pvp.models import PvPGroupResults
from validator.evaluation.pvp.models import PvPMatchupConfig
from validator.evaluation.pvp.models import PvPMode
from validator.evaluation.pvp.models import PvPModelSpec
from validator.evaluation.pvp.models import PvPPairResult
from validator.evaluation.result_processing import normalize_rewards_and_compute_loss
from validator.evaluation.result_processing import process_evaluation_results
from validator.scoring.models import IndividualEvalResult
from validator.scoring.models import MinerRepos
from validator.tasks.datasets.constants import CONTAINER_EVAL_RESULTS_PATH


try:
    import basilica
except ImportError:
    basilica = None


logger = get_logger(__name__)


def _deployment_url(deployment) -> str | None:
    return getattr(deployment, "url", None)


def _first_environment_name(dataset_type: EnvironmentDatasetType) -> env_cst.EnvironmentName | None:
    environment_names = dataset_type.environment_names or []
    return environment_names[0] if environment_names else None


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
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType | EnvironmentDatasetType,
    file_format: FileFormat,
    num_gpus: int,
    eval_seed: int | None = None,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
    local_logging: bool | None = False,
    use_kl: bool = False,
    kl_coef: float | None = None,
) -> DockerEvaluationResults:
    deployment_ids_by_repo = {}
    db_deployment_ids_by_repo, repo_to_hotkey = await _db_read_with_retry(
        lambda: load_eval_pair_state_for_models(task_id, psql_db, models),
        "load_eval_pair_state_for_models",
    )
    for repo, dep_info in db_deployment_ids_by_repo.items():
        deployment_ids_by_repo.setdefault(repo, dep_info)
    task_type = type(dataset_type).__name__
    is_environment_eval = isinstance(dataset_type, EnvironmentDatasetType)
    environment_name = _first_environment_name(dataset_type) if is_environment_eval else None
    environment_name_value = getattr(environment_name, "value", environment_name)
    is_intercode_eval = is_environment_eval and environment_name_value == env_cst.EnvironmentName.INTERCODE.value
    if is_intercode_eval:
        basilica_image = docker_cst.VALIDATOR_DOCKER_IMAGE_INTERCODE
    elif is_environment_eval:
        basilica_image = docker_cst.VALIDATOR_DOCKER_IMAGE_ENV
    else:
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
    elif isinstance(dataset_type, EnvironmentDatasetType):
        if is_intercode_eval:
            command = ["python", "-m", "validator.evaluation.evaluators.intercode"]
        else:
            command = ["python", "-m", "validator.evaluation.evaluators.environment"]
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")
    if not is_environment_eval and not dataset.startswith("http://") and not dataset.startswith("https://"):
        raise ValueError(
            "Basilica text eval expects dataset to be an S3/HTTP URL. "
            "Use validator.evaluation.local_evaluation.run_evaluation_docker_text for local file paths."
        )
    dataset_type_str = dataset_type.model_dump_json()
    source = create_basilica_eval_runner_source(command, CONTAINER_EVAL_RESULTS_PATH)

    base_env = {
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        **vcst.HF_CONTAINER_ENV,
    }
    if use_kl:
        base_env[docker_cst.USE_KL_ENV] = "1"
        if kl_coef is not None:
            base_env[docker_cst.KL_COEF_ENV] = str(kl_coef)
    if is_environment_eval:
        env_name = env_cst.EnvironmentName(environment_name_value) if environment_name_value else None
        if env_name not in env_cst.ENVIRONMENT_CONFIGS:
            raise ValueError(f"Environment '{env_name}' not found. Supported: {[e.value for e in env_cst.EnvironmentName]}")
        base_seed = eval_seed if eval_seed is not None else vcst.ENV_EVAL_DEFAULT_SEED
        base_env["ENVIRONMENT_NAME"] = env_name.value
        base_env["EVAL_SEED"] = str(base_seed)
        base_env["ENV_EVAL_TEMPERATURE"] = str(vcst.ENV_EVAL_TEMPERATURE)
        # InterCode runs bash actions in-process, so only generic envs get ENV_SERVER_CMD.
        if not is_intercode_eval:
            base_env["ENV_SERVER_CMD"] = vcst.ENV_SERVER_CMD_DEFAULT

    logger.debug(f"Running Basilica {task_type} evaluation (per-repo deployments) for models: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_env)
        repo_env["MODELS"] = repo
        if not is_environment_eval:
            repo_env["DATASET_URL"] = dataset
        return repo_env

    deployment_ids_str = {
        r: v for r, v in deployment_ids_by_repo.items()
        if isinstance(v, str) and not is_environment_eval
    }

    repo_results = await run_basilica_eval_repos(
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
        persist_deployment_ids=not is_environment_eval,
        reserve_deployment_id=False,
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

    base_environment = {
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
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

    repo_results = await run_basilica_eval_repos(
        repos=models,
        model_name=original_model,
        task_type="grpo",
        image=docker_cst.VALIDATOR_DOCKER_IMAGE,
        source=source,
        build_env_for_repo=build_env_for_repo,
        gpu_count=max(1, num_gpus),
        gpu_models=vcst.BASILICA_GPU_MODELS,
        min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo=deployment_ids_str,
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
    command = ["/app/start.sh"]
    source = create_basilica_eval_runner_source(command, CONTAINER_EVAL_RESULTS_PATH)

    base_env = {
        "ORIGINAL_MODEL_REPO": original_model_repo,
        "MODEL_TYPE": model_type.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        **vcst.HF_CONTAINER_ENV_IMAGE,
    }

    logger.debug(f"Starting Basilica image evaluation for {len(models)} repos: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_env)
        repo_env["MODELS"] = repo
        repo_env["TEST_SPLIT_URL"] = test_split_url
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await run_basilica_eval_repos(
        repos=models,
        model_name=original_model_repo,
        task_type="image",
        image="diagonalge/tuning_validator_diffusion:basilica",
        source=source,
        build_env_for_repo=build_env_for_repo,
        gpu_count=max(1, num_gpus),
        gpu_models=vcst.BASILICA_GPU_MODELS,
        min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo=deployment_ids_str,
    )

    evaluation_results = _collect_repo_evaluation_results(models, repo_results)
    return process_evaluation_results(evaluation_results, is_image=True)

async def _persist_pvp_deployment_id(
    *,
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    hotkeys: list[str],
    deployment_name: str,
    ctx: _BasilicaEvalContext,
) -> None:
    if task_id is None or psql_db is None or not hotkeys:
        return

    ctx.log_eval_step("deployment_id_persist_start", deployment=deployment_name)
    if len(hotkeys) == 2:
        await _db_call_with_retry(
            lambda: tournament_sql.set_pvp_pair_deployment_id(
                str(task_id),
                hotkeys[0],
                hotkeys[1],
                deployment_name,
                psql_db,
            ),
            "set_pvp_pair_deployment_id",
            ctx.eval_logger,
            ctx.repo,
        )
    ctx.log_eval_step("deployment_id_persist_complete", deployment=deployment_name)


async def _deploy_pvp_eval(
    pvp_config: PvPEvalConfig,
    label: str,
    repos_label: str,
    image: str | None = None,
    gpu_count: int | None = None,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
    hotkeys: list[str] | None = None,
) -> dict:
    """Deploy a PvP eval container via Basilica and return the raw result dict.

    Shared by PvP pair eval calls.
    """
    hotkeys = hotkeys or []
    image = image or docker_cst.VALIDATOR_DOCKER_IMAGE_PVP
    gpu_count = gpu_count or vcst.PVP_BASILICA_GPU_COUNT
    env = {
        vcst.PVP_CONFIG_ENV_VAR: pvp_config.model_dump_json(),
        **vcst.HF_CONTAINER_ENV,
    }
    command = ["python", "-m", "validator.evaluation.pvp"]
    source = create_basilica_eval_runner_source(command, vcst.PVP_RESULTS_PATH)

    existing_deployment_name = None
    if task_id is not None and psql_db is not None and len(hotkeys) == 2:
        existing_deployment_name = await _db_read_with_retry(
            lambda: tournament_sql.get_pvp_pair_deployment_id(str(task_id), hotkeys[0], hotkeys[1], psql_db),
            "get_pvp_pair_deployment_id",
        )
    eval_id = str(uuid.uuid4())
    eval_logger = get_environment_logger(
        name=f"pvp-{label}-{eval_id[:8]}",
        repo_id=repos_label,
        eval_id=eval_id,
        model=pvp_config.base_model or "",
        task_type=TaskType.ENVIRONMENTTASK.value,
        task_id=str(task_id) if task_id else "unknown",
        hotkey_a=hotkeys[0] if len(hotkeys) > 0 else None,
        hotkey_b=hotkeys[1] if len(hotkeys) > 1 else None,
        deployment_id=existing_deployment_name,
    )

    def log_step(step: str, **fields) -> None:
        _log_eval_step(eval_logger, step, **fields)

    ctx = _BasilicaEvalContext(
        repo=f"pvp-{label}",
        eval_logger=eval_logger,
        deleted_deployment_names=set(),
        log_eval_step=log_step,
    )

    if existing_deployment_name:
        resume_deployment = await _get_healthy_existing_basilica_deployment(
            existing_deployment_name=existing_deployment_name,
            ctx=ctx,
        )
        if resume_deployment is not None:
            client, deployment, deployment_name = resume_deployment
            update_environment_logger_labels(
                eval_logger,
                deployment_id=deployment_name,
                deployment_url=_deployment_url(deployment),
            )
            result = await _poll_eval_deployment(
                ctx=ctx,
                client=client,
                deployment=deployment,
                deployment_name=deployment_name,
                success_cleanup_reason="pvp_resume_completed",
                failure_cleanup_reason="pvp_resume_failed_or_timed_out",
                timeout_cleanup_reason="pvp_resume_failed_or_timed_out",
                retry_on_failure=False,
                poll_interval_seconds=vcst.EVAL_BASILICA_POLL_INTERVAL_SECONDS,
                max_poll_seconds=vcst.PVP_BASILICA_TTL_SECONDS,
            )
            if isinstance(result, dict):
                await _release_reserved_gpus(
                    task_id=task_id,
                    psql_db=psql_db,
                    hotkeys=hotkeys,
                    deployment_name=None,
                    ctx=ctx,
                )
                return result
            if deployment_name not in ctx.deleted_deployment_names:
                raise EvaluationRetryableError(
                    f"Failed to verify deletion for resumed PvP deployment {deployment_name}"
                )
            eval_logger.error("PvP %s resume returned non-dict result; redeploying: %s", label, result)
        await _release_reserved_gpus(
            task_id=task_id,
            psql_db=psql_db,
            hotkeys=hotkeys,
            deployment_name=None,
            ctx=ctx,
        )

    for attempt in range(1, vcst.EVAL_BASILICA_MAX_RETRIES + 1):
        deployment = None
        deployment_name = str(uuid.uuid4())
        try:
            update_environment_logger_labels(eval_logger, deployment_id=deployment_name)
            log_step("attempt_start", attempt=f"{attempt}/{vcst.EVAL_BASILICA_MAX_RETRIES}", deployment=deployment_name)
            eval_logger.info("Starting PvP %s eval attempt %d/%d", label, attempt, vcst.EVAL_BASILICA_MAX_RETRIES)
            client = basilica.BasilicaClient()
            if task_id is not None and psql_db is not None and hotkeys and gpu_count > 0:
                reserved = await _db_call_with_retry(
                    lambda: tasks_sql.try_reserve_evaluation_gpus(
                        task_id,
                        hotkeys[:1],
                        None,
                        gpu_count,
                        psql_db,
                    ),
                    "try_reserve_evaluation_gpus(pvp)",
                    ctx.eval_logger,
                    ctx.repo,
                )
                if not reserved:
                    log_step("gpu_capacity_unavailable", deployment=deployment_name, gpu_count=gpu_count)
                    raise EvaluationCapacityUnavailable(
                        f"Not enough evaluation GPU capacity for PvP deployment {deployment_name} ({gpu_count} GPUs)"
                    )
            log_step(
                "deploy_start",
                deployment=deployment_name,
                image=image,
                gpu_count=gpu_count,
                gpu_models=",".join(vcst.BASILICA_GPU_MODELS),
                min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
            )

            async def persist_verified_pvp_deployment(verified_deployment_name: str) -> None:
                await _persist_pvp_deployment_id(
                    task_id=task_id,
                    psql_db=psql_db,
                    hotkeys=hotkeys,
                    deployment_name=verified_deployment_name,
                    ctx=ctx,
                )

            deployment, resolved_deployment_name = await _deploy_with_readiness_timeout(
                ctx=ctx,
                client=client,
                deployment_name=deployment_name,
                deploy_kwargs={
                    "name": deployment_name,
                    "source": source,
                    "image": image,
                    "port": vcst.PVP_BASILICA_PORT,
                    "cpu": vcst.EVAL_BASILICA_CPU,
                    "memory": vcst.EVAL_BASILICA_MEMORY,
                    "ttl_seconds": vcst.PVP_BASILICA_TTL_SECONDS,
                    "timeout": vcst.EVAL_BASILICA_TIMEOUT,
                    "env": env,
                    "gpu_count": gpu_count,
                    "gpu_models": vcst.BASILICA_GPU_MODELS,
                    "min_gpu_memory_gb": vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
                },
                on_verified_deployment_name=persist_verified_pvp_deployment,
            )
            update_environment_logger_labels(
                eval_logger,
                deployment_id=resolved_deployment_name,
                deployment_url=_deployment_url(deployment),
            )
            log_step("deploy_complete", deployment=resolved_deployment_name)
            if resolved_deployment_name != deployment_name:
                await _release_reserved_gpus(
                    task_id=task_id,
                    psql_db=psql_db,
                    hotkeys=hotkeys,
                    deployment_name=None,
                    ctx=ctx,
                )
                if task_id is not None and psql_db is not None and hotkeys and gpu_count > 0:
                    reserved = await _db_call_with_retry(
                        lambda: tasks_sql.try_reserve_evaluation_gpus(
                            task_id,
                            hotkeys[:1],
                            None,
                            gpu_count,
                            psql_db,
                        ),
                        "try_reserve_evaluation_gpus(pvp-resolved-name)",
                        ctx.eval_logger,
                        ctx.repo,
                    )
                    if not reserved:
                        deleted = await _delete_eval_deployment(
                            ctx, client, deployment, resolved_deployment_name, "pvp_resolved_name_capacity_unavailable"
                        )
                        if not deleted:
                            raise EvaluationRetryableError(
                                f"Failed to verify deletion for PvP deployment {resolved_deployment_name}"
                            )
                        raise EvaluationCapacityUnavailable(
                            f"Not enough evaluation GPU capacity for PvP deployment {resolved_deployment_name} "
                            f"({gpu_count} GPUs)"
                        )
            eval_logger.info("PvP %s deployment started: %s", label, resolved_deployment_name)

            result = await _poll_eval_deployment(
                ctx=ctx,
                client=client,
                deployment=deployment,
                deployment_name=resolved_deployment_name,
                success_cleanup_reason="pvp_completed",
                failure_cleanup_reason="pvp_failed",
                timeout_cleanup_reason="pvp_timed_out",
                retry_on_failure=True,
                poll_interval_seconds=vcst.EVAL_BASILICA_POLL_INTERVAL_SECONDS,
                max_poll_seconds=vcst.PVP_BASILICA_TTL_SECONDS,
            )
            if isinstance(result, dict):
                await _release_reserved_gpus(
                    task_id=task_id,
                    psql_db=psql_db,
                    hotkeys=hotkeys,
                    deployment_name=None,
                    ctx=ctx,
                )
                return result

            raise RuntimeError(str(result))

        except Exception as exc:
            remaining = vcst.EVAL_BASILICA_MAX_RETRIES - attempt
            dep_name = (getattr(deployment, "name", None) or deployment_name) if deployment is not None else deployment_name
            if deployment is not None:
                deleted = await _delete_eval_deployment(ctx, client, deployment, dep_name, "pvp_attempt_exception")
                if not deleted:
                    raise EvaluationRetryableError(
                        f"Failed to verify deletion for PvP deployment {dep_name}"
                    ) from exc
            await _release_reserved_gpus(
                task_id=task_id,
                psql_db=psql_db,
                hotkeys=hotkeys,
                deployment_name=None,
                ctx=ctx,
            )
            log_step(
                "attempt_failed",
                attempt=f"{attempt}/{vcst.EVAL_BASILICA_MAX_RETRIES}",
                deployment=deployment_name,
                remaining=remaining,
                error=exc,
            )
            eval_logger.error("PvP %s eval attempt %d failed: %s", label, attempt, exc, exc_info=True)
            if isinstance(exc, EvaluationRetryableError):
                raise
            if remaining > 0:
                eval_logger.info("Retrying in %ds", vcst.EVAL_BASILICA_RETRY_DELAY_SECONDS)
                await asyncio.sleep(vcst.EVAL_BASILICA_RETRY_DELAY_SECONDS)
            else:
                raise RuntimeError(f"PvP {label} eval failed after {vcst.EVAL_BASILICA_MAX_RETRIES} attempts") from exc
        finally:
            if deployment is not None:
                await _fetch_attempt_logs(ctx, deployment, deployment_name)

    raise RuntimeError(f"PvP {label} evaluation failed")


async def run_evaluation_individual(
    miners: MinerRepos,
    base_model: str,
    environment_name: env_cst.EnvironmentName,
    seed: int,
    image: str,
    gpu_count: int,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
    base_chains: dict[str, list[str]] | None = None,
) -> IndividualEvalResult:
    """Run individual (per-miner) eval containers for a single environment.

    Each miner gets its own container. The container runs one model and returns
    a score via the standard eval_loss result format.

    base_chains maps hotkey -> adapter lineage, piped to the container as BASE_CHAIN.
    """
    env_config = env_cst.ENVIRONMENT_CONFIGS[environment_name]
    if not env_config.tournament_eval_command:
        raise ValueError(f"No tournament_eval_command configured for {environment_name.value}")
    command = env_config.tournament_eval_command
    source = create_basilica_eval_runner_source(command, CONTAINER_EVAL_RESULTS_PATH)

    base_env = {
        "ORIGINAL_MODEL": base_model,
        "ENVIRONMENT_NAME": environment_name.value,
        "EVAL_SEED": str(seed),
        "ENV_EVAL_TEMPERATURE": str(vcst.ENV_EVAL_TEMPERATURE),
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        **vcst.HF_CONTAINER_ENV,
    }

    repo_to_hotkey = {repo: hotkey for hotkey, repo in miners.by_hotkey.items()}
    base_chains = base_chains or {}
    existing_deployment_ids_by_hotkey = await _db_read_with_retry(
        lambda: tournament_sql.get_individual_deployment_ids(
            str(task_id),
            miners.hotkeys,
            [environment_name.value],
            psql_db,
        ) if task_id is not None and psql_db is not None else {},
        "get_individual_deployment_ids",
    )
    deployment_ids_by_repo = {
        repo: existing_deployment_ids_by_hotkey[hotkey]
        for hotkey, repo in miners.by_hotkey.items()
        if hotkey in existing_deployment_ids_by_hotkey
    }

    async def persist_individual_deployment_id(repo: str, deployment_name: str) -> None:
        hotkey = repo_to_hotkey.get(repo)
        if not hotkey or task_id is None or psql_db is None:
            return
        await tournament_sql.set_individual_score_deployment_id(
            str(task_id),
            hotkey,
            environment_name.value,
            deployment_name,
            psql_db,
        )

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_env)
        repo_env["MODELS"] = repo
        chain = base_chains.get(repo_to_hotkey.get(repo, repo))
        if chain:
            repo_env["BASE_CHAIN"] = json.dumps(chain)
        return repo_env

    repo_results = await run_basilica_eval_repos(
        repos=miners.repos,
        model_name=base_model,
        task_type=f"ITournEval[{environment_name.value}]",
        image=image,
        source=source,
        build_env_for_repo=build_env_for_repo,
        gpu_count=max(1, gpu_count),
        gpu_models=vcst.BASILICA_GPU_MODELS,
        min_gpu_memory_gb=vcst.BASILICA_SGLANG_MIN_GPU_MEMORY_GB,
        task_id=task_id,
        psql_db=psql_db,
        repo_to_hotkey=repo_to_hotkey,
        deployment_ids_by_repo=deployment_ids_by_repo,
        persist_deployment_ids=False,
        deployment_id_persister=persist_individual_deployment_id,
        reserve_deployment_id=False,
    )

    scores: dict[str, float] = {}
    for repo, result in repo_results.items():
        hotkey = repo_to_hotkey.get(repo, repo)
        if isinstance(result, dict):
            inner = result.get(repo, result)
            if isinstance(inner, dict):
                scores[hotkey] = float(inner.get("eval_loss", 0.0))
            else:
                logger.warning(f"Individual eval unexpected result for {repo}: {inner}")
        else:
            logger.warning(f"Individual eval failed for {repo}: {result}")

    return IndividualEvalResult(
        environment_name=environment_name,
        scores_by_hotkey=scores,
    )


async def run_evaluation_pvp_pair(
    model_a_repo: str,
    model_b_repo: str,
    hotkey_a: str,
    hotkey_b: str,
    base_model: str,
    environment_names: list[env_cst.EnvironmentName],
    seed: int,
    image: str | None = None,
    gpu_count: int | None = None,
    temperature: float = 0.0,
    task_id: UUID | None = None,
    psql_db: PSQLDB | None = None,
    base_chain_a: list[str] | None = None,
    base_chain_b: list[str] | None = None,
) -> PvPGroupResults:
    """Run PvP 1v1 pair evaluation via Basilica.

    Returns PvPGroupResults (single pair) for consistent downstream processing.

    base_chain_a/base_chain_b reconstruct continuation miners' trained bases.
    """
    matchups = {
        env: PvPMatchupConfig(time_budget_seconds=vcst.PVP_MATCHUP_TIME_BUDGET_SECONDS)
        for env in environment_names
    }
    pvp_config = PvPEvalConfig(
        mode=PvPMode.PAIR,
        model_a=PvPModelSpec(repo=model_a_repo, original_model=base_model, base_chain=base_chain_a or []),
        model_b=PvPModelSpec(repo=model_b_repo, original_model=base_model, base_chain=base_chain_b or []),
        matchups=matchups,
        seed=seed,
        temperature=temperature,
    )
    repos_label = f"{model_a_repo.split('/')[-1]},{model_b_repo.split('/')[-1]}"
    result = await _deploy_pvp_eval(
        pvp_config,
        "pair",
        repos_label,
        image=image,
        gpu_count=gpu_count,
        task_id=task_id,
        psql_db=psql_db,
        hotkeys=[hotkey_a, hotkey_b],
    )

    pair_eval = PvPEvalResults.model_validate(result)
    return PvPGroupResults(
        base_model=base_model,
        hotkeys=[hotkey_a, hotkey_b],
        pair_results=[PvPPairResult(
            hotkey_a=hotkey_a,
            hotkey_b=hotkey_b,
            results=pair_eval.results,
        )],
        metadata=pair_eval.metadata,
    )
