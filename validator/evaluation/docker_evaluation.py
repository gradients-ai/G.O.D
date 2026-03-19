import asyncio
import io
import json
import logging
import os
import re
import tarfile
import uuid
from uuid import UUID
import requests
import time
import random
import basilica

from core import constants as cst
from core.models.payload_models import DockerEvaluationResults
from core.models.payload_models import EvaluationResultImage
from core.models.payload_models import EvaluationResultText
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import InstructTextDatasetType
from validator.core import constants as vcst
from validator.db.database import PSQLDB
from validator.utils.logging import get_logger
from validator.utils.logging import get_environment_logger
from validator.evaluation.utils import (
    EVAL_RESULT_STATUS_PATH,
    cleanup_basilica_deployments_by_name,
    deployment_is_healthy,
    create_basilica_eval_runner_source,
    load_eval_pair_state_for_models,
    log_basilica_logs_block,
    persist_deployment_ids_for_repo,
)


logger = get_logger(__name__)
_EVAL_DB_WRITE_SEMAPHORE = asyncio.Semaphore(vcst.EVAL_DB_MAX_CONCURRENT_WRITES)


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


async def cleanup_resources(client):
    """Clean up Docker resources including containers, images, and volumes."""
    try:
        await asyncio.to_thread(client.containers.prune)
        await asyncio.to_thread(client.images.prune, filters={"dangling": True})
        await asyncio.to_thread(client.volumes.prune)
        logger.debug("Completed Docker resource cleanup")
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")


async def get_evaluation_results(container):
    archive_data = await asyncio.to_thread(container.get_archive, cst.CONTAINER_EVAL_RESULTS_PATH)
    tar_stream = archive_data[0]

    file_like_object = io.BytesIO()
    for chunk in tar_stream:
        file_like_object.write(chunk)
    file_like_object.seek(0)

    with tarfile.open(fileobj=file_like_object) as tar:
        members = tar.getnames()
        logger.debug(f"Tar archive members: {members}")
        eval_results_file = None
        for member_info in tar.getmembers():
            if member_info.name.endswith(("evaluation_results.json")):
                eval_results_file = tar.extractfile(member_info)
                break

        if eval_results_file is None:
            raise Exception("Evaluation results file not found in tar archive")

        eval_results_content = eval_results_file.read().decode("utf-8")
        return json.loads(eval_results_content)


def normalize_rewards_and_compute_loss(evaluation_results: dict) -> dict:
    """
    Normalize rewards across repos and compute final evaluation loss with KL penalty.

    Steps:
    1. For each reward type, normalize values across repos by dividing by max (after shifting if negative)
    2. Apply weights to normalized rewards (weights sum to 1)
    3. Sum weighted rewards to get final score in [0,1] range
    4. Apply KL penalty: score - (BETA_GRPO * kl_divergence)

    Special case: 2 repos with negative rewards map to [0.25, 0.75] to avoid extreme scores.

    Args:
        evaluation_results: Dict with model repos as keys and evaluation data as values

    Returns:
        Modified evaluation_results dict with updated eval_loss values
    """
    # Filter out non-repo keys (like model_params_count)
    repo_keys = [key for key in evaluation_results.keys() if key != "model_params_count"]

    if len(repo_keys) < 2:
        # Need at least 2 repos for meaningful normalization
        return evaluation_results

    reward_collections = {}
    for repo_key in repo_keys:
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):  # Skip error entries
            continue

        final_raw_rewards = repo_data.get('final_raw_rewards', {})

        for reward_name, reward_value in final_raw_rewards.items():
            if reward_name not in reward_collections:
                reward_collections[reward_name] = []
            reward_collections[reward_name].append((repo_key, reward_value))

    # Step 1: Normalize each reward type using shift + divide by max
    normalized_rewards_per_repo = {repo_key: {} for repo_key in repo_keys}

    for reward_name, repo_value_pairs in reward_collections.items():
        if len(repo_value_pairs) < 2:
            # Only one value, set to 1.0
            for repo_key, value in repo_value_pairs:
                normalized_rewards_per_repo[repo_key][reward_name] = 1.0
            continue

        values = [value for _, value in repo_value_pairs]
        min_value = min(values)

        # Check if we need to shift (have negatives)
        has_negatives = min_value < 0

        # Shift to positive if needed
        if has_negatives:
            shifted_values = [(repo, value - min_value) for repo, value in repo_value_pairs]
        else:
            shifted_values = repo_value_pairs

        # Find max of shifted values
        max_shifted = max(value for _, value in shifted_values)

        # Special case: 2 repos with negatives -> map to [0.25, 0.75]
        if len(repo_value_pairs) == 2 and has_negatives:
            sorted_pairs = sorted(shifted_values, key=lambda x: x[1])
            normalized_rewards_per_repo[sorted_pairs[0][0]][reward_name] = 0.25
            normalized_rewards_per_repo[sorted_pairs[1][0]][reward_name] = 0.75
        elif max_shifted > 0:
            # Normal case: divide by max
            for repo, shifted_value in shifted_values:
                normalized_rewards_per_repo[repo][reward_name] = shifted_value / max_shifted
        else:
            # All values are zero after shift (all were equal and negative or zero)
            for repo, _ in repo_value_pairs:
                normalized_rewards_per_repo[repo][reward_name] = 1.0

    # Step 2-3: Apply weights and sum (weights already sum to 1)
    final_scores = []

    for repo_key in repo_keys:
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):  # Skip error entries
            continue

        weights = repo_data.get('weights', {})
        normalized_rewards = normalized_rewards_per_repo.get(repo_key, {})

        # Calculate weighted sum
        weighted_sum = 0.0
        for reward_name, normalized_value in normalized_rewards.items():
            weight = weights.get(reward_name, 1.0)
            weighted_sum += normalized_value * weight

        final_scores.append(weighted_sum)

    # Step 4: Apply KL penalty and update eval_loss
    for i, repo_key in enumerate(repo_keys):
        repo_data = evaluation_results[repo_key]
        if isinstance(repo_data, str):  # Skip error entries
            continue

        if i < len(final_scores):
            kl_divergence = repo_data.get('kl_divergence', 0.0)
            # Final score: weighted_sum - BETA_GRPO * kl_divergence
            new_eval_loss = final_scores[i] - (vcst.BETA_GRPO * kl_divergence)
            repo_data['eval_loss'] = new_eval_loss

    return evaluation_results


def process_evaluation_results(results: dict, is_image: bool = False) -> DockerEvaluationResults:
    model_params_count = results.pop("model_params_count", 0)

    processed_results = {}
    for repo, result in results.items():
        if isinstance(result, str) and not isinstance(result, dict):
            processed_results[repo] = Exception(result)
        else:
            if is_image:
                result["is_finetune"] = True
                processed_results[repo] = EvaluationResultImage.model_validate(result)
            else:
                processed_results[repo] = EvaluationResultText.model_validate(result)

    return DockerEvaluationResults(
        results=processed_results,
        base_model_params_count=model_params_count
    )


async def _poll_basilica_result(
    deployment,
    repo: str,
    eval_logger: logging.Logger,
    poll_interval_seconds: int = vcst.EVAL_BASILICA_POLL_INTERVAL_SECONDS,
    max_poll_seconds: int = vcst.EVAL_BASILICA_MAX_POLL_SECONDS,
) -> dict | str:
    """Poll Basilica /result endpoint. Handles status: completed, failed, running, in_progress."""
    started = time.time()
    deployment_name = getattr(deployment, "name", "unknown")
    while time.time() - started < max_poll_seconds:
        try:
            await asyncio.to_thread(log_basilica_logs_block, eval_logger, repo, deployment_name, deployment)
            response = await asyncio.to_thread(
                requests.get,
                f"{deployment.url}{EVAL_RESULT_STATUS_PATH}",
                timeout=30,
            )
            if response.status_code == 200:
                payload = response.json()
                status = payload.get("status")
                if status == "completed":
                    result = payload.get("result")
                    if isinstance(result, dict):
                        eval_logger.info(f"[{repo}] Poll successful. Evaluation completed and result payload received.")
                        return result
                    return f"Completed but result payload invalid: {result}"
                if status == "failed":
                    return payload.get("error", "Basilica eval reported failure")
                eval_logger.info(f"[{repo}] Poll ping: status={status}.")
        except Exception as e:
            eval_logger.error(f"[{repo}] error polling Basilica result: {e}", exc_info=True)
            pass

        eval_logger.info(
            f"[{repo}] result not ready yet (status may be running/in_progress), "
            f"polling again in {poll_interval_seconds // 60} minutes..."
        )
        await asyncio.sleep(poll_interval_seconds)
    return f"Timed out waiting for result after {max_poll_seconds}s"


async def _run_single_basilica_eval_repo(
    *,
    repo: str,
    model_name: str,
    task_type: str,
    image: str,
    source: str,
    env: dict[str, str],
    gpu_count: int,
    gpu_models: list[str],
    min_gpu_memory_gb: int,
    cleanup_names: set[str],
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    repo_to_hotkey: dict[str, str],
    existing_deployment_name: str | None = None,
) -> dict | str:
    """Run one repo eval with retries. Supports resume via existing_deployment_name."""
    eval_id = str(uuid.uuid4())
    eval_logger = get_environment_logger(
        name=f"basilica-{repo.split('/')[-1]}-{eval_id[:8]}",
        repo_id=repo,
        eval_id=eval_id,
        model=model_name,
        task_type=task_type,
    )

    async def _db_call_with_retry(coro_factory, op_name: str):
        last_exc = None
        for attempt in range(1, vcst.EVAL_DB_RETRY_ATTEMPTS + 1):
            try:
                return await coro_factory()
            except Exception as exc:
                last_exc = exc
                delay = vcst.EVAL_DB_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, 0.3)
                if attempt < vcst.EVAL_DB_RETRY_ATTEMPTS:
                    eval_logger.warning(
                        f"[{repo}] DB op '{op_name}' failed attempt {attempt}/{vcst.EVAL_DB_RETRY_ATTEMPTS}: {exc}; "
                        f"retrying in {delay + jitter:.2f}s"
                    )
                    await asyncio.sleep(delay + jitter)
                else:
                    eval_logger.error(
                        f"[{repo}] DB op '{op_name}' failed after {vcst.EVAL_DB_RETRY_ATTEMPTS} attempts: {exc}"
                    )
        raise last_exc

    if existing_deployment_name:
        try:
            client = basilica.BasilicaClient()
            deployments = await asyncio.to_thread(client.list)
            by_name = {getattr(dep, "name", None): dep for dep in deployments}
            deployment = by_name.get(existing_deployment_name)
            if deployment is None:
                eval_logger.warning(f"[{repo}] resume: deployment {existing_deployment_name} not found, redeploying")
            elif not deployment_is_healthy(deployment):
                eval_logger.warning(f"[{repo}] resume: deployment {existing_deployment_name} unhealthy, redeploying")
                await asyncio.to_thread(deployment.delete)
            else:
                cleanup_names.add(existing_deployment_name)
                eval_logger.info(f"[{repo}] resuming polling deployment {existing_deployment_name}")
                result = await _poll_basilica_result(deployment, repo, eval_logger=eval_logger)
                if isinstance(result, dict):
                    return result
                return str(result) if result else "Resume poll returned empty"
        except Exception as e:
            eval_logger.error(f"[{repo}] resume failed, redeploying: {e}", exc_info=True)

    for attempt in range(1, vcst.EVAL_BASILICA_MAX_RETRIES + 1):
        deployment = None
        deployment_name = str(uuid.uuid4())
        cleanup_deployment = True
        try:
            eval_logger.info(f"[{repo}] starting Basilica evaluation attempt {attempt}/{vcst.EVAL_BASILICA_MAX_RETRIES}")
            client = basilica.BasilicaClient()
            await asyncio.sleep(random.uniform(0.0, 0.25))
            async with _EVAL_DB_WRITE_SEMAPHORE:
                await _db_call_with_retry(
                    lambda: persist_deployment_ids_for_repo(
                        task_id,
                        psql_db,
                        repo_to_hotkey,
                        repo,
                        deployment_name,
                        None,
                    ),
                    "persist_deployment_ids_for_repo(pre-deploy)",
                )
            deployment = await asyncio.to_thread(
                client.deploy,
                name=deployment_name,
                source=source,
                image=image,
                port=8000,
                cpu=vcst.EVAL_BASILICA_CPU,
                memory=vcst.EVAL_BASILICA_MEMORY,
                ttl_seconds=vcst.EVAL_BASILICA_TTL_SECONDS,
                timeout=vcst.EVAL_BASILICA_TIMEOUT,
                env=env,
                gpu_count=gpu_count,
                gpu_models=gpu_models,
                min_gpu_memory_gb=min_gpu_memory_gb,
            )
            resolved_deployment_name = getattr(deployment, "name", None) or deployment_name
            if resolved_deployment_name != deployment_name:
                await asyncio.sleep(random.uniform(0.0, 0.25))
                async with _EVAL_DB_WRITE_SEMAPHORE:
                    await _db_call_with_retry(
                        lambda: persist_deployment_ids_for_repo(
                            task_id,
                            psql_db,
                            repo_to_hotkey,
                            repo,
                            resolved_deployment_name,
                            None,
                        ),
                        "persist_deployment_ids_for_repo(post-deploy)",
                    )
            cleanup_names.add(resolved_deployment_name)
            eval_logger.info(f"[{repo}] deployment started: {resolved_deployment_name}")
            result = await _poll_basilica_result(deployment, repo, eval_logger=eval_logger)
            if isinstance(result, dict):
                return result
            if "Timed out" in str(result):
                logger.error(f"[{repo}] poll timeout, skipping retries: {result}")
                return result
            raise RuntimeError(str(result))
        except asyncio.CancelledError:
            cleanup_deployment = False 
            raise
        except Exception as e:
            remaining = vcst.EVAL_BASILICA_MAX_RETRIES - attempt
            eval_logger.error(
                f"[{repo}] attempt {attempt}/{vcst.EVAL_BASILICA_MAX_RETRIES} failed: {e}",
                exc_info=True,
            )
            if remaining > 0:
                eval_logger.info(
                    f"[{repo}] retrying in {vcst.EVAL_BASILICA_RETRY_DELAY_SECONDS // 60} minutes "
                    f"({remaining} attempts remaining)"
                )
                await asyncio.sleep(vcst.EVAL_BASILICA_RETRY_DELAY_SECONDS)
            else:
                return f"Evaluation failed after {vcst.EVAL_BASILICA_MAX_RETRIES} attempts: {e}"
        finally:
            if deployment is not None:
                try:
                    dep_name = getattr(deployment, "name", None) or deployment_name
                    await asyncio.to_thread(log_basilica_logs_block, eval_logger, repo, dep_name, deployment)
                    if cleanup_deployment:
                        await asyncio.to_thread(deployment.delete)
                except Exception as e:
                    eval_logger.warning(f"[{repo}] failed to cleanup deployment {dep_name}: {e}")

    return "Evaluation failed"


async def _run_basilica_eval_repos(
    *,
    repos: list[str],
    model_name: str,
    task_type: str,
    image: str,
    source: str,
    build_env_for_repo,
    gpu_count: int,
    gpu_models: list[str],
    min_gpu_memory_gb: int,
    task_id: UUID | None,
    psql_db: PSQLDB | None,
    repo_to_hotkey: dict[str, str],
    deployment_ids_by_repo: dict[str, str] | None = None,
) -> dict[str, dict | str]:
    deployment_ids_by_repo = deployment_ids_by_repo or {}
    cleanup_names: set[str] = set()
    task_results = await asyncio.gather(
        *[
            _run_single_basilica_eval_repo(
                repo=repo,
                model_name=model_name,
                task_type=task_type,
                image=image,
                source=source,
                env=build_env_for_repo(repo),
                gpu_count=gpu_count,
                gpu_models=gpu_models,
                min_gpu_memory_gb=min_gpu_memory_gb,
                cleanup_names=cleanup_names,
                task_id=task_id,
                psql_db=psql_db,
                repo_to_hotkey=repo_to_hotkey,
                existing_deployment_name=deployment_ids_by_repo.get(repo) if isinstance(deployment_ids_by_repo.get(repo), str) else None,
            )
            for repo in repos
        ],
        return_exceptions=True,
    )
    await cleanup_basilica_deployments_by_name(cleanup_names)
    out: dict[str, dict | str] = {}
    for repo, result in zip(repos, task_results):
        if isinstance(result, Exception):
            out[repo] = f"Evaluation failed: {result}"
        else:
            out[repo] = result
    return out


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
    basilica_image = vcst.ENV_EVAL_IMAGE if is_environment_eval else cst.VALIDATOR_DOCKER_IMAGE
    if isinstance(dataset_type, (InstructTextDatasetType, ChatTemplateDatasetType)):
        command = ["python", "-m", "validator.evaluation.eval_instruct_text"]
    elif isinstance(dataset_type, DpoDatasetType):
        command = ["python", "-m", "validator.evaluation.eval_dpo"]
    elif isinstance(dataset_type, GrpoDatasetType):
        return await run_evaluation_basilica_grpo(
            dataset, models, original_model, dataset_type, file_format, num_gpus,
            task_id=task_id,
            psql_db=psql_db,
            deployment_ids_by_repo=deployment_ids_by_repo,
        )
    elif isinstance(dataset_type, EnvironmentDatasetType):
        command = ["python", "-m", "validator.evaluation.eval_environment"]
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")
    if not is_environment_eval and not dataset.startswith("http://") and not dataset.startswith("https://"):
        raise ValueError(
            "Basilica text eval expects dataset to be an S3/HTTP URL. "
            "Use validator.evaluation.local_evaluation.run_evaluation_docker_text for local file paths."
        )
    dataset_type_str = dataset_type.model_dump_json()
    source = create_basilica_eval_runner_source(command, cst.CONTAINER_EVAL_RESULTS_PATH)

    base_env = {
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
        "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if is_environment_eval:
        env_name = dataset_type.environment_name
        if env_name not in vcst.ENVIRONMENTS:
            raise ValueError(f"Environment '{env_name}' not found. Supported: {list(vcst.ENVIRONMENTS.keys())}")
        base_seed = eval_seed if eval_seed is not None else vcst.ENV_EVAL_DEFAULT_SEED
        base_env["ENVIRONMENT_NAME"] = env_name
        base_env["EVAL_SEED"] = str(base_seed)
        base_env["ENV_EVAL_TEMPERATURE"] = str(vcst.ENV_EVAL_TEMPERATURE)
        base_env["ENV_SERVER_CMD"] = vcst.ENV_SERVER_CMD_DEFAULT

    logger.debug(f"Running Basilica {task_type} evaluation (per-repo deployments) for models: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_env)
        repo_env["MODELS"] = repo
        if not is_environment_eval:
            repo_env["DATASET_URL"] = dataset
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await _run_basilica_eval_repos(
        repos=models,
        model_name=original_model,
        task_type=task_type,
        image=basilica_image,
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

    evaluation_results: dict[str, dict | str] = {}
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
        else:
            candidate_keys = [k for k in raw_result.keys() if k != "model_params_count"]
            if len(candidate_keys) == 1:
                evaluation_results[repo] = raw_result[candidate_keys[0]]
            else:
                evaluation_results[repo] = f"Evaluation failed: missing result key for repo {repo}"

    if model_params_count:
        evaluation_results["model_params_count"] = model_params_count

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
    command = ["python", "-m", "validator.evaluation.eval_grpo"]
    if not dataset.startswith("http://") and not dataset.startswith("https://"):
        raise ValueError(
            "Basilica GRPO eval expects dataset to be an S3/HTTP URL. "
            "Use validator.evaluation.local_evaluation.run_evaluation_docker_grpo for local file paths."
        )
    dataset_type_str = dataset_type.model_dump_json()
    source = create_basilica_eval_runner_source(command, cst.CONTAINER_EVAL_RESULTS_PATH)

    base_environment = {
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
        "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }

    logger.debug(f"Starting Basilica GRPO evaluation for {len(models)} repos: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_environment)
        repo_env["MODELS"] = repo
        repo_env["DATASET_URL"] = dataset
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await _run_basilica_eval_repos(
        repos=models,
        model_name=original_model,
        task_type="grpo",
        image=cst.VALIDATOR_DOCKER_IMAGE,
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
        else:
            candidate_keys = [k for k in raw_result.keys() if k != "model_params_count"]
            if len(candidate_keys) == 1:
                evaluation_results[repo] = raw_result[candidate_keys[0]]
            else:
                evaluation_results[repo] = f"Evaluation failed: missing result key for repo {repo}"

    if model_params_count:
        evaluation_results["model_params_count"] = model_params_count

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
    source = create_basilica_eval_runner_source(command, cst.CONTAINER_EVAL_RESULTS_PATH)

    base_env = {
        "ORIGINAL_MODEL_REPO": original_model_repo,
        "MODEL_TYPE": model_type.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
        "HUGGINGFACE_HUB_CACHE": "/root/.cache/huggingface/hub"
    }

    logger.debug(f"Starting Basilica image evaluation for {len(models)} repos: {models}")

    def build_env_for_repo(repo: str) -> dict[str, str]:
        repo_env = dict(base_env)
        repo_env["MODELS"] = repo
        repo_env["TEST_SPLIT_URL"] = test_split_url
        return repo_env

    deployment_ids_str = {r: v for r, v in deployment_ids_by_repo.items() if isinstance(v, str)}

    repo_results = await _run_basilica_eval_repos(
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

    evaluation_results: dict[str, dict | str] = {}
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
        else:
            candidate_keys = [k for k in raw_result.keys() if k != "model_params_count"]
            if len(candidate_keys) == 1:
                evaluation_results[repo] = raw_result[candidate_keys[0]]
            else:
                evaluation_results[repo] = f"Evaluation failed: missing result key for repo {repo}"

    if model_params_count:
        evaluation_results["model_params_count"] = model_params_count

    return process_evaluation_results(evaluation_results, is_image=True)
