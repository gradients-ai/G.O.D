import asyncio
import glob
import io
import json
import logging
import os
import re
import shutil
import tarfile
import uuid
from datetime import datetime
from typing import Optional

import docker
from docker.models.containers import Container
from docker.types import Mount
from huggingface_hub import snapshot_download
import aiohttp
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
from core.utils import download_s3_file
from validator.core import constants as vcst
from validator.tasks.task_prep import unzip_to_temp_path
from validator.utils.logging import get_all_context_tags
from validator.utils.logging import get_logger
from validator.utils.logging import get_environment_logger
from validator.utils.logging import stream_container_logs
from validator.evaluation.utils import (
    deploy_sglang_basilica,
    deploy_env_basilica,
    wait_for_basilica_health,
    check_for_lora,
)


logger = get_logger(__name__)


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


async def run_evaluation_docker_text(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType | EnvironmentDatasetType,
    file_format: FileFormat,
    gpu_ids: list[int],
    eval_seed: int | None = None,
) -> DockerEvaluationResults:

    if isinstance(dataset_type, (InstructTextDatasetType, ChatTemplateDatasetType)):
        command = ["python", "-m", "validator.evaluation.eval_instruct_text"]
    elif isinstance(dataset_type, DpoDatasetType):
        command = ["python", "-m", "validator.evaluation.eval_dpo"]
    elif isinstance(dataset_type, GrpoDatasetType):
        return await run_evaluation_docker_grpo(dataset, models, original_model, dataset_type, file_format, gpu_ids)
    elif isinstance(dataset_type, EnvironmentDatasetType):
        return await run_evaluation_docker_environment(dataset, models, original_model, dataset_type, file_format, gpu_ids, eval_seed)
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")
    task_type = type(dataset_type).__name__

    client = docker.from_env()
    dataset_type_str = dataset_type.model_dump_json()
    dataset_filename = os.path.basename(dataset)
    dataset_dir = os.path.dirname(os.path.abspath(dataset))

    environment = {
        "DATASET": f"/workspace/input_data/{dataset_filename}",
        "MODELS": ",".join(models),
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
    }
    logger.info(f"Running {task_type} evaluation for models: {models}")

    volume_bindings = {
        dataset_dir: {
            "bind": "/workspace/input_data",
            "mode": "ro",
        },
        os.path.expanduser(cst.CACHE_DIR_HUB): {
            "bind": "/root/.cache/huggingface/hub",
            "mode": "rw",
        }
    }

    container = None
    retry_delay = 5.0
    
    try:
        while True:
            try:
                container: Container = await asyncio.to_thread(
                    client.containers.run,
                    cst.VALIDATOR_DOCKER_IMAGE,
                    command=command,
                    environment=environment,
                    volumes=volume_bindings,
                    runtime="nvidia",
                    device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gid) for gid in gpu_ids])],
                    detach=True,
                )
                log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, None, get_all_context_tags()))
                result = await asyncio.to_thread(container.wait)
                log_task.cancel()

                if result["StatusCode"] != 0:
                    raise Exception(f"Container exited with status {result['StatusCode']}")

                eval_results = await get_evaluation_results(container)
                return process_evaluation_results(eval_results, is_image=False)

            except Exception as e:
                logger.error(f"Failed to retrieve {task_type} evaluation results: {str(e)}, retrying in {retry_delay}s...", exc_info=True)
                if container is not None:
                    try:
                        await asyncio.to_thread(container.remove, force=True)
                        container = None
                    except:
                        pass
                await asyncio.sleep(retry_delay)
                continue

    finally:
        try:
            if container is not None:
                await asyncio.to_thread(container.remove, force=True)
            await cleanup_resources(client)
        except Exception as e:
            logger.info(f"A problem with cleaning up {e}")
        client.close()


async def run_evaluation_docker_grpo(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: GrpoDatasetType,
    file_format: FileFormat,
    gpu_ids: list[int],
) -> DockerEvaluationResults:
    """
    Run GRPO evaluation with separate containers for each model repo.
    This approach launches one container per repo and merges results.
    """
    logger.info(f"Downloading original GRPO model: {original_model}")
    cache_dir = os.path.expanduser(cst.CACHE_DIR_HUB)
    original_model_path = await asyncio.to_thread(
        snapshot_download,
        repo_id=original_model,
        cache_dir=cache_dir,
        ignore_patterns=None
    )

    command = ["python", "-m", "validator.evaluation.eval_grpo"]
    dataset_type_str = dataset_type.model_dump_json()
    dataset_filename = os.path.basename(dataset)
    dataset_dir = os.path.dirname(os.path.abspath(dataset))

    # Shared environment settings
    base_environment = {
        "DATASET": f"/workspace/input_data/{dataset_filename}",
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/hub",
        "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
    }

    volume_bindings = {
        dataset_dir: {
            "bind": "/workspace/input_data",
            "mode": "ro",
        },
        os.path.expanduser(cst.CACHE_DIR_HUB): {
            "bind": "/root/.cache/huggingface/hub",
            "mode": "rw",
        }
    }

    logger.info(f"Starting sequential GRPO evaluation for {len(models)} repos: {models}")

    evaluation_results = {}
    for repo in models:
        client = docker.from_env()
        environment = base_environment.copy()
        environment["MODELS"] = repo
        retry_delay = 5.0
        
        # Infinite retry for model download
        model_path = None
        while model_path is None:
            try:
                model_path = await asyncio.to_thread(
                    snapshot_download,
                    repo_id=repo,
                    cache_dir=cache_dir,
                    ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.pkl", "*.pth"]
                )
            except Exception as e:
                logger.error(f"Failed to download {repo}: {str(e)}, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)

        container = None  # Initialize container variable
        
        # Infinite retry for container execution
        while True:
            try:
                container: Container = await asyncio.to_thread(
                    client.containers.run,
                    cst.VALIDATOR_DOCKER_IMAGE,
                    command=command,
                    environment=environment,
                    volumes=volume_bindings,
                    runtime="nvidia",
                    device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gid) for gid in gpu_ids])],
                    detach=True,
                    network_mode="none",
                )

                log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, None, get_all_context_tags()))
                result = await asyncio.to_thread(container.wait)
                log_task.cancel()

                if result["StatusCode"] != 0:
                    raise Exception(f"Container for {repo} exited with non-zero status: {result['StatusCode']}")

                eval_results = await get_evaluation_results(container)
                evaluation_results[repo] = eval_results[repo]
                if "model_params_count" in eval_results and "model_params_count" not in evaluation_results:
                    evaluation_results["model_params_count"] = eval_results["model_params_count"]
                break  # Success, exit retry loop

            except Exception as e:
                logger.error(f"Failed to evaluate repo {repo}: {str(e)}, retrying in {retry_delay}s...", exc_info=True)
                if container is not None:
                    try:
                        await asyncio.to_thread(container.remove, force=True)
                    except:
                        pass
                await asyncio.sleep(retry_delay)
                continue

            finally:
                if container is not None:
                    try:
                        await asyncio.to_thread(container.remove, force=True)
                        await cleanup_resources(client)
                    except Exception as e:
                        logger.info(f"Problem with cleaning up container for {repo}: {e}")
        client.close()

    evaluation_results = normalize_rewards_and_compute_loss(evaluation_results)
    logger.info(f"Grpo evaluation results post normalization: {evaluation_results}")
    return process_evaluation_results(evaluation_results, is_image=False)


async def run_evaluation_docker_environment(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: EnvironmentDatasetType,
    file_format: FileFormat,
    gpu_ids: list[int],
    eval_seed: int | None = None,
) -> DockerEvaluationResults:
    """Run environment evaluation using Basilica deployments for SGLang and environment server.

    Each model repo gets its own Basilica deployments with retry logic.
    Repos are evaluated in parallel.
    """
    logger.info(f"Starting Basilica environment evaluation for {len(models)} repos: {models}")

    env_name = dataset_type.environment_name
    if env_name not in vcst.ENVIRONMENTS:
        raise ValueError(f"Environment '{env_name}' not found. Supported: {list(vcst.ENVIRONMENTS.keys())}")

    env_config = vcst.ENVIRONMENTS[env_name]
    task_id_min, task_id_max = env_config["task_id_range"]
    env_image = env_config["env_image"]
    env_payload_extra = env_config.get("eval_payload_extra", {})

    base_seed = eval_seed if eval_seed is not None else vcst.ENV_EVAL_DEFAULT_SEED
    seed_generator = random.Random(base_seed)
    eval_seeds = [seed_generator.randint(1, 1000000) for _ in range(vcst.ENV_EVAL_NUM_SEEDS)]
    logger.info(f"Generated {vcst.ENV_EVAL_NUM_SEEDS} seeds from base_seed={base_seed}")

    async def evaluate_single_repo(repo: str, repo_idx: int) -> tuple[str, dict | str]:
        """Deploy, evaluate, and cleanup a single repo on Basilica."""
        eval_id = str(uuid.uuid4())
        repo_name = repo.split("/")[-1]
        env_logger = get_environment_logger(
            name=repo_name, repo_id=repo, eval_id=eval_id, model=original_model,
        )
        deployments = {}
        repo_result = None

        def _log_deployment_logs(deployment, deployment_type: str):
            """Fetch and log Basilica deployment logs."""
            try:
                logs = deployment.logs()
                if not logs:
                    return
                for line in logs.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        log_data = json.loads(line)
                        message = log_data.get("message", "")
                        if message:
                            message = re.sub(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*", "", message)
                            message = re.sub(r"^data:\s*", "", message).rstrip(", ")
                            if message.strip():
                                env_logger.info(f"[{deployment_type}] {message}")
                    except (json.JSONDecodeError, AttributeError):
                        cleaned = line.strip()
                        if cleaned:
                            env_logger.info(f"[{deployment_type}] {cleaned}")
            except Exception as e:
                env_logger.warning(f"Failed to fetch {deployment_type} logs: {e}")

        async def _cleanup_deployments(deployments_dict: dict, fetch_logs: bool = False):
            """Clean up Basilica deployments."""
            for name, deployment in deployments_dict.items():
                try:
                    if fetch_logs:
                        await asyncio.to_thread(_log_deployment_logs, deployment, name)
                    deployment.delete()
                    env_logger.info(f"Cleaned up {name} deployment")
                except Exception as e:
                    env_logger.warning(f"Failed to cleanup {name}: {e}", exc_info=True)
            deployments_dict.clear()

        for attempt in range(1, vcst.ENV_EVAL_MAX_RETRIES + 1):
            try:
                is_lora = await asyncio.to_thread(check_for_lora, repo, local_files_only=False)

                if is_lora:
                    base_model = original_model
                    lora_model = repo
                    inference_model_name = f"{original_model}:trained_lora"
                    env_logger.info(f"Deploying SGLang: {original_model} w/ LoRA {repo}")
                else:
                    base_model = repo
                    lora_model = None
                    inference_model_name = repo
                    env_logger.info(f"Deploying SGLang: {repo}")

                sglang_deployment = await asyncio.to_thread(
                    deploy_sglang_basilica, base_model, lora_model, f"{eval_id}-sglang", base_seed,
                )
                deployments["sglang"] = sglang_deployment
                await asyncio.to_thread(wait_for_basilica_health, sglang_deployment.url)
                env_logger.info(f"SGLang ready at: {sglang_deployment.url}")

                env_deployment = await asyncio.to_thread(
                    deploy_env_basilica, f"{eval_id}-env", env_image,
                )
                deployments["env"] = env_deployment
                await asyncio.to_thread(wait_for_basilica_health, env_deployment.url, 1800, "/health")
                env_logger.info(f"Environment server ready at: {env_deployment.url}")

                avg_score = await _run_environment_evaluation(
                    sglang_deployment.url,
                    env_deployment.url,
                    eval_seeds,
                    task_id_max,
                    vcst.ENV_EVAL_TEMPERATURE,
                    env_logger,
                    inference_model_name,
                    task_id_min,
                    env_payload_extra=env_payload_extra,
                )

                repo_result = {"is_finetune": True, "eval_loss": avg_score}

                await asyncio.to_thread(_log_deployment_logs, sglang_deployment, "SGLang")
                await asyncio.to_thread(_log_deployment_logs, env_deployment, "Env")
                await _cleanup_deployments(deployments)
                break

            except Exception as e:
                remaining = vcst.ENV_EVAL_MAX_RETRIES - attempt
                if remaining > 0:
                    env_logger.error(
                        f"Attempt {attempt}/{vcst.ENV_EVAL_MAX_RETRIES} failed: {e}, "
                        f"retrying in {vcst.ENV_EVAL_DEPLOYMENT_RETRY_DELAY / 60:.0f} min...",
                        exc_info=True,
                    )
                else:
                    env_logger.error(
                        f"Attempt {attempt}/{vcst.ENV_EVAL_MAX_RETRIES} failed: {e}, max retries reached.",
                        exc_info=True,
                    )
                await _cleanup_deployments(deployments, fetch_logs=True)
                if remaining > 0:
                    await asyncio.sleep(vcst.ENV_EVAL_DEPLOYMENT_RETRY_DELAY)

        return (repo, repo_result if repo_result is not None else "Evaluation failed")

    logger.info(f"Starting {len(models)} parallel Basilica evaluations...")
    tasks = [evaluate_single_repo(repo, idx) for idx, repo in enumerate(models)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    evaluation_results = {}
    for idx, result in enumerate(results):
        repo = models[idx]
        if isinstance(result, Exception):
            logger.error(f"Evaluation for {repo} failed: {result}", exc_info=True)
            evaluation_results[repo] = f"Evaluation failed: {str(result)}"
        else:
            _, result_data = result
            evaluation_results[repo] = result_data

    logger.info(f"Environment evaluation results: {evaluation_results}")
    return process_evaluation_results(evaluation_results, is_image=False)


async def run_evaluation_local_environment(
    models: list[str],
    original_model: str,
    dataset_type: EnvironmentDatasetType,
    gpu_id: int = 0,
    eval_seed: int | None = None,
) -> DockerEvaluationResults:
    """Run environment evaluation using local Docker containers.

    Simple single-GPU sequential setup for local testing and development.
    Starts a local SGLang and environment container per repo, evaluates, then cleans up.
    """
    logger.info(f"Starting local Docker environment evaluation for {len(models)} repos: {models}")

    env_name = dataset_type.environment_name
    if env_name not in vcst.ENVIRONMENTS:
        raise ValueError(f"Environment '{env_name}' not found. Supported: {list(vcst.ENVIRONMENTS.keys())}")

    env_config = vcst.ENVIRONMENTS[env_name]
    task_id_min, task_id_max = env_config["task_id_range"]
    env_image = env_config["env_image"]
    env_payload_extra = env_config.get("eval_payload_extra", {})

    base_seed = eval_seed if eval_seed is not None else vcst.ENV_EVAL_DEFAULT_SEED
    seed_generator = random.Random(base_seed)
    eval_seeds = [seed_generator.randint(1, 1000000) for _ in range(vcst.ENV_EVAL_NUM_SEEDS)]
    logger.info(f"Generated {vcst.ENV_EVAL_NUM_SEEDS} seeds from base_seed={base_seed}")

    docker_client = docker.from_env()

    try:
        networks = docker_client.networks.list(names=[vcst.LOCAL_ENV_DOCKER_NETWORK])
        if not networks:
            docker_client.networks.create(vcst.LOCAL_ENV_DOCKER_NETWORK, driver="bridge")
            logger.info(f"Created Docker network: {vcst.LOCAL_ENV_DOCKER_NETWORK}")
    except Exception as e:
        logger.warning(f"Docker network setup issue: {e}")

    evaluation_results = {}

    for repo in models:
        eval_id = str(uuid.uuid4())
        repo_name = repo.split("/")[-1]
        env_logger = get_environment_logger(
            name=repo_name, repo_id=repo, eval_id=eval_id, model=original_model,
        )

        containers = {}
        lora_dir = None

        try:
            is_lora = await asyncio.to_thread(check_for_lora, repo, local_files_only=False)

            if is_lora:
                base_model = original_model
                inference_model_name = f"{original_model}:trained_lora"
                env_logger.info(f"LoRA detected: {original_model} + LoRA {repo}")
                safe_lora_name = repo.replace("/", "_")
                lora_dir = f"/tmp/sglang_lora/{safe_lora_name}"
                await asyncio.to_thread(
                    snapshot_download, repo_id=repo, local_dir=lora_dir,
                    local_dir_use_symlinks=False, tqdm_class=None,
                )
                # Remove incompatible full model safetensors (keep adapter files only)
                for model_file in glob.glob(os.path.join(lora_dir, "model-*.safetensors")):
                    try:
                        os.remove(model_file)
                        env_logger.info(f"Removed incompatible file: {os.path.basename(model_file)}")
                    except Exception as e:
                        env_logger.warning(f"Failed to remove {model_file}: {e}")
                index_file = os.path.join(lora_dir, "model.safetensors.index.json")
                if os.path.exists(index_file):
                    try:
                        os.remove(index_file)
                    except Exception as e:
                        env_logger.warning(f"Failed to remove index file: {e}")
            else:
                base_model = repo
                inference_model_name = repo
                env_logger.info(f"Base model: {repo}")

            # Build SGLang launch command
            sglang_args = (
                f"python3 -m sglang.launch_server --model-path {base_model} "
                f"--host 0.0.0.0 --port {vcst.LOCAL_ENV_SGLANG_PORT} "
                f"--tensor-parallel-size 1 --dtype float16 "
                f"--enable-deterministic-inference --random-seed {base_seed}"
            )
            if is_lora:
                sglang_args = (
                    f"python3 -m sglang.launch_server --model-path {base_model} "
                    f"--enable-lora --lora-paths trained_lora=/lora/trained_lora --lora-backend triton "
                    f"--host 0.0.0.0 --port {vcst.LOCAL_ENV_SGLANG_PORT} "
                    f"--tensor-parallel-size 1 --dtype float16 "
                    f"--enable-deterministic-inference --random-seed {base_seed}"
                )

            sglang_container_name = f"{eval_id}-sglang"
            env_container_name = f"{eval_id}-env"

            # Prepare SGLang volumes
            sglang_volumes = {vcst.LOCAL_ENV_HF_CACHE_PATH: {"bind": "/hf", "mode": "rw"}}
            if is_lora and lora_dir:
                sglang_volumes[lora_dir] = {"bind": "/lora/trained_lora", "mode": "ro"}

            # Start SGLang container
            env_logger.info(f"Starting SGLang container: {sglang_container_name} (GPU {gpu_id})")
            sglang_container = await asyncio.to_thread(
                docker_client.containers.run,
                vcst.BASILICA_SGLANG_IMAGE,
                command=sglang_args,
                name=sglang_container_name,
                detach=True,
                network=vcst.LOCAL_ENV_DOCKER_NETWORK,
                ports={f"{vcst.LOCAL_ENV_SGLANG_PORT}/tcp": vcst.LOCAL_ENV_SGLANG_PORT},
                device_requests=[docker.types.DeviceRequest(device_ids=[str(gpu_id)], capabilities=[["gpu"]])],
                environment={
                    "HF_HOME": "/hf",
                    "TRANSFORMERS_CACHE": "/hf",
                    "HUGGINGFACE_HUB_CACHE": "/hf",
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "PYTHONHASHSEED": str(base_seed),
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "NVIDIA_TF32_OVERRIDE": "0",
                },
                volumes=sglang_volumes,
                ipc_mode="host",
                remove=False,
            )
            containers["sglang"] = sglang_container

            sglang_host_url = f"http://localhost:{vcst.LOCAL_ENV_SGLANG_PORT}"
            await asyncio.to_thread(
                wait_for_basilica_health, sglang_host_url, vcst.LOCAL_ENV_SGLANG_HEALTH_TIMEOUT,
            )
            env_logger.info(f"SGLang ready at {sglang_host_url}")

            # Start environment container
            env_logger.info(f"Starting environment container: {env_container_name}")
            env_container = await asyncio.to_thread(
                docker_client.containers.run,
                env_image,
                name=env_container_name,
                detach=True,
                network=vcst.LOCAL_ENV_DOCKER_NETWORK,
                ports={"8000/tcp": vcst.LOCAL_ENV_SERVER_PORT},
                remove=False,
            )
            containers["env"] = env_container

            env_host_url = f"http://localhost:{vcst.LOCAL_ENV_SERVER_PORT}"
            await asyncio.to_thread(
                wait_for_basilica_health, env_host_url, vcst.LOCAL_ENV_SERVER_HEALTH_TIMEOUT, "/health",
            )
            env_logger.info(f"Environment server ready at {env_host_url}")

            sglang_internal_url = f"http://{sglang_container_name}:{vcst.LOCAL_ENV_SGLANG_PORT}"

            avg_score = await _run_environment_evaluation(
                sglang_internal_url,
                env_host_url,
                eval_seeds,
                task_id_max,
                vcst.ENV_EVAL_TEMPERATURE,
                env_logger,
                inference_model_name,
                task_id_min,
                env_payload_extra=env_payload_extra,
            )

            evaluation_results[repo] = {"is_finetune": True, "eval_loss": avg_score}

        except Exception as e:
            env_logger.error(f"Evaluation failed for {repo}: {e}", exc_info=True)
            evaluation_results[repo] = f"Evaluation failed: {str(e)}"

        finally:
            for name, container in containers.items():
                try:
                    container.remove(force=True)
                    env_logger.info(f"Cleaned up {name} container")
                except Exception as e:
                    env_logger.warning(f"Failed to cleanup {name}: {e}")
            if lora_dir and os.path.exists(lora_dir):
                try:
                    shutil.rmtree(lora_dir)
                except Exception as e:
                    env_logger.warning(f"Failed to cleanup LoRA dir: {e}")

    docker_client.close()
    logger.info(f"Local environment evaluation results: {evaluation_results}")
    return process_evaluation_results(evaluation_results, is_image=False)


async def _run_environment_evaluation(
    sglang_url: str,
    env_url: str,
    eval_seeds: list[int],
    data_len_range: int,
    temperature: float,
    env_logger: logging.Logger,
    inference_model_name: str,
    task_id_min: int = 0,
    env_payload_extra: dict | None = None,
) -> float:
    """Shared evaluation loop for environment tasks.

    Used by both Basilica and local Docker evaluation functions.
    For each seed, picks one random task_id and evaluates it concurrently.

    Returns:
        Average score across all successful evaluations.
    """
    eval_list = []
    for seed in eval_seeds:
        rng = random.Random(seed)
        task_id = rng.randint(task_id_min + 1, data_len_range)
        eval_list.append((seed, task_id))

    num_eval_samples = len(eval_list)
    all_results = []

    async def evaluate_single_task(
        session: aiohttp.ClientSession, seed: int, task_id: int, task_idx: int,
    ) -> dict | None:
        """Evaluate a single task with infinite retry."""
        payload = {
            "model": inference_model_name,
            "base_url": f"{sglang_url}/v1",
            "task_id": task_id,
            "temperature": temperature,
            "seed": seed,
        }
        if env_payload_extra:
            payload.update(env_payload_extra)
        if env_name == "goofspiel":
            payload["opponent"] = "random"
            payload["api_key"] = "dummy-key"
        elif env_name == "gin_rummy":
            # TODO Ensure basilica uses correct image phoenixbeaudry/game:mcts-api
            payload["opponent"] = "mcts"
            payload["mcts_max_simulations"] = 25
            payload["mcts_num_rollouts"] = 1
            payload["api_key"] = "dummy-key"
        else:
            payload["max_round"] = 30
        
        last_error = None

        attempt = 0
        while True:
            attempt += 1
            start_ts = time.time()
            try:
                env_logger.info(f"[{task_idx + 1}/{num_eval_samples}] Seed: {seed}, Task ID: {task_id}...")

                timeout = aiohttp.ClientTimeout(total=vcst.ENV_EVAL_TASK_TIMEOUT)
                async with session.post(
                    f"{env_url}/evaluate",
                    json=payload,
                    timeout=timeout,
                    headers={"Connection": "close"},
                ) as response:
                    if response.status != 200:
                        try:
                            error_text = await response.text()
                            error_detail = f": {error_text[:200]}" if error_text else ""
                        except Exception:
                            error_detail = ""
                        raise Exception(f"HTTP {response.status}{error_detail}")

                    response_data = await response.json()
                    result = response_data.get("result", response_data)
                    latency = result.get("time_taken", time.time() - start_ts)
                    score = result.get("score", 0.0)

                    if attempt > 1:
                        env_logger.info(f"Task ID {task_id}: Done (Score: {score}) - after {attempt - 1} retries")
                    else:
                        env_logger.info(f"Task ID {task_id}: Done (Score: {score})")

                    return {"task_id": task_id, "score": score, "time": latency}

            except Exception as e:
                env_logger.warning(
                    f"Task ID {task_id}: Error (retry {attempt} in {vcst.ENV_EVAL_TASK_RETRY_DELAY:.0f}s): {e}"
                )
                await asyncio.sleep(vcst.ENV_EVAL_TASK_RETRY_DELAY)

    semaphore = asyncio.Semaphore(vcst.ENV_EVAL_MAX_CONCURRENT_REQUESTS)

    async def evaluate_with_semaphore(
        session: aiohttp.ClientSession, seed: int, task_id: int, task_idx: int,
    ) -> dict | None:
        async with semaphore:
            return await evaluate_single_task(session, seed, task_id, task_idx)

    session_timeout = aiohttp.ClientTimeout(total=vcst.ENV_EVAL_SESSION_TIMEOUT)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        env_logger.info(
            f"Starting {num_eval_samples} evaluations with concurrency={vcst.ENV_EVAL_MAX_CONCURRENT_REQUESTS}..."
        )

        tasks = [
            evaluate_with_semaphore(session, seed, task_id, idx)
            for idx, (seed, task_id) in enumerate(eval_list)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                seed, task_id = eval_list[idx]
                env_logger.error(f"Seed {seed}, Task {task_id}: Failed with exception: {result}")
            elif result is not None:
                all_results.append(result)

    total_score = sum(r.get("score", 0.0) for r in all_results)
    total_time = sum(r.get("time", 0.0) for r in all_results)
    avg_score = total_score / len(all_results) if all_results else 0.0
    avg_time = total_time / len(all_results) if all_results else 0.0

    env_logger.info(
        f"Summary: {len(all_results)}/{len(eval_list)} successful, "
        f"Avg Score: {avg_score:.4f}, Avg Time: {avg_time:.2f}s"
    )

    return avg_score


async def run_evaluation_docker_image(
    test_split_url: str,
    original_model_repo: str,
    models: list[str],
    model_type: ImageModelType,
    gpu_ids: list[int]
) -> DockerEvaluationResults:
    raw_data = await download_s3_file(test_split_url)
    test_split_path = unzip_to_temp_path(raw_data)
    dataset_dir = os.path.abspath(test_split_path)
    container_dataset_path = "/workspace/input_data"

    client = docker.from_env()

    base_path = "/app/validator/evaluation/ComfyUI/models"
    mounts = [
        Mount(
            target=container_dataset_path,
            source=dataset_dir,
            type='bind',
            read_only=True
        ),
        Mount(
            target=f"{base_path}/checkpoints",
            source=cst.CACHE_DIR_HUB,
            type='bind',
            read_only=False
        ),
        Mount(
            target=f"{base_path}/diffusers",
            source=cst.CACHE_DIR_HUB,
            type='bind',
            read_only=False
        )
    ]

    environment = {
        "DATASET": container_dataset_path,
        "MODELS": ",".join(models),
        "ORIGINAL_MODEL_REPO": original_model_repo,
        "MODEL_TYPE": model_type.value,
        "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
    }

    container = None
    retry_delay = 5.0
    
    try:
        while True:
            try:
                container = await asyncio.to_thread(
                    client.containers.run,
                    cst.VALIDATOR_DOCKER_IMAGE_DIFFUSION,
                    mounts=mounts,
                    environment=environment,
                    runtime="nvidia",
                    device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gid) for gid in gpu_ids])],
                    detach=True,
                )
                log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, None, get_all_context_tags()))
                result = await asyncio.to_thread(container.wait)
                log_task.cancel()

                if result["StatusCode"] != 0:
                    raise Exception(f"Container exited with status {result['StatusCode']}")

                eval_results_dict = await get_evaluation_results(container)
                return process_evaluation_results(eval_results_dict, is_image=True)

            except Exception as e:
                logger.error(f"Failed to retrieve evaluation results: {str(e)}, retrying in {retry_delay}s...")
                if container is not None:
                    try:
                        await asyncio.to_thread(container.remove, force=True)
                        container = None
                    except:
                        pass
                await asyncio.sleep(retry_delay)
                continue

    finally:
        try:
            if container is not None:
                await asyncio.to_thread(container.remove, force=True)
            await cleanup_resources(client)
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
        except Exception as e:
            logger.info(f"A problem with cleaning up {e}")
        client.close()
