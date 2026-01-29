import asyncio
import io
import json
import logging
import os
import shutil
import tarfile
from datetime import datetime
from typing import Optional

import docker
from docker.models.containers import Container
from docker.types import Mount
from huggingface_hub import snapshot_download
import aiohttp
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
    """
    Run environment evaluation using Basilica deployments for vLLM and AgentGym.
    Each model repo gets its own deployments with separate logging and retry logic.
    
    Args:
        eval_seed: Random seed for evaluation reproducibility. If None, falls back to 42.
    """
    logger.info(f"Starting Basilica-based environment evaluation for {len(models)} repos: {models}")

    env_name = dataset_type.environment_name
    if env_name not in vcst.ENVIRONMENTS:
        raise ValueError(f"Environment '{env_name}' not found in ENVIRONMENTS. Supported environments: {list(vcst.ENVIRONMENTS.keys())}")
    
    env_config = vcst.ENVIRONMENTS[env_name]
    task_id_range = env_config["task_id_range"]
    env_image = env_config["env_image"]
    
    task_id_min, task_id_max = task_id_range
    DATA_LEN_RANGE = task_id_max
    TASK_ID_MIN = task_id_min
    
    RANDOM_SEED = eval_seed if eval_seed is not None else 42
    TEMPERATURE = 0.0
    logger.info(f"Using eval_seed={RANDOM_SEED} for environment evaluation")
    retry_delay = 5.0  # for individual task retries
    eval_retry_delay = 300.0  # for evaluation retries (deployment failures)
    
    async def evaluate_single_repo(repo: str, repo_idx: int) -> tuple[str, dict | str]:
        """Evaluate a single repo and return (repo, result)."""
        eval_id = str(random.randint(1, 1000000))
        repo_name_stripped = repo.split("/")[-1]

        env_logger = get_environment_logger(
            name=repo_name_stripped,
            repo_id=repo,
            eval_id=eval_id,
            model=original_model,
        )
        deployments = {}
        success = False
        repo_result = None
        
        def log_deployment_logs(deployment, deployment_type: str):
            """Fetch and log deployment logs."""
            try:
                logs = deployment.logs()
                if logs:
                    for line in logs.strip().split('\n'):
                        if not line.strip():
                            continue
                        
                        try:
                            log_data = json.loads(line)
                            message = log_data.get("message", "")
                            if message:
                                message = re.sub(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', message)
                                message = re.sub(r'^data:\s*', '', message)
                                message = message.rstrip(', ')
                                if message.strip():
                                    env_logger.info(f"[{deployment_type}] {message}")
                        except (json.JSONDecodeError, AttributeError):
                            cleaned_line = line.strip()
                            if cleaned_line:
                                env_logger.info(f"[{deployment_type}] {cleaned_line}")
            except Exception as e:
                env_logger.warning(f"Failed to fetch {deployment_type} logs: {e}")
        
        async def cleanup_deployments(deployments_dict: dict, fetch_logs: bool = False):
            """Clean up all deployments and optionally fetch their logs first."""
            for name, deployment in deployments_dict.items():
                try:
                    if fetch_logs:
                        env_logger.info(f"Dumping logs for {name} deployment before cleanup...")
                        await asyncio.to_thread(log_deployment_logs, deployment, name)
                        env_logger.info(f"Finished dumping logs for {name} deployment")
                    
                    deployment.delete()
                    env_logger.info(f"Cleaned up {name} deployment")
                except Exception as e:
                    env_logger.warning(f"Failed to cleanup {name}: {e}", exc_info=True)
                    if fetch_logs:
                        try:
                            env_logger.info(f"Attempting to dump logs for {name} after cleanup error...")
                            await asyncio.to_thread(log_deployment_logs, deployment, name)
                        except Exception as log_error:
                            env_logger.warning(f"Failed to dump logs for {name}: {log_error}")
            deployments_dict.clear()
        
        MAX_EVAL_RETRIES = 5
        retry_attempt = 0
        while retry_attempt < MAX_EVAL_RETRIES:
            retry_attempt += 1
            try:
                sglang_deployment_name = f"sglang-{repo_name_stripped}-{eval_id}"
                env_deployment_name = f"agentgym-{repo_name_stripped}-{eval_id}"
                
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
                    env_logger.info(f"Deploying SGLang: {repo} (base model, ignoring original_model={original_model})")
                
                sglang_deployment = await asyncio.to_thread(
                    deploy_sglang_basilica,
                    base_model,
                    lora_model,
                    sglang_deployment_name,
                    RANDOM_SEED,
                )
                deployments['sglang'] = sglang_deployment
                
                await asyncio.to_thread(wait_for_basilica_health, sglang_deployment.url)
                env_logger.info(f"SGLang Ready at: {sglang_deployment.url}")
                
                env_logger.info(f"Deploying Environment Server...")
                
                env_deployment = await asyncio.to_thread(
                    deploy_env_basilica,
                    env_deployment_name,
                    env_image
                )
                deployments['env'] = env_deployment
                
                await asyncio.to_thread(wait_for_basilica_health, env_deployment.url, timeout=300, path="/health")
                env_logger.info(f"Environment Server Ready at: {env_deployment.url}")
                
                avg_score = await _run_basilica_evaluation(
                    sglang_deployment.url,
                    env_deployment.url,
                    vcst.NUM_EVAL_SAMPLES,
                    DATA_LEN_RANGE,
                    RANDOM_SEED,
                    TEMPERATURE,
                    env_logger,
                    inference_model_name,
                    TASK_ID_MIN,
                    env_name=env_name
                )
                
                repo_result = {
                    'is_finetune': True,
                    'eval_loss': avg_score
                }
                
                await asyncio.to_thread(log_deployment_logs, sglang_deployment, "SGLang")
                await asyncio.to_thread(log_deployment_logs, env_deployment, "Env")
                
                await cleanup_deployments(deployments)
                
                success = True
                break
                
            except Exception as e:
                if retry_attempt < MAX_EVAL_RETRIES:
                    env_logger.error(f"Evaluation attempt {retry_attempt}/{MAX_EVAL_RETRIES} failed: {str(e)}, retrying in {eval_retry_delay/60:.1f} minutes...", exc_info=True)
                else:
                    env_logger.error(f"Evaluation attempt {retry_attempt}/{MAX_EVAL_RETRIES} failed: {str(e)}, max retries reached.", exc_info=True)
                
                await cleanup_deployments(deployments, fetch_logs=True)
                if retry_attempt < MAX_EVAL_RETRIES:
                    await asyncio.sleep(eval_retry_delay)
        
        if success:
            env_logger.info(f"Evaluation completed successfully after {retry_attempt} attempt(s).")
        else:
            env_logger.error(f"Evaluation failed after {MAX_EVAL_RETRIES} attempts.")
        
        return (repo, repo_result if repo_result is not None else "Evaluation failed")
    
    logger.info(f"Starting {len(models)} parallel evaluations...")
    tasks = [evaluate_single_repo(repo, idx) for idx, repo in enumerate(models)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    evaluation_results = {}
    for idx, result in enumerate(results):
        repo = models[idx]  # Get repo name from original list by index
        if isinstance(result, Exception):
            logger.error(f"Evaluation task for {repo} failed with exception: {result}", exc_info=True)
            evaluation_results[repo] = f"Evaluation failed: {str(result)}"
        else:
            _, result_data = result
            evaluation_results[repo] = result_data

    logger.info(f"Environment evaluation results: {evaluation_results}")
    return process_evaluation_results(evaluation_results, is_image=False)


async def _run_basilica_evaluation(
    vllm_url: str,
    env_url: str,
    num_eval_samples: int,
    data_len_range: int,
    random_seed: int,
    temperature: float,
    env_logger: logging.Logger,
    inference_model_name: str,
    task_id_min: int = 0,
    env_name: str = "alfworld"
) -> float:
    """Run evaluation loop using Basilica deployments with sequential task processing."""
    random.seed(random_seed)
    eval_list = random.sample(range(task_id_min + 1, data_len_range + 1), num_eval_samples)
    max_retries = 5
    retry_delay = 10.0
    
    all_results = []
    
    async def evaluate_single_task(session: aiohttp.ClientSession, task_id: int, task_idx: int) -> dict:
        """Evaluate a single task with retry logic."""
        payload = {
            "model": inference_model_name,
            "base_url": f"{vllm_url}/v1",
            "task_id": task_id,
            "temperature": temperature,
            "seed": random_seed,
        }
        
        if env_name == "goofspiel":
            payload["opponent"] = "random"
            payload["api_key"] = "dummy-key"
        else:
            payload["max_round"] = 30
        
        last_error = None
        attempt = 0
        
        while True:
            attempt += 1
            start_ts = time.time()
            try:
                env_logger.info(f"[{task_idx+1}/{num_eval_samples}] Task ID: {task_id}...")
                
                timeout = aiohttp.ClientTimeout(total=120)
                async with session.post(
                    f"{env_url}/evaluate",
                    json=payload,
                    timeout=timeout,
                    headers={'Connection': 'close'}
                ) as response:
                    if response.status != 200:
                        try:
                            error_text = await response.text()
                            error_detail = f": {error_text[:200]}" if error_text else ""
                        except:
                            error_detail = ""
                        raise Exception(f"HTTP {response.status}{error_detail}")
                    
                    response_data = await response.json()
                    if 'result' in response_data:
                        result = response_data.get('result', {})
                    else:
                        result = response_data
                    
                    latency = result.get('time_taken', time.time() - start_ts)
                    score = result.get('score', 0.0)
                    
                    if attempt > 1:
                        env_logger.info(f"Task ID {task_id}: Done (Score: {score}) - succeeded after {attempt - 1} retries")
                    else:
                        env_logger.info(f"Task ID {task_id}: Done (Score: {score})")
                    
                    return {
                        "task_id": task_id,
                        "score": score,
                        "time": latency
                    }
                    
            except Exception as e:
                last_error = str(e)
                env_logger.warning(f"Task ID {task_id}: Error (retry {attempt} in {retry_delay:.1f}s): {last_error}")
                await asyncio.sleep(retry_delay)
                continue
    
    # Concurrency settings
    max_concurrent = 4 
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(session: aiohttp.ClientSession, task_id: int, task_idx: int) -> dict:
        async with semaphore:
            return await evaluate_single_task(session, task_id, task_idx)
    
    session_timeout = aiohttp.ClientTimeout(total=7200)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        env_logger.info(f"Starting {len(eval_list)} evaluations with concurrency={max_concurrent}...")
        
        tasks = [
            evaluate_with_semaphore(session, task_id, idx) 
            for idx, task_id in enumerate(eval_list)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                env_logger.error(f"Task {eval_list[idx]}: Failed with exception: {result}")
            else:
                all_results.append(result)
    
    total_score = sum(r.get('score', 0.0) for r in all_results)
    total_time = sum(r.get('time', 0.0) for r in all_results)
    avg_score = total_score / len(all_results) if all_results else 0.0
    avg_time = total_time / len(all_results) if all_results else 0.0
    
    successful_tasks = len(all_results)
    total_attempted = len(eval_list)
    env_logger.info(f"Summary: Successful Tasks: {successful_tasks}/{total_attempted}, Average Score: {avg_score:.4f}, Average Time: {avg_time:.2f}s")
    
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
