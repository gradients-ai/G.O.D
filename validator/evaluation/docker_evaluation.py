import asyncio
import io
import json
import os
import shutil
import tarfile
from datetime import datetime
from typing import Optional

import docker
from docker.models.containers import Container
from docker.types import Mount
from huggingface_hub import snapshot_download
import requests
import time
import random
import basilica
from requests.adapters import HTTPAdapter

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
from validator.utils.logging import stream_container_logs
from validator.evaluation.utils import (
    deploy_vllm_basilica,
    deploy_agentgym_basilica,
    wait_for_basilica_health,
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
) -> DockerEvaluationResults:

    if isinstance(dataset_type, (InstructTextDatasetType, ChatTemplateDatasetType)):
        command = ["python", "-m", "validator.evaluation.eval_instruct_text"]
    elif isinstance(dataset_type, DpoDatasetType):
        command = ["python", "-m", "validator.evaluation.eval_dpo"]
    elif isinstance(dataset_type, GrpoDatasetType):
        return await run_evaluation_docker_grpo(dataset, models, original_model, dataset_type, file_format, gpu_ids)
    elif isinstance(dataset_type, EnvironmentDatasetType):
        return await run_evaluation_docker_environment(dataset, models, original_model, dataset_type, file_format, gpu_ids, num_eval_samples=250)
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
        logger.error(f"Failed to retrieve {task_type} evaluation results: {str(e)}", exc_info=True)
        raise Exception(f"Failed to retrieve {task_type} evaluation results: {str(e)}")

    finally:
        try:
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
        try:
            model_path = await asyncio.to_thread(
                snapshot_download,
                repo_id=repo,
                cache_dir=cache_dir,
                ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.pkl", "*.pth"]
            )

        except Exception as e:
            logger.error(f"Failed to download {repo}: {str(e)}")
            evaluation_results[repo] = f"Failed to download model: {str(e)}"
            continue

        container = None  # Initialize container variable
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

                logger.error(f"Container for {repo} exited with non-zero status: {result['StatusCode']}")
                evaluation_results[repo] = f"Container for {repo} exited with status {result['StatusCode']}"

            else:
                eval_results = await get_evaluation_results(container)
                evaluation_results[repo] = eval_results[repo]
                if "model_params_count" in eval_results and "model_params_count" not in evaluation_results:
                    evaluation_results["model_params_count"] = eval_results["model_params_count"]

        except Exception as e:
            logger.error(f"Failed to evaluate repo {repo}: {str(e)}", exc_info=True)
            evaluation_results[repo] = str(e)

        finally:
            try:
                if container is not None:
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
    num_eval_samples: int,
) -> DockerEvaluationResults:
    """
    Run environment evaluation using Basilica deployments for vLLM and AgentGym.
    Each model repo gets its own deployments with separate logging and retry logic.
    """
    logger.info(f"Starting Basilica-based environment evaluation for {len(models)} repos: {models}")

    # Evaluation configuration
    DATA_LEN_RANGE = 2500
    RANDOM_SEED = 42
    TEMPERATURE = 0.0
    MAX_RETRIES_PER_MODEL = 3  # Retry entire evaluation if deployment fails
    
    async def evaluate_single_repo(repo: str, repo_idx: int) -> tuple[str, dict | str]:
        """Evaluate a single repo and return (repo, result)."""
        log_prefix = f"[Repo-{repo_idx}] "
        # Create logs directory if it doesn't exist
        logs_dir = "logs/basilica_eval"
        os.makedirs(logs_dir, exist_ok=True)
        safe_repo_name = repo.split('/')[-1]
        # Use timestamp and repo_idx to ensure unique log files even with duplicate repos
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
        log_file = os.path.join(logs_dir, f"basilica_eval_{safe_repo_name}_{repo_idx}_{timestamp}.log")
        
        # Initialize log file
        with open(log_file, "w") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Basilica Environment Evaluation Log\n")
            f.write(f"Repo: {repo}\n")
            f.write(f"Repo Index: {repo_idx}\n")
            f.write(f"Base Model: {original_model}\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write(f"{'='*60}\n\n")
        
        deployments = {}
        success = False
        repo_result = None
        
        # Retry logic for entire evaluation
        for retry_attempt in range(MAX_RETRIES_PER_MODEL):
            try:
                with open(log_file, "a") as f:
                    f.write(f"{log_prefix}Attempt {retry_attempt + 1}/{MAX_RETRIES_PER_MODEL}\n")
                
                # Create unique deployment names using repo_idx and timestamp to avoid conflicts
                # Even if same repo appears multiple times, each gets unique deployment
                safe_repo_name = repo.split("/")[-1][:30]  # Shorter to leave room for suffix
                unique_suffix = f"{repo_idx}-{int(time.time() * 1000) % 100000}"  # repo_idx + timestamp ms
                vllm_deployment_name = f"vllm-{safe_repo_name}-{unique_suffix}"
                agentgym_deployment_name = f"agentgym-{safe_repo_name}-{unique_suffix}"
                
                # Deploy vLLM
                logger.info(f"{log_prefix}Deploying vLLM: {original_model} w/ LoRA {repo}")
                with open(log_file, "a") as f:
                    f.write(f"{log_prefix}Deploying vLLM...\n")
                
                vllm_deployment = await asyncio.to_thread(
                    deploy_vllm_basilica,
                    original_model,
                    repo,
                    vllm_deployment_name,
                    log_file
                )
                deployments['vllm'] = vllm_deployment
                
                # Wait for vLLM health
                await asyncio.to_thread(wait_for_basilica_health, vllm_deployment.url, log_file=log_file)
                logger.info(f"{log_prefix}vLLM Ready at: {vllm_deployment.url}")
                
                # Deploy AgentGym
                logger.info(f"{log_prefix}Deploying AgentGym...")
                with open(log_file, "a") as f:
                    f.write(f"{log_prefix}Deploying AgentGym...\n")
                
                agentgym_deployment = await asyncio.to_thread(
                    deploy_agentgym_basilica,
                    agentgym_deployment_name,
                    log_file
                )
                deployments['agentgym'] = agentgym_deployment
                
                # Wait for AgentGym health
                try:
                    await asyncio.to_thread(wait_for_basilica_health, agentgym_deployment.url, path="/health", log_file=log_file)
                except:
                    await asyncio.to_thread(wait_for_basilica_health, agentgym_deployment.url, path="/v1/models", log_file=log_file)
                logger.info(f"{log_prefix}AgentGym Ready at: {agentgym_deployment.url}")
                
                # Run evaluation
                avg_score = await _run_basilica_evaluation(
                    vllm_deployment.url,
                    agentgym_deployment.url,
                    num_eval_samples,
                    DATA_LEN_RANGE,
                    RANDOM_SEED,
                    TEMPERATURE,
                    log_prefix,
                    log_file
                )
                
                repo_result = {
                    'is_finetune': True,
                    'eval_loss': avg_score
                }
                
                with open(log_file, "a") as f:
                    f.write(f"{log_prefix}✅ Evaluation completed successfully. Average score: {avg_score:.4f}\n")
                
                success = True
                break  # Success, exit retry loop
                
            except Exception as e:
                error_msg = f"{log_prefix}Evaluation attempt {retry_attempt + 1} failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                with open(log_file, "a") as f:
                    f.write(f"{error_msg}\n")
                    import traceback
                    f.write(traceback.format_exc() + "\n")
                
                # Cleanup deployments on failure
                for name, deployment in deployments.items():
                    try:
                        deployment.delete()
                        with open(log_file, "a") as f:
                            f.write(f"{log_prefix}Cleaned up {name} deployment\n")
                    except Exception as cleanup_error:
                        with open(log_file, "a") as f:
                            f.write(f"{log_prefix}Failed to cleanup {name}: {cleanup_error}\n")
                deployments = {}
                
                if retry_attempt < MAX_RETRIES_PER_MODEL - 1:
                    wait_time = 5 * (2 ** retry_attempt)
                    with open(log_file, "a") as f:
                        f.write(f"{log_prefix}Retrying in {wait_time}s...\n")
                    await asyncio.sleep(wait_time)
                else:
                    # Final failure after all retries
                    repo_result = str(e)
                    with open(log_file, "a") as f:
                        f.write(f"{log_prefix}❌ All retry attempts exhausted\n")
            
            finally:
                # Cleanup deployments
                for name, deployment in deployments.items():
                    try:
                        deployment.delete()
                        logger.info(f"{log_prefix}Cleaned up {name} deployment")
                    except Exception as e:
                        logger.warning(f"{log_prefix}Failed to cleanup {name}: {e}")
        
        if success:
            logger.info(f"{log_prefix}✅ Evaluation completed. Log saved to: {log_file}")
        else:
            logger.error(f"{log_prefix}❌ Evaluation failed after {MAX_RETRIES_PER_MODEL} attempts. Log saved to: {log_file}")
        
        # Return result (or error string if failed)
        return (repo, repo_result if repo_result is not None else "Evaluation failed")
    
    # Run all evaluations in parallel
    logger.info(f"🚀 Starting {len(models)} parallel evaluations...")
    tasks = [evaluate_single_repo(repo, idx) for idx, repo in enumerate(models)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect results
    evaluation_results = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Evaluation task failed with exception: {result}", exc_info=True)
            # Can't identify which repo failed from exception alone
            continue
        repo, result_data = result
        evaluation_results[repo] = result_data

    logger.info(f"Environment evaluation results: {evaluation_results}")
    return process_evaluation_results(evaluation_results, is_image=False)


async def _run_basilica_evaluation(
    vllm_url: str,
    agentgym_url: str,
    num_eval_samples: int,
    data_len_range: int,
    random_seed: int,
    temperature: float,
    log_prefix: str,
    log_file: str
) -> float:
    """Run evaluation loop using Basilica deployments."""
    # Create session with connection pooling disabled
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=1,
        pool_maxsize=1,
        max_retries=0
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Evaluation loop
    random.seed(random_seed)
    eval_list = random.sample(range(1, data_len_range + 1), num_eval_samples)
    total_score = 0.0
    total_time = 0.0
    all_results = []
    
    max_retries = 5
    retry_delay = 2.0
    
    for i, task_id in enumerate(eval_list):
        with open(log_file, "a") as f:
            f.write(f"{log_prefix}[{i+1}/{num_eval_samples}] Task ID: {task_id}...\n")
        
        payload = {
            "model": "trained_lora",
            "base_url": f"{vllm_url}/v1",
            "task_id": task_id,
            "temperature": temperature,
            "max_round": 30
        }
        
        # Retry logic for individual task
        for attempt in range(max_retries):
            try:
                start_ts = time.time()
                response = session.post(
                    f"{agentgym_url}/evaluate",
                    json=payload,
                    timeout=2500,
                    headers={'Connection': 'close'}
                )
                
                # Check response status
                if response.status_code != 200:
                    if response.status_code >= 500 or response.status_code == 503:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)
                            with open(log_file, "a") as f:
                                f.write(f"{log_prefix}HTTP {response.status_code} (retry {attempt + 1}/{max_retries} in {wait_time:.1f}s)...\n")
                            await asyncio.sleep(wait_time)
                            continue
                    
                    error_msg = f"HTTP {response.status_code}"
                    with open(log_file, "a") as f:
                        f.write(f"{log_prefix}Failed: {error_msg}\n")
                    all_results.append({
                        "task_id": task_id,
                        "score": 0.0,
                        "time": time.time() - start_ts,
                        "error": error_msg
                    })
                    break
                
                # Parse JSON
                try:
                    result = response.json()
                except ValueError as e:
                    error_msg = f"Invalid JSON: {e}"
                    with open(log_file, "a") as f:
                        f.write(f"{log_prefix}Failed: {error_msg}\n")
                    all_results.append({
                        "task_id": task_id,
                        "score": 0.0,
                        "time": time.time() - start_ts,
                        "error": error_msg
                    })
                    break
                
                latency = result.get('time_taken', time.time() - start_ts)
                score = result.get('score', 0.0)
                
                total_score += score
                total_time += latency
                
                all_results.append({
                    "task_id": task_id,
                    "score": score,
                    "time": latency
                })
                
                with open(log_file, "a") as f:
                    f.write(f"{log_prefix}Done (Score: {score})\n")
                break  # Success
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    with open(log_file, "a") as f:
                        f.write(f"{log_prefix}Connection error (retry {attempt + 1}/{max_retries} in {wait_time:.1f}s)...\n")
                    await asyncio.sleep(wait_time)
                    continue
                
                with open(log_file, "a") as f:
                    f.write(f"{log_prefix}Failed: {str(e)}\n")
                all_results.append({
                    "task_id": task_id,
                    "score": 0.0,
                    "time": 2500.0 if isinstance(e, requests.exceptions.Timeout) else time.time() - start_ts,
                    "error": str(e)
                })
                break
            except Exception as e:
                with open(log_file, "a") as f:
                    f.write(f"{log_prefix}Failed: {str(e)}\n")
                all_results.append({
                    "task_id": task_id,
                    "score": 0.0,
                    "time": time.time() - start_ts if 'start_ts' in locals() else 0.0,
                    "error": str(e)
                })
                break
        
        # Small delay between evaluations
        if i < len(eval_list) - 1:
            await asyncio.sleep(1.0)
    
    session.close()
    
    # Calculate average score
    avg_score = total_score / len(all_results) if all_results else 0.0
    avg_time = total_time / len(all_results) if all_results else 0.0
    
    with open(log_file, "a") as f:
        f.write(f"\n{log_prefix}Summary:\n")
        f.write(f"  Total Tasks: {len(all_results)}\n")
        f.write(f"  Average Score: {avg_score:.4f}\n")
        f.write(f"  Average Time: {avg_time:.2f}s\n")
    
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
        logger.error(f"Failed to retrieve evaluation results: {str(e)}")
        raise Exception(f"Failed to retrieve evaluation results: {str(e)}")

    finally:
        try:
            await asyncio.to_thread(container.remove, force=True)
            await cleanup_resources(client)
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
        except Exception as e:
            logger.info(f"A problem with cleaning up {e}")
        client.close()
