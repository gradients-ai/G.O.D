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
    """
    Run environment evaluation using local Docker containers for vLLM and AgentGym.
    Each model repo gets its own containers with separate logging and retry logic.
    
    Args:
        eval_seed: Random seed for evaluation reproducibility. If None, falls back to 42.
    """
    logger.info(f"Starting local Docker-based environment evaluation for {len(models)} repos: {models}")

    env_name = dataset_type.environment_name
    if env_name not in vcst.ENVIRONMENTS:
        raise ValueError(f"Environment '{env_name}' not found in ENVIRONMENTS. Supported environments: {list(vcst.ENVIRONMENTS.keys())}")
    
    env_config = vcst.ENVIRONMENTS[env_name]
    task_id_range = env_config["task_id_range"]
    env_image = env_config["env_image"]
    
    task_id_min, task_id_max = task_id_range
    DATA_LEN_RANGE = task_id_max
    TASK_ID_MIN = task_id_min
    
    # Generate 2000 random seeds at the start of evaluation (to be used by every repo)
    # Use eval_seed for reproducibility so same eval_seed always generates same task sequence
    seed_generator = random.Random(eval_seed if eval_seed is not None else 42)
    EVAL_SEEDS = [seed_generator.randint(1, 1000000) for _ in range(2000)]
    logger.info(f"Generated 2000 random seeds for evaluation (eval_seed={eval_seed}): first 10={EVAL_SEEDS[:10]}")
    TEMPERATURE = 0.0
    retry_delay = 5.0  # for individual task retries
    eval_retry_delay = 300.0  # for evaluation retries (deployment failures)
    
    # Initialize Docker client and network
    docker_client = docker.from_env()
    NETWORK_NAME = "agent_eval_net"
    
    # Create network if it doesn't exist
    try:
        networks = docker_client.networks.list(names=[NETWORK_NAME])
        if not networks:
            docker_client.networks.create(NETWORK_NAME, driver="bridge")
            logger.info(f"Created Docker network: {NETWORK_NAME}")
    except Exception as e:
        logger.warning(f"Network setup issue (may already exist): {e}")
    
    # Clean up any existing containers that might be using our ports
    async def cleanup_existing_containers():
        """Clean up any existing stopped/exited sglang or agentgym containers."""
        try:
            all_containers = docker_client.containers.list(all=True)
            for container in all_containers:
                try:
                    if container.name.startswith(("sglang-", "agentgym-")):
                        # Only clean up stopped/exited containers, not running ones
                        container_status = container.status.lower()
                        if container_status in ['exited', 'stopped', 'created']:
                            logger.info(f"Cleaning up existing stopped container: {container.name} (status: {container_status})")
                            container.remove(force=True)
                        else:
                            logger.info(f"Skipping running container: {container.name} (status: {container_status})")
                except Exception as e:
                    logger.warning(f"Failed to remove container {container.name}: {e}")
        except Exception as e:
            logger.warning(f"Error during container cleanup: {e}")
    
    await asyncio.to_thread(cleanup_existing_containers)
    
    # GPU and port configuration for concurrent evaluations
    NUM_GPUS = 1  # Hardcoded to 1 GPU (one evaluation at a time)
    SGLANG_BASE_PORT = 30000
    ENV_BASE_PORT = 8001
    
    # Semaphore to limit concurrent evaluations to number of GPUs
    eval_semaphore = asyncio.Semaphore(NUM_GPUS)
    gpu_available = asyncio.Queue()  # Queue to track available GPU slots
    for gpu_id in range(NUM_GPUS):
        await gpu_available.put(gpu_id)  # Initialize with all GPU slots available
    
    async def evaluate_single_repo(repo: str, repo_idx: int, gpu_id: int) -> tuple[str, dict | str]:
        """Evaluate a single repo 2000 times with different seeds and return (repo, result)."""
        eval_id = str(uuid.uuid4())[:8]
        repo_name_stripped = repo.split("/")[-1]

        env_logger = get_environment_logger(
            name=repo_name_stripped,
            repo_id=repo,
            eval_id=eval_id,
            model=original_model,
        )
        
        # Allocate ports based on GPU ID (each GPU gets its own port range)
        sglang_port = SGLANG_BASE_PORT + gpu_id
        env_port = ENV_BASE_PORT + gpu_id
        
        # Generate unique container names using UUID and GPU ID (same names for all seeds)
        sglang_uuid = str(uuid.uuid4())[:8]
        env_uuid = str(uuid.uuid4())[:8]
        sglang_container_name = f"sglang-gpu{gpu_id}-{sglang_uuid}"
        env_container_name = f"agentgym-gpu{gpu_id}-{env_uuid}"
        
        containers = {}
        
        async def cleanup_containers(containers_dict: dict):
            """Clean up all containers."""
            for name, container in containers_dict.items():
                try:
                    container.remove(force=True)
                    env_logger.info(f"Cleaned up {name} container")
                except Exception as e:
                    env_logger.warning(f"Failed to cleanup {name}: {e}", exc_info=True)
            containers_dict.clear()
        
        # Check if repo is LoRA (only need to check once)
        is_lora = await asyncio.to_thread(check_for_lora, repo, local_files_only=False)
        lora_dir = None
        if is_lora:
            base_model = original_model
            lora_model = repo
            inference_model_name = f"{original_model}:trained_lora"
            env_logger.info(f"Repo uses LoRA: {original_model} w/ LoRA {repo}")
            
            # Download LoRA (only once)
            safe_lora_name = lora_model.replace("/", "_")
            lora_dir = f"/tmp/sglang_lora/{safe_lora_name}"
            await asyncio.to_thread(
                snapshot_download,
                repo_id=lora_model,
                local_dir=lora_dir,
                local_dir_use_symlinks=False,
                tqdm_class=None,
            )
            # Remove incompatible model safetensors files
            model_files = glob.glob(os.path.join(lora_dir, "model-*.safetensors"))
            for model_file in model_files:
                try:
                    os.remove(model_file)
                    env_logger.info(f"Removed incompatible LoRA file: {os.path.basename(model_file)}")
                except Exception as e:
                    env_logger.warning(f"Failed to remove {model_file}: {e}")
            index_file = os.path.join(lora_dir, "model.safetensors.index.json")
            if os.path.exists(index_file):
                try:
                    os.remove(index_file)
                    env_logger.info("Removed model.safetensors.index.json")
                except Exception as e:
                    env_logger.warning(f"Failed to remove index file: {e}")
        else:
            base_model = repo
            lora_model = None
            inference_model_name = repo
            env_logger.info(f"Repo is base model: {repo}")
        
        # Create containers once before the seed loop (use first seed for container initialization)
        INIT_SEED = EVAL_SEEDS[0] if EVAL_SEEDS else 42
        
        def wait_for_local_health(url: str, timeout: int = 600, path: str = "/v1/models") -> bool:
            """Wait for local service to be healthy."""
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(f"{url}{path}", timeout=5)
                    if response.status_code == 200:
                        return True
                except:
                    pass
                time.sleep(5)
            
            error_msg = f"Service at {url} did not become healthy within {timeout} seconds"
            raise TimeoutError(error_msg)
        
        # Build SGLang command (use INIT_SEED for container startup)
        if is_lora:
            sglang_command = (
                f"python3 -m sglang.launch_server --model-path {base_model} "
                "--enable-lora --lora-paths trained_lora=/lora/trained_lora "
                "--lora-backend triton "
                f"--host 0.0.0.0 --port 30000 --tensor-parallel-size 1 --dtype float16 --enable-deterministic-inference "
                f"--random-seed {INIT_SEED}"
            )
        else:
            sglang_command = (
                f"python3 -m sglang.launch_server --model-path {base_model} "
                f"--host 0.0.0.0 --port 30000 --tensor-parallel-size 1 --dtype float16 --enable-deterministic-inference "
                f"--random-seed {INIT_SEED}"
            )
        
        env_logger.info(f"Creating containers for repo {repo} (will be reused for all 2000 seeds)")
        env_logger.info(f"SGLang container name: {sglang_container_name}, GPU: {gpu_id}, port: {sglang_port}")
        
        # Check if container name already exists and remove it
        try:
            existing = docker_client.containers.get(sglang_container_name)
            env_logger.warning(f"Container {sglang_container_name} already exists, removing it...")
            existing.remove(force=True)
            await asyncio.sleep(1)
        except docker.errors.NotFound:
            pass
        except Exception as e:
            env_logger.warning(f"Error checking for existing container {sglang_container_name}: {e}")
        
        # Start SGLang container
        try:
            sglang_container = await asyncio.to_thread(
                docker_client.containers.run,
                vcst.BASILICA_SGLANG_IMAGE,
                command=sglang_command,
                name=sglang_container_name,
                detach=True,
                network=NETWORK_NAME,
                ports={f"30000/tcp": sglang_port},
                device_requests=[docker.types.DeviceRequest(device_ids=[str(gpu_id)], capabilities=[['gpu']])],
                environment={
                    "HF_HOME": "/hf",
                    "TRANSFORMERS_CACHE": "/hf",
                    "HUGGINGFACE_HUB_CACHE": "/hf",
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "PYTHONHASHSEED": str(INIT_SEED),
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "NVIDIA_TF32_OVERRIDE": "0",
                },
                volumes={
                    "/mnt/hf_cache": {"bind": "/hf", "mode": "rw"},
                    **({lora_dir: {"bind": "/lora/trained_lora", "mode": "ro"}} if is_lora and lora_dir else {}),
                },
                ipc_mode="host",
                remove=False,
            )
            containers['sglang'] = sglang_container
        except docker.errors.APIError as e:
            error_str = str(e)
            if "port is already allocated" in error_str or "409" in error_str or "Conflict" in error_str:
                env_logger.error(f"Container creation conflict (port or name). Attempting to find and remove conflicting containers...")
                try:
                    all_containers = docker_client.containers.list(all=True)
                    removed_any = False
                    for container in all_containers:
                        try:
                            if container.name == sglang_container_name:
                                env_logger.info(f"Found container with same name {container.name}, removing it...")
                                container.remove(force=True)
                                removed_any = True
                                continue
                            ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                            if ports:
                                for port_binding in ports.values():
                                    if port_binding and any(binding.get('HostPort') == str(sglang_port) for binding in port_binding):
                                        env_logger.info(f"Found container {container.name} using port {sglang_port}, removing it...")
                                        container.remove(force=True)
                                        removed_any = True
                                        break
                        except Exception as container_err:
                            env_logger.warning(f"Error processing container {container.name}: {container_err}")
                    
                    if removed_any:
                        await asyncio.sleep(2)
                    
                    sglang_container = await asyncio.to_thread(
                        docker_client.containers.run,
                        vcst.BASILICA_SGLANG_IMAGE,
                        command=sglang_command,
                        name=sglang_container_name,
                        detach=True,
                        network=NETWORK_NAME,
                        ports={f"30000/tcp": sglang_port},
                        device_requests=[docker.types.DeviceRequest(device_ids=[str(gpu_id)], capabilities=[['gpu']])],
                        environment={
                            "HF_HOME": "/hf",
                            "TRANSFORMERS_CACHE": "/hf",
                            "HUGGINGFACE_HUB_CACHE": "/hf",
                            "HF_HUB_ENABLE_HF_TRANSFER": "1",
                            "PYTHONHASHSEED": str(INIT_SEED),
                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "NVIDIA_TF32_OVERRIDE": "0",
                        },
                        volumes={
                            "/mnt/hf_cache": {"bind": "/hf", "mode": "rw"},
                            **({lora_dir: {"bind": "/lora/trained_lora", "mode": "ro"}} if is_lora and lora_dir else {}),
                        },
                        ipc_mode="host",
                        remove=False,
                    )
                    containers['sglang'] = sglang_container
                except Exception as retry_error:
                    env_logger.error(f"Failed to recover from container conflict: {retry_error}")
                    raise
            else:
                raise
        
        sglang_url = f"http://localhost:{sglang_port}"
        await asyncio.to_thread(wait_for_local_health, sglang_url)
        env_logger.info(f"SGLang Ready at: {sglang_url}")
        
        env_logger.info(f"Starting Environment Server locally...")
        env_logger.info(f"Environment container name: {env_container_name}, GPU: {gpu_id}, port: {env_port}")
        
        # Check if container name already exists and remove it
        try:
            existing = docker_client.containers.get(env_container_name)
            env_logger.warning(f"Container {env_container_name} already exists, removing it...")
            existing.remove(force=True)
            await asyncio.sleep(1)
        except docker.errors.NotFound:
            pass
        except Exception as e:
            env_logger.warning(f"Error checking for existing container {env_container_name}: {e}")
        
        # Start Environment container
        try:
            env_container = await asyncio.to_thread(
                docker_client.containers.run,
                env_image,
                name=env_container_name,
                detach=True,
                network=NETWORK_NAME,
                ports={'8000/tcp': env_port},
                remove=False,
            )
            containers['env'] = env_container
        except docker.errors.APIError as e:
            error_str = str(e)
            if "port is already allocated" in error_str or "409" in error_str or "Conflict" in error_str:
                env_logger.error(f"Container creation conflict (port or name). Attempting to find and remove conflicting containers...")
                try:
                    all_containers = docker_client.containers.list(all=True)
                    removed_any = False
                    for container in all_containers:
                        try:
                            if container.name == env_container_name:
                                env_logger.info(f"Found container with same name {container.name}, removing it...")
                                container.remove(force=True)
                                removed_any = True
                                continue
                            ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                            if ports:
                                for port_binding in ports.values():
                                    if port_binding and any(binding.get('HostPort') == str(env_port) for binding in port_binding):
                                        env_logger.info(f"Found container {container.name} using port {env_port}, removing it...")
                                        container.remove(force=True)
                                        removed_any = True
                                        break
                        except Exception as container_err:
                            env_logger.warning(f"Error processing container {container.name}: {container_err}")
                    
                    if removed_any:
                        await asyncio.sleep(2)
                    
                    env_container = await asyncio.to_thread(
                        docker_client.containers.run,
                        env_image,
                        name=env_container_name,
                        detach=True,
                        network=NETWORK_NAME,
                        ports={'8000/tcp': env_port},
                        remove=False,
                    )
                    containers['env'] = env_container
                except Exception as retry_error:
                    env_logger.error(f"Failed to recover from container conflict: {retry_error}")
                    raise
            else:
                raise
        
        env_url = f"http://localhost:{env_port}"
        await asyncio.to_thread(wait_for_local_health, env_url, timeout=300, path="/health")
        env_logger.info(f"Environment Server Ready at: {env_url}")
        
        sglang_container_url = f"http://{sglang_container_name}:30000"
        
        # Evaluate repo 2000 times with different seeds, then average
        seed_scores = []
        for seed_idx, RANDOM_SEED in enumerate(EVAL_SEEDS, 1):
            env_logger.info(f"Starting evaluation {seed_idx}/2000 for repo {repo} with seed {RANDOM_SEED}")
            seed_score = None
            MAX_EVAL_RETRIES = 5
            retry_attempt = 0
            while retry_attempt < MAX_EVAL_RETRIES:
                retry_attempt += 1
                try:
                    # Evaluate with 1 sample per seed (containers are already running)
                    avg_score = await _run_basilica_evaluation(
                        sglang_container_url,
                        env_url,
                        1,  # Use 1 sample per seed (2000 seeds total)
                        DATA_LEN_RANGE,
                        RANDOM_SEED,
                        TEMPERATURE,
                        env_logger,
                        inference_model_name,
                        TASK_ID_MIN,
                        env_name=env_name
                    )
                    
                    seed_score = avg_score
                    break
                except Exception as e:
                    if retry_attempt < MAX_EVAL_RETRIES:
                        env_logger.error(f"Evaluation attempt {retry_attempt}/{MAX_EVAL_RETRIES} failed: {str(e)}, retrying...", exc_info=True)
                        await asyncio.sleep(10)  # Short delay before retry
                    else:
                        env_logger.error(f"Evaluation attempt {retry_attempt}/{MAX_EVAL_RETRIES} failed: {str(e)}, max retries reached.", exc_info=True)
            
            if seed_score is not None:
                env_logger.info(f"Evaluation {seed_idx}/2000 completed successfully with seed {RANDOM_SEED}, score: {seed_score:.4f}")
                seed_scores.append(seed_score)
            else:
                env_logger.error(f"Evaluation {seed_idx}/2000 failed after {MAX_EVAL_RETRIES} attempts with seed {RANDOM_SEED}.")
        
        # Clean up containers after all seeds are done
        env_logger.info(f"All 2000 seeds completed for repo {repo}. Cleaning up containers...")
        await cleanup_containers(containers)
        
        # Average the scores from all 2000 seeds
        if seed_scores:
            final_avg_score = sum(seed_scores) / len(seed_scores)
            env_logger.info(f"Repo {repo} evaluation complete. Scores: {seed_scores}, Final averaged score: {final_avg_score:.4f}")
            repo_result = {
                'is_finetune': True,
                'eval_loss': final_avg_score
            }
        else:
            env_logger.error(f"Repo {repo} evaluation failed on all 2000 seeds.")
            repo_result = "Evaluation failed on all seeds"
        
        return (repo, repo_result)
    
    # Evaluate repos concurrently (up to 4 at a time, one per GPU)
    # Each repo evaluation still uses 4 concurrent requests internally
    logger.info(f"Starting {len(models)} evaluations with {NUM_GPUS} concurrent evaluations (one per GPU)...")
    
    async def evaluate_with_gpu_slot(repo: str, repo_idx: int) -> tuple[str, dict | str]:
        """Wrapper to acquire GPU slot, run evaluation, and release GPU slot."""
        # Acquire GPU slot
        async with eval_semaphore:
            gpu_id = await gpu_available.get()
            try:
                logger.info(f"Repo {repo} (idx {repo_idx}): Acquired GPU {gpu_id}, ports: SGLang={SGLANG_BASE_PORT + gpu_id}, Env={ENV_BASE_PORT + gpu_id}")
                result = await evaluate_single_repo(repo, repo_idx, gpu_id)
                logger.info(f"Repo {repo} (idx {repo_idx}): Completed evaluation on GPU {gpu_id}")
                return result
            finally:
                # Release GPU slot
                await gpu_available.put(gpu_id)
                logger.info(f"Repo {repo} (idx {repo_idx}): Released GPU {gpu_id}")
    
    # Run evaluations concurrently (limited by semaphore to NUM_GPUS)
    tasks = [evaluate_with_gpu_slot(repo, idx) for idx, repo in enumerate(models)]
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
    
    async def evaluate_single_task(session: aiohttp.ClientSession, task_id: int, task_idx: int) -> dict | None:
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
        
        while attempt < max_retries:
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
                if attempt >= max_retries:
                    env_logger.error(f"Task ID {task_id}: Failed after {max_retries} attempts. Last error: {last_error}. This task will be excluded from average score calculation.")
                    return None
                env_logger.warning(f"Task ID {task_id}: Error (retry {attempt}/{max_retries} in {retry_delay:.1f}s): {last_error}")
                await asyncio.sleep(retry_delay)
        
        # If we exit the loop without returning, all retries failed
        env_logger.error(f"Task ID {task_id}: Failed after {max_retries} attempts. Last error: {last_error}. This task will be excluded from average score calculation.")
        return None
    
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
            elif result is None:
                env_logger.warning(f"Task {eval_list[idx]}: Failed after {max_retries} retries, excluded from average")
            else:
                all_results.append(result)
    
    total_score = sum(r.get('score', 0.0) for r in all_results)
    total_time = sum(r.get('time', 0.0) for r in all_results)
    avg_score = total_score / len(all_results) if all_results else 0.0
    avg_time = total_time / len(all_results) if all_results else 0.0
    
    successful_tasks = len(all_results)
    total_attempted = len(eval_list)
    failed_tasks = total_attempted - successful_tasks
    env_logger.info(f"Summary: Successful Tasks: {successful_tasks}/{total_attempted}, Failed Tasks: {failed_tasks}, Average Score: {avg_score:.4f}, Average Time: {avg_time:.2f}s")
    
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
