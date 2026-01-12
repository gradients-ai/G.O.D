import asyncio
import io
import json
import os
import shutil
import tarfile

import docker
from docker.models.containers import Container
from docker.types import Mount
from huggingface_hub import snapshot_download
import requests
import time
import random
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
    Run environment evaluation with separate containers for each model repo.
    This approach launches one container per repo and merges results.
    """

    VLLM_HOST_PORT = 53421
    AGENT_HOST_PORT = 53422

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

    logger.info(f"Starting sequential environment evaluation for {len(models)} repos: {models}")

    environment_server_image = ""
    if dataset_type.environment_name == "alfworld":
        environment_server_image = "affinefoundation/agentgym:alfworld"

    evaluation_results = {}
    for repo in models:

        client = docker.from_env()
        environment = base_environment.copy()
        environment["MODELS"] = repo

        containers = {}
        all_results = []
        vllm_log_task = None
        agent_log_task = None

        # Pre-cleanup: Ensure names are free to prevent "Conflict" errors
        try: client.containers.get("vllm-server").remove(force=True)
        except: pass
        try: client.containers.get("agent-server").remove(force=True)
        except: pass

        # Start VLLM server for model inference
        try:
            # Docker Network Setup
            networks = client.networks.list(names=["agent_eval_net"])
            if not networks: client.networks.create("agent_eval_net", driver="bridge")
            logger.info(f"Starting vLLM: {original_model} w/ lora {repo}")
            vllm_command = f"--model {original_model} --enable-lora --lora-modules trained_lora={repo} --max-lora-rank 256 --port 8000 --trust-remote-code"

            vllm_container: Container = await asyncio.to_thread(
                client.containers.run,
                "vllm/vllm-openai:latest",
                name="vllm-server",
                command=vllm_command,
                volumes=volume_bindings,
                runtime="nvidia",
                device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gid) for gid in gpu_ids])],
                detach=True,
                network="agent_eval_net",
                ports={'8000/tcp': VLLM_HOST_PORT},
            )
            containers['vllm'] = vllm_container
            vllm_log_context = {**get_all_context_tags(), "container_type": "vllm", "repo": repo}
            vllm_log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, vllm_container, None, vllm_log_context))

            logger.info("Starting AgentGym Server...")
            environment_container: Container = await asyncio.to_thread(
                client.containers.run,
                environment_server_image,
                name="agent-server",
                detach=True,
                network="agent_eval_net",
                ports={'8000/tcp': AGENT_HOST_PORT} 
            )
            containers['agent'] = environment_container
            agent_log_context = {**get_all_context_tags(), "container_type": "agentgym", "repo": repo}
            agent_log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, environment_container, None, agent_log_context))

            logger.info("Waiting for vLLM health check...")
            max_wait_time = 300  # 5 minutes timeout
            start_time = time.time()
            while True:
                try:
                    vllm_container.reload()
                    if vllm_container.status == 'exited':
                        exit_code = vllm_container.attrs['State']['ExitCode']
                        raise Exception(f"vLLM container exited with code {exit_code}. Check logs for details.")
                except Exception as container_error:
                    if "exited" in str(container_error).lower():
                        raise container_error
                
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError(f"vLLM health check timeout after {max_wait_time} seconds")
                
                try:
                    if requests.get(f"http://localhost:{VLLM_HOST_PORT}/v1/models", timeout=2).status_code == 200:
                        break
                except:
                    time.sleep(5)
            logger.info("vLLM Ready.\n")

            # Evaluation Loop
            DATA_LEN_RANGE = 2500
            random.seed(42)
            eval_list = random.sample(range(1, DATA_LEN_RANGE + 1), num_eval_samples)
            total_score = 0.0
            total_time = 0.0

            for i, task_id in enumerate(eval_list):
                logger.info(f"[{i+1}/{num_eval_samples}] Task ID: {task_id}...")

                payload = {
                    "model": "trained_lora",
                    "base_url": "http://vllm-server:8000/v1",
                    "task_id": task_id,
                    "temperature": 0.0,
                    "max_round": 30
                }

                try:
                    start_ts = time.time()
                    response = requests.post(f"http://localhost:{AGENT_HOST_PORT}/evaluate", json=payload, timeout=2500)
                    result = response.json()

                    latency = result.get('time_taken', time.time() - start_ts)
                    score = result.get('score', 0.0)

                    total_score += score
                    total_time += latency

                    all_results.append({
                        "task_id": task_id,
                        "task_name": result.get('task_name', 'unknown'),
                        "score": score,
                        "success": result.get('success', False),
                        "time": latency,
                        "error": result.get('error')
                    })
                    logger.info(f" Done (Score: {score})")
                except Exception as e:
                    logger.info(f" Failed: {e}")

            # Final Aggregation & File Writing
            avg_score = total_score / len(all_results) if all_results else 0
            logger.info(f"Calculated average score of model to be: {avg_score}")
            avg_time = total_time / len(all_results) if all_results else 0
            logger.info(f"Calculated average time of model to be: {avg_time}")

            evaluation_results[repo] = {
                'is_finetune': True,
                'eval_loss': avg_score
            }

        except Exception as e:
            logger.error(f"Failed to evaluate repo {repo}: {str(e)}", exc_info=True)
            evaluation_results[repo] = str(e)
            
        finally:
            try:
                if vllm_log_task is not None:
                    vllm_log_task.cancel()
                if agent_log_task is not None:
                    agent_log_task.cancel()
            except:
                pass
            
            for c in containers.values():
                try: c.remove(force=True)
                except: pass
            client.close()

    logger.info(f"Environment evaluation results: {evaluation_results}")
    return process_evaluation_results(evaluation_results, is_image=False)


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
