import asyncio
import glob
import json
import logging
import os
import random
import shutil
import time
import uuid

import aiohttp
import docker
from docker.types import Mount
from huggingface_hub import snapshot_download

from core import constants as cst
from core.models.payload_models import DockerEvaluationResults
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import InstructTextDatasetType
from core.utils import download_s3_file
from validator.core import constants as vcst
from validator.evaluation.docker_evaluation import cleanup_resources
from validator.evaluation.docker_evaluation import get_evaluation_results
from validator.evaluation.docker_evaluation import normalize_rewards_and_compute_loss
from validator.evaluation.docker_evaluation import process_evaluation_results
from validator.evaluation.utils import check_for_lora
from validator.evaluation.utils import wait_for_basilica_health
from validator.tasks.task_prep import unzip_to_temp_path
from validator.utils.logging import get_all_context_tags
from validator.utils.logging import get_environment_logger
from validator.utils.logging import get_logger
from validator.utils.logging import stream_container_logs


logger = get_logger(__name__)


def stream_container_logs_to_file(
    container,
    output_path: str,
    logger_obj: logging.Logger | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as out:
        for log_chunk in container.logs(stream=True, follow=True):
            log_text = log_chunk.decode("utf-8", errors="replace")
            out.write(log_text)
            out.flush()
            if logger_obj is not None:
                for line in log_text.splitlines():
                    if line:
                        logger_obj.info(line)


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
        gpu_id = gpu_ids[0] if gpu_ids else 0
        return await run_evaluation_local_environment(models, original_model, dataset_type, gpu_id=gpu_id, eval_seed=eval_seed)
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
        dataset_dir: {"bind": "/workspace/input_data", "mode": "ro"},
        os.path.expanduser(cst.CACHE_DIR_HUB): {"bind": "/root/.cache/huggingface/hub", "mode": "rw"},
    }

    container = None
    retry_delay = 5.0
    try:
        while True:
            try:
                container = await asyncio.to_thread(
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
                    except Exception:
                        pass
                await asyncio.sleep(retry_delay)
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
    logger.info(f"Downloading original GRPO model: {original_model}")
    cache_dir = os.path.expanduser(cst.CACHE_DIR_HUB)
    await asyncio.to_thread(snapshot_download, repo_id=original_model, cache_dir=cache_dir, ignore_patterns=None)

    command = ["python", "-m", "validator.evaluation.eval_grpo"]
    dataset_type_str = dataset_type.model_dump_json()
    dataset_filename = os.path.basename(dataset)
    dataset_dir = os.path.dirname(os.path.abspath(dataset))

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
        dataset_dir: {"bind": "/workspace/input_data", "mode": "ro"},
        os.path.expanduser(cst.CACHE_DIR_HUB): {"bind": "/root/.cache/huggingface/hub", "mode": "rw"},
    }

    logger.info(f"Starting sequential GRPO evaluation for {len(models)} repos: {models}")
    evaluation_results = {}
    for repo in models:
        client = docker.from_env()
        environment = base_environment.copy()
        environment["MODELS"] = repo
        retry_delay = 5.0

        model_path = None
        while model_path is None:
            try:
                model_path = await asyncio.to_thread(
                    snapshot_download,
                    repo_id=repo,
                    cache_dir=cache_dir,
                    ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.pkl", "*.pth"],
                )
            except Exception as e:
                logger.error(f"Failed to download {repo}: {str(e)}, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)

        container = None
        while True:
            try:
                container = await asyncio.to_thread(
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
                break
            except Exception as e:
                logger.error(f"Failed to evaluate repo {repo}: {str(e)}, retrying in {retry_delay}s...", exc_info=True)
                if container is not None:
                    try:
                        await asyncio.to_thread(container.remove, force=True)
                    except Exception:
                        pass
                await asyncio.sleep(retry_delay)
            finally:
                if container is not None:
                    try:
                        await asyncio.to_thread(container.remove, force=True)
                        await cleanup_resources(client)
                    except Exception as e:
                        logger.info(f"Problem with cleaning up container for {repo}: {e}")
        client.close()

    evaluation_results = normalize_rewards_and_compute_loss(evaluation_results)
    logger.debug(f"Grpo evaluation results post normalization: {evaluation_results}")
    return process_evaluation_results(evaluation_results, is_image=False)


async def run_evaluation_local_environment(
    models: list[str],
    original_model: str,
    dataset_type: EnvironmentDatasetType,
    gpu_id: int = 0,
    eval_seed: int | None = None,
) -> DockerEvaluationResults:
    logger.info(f"Starting local Docker environment evaluation for {len(models)} repos: {models}")
    stream_sglang_logs = os.getenv("LOCAL_ENV_STREAM_SGLANG_LOGS", "0").strip().lower() in {"1", "true", "yes", "on"}
    raw_sglang_log_file = os.getenv("LOCAL_ENV_SGLANG_RAW_LOG_FILE", "").strip()
    sglang_log_requests = os.getenv("LOCAL_ENV_SGLANG_LOG_REQUESTS", "0").strip().lower() in {"1", "true", "yes", "on"}
    sglang_log_requests_level = os.getenv("LOCAL_ENV_SGLANG_LOG_REQUESTS_LEVEL", "3")
    sglang_log_requests_format = os.getenv("LOCAL_ENV_SGLANG_LOG_REQUESTS_FORMAT", "json")
    sglang_log_requests_target = os.getenv("LOCAL_ENV_SGLANG_LOG_REQUESTS_TARGET", "stdout")

    env_name = dataset_type.environment_name
    if env_name not in vcst.ENVIRONMENTS:
        raise ValueError(f"Environment '{env_name}' not found. Supported: {list(vcst.ENVIRONMENTS.keys())}")

    env_config = vcst.ENVIRONMENTS[env_name]
    task_id_min, task_id_max = env_config["task_id_range"]
    num_seeds = env_config.get("num_seeds", vcst.ENV_EVAL_NUM_SEEDS)
    env_image = env_config["env_image"]
    env_payload_extra = env_config.get("eval_payload_extra", {})

    base_seed = eval_seed if eval_seed is not None else vcst.ENV_EVAL_DEFAULT_SEED
    seed_generator = random.Random(base_seed)
    eval_seeds = [seed_generator.randint(1, 1000000) for _ in range(num_seeds)]
    logger.info(f"Generated {num_seeds} seeds from base_seed={base_seed}")

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
        env_logger = get_environment_logger(name=f"{repo_name}-{eval_id[:8]}", repo_id=repo, eval_id=eval_id, model=original_model)

        containers = {}
        lora_dir = None
        sglang_log_task = None
        try:
            is_lora = await asyncio.to_thread(check_for_lora, repo, local_files_only=False)
            if is_lora:
                base_model = original_model
                inference_model_name = f"{original_model}:trained_lora"
                env_logger.info(f"LoRA detected: {original_model} + LoRA {repo}")
                safe_lora_name = repo.replace("/", "_")
                lora_dir = f"/tmp/sglang_lora/{safe_lora_name}"
                await asyncio.to_thread(snapshot_download, repo_id=repo, local_dir=lora_dir, local_dir_use_symlinks=False, tqdm_class=None)
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
            if sglang_log_requests:
                sglang_args += (
                    f" --log-requests --log-requests-level {sglang_log_requests_level}"
                    f" --log-requests-format {sglang_log_requests_format}"
                    f" --log-requests-target {sglang_log_requests_target}"
                )

            sglang_container_name = f"{eval_id}-sglang"
            env_container_name = f"{eval_id}-env"

            sglang_volumes = {vcst.LOCAL_ENV_HF_CACHE_PATH: {"bind": "/hf", "mode": "rw"}}
            if is_lora and lora_dir:
                sglang_volumes[lora_dir] = {"bind": "/lora/trained_lora", "mode": "ro"}

            env_logger.info(f"Starting SGLang container: {sglang_container_name} (GPU {gpu_id})")
            sglang_container = await asyncio.to_thread(
                docker_client.containers.run,
                "lmsysorg/sglang:latest",
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
            if stream_sglang_logs:
                if raw_sglang_log_file:
                    sglang_log_task = asyncio.create_task(
                        asyncio.to_thread(stream_container_logs_to_file, sglang_container, raw_sglang_log_file, env_logger)
                    )
                else:
                    sglang_log_task = asyncio.create_task(
                        asyncio.to_thread(stream_container_logs, sglang_container, env_logger, {"log_source": "sglang"})
                    )

            sglang_host_url = f"http://localhost:{vcst.LOCAL_ENV_SGLANG_PORT}"
            await asyncio.to_thread(wait_for_basilica_health, sglang_host_url, vcst.LOCAL_ENV_SGLANG_HEALTH_TIMEOUT)
            env_logger.info(f"SGLang ready at {sglang_host_url}")

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
            await asyncio.to_thread(wait_for_basilica_health, env_host_url, vcst.LOCAL_ENV_SERVER_HEALTH_TIMEOUT, "/health")
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
            if sglang_log_task:
                sglang_log_task.cancel()
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
    eval_list = []
    for seed in eval_seeds:
        rng = random.Random(seed)
        task_id = rng.randint(task_id_min + 1, data_len_range)
        eval_list.append((seed, task_id))

    num_eval_samples = len(eval_list)
    all_results = []
    retry_statuses = {404, 500, 501}

    async def evaluate_single_task(session: aiohttp.ClientSession, seed: int, task_id: int, task_idx: int) -> dict | None:
        payload = {
            "model": inference_model_name,
            "base_url": f"{sglang_url}/v1",
            "task_id": task_id,
            "temperature": temperature,
            "seed": seed,
        }
        if env_payload_extra:
            payload.update(env_payload_extra)

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
                    raw_text = await response.text()
                    if response.status != 200:
                        error_detail = f": {raw_text[:500]}" if raw_text else ""
                        raise Exception(f"HTTP {response.status}{error_detail}")
                    response_data = json.loads(raw_text)
                    result = response_data.get("result", response_data)
                    latency = result.get("time_taken", time.time() - start_ts)
                    score = result.get("score", 0.0)
                    env_logger.info(f"Task ID {task_id}: Done (Score: {score})")
                    return {"task_id": task_id, "score": score, "time": latency}
            except Exception as e:
                if any(f"HTTP {c}" in str(e) for c in retry_statuses):
                    if attempt >= vcst.ENV_EVAL_TASK_MAX_RETRIES:
                        env_logger.warning("Task ID %s: Basilica failure after %d attempts, excluding from average", task_id, attempt)
                        return None
                    await asyncio.sleep(vcst.ENV_EVAL_TASK_RETRY_DELAY)
                else:
                    env_logger.error("Task ID %s: Non-retryable error, scoring 0: %s", task_id, e)
                    return {"task_id": task_id, "score": 0.0, "time": 0.0}

    semaphore = asyncio.Semaphore(vcst.ENV_EVAL_MAX_CONCURRENT_REQUESTS)

    async def evaluate_with_semaphore(session: aiohttp.ClientSession, seed: int, task_id: int, task_idx: int) -> dict | None:
        async with semaphore:
            return await evaluate_single_task(session, seed, task_id, task_idx)

    session_timeout = aiohttp.ClientTimeout(total=vcst.ENV_EVAL_SESSION_TIMEOUT)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        env_logger.info(f"Starting {num_eval_samples} evaluations with concurrency={vcst.ENV_EVAL_MAX_CONCURRENT_REQUESTS}...")
        tasks = [evaluate_with_semaphore(session, seed, task_id, idx) for idx, (seed, task_id) in enumerate(eval_list)]
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
    env_logger.info(f"Summary: {len(all_results)}/{len(eval_list)} successful, Avg Score: {avg_score:.4f}, Avg Time: {avg_time:.2f}s")
    return avg_score


async def run_evaluation_docker_image(
    test_split_url: str,
    original_model_repo: str,
    models: list[str],
    model_type: ImageModelType,
    gpu_ids: list[int],
) -> DockerEvaluationResults:
    raw_data = await download_s3_file(test_split_url)
    test_split_path = unzip_to_temp_path(raw_data)
    dataset_dir = os.path.abspath(test_split_path)
    container_dataset_path = "/workspace/input_data"

    client = docker.from_env()
    base_path = "/app/validator/evaluation/ComfyUI/models"
    mounts = [
        Mount(target=container_dataset_path, source=dataset_dir, type="bind", read_only=True),
        Mount(target=f"{base_path}/checkpoints", source=cst.CACHE_DIR_HUB, type="bind", read_only=False),
        Mount(target=f"{base_path}/diffusers", source=cst.CACHE_DIR_HUB, type="bind", read_only=False),
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
                    except Exception:
                        pass
                await asyncio.sleep(retry_delay)
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
