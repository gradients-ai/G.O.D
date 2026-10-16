import asyncio
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

import docker
from docker.errors import APIError
from docker.errors import BuildError
from docker.models.containers import Container

import core.constants as core_cst
import trainer.training_paths as train_paths
from core.logging import get_all_context_tags
from core.logging import stream_container_logs
from core.logging import stream_image_build_logs
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import EnvironmentDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import GrpoDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.image_models import ImageModelType
from core.models.model_prep_models import BaselineStats
from core.models.payload_models import ModelPrepResponse
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.task_models import TaskType
from trainer import constants as cst
from trainer.host import build_wandb_env
from trainer.host import extract_container_error
from trainer.job_state import complete_task
from trainer.job_state import log_task
from trainer.job_state import update_container_name
from trainer.job_state import update_training_started_at
from trainer.job_state import update_wandb_url
from trainer.model_artifacts import get_anonymous_model_dir
from trainer.telemetry import logger


# logger = get_logger(__name__)


_warned_missing_hf_token = False


def _hf_container_env() -> dict[str, str]:
    """Shared HF credentials passed to trainer-side containers."""
    global _warned_missing_hf_token
    token = os.environ.get("HUGGINGFACE_TOKEN", "")
    if not token and not _warned_missing_hf_token:
        logger.warning("HUGGINGFACE_TOKEN is empty; HF-gated model pulls/pushes may fail with auth header errors.")
        _warned_missing_hf_token = True
    return {
        "HUGGINGFACE_TOKEN": token,
        "HUGGINGFACE_USERNAME": os.environ.get("HUGGINGFACE_USERNAME", ""),
    }


def ensure_internal_network(name: str = cst.INTERNAL_BRIDGE_NAME):
    client = docker.from_env()
    try:
        client.networks.get(name)
    except docker.errors.NotFound:
        client.networks.create(name, driver="bridge", internal=True)


def calculate_container_resources(gpu_ids: list[int]) -> tuple[str, int]:
    """Calculate memory limit and CPU limit based on GPU count.

    Returns:
        tuple: (memory_limit_str, cpu_limit_nanocpus)
    """
    num_gpus = len(gpu_ids)
    memory_limit = f"{num_gpus * cst.MEMORY_PER_GPU_GB}g"
    cpu_limit_nanocpus = num_gpus * cst.CPUS_PER_GPU * 1_000_000_000

    logger.info(f"Allocating resources for {num_gpus} GPUs: {memory_limit} memory, {num_gpus * cst.CPUS_PER_GPU} CPUs")
    return memory_limit, cpu_limit_nanocpus


def build_docker_image(
    dockerfile_path: str,
    log_labels: dict[str, str] | None = None,
    context_path: str = ".",
    is_image_task: bool = False,
    tag: str = None,
    no_cache: bool = True,
    timeout_seconds: int | None = None,
) -> tuple[str, str | None]:
    client: docker.DockerClient = docker.from_env()

    if tag is None:
        tag = f"standalone-image-trainer:{uuid.uuid4()}" if is_image_task else f"standalone-text-trainer:{uuid.uuid4()}"

    if timeout_seconds is None:
        timeout_seconds = cst.DOCKER_BUILD_TIMEOUT_MINUTES * 60

    logger.info(
        f"Building Docker image '{tag}' (timeout={timeout_seconds}s)...",
        extra=log_labels,
    )

    build_output = None
    executor = None
    try:
        build_output = client.api.build(
            path=context_path,
            dockerfile=dockerfile_path,
            tag=tag,
            nocache=no_cache,
            decode=True,
        )
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(stream_image_build_logs, build_output, logger, log_labels)
        try:
            future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            message = f"Docker image build exceeded {timeout_seconds} seconds"
            logger.error(message, extra=log_labels)
            if build_output is not None:
                try:
                    build_output.close()
                except Exception:
                    pass
            return None, message

        logger.info("Docker image built successfully.", extra=log_labels)
        return tag, None
    except (BuildError, APIError) as e:
        logger.error(f"Docker build failed: {str(e)}", extra=log_labels)
        return None, str(e)
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def delete_image_and_cleanup(tag: str):
    client = docker.from_env()
    try:
        client.images.remove(image=tag, force=True)
        logger.info(f"Deleted Docker image with tag: {tag}")
    except docker.errors.ImageNotFound:
        logger.error(f"No Docker image found with tag: {tag}")
    except Exception as e:
        logger.error(f"Failed to delete image '{tag}': {e}")

    try:
        client.images.prune(filters={"dangling": True})
        client.api.prune_builds()
        logger.info("Cleaned up dangling images and build cache.")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


async def run_trainer_container_image(
    task_id: str,
    tag: str,
    model: str,
    dataset_zip: str,
    model_type: str,
    expected_repo_name: str,
    hours_to_complete: float,
    hotkey: str,
    trigger_word: str | None = None,
    baseline_stats: BaselineStats | None = None,
    log_labels: dict[str, str] | None = None,
    gpu_ids: list[int] = [0],
) -> Container:
    client: docker.DockerClient = docker.from_env()

    await asyncio.to_thread(ensure_internal_network)

    command: list[str] = [
        "--task-id",
        task_id,
        "--model",
        model,
        "--dataset-zip",
        dataset_zip,
        "--model-type",
        model_type,
        "--expected-repo-name",
        expected_repo_name,
        "--hours-to-complete",
        str(hours_to_complete),
    ]

    if trigger_word:
        command += ["--trigger-word", trigger_word]

    environment: dict[str, str] = {
        "TRANSFORMERS_CACHE": cst.HUGGINGFACE_CACHE_PATH,
        "HF_HOME": cst.HUGGINGFACE_CACHE_PATH,
        **_hf_container_env(),
    }
    if baseline_stats:
        vol = client.volumes.get(cst.CACHE_VOLUME_NAME)
        stats_filename = f"baseline_stats_{task_id}.json"
        with open(os.path.join(vol.attrs["Mountpoint"], stats_filename), "w") as f:
            json.dump(baseline_stats.model_dump(), f)
        environment["BASELINE_STATS_PATH"] = os.path.join(cst.CACHE_ROOT_PATH, stats_filename)

    container_name = f"image-trainer-{uuid.uuid4().hex}"

    # Calculate resources based on GPU count
    memory_limit, cpu_limit_nanocpus = calculate_container_resources(gpu_ids)

    # Set shared memory size based on GPU count
    shm_size = "16g" if len(gpu_ids) >= 4 else "8g"

    max_retries = cst.CONTAINER_START_MAX_RETRIES
    retry_delay = cst.CONTAINER_START_RETRY_DELAY_SECONDS

    for attempt in range(max_retries):
        try:
            container: Container = await asyncio.to_thread(
                client.containers.run,
                image=tag,
                command=command,
                volumes={
                    cst.CHECKPOINTS_VOLUME_NAME: {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"},
                    cst.CACHE_VOLUME_NAME: {"bind": cst.CACHE_ROOT_PATH, "mode": "ro"},
                },
                remove=False,
                shm_size=shm_size,
                name=container_name,
                labels=log_labels,
                mem_limit=memory_limit,
                nano_cpus=cpu_limit_nanocpus,
                device_requests=[docker.types.DeviceRequest(device_ids=[str(i) for i in gpu_ids], capabilities=[["gpu"]])],
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
                network=cst.INTERNAL_BRIDGE_NAME,
                environment=environment,
                detach=True,
            )

            _log_streaming_task = asyncio.create_task(
                asyncio.to_thread(stream_container_logs, container, get_all_context_tags())
            )
            return container

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Error starting container (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s: {str(e)[:150]}",
                    extra=log_labels,
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to start image trainer container after {max_retries} attempts: {e}", extra=log_labels)
                raise


async def run_trainer_container_text(
    task_id: str,
    hotkey: str,
    tag: str,
    model: str,
    dataset: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType | EnvironmentDatasetType,
    task_type: TaskType,
    file_format: FileFormat,
    expected_repo_name: str,
    hours_to_complete: float,
    baseline_stats: BaselineStats | None = None,
    log_labels: dict[str, str] | None = None,
    gpu_ids: list[int] = [0],
    env_server_urls: str | None = None,
    miner_datasets: list[str] | None = None,
    use_kl: bool = False,
    kl_coef: float | None = None,
) -> Container:
    client: docker.DockerClient = docker.from_env()

    await asyncio.to_thread(ensure_internal_network)

    environment = build_wandb_env(task_id, hotkey)
    if baseline_stats:
        vol = client.volumes.get(cst.CACHE_VOLUME_NAME)
        stats_filename = f"baseline_stats_{task_id}_{hotkey[:8]}.json"
        with open(os.path.join(vol.attrs["Mountpoint"], stats_filename), "w") as f:
            json.dump(baseline_stats.model_dump(), f)
        environment["BASELINE_STATS_PATH"] = os.path.join(cst.CACHE_ROOT_PATH, stats_filename)
    if env_server_urls:
        environment["ENVIRONMENT_SERVER_URLS"] = env_server_urls
    if miner_datasets:
        environment[cst.MINER_DATASETS_DIR_ENV] = cst.MINER_DATASETS_CACHE_DIR
        environment[cst.MINER_DATASETS_ENV] = ",".join(miner_datasets)
    if use_kl:
        environment[core_cst.USE_KL_ENV] = "1"
        if kl_coef is not None:
            environment[core_cst.KL_COEF_ENV] = str(kl_coef)

    command: list[str] = [
        "--task-id",
        task_id,
        "--model",
        model,
        "--dataset",
        dataset,
        "--dataset-type",
        json.dumps(dataset_type.model_dump()),
        "--task-type",
        task_type,
        "--file-format",
        file_format,
        "--expected-repo-name",
        expected_repo_name,
        "--hours-to-complete",
        str(hours_to_complete),
    ]

    container_name = f"text-trainer-{uuid.uuid4().hex}"

    # Calculate resources based on GPU count
    memory_limit, cpu_limit_nanocpus = calculate_container_resources(gpu_ids)

    # Set shared memory size based on GPU count
    shm_size = "16g" if len(gpu_ids) >= 4 else "8g"

    max_retries = cst.CONTAINER_START_MAX_RETRIES
    retry_delay = cst.CONTAINER_START_RETRY_DELAY_SECONDS

    for attempt in range(max_retries):
        try:
            container: Container = await asyncio.to_thread(
                client.containers.run,
                image=tag,
                command=command,
                volumes={
                    cst.CHECKPOINTS_VOLUME_NAME: {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"},
                    cst.CACHE_VOLUME_NAME: {"bind": cst.CACHE_ROOT_PATH, "mode": "ro"},
                },
                remove=False,
                shm_size=shm_size,
                name=container_name,
                labels=log_labels,
                mem_limit=memory_limit,
                nano_cpus=cpu_limit_nanocpus,
                device_requests=[docker.types.DeviceRequest(device_ids=[str(i) for i in gpu_ids], capabilities=[["gpu"]])],
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
                detach=True,
                network=cst.INTERNAL_BRIDGE_NAME,
                environment=environment,
            )

            _log_streaming_task = asyncio.create_task(
                asyncio.to_thread(stream_container_logs, container, get_all_context_tags())
            )
            return container

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Error starting container (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s: {str(e)[:150]}",
                    extra=log_labels,
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to start text trainer container after {max_retries} attempts: {e}", extra=log_labels)
                raise


def _create_volumes_sync():
    client: docker.DockerClient = docker.from_env()
    volume_names = cst.VOLUME_NAMES
    for volume_name in volume_names:
        try:
            client.volumes.get(volume_name)
        except docker.errors.NotFound:
            client.volumes.create(name=volume_name)
            logger.info(f"Volume '{volume_name}' created.")


async def create_volumes_if_dont_exist():
    await asyncio.to_thread(_create_volumes_sync)


def run_downloader_container(
    task_id: str,
    model: str,
    dataset_url: str,
    task_type: TaskType,
    hotkey: str,
    file_format: FileFormat | None = None,
    model_type: ImageModelType | None = None,
    log_labels: dict[str, str] | None = None,
    anonymize: bool = True,
) -> tuple[int, Exception | None]:
    client = docker.from_env()

    command = [
        "--task-id",
        task_id,
        "--model",
        model,
        "--task-type",
        task_type,
        "--dataset",
        dataset_url,
    ]
    if file_format:
        command += ["--file-format", file_format]

    if model_type:
        command += ["--model-type", model_type]

    if anonymize:
        command += ["--anonymize"]

    container_name = f"downloader-{task_id}-{str(uuid.uuid4())[:8]}"
    container = None

    environment = _hf_container_env()
    if anonymize:
        environment["MODEL_HASH_SALT"] = os.environ.get("MODEL_HASH_SALT", "")
    try:
        logger.info(f"Starting downloader container: {container_name}", extra=log_labels)
        container = client.containers.run(
            image=cst.TRAINER_DOWNLOADER_DOCKER_IMAGE,
            name=container_name,
            command=command,
            labels=log_labels,
            volumes={cst.CACHE_VOLUME_NAME: {"bind": "/cache", "mode": "rw"}},
            environment=environment,
            remove=False,
            detach=True,
        )

        try:
            stream_container_logs(container, get_all_context_tags())
        except Exception as log_err:
            logger.warning(f"Log streaming error (non-fatal): {log_err}", extra=log_labels)

        result = container.wait()
        exit_code = result.get("StatusCode", -1)

        if exit_code == 0:
            logger.info(f"Download completed successfully for task {task_id}", extra=log_labels)
        else:
            logs = container.logs().decode("utf-8", errors="ignore")
            error_message = extract_container_error(logs)
            return exit_code, error_message

        return exit_code, None

    except docker.errors.ContainerError as e:
        logger.error(f"Downloader container failed for task {task_id}: {e}", extra=log_labels)
        return 1, e

    except Exception as ex:
        logger.error(f"Unexpected error in downloader for task {task_id}: {ex}", extra=log_labels)
        return 1, ex

    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove container {container_name}: {cleanup_err}", extra=log_labels)


def run_model_prep_container(
    task_id: str,
    model_id: str,
    training_data_url: str,
    task_type: TaskType = TaskType.INSTRUCTTEXTTASK,
    augmentation_config=None,
    gpu_ids: list[int] = [0],
    reward_functions=None,
    log_labels: dict[str, str] | None = None,
) -> ModelPrepResponse:
    """Run model prep container: augment model + compute baseline stats.
    Downloads model to cache via downloader first."""
    client = docker.from_env()

    # Download model to cache volume
    download_exit, download_err = run_downloader_container(
        task_id=task_id,
        model=model_id,
        dataset_url=training_data_url,
        task_type=task_type,
        hotkey="",
        file_format=FileFormat.S3,
        log_labels=log_labels,
    )
    if download_exit != 0:
        raise RuntimeError(f"Downloader failed: {download_err}")

    anonymous_model = get_anonymous_model_dir(model_id)
    model_cache_path = f"/cache/models/{anonymous_model}"

    command = [
        "--model", model_cache_path,
        "--training-data", training_data_url,
        "--task-type", task_type,
        # Original HF repo id: the anonymized --model path may already be a flat
        # merge (downloader flattened the adapter), so merge-publish keys on this.
        "--source-model-id", model_id,
    ]

    if augmentation_config is not None:
        command += [
            "--aug-type", augmentation_config.aug_type.value,
            "--scope", augmentation_config.scope.value,
            "--seed", str(augmentation_config.seed),
            "--intensity", str(augmentation_config.intensity),
        ]

    if reward_functions:
        reward_functions_payload = [
            rf.model_dump() if hasattr(rf, "model_dump") else rf
            for rf in reward_functions
        ]
        command += ["--reward-functions", json.dumps(reward_functions_payload)]

    env = _hf_container_env()

    container_name = f"model-prep-{str(uuid.uuid4())[:8]}"
    container = None
    memory_limit, cpu_limit_nanocpus = calculate_container_resources(gpu_ids)

    model_prep_image = cst.MODEL_PREP_TEXT_DOCKER_IMAGE

    try:
        logger.info(f"Starting model prep container: {container_name} (image={model_prep_image})", extra=log_labels)
        container = client.containers.run(
            image=model_prep_image,
            name=container_name,
            command=command,
            labels=log_labels,
            environment=env,
            volumes={cst.CACHE_VOLUME_NAME: {"bind": "/cache", "mode": "rw"}},
            device_requests=[docker.types.DeviceRequest(
                device_ids=[str(i) for i in gpu_ids],
                capabilities=[["gpu"]],
            )],
            mem_limit=memory_limit,
            nano_cpus=cpu_limit_nanocpus,
            network=None,
            remove=False,
            detach=True,
        )

        stream_container_logs(container, get_all_context_tags())

        result = container.wait()
        exit_code = result.get("StatusCode", -1)
        logs_output = container.logs().decode("utf-8", errors="ignore")

        if exit_code != 0:
            error_message = extract_container_error(logs_output)
            raise RuntimeError(f"Model prep container failed (exit {exit_code}): {error_message}")

        # Container writes JSON result to last line of stdout
        result_line = logs_output.strip().rsplit("\n", 1)[-1]
        return ModelPrepResponse.model_validate_json(result_line)

    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove container {container_name}: {cleanup_err}", extra=log_labels)


async def upload_repo_to_hf(
    task_id: str,
    hotkey: str,
    expected_repo_name: str,
    huggingface_token: str,
    huggingface_username: str,
    model: str,
    docker_labels: dict[str, str] | None = None,
    wandb_token: str | None = None,
    path_in_repo: str | None = None,
):
    container = None
    try:
        client = docker.from_env()
        local_container_folder = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)

        environment = {
            "HUGGINGFACE_TOKEN": huggingface_token,
            "HUGGINGFACE_USERNAME": huggingface_username,
            "WANDB_TOKEN": wandb_token or None,
            "WANDB_LOGS_PATH": f"{cst.WANDB_LOGS_DIR}/{task_id}_{hotkey}",
            "LOCAL_FOLDER": local_container_folder,
            "MODEL": model,
            "TASK_ID": task_id,
            "EXPECTED_REPO_NAME": expected_repo_name,
            "HF_REPO_SUBFOLDER": path_in_repo,
        }

        volumes = {
            cst.CHECKPOINTS_VOLUME_NAME: {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"},
            cst.CACHE_VOLUME_NAME: {"bind": cst.CACHE_ROOT_PATH, "mode": "rw"},
        }

        container_name = f"hf-upload-{uuid.uuid4().hex}"

        logger.info(f"Starting upload container {container_name} for task {task_id}...", extra=docker_labels)

        container = await asyncio.to_thread(
            client.containers.run,
            image=cst.HF_UPLOAD_DOCKER_IMAGE,
            environment=environment,
            volumes=volumes,
            labels=docker_labels,
            detach=True,
            remove=False,
            name=container_name,
        )

        _log_streaming_task = asyncio.create_task(
            asyncio.to_thread(stream_container_logs, container, get_all_context_tags())
        )

        result = await asyncio.to_thread(container.wait)
        logs = (await asyncio.to_thread(container.logs)).decode("utf-8", errors="ignore")
        exit_code = result.get("StatusCode", -1)
        wandb_url = None
        if wandb_token:
            m = re.search(r"https://wandb\.ai/\S+", logs)
            wandb_url = m.group(0) if m else None
            if wandb_url:
                await update_wandb_url(task_id, hotkey, wandb_url)

        if exit_code != 0:
            last_err = extract_container_error(logs) or "unknown error"
            msg = f"HF upload failed | exit_code={exit_code} | container={container_name} | last_error={last_err}"
            await log_task(task_id, hotkey, f"[ERROR] {msg}")
            raise RuntimeError(msg)

    except Exception as e:
        logger.exception(f"Unexpected error during upload_repo_to_hf for task {task_id}: {e}", extra=docker_labels)
        raise

    finally:
        if container and isinstance(container, Container):
            try:
                await asyncio.to_thread(container.reload)
                if container.status == "running":
                    await asyncio.to_thread(container.kill)
                await asyncio.to_thread(container.remove, force=True)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove upload container {container.name}: {cleanup_err}")


def get_task_type(request: TrainerProxyRequest) -> TaskType:
    training_data = request.training_data

    if isinstance(training_data, TrainRequestImage):
        return TaskType.IMAGETASK

    elif isinstance(training_data, TrainRequestText):
        if isinstance(training_data.dataset_type, DpoDatasetType):
            return TaskType.DPOTASK
        elif isinstance(training_data.dataset_type, EnvironmentDatasetType):
            return TaskType.ENVIRONMENTTASK
        elif isinstance(training_data.dataset_type, InstructTextDatasetType):
            return TaskType.INSTRUCTTEXTTASK
        elif isinstance(training_data.dataset_type, ChatTemplateDatasetType):
            return TaskType.CHATTASK
        elif isinstance(training_data.dataset_type, GrpoDatasetType):
            return TaskType.GRPOTASK
        else:
            raise ValueError(f"Unsupported dataset_type for text task: {type(training_data.dataset_type)}")

    raise ValueError(f"Unsupported training_data type: {type(training_data)}")


def get_dockerfile_path(task_type: TaskType, training_data, local_repo_path: str) -> str:
    """Get the appropriate dockerfile path based on task type and model type"""
    if task_type == TaskType.IMAGETASK:
        if training_data.model_type == ImageModelType.FLUX:
            return _resolve_dockerfile_path(local_repo_path, cst.IMAGE_DOCKERFILE_PATHS)
        return _resolve_dockerfile_path(local_repo_path, cst.IMAGE_TOOLKIT_DOCKERFILE_PATHS)

    else:
        return _resolve_dockerfile_path(local_repo_path, cst.TEXT_DOCKERFILE_PATHS)


def _resolve_dockerfile_path(local_repo_path: str, candidate_paths: tuple[str, ...]) -> str:
    for candidate_path in candidate_paths:
        full_path = os.path.join(local_repo_path, candidate_path)
        if os.path.exists(full_path):
            return full_path

    searched = ", ".join(candidate_paths)
    raise FileNotFoundError(f"Training repository is missing a supported Dockerfile. Checked: {searched}")


async def start_training_task(task: TrainerProxyRequest, local_repo_path: str):
    cancelled_exc: asyncio.CancelledError | None = None
    cancel_log_message: str | None = None

    try:
        training_data = task.training_data
        success = False
        container = None
        env_server_containers = []
        tag = None
        timeout_seconds = max(1, int(training_data.hours_to_complete * 3600))
        task_type = get_task_type(task)
        await create_volumes_if_dont_exist()

        log_labels = {
            "task_id": training_data.task_id,
            "hotkey": task.hotkey,
            "model": training_data.model,
            "task_type": task_type,
            "expected_repo": training_data.expected_repo_name,
            **(
                {"dataset_type": str(training_data.dataset_type)}
                if getattr(training_data, "dataset_type", None) is not None
                else {}
            ),
        }

        dockerfile_path = get_dockerfile_path(task_type, training_data, local_repo_path)

        logger.info("Running Cache Download Container", extra=log_labels)
        await log_task(training_data.task_id, task.hotkey, "Downloading data")

        model_prep_ran = training_data.baseline_stats is not None

        download_status, exc = await asyncio.to_thread(
            run_downloader_container,
            task_id=training_data.task_id,
            model=training_data.model,
            dataset_url=training_data.dataset_zip if task_type == TaskType.IMAGETASK else training_data.dataset,
            task_type=task_type,
            hotkey=task.hotkey,
            file_format=getattr(training_data, "file_format", None),
            model_type=training_data.model_type if task_type == TaskType.IMAGETASK else None,
            log_labels=log_labels,
            anonymize=model_prep_ran,
        )

        if download_status == 0:
            message = "Download container completed successfully"
            await log_task(training_data.task_id, task.hotkey, message)
        else:
            message = f"[ERROR] Download container failed | ExitCode: {download_status} | LastError: {exc}"
            await log_task(training_data.task_id, task.hotkey, message)
            await complete_task(training_data.task_id, task.hotkey, success=False)
            raise RuntimeError(f"Downloader container failed: {exc}")

        tag, exc = await asyncio.to_thread(
            build_docker_image,
            dockerfile_path=dockerfile_path,
            log_labels=log_labels,
            is_image_task=(task_type == TaskType.IMAGETASK),
            context_path=local_repo_path,
        )

        if not tag:
            message = f"[ERROR] Image Build failed | ExitCode: Unknown | LastError: {exc}"
            logger.error(f"Image build failed: {exc}", extra=log_labels)
            await log_task(training_data.task_id, task.hotkey, message)
            await complete_task(training_data.task_id, task.hotkey, success=False)
            raise RuntimeError(f"Image build failed: {exc}")

        await log_task(training_data.task_id, task.hotkey, f"Docker image built with tag: {tag}")

        env_server_url_str = None
        if task_type == TaskType.ENVIRONMENTTASK:
            logger.warning(
                "Environment tasks are no longer supported; skipping environment server setup",
                extra=log_labels,
            )
            await log_task(training_data.task_id, task.hotkey, "Environment tasks are no longer supported; skipping.")

        if model_prep_ran:
            model_for_container = get_anonymous_model_dir(training_data.model)
        else:
            model_for_container = training_data.model

        if task_type == TaskType.IMAGETASK:
            container = await asyncio.wait_for(
                run_trainer_container_image(
                    task_id=training_data.task_id,
                    tag=tag,
                    model=model_for_container,
                    dataset_zip=training_data.dataset_zip,
                    model_type=training_data.model_type,
                    expected_repo_name=training_data.expected_repo_name,
                    hours_to_complete=training_data.hours_to_complete,
                    hotkey=task.hotkey,
                    trigger_word=training_data.trigger_word if training_data.trigger_word else None,
                    baseline_stats=training_data.baseline_stats,
                    log_labels=log_labels,
                    gpu_ids=task.gpu_ids,
                ),
                timeout=60,
            )
        else:
            use_kl = training_data.use_kl if isinstance(training_data, TrainRequestText) else False
            kl_coef = training_data.kl_coef if isinstance(training_data, TrainRequestText) else None
            container = await asyncio.wait_for(
                run_trainer_container_text(
                    task_id=training_data.task_id,
                    hotkey=task.hotkey,
                    tag=tag,
                    model=model_for_container,
                    dataset=training_data.dataset,
                    dataset_type=training_data.dataset_type,
                    task_type=task_type,
                    file_format=training_data.file_format,
                    expected_repo_name=training_data.expected_repo_name,
                    hours_to_complete=training_data.hours_to_complete,
                    baseline_stats=training_data.baseline_stats,
                    log_labels=log_labels,
                    gpu_ids=task.gpu_ids,
                    env_server_urls=env_server_url_str,
                    miner_datasets=task.requested_datasets,
                    use_kl=use_kl,
                    kl_coef=kl_coef,
                ),
                timeout=60,
            )

        await update_container_name(training_data.task_id, task.hotkey, container.name)
        # Align cleanup's stale-timeout clock with container.wait (not job-accept time).
        await update_training_started_at(training_data.task_id, task.hotkey)
        await log_task(training_data.task_id, task.hotkey, f"Container started: {container.name}")
        await log_task(training_data.task_id, task.hotkey, f"Waiting for container to finish (timeout={timeout_seconds})...")
        wait_task = asyncio.create_task(asyncio.to_thread(container.wait))
        done, pending = await asyncio.wait({wait_task}, timeout=timeout_seconds)
        await log_task(training_data.task_id, task.hotkey, "Container wait completed or timed out.")

        if wait_task in done:
            result = await wait_task
            logger.info(f"Container.wait() returned: {result}", extra=log_labels)
            status_code = result.get("StatusCode", -1)
            if status_code == 0:
                await log_task(training_data.task_id, task.hotkey, "Training completed successfully.")
                success = True
            else:
                logs = container.logs().decode("utf-8", errors="ignore")
                error_message = extract_container_error(logs)
                if error_message:
                    log_message = f"[ERROR] Training container failed | ExitCode: {status_code} | LastError: {error_message}"
                    await log_task(training_data.task_id, task.hotkey, log_message)
                    logger.error(f"Training container failed: {error_message}", extra=log_labels)
                await complete_task(training_data.task_id, task.hotkey, success=success)
                await log_task(training_data.task_id, task.hotkey, f"Training failed with status code {status_code}")
        else:
            await log_task(training_data.task_id, task.hotkey, f"Timeout reached ({timeout_seconds}s). Killing container...")
            # Report completion from _final_cleanup only, after the checkpoint upload, exactly like the
            # clean-exit path above. Calling complete_task(success=True) here marked the task SUCCESS
            # before upload_repo_to_hf ran, so the validator could start evaluating a repo whose weights
            # had not finished uploading yet.
            success = True

    except asyncio.CancelledError as cancel:
        cancel_log_message = "[INFO] Training cancelled."
        logger.info("Training cancelled", extra=log_labels)
        cancelled_exc = cancel
    except Exception as e:
        log_message = f"[ERROR] Job failed: {e}"
        await log_task(training_data.task_id, task.hotkey, log_message)
        logger.exception(f"Training job failed: {training_data.task_id}", extra=log_labels)
        await complete_task(training_data.task_id, task.hotkey, success=success)

    finally:

        async def _final_cleanup():
            nonlocal success

            if cancel_log_message:
                await log_task(training_data.task_id, task.hotkey, cancel_log_message)

            # Clean up all environment servers
            for srv in env_server_containers:
                try:
                    await asyncio.to_thread(srv.stop)
                    await asyncio.to_thread(srv.remove, force=True)
                except Exception as e:
                    logger.warning(f"Failed to cleanup server {srv.name}: {e}")

            if container and isinstance(container, Container):
                try:
                    await asyncio.to_thread(container.reload)
                    if container.status == "running":
                        await asyncio.to_thread(container.kill)
                    await asyncio.to_thread(container.remove, force=True)
                    await log_task(training_data.task_id, task.hotkey, f"Container {container.name} cleaned up.")

                except Exception as cleanup_err:
                    await log_task(training_data.task_id, task.hotkey, f"Error during container cleanup: {cleanup_err}")

            logger.info("Cleaning up", extra=log_labels)
            if tag:
                await asyncio.to_thread(delete_image_and_cleanup, tag)
                logger.info("Cleaned up Docker resources.", extra=log_labels)
            else:
                logger.info("No Docker image to clean up.", extra=log_labels)

            if success:
                try:
                    path_in_repo = cst.IMAGE_TASKS_HF_SUBFOLDER_PATH if task_type == TaskType.IMAGETASK else None
                    wandb_token = os.getenv("WANDB_TOKEN") if task_type != TaskType.IMAGETASK else None
                    await upload_repo_to_hf(
                        task_id=training_data.task_id,
                        hotkey=task.hotkey,
                        expected_repo_name=training_data.expected_repo_name,
                        huggingface_username=os.getenv("HUGGINGFACE_USERNAME"),
                        huggingface_token=os.getenv("HUGGINGFACE_TOKEN"),
                        model=training_data.model,
                        docker_labels=log_labels,
                        wandb_token=wandb_token,
                        path_in_repo=path_in_repo,
                    )

                    await log_task(training_data.task_id, task.hotkey, "Repo uploaded successfully.")
                except Exception as upload_err:
                    log_message = f"[ERROR] Upload container failed | ExitCode: Unknown | LastError: {upload_err}"
                    await log_task(training_data.task_id, task.hotkey, log_message)
                    success = False

            await complete_task(training_data.task_id, task.hotkey, success=success)

        try:
            await asyncio.shield(_final_cleanup())
        finally:
            if cancelled_exc:
                raise cancelled_exc
