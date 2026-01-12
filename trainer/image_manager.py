import asyncio
import json
import os
import re
import uuid

import docker
from docker.errors import APIError
from docker.errors import BuildError
from docker.models.containers import Container

import trainer.utils.training_paths as train_paths
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import EnvironmentDatasetType
from core.models.utility_models import TaskType
from trainer import constants as cst
from trainer.tasks import complete_task
from trainer.tasks import log_task
from trainer.tasks import update_wandb_url
from trainer.utils.trainer_logging import logger
from trainer.utils.misc import build_wandb_env
from trainer.utils.misc import extract_container_error
from validator.utils.logging import get_all_context_tags
from validator.utils.logging import stream_container_logs
from validator.utils.logging import stream_image_build_logs


# logger = get_logger(__name__)


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
) -> tuple[str, str | None]:
    client: docker.DockerClient = docker.from_env()

    if tag is None:
        tag = f"standalone-image-trainer:{uuid.uuid4()}" if is_image_task else f"standalone-text-trainer:{uuid.uuid4()}"

    logger.info(f"Building Docker image '{tag}', Dockerfile path: {dockerfile_path}, Context Path: {context_path}...")

    try:
        build_output = client.api.build(
            path=context_path,
            dockerfile=dockerfile_path,
            tag=tag,
            nocache=no_cache,
            decode=True,
        )
        stream_image_build_logs(build_output, logger=logger, log_context=log_labels)

        logger.info("Docker image built successfully.", extra=log_labels)
        return tag, None
    except (BuildError, APIError) as e:
        logger.error(f"Docker build failed: {str(e)}", extra=log_labels)
        return None, str(e)


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


async def wait_for_env_container_ip(environment_server_container) -> str:
    ip_address = None
    for _ in range(10):
        environment_server_container.reload()
        settings = environment_server_container.attrs.get("NetworkSettings", {})

        # First, try to get IP from the specific internal_bridge network
        networks = settings.get("Networks", {})
        if cst.INTERNAL_BRIDGE_NAME in networks:
            ip_address = networks[cst.INTERNAL_BRIDGE_NAME].get("IPAddress")
        
        # Fallback: try the direct field (for default bridge network)
        if not ip_address:
            ip_address = settings.get("IPAddress")

        # Fallback: check any other network (shouldn't happen, but safer)
        if not ip_address:
            for net_name in networks:
                ip_address = networks[net_name].get("IPAddress")
                if ip_address:
                    break

        if ip_address:
            break
        await asyncio.sleep(0.5)

    if not ip_address:
        raise RuntimeError("Environment server started but could not retrieve internal IP.")

    return ip_address


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
    log_labels: dict[str, str] | None = None,
    gpu_ids: list[int] = [0],
) -> Container:
    client: docker.DockerClient = docker.from_env()

    ensure_internal_network()

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

    container_name = f"image-trainer-{uuid.uuid4().hex}"

    # Calculate resources based on GPU count
    memory_limit, cpu_limit_nanocpus = calculate_container_resources(gpu_ids)

    # Set shared memory size based on GPU count
    shm_size = "16g" if len(gpu_ids) >= 4 else "8g"

    max_retries = cst.CONTAINER_START_MAX_RETRIES
    retry_delay = cst.CONTAINER_START_RETRY_DELAY_SECONDS

    for attempt in range(max_retries):
        try:
            container: Container = client.containers.run(
                image=tag,
                command=command,
                volumes={
                    cst.VOLUME_NAMES[0]: {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"},
                    cst.VOLUME_NAMES[1]: {"bind": cst.CACHE_ROOT_PATH, "mode": "ro"},
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
                environment={"TRANSFORMERS_CACHE": cst.HUGGINGFACE_CACHE_PATH},
                detach=True,
            )

            log_streaming_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))
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
    log_labels: dict[str, str] | None = None,
    gpu_ids: list[int] = [0],
    env_server_urls: str | None = None,
) -> Container:
    client: docker.DockerClient = docker.from_env()

    ensure_internal_network()

    environment = build_wandb_env(task_id, hotkey)
    if env_server_urls:
        environment["ENVIRONMENT_SERVER_URLS"] = env_server_urls

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
            container: Container = client.containers.run(
                image=tag,
                command=command,
                volumes={
                    cst.VOLUME_NAMES[0]: {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"},
                    cst.VOLUME_NAMES[1]: {"bind": cst.CACHE_ROOT_PATH, "mode": "ro"},  # NOTE: may require rw fixing
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

            log_streaming_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))
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


async def create_volumes_if_dont_exist():
    client: docker.DockerClient = docker.from_env()
    volume_names = cst.VOLUME_NAMES
    for volume_name in volume_names:
        try:
            volume = client.volumes.get(volume_name)
        except docker.errors.NotFound:
            volume = client.volumes.create(name=volume_name)
            logger.info(f"Volume '{volume_name}' created.")


def run_downloader_container(
    task_id: str,
    model: str,
    dataset_url: str,
    task_type: TaskType,
    hotkey: str,
    file_format: FileFormat | None = None,
    model_type: ImageModelType | None = None,
    log_labels: dict[str, str] | None = None,
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

    container_name = f"downloader-{task_id}-{str(uuid.uuid4())[:8]}"
    container = None

    try:
        logger.info(f"Starting downloader container: {container_name}", extra=log_labels)
        container = client.containers.run(
            image=cst.TRAINER_DOWNLOADER_DOCKER_IMAGE,
            name=container_name,
            command=command,
            labels=log_labels,
            volumes={cst.VOLUME_NAMES[1]: {"bind": "/cache", "mode": "rw"}},
            remove=False,
            detach=True,
        )

        stream_container_logs(container, get_all_context_tags())

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


async def run_environment_server_container(environment_name: str, log_labels: dict) -> Container:
    client = docker.from_env()

    ensure_internal_network()

    container_name = f"environment-server-{uuid.uuid4().hex[:8]}"
    logger.info(f"Starting env server container: {container_name}", extra=log_labels)

    if environment_name == "alfworld":
        # Run the alfworld server
        container = await asyncio.to_thread(
            client.containers.run,
            image="affinefoundation/agentgym:alfworld",
            name=container_name,
            detach=True,
            labels=log_labels,
            network=cst.INTERNAL_BRIDGE_NAME,
        )
        return container
    else:
        return None


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
            cst.VOLUME_NAMES[0]: {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"},
            cst.VOLUME_NAMES[1]: {"bind": cst.CACHE_ROOT_PATH, "mode": "rw"},
        }

        container_name = f"hf-upload-{uuid.uuid4().hex}"

        logger.info(f"Starting upload container {container_name} for task {task_id}...", extra=docker_labels)

        container = client.containers.run(
            image=cst.HF_UPLOAD_DOCKER_IMAGE,
            environment=environment,
            volumes=volumes,
            labels=docker_labels,
            detach=True,
            remove=False,
            name=container_name,
        )

        log_streaming_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))

        result = container.wait()
        logs = container.logs().decode("utf-8", errors="ignore")
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
                container.reload()
                if container.status == "running":
                    container.kill()
                container.remove(force=True)
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
        model_type = training_data.model_type
        if model_type in [ImageModelType.Z_IMAGE, ImageModelType.QWEN_IMAGE]:
            return f"{local_repo_path}/{cst.DEFAULT_IMAGE_TOOLKIT_DOCKERFILE_PATH}"
        else:
            return f"{local_repo_path}/{cst.DEFAULT_IMAGE_DOCKERFILE_PATH}"

    else:
        return f"{local_repo_path}/{cst.DEFAULT_TEXT_DOCKERFILE_PATH}"


async def start_training_task(task: TrainerProxyRequest, local_repo_path: str):
    cancelled_exc: asyncio.CancelledError | None = None
    cancel_log_message: str | None = None

    try:
        training_data = task.training_data
        success = False
        container = None
        env_server_containers = []
        tag = None
        timeout_seconds = int(training_data.hours_to_complete * 3600)
        task_type = get_task_type(task)
        training_data.hours_to_complete = int(training_data.hours_to_complete)
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

        env_urls = []
        env_server_url_str = None
        if task_type == TaskType.ENVIRONMENTTASK:
            logger.info("Running Environment Server Containers", extra=log_labels)
            await log_task(training_data.task_id, task.hotkey, "Starting Environment Servers...")
            for gpu in task.gpu_ids:
                environment_server_container = await run_environment_server_container(
                    task.training_data.dataset_type.environment_name, log_labels
                )
                env_server_containers.append(environment_server_container)
                ip_address = await wait_for_env_container_ip(environment_server_container)
                env_urls.append(f"http://{ip_address}:8000")
            env_server_url_str = ",".join(env_urls)
            await log_task(training_data.task_id, task.hotkey, f"Environment servers ready.")

        if task_type == TaskType.IMAGETASK:
            container = await asyncio.wait_for(
                run_trainer_container_image(
                    task_id=training_data.task_id,
                    tag=tag,
                    model=training_data.model,
                    dataset_zip=training_data.dataset_zip,
                    model_type=training_data.model_type,
                    expected_repo_name=training_data.expected_repo_name,
                    hours_to_complete=training_data.hours_to_complete,
                    hotkey=task.hotkey,
                    trigger_word=training_data.trigger_word if training_data.trigger_word else None,
                    log_labels=log_labels,
                    gpu_ids=task.gpu_ids,
                ),
                timeout=60,
            )
        else:
            container = await asyncio.wait_for(
                run_trainer_container_text(
                    task_id=training_data.task_id,
                    hotkey=task.hotkey,
                    tag=tag,
                    model=training_data.model,
                    dataset=training_data.dataset,
                    dataset_type=training_data.dataset_type,
                    task_type=task_type,
                    file_format=training_data.file_format,
                    expected_repo_name=training_data.expected_repo_name,
                    hours_to_complete=training_data.hours_to_complete,
                    log_labels=log_labels,
                    gpu_ids=task.gpu_ids,
                    env_server_urls=env_server_url_str,
                ),
                timeout=60,
            )

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
            success = True
            await complete_task(training_data.task_id, task.hotkey, success=success)

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
                    srv.stop()
                    srv.remove(force=True)
                except Exception as e:
                    logger.warning(f"Failed to cleanup server {srv.name}: {e}")

            if container and isinstance(container, Container):
                try:
                    container.reload()
                    if container.status == "running":
                        container.kill()
                    container.remove(force=True)
                    await log_task(training_data.task_id, task.hotkey, f"Container {container.name} cleaned up.")

                except Exception as cleanup_err:
                    await log_task(training_data.task_id, task.hotkey, f"Error during container cleanup: {cleanup_err}")

            logger.info("Cleaning up", extra=log_labels)
            if tag:
                delete_image_and_cleanup(tag)
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