import asyncio
import io
import json
import os
import shutil
import tarfile

import docker
from docker.types import Mount
from huggingface_hub import snapshot_download

import core.constants.docker as docker_cst
import validator.evaluation.constants as vcst
from core.constants.paths import CACHE_DIR_HUB
from core.downloads import download_s3_file
from core.logging import get_all_context_tags
from core.logging import get_logger
from core.logging import stream_container_logs
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import GrpoDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.image_models import ImageModelType
from core.models.payload_models import DockerEvaluationResults
from validator.evaluation.result_processing import normalize_rewards_and_compute_loss
from validator.evaluation.result_processing import process_evaluation_results
from validator.tasks.datasets.constants import CONTAINER_EVAL_RESULTS_PATH
from validator.tasks.datasets.preparation import unzip_to_temp_path


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


async def get_json_results_from_container(container, results_path: str):
    archive_data = await asyncio.to_thread(container.get_archive, results_path)
    tar_stream = archive_data[0]
    results_filename = os.path.basename(results_path)

    file_like_object = io.BytesIO()
    for chunk in tar_stream:
        file_like_object.write(chunk)
    file_like_object.seek(0)

    with tarfile.open(fileobj=file_like_object) as tar:
        members = tar.getnames()
        logger.debug(f"Tar archive members: {members}")
        eval_results_file = None
        for member_info in tar.getmembers():
            if member_info.name.endswith(results_filename):
                eval_results_file = tar.extractfile(member_info)
                break

        if eval_results_file is None:
            raise Exception(f"Evaluation results file {results_filename} not found in tar archive")

        eval_results_content = eval_results_file.read().decode("utf-8")
        return json.loads(eval_results_content)


async def get_evaluation_results(container):
    return await get_json_results_from_container(container, CONTAINER_EVAL_RESULTS_PATH)


async def run_evaluation_docker_text(
    dataset: str,
    models: list[str],
    original_model: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType | ChatTemplateDatasetType,
    file_format: FileFormat,
    gpu_ids: list[int],
    eval_seed: int | None = None,
    continuous_sft_tokenizer_repo: str | None = None,
) -> DockerEvaluationResults:
    if isinstance(dataset_type, (InstructTextDatasetType, ChatTemplateDatasetType)):
        command = ["python", "-m", "validator.evaluation.evaluators.instruct_text"]
    elif isinstance(dataset_type, DpoDatasetType):
        command = ["python", "-m", "validator.evaluation.evaluators.dpo"]
    elif isinstance(dataset_type, GrpoDatasetType):
        return await run_evaluation_docker_grpo(dataset, models, original_model, dataset_type, file_format, gpu_ids)
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
    if continuous_sft_tokenizer_repo:
        environment[docker_cst.CONTINUOUS_SFT_TOKENIZER_REPO_ENV] = continuous_sft_tokenizer_repo
    logger.info(f"Running {task_type} evaluation for models: {models}")

    volume_bindings = {
        dataset_dir: {"bind": "/workspace/input_data", "mode": "ro"},
        os.path.expanduser(CACHE_DIR_HUB): {"bind": "/root/.cache/huggingface/hub", "mode": "rw"},
    }

    container = None
    retry_delay = 5.0
    try:
        while True:
            try:
                container = await asyncio.to_thread(
                    client.containers.run,
                    docker_cst.VALIDATOR_DOCKER_IMAGE,
                    command=command,
                    environment=environment,
                    volumes=volume_bindings,
                    runtime="nvidia",
                    device_requests=[
                        docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gid) for gid in gpu_ids])
                    ],
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
                logger.error(
                    f"Failed to retrieve {task_type} evaluation results: {str(e)}, retrying in {retry_delay}s...",
                    exc_info=True,
                )
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
    cache_dir = os.path.expanduser(CACHE_DIR_HUB)
    await asyncio.to_thread(snapshot_download, repo_id=original_model, cache_dir=cache_dir, ignore_patterns=None)

    command = ["python", "-m", "validator.evaluation.evaluators.grpo"]
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
        os.path.expanduser(CACHE_DIR_HUB): {"bind": "/root/.cache/huggingface/hub", "mode": "rw"},
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
                    docker_cst.VALIDATOR_DOCKER_IMAGE,
                    command=command,
                    environment=environment,
                    volumes=volume_bindings,
                    runtime="nvidia",
                    device_requests=[
                        docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gid) for gid in gpu_ids])
                    ],
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


async def run_evaluation_docker_image(
    test_split_url: str,
    original_model_repo: str,
    models: list[str],
    model_type: ImageModelType,
    gpu_ids: list[int],
    docker_image: str | None = None,
) -> DockerEvaluationResults:
    image = docker_image or docker_cst.VALIDATOR_DOCKER_IMAGE_DIFFUSION
    raw_data = await download_s3_file(test_split_url)
    test_split_path = unzip_to_temp_path(raw_data)
    dataset_dir = os.path.abspath(test_split_path)
    container_dataset_path = "/workspace/input_data"

    client = docker.from_env()
    mounts = [
        Mount(target=container_dataset_path, source=dataset_dir, type="bind", read_only=True),
        Mount(target="/root/.cache/huggingface/hub", source=CACHE_DIR_HUB, type="bind", read_only=False),
    ]
    environment = {
        "DATASET": container_dataset_path,
        "MODELS": ",".join(models),
        "ORIGINAL_MODEL_REPO": original_model_repo,
        "MODEL_TYPE": model_type.value,
        **vcst.HF_CONTAINER_ENV_IMAGE,
    }
    command = ["/app/start.sh"]

    container = None
    retry_delay = 5.0
    try:
        while True:
            try:
                container = await asyncio.to_thread(
                    client.containers.run,
                    image,
                    command=command,
                    mounts=mounts,
                    environment=environment,
                    runtime="nvidia",
                    device_requests=[
                        docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gid) for gid in gpu_ids])
                    ],
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
