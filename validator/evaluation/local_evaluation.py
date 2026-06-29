import asyncio
import glob
import io
import itertools
import json
import logging
import os
import random
import shutil
import tarfile
import uuid

import docker
from docker.types import Mount
from huggingface_hub import snapshot_download

import core.constants.docker as docker_cst
import core.constants.environments as env_cst
import validator.evaluation.constants as vcst
from core.constants.paths import CACHE_DIR_HUB
from core.downloads import download_s3_file
from core.logging import get_all_context_tags
from core.logging import get_environment_logger
from core.logging import get_logger
from core.logging import stream_container_logs
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import EnvironmentDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import GrpoDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.image_models import ImageModelType
from core.models.payload_models import DockerEvaluationResults
from validator.evaluation.evaluators.environment import _build_sglang_command
from validator.evaluation.evaluators.environment import _download_lora_with_retry
from validator.evaluation.evaluators.environment import _download_model_with_retry
from validator.evaluation.evaluators.environment import _merge_base_and_lora
from validator.evaluation.evaluators.environment import _run_environment_evaluation as _run_eval_environment_rollouts
from validator.evaluation.model_checks import check_for_lora
from validator.evaluation.model_checks import check_lora_has_added_tokens
from validator.evaluation.pvp.models import PvPEvalConfig
from validator.evaluation.pvp.models import PvPEvalMetadata
from validator.evaluation.pvp.models import PvPEvalResults
from validator.evaluation.pvp.models import PvPGroupResults
from validator.evaluation.pvp.models import PvPMatchupConfig
from validator.evaluation.pvp.models import PvPMode
from validator.evaluation.pvp.models import PvPModelSpec
from validator.evaluation.pvp.models import PvPPairResult
from validator.evaluation.result_processing import normalize_rewards_and_compute_loss
from validator.evaluation.result_processing import process_evaluation_results
from validator.evaluation.runtime import wait_for_basilica_health
from validator.tasks.datasets.constants import CONTAINER_EVAL_RESULTS_PATH
from validator.tasks.datasets.preparation import unzip_to_temp_path


logger = get_logger(__name__)


def _first_environment_name(dataset_type: EnvironmentDatasetType) -> env_cst.EnvironmentName | None:
    environment_names = dataset_type.environment_names or []
    return environment_names[0] if environment_names else None


def _is_intercode_environment(dataset_type: EnvironmentDatasetType) -> bool:
    env_name = _first_environment_name(dataset_type)
    return (
        env_name == env_cst.EnvironmentName.INTERCODE
        or getattr(env_name, "value", env_name) == env_cst.EnvironmentName.INTERCODE.value
    )


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


def _build_local_sglang_command(base_model: str, base_seed: int) -> str:
    return _build_sglang_command(base_model, base_seed)


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
    continuous_sft_remote_code_repo: str | None = None,
    continuous_sft_tokenizer_repo: str | None = None,
) -> DockerEvaluationResults:
    if isinstance(dataset_type, (InstructTextDatasetType, ChatTemplateDatasetType)):
        command = ["python", "-m", "validator.evaluation.evaluators.instruct_text"]
    elif isinstance(dataset_type, DpoDatasetType):
        command = ["python", "-m", "validator.evaluation.evaluators.dpo"]
    elif isinstance(dataset_type, GrpoDatasetType):
        return await run_evaluation_docker_grpo(dataset, models, original_model, dataset_type, file_format, gpu_ids)
    elif isinstance(dataset_type, EnvironmentDatasetType):
        gpu_id = gpu_ids[0] if gpu_ids else 0
        if _is_intercode_environment(dataset_type):
            return await run_evaluation_local_intercode(
                models,
                original_model,
                dataset_type,
                file_format=file_format,
                gpu_id=gpu_id,
                eval_seed=eval_seed,
            )
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
    if continuous_sft_remote_code_repo:
        environment[docker_cst.CONTINUOUS_SFT_REMOTE_CODE_REPO_ENV] = continuous_sft_remote_code_repo
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

    env_name = (dataset_type.environment_names or [None])[0]
    if env_name not in env_cst.ENVIRONMENT_CONFIGS:
        raise ValueError(f"Environment '{env_name}' not found. Supported: {[e.value for e in env_cst.EnvironmentName]}")

    env_config = env_cst.ENVIRONMENT_CONFIGS[env_name]
    task_id_min = env_config.task_id_min
    task_id_max = env_config.task_id_max
    num_seeds_override = os.getenv("ENV_EVAL_NUM_SEEDS", "").strip()
    if num_seeds_override:
        num_seeds = int(num_seeds_override)
    else:
        num_seeds = env_config.num_seeds
    env_image = env_config.env_image
    env_payload_extra = env_config.eval_payload_extra
    temperature = float(os.getenv("ENV_EVAL_TEMPERATURE", str(vcst.ENV_EVAL_TEMPERATURE)))

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
        env_logger = get_environment_logger(
            name=f"{repo_name}-{eval_id[:8]}",
            repo_id=repo,
            eval_id=eval_id,
            model=original_model,
        )
        local_env_server_port = int(os.getenv("LOCAL_ENV_SERVER_PORT", str(vcst.LOCAL_ENV_SERVER_PORT)))
        sglang_health_timeout = int(os.getenv("SGLANG_HEALTH_TIMEOUT", "1800"))
        env_health_timeout = int(os.getenv("ENV_SERVER_HEALTH_TIMEOUT", "600"))

        containers = {}
        lora_dir = None
        merged_model_dir = None
        sglang_log_task = None
        try:
            is_lora = await asyncio.to_thread(check_for_lora, repo, local_files_only=False)
            should_merge_lora = False
            if is_lora:
                should_merge_lora = await asyncio.to_thread(check_lora_has_added_tokens, repo, False)
            env_logger.info(
                "LoRA detection: is_lora=%s merge_lora_to_base=%s",
                is_lora,
                should_merge_lora,
            )

            inference_model_name = repo
            model_path_for_sglang = repo
            sglang_args = os.getenv("SGLANG_START_CMD")
            if sglang_args:
                env_logger.info("Using SGLANG_START_CMD override from environment")

            if not sglang_args and is_lora and not should_merge_lora:
                env_logger.info("LoRA detected: using base snapshot + native SGLang LoRA loading")
                model_path_for_sglang = await asyncio.to_thread(_download_model_with_retry, original_model)
                safe_lora_name = repo.replace("/", "_")
                lora_dir = f"/tmp/sglang_lora/{safe_lora_name}"
                await asyncio.to_thread(_download_lora_with_retry, repo, lora_dir)
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
                inference_model_name = f"{original_model}:trained_lora"
                sglang_args = (
                    _build_local_sglang_command(model_path_for_sglang, base_seed)
                    + " --enable-lora --lora-paths trained_lora=/lora/trained_lora --lora-backend triton"
                )
            elif not sglang_args and is_lora and should_merge_lora:
                env_logger.info("LoRA detected: merging into base before SGLang launch")
                base_path = await asyncio.to_thread(_download_model_with_retry, original_model)
                safe_lora_name = repo.replace("/", "_")
                lora_dir = f"/tmp/sglang_lora/{safe_lora_name}"
                await asyncio.to_thread(_download_lora_with_retry, repo, lora_dir)
                merged_model_dir = f"/tmp/sglang_merged/{safe_lora_name}"
                model_path_for_sglang = await asyncio.to_thread(
                    _merge_base_and_lora, base_path, lora_dir, merged_model_dir
                )
                inference_model_name = repo
                sglang_args = _build_local_sglang_command(model_path_for_sglang, base_seed)
            elif not sglang_args:
                env_logger.info(f"Base model: {repo}")
                model_path_for_sglang = await asyncio.to_thread(_download_model_with_retry, repo)
                inference_model_name = repo
                sglang_args = _build_local_sglang_command(model_path_for_sglang, base_seed)

            local_sglang_port = int(os.getenv("SGLANG_PORT", str(vcst.LOCAL_ENV_SGLANG_PORT)))
            sglang_container_name = f"{eval_id}-sglang"
            env_container_name = f"{eval_id}-env"

            sglang_volumes = {vcst.LOCAL_ENV_HF_CACHE_PATH: {"bind": "/hf", "mode": "rw"}}
            if is_lora and lora_dir:
                sglang_volumes[lora_dir] = {"bind": "/lora/trained_lora", "mode": "ro"}
            if os.path.isabs(model_path_for_sglang) and os.path.exists(model_path_for_sglang):
                sglang_volumes[model_path_for_sglang] = {"bind": model_path_for_sglang, "mode": "ro"}

            try:
                current_ws = int(os.environ.get("SGLANG_FLASHINFER_WORKSPACE_SIZE", "0") or "0")
            except ValueError:
                current_ws = 0
            flashinfer_workspace_size = str(max(current_ws, vcst.SGLANG_FLASHINFER_WORKSPACE_MIN_BYTES))

            env_logger.info(f"Starting SGLang container: {sglang_container_name} (GPU {gpu_id})")
            sglang_container = await asyncio.to_thread(
                docker_client.containers.run,
                "lmsysorg/sglang:latest",
                command=sglang_args,
                name=sglang_container_name,
                detach=True,
                network=vcst.LOCAL_ENV_DOCKER_NETWORK,
                ports={f"{local_sglang_port}/tcp": local_sglang_port},
                device_requests=[docker.types.DeviceRequest(device_ids=[str(gpu_id)], capabilities=[["gpu"]])],
                environment={
                    "HF_HOME": "/hf",
                    "TRANSFORMERS_CACHE": "/hf",
                    "HUGGINGFACE_HUB_CACHE": "/hf",
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "PYTHONHASHSEED": str(base_seed),
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "NVIDIA_TF32_OVERRIDE": "0",
                    "SGLANG_FLASHINFER_WORKSPACE_SIZE": flashinfer_workspace_size,
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

            sglang_host_url = f"http://localhost:{local_sglang_port}"
            await asyncio.to_thread(wait_for_basilica_health, sglang_host_url, sglang_health_timeout)
            env_logger.info(f"SGLang ready at {sglang_host_url}")

            env_logger.info(f"Starting environment container: {env_container_name}")
            env_container = await asyncio.to_thread(
                docker_client.containers.run,
                env_image,
                name=env_container_name,
                detach=True,
                network=vcst.LOCAL_ENV_DOCKER_NETWORK,
                ports={"8000/tcp": local_env_server_port},
                remove=False,
            )
            containers["env"] = env_container

            env_host_url = f"http://localhost:{local_env_server_port}"
            await asyncio.to_thread(wait_for_basilica_health, env_host_url, env_health_timeout, "/health")
            env_logger.info(f"Environment server ready at {env_host_url}")

            sglang_internal_url = f"http://{sglang_container_name}:{local_sglang_port}"
            avg_score = await _run_environment_evaluation(
                sglang_internal_url,
                env_host_url,
                eval_seeds,
                task_id_max,
                temperature,
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
            if merged_model_dir and os.path.exists(merged_model_dir):
                try:
                    shutil.rmtree(merged_model_dir)
                except Exception as e:
                    env_logger.warning(f"Failed to cleanup merged model dir: {e}")

    docker_client.close()
    logger.info(f"Local environment evaluation results: {evaluation_results}")
    return process_evaluation_results(evaluation_results, is_image=False)


async def run_evaluation_local_intercode(
    models: list[str],
    original_model: str,
    dataset_type: EnvironmentDatasetType,
    file_format: FileFormat = FileFormat.JSON,
    gpu_id: int = 0,
    eval_seed: int | None = None,
) -> DockerEvaluationResults:
    logger.info(f"Starting local Docker InterCode evaluation for {len(models)} repos: {models}")
    if not _is_intercode_environment(dataset_type):
        actual_env_name = _first_environment_name(dataset_type)
        raise ValueError(
            f"run_evaluation_local_intercode requires environment_names=['intercode'], got {actual_env_name!r}"
        )

    base_seed = eval_seed if eval_seed is not None else vcst.ENV_EVAL_DEFAULT_SEED
    temperature = float(os.getenv("ENV_EVAL_TEMPERATURE", str(vcst.ENV_EVAL_TEMPERATURE)))
    dataset_type_str = dataset_type.model_dump_json()
    cache_dir = os.path.expanduser(CACHE_DIR_HUB)
    volume_bindings = {
        cache_dir: {"bind": "/root/.cache/huggingface/hub", "mode": "rw"},
    }
    command = ["python", "-m", "validator.evaluation.evaluators.intercode"]
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
        "ENVIRONMENT_NAME": env_cst.EnvironmentName.INTERCODE.value,
        "EVAL_SEED": str(base_seed),
        "ENV_EVAL_TEMPERATURE": str(temperature),
    }

    client = docker.from_env()
    evaluation_results: dict[str, dict | str | int] = {}
    try:
        for repo in models:
            container = None
            environment = dict(base_environment)
            environment["MODELS"] = repo
            try:
                logger.info(f"Running local InterCode evaluation for repo: {repo}")
                container = await asyncio.to_thread(
                    client.containers.run,
                    docker_cst.VALIDATOR_DOCKER_IMAGE_INTERCODE,
                    command=command,
                    environment=environment,
                    volumes=volume_bindings,
                    runtime="nvidia",
                    device_requests=[
                        docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gpu_id)])
                    ],
                    ipc_mode="host",
                    detach=True,
                )
                log_task = asyncio.create_task(
                    asyncio.to_thread(stream_container_logs, container, None, get_all_context_tags())
                )
                result = await asyncio.to_thread(container.wait)
                log_task.cancel()

                if result["StatusCode"] != 0:
                    status_code = result["StatusCode"]
                    raise Exception(f"InterCode container for {repo} exited with non-zero status: {status_code}")

                eval_results = await get_evaluation_results(container)
                raw_result = eval_results.get(repo)
                if raw_result is None:
                    raise Exception(f"InterCode results missing repo key {repo!r}: {eval_results}")
                evaluation_results[repo] = raw_result
                if "model_params_count" in eval_results and "model_params_count" not in evaluation_results:
                    evaluation_results["model_params_count"] = eval_results["model_params_count"]
            except Exception as e:
                logger.error(f"Failed to evaluate InterCode repo {repo}: {str(e)}", exc_info=True)
                evaluation_results[repo] = f"Evaluation failed: {str(e)}"
            finally:
                if container is not None:
                    try:
                        await asyncio.to_thread(container.remove, force=True)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup InterCode container for {repo}: {e}")
    finally:
        try:
            await cleanup_resources(client)
        except Exception as e:
            logger.info(f"A problem with cleaning up {e}")
        client.close()

    logger.info(f"Local InterCode evaluation results: {evaluation_results}")
    return process_evaluation_results(evaluation_results, is_image=False)


def _normalize_environment_name(environment_name: env_cst.EnvironmentName | str) -> env_cst.EnvironmentName:
    if isinstance(environment_name, env_cst.EnvironmentName):
        return environment_name
    return env_cst.EnvironmentName(environment_name)


def _get_shared_pvp_eval_image(environment_names: list[env_cst.EnvironmentName]) -> str:
    configs = [env_cst.ENVIRONMENT_CONFIGS[environment_name] for environment_name in environment_names]
    image = configs[0].tournament_eval_image
    if not all(config.tournament_eval_image == image for config in configs):
        raise ValueError(
            "All PvP environments must share a tournament_eval_image, got: "
            f"{[(env.value, config.tournament_eval_image) for env, config in zip(environment_names, configs)]}"
        )
    return image


async def run_evaluation_local_pvp_pair(
    model_a_repo: str,
    model_b_repo: str,
    hotkey_a: str,
    hotkey_b: str,
    base_model: str,
    environment_names: list[env_cst.EnvironmentName | str],
    gpu_ids: list[int],
    seed: int,
    image: str | None = None,
    temperature: float = 0.0,
) -> PvPGroupResults:
    """Run one PvP pair locally using the same PvP evaluator container as production."""
    if len(gpu_ids) < 2:
        raise ValueError("PvP local evaluation requires at least two GPU IDs, for example: --gpu_ids 0 1")

    pvp_envs = [_normalize_environment_name(environment_name) for environment_name in environment_names]
    if not pvp_envs:
        raise ValueError("At least one PvP environment is required")

    image = image or _get_shared_pvp_eval_image(pvp_envs)
    time_budget = float(os.getenv("PVP_MATCHUP_TIME_BUDGET_SECONDS", str(vcst.PVP_MATCHUP_TIME_BUDGET_SECONDS)))
    matchups = {environment_name: PvPMatchupConfig(time_budget_seconds=time_budget) for environment_name in pvp_envs}
    pvp_config = PvPEvalConfig(
        mode=PvPMode.PAIR,
        model_a=PvPModelSpec(
            repo=model_a_repo,
            original_model=base_model,
            gpu_id=0,
            port=vcst.PVP_SGLANG_PORT_A,
        ),
        model_b=PvPModelSpec(
            repo=model_b_repo,
            original_model=base_model,
            gpu_id=1,
            port=vcst.PVP_SGLANG_PORT_B,
        ),
        matchups=matchups,
        seed=seed,
        temperature=temperature,
    )

    logger.info(
        "Prepared local PvP pair: hotkey_a=%s repo_a=%s hotkey_b=%s repo_b=%s envs=%s seed=%s temperature=%s image=%s GPUs=%s",
        hotkey_a,
        model_a_repo,
        hotkey_b,
        model_b_repo,
        [env.value for env in pvp_envs],
        seed,
        temperature,
        image,
        gpu_ids[:2],
    )
    logger.debug("PvP config JSON: %s", pvp_config.model_dump_json())

    logger.info("Running local PvP pair %s vs %s on GPUs %s", hotkey_a, hotkey_b, gpu_ids[:2])
    client = docker.from_env()
    container = None
    try:
        environment = {
            vcst.PVP_CONFIG_ENV_VAR: pvp_config.model_dump_json(),
            "TRANSFORMERS_ALLOW_TORCH_LOAD": "true",
            **vcst.HF_CONTAINER_ENV,
        }
        volume_bindings = {
            os.path.expanduser(CACHE_DIR_HUB): {"bind": "/root/.cache/huggingface/hub", "mode": "rw"},
        }
        container = await asyncio.to_thread(
            client.containers.run,
            image,
            command=["python", "-m", "validator.evaluation.pvp"],
            environment=environment,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[
                docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=[str(gpu_id) for gpu_id in gpu_ids[:2]])
            ],
            ipc_mode="host",
            detach=True,
        )
        log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, None, get_all_context_tags()))
        result = await asyncio.to_thread(container.wait)
        log_task.cancel()

        if result["StatusCode"] != 0:
            raise Exception(f"PvP container for {hotkey_a}:{hotkey_b} exited with status {result['StatusCode']}")

        raw_results = await get_json_results_from_container(container, vcst.PVP_RESULTS_PATH)
        pair_eval = PvPEvalResults.model_validate(raw_results)
        return PvPGroupResults(
            base_model=base_model,
            hotkeys=[hotkey_a, hotkey_b],
            pair_results=[
                PvPPairResult(
                    hotkey_a=hotkey_a,
                    hotkey_b=hotkey_b,
                    results=pair_eval.results,
                )
            ],
            metadata=pair_eval.metadata,
        )
    finally:
        if container is not None:
            try:
                await asyncio.to_thread(container.remove, force=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup PvP container for {hotkey_a}:{hotkey_b}: {e}")
        client.close()


async def run_evaluation_local_pvp_pairs(
    miner_repos: dict[str, str],
    original_model: str,
    environment_names: list[env_cst.EnvironmentName | str],
    gpu_ids: list[int],
    eval_seed: int | None = None,
    temperature: float | None = None,
) -> PvPGroupResults:
    """Run local round-robin PvP evaluation for every hotkey pair in miner_repos."""
    if len(miner_repos) < 2:
        raise ValueError("PvP evaluation requires at least two hotkeys with repos")

    pvp_envs = [_normalize_environment_name(environment_name) for environment_name in environment_names]
    invalid_envs = [
        environment_name
        for environment_name in pvp_envs
        if env_cst.ENVIRONMENT_CONFIGS[environment_name].eval_type != env_cst.EvalType.PVP
    ]
    if invalid_envs:
        raise ValueError(f"Non-PvP environments cannot be run with PvP evaluation: {[env.value for env in invalid_envs]}")

    seed = eval_seed if eval_seed is not None else vcst.ENV_EVAL_DEFAULT_SEED
    eval_temperature = (
        temperature
        if temperature is not None
        else float(os.getenv("ENV_EVAL_TEMPERATURE", str(vcst.ENV_EVAL_TEMPERATURE)))
    )
    image = _get_shared_pvp_eval_image(pvp_envs)
    pair_results: list[PvPPairResult] = []

    hotkeys = list(miner_repos.keys())
    pair_count = len(list(itertools.combinations(hotkeys, 2)))
    logger.info(
        "Prepared local PvP evaluation: hotkeys=%s envs=%s base_model=%s seed=%s temperature=%s image=%s pair_count=%d",
        hotkeys,
        [env.value for env in pvp_envs],
        original_model,
        seed,
        eval_temperature,
        image,
        pair_count,
    )
    for hotkey_a, hotkey_b in itertools.combinations(hotkeys, 2):
        pair_group = await run_evaluation_local_pvp_pair(
            model_a_repo=miner_repos[hotkey_a],
            model_b_repo=miner_repos[hotkey_b],
            hotkey_a=hotkey_a,
            hotkey_b=hotkey_b,
            base_model=original_model,
            environment_names=pvp_envs,
            gpu_ids=gpu_ids,
            seed=seed,
            image=image,
            temperature=eval_temperature,
        )
        pair_results.extend(pair_group.pair_results)

    return PvPGroupResults(
        base_model=original_model,
        hotkeys=hotkeys,
        pair_results=pair_results,
        metadata=PvPEvalMetadata(seed=seed, temperature=eval_temperature, wall_time_seconds=0),
    )


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
    return await _run_eval_environment_rollouts(
        sglang_url=sglang_url,
        env_url=env_url,
        eval_seeds=eval_seeds,
        task_id_max=data_len_range,
        task_id_min=task_id_min,
        inference_model_name=inference_model_name,
        temperature=temperature,
        env_payload_extra=env_payload_extra or {},
    )


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
        Mount(target=f"{base_path}/checkpoints", source=CACHE_DIR_HUB, type="bind", read_only=False),
        Mount(target=f"{base_path}/diffusers", source=CACHE_DIR_HUB, type="bind", read_only=False),
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
                    docker_cst.VALIDATOR_DOCKER_IMAGE_DIFFUSION,
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
