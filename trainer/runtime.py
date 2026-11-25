import asyncio
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from functools import partial
from pathlib import Path

import docker
from docker.errors import APIError
from docker.errors import BuildError
from docker.errors import ImageNotFound
from docker.errors import NotFound
from docker.models.containers import Container

import core.constants as core_cst
import trainer.training_paths as train_paths
from core.build_context import describe_large_files
from core.build_context import find_large_files
from core.constants.quant import QUANT_LARGE_FILE_REVIEW_BYTES
from core.logging import get_all_context_tags
from core.logging import stream_container_logs
from core.logging import stream_image_build_logs
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import DpoDatasetType
from core.models.dataset_models import EnvironmentDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import GrpoDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.image_models import FLUX_CLIP_ARTIFACT
from core.models.image_models import FLUX_T5_ARTIFACT
from core.models.image_models import FLUX_VAE_ARTIFACT
from core.models.image_models import ImageModelType
from core.models.image_models import ImageTrainingMode
from core.models.model_prep_models import BaselineStats
from core.models.payload_models import ModelPrepResponse
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestQuant
from core.models.payload_models import TrainRequestText
from core.models.quant_artifact_models import QuantDecodeResult
from core.models.quant_models import QuantCalibrationRequest
from core.models.quant_models import QuantFixedSpec
from core.models.quant_models import QuantTarget
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskType
from trainer import constants as cst
from trainer.host import build_wandb_env
from trainer.host import extract_container_error
from trainer.host_resources import container_resources
from trainer.host_resources import pinned_helper_kwargs
from trainer.image_builds import ReusedLayers
from trainer.image_builds import image_build_in_flight
from trainer.image_builds import layer_too_old
from trainer.image_builds import prune_build_leftovers
from trainer.job_checkpoints import job_checkpoints_host_dir
from trainer.job_checkpoints import job_checkpoints_volume
from trainer.job_state import complete_task
from trainer.job_state import is_gpu_throttle_failure
from trainer.job_state import log_task
from trainer.job_state import update_container_name
from trainer.job_state import update_quant_decode_result
from trainer.job_state import update_training_started_at
from trainer.job_state import update_wandb_url
from trainer.model_artifacts import get_anonymous_model_dir
from trainer.output_reclaim import clear_job_checkpoints
from trainer.output_reclaim import reclaim_output_dir
from trainer.quant_calibration import write_quant_calibration_file
from trainer.quant_decode import upload_retry
from trainer.quant_decode.job import JOB_BUSY_FAILURE
from trainer.quant_decode.job import run_freeform_decode
from trainer.quant_decode.sandbox import sanitize_failure
from trainer.quant_upload_check import QuantUploadVerdict
from trainer.quant_upload_check import check_quant_output
from trainer.quant_upload_check import quant_output_max_bytes
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


# The classic builder runs each RUN step in a container of its own. It names the container on the line after the
# step's own line, before the step produces any output:
#   Step 4/9 : RUN pip install ...
#    ---> Running in 0123456789ab
_BUILD_STEP_LINE = re.compile(r"^Step \d+/\d+ : ")
_STEP_CONTAINER_LINE = re.compile(r"^\s*--->\s+Running in ([0-9a-f]{12,64})\s*$")
_BUILD_STREAM_LINE_MAX_CHARS = 64 * 1024


class BuildStepContainers:
    """Follows one build's stream to know its running step container, so the build can be stopped.

    Killing a step's container fails that step, which ends the build on the daemon. Once stopped, any step the build
    still starts is killed as soon as the stream names it. Nothing is removed.

    The stream also carries the steps' own output, so a container line counts only as the first one after a step line,
    and a container under one of the trainer's own name prefixes is never killed.
    """

    def __init__(self, client: docker.DockerClient, log_labels: dict[str, str] | None = None):
        self._client = client
        self._log_labels = log_labels
        self._current: str | None = None
        self._stopped = False
        self._buffer = ""
        self._awaiting_step_container = False

    def watch(self, build_output: Iterator[dict]) -> Iterator[dict]:
        """Pass the stream through. `stop` is called from the code reading it, so there is no locking."""
        for chunk in build_output:
            text = chunk.get("stream") if isinstance(chunk, dict) else None
            if isinstance(text, str):
                self._feed(text)
            yield chunk

    def _feed(self, text: str) -> None:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._observe(line)
        if len(self._buffer) > _BUILD_STREAM_LINE_MAX_CHARS:
            self._buffer = ""

    def _observe(self, line: str) -> None:
        if _BUILD_STEP_LINE.match(line):
            self._awaiting_step_container = True
            return
        named = _STEP_CONTAINER_LINE.match(line) if self._awaiting_step_container else None
        if named is None:
            return
        self._awaiting_step_container = False
        self._current = named.group(1)
        if self._stopped:
            self._kill(self._current)

    def stop(self) -> None:
        self._stopped = True
        if self._current is not None:
            self._kill(self._current)

    def _kill(self, container_id: str) -> None:
        """Kill one step container. The stream names it between create and start, and a container that has not
        started cannot be killed (409), so that case is retried until the kill lands or the container is gone."""
        deadline = time.monotonic() + cst.BUILD_STOP_KILL_TIMEOUT_SECONDS
        while True:
            try:
                container = self._client.api.inspect_container(container_id)
            except APIError:  # NotFound included: the step ended on its own
                return
            if container["Name"].lstrip("/").startswith(cst.TRAINER_CONTAINER_NAME_PREFIXES):
                return
            try:
                self._client.api.kill(container_id)
                logger.info(f"Stopped image build: killed step container {container_id[:12]}", extra=self._log_labels)
                return
            except NotFound:
                return
            except APIError as e:
                not_started = e.status_code == 409 and container["State"]["Status"] == "created"
                if not not_started or time.monotonic() >= deadline:
                    logger.info(f"Image build step container {container_id[:12]} not killed: {e}", extra=self._log_labels)
                    return
            time.sleep(cst.BUILD_STOP_KILL_RETRY_SECONDS)


class BuildStream:
    """One build's stream. A build that takes a too-old layer from the cache is stopped and run again without the cache.

    Every layer the build reuses is checked (image_builds.ReusedLayers). The first build is read to its end before the
    second starts, so it can no longer tag the image, and a tag it did set is removed: the image must come from the
    second build or not at all. Both builds run inside the caller's one time limit, and no second build starts once the
    caller has set `abandoned`.

    The stream is read by a thread that logs errors and carries on, so a build that could not be started or run again
    leaves its reason in `error`.
    """

    def __init__(self, client: docker.DockerClient, log_labels: dict[str, str] | None, **build_kwargs):
        self._client = client
        self._log_labels = log_labels
        self._build_kwargs = build_kwargs
        self.abandoned = threading.Event()
        self.error: str | None = None

    def __iter__(self) -> Iterator[dict]:
        stream = self._start(self._build_kwargs)
        if stream is None:
            return
        if self._build_kwargs["nocache"]:
            yield from stream
            return
        steps = BuildStepContainers(self._client, self._log_labels)
        reused = ReusedLayers()
        rebuild = False
        for chunk in steps.watch(stream):
            yield chunk
            for layer_id in [] if rebuild else reused.feed(chunk):
                if layer_too_old(self._client, layer_id):
                    logger.info(
                        f"Cached layer {layer_id} is over {cst.IMAGE_BUILD_CACHE_MAX_AGE_HOURS:g} h old: "
                        "building again without cache",
                        extra=self._log_labels,
                    )
                    rebuild = True
                    steps.stop()
                    break
        if not rebuild:
            return
        tag = self._build_kwargs["tag"]
        try:
            self._client.images.remove(image=tag, force=True, noprune=True)
        except ImageNotFound:
            pass
        except APIError as e:
            self.error = f"Docker image build failed: could not remove the image built on a too-old cached layer: {e}"
            logger.warning(self.error, extra=self._log_labels)
            return
        if self.abandoned.is_set():
            return
        stream = self._start({**self._build_kwargs, "nocache": True})
        if stream is not None:
            yield from stream

    def _start(self, build_kwargs: dict) -> Iterator[dict] | None:
        try:
            return self._client.api.build(**build_kwargs)
        except (BuildError, APIError) as e:
            self.error = str(e)
            return None


def build_docker_image(
    dockerfile_path: str,
    log_labels: dict[str, str] | None = None,
    context_path: str = ".",
    is_image_task: bool = False,
    tag: str = None,
    no_cache: bool | None = None,
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

    with image_build_in_flight():
        build_output = None
        executor = None
        try:
            build_output = BuildStream(
                client,
                log_labels,
                path=context_path,
                dockerfile=dockerfile_path,
                tag=tag,
                nocache=cst.IMAGE_BUILD_NO_CACHE if no_cache is None else no_cache,
                pull=cst.IMAGE_BUILD_PULL_BASE,
                # docker-py sends rm=False by default, which leaves intermediate containers behind; remove them even on
                # failure.
                rm=True,
                forcerm=True,
                decode=True,
            )
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_stream_build_logs, build_output, log_labels)
            try:
                future.result(timeout=timeout_seconds)
            except FuturesTimeoutError:
                build_output.abandoned.set()
                message = f"Docker image build exceeded {timeout_seconds} seconds"
                logger.error(message, extra=log_labels)
                return None, message
            if build_output.error is not None:
                logger.error(f"Docker build failed: {build_output.error}", extra=log_labels)
                return None, build_output.error

            # A failed build step reaches the low-level build stream as an {"error": ...} chunk, not an exception: the stream
            # ends normally with no image, and the job would only fail later on a missing tag at container start.
            try:
                client.images.get(tag)
            except ImageNotFound:
                message = f"Docker image build failed: no image '{tag}' was produced (a build step failed; see the build log)"
                logger.error(message, extra=log_labels)
                return None, message

            logger.info("Docker image built successfully.", extra=log_labels)
            return tag, None
        except (BuildError, APIError) as e:
            logger.error(f"Docker build failed: {str(e)}", extra=log_labels)
            return None, str(e)
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)


def _stream_build_logs(build_output: Iterator[dict], log_labels: dict[str, str] | None) -> None:
    # A build abandoned on timeout keeps running on the daemon until its stream ends, so it stays counted until then.
    with image_build_in_flight():
        stream_image_build_logs(build_output, logger, log_labels)


def flag_large_build_context_files(context_path: str, log_labels: dict[str, str] | None = None) -> str | None:
    """List build-context files over the review size for manual review; never fails the build.

    Weights-derived payloads over about 1 MB are banned and enforced by review. This checkout is what the image
    is built from, so unlike the registration scan it includes any git LFS objects the clone fetched.
    """
    large_files = find_large_files(Path(context_path), QUANT_LARGE_FILE_REVIEW_BYTES)
    if not large_files:
        return None
    message = (
        f"[REVIEW] Build context has {len(large_files)} file(s) over {QUANT_LARGE_FILE_REVIEW_BYTES} bytes, flagged for "
        f"manual review (weights-derived payloads are banned): {describe_large_files(large_files)}"
    )
    logger.warning(message, extra=log_labels)
    return message


def delete_image_and_cleanup(tag: str):
    client = docker.from_env()
    try:
        # With the layer cache on, the image's layers stay behind as the cache for later builds until
        # prune_build_leftovers empties it.
        client.images.remove(image=tag, force=True, noprune=not cst.IMAGE_BUILD_NO_CACHE)
        logger.info(f"Deleted Docker image with tag: {tag}")
    except docker.errors.ImageNotFound:
        logger.error(f"No Docker image found with tag: {tag}")
    except Exception as e:
        logger.error(f"Failed to delete image '{tag}': {e}")

    try:
        prune_build_leftovers()
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
    training_mode: str = ImageTrainingMode.TEXT_TO_IMAGE.value,
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

    if training_mode != ImageTrainingMode.TEXT_TO_IMAGE.value:
        command += ["--training-mode", training_mode]

    if trigger_word:
        command += ["--trigger-word", trigger_word]

    # No Hugging Face credentials: the job runs on the internal network from the read-only cache, and the uploader
    # container does the upload.
    environment: dict[str, str] = {
        "TRANSFORMERS_CACHE": cst.HUGGINGFACE_CACHE_PATH,
        "HF_HOME": cst.HUGGINGFACE_CACHE_PATH,
    }
    if model_type == ImageModelType.FLUX.value:
        environment.update(
            {
                "FLUX_T5_PATH": os.path.join(cst.HUGGINGFACE_CACHE_PATH, FLUX_T5_ARTIFACT[1]),
                "FLUX_CLIP_PATH": os.path.join(cst.HUGGINGFACE_CACHE_PATH, FLUX_CLIP_ARTIFACT[1]),
                "FLUX_VAE_PATH": os.path.join(cst.HUGGINGFACE_CACHE_PATH, FLUX_VAE_ARTIFACT[1]),
            }
        )
    if baseline_stats:
        vol = client.volumes.get(cst.CACHE_VOLUME_NAME)
        stats_filename = f"baseline_stats_{task_id}.json"
        with open(os.path.join(vol.attrs["Mountpoint"], stats_filename), "w") as f:
            json.dump(baseline_stats.model_dump(), f)
        environment["BASELINE_STATS_PATH"] = os.path.join(cst.CACHE_ROOT_PATH, stats_filename)

    # Only this job's directory of the checkpoints volume, and the shared cache read-only.
    volumes = {
        **job_checkpoints_volume(client, task_id, hotkey),
        cst.CACHE_VOLUME_NAME: {"bind": cst.CACHE_ROOT_PATH, "mode": "ro"},
    }

    container_name = f"image-trainer-{uuid.uuid4().hex}"

    resources = container_resources(gpu_ids)
    environment.update(resources.thread_env())

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
                volumes=volumes,
                remove=False,
                shm_size=shm_size,
                name=container_name,
                labels=log_labels,
                **resources.docker_kwargs(),
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


async def _write_quant_calibration(task_id: str, hotkey: str, calibration: QuantCalibrationRequest) -> dict[str, str]:
    """The env that points the miner at its calibration file in the cache volume; raises when the file cannot be written
    within the budget of all its attempts (the training container has no network, so this runs on the host)."""
    per_attempt = (
        cst.QUANT_CALIBRATION_FETCH_TIMEOUT_SECONDS
        + cst.QUANT_CALIBRATION_RETRY_SLEEP_SECONDS
        + cst.QUANT_CALIBRATION_ATTEMPT_SLACK_SECONDS
    )
    budget = cst.QUANT_CALIBRATION_FETCH_ATTEMPTS * per_attempt
    volume_dir = docker.from_env().volumes.get(cst.CACHE_VOLUME_NAME).attrs["Mountpoint"]
    return await asyncio.wait_for(write_quant_calibration_file(task_id, hotkey, calibration, volume_dir), timeout=budget)


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
    quant_spec: QuantFixedSpec | None = None,
    quant_target: QuantTarget | None = None,
    extra_environment: dict[str, str] | None = None,
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
    if extra_environment:
        environment.update(extra_environment)
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
    if quant_spec is not None:
        quant_spec_json = quant_spec.model_dump_json()
        command += ["--quant-spec", quant_spec_json]
        environment[core_cst.QUANT_SPEC_ENV] = quant_spec_json
    if quant_target is not None:
        # Env only, never a flag: a miner's strict argparse would reject an unknown --quant-target and fail the job.
        environment[core_cst.QUANT_TARGET_ENV] = quant_target.model_dump_json()

    # Only this job's directory of the checkpoints volume, and the shared cache read-only.
    volumes = {
        **job_checkpoints_volume(client, task_id, hotkey),
        cst.CACHE_VOLUME_NAME: {"bind": cst.CACHE_ROOT_PATH, "mode": "ro"},
    }

    container_name = f"text-trainer-{uuid.uuid4().hex}"

    resources = container_resources(gpu_ids)
    environment.update(resources.thread_env())

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
                volumes=volumes,
                remove=False,
                shm_size=shm_size,
                name=container_name,
                labels=log_labels,
                **resources.docker_kwargs(),
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
    training_mode: ImageTrainingMode | None = None,
    log_labels: dict[str, str] | None = None,
    anonymize: bool = True,
    gpu_ids: list[int] | None = None,
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
    if training_mode:
        command += ["--training-mode", training_mode]

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
            **pinned_helper_kwargs(gpu_ids),
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
                container.remove(force=True, v=True)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove container {container_name}: {cleanup_err}", extra=log_labels)


def run_model_prep_container(
    task_id: str,
    model_id: str,
    training_data_url: str,
    task_type: TaskType = TaskType.INSTRUCTTEXTTASK,
    augmentation_config=None,
    gpu_ids: list[int] = [0],
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
        gpu_ids=gpu_ids,
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

    env = _hf_container_env()

    container_name = f"model-prep-{str(uuid.uuid4())[:8]}"
    container = None
    resources = container_resources(gpu_ids)
    env.update(resources.thread_env())

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
            **resources.docker_kwargs(),
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
                container.remove(force=True, v=True)
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
    verbatim: bool = False,
    gpu_ids: list[int] | None = None,
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
        if verbatim:
            environment["UPLOAD_VERBATIM"] = "1"

        volumes = {
            **job_checkpoints_volume(client, task_id, hotkey),
            cst.CACHE_VOLUME_NAME: {"bind": cst.CACHE_ROOT_PATH, "mode": "ro"},
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
            **pinned_helper_kwargs(gpu_ids),
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
                await asyncio.to_thread(container.remove, force=True, v=True)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove upload container {container.name}: {cleanup_err}")


def get_task_type(request: TrainerProxyRequest) -> TaskType:
    training_data = request.training_data

    if isinstance(training_data, TrainRequestImage):
        return TaskType.IMAGETASK

    elif isinstance(training_data, TrainRequestQuant):
        return training_data.task_type

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


def _decode_base_model_dir(training_data: TrainRequestQuant, cache_mountpoint: Path) -> Path:
    """Host path of the base model the training container saw: the anonymised dir when model prep ran.

    task_history.json drops baseline_stats, so a job reloaded from it (the decode re-run) falls back to the anonymised
    dir when the plain one is absent. Both hold the same tensors; anonymisation only scrubs config identity.
    """
    plain = _cache_host_model_dir(training_data.model, cache_mountpoint)
    if training_data.baseline_stats is None and plain.exists():
        return plain
    return _cache_host_model_dir(get_anonymous_model_dir(training_data.model), cache_mountpoint)


def _cache_host_model_dir(model_dir_name: str, cache_mountpoint: Path) -> Path:
    container_path = train_paths.get_text_base_model_path(model_dir_name)
    return cache_mountpoint / Path(container_path).relative_to(cst.CACHE_ROOT_PATH)


async def freeform_decode_result(task: TrainerProxyRequest) -> QuantDecodeResult:
    """Decode one freeform job's uploaded artifact."""
    training_data = task.training_data
    try:
        client = docker.from_env()
        cache_mountpoint = Path(client.volumes.get(cst.CACHE_VOLUME_NAME).attrs["Mountpoint"])
        return await run_freeform_decode(
            training_data.decode,
            task_id=training_data.task_id,
            hotkey=task.hotkey,
            artifact_repo=f"{os.getenv('HUGGINGFACE_USERNAME')}/{training_data.expected_repo_name}",
            base_model_dir=_decode_base_model_dir(training_data, cache_mountpoint),
            hf_token=os.getenv("HUGGINGFACE_TOKEN") or None,
            work_root=cst.QUANT_DECODE_WORK_ROOT,
            docker_client=client,
            report=partial(log_task, training_data.task_id, task.hotkey),
            cpu_share=container_resources(task.gpu_ids),
        )
    except Exception as e:  # anything the job did not classify is ours, never the miner's
        logger.exception(f"Freeform decode crashed for task {training_data.task_id}")
        return QuantDecodeResult(infra_error=sanitize_failure(str(e)) or type(e).__name__)


async def _run_freeform_decode_step(task: TrainerProxyRequest, log_labels: dict[str, str]) -> None:
    """Decode a freeform artifact after its upload and store the outcome on the job log for the validator.

    When the decoded files could not be uploaded yet, the job still finishes here: the stored outcome is
    `upload_pending` and upload_retry replaces it once the upload lands or its retry window closes.
    """
    training_data = task.training_data
    if not isinstance(training_data, TrainRequestQuant) or training_data.decode is None:
        await log_task(training_data.task_id, task.hotkey, "freeform decode skipped: no decode request")
        return

    await log_task(training_data.task_id, task.hotkey, "Running freeform decode...")
    upload_retry.cancel((training_data.task_id, task.hotkey))  # this decode replaces an earlier one's pending upload
    result = await freeform_decode_result(task)
    if result.infra_error == JOB_BUSY_FAILURE:
        # Another process (the re-run CLI) is decoding this job and stores its own result: keep the stored one.
        message = f"[ERROR] freeform decode not run: {result.infra_error}"
        logger.error(message, extra=log_labels)
        await log_task(training_data.task_id, task.hotkey, message)
        return
    await update_quant_decode_result(training_data.task_id, task.hotkey, result)
    if result.upload_pending:
        upload_retry.start(cst.QUANT_DECODE_WORK_ROOT, training_data.task_id, task.hotkey)
    message = upload_retry.result_message(result)
    logger.info(message, extra=log_labels)
    await log_task(training_data.task_id, task.hotkey, message)


async def _quant_output_accepted(task: TrainerProxyRequest, training_data: TrainRequestQuant, log_labels: dict[str, str]) -> bool:
    """Check and prune the quant output dir against its quant_manifest.json before upload; log the reason on rejection.

    The miner's container wrote the output as root, so it is first chowned to the trainer (reclaim_output_dir), which
    then owns everything it must prune. A reclaim that cannot run raises: the caller fails the job without upload.
    """
    client = docker.from_env()
    job_dir = job_checkpoints_host_dir(client, training_data.task_id, task.hotkey)
    cache_mountpoint = Path(client.volumes.get(cst.CACHE_VOLUME_NAME).attrs["Mountpoint"])
    output_dir = job_dir / training_data.task_id / training_data.expected_repo_name
    rejection = await reclaim_output_dir(
        client, task_id=training_data.task_id, repo_name=training_data.expected_repo_name, checkpoints_source=str(job_dir)
    )
    if rejection is not None:
        verdict = QuantUploadVerdict(ok=False, failure=f"{cst.QUANT_OUTPUT_REJECTED_PREFIX}{rejection}")
    else:
        max_bytes = await asyncio.to_thread(quant_output_max_bytes, _decode_base_model_dir(training_data, cache_mountpoint))
        verdict = await asyncio.to_thread(
            check_quant_output,
            output_dir,
            task_id=training_data.task_id,
            task_type=training_data.task_type,
            max_bytes=max_bytes,
        )
    if not verdict.ok:
        logger.error(verdict.failure, extra=log_labels)
        await log_task(training_data.task_id, task.hotkey, f"[ERROR] {verdict.failure}")
        return False
    await log_task(
        training_data.task_id, task.hotkey, f"Quant output accepted: {verdict.n_files} files, {verdict.total_bytes} bytes"
    )
    return True


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
            training_mode=training_data.training_mode if task_type == TaskType.IMAGETASK else None,
            log_labels=log_labels,
            anonymize=model_prep_ran,
            gpu_ids=task.gpu_ids,
        )

        if download_status == 0:
            message = "Download container completed successfully"
            await log_task(training_data.task_id, task.hotkey, message)
        else:
            message = f"[ERROR] Download container failed | ExitCode: {download_status} | LastError: {exc}"
            await log_task(training_data.task_id, task.hotkey, message)
            await complete_task(training_data.task_id, task.hotkey, success=False)
            raise RuntimeError(f"Downloader container failed: {exc}")

        # The compression job's calibration file is written on the host before the image build, with its own time budget
        # (the container-start timeout must not cut its retries short): no file, no build, no container.
        extra_environment = (
            await _write_quant_calibration(training_data.task_id, task.hotkey, training_data.calibration)
            if isinstance(training_data, TrainRequestQuant)
            else None
        )

        if task_type in QUANT_TASK_TYPES:
            review_message = await asyncio.to_thread(flag_large_build_context_files, local_repo_path, log_labels)
            if review_message:
                await log_task(training_data.task_id, task.hotkey, review_message)

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

        # A new attempt starts from an empty directory of its own, whatever an earlier attempt left there.
        docker_client = docker.from_env()
        await clear_job_checkpoints(
            docker_client, job_checkpoints_host_dir(docker_client, training_data.task_id, task.hotkey)
        )

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
                    training_mode=training_data.training_mode,
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
            quant_spec = training_data.quant_spec if isinstance(training_data, TrainRequestQuant) else None
            quant_target = training_data.quant_target if isinstance(training_data, TrainRequestQuant) else None
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
                    quant_spec=quant_spec,
                    quant_target=quant_target,
                    extra_environment=extra_environment,
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

            if is_gpu_throttle_failure(training_data.task_id, task.hotkey):
                success = False

            if cancel_log_message:
                await log_task(training_data.task_id, task.hotkey, cancel_log_message)

            # Clean up all environment servers
            for srv in env_server_containers:
                try:
                    await asyncio.to_thread(srv.stop)
                    await asyncio.to_thread(srv.remove, force=True, v=True)
                except Exception as e:
                    logger.warning(f"Failed to cleanup server {srv.name}: {e}")

            if container and isinstance(container, Container):
                try:
                    await asyncio.to_thread(container.reload)
                    if container.status == "running":
                        await asyncio.to_thread(container.kill)
                    await asyncio.to_thread(container.remove, force=True, v=True)
                    await log_task(training_data.task_id, task.hotkey, f"Container {container.name} cleaned up.")

                except Exception as cleanup_err:
                    await log_task(training_data.task_id, task.hotkey, f"Error during container cleanup: {cleanup_err}")

            logger.info("Cleaning up", extra=log_labels)
            if tag:
                await asyncio.to_thread(delete_image_and_cleanup, tag)
                logger.info("Cleaned up Docker resources.", extra=log_labels)
            else:
                logger.info("No Docker image to clean up.", extra=log_labels)

            if success and task_type in QUANT_TASK_TYPES:
                try:
                    success = await _quant_output_accepted(task, training_data, log_labels)
                except Exception as check_err:  # never upload an unchecked quant output
                    logger.exception("Quant output check crashed", extra=log_labels)
                    await log_task(training_data.task_id, task.hotkey, f"[ERROR] Quant output check crashed: {check_err}")
                    success = False

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
                        verbatim=task_type in QUANT_TASK_TYPES,
                        gpu_ids=task.gpu_ids,
                    )

                    await log_task(training_data.task_id, task.hotkey, "Repo uploaded successfully.")
                except Exception as upload_err:
                    log_message = f"[ERROR] Upload container failed | ExitCode: Unknown | LastError: {upload_err}"
                    await log_task(training_data.task_id, task.hotkey, log_message)
                    success = False

            if success and task_type == TaskType.QUANTFREEFORMTASK:
                # Never changes success: a decoder fault is a scored fail, kept out of the majority-failure hold.
                await _run_freeform_decode_step(task, log_labels)

            await complete_task(training_data.task_id, task.hotkey, success=success)

        try:
            await asyncio.shield(_final_cleanup())
        finally:
            if cancelled_exc:
                raise cancelled_exc
