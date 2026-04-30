import uuid

import docker

from core.whitelisted_sft_datasets import validate_requested_datasets
from trainer import constants as cst
from trainer.utils.trainer_logging import logger


def download_whitelisted_datasets(
    requested_datasets: list[str] | None,
    hotkey: str,
    task_id: str,
) -> list[str]:
    """Validate and download requested datasets into the shared cache volume.

    Runs the trainer-downloader container with the cache volume mounted rw.
    Downloaded datasets persist at /cache/miner_datasets/{org--name}/ and are
    readable by training containers which mount the same volume at /cache (ro).

    Returns the list of dataset directory names successfully downloaded.
    Individual failures are logged and skipped.
    """
    if not requested_datasets:
        return []

    validated = validate_requested_datasets(requested_datasets)
    if not validated:
        logger.warning(
            f"Miner {hotkey} requested datasets {requested_datasets} but none matched whitelist (task {task_id})"
        )
        return []

    logger.info(f"Validated datasets for hotkey {hotkey}, task {task_id}: {validated}")

    downloaded = []
    for dataset_repo_id in validated:
        try:
            _download_dataset_via_container(dataset_repo_id, task_id)
            downloaded.append(dataset_repo_id.replace("/", "--"))
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_repo_id} for task {task_id}: {e}")

    return downloaded


def _download_dataset_via_container(dataset_repo_id: str, task_id: str) -> None:
    """Download a single HF dataset into the shared cache volume using the trainer-downloader container."""
    client = docker.from_env()
    container_name = f"miner-ds-{task_id[:8]}-{uuid.uuid4().hex[:8]}"

    container = None
    try:
        logger.info(f"Starting dataset download container for {dataset_repo_id}", extra={"task_id": task_id})
        container = client.containers.run(
            image=cst.TRAINER_DOWNLOADER_DOCKER_IMAGE,
            command=[
                "download-miner-dataset",
                "--repo-id", dataset_repo_id,
                "--cache-dir", cst.MINER_DATASETS_CACHE_DIR,
            ],
            name=container_name,
            volumes={cst.CACHE_VOLUME_NAME: {"bind": "/cache", "mode": "rw"}},
            remove=False,
            detach=True,
        )

        result = container.wait(timeout=300)
        exit_code = result.get("StatusCode", -1)

        if exit_code == 0:
            logger.info(f"Dataset {dataset_repo_id} downloaded successfully", extra={"task_id": task_id})
        else:
            logs = container.logs().decode("utf-8", errors="ignore")
            logger.error(
                f"Dataset download failed for {dataset_repo_id} | exit_code={exit_code} | logs={logs[-500:]}",
                extra={"task_id": task_id},
            )
            raise RuntimeError(f"Download failed with exit code {exit_code}")

    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove dataset download container {container_name}: {cleanup_err}")
