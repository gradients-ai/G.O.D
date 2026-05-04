import os
import uuid

import docker

from trainer import constants as cst
from trainer.utils.trainer_logging import logger


def download_whitelisted_datasets(
    requested_datasets: list[str],
    hotkey: str,
    task_id: str,
) -> list[str]:
    """Download pre-validated datasets into the shared cache volume.

    Returns directory names (org--name) for datasets that were successfully downloaded.
    """
    logger.info(f"Downloading datasets for hotkey {hotkey}, task {task_id}: {requested_datasets}")

    downloaded = []
    for dataset_repo_id in requested_datasets:
        dir_name = dataset_repo_id.replace("/", "--")
        try:
            _download_dataset_via_container(dataset_repo_id, task_id)
            downloaded.append(dir_name)
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_repo_id} for task {task_id}: {e}")

    return downloaded


def _download_dataset_via_container(dataset_repo_id: str, task_id: str) -> None:
    """Download a single HF dataset into the shared cache volume."""
    client = docker.from_env()
    container_name = f"miner-ds-{task_id[:8]}-{uuid.uuid4().hex[:8]}"

    container = None
    try:
        logger.info(f"Starting dataset download container for {dataset_repo_id}", extra={"task_id": task_id})
        environment = {}
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            environment["HF_TOKEN"] = hf_token

        container = client.containers.run(
            image=cst.TRAINER_DOWNLOADER_DOCKER_IMAGE,
            command=[
                "download-miner-dataset",
                "--repo-id", dataset_repo_id,
                "--cache-dir", cst.MINER_DATASETS_CACHE_DIR,
            ],
            name=container_name,
            volumes={cst.CACHE_VOLUME_NAME: {"bind": "/cache", "mode": "rw"}},
            environment=environment,
            remove=False,
            detach=True,
        )

        result = container.wait(timeout=1800)
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
