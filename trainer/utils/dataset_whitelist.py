import os
import shutil

from huggingface_hub import snapshot_download

from core.whitelisted_sft_datasets import validate_requested_datasets
from trainer import constants as cst
from trainer.utils.trainer_logging import logger


def download_and_place_whitelisted_datasets(
    requested_datasets: list[str] | None,
    local_repo_path: str,
    hotkey: str,
    task_id: str,
) -> None:
    """Validate, download (with caching), and place requested datasets into the repo.

    Filters to whitelisted datasets only, downloads to a persistent cache,
    then copies into {local_repo_path}/Datasets/{dataset_name}/.
    Individual failures are logged and skipped — training continues regardless.
    """
    if not requested_datasets:
        return

    validated = validate_requested_datasets(requested_datasets)
    if not validated:
        logger.warning(
            f"Miner {hotkey} requested datasets {requested_datasets} but none matched whitelist (task {task_id})"
        )
        return

    logger.info(f"Validated datasets for hotkey {hotkey}, task {task_id}: {validated}")

    datasets_dir = os.path.join(local_repo_path, cst.REPO_DATASETS_SUBDIR)
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(cst.MINER_DATASETS_CACHE_DIR, exist_ok=True)

    for dataset_repo_id in validated:
        cache_name = dataset_repo_id.replace("/", "--")
        cache_path = os.path.join(cst.MINER_DATASETS_CACHE_DIR, cache_name)
        dest_path = os.path.join(datasets_dir, cache_name)

        try:
            if not os.path.exists(cache_path):
                logger.info(f"Downloading dataset {dataset_repo_id} to cache {cache_path}")
                snapshot_download(
                    repo_id=dataset_repo_id,
                    repo_type="dataset",
                    local_dir=cache_path,
                    local_dir_use_symlinks=False,
                )
            else:
                logger.info(f"Dataset {dataset_repo_id} already cached at {cache_path}")

            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(cache_path, dest_path)
            logger.info(f"Placed dataset {dataset_repo_id} into {dest_path}")

        except Exception as e:
            logger.error(f"Failed to download/place dataset {dataset_repo_id} for task {task_id}: {e}")
