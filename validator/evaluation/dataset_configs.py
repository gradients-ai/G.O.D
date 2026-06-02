from datasets import get_dataset_config_names

from core.logging import get_logger


logger = get_logger(__name__)


def get_default_dataset_config(dataset_name: str) -> str | None:
    try:
        logger.info(dataset_name)
        config_names = get_dataset_config_names(dataset_name)
    except Exception:
        return None
    if config_names:
        logger.info(f"Taking the first config name: {config_names[0]} for dataset: {dataset_name}")
        # logger.info(f"Dataset {dataset_name} has configs: {config_names}. Taking the first config name: {config_names[0]}")
        return config_names[0]
    else:
        return None
