import glob
import os
import shutil
import tempfile

import core.constants.paths as cst
from core.logging import get_logger


logger = get_logger(__name__)


def cleanup_temp_files():
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.endswith(".json") and (
            any(prefix in filename for prefix in ["train_data_", "test_data_", "synth_data_", "latest_scores_"])
            or any(suffix in filename for suffix in ["_test_data.json", "_train_data.json", "_synth_data.json", "_scores.json"])
        ):
            try:
                os.remove(os.path.join(temp_dir, filename))
                logger.info(f"Removed temporary file: {filename}")
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {filename}: {e}")


def delete_dataset_from_cache(dataset_name):
    """
    Delete dataset and associated lock files from HuggingFace cache.
    Case-insensitive matching for dataset names.
    """
    dataset_name = dataset_name.lower().replace("/", "___")
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")

    all_datasets = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
    matching_datasets = [d for d in all_datasets if d.lower() == dataset_name]

    deleted = False

    for dataset_dir in matching_datasets:
        dataset_path = os.path.join(cache_dir, dataset_dir)
        try:
            shutil.rmtree(dataset_path)
            logger.info(f"Deleted dataset directory: {dataset_dir}")
            deleted = True
        except Exception as e:
            logger.error(f"Error deleting dataset directory '{dataset_dir}': {str(e)}")

    try:
        all_lock_files = glob.glob(os.path.join(cache_dir, "*.lock"))
        matching_locks = [
            f for f in all_lock_files if f"cache_huggingface_datasets_{dataset_name}" in os.path.basename(f).lower()
        ]

        for lock_file in matching_locks:
            try:
                os.remove(lock_file)
                logger.info(f"Deleted lock file: {os.path.basename(lock_file)}")
                deleted = True
            except Exception as e:
                logger.error(f"Error deleting lock file {os.path.basename(lock_file)}: {str(e)}")
    except Exception as e:
        logger.error(f"Error searching for lock files: {str(e)}")

    if not deleted:
        logger.info(f"No files found for dataset '{dataset_name}'")


def clean_all_hf_datasets_cache():
    """Clean the entire Huggingface datasets cache directory."""
    try:
        hf_cache_path = os.path.expanduser("~/.cache/huggingface/datasets/")
        if os.path.exists(hf_cache_path):
            shutil.rmtree(hf_cache_path)
            logger.info(f"Cleaned Huggingface datasets cache at {hf_cache_path}")
    except Exception as e:
        logger.error(f"Error cleaning Huggingface datasets cache: {e}")


def remove_cache_models_except(models_to_keep: list[str]):
    """Keep only specified models in HuggingFace cache, remove all others.

    Args:
        models_to_keep: List of model identifiers in format 'org/model-name' to preserve
    """
    keep_patterns = {f"models--{name.lower().replace('/', '--')}" for name in models_to_keep}

    # Handle directories
    existing_models = {
        orig_dir: orig_dir.lower()
        for orig_dir in os.listdir(cst.CACHE_DIR_HUB)
        if orig_dir.startswith("models--") and os.path.isdir(os.path.join(cst.CACHE_DIR_HUB, orig_dir))
    }

    to_delete = {orig_dir for orig_dir, lower_dir in existing_models.items() if lower_dir not in keep_patterns}

    deleted_count = 0
    for dir_name in to_delete:
        try:
            model_path = os.path.join(cst.CACHE_DIR_HUB, dir_name)
            shutil.rmtree(model_path)
            deleted_count += 1
        except Exception as e:
            logger.error(f"Error deleting cache for directory {dir_name}: {e}")

    # Handle safetensor files
    safetensor_files = [f for f in os.listdir(cst.CACHE_DIR_HUB) if f.endswith(".safetensors")]

    deleted_files = 0
    for file_name in safetensor_files:
        # Check if file belongs to a model we want to keep
        if not any(file_name.lower().startswith(pattern) for pattern in keep_patterns):
            try:
                file_path = os.path.join(cst.CACHE_DIR_HUB, file_name)
                os.remove(file_path)
                deleted_files += 1
            except Exception as e:
                logger.error(f"Error deleting safetensor file {file_name}: {e}")

    logger.info(f"Cleaned cache: removed {deleted_count} dirs and {deleted_files} safetensors files.")


def get_directory_size(path: str) -> int:
    """Calculate total size of a directory in bytes.
    Returns 0 if directory doesn't exist."""
    try:
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += get_directory_size(entry.path)
        return total
    except (FileNotFoundError, PermissionError):
        return 0


def get_hf_model_cache_size(model_id: str) -> int:
    """Get size of a specific model in HF cache (case insensitive).
    Returns 0 if model not found."""
    model_pattern = f"models--{model_id.lower().replace('/', '--')}"
    total_size = 0

    # Find and measure directory size (case insensitive)
    if os.path.exists(cst.CACHE_DIR_HUB):
        for dir_name in os.listdir(cst.CACHE_DIR_HUB):
            if dir_name.lower() == model_pattern:
                total_size += get_directory_size(os.path.join(cst.CACHE_DIR_HUB, dir_name))

        # Check for matching safetensor files
        for file_name in os.listdir(cst.CACHE_DIR_HUB):
            if file_name.endswith(".safetensors") and file_name.lower().startswith(model_pattern):
                try:
                    file_path = os.path.join(cst.CACHE_DIR_HUB, file_name)
                    total_size += os.path.getsize(file_path)
                except (OSError, FileNotFoundError):
                    pass

    return total_size


def manage_models_cache(model_stats: dict[str, dict], max_size: int) -> None:
    """Manage HF models cache based on model usage statistics."""
    current_size = get_directory_size(cst.CACHE_DIR_HUB)
    if current_size <= max_size:
        return

    logger.info(f"Cache size ({current_size / 1024**3:.2f}GB) exceeds limit ({max_size / 1024**3:.2f}GB). Starting cleanup...")
    logger.info(f"Model stats: {model_stats}")

    # First pass: keep only models in stats
    allowed_models = list(model_stats.keys())
    logger.info(f"Cache cleanup: Will remove models with no cache score record ({len(allowed_models)} records)")
    remove_cache_models_except(allowed_models)

    current_size = get_directory_size(cst.CACHE_DIR_HUB)
    if current_size <= max_size:
        logger.info(f"Cache size now {current_size / 1024**3:.2f}GB after removing unused models.")
        return

    # Second pass: calculate which models to remove based on scores and sizes
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["cache_score"])

    size_to_free = current_size - max_size
    cumulative_size = 0
    models_to_remove = []

    for model_id, stats in sorted_models:
        model_size = get_hf_model_cache_size(model_id)
        cumulative_size += model_size
        models_to_remove.append(model_id)

        if cumulative_size >= size_to_free:
            break

    models_to_keep = [model_id for model_id in model_stats.keys() if model_id not in models_to_remove]

    logger.info(f"Cache cleanup: Will keep {len(models_to_keep)} models")
    remove_cache_models_except(models_to_keep)
    final_size = get_directory_size(cst.CACHE_DIR_HUB)
    logger.info(f"Cache cleanup complete. Final size: {final_size / 1024**3:.2f}GB")
