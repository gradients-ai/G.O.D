import json

from datasets import load_dataset
from fiber import Keypair

from core.models.payload_models import TaskType
from validator.core.constants import STANDARD_DPO_CHOSEN_COLUMN
from validator.core.constants import STANDARD_DPO_PROMPT_COLUMN
from validator.core.constants import STANDARD_DPO_REJECTED_COLUMN
from validator.core.constants import STANDARD_GRPO_PROMPT_COLUMN
from validator.core.constants import STANDARD_INPUT_COLUMN
from validator.core.constants import STANDARD_INSTRUCT_COLUMN
from validator.core.constants import STANDARD_OUTPUT_COLUMN
from validator.core.constants import STANDARD_SYSTEM_COLUMN
from validator.core.models import AnyTextTypeRawTask
from validator.evaluation.utils import get_default_dataset_config
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def get_dataset_column_mapping(dataset_id: str, task_type: TaskType, keypair: Keypair) -> dict[str, str]:
    """Get column mapping for a specific dataset based on task type."""
    from validator.core.constants import CONTENT_BASE_URL
    from validator.utils.call_endpoint import call_content_service_fast

    url = f"{CONTENT_BASE_URL}/dataset/{dataset_id}/columns/suggest"
    response = await call_content_service_fast(url, keypair)

    logger.info(f"Raw response from content service for {dataset_id}: {response}")

    if not isinstance(response, dict):
        raise ValueError(f"Invalid response from content service for dataset {dataset_id}")

    if task_type == TaskType.DPOTASK:
        mapping = {
            "prompt": response.get("field_prompt", "prompt"),
            "chosen": response.get("field_chosen", "chosen"),
            "rejected": response.get("field_rejected", "rejected"),
        }
        logger.info(f"DPO column mapping for {dataset_id}: response={response}, mapping={mapping}")
        return mapping
    elif task_type == TaskType.INSTRUCTTEXTTASK:
        column_mapping = {}
        if "field_instruction" in response:
            column_mapping["instruction"] = response["field_instruction"]
        if "field_output" in response:
            column_mapping["output"] = response["field_output"]
        if response.get("field_input"):
            column_mapping["input"] = response["field_input"]
        if response.get("field_system"):
            column_mapping["system"] = response["field_system"]
        return column_mapping
    elif task_type == TaskType.CHATTASK:
        column_mapping = {}
        if "field_instruction" in response:
            column_mapping["instruction"] = response["field_instruction"]
        if "field_output" in response:
            column_mapping["output"] = response["field_output"]
        if response.get("field_input"):
            column_mapping["input"] = response["field_input"]
        if response.get("field_system"):
            column_mapping["system"] = response["field_system"]
        return column_mapping
    elif task_type == TaskType.GRPOTASK:
        return {"prompt": response.get("field_prompt", "prompt")}
    elif task_type == TaskType.ENVIRONMENTTASK:
        return {"prompt": "prompt"}
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def standardize_instruct_sample(sample: dict, task: AnyTextTypeRawTask) -> dict:
    """Standardize a single instruct/grpo sample to use standard column names."""
    std_sample = {}
    std_sample[STANDARD_INSTRUCT_COLUMN] = sample.get(task.field_instruction, "")
    std_sample[STANDARD_OUTPUT_COLUMN] = sample.get(task.field_output, "")
    if hasattr(task, "field_input") and task.field_input:
        std_sample[STANDARD_INPUT_COLUMN] = sample.get(task.field_input, "")
    if hasattr(task, "field_system") and task.field_system:
        std_sample[STANDARD_SYSTEM_COLUMN] = sample.get(task.field_system, "")
    return std_sample


def standardize_dpo_sample(sample: dict, task: AnyTextTypeRawTask) -> dict:
    """Standardize a single DPO sample."""
    std_sample = {
        STANDARD_DPO_PROMPT_COLUMN: sample.get(task.field_prompt, ""),
        STANDARD_DPO_CHOSEN_COLUMN: sample.get(task.field_chosen, ""),
        STANDARD_DPO_REJECTED_COLUMN: sample.get(task.field_rejected, ""),
    }
    if hasattr(task, "field_system") and task.field_system:
        std_sample[STANDARD_SYSTEM_COLUMN] = sample.get(task.field_system, "")
    return std_sample


def standardize_grpo_sample(sample: dict, task: AnyTextTypeRawTask) -> dict:
    """Standardize a single GRPO sample to use standard column names."""
    std_sample = {}
    std_sample[STANDARD_GRPO_PROMPT_COLUMN] = sample.get(task.field_prompt, "")
    return std_sample


def standardize_samples(samples: list[dict], task: AnyTextTypeRawTask) -> list[dict]:
    """Standardize a list of samples based on task type."""
    from validator.core.models import DpoRawTask
    from validator.core.models import GrpoRawTask
    from validator.core.models import InstructTextRawTask

    logger.info(f"Standardizing {len(samples)} samples with task type: {type(task).__name__}")
    if hasattr(task, "__dict__"):
        logger.info(f"Task fields: {vars(task)}")

    standardized = []
    for i, sample in enumerate(samples):
        try:
            processed_sample = {}
            for key, value in sample.items():
                if isinstance(value, dict):
                    processed_sample[key] = json.dumps(value)
                else:
                    processed_sample[key] = str(value) if value is not None else ""

            if isinstance(task, InstructTextRawTask):
                result = standardize_instruct_sample(processed_sample, task)
            elif isinstance(task, DpoRawTask):
                result = standardize_dpo_sample(processed_sample, task)
            elif isinstance(task, GrpoRawTask):
                result = standardize_grpo_sample(processed_sample, task)
            else:
                # Handle temp task objects that don't inherit from the models
                if hasattr(task, "task_type"):
                    from validator.core.models import TaskType

                    if task.task_type == TaskType.GRPOTASK:
                        result = standardize_grpo_sample(processed_sample, task)
                    elif task.task_type == TaskType.DPOTASK:
                        result = standardize_dpo_sample(processed_sample, task)
                    else:
                        result = standardize_instruct_sample(processed_sample, task)
                else:
                    # Fallback for old behavior
                    result = standardize_instruct_sample(processed_sample, task)

            if i == 0:  # Log first sample
                logger.info(f"Sample {i}: {sample} -> {result}")

            standardized.append(result)
        except Exception as e:
            logger.error(f"Failed to standardize sample {i}: {sample}, error: {e}")

    logger.info(f"Standardization complete: {len(samples)} -> {len(standardized)} samples")
    return standardized


def get_task_columns(task: AnyTextTypeRawTask) -> list[str]:
    """Extract column names from task based on task type."""
    from validator.core.models import DpoRawTask
    from validator.core.models import GrpoRawTask
    from validator.core.models import InstructTextRawTask

    if isinstance(task, InstructTextRawTask):
        columns = [task.field_instruction, task.field_output]
        if task.field_input:
            columns.append(task.field_input)
        if task.field_system:
            columns.append(task.field_system)
    elif isinstance(task, DpoRawTask):
        columns = [task.field_prompt, task.field_chosen, task.field_rejected]
        if hasattr(task, "field_system") and task.field_system:
            columns.append(task.field_system)
    elif isinstance(task, GrpoRawTask):
        columns = [task.field_prompt]
    else:
        columns = []
    return columns


def create_temp_task_from_mapping(column_mapping: dict[str, str], task_type):
    """Create a temporary task object with column mappings.

    column_mapping maps standard keys to actual dataset column names.
    We need to create a task that maps those actual column names to standard names.
    """
    from validator.core.models import TaskType

    if task_type == TaskType.INSTRUCTTEXTTASK:
        temp_task_dict = {"task_type": task_type}
        if "instruction" in column_mapping:
            temp_task_dict["field_instruction"] = column_mapping["instruction"]
        if "output" in column_mapping:
            temp_task_dict["field_output"] = column_mapping["output"]
        if "input" in column_mapping:
            temp_task_dict["field_input"] = column_mapping["input"]
        if "system" in column_mapping:
            temp_task_dict["field_system"] = column_mapping["system"]
        return type("obj", (object,), temp_task_dict)
    elif task_type == TaskType.DPOTASK:
        temp_task_dict = {"task_type": task_type}
        if "prompt" in column_mapping:
            temp_task_dict["field_prompt"] = column_mapping["prompt"]
        if "chosen" in column_mapping:
            temp_task_dict["field_chosen"] = column_mapping["chosen"]
        if "rejected" in column_mapping:
            temp_task_dict["field_rejected"] = column_mapping["rejected"]
        if "system" in column_mapping:
            temp_task_dict["field_system"] = column_mapping["system"]
        return type("obj", (object,), temp_task_dict)
    elif task_type == TaskType.GRPOTASK:
        temp_task_dict = {"task_type": task_type}
        if "prompt" in column_mapping:
            temp_task_dict["field_prompt"] = column_mapping["prompt"]
        return type("obj", (object,), temp_task_dict)


async def load_and_merge_multiple_datasets(dataset_ids: list[str], task: AnyTextTypeRawTask, keypair: Keypair) -> list[dict]:
    """Load and merge multiple datasets, returning average size."""
    import random

    logger.info(f"Loading and merging {len(dataset_ids)} datasets")

    all_samples = []
    dataset_sizes = []

    primary_id = dataset_ids[0]
    primary_columns = get_task_columns(task)

    try:
        config_name = get_default_dataset_config(primary_id)
        dataset = load_dataset(primary_id, config_name, trust_remote_code=True)

        if isinstance(dataset, dict):
            if "train" in dataset:
                dataset = dataset["train"]
            elif len(dataset) > 0:
                first_split = list(dataset.keys())[0]
                logger.info(f"Using split '{first_split}' from primary dataset {primary_id}")
                dataset = dataset[first_split]
            else:
                raise ValueError(f"No valid splits found in dataset {primary_id}")

        dataset = dataset.select_columns(primary_columns)
        samples = list(dataset)

        standardized = standardize_samples(samples, task)
        all_samples.extend(standardized)
        dataset_sizes.append(len(standardized))
        logger.info(f"Loaded {len(samples)} samples from primary dataset {primary_id}")
    except Exception as e:
        logger.error(f"Failed to load primary dataset {primary_id}: {e}")
        raise e

    if len(dataset_ids) > 1:
        logger.info(f"Loading {len(dataset_ids) - 1} additional datasets")

        for dataset_id in dataset_ids[1:]:
            try:
                logger.info(f"Loading additional dataset: {dataset_id}")

                # For DPO tasks, use the detectcolumns endpoint to get proper column mapping
                if task.task_type == TaskType.DPOTASK:
                    try:
                        from validator.core.constants import CONTENT_BASE_URL
                        from validator.utils.call_endpoint import call_content_service_fast

                        url = f"{CONTENT_BASE_URL}/dataset/{dataset_id}/detectcolumns"
                        response = await call_content_service_fast(url, keypair)

                        if response.get("is_dpo") and response.get("columns"):
                            columns = response["columns"]
                            column_mapping = {
                                "prompt": columns.get("prompt", "prompt"),
                                "chosen": columns.get("accepted", "chosen"),
                                "rejected": columns.get("rejected", "rejected"),
                            }
                            logger.info(f"DPO column mapping from detectcolumns for {dataset_id}: {column_mapping}")
                        else:
                            raise ValueError(f"Dataset {dataset_id} is not DPO compatible")
                    except Exception as e:
                        logger.error(f"Failed to get DPO column mapping for {dataset_id}: {e}")
                        raise
                else:
                    try:
                        column_mapping = await get_dataset_column_mapping(dataset_id, task.task_type, keypair)
                        logger.info(f"Column mapping for {dataset_id}: {column_mapping}")
                    except Exception as e:
                        logger.error(f"Failed to get column mapping for {dataset_id}: {e}")
                        raise

                columns = list(column_mapping.values())
                config_name = get_default_dataset_config(dataset_id)
                logger.info(f"Loading {dataset_id} with config {config_name} and columns {columns}")
                logger.info(f"Column mapping: {column_mapping}")

                dataset = load_dataset(dataset_id, config_name, trust_remote_code=True)

                # Handle DatasetDict vs Dataset
                if hasattr(dataset, "column_names") and not isinstance(dataset, dict):
                    # It's already a Dataset
                    pass
                else:
                    # It's a DatasetDict - try to get the best split
                    if "train" in dataset:
                        dataset = dataset["train"]
                    elif "train_prefs" in dataset:
                        dataset = dataset["train_prefs"]
                    elif list(dataset.keys()):
                        # Take the first available split
                        first_split = list(dataset.keys())[0]
                        logger.info(f"Using split '{first_split}' from DatasetDict")
                        dataset = dataset[first_split]
                    else:
                        raise ValueError(f"DatasetDict is empty for {dataset_id}")

                logger.info(f"Dataset {dataset_id} loaded with {len(dataset)} total samples")
                logger.info(f"Available columns in dataset: {dataset.column_names}")

                # Check if all required columns exist
                missing_columns = [col for col in columns if col not in dataset.column_names]
                if missing_columns:
                    logger.error(f"Missing columns in {dataset_id}: {missing_columns}")
                    logger.error(f"Available columns: {dataset.column_names}")
                    logger.error(f"Required columns from mapping: {columns}")
                    raise ValueError(f"Missing required columns: {missing_columns}")

                dataset = dataset.select_columns(columns)
                samples = list(dataset)
                logger.info(f"After column selection, {dataset_id} has {len(samples)} samples")

                if samples:
                    logger.info(f"Sample data from {dataset_id}: {samples[0]}")

                temp_task = create_temp_task_from_mapping(column_mapping, task.task_type)
                logger.info(f"Created temp task for {dataset_id}: {vars(temp_task)}")

                standardized = standardize_samples(samples, temp_task)
                logger.info(f"After standardization, {dataset_id} has {len(standardized)} samples")

                if standardized:
                    logger.info(f"Standardized sample from {dataset_id}: {standardized[0]}")
                all_samples.extend(standardized)
                dataset_sizes.append(len(standardized))
                logger.info(f"Successfully loaded {len(samples)} samples from {dataset_id}")
                if standardized:
                    logger.info(f"Example from {dataset_id}: {standardized[0]}")
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_id}: {e}")
                logger.info("Continuing with remaining datasets...")
                continue

    # Ensure we have at least one dataset loaded
    if not all_samples:
        raise ValueError("Failed to load any datasets successfully")

    if len(dataset_sizes) == 0:
        raise ValueError("No valid dataset sizes found")

    avg_size = sum(dataset_sizes) // len(dataset_sizes)
    min_size = min(dataset_sizes)
    max_size = max(dataset_sizes)
    total_available = sum(dataset_sizes)
    logger.info(f"Dataset sizes: {dataset_sizes}")
    logger.info(f"Min size: {min_size}, Average size: {avg_size}, Max size: {max_size}, Total available: {total_available}")

    # Calculate max samples per dataset (50% of total)
    max_samples_per_dataset = total_available // 2

    # Calculate how many samples we can take from each dataset
    available_per_dataset = []
    for size in dataset_sizes:
        available = min(size, max_samples_per_dataset)
        available_per_dataset.append(available)

    total_available_balanced = sum(available_per_dataset)
    logger.info(f"After applying 50% cap, available samples per dataset: {available_per_dataset}")
    logger.info(f"Total available for balanced sampling: {total_available_balanced}")

    # Now calculate equal samples from what's available
    samples_per_dataset = total_available_balanced // len(dataset_sizes)
    remainder = total_available_balanced % len(dataset_sizes)

    logger.info(f"Taking {samples_per_dataset} samples from each dataset (with {remainder} extra distributed)")

    final_samples = []
    start_idx = 0

    for i, size in enumerate(dataset_sizes):
        dataset_name = dataset_ids[i] if i < len(dataset_ids) else f"Dataset_{i}"
        end_idx = start_idx + size
        dataset_samples = all_samples[start_idx:end_idx]

        # Shuffle samples from this dataset
        random.shuffle(dataset_samples)

        # Take appropriate number of samples (respecting both equal distribution and max cap)
        num_to_take = samples_per_dataset + (1 if i < remainder else 0)
        num_to_take = min(num_to_take, len(dataset_samples), max_samples_per_dataset)

        logger.info(f"Dataset {i} ({dataset_name}): has {len(dataset_samples)} samples, taking {num_to_take}")

        # Take appropriate number of samples
        num_to_take = samples_per_dataset + (1 if i < remainder else 0)
        num_to_take = min(num_to_take, len(dataset_samples))

        logger.info(f"Dataset {i}: has {len(dataset_samples)} samples, taking {num_to_take}")

        final_samples.extend(dataset_samples[:num_to_take])
        start_idx = end_idx

    # Final shuffle to mix samples from all datasets
    random.shuffle(final_samples)

    logger.info(f"Merged {len(dataset_sizes)} datasets, returning {len(final_samples)} samples")
    return final_samples
