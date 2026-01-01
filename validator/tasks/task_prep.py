import ast
import json
import os
import random
import shutil
import tempfile
import uuid
import zipfile
from math import ceil
from pathlib import Path

from datasets import Dataset
from datasets import DatasetDict
from datasets import concatenate_datasets
from datasets import load_dataset
from fiber import Keypair

import validator.core.constants as cst
from core.models.payload_models import ImageTextPair
from core.models.utility_models import FileFormat
from core.utils import download_s3_file
from validator.core.models import AnyTextTypeRawTask
from validator.core.models import ChatRawTask
from validator.core.models import DpoRawTask
from validator.core.models import EnvRawTask
from validator.core.models import GrpoRawTask
from validator.core.models import InstructTextRawTask
from validator.db.sql.tasks import update_task
from validator.evaluation.utils import get_default_dataset_config
from validator.utils.cache_clear import delete_dataset_from_cache
from validator.utils.logging import get_logger
from validator.utils.reward_functions import validate_reward_function
from validator.utils.util import save_json_to_temp_file
from validator.utils.util import upload_file_to_minio


logger = get_logger(__name__)


def create_zip_for_image_dataset(split_keys: set, zip_name: str, entries: dict, dataset_root: Path) -> Path:
    subfolder_name = Path(zip_name).stem
    zip_path = dataset_root / zip_name

    if zip_path.exists():
        logger.error(f"Zip path {zip_path} exists. This should not happen. Deleting it.")
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for key in split_keys:
            img_file, txt_file = entries[key]
            with open(txt_file, "r") as f:
                logger.info(f"Adding the following prompt to the zip: {f.read()}")
            zipf.write(img_file, Path(subfolder_name) / img_file.relative_to(dataset_root))
            zipf.write(txt_file, Path(subfolder_name) / txt_file.relative_to(dataset_root))

    return zip_path


def unzip_to_temp_path(zip_file_path: str) -> str:
    random_tmp_id = uuid.uuid4()
    tmp_dir = f"{cst.TEMP_PATH_FOR_IMAGES}/{random_tmp_id}"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    return tmp_dir


async def load_dataset_from_s3(dataset_url: str, max_file_size_bytes: int = None) -> Dataset | DatasetDict:
    """Load a dataset from S3 storage."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_file_path = await download_s3_file(dataset_url, save_path=temp_dir)
            if max_file_size_bytes:
                file_size = os.path.getsize(local_file_path)
                if file_size > max_file_size_bytes:
                    raise ValueError(f"File size {file_size} exceeds max file size {max_file_size_bytes}")

            dataset = load_dataset("json", data_files=local_file_path, split="train", trust_remote_code=False)

            return dataset
    except Exception as e:
        logger.exception(f"Failed to load dataset from S3: {e}")
        raise e


async def train_test_split(dataset: Dataset, test_size: float = cst.TRAIN_TEST_SPLIT_PERCENTAGE) -> DatasetDict:
    logger.info(f"Splitting dataset into train and test with test size {test_size}")

    test_size = min(
        int(len(dataset) * test_size),
        cst.MAX_TEST_DATA_POINTS,
    )
    split_dataset = dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)
    logger.info(f"Train set size: {len(split_dataset['train'])}")
    logger.info(f"Test set size: {len(split_dataset['test'])}")

    return split_dataset


def train_test_split_image(dataset_path: str) -> tuple[str, str]:
    """
    Dataset path is a folder containing the images and text files.
    """
    dataset_path = Path(dataset_path)

    has_images = any(dataset_path.glob(f"*.{ext.lstrip('.')}") for ext in cst.SUPPORTED_IMAGE_FILE_EXTENSIONS)

    if not has_images:
        sub_folder = [
            folder
            for folder in dataset_path.iterdir()
            if folder.is_dir() and any(folder.glob(f"*.{ext.lstrip('.')}") for ext in cst.SUPPORTED_IMAGE_FILE_EXTENSIONS)
        ]
        if not sub_folder:
            raise ValueError(f"No folder containing images found in: {dataset_path}")
        dataset_path = sub_folder[0]

    dataset_entries = {}
    for file in dataset_path.iterdir():
        if file.suffix in cst.SUPPORTED_IMAGE_FILE_EXTENSIONS:
            txt_file = file.with_suffix(".txt")
            if txt_file.exists():
                dataset_entries[file.stem] = (file, txt_file)

    keys = list(dataset_entries.keys())
    random.shuffle(keys)
    split_idx = ceil(len(keys) * cst.TRAIN_TEST_SPLIT_PERCENTAGE)
    test_keys = set(keys[:split_idx])
    train_keys = set(keys[split_idx:])

    test_zip_path = create_zip_for_image_dataset(
        split_keys=test_keys, zip_name=cst.IMAGE_TEST_SPLIT_ZIP_NAME, entries=dataset_entries, dataset_root=dataset_path
    )
    train_zip_path = create_zip_for_image_dataset(
        split_keys=train_keys, zip_name=cst.IMAGE_TRAIN_SPLIT_ZIP_NAME, entries=dataset_entries, dataset_root=dataset_path
    )

    return test_zip_path, train_zip_path


async def download_and_load_dataset(
    dataset_name: str, file_format: FileFormat, max_file_size_bytes: int = cst.MAX_FILE_SIZE_BYTES
) -> Dataset:
    if file_format == FileFormat.S3:
        dataset = await load_dataset_from_s3(dataset_name, max_file_size_bytes)
    else:
        config_name = get_default_dataset_config(dataset_name)
        dataset = load_dataset(dataset_name, config_name, trust_remote_code=True)

    if isinstance(dataset, DatasetDict):
        combined_dataset = concatenate_datasets([split for split in dataset.values()])
    else:
        combined_dataset = dataset

    return combined_dataset


def process_chat_row(value, role_field: str, content_field: str):
    if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass

    if isinstance(value, list):
        cleaned_messages = []
        for msg in value:
            if isinstance(msg, dict) and msg.get(content_field) is not None and msg.get(role_field) is not None:
                cleaned_messages.append(msg)
        return cleaned_messages if len(cleaned_messages) > 1 else []
    else:
        return value if value is not None else ""


def change_to_json_format(dataset: Dataset, columns: list[str], task: AnyTextTypeRawTask = None):
    result = []
    total_rows = 0
    fully_empty_rows = 0
    is_chat_task = isinstance(task, ChatRawTask)

    for idx, row in enumerate(dataset):
        row_dict = {}
        is_row_empty = True
        for col in columns:
            if col in row:
                value = row[col]

                # Ensure consistent data types: strings for non-ChatTask, preserve type for ChatTask
                if is_chat_task:
                    processed_value = process_chat_row(value, task.chat_role_field, task.chat_content_field)
                else:
                    processed_value = str(value) if value is not None else ""

                row_dict[col] = processed_value
                if processed_value != "" and processed_value != []:
                    is_row_empty = False

        if is_chat_task and is_row_empty:
            continue

        result.append(row_dict)
        total_rows += 1
        if is_row_empty:
            fully_empty_rows += 1

    if total_rows > 0 and (fully_empty_rows / total_rows) > 0.8:
        raise ValueError(f"More than 80% of rows are fully empty ({fully_empty_rows}/{total_rows} rows)")

    result = _validate_dpo_data(result, task)

    return result


def _validate_dpo_data(result: list[dict], task: AnyTextTypeRawTask) -> list[dict]:
    if not isinstance(task, DpoRawTask):
        return result

    original_count = len(result)
    filtered_result = [
        row
        for row in result
        if (
            row.get(cst.STANDARD_DPO_PROMPT_COLUMN, "").strip()
            and row.get(cst.STANDARD_DPO_CHOSEN_COLUMN, "").strip()
            and row.get(cst.STANDARD_DPO_REJECTED_COLUMN, "").strip()
        )
    ]

    if len(filtered_result) < original_count:
        logger.warning(f"Filtered out {original_count - len(filtered_result)} DPO rows with empty prompt/chosen/rejected fields")

    if len(filtered_result) == 0:
        raise ValueError("All DPO data points have empty prompt, chosen, or rejected fields")

    return filtered_result


async def _process_and_upload_datasets(
    task: AnyTextTypeRawTask,
    train_dataset,
    test_dataset,
    columns_to_sample,
    should_reupload_train,
    should_reupload_test,
    ds_hf_name=None,
    psql_db=None,
):
    files_to_delete = []
    logger.info("Processing and uploading datasets to MinIO storage")
    logger.info(f"Train dataset: {train_dataset}\nTest dataset: {test_dataset}")
    logger.info(f"Columns to sample: {columns_to_sample}")
    logger.info(f"Should reupload train: {should_reupload_train}, test: {should_reupload_test}")
    logger.info(f"Dataset HF name: {ds_hf_name}")

    try:
        if should_reupload_train:
            train_data_json = change_to_json_format(train_dataset, columns_to_sample, task)

            await _validate_and_filter_grpo_reward_functions(task, train_data_json, psql_db)

            train_json_path, train_json_size = await save_json_to_temp_file(train_data_json, prefix="train_data_")
            files_to_delete.append(train_json_path)
            await _check_file_size(train_json_size, "train_data")
            train_json_url = await upload_file_to_minio(
                train_json_path, cst.BUCKET_NAME, f"{os.urandom(8).hex()}_train_data.json"
            )
        else:
            train_json_url = train_dataset
        if should_reupload_test:
            test_data_json = change_to_json_format(test_dataset, columns_to_sample, task)
            test_json_path, test_json_size = await save_json_to_temp_file(test_data_json, prefix="test_data_")
            files_to_delete.append(test_json_path)
            await _check_file_size(test_json_size, "test_data")
            test_json_url = await upload_file_to_minio(test_json_path, cst.BUCKET_NAME, f"{os.urandom(8).hex()}_test_data.json")
        else:
            test_json_url = test_dataset
    except Exception as e:
        logger.error(f"There was a problem going to json {e}")
        raise e

    logger.info(f"Train json url: {train_json_url}\nTest json url: {test_json_url}")

    if not train_json_url:
        raise Exception("Failed to upload training data to MinIO storage")
    if not test_json_url:
        raise Exception("Failed to upload test data to MinIO storage")

    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)

    if ds_hf_name:
        delete_dataset_from_cache(ds_hf_name)

    return (
        test_json_url.strip('"'),
        train_json_url.strip('"'),
    )


async def _validate_and_filter_grpo_reward_functions(task: GrpoRawTask, json_data: list[dict], psql_db) -> bool:
    """
    Validate GRPO reward functions with real dataset sample and update database if needed.
    Returns True if any functions were filtered out.
    """
    if not isinstance(task, GrpoRawTask) or not task.reward_functions:
        return False

    sample_size = min(5, len(json_data))
    json_sample = json_data[:sample_size]

    valid_reward_functions = []
    for rf in task.reward_functions:
        is_valid, error_msg, func = validate_reward_function(rf.reward_func, json_sample)
        if is_valid:
            valid_reward_functions.append(rf)
        else:
            logger.warning(f"Removing invalid reward function: {error_msg}")

    if not valid_reward_functions:
        raise ValueError("No valid reward functions remain after dataset validation")

    # Update database if any functions were filtered out
    if len(valid_reward_functions) != len(task.reward_functions):
        logger.info(f"Filtering reward functions: {len(task.reward_functions)} -> {len(valid_reward_functions)}")

        # Update task object
        task.reward_functions = valid_reward_functions

        # Update database with filtered reward functions
        await update_task(task, psql_db)

        return True

    return False


def extract_grpo_extra_columns(task: GrpoRawTask) -> list[str]:
    """
    Extract all unique arguments from reward functions excluding field_prompt.
    """
    all_args = set()

    for reward_function in task.reward_functions:
        parsed = ast.parse(reward_function.reward_func)

        for node in ast.walk(parsed):
            if isinstance(node, ast.FunctionDef):
                all_args |= {arg.arg for arg in node.args.args}
                break

    return list(all_args - {task.field_prompt, "completions"})


def pick_columns_to_sample(task: AnyTextTypeRawTask, dataset: Dataset = None) -> list[str]:
    if isinstance(task, InstructTextRawTask):
        columns_to_sample = [cst.STANDARD_INSTRUCT_COLUMN, cst.STANDARD_OUTPUT_COLUMN]

        if task.field_input and (dataset is None or cst.STANDARD_INPUT_COLUMN in dataset.column_names):
            columns_to_sample.append(cst.STANDARD_INPUT_COLUMN)
        if task.field_system and (dataset is None or cst.STANDARD_SYSTEM_COLUMN in dataset.column_names):
            columns_to_sample.append(cst.STANDARD_SYSTEM_COLUMN)

        return columns_to_sample
    elif isinstance(task, DpoRawTask):
        columns_to_sample = [cst.STANDARD_DPO_PROMPT_COLUMN, cst.STANDARD_DPO_CHOSEN_COLUMN, cst.STANDARD_DPO_REJECTED_COLUMN]
        if task.field_system:
            columns_to_sample.append(cst.STANDARD_SYSTEM_COLUMN)
    elif isinstance(task, GrpoRawTask):
        columns_to_sample = [cst.STANDARD_GRPO_PROMPT_COLUMN]
        if task.extra_column:
            columns_to_sample.append(cst.STANDARD_GRPO_EXTRA_COLUMN)
    elif isinstance(task, ChatRawTask):
        columns_to_sample = [task.chat_column if task.chat_column else cst.STANDARD_CHAT_MESSAGES_COLUMN]
    elif isinstance(task, EnvRawTask):
        raise ValueError("Environment tasks do not use datasets")
    else:
        raise ValueError(f"Unsupported task type: {task.task_type}")
    return columns_to_sample


def standardize_column_names(dataset: Dataset, task: InstructTextRawTask) -> Dataset:
    column_mapping = {}

    if task.field_instruction in dataset.column_names:
        column_mapping[task.field_instruction] = cst.STANDARD_INSTRUCT_COLUMN
    else:
        raise ValueError(f"Instruction column {task.field_instruction} not found in dataset")

    if task.field_input:
        if task.field_input in dataset.column_names:
            column_mapping[task.field_input] = cst.STANDARD_INPUT_COLUMN
        else:
            raise ValueError(f"Input column {task.field_input} not found in dataset")

    if task.field_output in dataset.column_names:
        column_mapping[task.field_output] = cst.STANDARD_OUTPUT_COLUMN
    else:
        raise ValueError(f"Output column {task.field_output} not found in dataset")

    if task.field_system:
        if task.field_system in dataset.column_names:
            column_mapping[task.field_system] = cst.STANDARD_SYSTEM_COLUMN
        else:
            raise ValueError(f"System column {task.field_system} not found in dataset")

    for old_name, new_name in column_mapping.items():
        if old_name != new_name:
            dataset = dataset.rename_column(old_name, new_name)

    return dataset


def standardize_dpo_column_names(dataset: Dataset, task: DpoRawTask) -> Dataset:
    column_mapping = {}

    if task.field_prompt in dataset.column_names:
        column_mapping[task.field_prompt] = cst.STANDARD_DPO_PROMPT_COLUMN
    else:
        raise ValueError(f"Prompt column {task.field_prompt} not found in dataset")

    if task.field_chosen in dataset.column_names:
        column_mapping[task.field_chosen] = cst.STANDARD_DPO_CHOSEN_COLUMN
    else:
        raise ValueError(f"Chosen column {task.field_chosen} not found in dataset")

    if task.field_rejected in dataset.column_names:
        column_mapping[task.field_rejected] = cst.STANDARD_DPO_REJECTED_COLUMN
    else:
        raise ValueError(f"Rejected column {task.field_rejected} not found in dataset")

    if task.field_system:
        if task.field_system in dataset.column_names:
            column_mapping[task.field_system] = cst.STANDARD_SYSTEM_COLUMN
        else:
            raise ValueError(f"System column {task.field_system} not found in dataset")

    for old_name, new_name in column_mapping.items():
        if old_name != new_name:
            dataset = dataset.rename_column(old_name, new_name)

    return dataset


def standardize_grpo_column_names(dataset: Dataset, task: GrpoRawTask) -> Dataset:
    column_mapping = {}

    if task.field_prompt in dataset.column_names:
        column_mapping[task.field_prompt] = cst.STANDARD_GRPO_PROMPT_COLUMN
    else:
        raise ValueError(f"Prompt column {task.field_prompt} not found in dataset")

    if task.extra_column and task.extra_column in dataset.column_names:
        column_mapping[task.extra_column] = cst.STANDARD_GRPO_EXTRA_COLUMN

    for old_name, new_name in column_mapping.items():
        if old_name != new_name:
            dataset = dataset.rename_column(old_name, new_name)

    return dataset

# TODO changes needed here?
async def prepare_text_task(task: AnyTextTypeRawTask, keypair: Keypair, psql_db=None) -> tuple[str, str, str]:
    # Environment tasks: trainer ignores the dataset and generates its own proxy dataset
    # Just return dummy URLs to satisfy the interface - they won't be used
    if isinstance(task, EnvRawTask):
        logger.info("Environment task detected - returning dummy dataset URLs (trainer will generate proxy dataset)")
        dummy_url = "env_task_dummy_dataset"
        return (dummy_url, dummy_url)
    
    train_dataset_name = task.training_data if task.training_data else task.ds

    if not task.test_data:
        logger.info(f"Preparing {train_dataset_name}")

        try:
            dataset = await download_and_load_dataset(train_dataset_name, task.file_format)
        except Exception as e:
            logger.info(f"There was an issue loading the dataset: {e}")
            raise e

        if isinstance(task, InstructTextRawTask):
            dataset = standardize_column_names(dataset, task)
        elif isinstance(task, DpoRawTask):
            dataset = standardize_dpo_column_names(dataset, task)
        elif isinstance(task, GrpoRawTask):
            dataset = standardize_grpo_column_names(dataset, task)

        dataset_dict = await train_test_split(dataset)
        train_ds = dataset_dict["train"]
        test_ds = dataset_dict["test"]
    else:
        logger.info(f"Preparing train and test datasets. Train: {task.training_data}, Test: {task.test_data}")
        try:
            train_ds = await download_and_load_dataset(task.training_data, task.file_format)
            test_ds = await download_and_load_dataset(task.test_data, task.file_format)

            if isinstance(task, InstructTextRawTask):
                train_ds = standardize_column_names(train_ds, task)
                test_ds = standardize_column_names(test_ds, task)
            elif isinstance(task, DpoRawTask):
                train_ds = standardize_dpo_column_names(train_ds, task)
                test_ds = standardize_dpo_column_names(test_ds, task)
            elif isinstance(task, GrpoRawTask):
                train_ds = standardize_grpo_column_names(train_ds, task)
                test_ds = standardize_grpo_column_names(test_ds, task)

        except Exception as e:
            logger.info(f"There was an issue loading the dataset: {e}")
            raise e

    total_size = len(train_ds) + len(test_ds)
    check_ds_num_rows(total_size)

    columns_to_sample = pick_columns_to_sample(task, train_ds)

    if any(col not in train_ds.column_names for col in columns_to_sample):
        raise ValueError(f"Column {columns_to_sample} not found in train dataset")

    return await _process_and_upload_datasets(
        task,
        train_ds,
        test_ds,
        columns_to_sample,
        True,  # always reupload train
        True,  # always reupload test
        train_dataset_name if task.file_format == FileFormat.HF else None,
        psql_db,
    )


async def prepare_image_task(image_text_pairs: list[ImageTextPair]) -> tuple[str, str]:
    Path(cst.TEMP_PATH_FOR_IMAGES).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cst.TEMP_PATH_FOR_IMAGES) as source_dir:
        for i, pair in enumerate(image_text_pairs):
            txt_path = Path(source_dir) / f"{i}.txt"
            await download_s3_file(pair.text_url, str(txt_path))

            tmp_img_path = Path(await download_s3_file(pair.image_url))
            img_extension = tmp_img_path.suffix
            img_path = Path(source_dir) / f"{i}{img_extension}"
            shutil.move(tmp_img_path, img_path)

        test_zip_path, train_zip_path = train_test_split_image(dataset_path=source_dir)

        test_url = await upload_file_to_minio(
            file_path=str(test_zip_path), bucket_name=cst.BUCKET_NAME, object_name=f"{os.urandom(8).hex()}_test_data.zip"
        )
        train_url = await upload_file_to_minio(
            file_path=str(train_zip_path), bucket_name=cst.BUCKET_NAME, object_name=f"{os.urandom(8).hex()}_train_data.zip"
        )

        return (test_url.strip('"'), train_url.strip('"'))


async def _check_file_size(file_size: int, file_type: str) -> None:
    if file_size > cst.MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"{file_type} data size ({file_size} bytes) exceeds maximum allowed size of {cst.MAX_FILE_SIZE_BYTES} bytes"
        )


def check_ds_num_rows(num_rows: int) -> int:
    if num_rows < cst.MINIMUM_DATASET_ROWS:
        error_msg = f"Dataset has only {num_rows} rows, minimum required is {cst.MINIMUM_DATASET_ROWS}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return num_rows
