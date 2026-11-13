import ast
import asyncio
import hashlib
import json
import os
import random
import shutil
import stat
import tempfile
import uuid
import zipfile
from collections.abc import Mapping
from math import ceil
from pathlib import Path

from datasets import Dataset
from datasets import DatasetDict
from datasets import concatenate_datasets
from datasets import load_dataset
from pydantic import BaseModel

import core.constants.datasets as core_cst
import validator.tasks.datasets.constants as cst
import validator.tournament.constants as t_cst
from core.constants.credentials import BUCKET_NAME
from core.downloads import download_s3_file
from core.interfaces import KeypairLike
from core.logging import get_logger
from core.logging import redact_url_queries
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import ImageTextPair
from core.models.dataset_models import InstructTextDatasetType
from core.near_duplicate import compute_near_duplicate_rate
from core.near_duplicate import near_duplicate_sample_indices
from validator.db.sql.calibration import get_calibration_row_hashes
from validator.db.sql.calibration import record_calibration_rows
from validator.db.sql.tasks import update_task
from validator.evaluation.dataset_configs import get_default_dataset_config
from validator.infrastructure.cache import delete_dataset_from_cache
from validator.tasks.datasets.calibration import ROW_HASH_COLUMN
from validator.tasks.datasets.calibration import carve_calibration
from validator.tasks.datasets.calibration import shared_split_seed
from validator.tasks.datasets.calibration import with_row_hashes
from validator.tasks.datasets.hub_safety import resolve_code_free_hub_dataset
from validator.tasks.details import save_json_to_temp_file
from validator.tasks.details import upload_file_to_minio
from validator.tasks.models import AnyTextTypeRawTask
from validator.tasks.models import ChatRawTask
from validator.tasks.models import DpoRawTask
from validator.tasks.models import EnvRawTask
from validator.tasks.models import GrpoRawTask
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantRawTask
from validator.tasks.rewards.functions import validate_reward_function
from validator.tournament.models import ContinuousRowsPart


logger = get_logger(__name__)


class PreparedTextSplit(BaseModel):
    """The uploaded JSON files of a prepared text task's split (never logged raw: the URLs are presigned)."""

    test_url: str
    train_url: str
    calibration_url: str | None = None  # a carved calibration set, or the one carried from the given split
    calibration_row_hashes: list[str] = []  # raw-row hashes of the rows carved at this prep, as recorded per dataset


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
    tmp_dir = Path(cst.TEMP_PATH_FOR_IMAGES) / str(random_tmp_id)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    root = tmp_dir.resolve()
    total_uncompressed = 0
    try:
        with zipfile.ZipFile(zip_file_path, "r") as archive:
            entries = archive.infolist()
            if len(entries) > cst.MAX_ARCHIVE_ENTRIES:
                raise ValueError(f"Archive contains too many entries: {len(entries)}")
            for entry in entries:
                if entry.flag_bits & 0x1:
                    raise ValueError("Encrypted archive entries are not allowed")
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Archive symlinks are not allowed")
                target = (root / entry.filename).resolve()
                if target != root and root not in target.parents:
                    raise ValueError("Archive entry escapes the extraction directory")
                total_uncompressed += entry.file_size
                if total_uncompressed > cst.MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ValueError("Archive uncompressed size exceeds the limit")
                if entry.file_size and entry.compress_size == 0:
                    raise ValueError("Archive entry has an invalid compression ratio")
                if entry.compress_size and entry.file_size / entry.compress_size > cst.MAX_ARCHIVE_COMPRESSION_RATIO:
                    raise ValueError("Archive compression ratio exceeds the limit")
            archive.extractall(root)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return str(root)


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


async def train_test_split(
    dataset: Dataset, test_size: float = cst.TRAIN_TEST_SPLIT_PERCENTAGE, n_test: int | None = None
) -> DatasetDict:
    """n_test, when given, is the exact test row count (used when recorded calibration rows were removed from dataset)."""
    logger.info(f"Splitting dataset into train and test with test size {test_size}")

    test_size = n_test if n_test is not None else min(
        int(len(dataset) * test_size),
        cst.MAX_TEST_DATA_POINTS,
    )
    split_dataset = dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)
    logger.info(f"Train set size: {len(split_dataset['train'])}")
    logger.info(f"Test set size: {len(split_dataset['test'])}")

    return split_dataset


def _instruct_row_key(row: Mapping) -> tuple:
    """Identity of an instruct row for de-duplication. Rows with equal keys are duplicates."""
    columns = (cst.STANDARD_SYSTEM_COLUMN, cst.STANDARD_INSTRUCT_COLUMN, cst.STANDARD_INPUT_COLUMN, cst.STANDARD_OUTPUT_COLUMN)
    return tuple(str(row.get(col) or "").strip() for col in columns)


def drop_duplicate_instruct_rows(dataset: Dataset) -> Dataset:
    """Drop duplicate rows (first occurrence kept) so no test row also appears in train."""
    seen: set[tuple] = set()
    keep: list[int] = []
    for idx, row in enumerate(dataset):
        key = _instruct_row_key(row)
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)

    dropped = len(dataset) - len(keep)
    if dropped:
        logger.info(f"Dropped {dropped}/{len(dataset)} duplicate rows before train/test split")
        dataset = dataset.select(keep)
    return dataset


class NearDuplicateDatasetError(ValueError):
    """The train split's near-duplicate rate is at or above the limit."""


def _near_duplicate_texts(task: AnyTextTypeRawTask, train_dataset: Dataset) -> list[str]:
    """The texts model prep measures the rate on, so both stages agree."""
    rows = train_dataset.select(near_duplicate_sample_indices(len(train_dataset)))
    if isinstance(task, InstructTextRawTask):
        prompt_columns = (cst.STANDARD_SYSTEM_COLUMN, cst.STANDARD_INSTRUCT_COLUMN, cst.STANDARD_INPUT_COLUMN)
        texts = []
        for row in rows:
            prompt = " ".join(str(row[column]) for column in prompt_columns if row.get(column))
            texts.append(f"{prompt} {row.get(cst.STANDARD_OUTPUT_COLUMN) or ''}")
        return texts
    return [str(row.get(cst.STANDARD_DPO_PROMPT_COLUMN) or "") for row in rows]


async def reject_near_duplicate_dataset(task: AnyTextTypeRawTask, train_dataset: Dataset) -> None:
    """Fail data prep on a near-duplicate dataset, before a GPU is spent on model prep. Organic tasks are exempt."""
    if task.is_organic or not isinstance(task, (InstructTextRawTask, DpoRawTask, GrpoRawTask)):
        return
    texts = _near_duplicate_texts(task, train_dataset)
    rate = await asyncio.to_thread(compute_near_duplicate_rate, texts)
    if rate >= t_cst.MAX_NEAR_DUPLICATE_RATE:
        raise NearDuplicateDatasetError(
            f"Dataset {task.ds} near-duplicate rate {rate:.3f} >= {t_cst.MAX_NEAR_DUPLICATE_RATE}"
        )
    logger.info(f"Dataset {task.ds} near-duplicate rate {rate:.3f} is under the limit")


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

    if len(dataset_entries) < core_cst.MIN_IMAGE_TEXT_PAIRS:
        raise ValueError(
            f"Image dataset has {len(dataset_entries)} image/text pairs, "
            f"minimum required is {core_cst.MIN_IMAGE_TEXT_PAIRS}"
        )
    if len(dataset_entries) > core_cst.MAX_IMAGE_TEXT_PAIRS:
        raise ValueError(
            f"Image dataset has {len(dataset_entries)} image/text pairs, "
            f"maximum allowed is {core_cst.MAX_IMAGE_TEXT_PAIRS}"
        )

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
        revision = resolve_code_free_hub_dataset(dataset_name)
        config_name = get_default_dataset_config(dataset_name, revision=revision)
        dataset = load_dataset(dataset_name, config_name, revision=revision, trust_remote_code=False)

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


def _is_chat_task(task: AnyTextTypeRawTask | QuantRawTask | None) -> bool:
    return isinstance(task, ChatRawTask) or (
        isinstance(task, QuantRawTask) and isinstance(task.dataset_type, ChatTemplateDatasetType)
    )


def _chat_fields(task: ChatRawTask | QuantRawTask) -> tuple[str, str]:
    """(role field, content field) of a chat task's messages."""
    source = task.dataset_type if isinstance(task, QuantRawTask) else task
    return source.chat_role_field, source.chat_content_field


def change_to_json_format(dataset: Dataset, columns: list[str], task: AnyTextTypeRawTask | QuantRawTask = None):
    result = []
    total_rows = 0
    fully_empty_rows = 0
    is_chat_task = _is_chat_task(task)
    if is_chat_task:
        role_field, content_field = _chat_fields(task)

    for idx, row in enumerate(dataset):
        row_dict = {}
        is_row_empty = True
        for col in columns:
            if col in row:
                value = row[col]

                # Ensure consistent data types: strings for non-ChatTask, preserve type for ChatTask
                if is_chat_task:
                    processed_value = process_chat_row(value, role_field, content_field)
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
    calibration_dataset=None,
) -> PreparedTextSplit:
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
                train_json_path, BUCKET_NAME, f"{os.urandom(8).hex()}_train_data.json"
            )
        else:
            train_json_url = train_dataset
        if should_reupload_test:
            test_data_json = change_to_json_format(test_dataset, columns_to_sample, task)
            test_json_path, test_json_size = await save_json_to_temp_file(test_data_json, prefix="test_data_")
            files_to_delete.append(test_json_path)
            await _check_file_size(test_json_size, "test_data")
            test_json_url = await upload_file_to_minio(test_json_path, BUCKET_NAME, f"{os.urandom(8).hex()}_test_data.json")
        else:
            test_json_url = test_dataset
        calibration_json_url = None
        if calibration_dataset is not None:
            calibration_json = change_to_json_format(calibration_dataset, columns_to_sample, task)
            calibration_path, calibration_size = await save_json_to_temp_file(calibration_json, prefix="calibration_data_")
            files_to_delete.append(calibration_path)
            await _check_file_size(calibration_size, "calibration_data")
            calibration_json_url = await upload_file_to_minio(
                calibration_path, BUCKET_NAME, f"{os.urandom(8).hex()}_calibration_data.json"
            )
            if not calibration_json_url:
                raise Exception("Failed to upload calibration data to MinIO storage")
    except Exception as e:
        logger.error(f"There was a problem going to json {e}")
        raise e

    logger.info(redact_url_queries(f"Train json url: {train_json_url}\nTest json url: {test_json_url}"))

    if not train_json_url:
        raise Exception("Failed to upload training data to MinIO storage")
    if not test_json_url:
        raise Exception("Failed to upload test data to MinIO storage")

    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)

    if ds_hf_name:
        delete_dataset_from_cache(ds_hf_name)

    return PreparedTextSplit(
        test_url=test_json_url.strip('"'),
        train_url=train_json_url.strip('"'),
        calibration_url=calibration_json_url.strip('"') if calibration_json_url else None,
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


def _instruct_columns_to_sample(field_input: str | None, field_system: str | None, dataset: Dataset | None) -> list[str]:
    columns_to_sample = [cst.STANDARD_INSTRUCT_COLUMN, cst.STANDARD_OUTPUT_COLUMN]
    if field_input and (dataset is None or cst.STANDARD_INPUT_COLUMN in dataset.column_names):
        columns_to_sample.append(cst.STANDARD_INPUT_COLUMN)
    if field_system and (dataset is None or cst.STANDARD_SYSTEM_COLUMN in dataset.column_names):
        columns_to_sample.append(cst.STANDARD_SYSTEM_COLUMN)
    return columns_to_sample


def pick_columns_to_sample(task: AnyTextTypeRawTask | QuantRawTask, dataset: Dataset = None) -> list[str]:
    if isinstance(task, InstructTextRawTask):
        return _instruct_columns_to_sample(task.field_input, task.field_system, dataset)
    elif isinstance(task, QuantRawTask):
        dataset_type = task.dataset_type
        if isinstance(dataset_type, InstructTextDatasetType):
            return _instruct_columns_to_sample(dataset_type.field_input, dataset_type.field_system, dataset)
        columns_to_sample = [dataset_type.chat_column or cst.STANDARD_CHAT_MESSAGES_COLUMN]
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
    return standardize_instruct_columns(dataset, task.field_instruction, task.field_input, task.field_output, task.field_system)


def standardize_instruct_columns(
    dataset: Dataset,
    field_instruction: str,
    field_input: str | None,
    field_output: str,
    field_system: str | None,
) -> Dataset:
    column_mapping = {}

    if field_instruction in dataset.column_names:
        column_mapping[field_instruction] = cst.STANDARD_INSTRUCT_COLUMN
    else:
        raise ValueError(f"Instruction column {field_instruction} not found in dataset")

    if field_input:
        if field_input in dataset.column_names:
            column_mapping[field_input] = cst.STANDARD_INPUT_COLUMN
        else:
            raise ValueError(f"Input column {field_input} not found in dataset")

    if field_output in dataset.column_names:
        column_mapping[field_output] = cst.STANDARD_OUTPUT_COLUMN
    else:
        raise ValueError(f"Output column {field_output} not found in dataset")

    if field_system:
        if field_system in dataset.column_names:
            column_mapping[field_system] = cst.STANDARD_SYSTEM_COLUMN
        else:
            raise ValueError(f"System column {field_system} not found in dataset")

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

def _is_quant_instruct_task(task: AnyTextTypeRawTask | QuantRawTask) -> bool:
    return isinstance(task, QuantRawTask) and isinstance(task.dataset_type, InstructTextDatasetType)


def _standardize_quant_instruct_columns(dataset: Dataset, dataset_type: InstructTextDatasetType) -> Dataset:
    # A ChatTemplateDatasetType quant task is left as-is, like ChatRawTask.
    return standardize_instruct_columns(
        dataset, dataset_type.field_instruction, dataset_type.field_input, dataset_type.field_output, dataset_type.field_system
    )


def _task_carves_calibration(task: AnyTextTypeRawTask | QuantRawTask) -> bool:
    """Only an R1 instruct task or a quant task (fixed or freeform) is flagged to carve a calibration set.

    On a quant task with a given train/test split (test_data set) the carve takes the rows from the given train file
    (the continuous-chunk compression tasks); an instruct task with a given split is never carved.
    """
    if not task.carve_calibration:
        return False
    if task.test_data:
        return isinstance(task, QuantRawTask) and not task.calibration_data
    return isinstance(task, (InstructTextRawTask, QuantRawTask))


async def _split_without_recorded(dataset: Dataset, recorded: set[str]) -> DatasetDict:
    """The usual split with rows recorded as calibration rows kept out of the test set and appended to train.

    The test size is still the one the full dataset gives. A no-op (the plain split of dataset) when none of the
    recorded rows are in dataset.
    """
    hashes = dataset[ROW_HASH_COLUMN]
    held = [i for i, row_hash in enumerate(hashes) if row_hash in recorded]
    if not held:
        return await train_test_split(dataset)
    kept = [i for i, row_hash in enumerate(hashes) if row_hash not in recorded]
    n_test = min(int(len(dataset) * cst.TRAIN_TEST_SPLIT_PERCENTAGE), cst.MAX_TEST_DATA_POINTS)
    logger.info(f"Keeping {len(held)} recorded calibration rows out of the test split")
    split = await train_test_split(dataset.select(kept), n_test=n_test)
    return DatasetDict(
        {"train": concatenate_datasets([split["train"], dataset.select(held)]), "test": split["test"]}
    )


def _completion_chars(dataset: Dataset, columns_to_sample: list[str]) -> list[int]:
    """Per-row characters of the supervised text: the output column of an instruct split, else every sampled column.

    On a chat (ShareGPT) split the sampled column is the whole conversation, so user turns are counted too: the token
    estimate overshoots the real supervised count and the carve lands nearer the QUANT_CALIBRATION_MIN_ROWS floor. It only
    sizes the carve; the published supervised-token count is logged at quant prep.
    """
    columns = [cst.STANDARD_OUTPUT_COLUMN] if cst.STANDARD_OUTPUT_COLUMN in columns_to_sample else columns_to_sample
    per_column = [[len(str(value)) if value is not None else 0 for value in dataset[column]] for column in columns]
    return [sum(lengths) for lengths in zip(*per_column)]


def _row_hash(seed: str, row: dict) -> str:
    return hashlib.sha256(f"{seed}\0{json.dumps(row, sort_keys=True, default=str)}".encode()).hexdigest()


def select_continuous_rows(task: AnyTextTypeRawTask | QuantRawTask, train_ds: Dataset) -> Dataset:
    """The continuous instruct task's or the compression tasks' share of one continuous chunk's train rows; any other task
    is returned unchanged.

    The rows are ordered by a hash of (tournament seed, row content), which does not depend on the order of the file. The
    instruct task takes the first QUANT_CONTINUOUS_INSTRUCT_TRAIN_ROWS of that order and the compression tasks the next ones,
    so the two shares are disjoint, each exactly that size, reproducible for a tournament and different for another, and the
    freeform and fixed compression tasks (same seed, same chunk) get identical rows. A chunk with fewer than twice as many
    rows is split into disjoint halves (WARNING). The test split is not touched. Taken here because prep already holds
    the downloaded train file and re-uploads it, and the baseline stats that size the hours are measured on what prep
    uploads (task.training_data).
    """
    selected = t_cst.continuous_rows_part(task)
    if selected is None:
        return train_ds
    part, seed = selected
    size = t_cst.QUANT_CONTINUOUS_INSTRUCT_TRAIN_ROWS
    if len(train_ds) < 2 * size:
        logger.warning(f"Task {task.task_id}: the chunk has {len(train_ds)} train rows, under {2 * size}: using disjoint halves")
        size = len(train_ds) // 2
    hashes = [_row_hash(seed, row) for row in train_ds.to_list()]  # one pass over the table, not one Arrow read per row
    order = sorted(range(len(train_ds)), key=hashes.__getitem__)
    start = 0 if part == ContinuousRowsPart.INSTRUCT else size
    logger.info(f"Task {task.task_id}: {part.value} share, {size} of {len(train_ds)} chunk rows")
    return train_ds.select(sorted(order[start : start + size]))


# TODO changes needed here?
async def prepare_text_task(
    task: AnyTextTypeRawTask | QuantRawTask,
    keypair: KeypairLike | None = None,
    psql_db=None,
) -> PreparedTextSplit:
    # Environment tasks: trainer ignores the dataset and generates its own proxy dataset
    # Just return dummy URLs to satisfy the interface - they won't be used
    if isinstance(task, EnvRawTask):
        logger.info("Environment task detected - returning dummy dataset URLs (trainer will generate proxy dataset)")
        dummy_url = "env_task_dummy_dataset"
        return PreparedTextSplit(test_url=dummy_url, train_url=dummy_url)
    
    train_dataset_name = task.training_data if task.training_data else task.ds
    carve = _task_carves_calibration(task)
    train_hashes: list[str] | None = None
    test_hashes: list[str] | None = None

    if not task.test_data:
        logger.info(f"Preparing {train_dataset_name}")
        recorded = (
            await get_calibration_row_hashes(train_dataset_name, psql_db, exclude_task_id=task.task_id)
            if isinstance(task, QuantRawTask)
            else set()
        )

        try:
            dataset = await download_and_load_dataset(train_dataset_name, task.file_format)
        except Exception as e:
            logger.info(f"There was an issue loading the dataset: {e}")
            raise e

        # Hashed on the raw rows, before standardisation renames columns; only when a recorded or carved row can matter.
        hashed = carve or bool(recorded)
        if hashed:
            dataset = with_row_hashes(dataset)

        if isinstance(task, InstructTextRawTask):
            dataset = standardize_column_names(dataset, task)
            dataset = drop_duplicate_instruct_rows(dataset)
        elif _is_quant_instruct_task(task):
            dataset = _standardize_quant_instruct_columns(dataset, task.dataset_type)
        elif isinstance(task, DpoRawTask):
            dataset = standardize_dpo_column_names(dataset, task)
        elif isinstance(task, GrpoRawTask):
            dataset = standardize_grpo_column_names(dataset, task)

        dataset_dict = await (_split_without_recorded(dataset, recorded) if recorded else train_test_split(dataset))
        train_ds = dataset_dict["train"]
        test_ds = dataset_dict["test"]
        if hashed:
            train_hashes = train_ds[ROW_HASH_COLUMN]
            test_hashes = test_ds[ROW_HASH_COLUMN]
            train_ds = train_ds.remove_columns([ROW_HASH_COLUMN])
            test_ds = test_ds.remove_columns([ROW_HASH_COLUMN])
    else:
        logger.info(redact_url_queries(f"Preparing train and test datasets. Train: {task.training_data}, Test: {task.test_data}"))
        try:
            train_ds = await download_and_load_dataset(task.training_data, task.file_format)
            train_ds = select_continuous_rows(task, train_ds)
            test_ds = await download_and_load_dataset(task.test_data, task.file_format)
            if carve:  # hashed on the raw rows, before standardisation, like the self-split path
                train_ds = with_row_hashes(train_ds)
                test_ds = with_row_hashes(test_ds)

            if isinstance(task, InstructTextRawTask):
                train_ds = standardize_column_names(train_ds, task)
                test_ds = standardize_column_names(test_ds, task)
            elif _is_quant_instruct_task(task):
                train_ds = _standardize_quant_instruct_columns(train_ds, task.dataset_type)
                test_ds = _standardize_quant_instruct_columns(test_ds, task.dataset_type)
            elif isinstance(task, DpoRawTask):
                train_ds = standardize_dpo_column_names(train_ds, task)
                test_ds = standardize_dpo_column_names(test_ds, task)
            elif isinstance(task, GrpoRawTask):
                train_ds = standardize_grpo_column_names(train_ds, task)
                test_ds = standardize_grpo_column_names(test_ds, task)

        except Exception as e:
            logger.info(f"There was an issue loading the dataset: {e}")
            raise e
        if carve:
            train_hashes = train_ds[ROW_HASH_COLUMN]
            test_hashes = test_ds[ROW_HASH_COLUMN]
            train_ds = train_ds.remove_columns([ROW_HASH_COLUMN])
            test_ds = test_ds.remove_columns([ROW_HASH_COLUMN])

    total_size = len(train_ds) + len(test_ds)
    check_ds_num_rows(total_size)
    await reject_near_duplicate_dataset(task, train_ds)

    columns_to_sample = pick_columns_to_sample(task, train_ds)

    if any(col not in train_ds.column_names for col in columns_to_sample):
        raise ValueError(f"Column {columns_to_sample} not found in train dataset")

    calibration_ds = None
    calibration_hashes: list[str] = []
    if carve:
        # A given split is carved on a seed shared by every task of the chunk, so sibling tasks carve identical rows.
        carved = carve_calibration(
            shared_split_seed(task.ds) if task.test_data else str(task.task_id),
            train_hashes, set(test_hashes), _completion_chars(train_ds, columns_to_sample)
        )
        if carved is not None:
            calibration_ds = train_ds.select(carved.calibration_indices)
            calibration_hashes = [train_hashes[i] for i in carved.calibration_indices]
            train_ds = train_ds.select(carved.train_indices)

    prepared = await _process_and_upload_datasets(
        task,
        train_ds,
        test_ds,
        columns_to_sample,
        True,  # always reupload train
        True,  # always reupload test
        train_dataset_name if task.file_format == FileFormat.HF else None,
        psql_db,
        calibration_dataset=calibration_ds,
    )
    if calibration_ds is not None and task.test_data:
        # Not recorded per dataset: a given split is never re-split, and its urls are presigned (no stable dataset key).
        prepared.calibration_row_hashes = calibration_hashes
    elif calibration_ds is not None:
        await record_calibration_rows(train_dataset_name, calibration_hashes, task.task_id, psql_db)
        prepared.calibration_row_hashes = calibration_hashes
    elif task.test_data:
        # A given split is never re-split or re-carved: its calibration rows (if any) are carried through.
        prepared.calibration_url = task.calibration_data
    return prepared


async def prepare_image_task(
    image_text_pairs: list[ImageTextPair] | None = None,
    dataset_zip_url: str | None = None,
) -> tuple[str, str]:
    Path(cst.TEMP_PATH_FOR_IMAGES).mkdir(parents=True, exist_ok=True)

    if dataset_zip_url:
        with tempfile.TemporaryDirectory(dir=cst.TEMP_PATH_FOR_IMAGES) as download_dir:
            local_zip_path = await download_s3_file(dataset_zip_url, save_path=download_dir)
            dataset_path = unzip_to_temp_path(local_zip_path)
            try:
                test_zip_path, train_zip_path = train_test_split_image(dataset_path=dataset_path)
                test_url = await upload_file_to_minio(
                    file_path=str(test_zip_path),
                    bucket_name=BUCKET_NAME,
                    object_name=f"{os.urandom(8).hex()}_test_data.zip",
                )
                train_url = await upload_file_to_minio(
                    file_path=str(train_zip_path),
                    bucket_name=BUCKET_NAME,
                    object_name=f"{os.urandom(8).hex()}_train_data.zip",
                )
            finally:
                shutil.rmtree(dataset_path, ignore_errors=True)

            return (test_url.strip('"'), train_url.strip('"'))

    if not image_text_pairs:
        raise ValueError("Image task requires either image_text_pairs or dataset_zip_url")

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
            file_path=str(test_zip_path), bucket_name=BUCKET_NAME, object_name=f"{os.urandom(8).hex()}_test_data.zip"
        )
        train_url = await upload_file_to_minio(
            file_path=str(train_zip_path), bucket_name=BUCKET_NAME, object_name=f"{os.urandom(8).hex()}_train_data.zip"
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
    if num_rows > cst.MAXIMUM_DATASET_ROWS:
        error_msg = f"Dataset has {num_rows} rows, maximum allowed is {cst.MAXIMUM_DATASET_ROWS}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return num_rows
