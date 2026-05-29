import asyncio
import random
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import AsyncGenerator
from uuid import UUID

from substrateinterface import Keypair

import validator.core.constants as vcst
from core.models.payload_models import ImageModelInfo
from core.models.payload_models import ImageModelsResponse
from core.models.payload_models import InstructTextDatasetColumnsResponse
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from core.reward_templates import sample_template_groups
from validator.core.config import Config
from validator.core.models import Dataset
from validator.core.models import DpoRawTask
from validator.core.models import EnvRawTask
from validator.core.models import GrpoRawTask
from validator.core.models import InstructTextRawTask
from validator.core.models import RawTask
from validator.core.models import RewardFunction
from validator.db.sql import grpo as grpo_sql
from validator.db.sql.tasks import add_task
from validator.utils.call_endpoint import call_content_service
from validator.utils.logging import get_logger
from validator.utils.util import retry_with_backoff


logger = get_logger(__name__)

SUPPORTED_ENV_MODELS = [
    "Qwen/Qwen2-7B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "codellama/CodeLlama-7b-Instruct-hf",
    "NousResearch/Hermes-3-Llama-3.2-3B",
]


def maybe_get_yarn_factor() -> int | None:
    """
    Randomly decide whether to apply YaRN extension and return the factor.
    """
    if random.random() < vcst.YARN_EXTENSION_PROBABILITY:
        return random.choice(vcst.YARN_TOURNAMENT_FACTORS)
    return None



async def _get_text_models(
    keypair: Keypair, smallest_size_b: float = 0.1, largest_size_b: float = 12.0
) -> AsyncGenerator[str, None]:
    min_params = int(smallest_size_b * 1_000_000_000)
    max_params = int(largest_size_b * 1_000_000_000)
    params = {"min_params": min_params, "max_params": max_params}

    while True:
        response = await call_content_service(
            vcst.GET_RANDOM_MODELS_ENDPOINT,
            keypair,
            params=params,
        )
        if not isinstance(response, list):
            raise TypeError("Expected a list of responses from GET_ALL_MODELS_ENDPOINT")
        models: list[dict[str, Any]] = response
        model_ids = [model.get(vcst.GET_ALL_MODELS_ID, "") for model in models]
        random.shuffle(model_ids)
        for model_id in model_ids:
            yield model_id


async def _get_image_models(keypair: Keypair) -> AsyncGenerator[ImageModelInfo, None]:
    while True:
        response_data = await call_content_service(vcst.GET_IMAGE_MODELS_ENDPOINT, keypair)
        try:
            response = ImageModelsResponse.model_validate(response_data)
        except Exception as e:
            logger.error(f"Invalid response format from {vcst.GET_IMAGE_MODELS_ENDPOINT}: {response_data}. Error: {e}")
            await asyncio.sleep(5)
            continue

        models = response.models
        random.shuffle(models)
        for model_info in models:
            yield model_info


async def _get_datasets_for_bin(min_rows: int, max_rows: int, keypair: Keypair, dpo: bool) -> AsyncGenerator[Dataset, None]:
    """Get datasets for a specific size bin."""
    while True:
        # params = {"min_rows": min_rows, "max_rows": max_rows, "dpo": dpo}
        params = {"dpo": dpo}
        try:
            response = await call_content_service(vcst.GET_RANDOM_DATASETS_ENDPOINT, keypair, params)
            if not isinstance(response, list):
                raise TypeError("Expected a list of responses from GET_ALL_DATASETS_ENDPOINT")

            dataset_dicts: list[dict[str, Any]] = response
            logger.info(f"[DATASET_BIN] Got {len(dataset_dicts)} dataset dicts from content service")
            datasets = []
            for idx, ds in enumerate(dataset_dicts):
                try:
                    dataset = Dataset.model_validate(ds)
                    datasets.append(dataset)
                except Exception as exc:
                    logger.warning(f"[DATASET_BIN] Failed to validate dataset {idx + 1}: {exc}")

            logger.info(f"[DATASET_BIN] Successfully validated {len(datasets)} datasets")
            random.shuffle(datasets)

            for dataset in datasets:
                logger.info(
                    f"[DATASET_BIN] Yielding dataset: {dataset.dataset_id} (rows: {dataset.num_rows}, "
                    f"bytes: {dataset.num_bytes_parquet_files}, "
                    f"dpo_available: {dataset.dpo_available})"
                )
                yield dataset

        except Exception as e:
            logger.error(f"[DATASET_BIN] Failed to fetch datasets for bin {min_rows}-{max_rows} rows: {e}")
            logger.info("[DATASET_BIN] Sleeping 5 seconds before retry...")
            await asyncio.sleep(5)


async def _get_instruct_text_datasets(keypair: Keypair) -> AsyncGenerator[Dataset, None]:
    """Round-robin generator that cycles through all dataset size bins."""

    bin_generators = [
        _get_datasets_for_bin(min_rows, max_rows, keypair, False) for min_rows, max_rows in vcst.DATASET_BINS_TO_SAMPLE
    ]

    while True:
        for generator in bin_generators:
            try:
                dataset = await anext(generator)
                yield dataset
            except StopAsyncIteration:
                continue
            except Exception as e:
                logger.warning(f"Error getting next dataset from bin: {e}")
                continue


async def _get_dpo_datasets(keypair: Keypair) -> AsyncGenerator[Dataset, None]:
    """Round-robin generator that cycles through all dataset size bins."""

    logger.info("I AM GETTIG THE DPO DATASETS")
    bin_generators = [
        _get_datasets_for_bin(min_rows, max_rows, keypair, True) for min_rows, max_rows in vcst.DATASET_BINS_TO_SAMPLE
    ]

    while True:
        for generator in bin_generators:
            try:
                logger.info(f"We have picked {generator}")
                dataset = await anext(generator)
                yield dataset
            except StopAsyncIteration:
                continue
            except Exception as e:
                logger.warning(f"Error getting next dataset from bin: {e}")
                continue


async def _get_columns_for_instruct_dataset(
    dataset_id: str,
    keypair: Keypair,
) -> InstructTextDatasetColumnsResponse:
    from validator.utils.call_endpoint import call_content_service_fast

    url = vcst.GET_COLUMNS_FOR_DATASET_ENDPOINT.replace("{dataset}", dataset_id)
    logger.info(f"Getting columns for dataset {dataset_id} - ACTUAL MAPPING CALL")

    response = await call_content_service_fast(url, keypair)
    if not isinstance(response, dict):
        raise TypeError(f"Expected dictionary response, got {type(response)}")
    try:
        columns = InstructTextDatasetColumnsResponse.model_validate(response)
    except Exception as exc:
        logger.error(f"The get columns for dataset endpoint should return a DatasetColumnsResponse type: {exc}")
        raise TypeError(f"The get columns for dataset endpoint should return a DatasetColumnsResponse type: {exc}")
    return columns


def _get_training_hours_from_num_rows(num_rows: int) -> int:
    """Randomly select training hours for a given dataset size in bytes based on range bins."""
    min_hours, max_hours = 0, 0
    for min_rows, max_rows in vcst.INSTRUCT_TEXT_DATASET_BINS_TO_TRAINING_HOURS_RANGE.keys():
        if min_rows <= num_rows <= max_rows:
            min_hours, max_hours = vcst.INSTRUCT_TEXT_DATASET_BINS_TO_TRAINING_HOURS_RANGE[(min_rows, max_rows)]
            break
    if min_hours == 0 and max_hours == 0:
        raise ValueError(f"No training hours range found for {num_rows} rows")
    return random.randint(min_hours, max_hours)


def _get_training_hours_for_environment_task() -> int:
    return 3


async def get_dataset(
    datasets_generator: AsyncGenerator[Dataset, None],
    task_type: TaskType | None = None,
    keypair: Keypair | None = None,
) -> Dataset:
    """Get a single dataset from the generator, validating column availability."""
    while True:
        dataset = await anext(datasets_generator)

        if task_type and keypair and task_type != TaskType.DPOTASK:
            try:
                from validator.utils.call_endpoint import call_content_service_fast

                url = vcst.GET_COLUMNS_FOR_DATASET_ENDPOINT.replace("{dataset}", dataset.dataset_id)
                logger.info(f"PRE-VALIDATION: Checking column mapping for dataset {dataset.dataset_id}")
                await call_content_service_fast(url, keypair)
                logger.info(f"PRE-VALIDATION: Dataset {dataset.dataset_id} column mapping validated successfully")
                logger.info(f"Selected dataset: {dataset.dataset_id}")
                return dataset
            except Exception as e:
                logger.warning(f"Dataset {dataset.dataset_id} failed column validation, skipping: {e}")
                continue
        else:
            logger.info(f"Selected dataset: {dataset.dataset_id}")
            return dataset


@retry_with_backoff
async def create_synthetic_dpo_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[Dataset, None],
) -> RawTask:
    logger.info("DPO task")
    model_id = await anext(models)
    logger.info(f"We picked {model_id}")

    dataset = await get_dataset(datasets, task_type=TaskType.DPOTASK, keypair=config.keypair)

    logger.info(f"Selected dataset: {dataset.dataset_id} (rows: {dataset.num_rows}, bytes: {dataset.num_bytes_parquet_files})")

    number_of_hours = _get_training_hours_from_num_rows(dataset.num_rows)
    assert dataset.dpo_rejected_column, "we should have a reject column"
    assert dataset.dpo_accepted_column, "we should have a accepted column"
    assert dataset.dpo_prompt_column, "we should have a prompt column"

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    yarn_factor = maybe_get_yarn_factor()
    task = DpoRawTask(
        model_id=model_id,
        ds=dataset.dataset_id,
        field_system=None,
        field_prompt=dataset.dpo_prompt_column,
        field_chosen=dataset.dpo_accepted_column,
        field_rejected=dataset.dpo_rejected_column,
        status=TaskStatus.PENDING,
        is_organic=False,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=number_of_hours,
        account_id=vcst.NULL_ACCOUNT_ID,
        yarn_factor=yarn_factor,
    )
    logger.info(f"New DPO task created with dataset {dataset.dataset_id}, yarn_factor={yarn_factor}")

    task = await add_task(task, config.psql_db)

    return task


async def _get_generic_reward_functions(config: Config) -> list[RewardFunction]:
    total_rewards = random.randint(vcst.MIN_NUM_REWARD_FUNCTIONS, vcst.MAX_NUM_REWARD_FUNCTIONS)

    code_strings = sample_template_groups(n=total_rewards, rng=random.Random())

    reward_functions = [
        RewardFunction(reward_func=code, is_generic=True, reward_weight=1.0)
        for code in code_strings
    ]

    reward_functions = _randomize_reward_weights(reward_functions)

    return reward_functions


def _randomize_reward_weights(reward_functions: list[RewardFunction]) -> list[RewardFunction]:
    # Generate random weights
    random_weights = [random.uniform(0.1, 10.0) for _ in reward_functions]

    # Normalize to sum to 1
    weight_sum = sum(random_weights)
    normalized_weights = [w / weight_sum for w in random_weights]

    return [
        RewardFunction(
            reward_id=reward_function.reward_id,
            reward_func=reward_function.reward_func,
            func_hash=reward_function.func_hash,
            is_generic=reward_function.is_generic,
            reward_weight=normalized_weight,
        )
        for reward_function, normalized_weight in zip(reward_functions, normalized_weights)
    ]


@retry_with_backoff
async def create_synthetic_grpo_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[Dataset, None],
) -> RawTask:
    model_id = await anext(models)

    dataset = await get_dataset(datasets, task_type=TaskType.GRPOTASK, keypair=config.keypair)

    number_of_hours = _get_training_hours_from_num_rows(dataset.num_rows)
    columns = await _get_columns_for_instruct_dataset(dataset.dataset_id, config.keypair)

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    reward_functions = await _get_generic_reward_functions(config)

    yarn_factor = maybe_get_yarn_factor()
    task = GrpoRawTask(
        model_id=model_id,
        ds=dataset.dataset_id,
        field_prompt=columns.field_instruction,
        reward_functions=reward_functions,
        status=TaskStatus.PENDING,
        is_organic=False,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=number_of_hours,
        account_id=vcst.NULL_ACCOUNT_ID,
        yarn_factor=yarn_factor,
    )
    logger.info(f"New GRPO task created with dataset {dataset.dataset_id}, yarn_factor={yarn_factor}")

    task = await add_task(task, config.psql_db)

    return task


@retry_with_backoff
async def create_synthetic_env_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[Dataset, None],
    exclude_environments: list[str] | None = None,
    force_environment: str | None = None,
) -> RawTask:
    # hardoced model for now. the model and ds generators kept for signature compatibility
    model_id = random.choice(SUPPORTED_ENV_MODELS)

    # Environment tasks don't use the actual dataset - trainer generates a dummy one
    # Use a placeholder to satisfy DB constraint
    dummy_dataset = "env_task_dummy_dataset"

    number_of_hours = _get_training_hours_for_environment_task()

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    if force_environment:
        selected_environment = force_environment
    else:
        game_candidates = ["gin_rummy", "liars_dice", "leduc_poker"]
        if exclude_environments:
            game_candidates = [g for g in game_candidates if g not in exclude_environments]
        selected_environment = random.choice(game_candidates)

    # Generate a random seed for evaluation reproducibility
    eval_seed = random.randint(0, 2**31 - 1)

    task = EnvRawTask(
        model_id=model_id,
        ds=dummy_dataset,
        status=TaskStatus.PENDING,
        environment_name=selected_environment,
        eval_seed=eval_seed,
        is_organic=False,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=number_of_hours,
        account_id=vcst.NULL_ACCOUNT_ID,
        yarn_factor=None,
    )
    logger.info(f"New Environment task created with eval_seed={eval_seed}")

    task = await add_task(task, config.psql_db)

    return task


@retry_with_backoff
async def create_synthetic_affine_grpo_task(
    config: Config,
    models: AsyncGenerator[str, None],
) -> RawTask:
    """Create a synthetic GRPO task using affine data from the content service."""
    model_id = await anext(models)

    try:
        response = await call_content_service(vcst.GET_AFFINE_GRPO_DATA_ENDPOINT, config.keypair)
        logger.info(f"Retrieved affine GRPO data: {response}")

        if not isinstance(response, dict):
            raise ValueError("Expected dict response from affine GRPO data endpoint")

        s3_url = response.get("s3_url")
        if not s3_url:
            raise ValueError("No s3_url in affine GRPO data response")

        logger.info(f"Looking for affine reward functions with IDs: {vcst.AFFINE_REWARD_FN_IDS}")

        affine_reward_functions = []
        for reward_id in vcst.AFFINE_REWARD_FN_IDS:
            logger.debug(f"Attempting to fetch reward function with ID: {reward_id}")
            reward_function = await grpo_sql.get_reward_function_by_id(config.psql_db, UUID(reward_id))
            if reward_function:
                affine_reward_functions.append(reward_function)
            else:
                logger.warning(f"Reward function {reward_id} not found in database")

        logger.info(f"Successfully loaded {len(affine_reward_functions)} affine reward functions")

        # Normalize weights to sum to 1
        if affine_reward_functions:
            num_functions = len(affine_reward_functions)
            normalized_weight = 1.0 / num_functions
            for reward_function in affine_reward_functions:
                logger.info(f"Setting weight for {reward_function.reward_id} to {normalized_weight:.4f}")
                reward_function.reward_weight = normalized_weight

        if not affine_reward_functions:
            logger.error("No affine reward functions found in database, falling back to generic functions")
            reward_functions = await _get_generic_reward_functions(config)
        else:
            logger.info(f"Using {len(affine_reward_functions)} affine-specific reward functions")
            reward_functions = affine_reward_functions

        num_entries = response.get("num_entries", 10_000)
        number_of_hours = _get_training_hours_from_num_rows(num_entries)

        current_time = datetime.utcnow()
        end_timestamp = current_time + timedelta(hours=number_of_hours)

        yarn_factor = maybe_get_yarn_factor()
        task = GrpoRawTask(
            model_id=model_id,
            ds=s3_url,
            field_prompt="prompt",
            reward_functions=reward_functions,
            status=TaskStatus.PENDING,
            is_organic=False,
            created_at=current_time,
            termination_at=end_timestamp,
            hours_to_complete=number_of_hours,
            account_id=vcst.NULL_ACCOUNT_ID,
            file_format=FileFormat.S3,
            extra_column="extra",
            yarn_factor=yarn_factor,
        )

        logger.info(f"New affine GRPO task created with S3 dataset: {s3_url}, yarn_factor={yarn_factor}")

        task = await add_task(task, config.psql_db)

        return task

    except Exception as e:
        logger.error(f"Failed to create affine GRPO task: {e}")


@retry_with_backoff
async def create_synthetic_instruct_text_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[Dataset, None],
) -> RawTask:
    model_id = await anext(models)

    logger.info("INSTRUCT_TASK: Starting dataset selection...")
    dataset = await get_dataset(datasets, task_type=TaskType.INSTRUCTTEXTTASK, keypair=config.keypair)
    logger.info(f"INSTRUCT_TASK: Selected dataset: {dataset.dataset_id}")

    number_of_hours = _get_training_hours_from_num_rows(dataset.num_rows)
    columns = await _get_columns_for_instruct_dataset(dataset.dataset_id, config.keypair)

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    yarn_factor = maybe_get_yarn_factor()
    task = InstructTextRawTask(
        model_id=model_id,
        ds=dataset.dataset_id,
        field_system=None,
        field_instruction=columns.field_instruction,
        field_input=columns.field_input,
        field_output=columns.field_output,
        status=TaskStatus.PENDING,
        is_organic=False,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=number_of_hours,
        account_id=vcst.NULL_ACCOUNT_ID,
        yarn_factor=yarn_factor,
    )
    logger.info(f"INSTRUCT_TASK: Successfully created task with dataset {dataset.dataset_id}, yarn_factor={yarn_factor}")

    task = await add_task(task, config.psql_db)
    logger.info(f"INSTRUCT_TASK: Task saved to database with ID: {task.task_id}")

    return task
