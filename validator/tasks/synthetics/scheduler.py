import asyncio
import math
import random
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import AsyncGenerator
from uuid import UUID

from substrateinterface import Keypair

import validator.infrastructure.service_constants as service_cst
import validator.tasks.datasets.constants as data_cst
import validator.tasks.prep.constants as prep_cst
import validator.tasks.synthetics.constants as synth_cst
from core.constants.environments import EnvironmentName
from core.constants.environments import TrainingStartPoint
from core.logging import get_logger
from core.models.dataset_models import FileFormat
from core.models.model_prep_models import EnvBaselineStats
from core.models.payload_models import ImageModelInfo
from core.models.payload_models import ImageModelsResponse
from core.models.payload_models import InstructTextDatasetColumnsResponse
from core.models.reward_models import RewardFunction
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.app.config import Config
from validator.db.database import PSQLDB
from validator.db.sql import grpo as grpo_sql
from validator.db.sql.tasks import add_task
from validator.db.sql.tasks import get_dataset_test_losses
from validator.infrastructure.content_service import call_content_service
from validator.scoring.models import EnvironmentWeight
from validator.tasks.datasets.models import Dataset
from validator.tasks.details import retry_with_backoff
from validator.tasks.models import DpoRawTask
from validator.tasks.models import EnvRawTask
from validator.tasks.models import GrpoRawTask
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import RawTask
from validator.tasks.prep.augmentation import maybe_get_augmentation_config
from validator.tasks.requests import get_model_num_params
from validator.tasks.rewards.templates import sample_template_groups
from validator.tournament import constants as t_cst
from validator.tournament.gpu_requirements import get_tournament_gpu_requirement


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
    if random.random() < prep_cst.YARN_EXTENSION_PROBABILITY:
        return random.choice(prep_cst.YARN_TOURNAMENT_FACTORS)
    return None


def maybe_get_kl_config() -> tuple[bool, float | None]:
    """
    Randomly decide whether this instruct task should ask miners to train with a KL term.

    Returns (use_kl, kl_coef). When enabled, kl_coef is the coefficient the evaluator
    will use to weight KL(finetuned || base) into the loss, and which we send to miners.
    """
    if random.random() < synth_cst.INSTRUCT_KL_TASK_PROBABILITY:
        kl_coef = random.uniform(synth_cst.INSTRUCT_KL_COEFFICIENT_MIN, synth_cst.INSTRUCT_KL_COEFFICIENT_MAX)
        return True, kl_coef
    return False, None


async def _get_text_models(
    keypair: Keypair, smallest_size_b: float = 0.1, largest_size_b: float = 12.0
) -> AsyncGenerator[str, None]:
    min_params = int(smallest_size_b * 1_000_000_000)
    max_params = int(largest_size_b * 1_000_000_000)
    params = {"min_params": min_params, "max_params": max_params}

    while True:
        response = await call_content_service(
            service_cst.GET_RANDOM_MODELS_ENDPOINT,
            keypair,
            params=params,
        )
        if not isinstance(response, list):
            raise TypeError("Expected a list of responses from GET_ALL_MODELS_ENDPOINT")
        models: list[dict[str, Any]] = response
        model_ids = [model.get(service_cst.GET_ALL_MODELS_ID, "") for model in models]
        random.shuffle(model_ids)
        for model_id in model_ids:
            yield model_id


async def _get_image_models(keypair: Keypair) -> AsyncGenerator[ImageModelInfo, None]:
    while True:
        response_data = await call_content_service(service_cst.GET_IMAGE_MODELS_ENDPOINT, keypair)
        try:
            response = ImageModelsResponse.model_validate(response_data)
        except Exception as e:
            logger.error(f"Invalid response format from {service_cst.GET_IMAGE_MODELS_ENDPOINT}: {response_data}. Error: {e}")
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
            response = await call_content_service(service_cst.GET_RANDOM_DATASETS_ENDPOINT, keypair, params)
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
            # The content service ignores row-range params, so enforce bins
            # client-side to keep task durations bounded.
            datasets = [ds for ds in datasets if min_rows <= ds.num_rows <= max_rows]
            logger.info(f"[DATASET_BIN] {len(datasets)} datasets within bin {min_rows}-{max_rows} rows")
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


async def _get_instruct_text_datasets(keypair: Keypair, small_only: bool = False) -> AsyncGenerator[Dataset, None]:
    """Round-robin generator that cycles through all dataset size bins."""

    bins = [t_cst.R1_TEXT_DATASET_BIN] if small_only else data_cst.DATASET_BINS_TO_SAMPLE
    bin_generators = [
        _get_datasets_for_bin(min_rows, max_rows, keypair, False) for min_rows, max_rows in bins
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
        _get_datasets_for_bin(min_rows, max_rows, keypair, True) for min_rows, max_rows in data_cst.DATASET_BINS_TO_SAMPLE
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
    from validator.infrastructure.content_service import call_content_service_fast

    url = service_cst.GET_COLUMNS_FOR_DATASET_ENDPOINT.replace("{dataset}", dataset_id)
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


def _analytic_tokens_per_sec_per_gpu(num_params: float) -> float:
    """Expected miner full-FT throughput per H100: peak_flops * MFU / (6N flops/token)."""
    return data_cst.H100_BF16_TFLOPS * 1e12 * data_cst.ASSUMED_TRAINING_MFU / (6.0 * num_params)


def compute_training_hours(
    tokens_per_epoch: float,
    num_params: float,
    task_type: TaskType,
    measured_tokens_per_sec: float | None = None,
) -> float:
    """Hours for TARGET_TRAINING_EPOCHS over the dataset at expected miner throughput."""
    gpus = get_tournament_gpu_requirement(task_type, int(num_params)).gpu_count
    analytic_tps = _analytic_tokens_per_sec_per_gpu(num_params)
    if measured_tokens_per_sec:
        lo, hi = data_cst.MEASURED_THROUGHPUT_CLAMP
        per_gpu_tps = measured_tokens_per_sec * data_cst.MEASURED_THROUGHPUT_MINER_RATIO
        per_gpu_tps = min(max(per_gpu_tps, analytic_tps * lo), analytic_tps * hi)
    else:
        per_gpu_tps = analytic_tps

    type_mult = data_cst.TASK_TYPE_HOURS_MULTIPLIER.get(task_type, 1.0)
    train_seconds = data_cst.TARGET_TRAINING_EPOCHS * tokens_per_epoch * type_mult / (per_gpu_tps * gpus)
    hours = train_seconds / 3600 + data_cst.TRAINING_OVERHEAD_HOURS

    hours = max(data_cst.TRAINING_HOURS_MIN, math.ceil(hours * 4) / 4)
    return min(hours, data_cst.MAX_TRAINING_HOURS)


def get_grpo_training_hours(num_params: float | None) -> float:
    """GRPO budget: fixed per model-size band, independent of dataset size."""
    params_b = (num_params or data_cst.DEFAULT_MODEL_PARAMS_FOR_HOURS) / 1e9
    hours = next(hours for bound_b, hours in data_cst.GRPO_HOURS_BY_PARAMS_B if params_b <= bound_b)
    return min(data_cst.MAX_TRAINING_HOURS, max(data_cst.TRAINING_HOURS_MIN, hours))


def _get_training_hours_from_num_rows(num_rows: int, model_id: str | None = None, task_type: TaskType | None = None) -> float:
    """Pre-prep estimate using ASSUMED_TOKENS_PER_ROW until measured stats arrive."""
    num_params = get_model_num_params(model_id) if model_id is not None else None
    if not num_params:
        num_params = data_cst.DEFAULT_MODEL_PARAMS_FOR_HOURS
    if task_type == TaskType.GRPOTASK:
        return get_grpo_training_hours(num_params)
    tokens_per_epoch = num_rows * data_cst.ASSUMED_TOKENS_PER_ROW
    return compute_training_hours(tokens_per_epoch, num_params, task_type or TaskType.INSTRUCTTEXTTASK)


def compute_hours_from_baseline_stats(
    current_hours: float,
    baseline_stats,
    task_type: TaskType,
    model_id: str | None = None,
    model_params_count: int | None = None,
) -> float:
    """Post-prep hours from real token counts plus measured fwd/bwd throughput."""
    if isinstance(baseline_stats, EnvBaselineStats) or baseline_stats is None:
        return current_hours

    num_params = model_params_count or (get_model_num_params(model_id) if model_id else None)
    if not num_params:
        num_params = data_cst.DEFAULT_MODEL_PARAMS_FOR_HOURS

    if task_type == TaskType.GRPOTASK:
        return get_grpo_training_hours(num_params)

    dataset_stats = baseline_stats.dataset
    if not dataset_stats.total_tokens or not dataset_stats.num_records:
        return current_hours

    effective_tokens = max(
        dataset_stats.total_tokens,
        dataset_stats.num_records * data_cst.EFFECTIVE_MIN_TOKENS_PER_ROW,
    )
    measured_tps = baseline_stats.throughput.tokens_per_sec if baseline_stats.throughput else None
    return compute_training_hours(effective_tokens, num_params, task_type, measured_tps)


def apply_baseline_ctx_scale(hours: float, baseline_stats) -> float:
    """Backward-compatible wrapper for callers not yet passing task metadata."""
    return hours if baseline_stats is None else hours


def _get_training_hours_for_environment_task(round_number: int = 1) -> float:
    return t_cst.ENV_TRAINING_HOURS


async def _is_dataset_degenerate(ds_name: str, task_type: TaskType, psql_db: PSQLDB) -> bool:
    """Check if a dataset has historically produced degenerate test_loss scores.

    Returns True (degenerate) if:
    - Any historical test_loss < 0.01 (model collapse)
    - For instruct tasks: best (min) test_loss > 2.0 (garbage / unlearnable data)
    - For DPO tasks: average test_loss in [0.68, 0.71] (random noise around ln(2))
    """
    try:
        losses = await get_dataset_test_losses(ds_name, psql_db)
    except Exception as e:
        logger.warning(f"Failed to query historical losses for {ds_name}, allowing dataset: {e}")
        return False

    if not losses:
        return False

    if any(loss < 0.01 for loss in losses):
        logger.warning(f"Dataset {ds_name} rejected: has test_loss < 0.01 (model collapse)")
        return True

    if task_type == TaskType.INSTRUCTTEXTTASK:
        best_loss = min(losses)
        if best_loss > 2.0:
            logger.warning(f"Dataset {ds_name} rejected: best instruct test_loss {best_loss:.4f} > 2.0 (garbage data)")
            return True

    if task_type == TaskType.DPOTASK:
        avg_loss = sum(losses) / len(losses)
        if 0.68 <= avg_loss <= 0.71:
            logger.warning(f"Dataset {ds_name} rejected: avg DPO test_loss {avg_loss:.4f} in noise range [0.68, 0.71]")
            return True

    return False


async def get_dataset(
    datasets_generator: AsyncGenerator[Dataset, None],
    task_type: TaskType | None = None,
    keypair: Keypair | None = None,
    psql_db: PSQLDB | None = None,
    min_rows: int | None = None,
) -> Dataset:
    """Get a single dataset from the generator, validating column availability."""
    while True:
        dataset = await anext(datasets_generator)

        if min_rows and dataset.num_rows < min_rows:
            continue

        if task_type and psql_db:
            if await _is_dataset_degenerate(dataset.dataset_id, task_type, psql_db):
                continue

        if task_type and keypair and task_type != TaskType.DPOTASK:
            try:
                from validator.infrastructure.content_service import call_content_service_fast

                url = service_cst.GET_COLUMNS_FOR_DATASET_ENDPOINT.replace("{dataset}", dataset.dataset_id)
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

    dataset = await get_dataset(datasets, task_type=TaskType.DPOTASK, keypair=config.keypair, psql_db=config.psql_db)

    logger.info(f"Selected dataset: {dataset.dataset_id} (rows: {dataset.num_rows}, bytes: {dataset.num_bytes_parquet_files})")

    number_of_hours = _get_training_hours_from_num_rows(dataset.num_rows, model_id, task_type=TaskType.DPOTASK)
    assert dataset.dpo_rejected_column, "we should have a reject column"
    assert dataset.dpo_accepted_column, "we should have a accepted column"
    assert dataset.dpo_prompt_column, "we should have a prompt column"

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    yarn_factor = maybe_get_yarn_factor()
    augmentation_config = maybe_get_augmentation_config(TaskType.DPOTASK)
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
        account_id=service_cst.NULL_ACCOUNT_ID,
        yarn_factor=yarn_factor,
        augmentation_config=augmentation_config,
    )
    logger.info(f"New DPO task created with dataset {dataset.dataset_id}, augmented={augmentation_config is not None}")

    task = await add_task(task, config.psql_db)

    return task


def _get_generic_reward_functions() -> list[RewardFunction]:
    total_rewards = random.randint(synth_cst.MIN_NUM_REWARD_FUNCTIONS, synth_cst.MAX_NUM_REWARD_FUNCTIONS)

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

    dataset = await get_dataset(
        datasets,
        task_type=TaskType.GRPOTASK,
        keypair=config.keypair,
        min_rows=data_cst.GRPO_MIN_SYNTH_ROWS,
    )

    number_of_hours = _get_training_hours_from_num_rows(dataset.num_rows, model_id, task_type=TaskType.GRPOTASK)
    columns = await _get_columns_for_instruct_dataset(dataset.dataset_id, config.keypair)

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    reward_functions = _get_generic_reward_functions()

    yarn_factor = maybe_get_yarn_factor()
    augmentation_config = maybe_get_augmentation_config(TaskType.GRPOTASK)
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
        account_id=service_cst.NULL_ACCOUNT_ID,
        yarn_factor=yarn_factor,
        augmentation_config=augmentation_config,
    )
    logger.info(f"New GRPO task created with dataset {dataset.dataset_id}, augmented={augmentation_config is not None}")

    task = await add_task(task, config.psql_db)

    return task


@retry_with_backoff
async def create_synthetic_env_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[Dataset, None],
    num_environments: int = 1,
    exclude_environments: list[EnvironmentName] | None = None,
    round_number: int = 1,
    model_id_override: str | None = None,
    training_start_point: TrainingStartPoint = TrainingStartPoint.DEFAULT,
    environment_names_override: list[EnvironmentName] | None = None,
    eval_seed_override: int | None = None,
    exclude_models: list[str] | None = None,
    hours_override: float | None = None,
) -> RawTask:
    if model_id_override:
        model_id = model_id_override
    else:
        candidates = [m for m in SUPPORTED_ENV_MODELS if m not in (exclude_models or [])]
        model_id = random.choice(candidates or SUPPORTED_ENV_MODELS)
    dummy_dataset = "env_task_dummy_dataset"

    number_of_hours = hours_override or _get_training_hours_for_environment_task(round_number)
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    if environment_names_override:
        selected_environments = environment_names_override
    else:
        all_envs = list(EnvironmentName)
        candidates = [g for g in all_envs if g not in (exclude_environments or [])]
        count = min(num_environments, len(candidates))
        selected_environments = random.sample(candidates, count) if candidates else []

    eval_seed = eval_seed_override if eval_seed_override is not None else random.randint(0, 2**31 - 1)

    augmentation_config = maybe_get_augmentation_config(TaskType.ENVIRONMENTTASK)
    weights = [EnvironmentWeight(environment=env) for env in selected_environments]

    augmentation_config = maybe_get_augmentation_config(TaskType.ENVIRONMENTTASK)
    task = EnvRawTask(
        model_id=model_id,
        ds=dummy_dataset,
        status=TaskStatus.PENDING,
        environment_names=selected_environments,
        environment_weights=weights,
        eval_seed=eval_seed,
        is_organic=False,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=number_of_hours,
        account_id=service_cst.NULL_ACCOUNT_ID,
        yarn_factor=None,
        augmentation_config=augmentation_config,
        training_start_point=training_start_point,
    )
    logger.info(
        f"New Environment task: {len(selected_environments)} envs={[e.value for e in selected_environments]}, "
        f"eval_seed={eval_seed}, augmented={augmentation_config is not None}"
    )

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
        response = await call_content_service(synth_cst.GET_AFFINE_GRPO_DATA_ENDPOINT, config.keypair)
        logger.info(f"Retrieved affine GRPO data: {response}")

        if not isinstance(response, dict):
            raise ValueError("Expected dict response from affine GRPO data endpoint")

        s3_url = response.get("s3_url")
        if not s3_url:
            raise ValueError("No s3_url in affine GRPO data response")

        logger.info(f"Looking for affine reward functions with IDs: {synth_cst.AFFINE_REWARD_FN_IDS}")

        affine_reward_functions = []
        for reward_id in synth_cst.AFFINE_REWARD_FN_IDS:
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
            reward_functions = _get_generic_reward_functions()
        else:
            logger.info(f"Using {len(affine_reward_functions)} affine-specific reward functions")
            reward_functions = affine_reward_functions

        num_entries = response.get("num_entries", 10_000)
        number_of_hours = _get_training_hours_from_num_rows(num_entries, model_id, task_type=TaskType.GRPOTASK)

        current_time = datetime.utcnow()
        end_timestamp = current_time + timedelta(hours=number_of_hours)

        yarn_factor = maybe_get_yarn_factor()
        augmentation_config = maybe_get_augmentation_config(TaskType.GRPOTASK)
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
            account_id=service_cst.NULL_ACCOUNT_ID,
            file_format=FileFormat.S3,
            extra_column="extra",
            yarn_factor=yarn_factor,
            augmentation_config=augmentation_config,
        )

        logger.info(f"New affine GRPO task created with S3 dataset: {s3_url}, augmented={augmentation_config is not None}")

        task = await add_task(task, config.psql_db)

        return task

    except Exception as e:
        logger.error(f"Failed to create affine GRPO task: {e}")


@retry_with_backoff
async def create_synthetic_instruct_text_task(
    config: Config,
    models: AsyncGenerator[str, None],
    datasets: AsyncGenerator[Dataset, None],
    enable_kl: bool = False,
) -> RawTask:
    model_id = await anext(models)

    logger.info("INSTRUCT_TASK: Starting dataset selection...")
    dataset = await get_dataset(datasets, task_type=TaskType.INSTRUCTTEXTTASK, keypair=config.keypair, psql_db=config.psql_db)
    logger.info(f"INSTRUCT_TASK: Selected dataset: {dataset.dataset_id}")

    number_of_hours = _get_training_hours_from_num_rows(dataset.num_rows, model_id, task_type=TaskType.INSTRUCTTEXTTASK)
    columns = await _get_columns_for_instruct_dataset(dataset.dataset_id, config.keypair)

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    yarn_factor = maybe_get_yarn_factor()
    augmentation_config = maybe_get_augmentation_config(TaskType.INSTRUCTTEXTTASK)
    use_kl, kl_coef = maybe_get_kl_config() if enable_kl else (False, None)
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
        account_id=service_cst.NULL_ACCOUNT_ID,
        yarn_factor=yarn_factor,
        augmentation_config=augmentation_config,
        use_kl=use_kl,
        kl_coef=kl_coef,
    )
    logger.info(
        f"INSTRUCT_TASK: Successfully created task with dataset {dataset.dataset_id}, "
        f"augmented={augmentation_config is not None}, use_kl={use_kl}"
    )
    task = await add_task(task, config.psql_db)
    logger.info(f"INSTRUCT_TASK: Task saved to database with ID: {task.task_id}")

    return task
