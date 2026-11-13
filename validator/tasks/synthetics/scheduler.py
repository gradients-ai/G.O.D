import asyncio
import math
import random
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import AsyncGenerator
from uuid import UUID

from bittensor_wallet.keypair import Keypair

import validator.infrastructure.service_constants as service_cst
import validator.tasks.datasets.constants as data_cst
import validator.tasks.prep.constants as prep_cst
import validator.tasks.synthetics.constants as synth_cst
from core.constants.environments import TrainingStartPoint
from core.logging import get_logger
from core.logging import redact_url_queries
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.image_models import ImageModelType
from core.models.model_prep_models import AugmentationConfig
from core.models.model_prep_models import EnvBaselineStats
from core.models.payload_models import ImageModelInfo
from core.models.payload_models import ImageModelsResponse
from core.models.payload_models import InstructTextDatasetColumnsResponse
from core.models.quant_models import LAUNCH_QUANT_FORMATS
from core.models.quant_models import QuantFixedSpec
from core.models.quant_models import QuantFormat
from core.models.quant_models import QuantHoursBreakdown
from core.models.quant_models import QuantWorkload
from core.models.reward_models import RewardFunction
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.app.config import Config
from validator.db.database import PSQLDB
from validator.db.sql import grpo as grpo_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql.continuous_sft import get_continuous_sft_state
from validator.db.sql.continuous_sft import record_continuous_sft_task_chunk
from validator.db.sql.tasks import add_task
from validator.db.sql.tasks import get_dataset_near_duplicate_rates
from validator.db.sql.tasks import get_dataset_test_losses
from validator.infrastructure.content_service import call_content_service
from validator.scoring.constants import QUANT_BAR_CAP
from validator.scoring.constants import QUANT_PENALTY_CURVATURE
from validator.scoring.constants import QUANT_PENALTY_KNEE
from validator.tasks.datasets.models import Dataset
from validator.tasks.details import retry_with_backoff
from validator.tasks.models import ChatRawTask
from validator.tasks.models import ContinuousChunk
from validator.tasks.models import DpoRawTask
from validator.tasks.models import GrpoRawTask
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tasks.models import QuantPreppedSplit
from validator.tasks.models import RawTask
from validator.tasks.prep.augmentation import maybe_get_augmentation_config
from validator.tasks.prep.augmentation import skip_augmentation_for_params
from validator.tasks.requests import get_model_num_params
from validator.tasks.rewards.templates import sample_template_groups
from validator.tournament import constants as t_cst
from validator.tournament.gpu_requirements import get_tournament_gpu_requirement
from validator.tournament.models import ContinuousSftTaskChunk


logger = get_logger(__name__)

SUPPORTED_TOURNAMENT_IMAGE_MODEL_TYPES = {
    ImageModelType.Z_IMAGE,
    ImageModelType.QWEN_IMAGE,
    ImageModelType.KREA2,
    ImageModelType.FLUX2_KLEIN_4B,
    ImageModelType.QWEN_IMAGE_EDIT_2511,
    ImageModelType.HIDREAM_O1,
    ImageModelType.MOSS_IMAGE_BASE,
}

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

        models = [
            model_info
            for model_info in response.models
            if model_info.model_type in SUPPORTED_TOURNAMENT_IMAGE_MODEL_TYPES
        ]
        if not models:
            logger.error(f"No supported image models returned from {service_cst.GET_IMAGE_MODELS_ENDPOINT}: {response_data}")
            await asyncio.sleep(5)
            continue
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


async def _get_instruct_text_datasets(keypair: Keypair) -> AsyncGenerator[Dataset, None]:
    """Round-robin generator that cycles through all dataset size bins."""

    bins = data_cst.DATASET_BINS_TO_SAMPLE
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


def get_training_overhead_hours(num_params: float) -> float:
    """Fixed per-task overhead on top of the token budget, scaled by model-size band."""
    params_b = num_params / 1e9
    return next(hours for bound_b, hours in data_cst.TRAINING_OVERHEAD_HOURS_BY_PARAMS_B if params_b < bound_b)


def compute_training_hours(
    tokens_per_epoch: float,
    num_params: float,
    task_type: TaskType,
    measured_tokens_per_sec: float | None = None,
    training_start_point: TrainingStartPoint = TrainingStartPoint.DEFAULT,
    max_hours: float | None = None,
    gpu_count_multiplier: int = 1,
) -> float:
    """Hours for TARGET_TRAINING_EPOCHS over the dataset at expected miner throughput.

    training_start_point matters for the GPU count: continuous-SFT trains on a forced 4xH100
    regardless of params, so sizing from params alone would budget for half the real GPUs.
    max_hours overrides the global MAX_TRAINING_HOURS ceiling when set. gpu_count_multiplier is the
    task's multiple of its size tier (the continuous instruct task trains on 2x: tournament constants
    gpu_count_multiplier), so the budget divides by the GPUs the task really gets.
    """
    gpus = get_tournament_gpu_requirement(
        task_type, int(num_params), training_start_point=training_start_point, gpu_count_multiplier=gpu_count_multiplier
    ).gpu_count
    analytic_tps = _analytic_tokens_per_sec_per_gpu(num_params)
    if measured_tokens_per_sec:
        lo, hi = data_cst.MEASURED_THROUGHPUT_CLAMP
        per_gpu_tps = measured_tokens_per_sec * data_cst.MEASURED_THROUGHPUT_MINER_RATIO
        per_gpu_tps = min(max(per_gpu_tps, analytic_tps * lo), analytic_tps * hi)
    else:
        per_gpu_tps = analytic_tps

    type_mult = data_cst.TASK_TYPE_HOURS_MULTIPLIER.get(task_type, 1.0)
    train_seconds = data_cst.TARGET_TRAINING_EPOCHS * tokens_per_epoch * type_mult / (per_gpu_tps * gpus)
    hours = train_seconds / 3600 + get_training_overhead_hours(num_params)

    hours = max(data_cst.TRAINING_HOURS_MIN, math.ceil(hours * 4) / 4)
    ceiling = data_cst.MAX_TRAINING_HOURS if max_hours is None else max_hours
    return min(hours, ceiling)


def get_grpo_training_hours(num_params: float | None) -> float:
    """GRPO budget: fixed per model-size band, independent of dataset size."""
    params_b = (num_params or data_cst.DEFAULT_MODEL_PARAMS_FOR_HOURS) / 1e9
    hours = next(hours for bound_b, hours in data_cst.GRPO_HOURS_BY_PARAMS_B if params_b <= bound_b)
    return min(data_cst.MAX_TRAINING_HOURS, max(data_cst.TRAINING_HOURS_MIN, hours))


def _get_training_hours_from_num_rows(
    num_rows: int, model_id: str | None = None, task_type: TaskType | None = None, gpu_count_multiplier: int = 1
) -> float:
    """Pre-prep estimate using ASSUMED_TOKENS_PER_ROW until measured stats arrive."""
    num_params = get_model_num_params(model_id) if model_id is not None else None
    if not num_params:
        num_params = data_cst.DEFAULT_MODEL_PARAMS_FOR_HOURS
    if task_type == TaskType.GRPOTASK:
        return get_grpo_training_hours(num_params)
    tokens_per_epoch = num_rows * data_cst.ASSUMED_TOKENS_PER_ROW
    return compute_training_hours(
        tokens_per_epoch, num_params, task_type or TaskType.INSTRUCTTEXTTASK, gpu_count_multiplier=gpu_count_multiplier
    )


def compute_hours_from_baseline_stats(
    current_hours: float,
    baseline_stats,
    task_type: TaskType,
    model_id: str | None = None,
    model_params_count: int | None = None,
    training_start_point: TrainingStartPoint = TrainingStartPoint.DEFAULT,
    ds: str | None = None,
    max_hours: float | None = None,
) -> float:
    """Post-prep hours from real token counts plus measured fwd/bwd throughput.

    A continuous-SFT task (CONTINUOUS_SFT start point, or a ds that resolves to a lineage) gets exactly
    CONTINUOUS_SFT_TRAINING_HOURS whatever the stats, current_hours or max_hours are. Other SFT tasks
    are sized from the measured tokens and throughput.
    """
    if training_start_point == TrainingStartPoint.CONTINUOUS_SFT or t_cst.continuous_sft_lineage_from_ds(ds):
        return t_cst.CONTINUOUS_SFT_TRAINING_HOURS
    if task_type in QUANT_TASK_TYPES:
        return current_hours
    if isinstance(baseline_stats, EnvBaselineStats) or baseline_stats is None:
        return current_hours if max_hours is None else min(current_hours, max_hours)

    num_params = model_params_count or (get_model_num_params(model_id) if model_id else None)
    if not num_params:
        num_params = data_cst.DEFAULT_MODEL_PARAMS_FOR_HOURS

    if task_type == TaskType.GRPOTASK:
        hours = get_grpo_training_hours(num_params)
        return hours if max_hours is None else min(hours, max_hours)

    dataset_stats = baseline_stats.dataset
    if not dataset_stats.total_tokens or not dataset_stats.num_records:
        return current_hours if max_hours is None else min(current_hours, max_hours)

    effective_tokens = max(
        dataset_stats.total_tokens,
        dataset_stats.num_records * data_cst.EFFECTIVE_MIN_TOKENS_PER_ROW,
    )
    measured_tps = baseline_stats.throughput.tokens_per_sec if baseline_stats.throughput else None
    return compute_training_hours(
        effective_tokens,
        num_params,
        task_type,
        measured_tps,
        training_start_point,
        max_hours=max_hours,
        gpu_count_multiplier=t_cst.gpu_count_multiplier(task_type, ds),
    )


def quant_fixed_hours(params_b: float) -> float:
    """Fixed-track budget: flat below the first anchor, linear to the second, flat above it."""
    (low_b, low_hours), (high_b, high_hours) = data_cst.QUANT_FIXED_HOURS_BY_PARAMS_B
    fraction = min(max((params_b - low_b) / (high_b - low_b), 0.0), 1.0)
    return low_hours + fraction * (high_hours - low_hours)


def quant_flat_hours(task_type: TaskType, model_params_count: int, boss: bool) -> float:
    """Flat quant budget: freeform a constant, fixed by model size; boss gets no extra. The no-stats fallback."""
    if task_type == TaskType.QUANTFREEFORMTASK:
        return data_cst.QUANT_FREEFORM_EARLY_HOURS
    if task_type == TaskType.QUANTFIXEDTASK:
        return quant_fixed_hours((model_params_count or data_cst.DEFAULT_MODEL_PARAMS_FOR_HOURS) / 1e9)
    raise ValueError(f"{task_type} is not a quant task type")


def quant_workload(task_type: TaskType) -> QuantWorkload:
    """The priced workload of the track."""
    if task_type == TaskType.QUANTFREEFORMTASK:
        return data_cst.QUANT_FREEFORM_WORKLOAD
    if task_type == TaskType.QUANTFIXEDTASK:
        return data_cst.QUANT_FIXED_WORKLOAD
    raise ValueError(f"{task_type} is not a quant task type")


def _has_usable_quant_stats(baseline_stats) -> bool:
    """Instruct/dpo/grpo stats with a positive measured throughput and a positive mean sequence length."""
    if baseline_stats is None or isinstance(baseline_stats, EnvBaselineStats):
        return False
    throughput = getattr(baseline_stats, "throughput", None)
    if throughput is None or throughput.tokens_per_sec <= 0:
        return False
    return baseline_stats.dataset.seq_length_distribution.mean > 0


def quant_measured_tokens_per_sec(baseline_stats) -> float:
    """The prep's per-GPU fwd+bwd rate: the hardened probe's median-of-windows reading, else the 3-step reading."""
    probes = getattr(baseline_stats, "device", None)
    hardened = getattr(probes, "train_tokens_per_sec", None)
    if hardened and hardened > 0:
        return hardened
    return baseline_stats.throughput.tokens_per_sec


def _quant_per_gpu_tokens_per_sec(measured_tokens_per_sec: float, num_params: float) -> float:
    """Measured per-GPU probe throughput, clamped to the analytic band like compute_training_hours."""
    analytic_tps = _analytic_tokens_per_sec_per_gpu(num_params)
    lo, hi = data_cst.MEASURED_THROUGHPUT_CLAMP
    return min(max(measured_tokens_per_sec, analytic_tps * lo), analytic_tps * hi)


def _quant_train_tokens(workload: QuantWorkload, dataset_stats) -> int:
    """Training tokens priced: the workload's budget, lowered to the epoch cap of the train split when one applies."""
    if workload.train_max_epochs is None or not dataset_stats.total_tokens or not dataset_stats.num_records:
        return workload.train_tokens
    split_tokens = max(dataset_stats.total_tokens, dataset_stats.num_records * data_cst.EFFECTIVE_MIN_TOKENS_PER_ROW)
    return min(workload.train_tokens, int(workload.train_max_epochs * split_tokens))


def _quant_token_work(workload: QuantWorkload, mean_seq: float, train_tokens: int) -> float:
    """Probe-token equivalents of the evaluation passes, the further forward-only tokens and the training tokens."""
    evaluation = workload.eval_passes * data_cst.QUANT_EVAL_ROWS_PER_PASS * mean_seq * data_cst.QUANT_FORWARD_TOKEN_FACTOR
    forward = workload.forward_tokens * data_cst.QUANT_FORWARD_TOKEN_FACTOR
    training = train_tokens * data_cst.QUANT_TRAIN_TOKEN_FACTOR
    return evaluation + training + forward


def quant_hours_breakdown(
    task_type: TaskType, num_params: float, baseline_stats, boss: bool
) -> QuantHoursBreakdown | None:
    """Token, model-size and host seconds plus the clamped quarter-hour budget; None when the stats are unusable."""
    if not _has_usable_quant_stats(baseline_stats):
        return None
    workload = quant_workload(task_type)
    num_params = num_params or data_cst.DEFAULT_MODEL_PARAMS_FOR_HOURS
    params_b = num_params / 1e9
    gpus = get_tournament_gpu_requirement(task_type, int(num_params)).gpu_count
    tps_per_gpu = _quant_per_gpu_tokens_per_sec(quant_measured_tokens_per_sec(baseline_stats), num_params)
    train_tokens = _quant_train_tokens(workload, baseline_stats.dataset)
    token_work = _quant_token_work(workload, baseline_stats.dataset.seq_length_distribution.mean, train_tokens)
    token_seconds = token_work / (tps_per_gpu * gpus)
    model_seconds = workload.model_seconds_per_b_params * params_b / gpus
    host_seconds = workload.host_seconds_per_b_params * params_b + workload.host_fixed_seconds
    boss_multiplier = data_cst.QUANT_BOSS_HOURS_MULTIPLIER[task_type] if boss else 1.0
    plan_seconds = (token_seconds + model_seconds + host_seconds) * workload.hours_slack
    raw_hours = plan_seconds / 3600 * boss_multiplier
    hours = min(data_cst.MAX_TRAINING_HOURS, max(data_cst.QUANT_TRAINING_HOURS_MIN, math.ceil(raw_hours * 4) / 4))
    breakdown = QuantHoursBreakdown(
        token_seconds=token_seconds,
        model_seconds=model_seconds,
        host_seconds=host_seconds,
        gpus=gpus,
        train_tokens=train_tokens,
        slack=workload.hours_slack,
        hours=hours,
    )
    logger.info(
        f"QUANT_HOURS: {task_type.value} params={params_b:.2f}B boss={boss} tps/gpu={tps_per_gpu:.0f} "
        f"token={token_seconds:.0f}s model={model_seconds:.0f}s host={host_seconds:.0f}s "
        f"gpus={gpus} train_tokens={train_tokens} slack={workload.hours_slack} -> {hours}h"
    )
    return breakdown


def quant_training_hours(task_type: TaskType, model_params_count: int, boss: bool, baseline_stats=None) -> float:
    """Stats-based quant budget when the stats are usable, else the flat role budget."""
    num_params = model_params_count or data_cst.DEFAULT_MODEL_PARAMS_FOR_HOURS
    breakdown = quant_hours_breakdown(task_type, num_params, baseline_stats, boss)
    if breakdown is None:
        return quant_flat_hours(task_type, model_params_count, boss)
    return breakdown.hours


def quant_hours_after_prep(task, baseline_stats, boss: bool) -> float:
    """Post-prep hours of a quant task from its own stats; the task's current hours when the stats are unusable."""
    params = task.model_params_count or get_model_num_params(task.model_id) or data_cst.DEFAULT_MODEL_PARAMS_FOR_HOURS
    if not _has_usable_quant_stats(baseline_stats):
        return task.hours_to_complete
    return quant_training_hours(task.task_type, int(params), boss, baseline_stats)


def _quant_creation_hours(task_type: TaskType, model_id: str, boss: bool) -> float:
    """Hours at creation: the flat role budget for the model's size (prep resizes from the task's own stats)."""
    params = get_model_num_params(model_id) or data_cst.DEFAULT_MODEL_PARAMS_FOR_HOURS
    return quant_flat_hours(task_type, int(params), boss)


def apply_baseline_ctx_scale(hours: float, baseline_stats) -> float:
    """Backward-compatible wrapper for callers not yet passing task metadata."""
    return hours if baseline_stats is None else hours


async def _is_dataset_degenerate(ds_name: str, task_type: TaskType, psql_db: PSQLDB) -> bool:
    """Check if a dataset has recently produced degenerate test_loss scores.

    Returns True (degenerate) if:
    - Any recent test_loss below the collapse floor (0.01, or 0.05 for DPO)
    - For instruct tasks: best (min) test_loss > 2.0 (garbage / unlearnable data)
    - For DPO tasks: best (min) test_loss in [0.68, 0.71] (nobody beat random noise around ln(2))

    Both task-type checks key off the *best* loss, not the mean: the mean is dominated by weak
    submissions across every miner that ever touched the dataset and effectively never lands in
    a rejection band.
    """
    try:
        losses = await get_dataset_test_losses(ds_name, psql_db, synth_cst.DEGENERATE_LOSS_LOOKBACK_DAYS, task_type)
    except Exception as e:
        logger.warning(f"Failed to query historical losses for {ds_name}, allowing dataset: {e}")
        return False

    if not losses:
        return False

    best_loss = min(losses)
    collapse_floor = (
        synth_cst.DEGENERATE_DPO_COLLAPSE_LOSS if task_type == TaskType.DPOTASK else synth_cst.DEGENERATE_COLLAPSE_LOSS
    )
    if best_loss < collapse_floor:
        logger.warning(f"Dataset {ds_name} rejected: test_loss {best_loss:.4f} < {collapse_floor} (model collapse)")
        return True

    if task_type == TaskType.INSTRUCTTEXTTASK and best_loss > synth_cst.DEGENERATE_INSTRUCT_MAX_BEST_LOSS:
        logger.warning(
            f"Dataset {ds_name} rejected: best instruct test_loss {best_loss:.4f} > "
            f"{synth_cst.DEGENERATE_INSTRUCT_MAX_BEST_LOSS} (garbage data)"
        )
        return True

    if task_type == TaskType.DPOTASK:
        noise_low, noise_high = synth_cst.DEGENERATE_DPO_NOISE_BAND
        if noise_low <= best_loss <= noise_high:
            logger.warning(
                f"Dataset {ds_name} rejected: best DPO test_loss {best_loss:.4f} in noise range "
                f"[{noise_low}, {noise_high}]"
            )
            return True

    return False


async def _has_known_high_near_duplicate_rate(ds_name: str, task_type: TaskType, psql_db: PSQLDB) -> bool:
    """Skip datasets whose persisted near_duplicate_rate already exceeded the gate, before paying for prep."""
    try:
        rates = await get_dataset_near_duplicate_rates(
            ds_name, psql_db, synth_cst.NEAR_DUP_HISTORY_LOOKBACK_DAYS, task_type
        )
    except Exception as e:
        logger.warning(f"Failed to query near-duplicate history for {ds_name}, allowing dataset: {e}")
        return False

    if not rates:
        return False

    worst = max(rates)
    if worst >= t_cst.MAX_NEAR_DUPLICATE_RATE:
        logger.warning(
            f"Dataset {ds_name} rejected: near_duplicate_rate {worst:.3f} >= {t_cst.MAX_NEAR_DUPLICATE_RATE} "
            f"in {len(rates)} prior model-prep(s), skipping before prep"
        )
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
            if await _has_known_high_near_duplicate_rate(dataset.dataset_id, task_type, psql_db):
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


def _augmentation_config_for_model(
    task_type: TaskType, model_id: str, allow_augmentation: bool = True
) -> AugmentationConfig | None:
    """Roll augmentation unless the caller disabled it or the base is too large to republish."""
    if not allow_augmentation:
        return None
    num_params = get_model_num_params(model_id)
    if skip_augmentation_for_params(num_params):
        size_b = (num_params or 0) / 1_000_000_000
        logger.info(
            f"Skipping augmentation for {model_id} ({size_b:.1f}B >= {prep_cst.AUGMENTATION_SKIP_MIN_SIZE_B}B)"
        )
        return None
    return maybe_get_augmentation_config(task_type)


@retry_with_backoff
async def create_synthetic_dpo_task(
    config: Config,
    models: AsyncGenerator[str, None] | None,
    datasets: AsyncGenerator[Dataset, None],
    model_id_override: str | None = None,
) -> RawTask:
    """models may be None only with model_id_override set (the pool is never drawn from then)."""
    logger.info("DPO task")
    if model_id_override:
        model_id = model_id_override
    else:
        assert models is not None, "a models pool is required when model_id_override is not set"
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
    augmentation_config = _augmentation_config_for_model(TaskType.DPOTASK, model_id)
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
    models: AsyncGenerator[str, None] | None,
    datasets: AsyncGenerator[Dataset, None],
    model_id_override: str | None = None,
) -> RawTask:
    """models may be None only with model_id_override set (the pool is never drawn from then)."""
    if model_id_override:
        model_id = model_id_override
    else:
        assert models is not None, "a models pool is required when model_id_override is not set"
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
    augmentation_config = _augmentation_config_for_model(TaskType.GRPOTASK, model_id)
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
        augmentation_config = _augmentation_config_for_model(TaskType.GRPOTASK, model_id)
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
    models: AsyncGenerator[str, None] | None,
    datasets: AsyncGenerator[Dataset, None],
    enable_kl: bool = False,
    model_id_override: str | None = None,
    allow_augmentation: bool = True,
    allow_yarn: bool = True,
    carve_calibration: bool = False,
) -> RawTask:
    """models may be None only with model_id_override set (the pool is never drawn from then).

    carve_calibration marks a tournament round-1 task: its prep carves a calibration set from its train side.
    """
    if model_id_override:
        model_id = model_id_override
    else:
        assert models is not None, "a models pool is required when model_id_override is not set"
        model_id = await anext(models)

    logger.info("INSTRUCT_TASK: Starting dataset selection...")
    dataset = await get_dataset(datasets, task_type=TaskType.INSTRUCTTEXTTASK, keypair=config.keypair, psql_db=config.psql_db)
    logger.info(f"INSTRUCT_TASK: Selected dataset: {dataset.dataset_id}")

    number_of_hours = _get_training_hours_from_num_rows(dataset.num_rows, model_id, task_type=TaskType.INSTRUCTTEXTTASK)
    columns = await _get_columns_for_instruct_dataset(dataset.dataset_id, config.keypair)

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    yarn_factor = maybe_get_yarn_factor() if allow_yarn else None
    augmentation_config = _augmentation_config_for_model(
        TaskType.INSTRUCTTEXTTASK, model_id, allow_augmentation=allow_augmentation
    )
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
        carve_calibration=carve_calibration,
    )
    logger.info(
        f"INSTRUCT_TASK: Successfully created task with dataset {dataset.dataset_id}, "
        f"augmented={augmentation_config is not None}, use_kl={use_kl}"
    )
    task = await add_task(task, config.psql_db)
    logger.info(f"INSTRUCT_TASK: Task saved to database with ID: {task.task_id}")

    return task


def draw_quant_format(rng: random.Random) -> QuantFormat:
    """Fixed-track format for one task. Callers seed rng from the round so a resumed round redraws the same one."""
    return rng.choice(LAUNCH_QUANT_FORMATS)


async def create_synthetic_quant_task(
    config: Config,
    task_type: TaskType,
    models: AsyncGenerator[str, None] | None,
    datasets: AsyncGenerator[Dataset, None] | None,
    *,
    quant_format: QuantFormat | None = None,
    model_id_override: str | None = None,
    ds_override: tuple[str, InstructTextDatasetType, int] | None = None,
    prepped_split: QuantPreppedSplit | None = None,
    rng: random.Random | None = None,
    boss: bool = False,
    training_start_point: TrainingStartPoint = TrainingStartPoint.DEFAULT,
) -> RawTask:
    """Create a fixed- or freeform-track quant task on instruct data.

    models may be None only with model_id_override set, datasets only with ds_override
    (dataset id, columns, num_rows). With prepped_split (only alongside ds_override, whose columns
    must then be the split's standard ones) prep reuses those files instead of splitting ds.
    A fixed task plays quant_format, else a format drawn from rng.
    The format is drawn once, before the retried body, so a retry never redraws from an advanced rng.
    boss selects the boss-round budget (no multiplier: QUANT_BOSS_HOURS_MULTIPLIER is 1.0 for both types).
    training_start_point OWN_R1_MODEL marks a task whose miners each start from their own round-1 model; R1-R3 do
    not use it (they are chained on a seed, see create_chained_quant_task).
    """
    if task_type == TaskType.QUANTFIXEDTASK:
        quant_format = quant_format or draw_quant_format(rng or random.Random())
    return await _create_synthetic_quant_task_pinned(
        config,
        task_type,
        models,
        datasets,
        quant_format=quant_format,
        model_id_override=model_id_override,
        ds_override=ds_override,
        prepped_split=prepped_split,
        boss=boss,
        training_start_point=training_start_point,
    )


@retry_with_backoff
async def _create_synthetic_quant_task_pinned(
    config: Config,
    task_type: TaskType,
    models: AsyncGenerator[str, None] | None,
    datasets: AsyncGenerator[Dataset, None] | None,
    *,
    quant_format: QuantFormat | None,
    model_id_override: str | None,
    ds_override: tuple[str, InstructTextDatasetType, int] | None,
    prepped_split: QuantPreppedSplit | None = None,
    boss: bool = False,
    training_start_point: TrainingStartPoint = TrainingStartPoint.DEFAULT,
) -> RawTask:
    """create_synthetic_quant_task's body with the fixed format already drawn (None for freeform).

    No YaRN (a rope change alters config.json, which the fixed-track structural check compares) and
    no KL.
    """
    if model_id_override:
        model_id = model_id_override
    else:
        assert models is not None, "a models pool is required when model_id_override is not set"
        model_id = await anext(models)

    if ds_override:
        ds, dataset_type, num_rows = ds_override
    else:
        assert datasets is not None, "a datasets pool is required when ds_override is not set"
        ds, dataset_type, num_rows = await _draw_quant_instruct_dataset(config, datasets)

    number_of_hours = _quant_creation_hours(task_type, model_id, boss)
    quant_spec = quant_format.spec() if task_type == TaskType.QUANTFIXEDTASK else None
    task = _new_quant_task(
        task_type,
        model_id=model_id,
        ds=ds,
        dataset_type=dataset_type,
        hours_to_complete=number_of_hours,
        quant_spec=quant_spec,
        training_start_point=training_start_point,
        **_prepped_split_fields(prepped_split),
    )
    logger.info(
        f"QUANT_TASK: {task_type.value} on {model_id} with dataset {ds}, spec={quant_spec.name if quant_spec else None}, "
        f"prepped_split={prepped_split is not None}"
    )
    task = await add_task(task, config.psql_db)
    logger.info(f"QUANT_TASK: Task saved to database with ID: {task.task_id}")
    return task


async def _draw_quant_instruct_dataset(
    config: Config, datasets: AsyncGenerator[Dataset, None]
) -> tuple[str, InstructTextDatasetType, int]:
    """A fresh instruct dataset for a quant task: (dataset id, columns, num_rows)."""
    # Quant tasks train on instruct data, so the instruct dataset-history filters apply.
    dataset = await get_dataset(datasets, task_type=TaskType.INSTRUCTTEXTTASK, keypair=config.keypair, psql_db=config.psql_db)
    columns = await _get_columns_for_instruct_dataset(dataset.dataset_id, config.keypair)
    dataset_type = InstructTextDatasetType(
        field_instruction=columns.field_instruction,
        field_input=columns.field_input,
        field_output=columns.field_output,
        field_system=None,
    )
    return dataset.dataset_id, dataset_type, dataset.num_rows


@retry_with_backoff
async def create_chained_quant_task(
    config: Config,
    task_type: TaskType,
    upstream: InstructTextRawTask,
    *,
    quant_format: QuantFormat | None,
    boss: bool = True,
) -> RawTask:
    """Create a chained task held in AWAITING_UPSTREAM on an instruct task (a boss-round instruct or an input seed).

    Both players compress the same upstream output, chosen at release, so model_id is the upstream's
    base until the release sets it to that output. No augmentation: it would change the trained model.
    The instruct slot carries the fine-tuning, so hours are the flat boss budget for the type; boss=False (the miner
    rounds R1-R3 on an input seed) takes the non-boss budget instead, which the release recomputes from the seed.

    The data is the upstream's (raw columns here; the release re-copies the prepped upstream's columns
    and URLs). The fixed format is drawn by the caller.
    """
    ds = upstream.ds
    dataset_type = _upstream_dataset_type(upstream)
    quant_spec = quant_format.spec() if task_type == TaskType.QUANTFIXEDTASK else None
    task = _new_quant_task(
        task_type,
        model_id=upstream.model_id,
        ds=ds,
        dataset_type=dataset_type,
        hours_to_complete=_chained_hours(upstream, task_type, boss=boss),
        quant_spec=quant_spec,
        upstream_task_id=upstream.task_id,
        status=TaskStatus.AWAITING_UPSTREAM,
        model_params_count=upstream.model_params_count,
    )
    logger.info(
        f"QUANT_TASK: chained {task_type.value} on upstream {upstream.task_id} ({upstream.model_id}) with dataset {ds}, "
        f"spec={quant_spec.name if quant_spec else None}"
    )
    task = await add_task(task, config.psql_db)
    logger.info(f"QUANT_TASK: Task saved to database with ID: {task.task_id}")
    return task


@retry_with_backoff
async def create_own_upstream_quant_task(
    config: Config,
    task_type: TaskType,
    upstream: InstructTextRawTask | ChatRawTask,
    *,
    quant_format: QuantFormat | None,
    train_index: int | None = None,
    tournament_id: str | None = None,
) -> RawTask:
    """Create a boss-round task whose players each compress their OWN output on the upstream instruct task.

    The start point is OWN_UPSTREAM_MODEL: there is no shared input, so the release sets each player's
    starting_model_repo to that player's own upstream output rather than model_id. Held in AWAITING_UPSTREAM.

    train_index None: the data is the upstream's (create_chained_quant_task's columns), copied from it at
    release, and the hours follow the upstream's stats. With train_index and tournament_id: the task carries
    the SAME continuous chunk as the upstream from creation, with model_id the upstream's base; at prep it takes the
    second, disjoint share of the chunk's train rows (select_continuous_rows), the release leaves that data alone,
    calibration is carved from those rows, and hours are the normal quant budget (boss multiplier 1.0, no instruct term)
    from the task's own prep stats once they exist.
    """
    quant_spec = quant_format.spec() if task_type == TaskType.QUANTFIXEDTASK else None
    if train_index is None:
        data_fields = dict(
            ds=upstream.ds,
            dataset_type=_upstream_dataset_type(upstream),
            hours_to_complete=_chained_hours(upstream, task_type),
        )
    else:
        chunk = await _fetch_continuous_quant_chunk(config, train_index)
        data_fields = dict(
            ds=t_cst.continuous_quant_ds(train_index, tournament_id, chunk.label),
            dataset_type=_continuous_quant_dataset_type(),
            hours_to_complete=_quant_creation_hours(task_type, upstream.model_id, boss=True),
            file_format=FileFormat.S3,
            training_data=chunk.train_url,
            test_data=chunk.test_url,
        )
    task = _new_quant_task(
        task_type,
        model_id=upstream.model_id,
        quant_spec=quant_spec,
        upstream_task_id=upstream.task_id,
        status=TaskStatus.AWAITING_UPSTREAM,
        model_params_count=upstream.model_params_count,
        training_start_point=TrainingStartPoint.OWN_UPSTREAM_MODEL,
        **data_fields,
    )
    logger.info(
        f"QUANT_TASK: own-upstream {task_type.value} on upstream {upstream.task_id} ({upstream.model_id}) with dataset "
        f"{task.ds}, spec={quant_spec.name if quant_spec else None}"
    )
    task = await add_task(task, config.psql_db)
    logger.info(f"QUANT_TASK: Task saved to database with ID: {task.task_id}")
    return task


@retry_with_backoff
async def create_chained_continuous_quant_task(
    config: Config,
    task_type: TaskType,
    upstream: ChatRawTask,
    *,
    quant_format: QuantFormat | None,
    train_index: int,
    tournament_id: str,
) -> RawTask:
    """Create a held quant task that compresses the model the text boss trains on a continuous-SFT task.

    The task carries the SAME continuous chunk as the upstream from creation (train_index is the one recorded for the
    upstream task, never read from a name), so its data is not copied at release; model_id is the upstream's base until
    the release sets it to the boss's output. Start point DEFAULT: it is released as a SHARED input, both players compress
    the text boss's model. Calibration is carved from the chunk's train rows. Hours are the flat boss-round quant budget for
    the base's size (the upstream's persisted size when prep has written one; its base may be an adapter repo, which
    cannot be sized from the hub); prep resizes them from the task's own stats.
    """
    quant_spec = quant_format.spec() if task_type == TaskType.QUANTFIXEDTASK else None
    chunk = await _fetch_continuous_quant_chunk(config, train_index)
    task = _new_quant_task(
        task_type,
        model_id=upstream.model_id,
        ds=t_cst.continuous_quant_ds(train_index, tournament_id, chunk.label),
        dataset_type=_continuous_quant_dataset_type(),
        hours_to_complete=quant_flat_hours(task_type, upstream.model_params_count, boss=True),
        quant_spec=quant_spec,
        file_format=FileFormat.S3,
        training_data=chunk.train_url,
        test_data=chunk.test_url,
        upstream_task_id=upstream.task_id,
        status=TaskStatus.AWAITING_UPSTREAM,
        model_params_count=upstream.model_params_count,
        training_start_point=TrainingStartPoint.DEFAULT,
    )
    logger.info(
        f"QUANT_TASK: chained continuous {task_type.value} on upstream {upstream.task_id} ({upstream.model_id}) with dataset "
        f"{task.ds}, spec={quant_spec.name if quant_spec else None}"
    )
    task = await add_task(task, config.psql_db)
    logger.info(f"QUANT_TASK: Task saved to database with ID: {task.task_id}")
    return task


def repoint_chained_quant_task(
    task: QuantFixedRawTask | QuantFreeformRawTask,
    upstream: InstructTextRawTask | ChatRawTask,
    *,
    pinned_data: bool,
    upstream_train_index: int | None = None,
) -> None:
    """Move a held chained task onto a replacement upstream, in place.

    A replacement redraws the base and the data, so the base, its size and the hours follow the new
    upstream; a task pinned to the old upstream's data moves to the new data (a task with data of its own
    keeps it).

    A task on its own continuous chunk (a compression of a continuous-SFT task) keeps its data: only the upstream, the base
    and the size move. Its chunk must be the replacement's, so upstream_train_index (the index recorded for the new upstream
    task) has to equal the task's own; a different chunk raises, because moving the task would compress a model trained on
    other data than the one the task carries (a replaced continuous task is the same chunk of the same lineage in the same week).
    """
    if t_cst.is_continuous_quant_task(task):
        own_index = t_cst.continuous_quant_train_index(task.ds)
        if upstream_train_index != own_index:
            raise ValueError(
                f"Task {task.task_id} carries continuous chunk {own_index} but the new upstream {upstream.task_id} trained on "
                f"chunk {upstream_train_index}; not re-pointing it"
            )
        task.upstream_task_id = upstream.task_id
        task.model_id = upstream.model_id
        task.model_params_count = upstream.model_params_count
        task.hours_to_complete = quant_flat_hours(task.task_type, upstream.model_params_count, boss=True)
        return
    task.upstream_task_id = upstream.task_id
    task.model_id = upstream.model_id
    task.model_params_count = upstream.model_params_count
    task.hours_to_complete = _chained_hours(upstream, task.task_type)
    if pinned_data:
        task.ds = upstream.ds
        task.dataset_type = _upstream_dataset_type(upstream)


def _upstream_dataset_type(upstream: InstructTextRawTask) -> InstructTextDatasetType:
    return InstructTextDatasetType(
        field_instruction=upstream.field_instruction,
        field_input=upstream.field_input,
        field_output=upstream.field_output,
        field_system=upstream.field_system,
        format=upstream.format,
        no_input_format=upstream.no_input_format,
    )


def _chained_hours(upstream: InstructTextRawTask, task_type: TaskType, *, boss: bool = True) -> float:
    return quant_training_hours(task_type, upstream.model_params_count, boss=boss, baseline_stats=upstream.baseline_stats)


async def recreate_chained_quant_task(config: Config, task: QuantFixedRawTask | QuantFreeformRawTask) -> RawTask:
    """A fresh held copy of a chained task: same upstream, data, columns, spec or bar cap, hours and start point (an
    own-upstream task stays OWN_UPSTREAM_MODEL, so it is released per player rather than as a shared input).

    A compression task of the continuous instruct task, on its own chunk, gets freshly signed urls for the same chunk: the failed
    task's own urls are never copied (after prep its test_data is the tokenised eval set).
    """
    chunk_fields: dict[str, Any] = {}
    train_index = t_cst.continuous_quant_train_index(task.ds)
    if train_index is not None:
        chunk = await _fetch_continuous_quant_chunk(config, train_index)
        chunk_fields = dict(
            file_format=FileFormat.S3,
            training_data=chunk.train_url,
            test_data=chunk.test_url,
        )
    fresh = _new_quant_task(
        task.task_type,
        model_id=task.model_id,
        ds=task.ds,
        dataset_type=task.dataset_type,
        hours_to_complete=task.hours_to_complete,
        quant_spec=getattr(task, "quant_spec", None),
        bar_cap=getattr(task, "bar_cap", QUANT_BAR_CAP),
        penalty_knee=getattr(task, "penalty_knee", None),
        penalty_curvature=getattr(task, "penalty_curvature", None),
        upstream_task_id=task.upstream_task_id,
        status=TaskStatus.AWAITING_UPSTREAM,
        model_params_count=task.model_params_count,
        training_start_point=task.training_start_point,
        **chunk_fields,
    )
    logger.info(f"QUANT_TASK: recreating chained {task.task_type.value} {task.task_id} on upstream {task.upstream_task_id}")
    return await add_task(fresh, config.psql_db)


def _continuous_quant_dataset_type() -> ChatTemplateDatasetType:
    """ShareGPT chat data under the base's own template (as the continuous-SFT task), the other fields ChatRawTask's."""
    chat_defaults = ChatRawTask.model_fields
    return ChatTemplateDatasetType(
        chat_template="tokenizer_default",
        chat_column=chat_defaults["chat_column"].default,
        chat_role_field=chat_defaults["chat_role_field"].default,
        chat_content_field=chat_defaults["chat_content_field"].default,
        chat_user_reference=chat_defaults["chat_user_reference"].default,
        chat_assistant_reference=chat_defaults["chat_assistant_reference"].default,
    )


async def _fetch_continuous_quant_chunk(config: Config, train_index: int) -> ContinuousChunk:
    """One continuous chunk (train url, test url, ds label), freshly signed by the content service.

    The urls are presigned, so neither the logs nor the errors raised here carry them.
    """
    response = await call_content_service(
        service_cst.GET_CONTINUOUS_SFT_DATA_ENDPOINT, config.keypair, {"train_index": train_index}
    )
    if not isinstance(response, dict):
        raise ValueError("Expected dict response from continuous-SFT data endpoint")
    train_url, test_url = response.get("train_s3_url"), response.get("test_s3_url")
    if not train_url or not test_url:
        raise ValueError(f"continuous-SFT data response for train_index={train_index} is missing a train or test url")
    logger.info(f"Retrieved continuous chunk data for train_index={train_index}")
    return ContinuousChunk(
        train_url=train_url, test_url=test_url, label=response.get("ds") or f"train-index-{train_index}"
    )


@retry_with_backoff
async def create_continuous_instruct_task(config: Config, *, model_id: str, train_index: int, tournament_id: str) -> RawTask:
    """The boss round's continuous instruct task: a raw quant-pool base trained on a continuous chat chunk.

    A ChatRawTask under the base's own chat template (ShareGPT, as the continuous-SFT task) with start point DEFAULT:
    CONTINUOUS_SFT would force 4xH100, while this task's GPU tier comes from its size doubled (gpu_count_multiplier). No KL, no
    augmentation, so the evaluator's tokenizer is the raw base's (original_model). The ds carries the chunk and the tournament
    that seeds the row split: prep takes the first QUANT_CONTINUOUS_INSTRUCT_TRAIN_ROWS rows of the chunk
    (select_continuous_rows), and the creation estimate prices that many rows.
    """
    chunk = await _fetch_continuous_quant_chunk(config, train_index)
    ds = t_cst.continuous_chat_ds(train_index, tournament_id, chunk.label)
    hours = _get_training_hours_from_num_rows(
        t_cst.QUANT_CONTINUOUS_INSTRUCT_TRAIN_ROWS,
        model_id,
        TaskType.CHATTASK,
        gpu_count_multiplier=t_cst.CONTINUOUS_INSTRUCT_GPU_MULTIPLIER,
    )
    current_time = datetime.utcnow()
    task = ChatRawTask(
        model_id=model_id,
        ds=ds,
        status=TaskStatus.PENDING,
        is_organic=False,
        created_at=current_time,
        termination_at=current_time + timedelta(hours=hours),
        hours_to_complete=hours,
        account_id=service_cst.NULL_ACCOUNT_ID,
        file_format=FileFormat.S3,
        training_data=chunk.train_url,
        test_data=chunk.test_url,
        training_start_point=TrainingStartPoint.DEFAULT,
        chat_template="tokenizer_default",
    )
    logger.info(f"CONTINUOUS_INSTRUCT_TASK: continuous chat chunk on {model_id} with dataset {ds}, hours={hours}")
    task = await add_task(task, config.psql_db)
    logger.info(f"CONTINUOUS_INSTRUCT_TASK: saved task {task.task_id}")
    return task


async def recreate_quant_task(
    config: Config,
    task: QuantFixedRawTask | QuantFreeformRawTask,
    *,
    dataset_type: InstructTextDatasetType | None = None,
    prepped_split: QuantPreppedSplit | None = None,
) -> RawTask:
    """A fresh copy of a quant task: same model, data, columns, spec or bar cap and hours.

    The failed task's own split URLs are never copied (a prepped task's test_data is the tokenised
    eval set, not the raw split); a caller that pins the data re-resolves it into dataset_type and
    prepped_split.
    """
    fresh = _new_quant_task(
        task.task_type,
        model_id=task.model_id,
        ds=task.ds,
        dataset_type=dataset_type or task.dataset_type,
        hours_to_complete=task.hours_to_complete,
        quant_spec=getattr(task, "quant_spec", None),
        bar_cap=getattr(task, "bar_cap", QUANT_BAR_CAP),
        penalty_knee=getattr(task, "penalty_knee", None),
        penalty_curvature=getattr(task, "penalty_curvature", None),
        model_params_count=task.model_params_count,
        training_start_point=task.training_start_point,
        **_prepped_split_fields(prepped_split),
    )
    recreating = f"QUANT_TASK: recreating {task.task_type.value} {task.task_id} on {task.model_id} with dataset {task.ds}"
    logger.info(redact_url_queries(recreating))
    return await add_task(fresh, config.psql_db)


def _prepped_split_fields(prepped_split: QuantPreppedSplit | None) -> dict[str, Any]:
    """_new_quant_task's data kwargs for a reused split: S3 files, so prep takes the given-split path."""
    if prepped_split is None:
        return {}
    return dict(
        file_format=FileFormat.S3,
        training_data=prepped_split.training_data,
        test_data=prepped_split.test_data,
        calibration_data=prepped_split.calibration_data,
    )


def _new_quant_task(
    task_type: TaskType,
    *,
    model_id: str,
    ds: str,
    dataset_type: InstructTextDatasetType | ChatTemplateDatasetType,
    hours_to_complete: float,
    quant_spec: QuantFixedSpec | None = None,
    bar_cap: float = QUANT_BAR_CAP,
    penalty_knee: float | None = QUANT_PENALTY_KNEE,
    penalty_curvature: float | None = QUANT_PENALTY_CURVATURE,
    file_format: FileFormat = FileFormat.HF,
    training_data: str | None = None,
    test_data: str | None = None,
    calibration_data: str | None = None,
    upstream_task_id: UUID | None = None,
    status: TaskStatus = TaskStatus.PENDING,
    model_params_count: int = 0,
    training_start_point: TrainingStartPoint = TrainingStartPoint.DEFAULT,
) -> QuantFixedRawTask | QuantFreeformRawTask:
    """A synthetic quant task (pending unless status says otherwise); quant_spec is required for the
    fixed track and ignored otherwise.

    Never augmented: quant tasks are text-tournament tasks outside round 1, which play their base unmodified.
    Both tracks take calibration_data (the raw rows carried from the source split) and are always flagged to carve:
    prep carves a calibration set from the task's own split, or from the given train file when test_data is given and the
    task is on its own continuous chunk, unless the task already carries calibration_data; any other given split without
    calibration_data fails prep. A task leaves prep only with its published calibration set.
    """
    current_time = datetime.utcnow()
    fields = dict(
        model_id=model_id,
        ds=ds,
        dataset_type=dataset_type,
        file_format=file_format,
        training_data=training_data,
        test_data=test_data,
        upstream_task_id=upstream_task_id,
        status=status,
        model_params_count=model_params_count,
        training_start_point=training_start_point,
        is_organic=False,
        created_at=current_time,
        termination_at=current_time + timedelta(hours=hours_to_complete),
        hours_to_complete=hours_to_complete,
        account_id=service_cst.NULL_ACCOUNT_ID,
        yarn_factor=None,
        augmentation_config=None,
    )
    calibration = dict(calibration_data=calibration_data, carve_calibration=True)
    if task_type == TaskType.QUANTFIXEDTASK:
        return QuantFixedRawTask(**fields, quant_spec=quant_spec, **calibration)
    if task_type == TaskType.QUANTFREEFORMTASK:
        return QuantFreeformRawTask(
            **fields,
            bar_cap=bar_cap,
            penalty_knee=penalty_knee,
            penalty_curvature=penalty_curvature,
            **calibration,
        )
    raise ValueError(f"{task_type} is not a quant task type")


@retry_with_backoff
async def create_continuous_sft_task(config: Config, lineage: str, seed_model: str) -> RawTask:
    """Create one lineage's continuous-SFT boss task: fixed chat-SFT from the carried-forward winner
    (or seed_model on first run), on this lineage's next stage-1 chunk.

    The content service is stateless: we pass the monotonic train_index and it re-materializes
    train+test at fresh randomized S3 URLs each call, so miners can't derive the held-out test set
    across tournaments. Lineage slug is encoded into the task ds so carry-forward routes the winner.
    Fixed 4xH100 and a fixed hours budget (the same after prep). No augmentation, so the carried base is
    never perturbed.
    """
    state = await get_continuous_sft_state(lineage, config.psql_db)
    if state.chunk_scheme != t_cst.CONTINUOUS_SFT_CHUNK_SCHEME:
        raise RuntimeError(
            f"continuous_sft_state[{lineage}] is in the {state.chunk_scheme.value} chunk scheme (train_index="
            f"{state.train_index}), but the content service now serves {t_cst.CONTINUOUS_SFT_CHUNK_SCHEME.value} "
            f"chunks; apply migration 20261113120000_continuous_sft_chunk_scheme.sql before creating this task"
        )
    base_model = state.last_winner_repo or seed_model

    response = await call_content_service(
        service_cst.GET_CONTINUOUS_SFT_DATA_ENDPOINT,
        config.keypair,
        {"train_index": state.train_index},
    )
    logger.info(
        f"Retrieved continuous-SFT chunk data for lineage={lineage} train_index={state.train_index}: {response}"
    )
    if not isinstance(response, dict):
        raise ValueError("Expected dict response from continuous-SFT data endpoint")
    train_url = response.get("train_s3_url")
    test_url = response.get("test_s3_url")
    if not train_url or not test_url:
        raise ValueError(f"continuous-SFT data response missing train/test url: {response}")
    label = response.get("ds") or f"train-index-{state.train_index}"
    ds_label = t_cst.continuous_sft_ds(lineage, label)

    number_of_hours = t_cst.CONTINUOUS_SFT_TRAINING_HOURS
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=number_of_hours)

    task = ChatRawTask(
        model_id=base_model,
        ds=ds_label,
        status=TaskStatus.PENDING,
        is_organic=False,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=number_of_hours,
        account_id=service_cst.NULL_ACCOUNT_ID,
        file_format=FileFormat.S3,
        training_data=train_url,
        test_data=test_url,
        training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
        # Base's own template, not chatml; eval loads the base tokenizer so train+eval match.
        chat_template="tokenizer_default",
        # other chat_* fields keep ChatRawTask ShareGPT defaults; augmentation_config stays None.
    )
    logger.info(
        f"CONTINUOUS_SFT_TASK: lineage={lineage} train_index={state.train_index} "
        f"base_model={base_model} hours={number_of_hours} ds={ds_label}"
    )
    task = await add_task(task, config.psql_db)
    try:
        await record_continuous_sft_task_chunk(
            ContinuousSftTaskChunk(
                task_id=task.task_id,
                lineage=lineage,
                train_index=state.train_index,
                chunk_scheme=t_cst.CONTINUOUS_SFT_CHUNK_SCHEME,
            ),
            config.psql_db,
        )
    except Exception:
        # No continuous-SFT task may exist without its chunk record: carry-forward would misread its cursor.
        logger.error(f"CONTINUOUS_SFT_TASK: recording chunk for task {task.task_id} failed; deleting the task")
        await task_sql.delete_task(task.task_id, config.psql_db)
        raise
    logger.info(f"CONTINUOUS_SFT_TASK: saved task {task.task_id}")

    return task
