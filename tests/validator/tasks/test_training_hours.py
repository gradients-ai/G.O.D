"""Tests for the throughput-based training-hours formula."""

import math

import pytest

import validator.tasks.datasets.constants as data_cst
import validator.tournament  # noqa: F401  (import-order: tournament package must init before scheduler)
from core.models.model_prep_models import InstructBaselineStats
from core.models.model_prep_models import InstructDatasetStats
from core.models.model_prep_models import InstructTrainingDynamics
from core.models.model_prep_models import SeqLengthDistribution
from core.models.model_prep_models import ThroughputStats
from core.models.model_prep_models import WeightStats
from core.models.task_models import TaskType
from validator.tasks.datasets.models import Dataset
from validator.tasks.synthetics.scheduler import _analytic_tokens_per_sec_per_gpu
from validator.tasks.synthetics.scheduler import _get_training_hours_from_num_rows
from validator.tasks.synthetics.scheduler import compute_hours_from_baseline_stats
from validator.tasks.synthetics.scheduler import compute_training_hours
from validator.tasks.synthetics.scheduler import get_dataset
from validator.tasks.synthetics.scheduler import get_grpo_training_hours
from validator.tournament.gpu_requirements import get_tournament_gpu_requirement


def _make_stats(total_tokens: int, num_records: int, tokens_per_sec: float | None = None) -> InstructBaselineStats:
    seq_dist = SeqLengthDistribution(mean=400.0, p50=350, p95=900, p99=1200, max=2000)
    throughput = None
    if tokens_per_sec is not None:
        throughput = ThroughputStats(tokens_per_sec=tokens_per_sec, seq_len=900, micro_batch_size=8, n_gpus=1)
    return InstructBaselineStats(
        dataset=InstructDatasetStats(
            total_tokens=total_tokens,
            num_records=num_records,
            seq_length_distribution=seq_dist,
            near_duplicate_rate=0.0,
            vocab_size=150_000,
            prompt_tokens=total_tokens // 2,
            completion_tokens=total_tokens // 2,
            completion_length_distribution=seq_dist,
        ),
        weights=WeightStats(by_group={}),
        training=InstructTrainingDynamics(
            init_loss=2.0,
            activation_rms={},
            output_entropy=3.0,
            masked_completion_loss=2.5,
        ),
        throughput=throughput,
    )


class TestComputeTrainingHours:
    def test_floor_for_tiny_tasks(self):
        assert compute_training_hours(8_000 * 200, 1.5e9, TaskType.INSTRUCTTEXTTASK) == data_cst.TRAINING_HOURS_MIN

    def test_cap_for_huge_tasks(self):
        assert compute_training_hours(175_000 * 1500, 32e9, TaskType.INSTRUCTTEXTTASK) == data_cst.MAX_TRAINING_HOURS

    def test_quarter_hour_granularity(self):
        hours = compute_training_hours(90_000 * 400, 8e9, TaskType.INSTRUCTTEXTTASK)
        assert hours * 4 == int(hours * 4)

    def test_quarter_hour_rounding_never_underbudgets(self):
        tokens = 10_000 * 400
        params = 12e9
        gpus = get_tournament_gpu_requirement(TaskType.INSTRUCTTEXTTASK, int(params)).gpu_count
        raw_hours = (
            data_cst.TARGET_TRAINING_EPOCHS
            * tokens
            / (_analytic_tokens_per_sec_per_gpu(params) * gpus)
            / 3600
            + data_cst.TRAINING_OVERHEAD_HOURS
        )

        assert compute_training_hours(tokens, params, TaskType.INSTRUCTTEXTTASK) == math.ceil(raw_hours * 4) / 4

    def test_two_epochs_fit_at_assumed_throughput(self):
        tokens = 90_000 * 400
        params = 8e9
        hours = compute_training_hours(tokens, params, TaskType.INSTRUCTTEXTTASK)
        train_hours = hours - data_cst.TRAINING_OVERHEAD_HOURS
        epochs = train_hours * 3600 * _analytic_tokens_per_sec_per_gpu(params) * 2 / tokens
        assert epochs >= data_cst.TARGET_TRAINING_EPOCHS * 0.9

    def test_dpo_costs_more_than_instruct_per_gpu(self):
        instruct = compute_training_hours(90_000 * 400, 1.3e9, TaskType.INSTRUCTTEXTTASK)
        dpo = compute_training_hours(90_000 * 400, 1.3e9, TaskType.DPOTASK)
        assert dpo >= instruct

    def test_measured_throughput_is_clamped(self):
        tokens = 90_000 * 400
        params = 8e9
        absurd_fast = compute_training_hours(tokens, params, TaskType.INSTRUCTTEXTTASK, measured_tokens_per_sec=1e9)
        analytic_hi = compute_training_hours(
            tokens,
            params,
            TaskType.INSTRUCTTEXTTASK,
            measured_tokens_per_sec=_analytic_tokens_per_sec_per_gpu(params) * data_cst.MEASURED_THROUGHPUT_CLAMP[1],
        )
        assert absurd_fast == analytic_hi


class TestComputeHoursFromBaselineStats:
    def test_uses_real_tokens(self):
        stats = _make_stats(total_tokens=90_000 * 400, num_records=90_000)
        hours = compute_hours_from_baseline_stats(2.0, stats, TaskType.INSTRUCTTEXTTASK, model_params_count=int(8e9))
        assert hours == compute_training_hours(90_000 * 400, 8e9, TaskType.INSTRUCTTEXTTASK)

    def test_legacy_stats_without_num_records_keep_current_hours(self):
        stats = _make_stats(total_tokens=40_000, num_records=0)
        assert compute_hours_from_baseline_stats(2.5, stats, TaskType.INSTRUCTTEXTTASK, model_params_count=int(8e9)) == 2.5

    def test_none_stats_keep_current_hours(self):
        assert compute_hours_from_baseline_stats(3.0, None, TaskType.INSTRUCTTEXTTASK) == 3.0

    def test_measured_throughput_used(self):
        tokens = 90_000 * 400
        stats = _make_stats(total_tokens=tokens, num_records=90_000, tokens_per_sec=4000.0)
        hours = compute_hours_from_baseline_stats(2.0, stats, TaskType.INSTRUCTTEXTTASK, model_params_count=int(8e9))
        assert hours == compute_training_hours(tokens, 8e9, TaskType.INSTRUCTTEXTTASK, measured_tokens_per_sec=4000.0)

    def test_short_rows_floored_per_row(self):
        stats = _make_stats(total_tokens=4_951_073, num_records=253_090, tokens_per_sec=4907.0)
        hours = compute_hours_from_baseline_stats(1.5, stats, TaskType.INSTRUCTTEXTTASK, model_params_count=int(8.54e9))
        floored_tokens = 253_090 * data_cst.EFFECTIVE_MIN_TOKENS_PER_ROW
        assert hours == compute_training_hours(
            floored_tokens,
            8.54e9,
            TaskType.INSTRUCTTEXTTASK,
            measured_tokens_per_sec=4907.0,
        )
        assert hours >= 1.5


class TestGrpoTrainingHours:
    def test_fixed_hours_by_model_size_band(self):
        assert get_grpo_training_hours(3e9) == 1.5
        assert get_grpo_training_hours(8e9) == 2.5
        assert get_grpo_training_hours(13e9) == 4.0
        assert get_grpo_training_hours(70e9) == data_cst.MAX_TRAINING_HOURS

    def test_grpo_num_rows_estimate_ignores_dataset_size(self):
        small = _get_training_hours_from_num_rows(8_000, task_type=TaskType.GRPOTASK)
        large = _get_training_hours_from_num_rows(175_000, task_type=TaskType.GRPOTASK)
        assert small == large == get_grpo_training_hours(data_cst.DEFAULT_MODEL_PARAMS_FOR_HOURS)

    def test_grpo_baseline_stats_ignore_token_shape(self):
        empty_stats = _make_stats(total_tokens=0, num_records=0)
        hours = compute_hours_from_baseline_stats(5.0, empty_stats, TaskType.GRPOTASK, model_params_count=int(3e9))
        assert hours == get_grpo_training_hours(3e9)


class TestGetDataset:
    @pytest.mark.asyncio
    async def test_min_rows_filter_skips_small_datasets(self):
        async def datasets():
            yield Dataset(dataset_id="too-small", num_rows=data_cst.GRPO_MIN_SYNTH_ROWS - 1, num_bytes_parquet_files=1)
            yield Dataset(dataset_id="large-enough", num_rows=data_cst.GRPO_MIN_SYNTH_ROWS, num_bytes_parquet_files=1)

        dataset = await get_dataset(datasets(), min_rows=data_cst.GRPO_MIN_SYNTH_ROWS)

        assert dataset.dataset_id == "large-enough"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
