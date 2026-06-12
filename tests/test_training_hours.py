"""Tests for the throughput-based training-hours formula."""

import pytest

import validator.tournament  # noqa: F401  (import-order: tournament package must init before synthetic_scheduler)
import validator.core.constants as vcst
from core.models.model_prep_models import (
    InstructBaselineStats,
    InstructDatasetStats,
    InstructTrainingDynamics,
    SeqLengthDistribution,
    ThroughputStats,
    WeightStats,
)
from core.models.utility_models import TaskType
from validator.tasks.synthetic_scheduler import (
    _analytic_tokens_per_sec_per_gpu,
    compute_hours_from_baseline_stats,
    compute_training_hours,
)


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
        assert compute_training_hours(8_000 * 200, 1.5e9, TaskType.INSTRUCTTEXTTASK) == vcst.TRAINING_HOURS_MIN

    def test_cap_for_huge_tasks(self):
        assert compute_training_hours(175_000 * 1500, 32e9, TaskType.INSTRUCTTEXTTASK) == vcst.MAX_TRAINING_HOURS

    def test_quarter_hour_granularity(self):
        hours = compute_training_hours(90_000 * 400, 8e9, TaskType.INSTRUCTTEXTTASK)
        assert hours * 4 == int(hours * 4)

    def test_two_epochs_fit_at_assumed_throughput(self):
        tokens = 90_000 * 400
        params = 8e9
        hours = compute_training_hours(tokens, params, TaskType.INSTRUCTTEXTTASK)
        train_hours = hours - vcst.TRAINING_OVERHEAD_HOURS
        # 8B -> 2xH100
        epochs = train_hours * 3600 * _analytic_tokens_per_sec_per_gpu(params) * 2 / tokens
        assert epochs >= vcst.TARGET_TRAINING_EPOCHS * 0.9

    def test_dpo_costs_more_than_instruct_per_gpu(self):
        # Same tokens/params; DPO gets more GPUs (3x param multiplier), so
        # compare the raw multiplier on a model small enough to stay at 1 GPU.
        instruct = compute_training_hours(90_000 * 400, 1.3e9, TaskType.INSTRUCTTEXTTASK)
        dpo = compute_training_hours(90_000 * 400, 1.3e9, TaskType.DPOTASK)
        assert dpo >= instruct

    def test_measured_throughput_is_clamped(self):
        tokens = 90_000 * 400
        params = 8e9
        absurd_fast = compute_training_hours(tokens, params, TaskType.INSTRUCTTEXTTASK, measured_tokens_per_sec=1e9)
        analytic_hi = compute_training_hours(
            tokens, params, TaskType.INSTRUCTTEXTTASK,
            measured_tokens_per_sec=_analytic_tokens_per_sec_per_gpu(params) * vcst.MEASURED_THROUGHPUT_CLAMP[1],
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
        # The gemma-7b/vi_en case: 253k rows of ~19.6 tokens. Even a packing
        # miner pays block-density loss and per-step overhead, so each row
        # costs at least EFFECTIVE_MIN_TOKENS_PER_ROW — the budget must not
        # collapse to the raw token count.
        stats = _make_stats(total_tokens=4_951_073, num_records=253_090, tokens_per_sec=4907.0)
        hours = compute_hours_from_baseline_stats(1.5, stats, TaskType.INSTRUCTTEXTTASK, model_params_count=int(8.54e9))
        floored_tokens = 253_090 * vcst.EFFECTIVE_MIN_TOKENS_PER_ROW
        assert hours == compute_training_hours(
            floored_tokens, 8.54e9, TaskType.INSTRUCTTEXTTASK, measured_tokens_per_sec=4907.0
        )
        assert hours >= 1.5  # must not shrink below the old grant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
