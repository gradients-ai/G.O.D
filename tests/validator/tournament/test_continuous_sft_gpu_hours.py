"""Continuous-SFT compute sizing: GPUs stay a forced 4xH100, hours are a fixed budget.

The GPU force must return before get_model_num_params is ever reached (the carried base is a
miner repo that may be gated and would throw during a lookup). Hours are pinned to
CONTINUOUS_SFT_TRAINING_HOURS by compute_hours_from_baseline_stats whatever the baseline stats say,
and never fetch model params (the carried winner may be an unfetchable adapter).
"""

from types import SimpleNamespace

import pytest

from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskType
from validator.tasks.synthetics import scheduler
from validator.tournament import constants as t_cst
from validator.tournament import gpu_requirements
from validator.tournament.models import GpuRequirement


LINEAGE_SEED = "Qwen/Qwen3-8B-Base"
SEED_PARAMS = 8_600_000_000


def _boom(*args, **kwargs):
    raise AssertionError("get_model_num_params must not be called for a continuous-SFT task")


class TestGpuRequirement:
    def test_continuous_sft_forces_4xh100_without_param_fetch(self, monkeypatch):
        monkeypatch.setattr(gpu_requirements, "get_model_num_params", _boom)
        req = gpu_requirements.get_tournament_gpu_requirement(
            TaskType.CHATTASK,
            model_params_count=0,  # would normally trigger the HF fetch
            model_id=LINEAGE_SEED,
            training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
        )
        assert req == GpuRequirement.H100_4X

    def test_non_continuous_task_still_uses_normal_routing(self, monkeypatch):
        # The force must not leak into normal tasks: image still 1xH100, no param fetch.
        monkeypatch.setattr(gpu_requirements, "get_model_num_params", _boom)
        req = gpu_requirements.get_tournament_gpu_requirement(
            TaskType.IMAGETASK, model_params_count=0, model_id=None, training_start_point=None
        )
        assert req == GpuRequirement.H100_1X


def _stats(total_tokens: int, num_records: int, tokens_per_sec: float | None):
    throughput = SimpleNamespace(tokens_per_sec=tokens_per_sec) if tokens_per_sec else None
    return SimpleNamespace(
        dataset=SimpleNamespace(total_tokens=total_tokens, num_records=num_records),
        throughput=throughput,
    )


class TestPinnedHours:
    def _hours(
        self,
        stats,
        *,
        current_hours=4.0,
        max_hours=None,
        model_id=LINEAGE_SEED,
        ds=None,
        start_point=TrainingStartPoint.CONTINUOUS_SFT,
    ):
        return scheduler.compute_hours_from_baseline_stats(
            current_hours=current_hours,
            baseline_stats=stats,
            task_type=TaskType.CHATTASK,
            model_id=model_id,
            model_params_count=SEED_PARAMS,
            training_start_point=start_point,
            ds=ds,
            max_hours=max_hours,
        )

    def test_constant_is_two_and_a_half(self):
        assert t_cst.CONTINUOUS_SFT_TRAINING_HOURS == 2.5

    def test_typical_stats(self):
        assert self._hours(_stats(200_000_000, 100_000, 8_000.0)) == t_cst.CONTINUOUS_SFT_TRAINING_HOURS

    def test_no_stats(self):
        assert self._hours(None) == t_cst.CONTINUOUS_SFT_TRAINING_HOURS

    def test_stats_that_would_compute_far_more(self):
        assert self._hours(_stats(50_000_000_000, 5_000_000, 500.0)) == t_cst.CONTINUOUS_SFT_TRAINING_HOURS

    def test_stats_that_would_compute_far_less(self):
        assert self._hours(_stats(1_000, 10, 1_000_000.0)) == t_cst.CONTINUOUS_SFT_TRAINING_HOURS

    @pytest.mark.parametrize("current_hours", [0.5, 2.5, 4.0, 30.0])
    def test_current_hours_does_not_change_it(self, current_hours):
        stats = _stats(200_000_000, 100_000, 8_000.0)
        assert self._hours(stats, current_hours=current_hours) == t_cst.CONTINUOUS_SFT_TRAINING_HOURS

    @pytest.mark.parametrize("max_hours", [1.0, 2.0, 3.0, 10.0, 48.0])
    def test_max_hours_does_not_change_it(self, max_hours):
        stats = _stats(200_000_000, 100_000, 8_000.0)
        assert self._hours(stats, max_hours=max_hours) == t_cst.CONTINUOUS_SFT_TRAINING_HOURS

    def test_ds_alone_identifies_a_continuous_task(self):
        hours = self._hours(
            _stats(200_000_000, 100_000, 8_000.0),
            current_hours=9.0,
            ds=t_cst.continuous_sft_ds("qwen3-8b", "chunk-00010"),
            start_point=TrainingStartPoint.DEFAULT,
        )
        assert hours == t_cst.CONTINUOUS_SFT_TRAINING_HOURS

    def test_no_param_fetch(self, monkeypatch):
        # The carried winner may be an unfetchable LoRA adapter; the pinned budget never looks at params.
        monkeypatch.setattr(scheduler, "get_model_num_params", _boom)
        hours = scheduler.compute_hours_from_baseline_stats(
            current_hours=4.0,
            baseline_stats=_stats(200_000_000, 100_000, 8_000.0),
            task_type=TaskType.CHATTASK,
            model_id="miner-org/carried-lora-winner",
            model_params_count=None,
            training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
            ds="continuous-sft:qwen:chunk-00003",
        )
        assert hours == t_cst.CONTINUOUS_SFT_TRAINING_HOURS

    def test_non_continuous_task_is_unaffected(self):
        stats = _stats(200_000_000, 100_000, 8_000.0)
        expected = scheduler.compute_training_hours(
            max(200_000_000, 100_000 * scheduler.data_cst.EFFECTIVE_MIN_TOKENS_PER_ROW),
            SEED_PARAMS,
            TaskType.INSTRUCTTEXTTASK,
            8_000.0,
            TrainingStartPoint.DEFAULT,
            max_hours=None,
        )
        got = scheduler.compute_hours_from_baseline_stats(
            4.0, stats, TaskType.INSTRUCTTEXTTASK, model_id="org/model", model_params_count=SEED_PARAMS
        )
        assert got == expected
        assert scheduler.compute_hours_from_baseline_stats(4.0, None, TaskType.INSTRUCTTEXTTASK, model_id="org/model") == 4.0

    def test_throughput_budget_still_divides_by_4_gpus_for_continuous_start_point(self):
        # The raw estimate (no longer used for continuous tasks) still sizes by the forced 4xH100.
        # A small dataset keeps both budgets under the MAX_TRAINING_HOURS cap.
        continuous = scheduler.compute_training_hours(
            20_000_000, SEED_PARAMS, TaskType.CHATTASK, 8_000.0, TrainingStartPoint.CONTINUOUS_SFT
        )
        default = scheduler.compute_training_hours(
            20_000_000, SEED_PARAMS, TaskType.CHATTASK, 8_000.0, TrainingStartPoint.DEFAULT
        )
        assert continuous < default
