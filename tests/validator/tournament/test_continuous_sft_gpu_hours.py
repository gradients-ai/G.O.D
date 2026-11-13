"""Continuous-SFT compute sizing: GPUs stay a forced 4xH100, hours come from the general
throughput pipeline.

The GPU force must return before get_model_num_params is ever reached (the carried base is a
miner repo that may be gated and would throw during a lookup). Hours, by contrast, flow through
compute_hours_from_baseline_stats like any SFT task — the two correctness points there are that
the budget divides by the REAL 4 GPUs (param-only sizing would give 2 for an ~8.6B chat model)
and that callers pass the lineage seed for the param fetch, never the carried winner.
"""

from types import SimpleNamespace

from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskType
from validator.tasks.synthetics import scheduler
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


class TestComputeHours:
    def _hours(self, training_start_point):
        return scheduler.compute_hours_from_baseline_stats(
            current_hours=4.0,
            baseline_stats=_stats(total_tokens=200_000_000, num_records=100_000, tokens_per_sec=8_000.0),
            task_type=TaskType.CHATTASK,
            model_id=LINEAGE_SEED,
            model_params_count=SEED_PARAMS,
            training_start_point=training_start_point,
        )

    def test_continuous_sft_hours_are_pinned(self):
        assert self._hours(TrainingStartPoint.CONTINUOUS_SFT) == 2.5

    def test_throughput_budget_divides_by_4_gpus_for_continuous_start_point(self):
        # The throughput estimate itself still sizes by the forced 4xH100 (DEFAULT sizes GPUs from
        # params: ~8.6B -> 2xH100), though continuous-SFT tasks do not use its result. A dataset small
        # enough that neither budget hits the MAX_TRAINING_HOURS cap.
        continuous = scheduler.compute_training_hours(
            20_000_000, SEED_PARAMS, TaskType.CHATTASK, 8_000.0, TrainingStartPoint.CONTINUOUS_SFT
        )
        default = scheduler.compute_training_hours(
            20_000_000, SEED_PARAMS, TaskType.CHATTASK, 8_000.0, TrainingStartPoint.DEFAULT
        )
        assert continuous < default

    def test_no_param_fetch_for_continuous_sft(self, monkeypatch):
        # The carried winner may be an unfetchable LoRA adapter; the pinned budget never looks at params.
        monkeypatch.setattr(scheduler, "get_model_num_params", _boom)
        hours = scheduler.compute_hours_from_baseline_stats(
            current_hours=4.0,
            baseline_stats=_stats(total_tokens=200_000_000, num_records=100_000, tokens_per_sec=8_000.0),
            task_type=TaskType.CHATTASK,
            model_id="miner-org/carried-lora-winner",
            model_params_count=None,
            training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
            ds="continuous-sft:qwen:chunk-00003",
        )
        assert hours == 2.5

    def test_no_baseline_stats_gives_the_pinned_budget(self, monkeypatch):
        monkeypatch.setattr(scheduler, "get_model_num_params", _boom)
        hours = scheduler.compute_hours_from_baseline_stats(
            current_hours=4.0,
            baseline_stats=None,
            task_type=TaskType.CHATTASK,
            model_id=LINEAGE_SEED,
            training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
        )
        assert hours == 2.5
