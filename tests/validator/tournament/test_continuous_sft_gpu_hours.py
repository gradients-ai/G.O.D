"""Continuous-SFT compute sizing: GPUs stay a forced 4xH100, hours come from the general
throughput pipeline.

The GPU force must return before get_model_num_params is ever reached (the carried base is
gated / custom-arch and would throw during a lookup). Hours, by contrast, flow through
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


QUASAR_SEED = "gradients-io-tournaments/continuous-sft-seed-quasar-king"
SEED_PARAMS = 8_600_000_000


def _boom(*args, **kwargs):
    raise AssertionError("get_model_num_params must not be called for a continuous-SFT task")


class TestGpuRequirement:
    def test_continuous_sft_forces_4xh100_without_param_fetch(self, monkeypatch):
        monkeypatch.setattr(gpu_requirements, "get_model_num_params", _boom)
        req = gpu_requirements.get_tournament_gpu_requirement(
            TaskType.CHATTASK,
            model_params_count=0,  # would normally trigger the HF fetch
            model_id=QUASAR_SEED,
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
            model_id=QUASAR_SEED,
            model_params_count=SEED_PARAMS,
            training_start_point=training_start_point,
        )

    def test_continuous_sft_budget_divides_by_4_gpus(self):
        # Identical stats, chat task: DEFAULT sizes GPUs from params (~8.6B -> 2xH100) while
        # CONTINUOUS_SFT uses the forced 4xH100, so its budget must be strictly smaller.
        continuous = self._hours(TrainingStartPoint.CONTINUOUS_SFT)
        default = self._hours(TrainingStartPoint.DEFAULT)
        assert continuous < default

    def test_params_fetched_from_seed_not_carried_winner(self, monkeypatch):
        # The exact orchestrator configuration: stats present, model_params_count unset, model_id
        # is the carried winner (round 2+: possibly a LoRA adapter whose safetensors total is
        # adapter-sized). The seed must be resolved from ds INSIDE the function so no call site
        # can feed the winner into the param fetch.
        fetched = []

        def _record(model_id):
            fetched.append(model_id)
            return SEED_PARAMS

        monkeypatch.setattr(scheduler, "get_model_num_params", _record)
        scheduler.compute_hours_from_baseline_stats(
            current_hours=4.0,
            baseline_stats=_stats(total_tokens=200_000_000, num_records=100_000, tokens_per_sec=8_000.0),
            task_type=TaskType.CHATTASK,
            model_id="miner-org/carried-lora-winner",
            model_params_count=None,
            training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
            ds="continuous-sft:quasar:chunk-00003",
        )
        assert fetched == [QUASAR_SEED]

    def test_no_baseline_stats_keeps_the_fallback_budget(self, monkeypatch):
        monkeypatch.setattr(scheduler, "get_model_num_params", _boom)
        hours = scheduler.compute_hours_from_baseline_stats(
            current_hours=4.0,
            baseline_stats=None,
            task_type=TaskType.CHATTASK,
            model_id=QUASAR_SEED,
            training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
        )
        assert hours == 4.0
