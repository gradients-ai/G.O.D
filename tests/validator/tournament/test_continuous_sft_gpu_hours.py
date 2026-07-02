"""Fixed compute for the continuous-SFT task: 4xH100 + fixed hours, short-circuiting the
HuggingFace param fetch (the base is gated / custom-arch and would throw during a lookup).

The short-circuit ORDER is the correctness point: the CONTINUOUS_SFT branch must return before
get_model_num_params is ever reached, else creating the boss round crashes on the gated repo.
"""

from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskType
from validator.tasks.synthetics import scheduler
from validator.tournament import gpu_requirements
from validator.tournament.models import GpuRequirement


def _boom(*args, **kwargs):
    raise AssertionError("get_model_num_params must not be called for a continuous-SFT task")


class TestGpuRequirement:
    def test_continuous_sft_forces_4xh100_without_param_fetch(self, monkeypatch):
        monkeypatch.setattr(gpu_requirements, "get_model_num_params", _boom)
        req = gpu_requirements.get_tournament_gpu_requirement(
            TaskType.CHATTASK,
            model_params_count=0,  # would normally trigger the HF fetch
            model_id="gradients-io-tournaments/continuous-sft-seed-quasar-king",
            training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
        )
        assert req == GpuRequirement.H100_4X

    def test_non_continuous_task_still_uses_normal_routing(self, monkeypatch):
        # The new branch must not leak into normal tasks: image still 1xH100, no param fetch.
        monkeypatch.setattr(gpu_requirements, "get_model_num_params", _boom)
        req = gpu_requirements.get_tournament_gpu_requirement(
            TaskType.IMAGETASK, model_params_count=0, model_id=None, training_start_point=None
        )
        assert req == GpuRequirement.H100_1X


class TestComputeHours:
    def test_continuous_sft_returns_current_hours_unchanged(self, monkeypatch):
        monkeypatch.setattr(scheduler, "get_model_num_params", _boom)
        hours = scheduler.compute_hours_from_baseline_stats(
            current_hours=6.0,
            baseline_stats=object(),  # would normally be inspected; must be skipped
            task_type=TaskType.CHATTASK,
            model_id="gradients-io-tournaments/continuous-sft-seed-quasar-king",
            training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
        )
        assert hours == 6.0
