"""GPU requirement computation for tournament evaluation and training."""

from core.constants.environments import TrainingStartPoint
from core.logging import get_logger
from core.models.task_models import TaskType
from validator.tasks.requests import get_model_num_params
from validator.tournament.models import GpuRequirement


logger = get_logger(__name__)

TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100 = 4.0
TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100 = 12.0
TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100 = 40.0
TOURNAMENT_DPO_GPU_MULTIPLIER = 3
TOURNAMENT_GRPO_GPU_MULTIPLIER = 2
TOURNAMENT_KL_GPU_MULTIPLIER = 2


def get_tournament_gpu_requirement(
    task_type: TaskType,
    model_params_count: int,
    model_id: str | None = None,
    gpu_multiplier: int | None = None,
    use_kl: bool = False,
    training_start_point: TrainingStartPoint | None = None,
) -> GpuRequirement:
    """Compute GPU requirement based on model size, task type, and optional multiplier.

    When ``use_kl`` is set (instruct KL tasks), the effective model size is scaled by
    TOURNAMENT_KL_GPU_MULTIPLIER to account for the frozen reference model held resident
    alongside the trainable model.

    Continuous-SFT gets a FIXED 4xH100, checked first so we never hit the (gated/custom-arch)
    HuggingFace param fetch below.
    """
    if training_start_point == TrainingStartPoint.CONTINUOUS_SFT:
        return GpuRequirement.H100_4X

    if task_type == TaskType.IMAGETASK:
        return GpuRequirement.H100_1X

    if not model_params_count and model_id:
        logger.info(f"model_params_count is {model_params_count}, fetching from HuggingFace for model {model_id}")
        try:
            model_params_count = get_model_num_params(model_id)
            logger.info(f"Fetched model_params_count: {model_params_count} for model {model_id}")
        except Exception:
            model_params_count = 0

        if not model_params_count:
            logger.warning(f"Could not determine model size for {model_id}, defaulting to H100_1X")
            return GpuRequirement.H100_1X

    params_b = model_params_count / 1_000_000_000

    if task_type == TaskType.DPOTASK:
        params_b *= TOURNAMENT_DPO_GPU_MULTIPLIER
    elif task_type == TaskType.GRPOTASK:
        params_b *= TOURNAMENT_GRPO_GPU_MULTIPLIER
    elif task_type == TaskType.ENVIRONMENTTASK:
        if gpu_multiplier is not None:
            params_b *= gpu_multiplier
        else:
            return GpuRequirement.H100_4X

    if use_kl:
        params_b *= TOURNAMENT_KL_GPU_MULTIPLIER

    if params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100:
        return GpuRequirement.H100_1X
    elif params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100:
        return GpuRequirement.H100_2X
    elif params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100:
        return GpuRequirement.H100_4X
    else:
        return GpuRequirement.H100_8X
