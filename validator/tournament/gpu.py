"""GPU requirement computation for tournament evaluation and training."""

from core.models.tournament_models import GpuRequirement
from core.models.utility_models import TaskType
from validator.core.constants import (
    TOURNAMENT_DPO_GPU_MULTIPLIER,
    TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100,
    TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100,
    TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100,
    TOURNAMENT_GRPO_GPU_MULTIPLIER,
)
from validator.cycle.util_functions import get_model_num_params
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def get_tournament_gpu_requirement(
    task_type: TaskType,
    model_params_count: int,
    model_id: str | None = None,
    gpu_multiplier: int | None = None,
) -> GpuRequirement:
    """Compute GPU requirement based on model size, task type, and optional multiplier."""
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
    elif task_type == TaskType.COMPOSITETASK:
        # Composites bundle instruct/dpo/grpo on one model — size for the heaviest method actually
        # present (passed in); fall back to DPO (the heaviest possible) when unknown.
        params_b *= gpu_multiplier if gpu_multiplier is not None else TOURNAMENT_DPO_GPU_MULTIPLIER
    elif task_type == TaskType.ENVIRONMENTTASK:
        if gpu_multiplier is not None:
            params_b *= gpu_multiplier
        else:
            return GpuRequirement.H100_4X

    if params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100:
        return GpuRequirement.H100_1X
    elif params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100:
        return GpuRequirement.H100_2X
    elif params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100:
        return GpuRequirement.H100_4X
    else:
        return GpuRequirement.H100_8X


async def composite_method_gpu_multiplier(composite_task_id: str, psql_db) -> int:
    """The heaviest method multiplier among a composite's subtasks (instruct 1x, grpo 2x, dpo 3x)."""
    from uuid import UUID

    from validator.db.sql.tasks import get_task
    from validator.db.sql.tournaments import get_composite_task_subtasks

    multipliers = {TaskType.DPOTASK: TOURNAMENT_DPO_GPU_MULTIPLIER, TaskType.GRPOTASK: TOURNAMENT_GRPO_GPU_MULTIPLIER}
    heaviest = 1
    for subtask in await get_composite_task_subtasks(composite_task_id, psql_db):
        subtask_task = await get_task(UUID(subtask.subtask_task_id), psql_db)
        heaviest = max(heaviest, multipliers.get(subtask_task.task_type, 1))
    return heaviest


async def tournament_gpu_requirement_for_task(task, psql_db) -> GpuRequirement:
    """GPU requirement for a tournament task; composites are sized by their heaviest sampled method."""
    multiplier = (
        await composite_method_gpu_multiplier(str(task.task_id), psql_db)
        if task.task_type == TaskType.COMPOSITETASK
        else None
    )
    return get_tournament_gpu_requirement(
        task.task_type, task.model_params_count or 0, task.model_id, gpu_multiplier=multiplier
    )
