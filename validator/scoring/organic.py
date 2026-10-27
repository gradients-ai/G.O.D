"""Database-free evaluation of a single organic training result.

The product worker owns persistence; this module only runs the evaluation and
returns a JSON-safe result. Passing ``config=None`` keeps every SQL read out of the
evaluation path (see ``load_eval_pair_state_for_models``).
"""

import math

import validator.lifecycle.constants as lifecycle_cst
from core.models.payload_models import EvaluationResultImage
from core.models.task_models import TaskType
from validator.scoring.tasks import _calculate_weighted_loss_for_image_eval
from validator.scoring.tasks import _evaluate_submissions
from validator.scoring.tasks import _get_dataset_type
from validator.tasks.models import AnyTypeRawTask
from validator.tasks.requests import get_model_num_params


_PARAMETER_MULTIPLIERS = {
    TaskType.DPOTASK: 2,
    TaskType.GRPOTASK: 3,
    TaskType.ENVIRONMENTTASK: 3,
}


def organic_eval_gpu_count(task: AnyTypeRawTask) -> int:
    num_params = task.model_params_count or get_model_num_params(task.model_id)
    if not num_params:
        return 1
    num_params *= _PARAMETER_MULTIPLIERS.get(task.task_type, 1)
    if num_params < lifecycle_cst.MODEL_SIZE_REQUIRING_2_GPUS:
        return 1
    if num_params < lifecycle_cst.MODEL_SIZE_REQUIRING_4_GPUS:
        return 2
    return 4


async def evaluate_organic_repo(task: AnyTypeRawTask, repo: str, num_gpus: int | None = None) -> dict:
    """Evaluate one trained repository against the task's held-out test split.

    Raises when evaluation fails or the repository is not a finetune of the base model.
    """
    results = await _evaluate_submissions(
        task,
        [repo],
        num_gpus or organic_eval_gpu_count(task),
        _get_dataset_type(task),
        config=None,
    )
    result = results.get(repo)
    if result is None:
        raise RuntimeError(f"evaluation returned no result for {repo}")
    if isinstance(result, Exception):
        raise result
    if not result.is_finetune:
        raise ValueError(f"{repo} is not a finetune of {task.model_id}")
    summary = result.model_dump(mode="json")
    if isinstance(result, EvaluationResultImage):
        summary["weighted_loss"] = _calculate_weighted_loss_for_image_eval(result)
    loss = summary.get("weighted_loss", summary.get("eval_loss"))
    if isinstance(loss, float) and not math.isfinite(loss):
        raise ValueError(f"evaluation of {repo} produced a non-finite loss")
    return summary
