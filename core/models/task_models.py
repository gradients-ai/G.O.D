from enum import Enum
from uuid import UUID

from pydantic import BaseModel


class TaskStatus(str, Enum):
    PENDING = "pending"
    # A chained boss task, held until its upstream instruct task is SUCCESS; released to PENDING.
    AWAITING_UPSTREAM = "awaiting_upstream"
    PREPARING_DATA = "preparing_data"
    PREP_TASK_FAILURE = "prep_task_failure"
    AWAITING_MODEL_PREP = "awaiting_model_prep"
    LOOKING_FOR_NODES = "looking_for_nodes"
    FAILURE_FINDING_NODES = "failure_finding_nodes"
    DELAYED = "delayed"
    READY = "ready"
    TRAINING = "training"
    PREEVALUATION = "preevaluation"
    EVALUATING = "evaluating"
    SUCCESS = "success"
    FAILURE = "failure"


class MinerTaskResult(BaseModel):
    hotkey: str
    quality_score: float
    test_loss: float | None
    synth_loss: float | None
    score_reason: str | None


# NOTE: Confusing name with the class above
class TaskMinerResult(BaseModel):
    task_id: UUID
    quality_score: float


class TaskType(str, Enum):
    INSTRUCTTEXTTASK = "InstructTextTask"
    IMAGETASK = "ImageTask"
    DPOTASK = "DpoTask"
    GRPOTASK = "GrpoTask"
    CHATTASK = "ChatTask"
    ENVIRONMENTTASK = "EnvTask"
    QUANTFIXEDTASK = "QuantFixedTask"
    QUANTFREEFORMTASK = "QuantFreeformTask"

    def __hash__(self):
        return hash(str(self))


QUANT_TASK_TYPES = frozenset({TaskType.QUANTFIXEDTASK, TaskType.QUANTFREEFORMTASK})

# Every type ranked through MinerResultsText (all but ImageTask).
TEXT_TASK_TYPES = (
    frozenset({TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK, TaskType.ENVIRONMENTTASK})
    | QUANT_TASK_TYPES
)

# Types whose test_loss is a reward: rank descending. Every other type ranks ascending on loss.
HIGHER_IS_BETTER_TASK_TYPES = frozenset({TaskType.GRPOTASK, TaskType.ENVIRONMENTTASK})


def higher_is_better(task_type: TaskType) -> bool:
    return task_type in HIGHER_IS_BETTER_TASK_TYPES
