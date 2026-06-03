from enum import Enum
from uuid import UUID

from pydantic import BaseModel


class TaskStatus(str, Enum):
    PENDING = "pending"
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

    def __hash__(self):
        return hash(str(self))
