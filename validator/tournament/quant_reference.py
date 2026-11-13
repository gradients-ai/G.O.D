"""Which bf16 model a quant row is normalised by.

On an own-model task (a miner compressing its own upstream model) that is that model; on any
other quant task it is the task's common input model. `reference_hotkey_for` names the key its measured loss is
stored under in quant_sku_references.
"""

from core.constants.environments import TrainingStartPoint
from validator.db.database import PSQLDB
from validator.db.sql import tasks as task_sql
from validator.tasks.models import AnyQuantRawTask


# Scored-fail reason of an own-model miner with no starting model: it has no bf16 reference, so its loss cannot be
# normalised; only that miner is unscoreable and the task stands.
NO_REFERENCE_MODEL_FAILURE = "no bf16 reference model"

def is_own_model_start_point(start_point: TrainingStartPoint | None) -> bool:
    """Whether every miner starts from, and is referenced against, its own model (R1 output or boss-round upstream output)."""
    return start_point in (TrainingStartPoint.OWN_R1_MODEL, TrainingStartPoint.OWN_UPSTREAM_MODEL)


def reference_hotkey_for(task: AnyQuantRawTask, hotkey: str) -> str:
    """The hotkey a row's bf16 reference is stored under: the miner's own on an own-model task, else '' (task-common)."""
    return hotkey if is_own_model_start_point(task.training_start_point) else ""


async def quant_reference_model(task: AnyQuantRawTask, hotkey: str, psql_db: PSQLDB) -> str | None:
    """The bf16 model the quant loss of `hotkey`'s row on the task is normalised by.

    On an own-model task, the miner's own starting model (task_nodes.starting_model_repo); None when unset,
    so that miner has no usable reference. Otherwise the common input as the entrants received it (augmented
    when it was).
    """
    if is_own_model_start_point(task.training_start_point):
        return await task_sql.get_starting_model_repo(str(task.task_id), hotkey, psql_db)
    return task.augmented_model_id or task.model_id
