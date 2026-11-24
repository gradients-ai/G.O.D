"""Everything the head-to-heads of one task read from the database, fetched once.

A head-to-head against the boss reads the task's ranked results, the training statuses and the two per-example loss
vectors (and, for compression, the eval GPUs and references). Run for every miner of a task that is a handful of queries
per miner; ``TaskReads`` holds the task's rows so any number of head-to-heads then run in memory.
"""

from pydantic import BaseModel
from pydantic import ConfigDict

from core.models.quant_models import QuantEvalGpu
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskType
from validator.db.database import PSQLDB
from validator.db.sql.quant_references import get_quant_sku_references
from validator.db.sql.submissions_and_scoring import StoredLosses
from validator.db.sql.submissions_and_scoring import get_per_example_losses_for_task
from validator.db.sql.submissions_and_scoring import get_task_node_eval_gpus
from validator.db.sql.tournaments import get_training_status_for_task
from validator.scoring.models import MinerResultsImage
from validator.scoring.models import MinerResultsText
from validator.scoring.models import QuantSkuReferences
from validator.tournament.task_results import get_task_results_for_ranking


class TaskReads(BaseModel):
    """One task's stored results, as the head-to-head functions read them."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    results: list[MinerResultsText | MinerResultsImage]
    statuses: dict[str, str]
    losses: dict[str, StoredLosses]
    eval_gpus: dict[str, QuantEvalGpu | None]
    references: QuantSkuReferences

    def fresh_results(self) -> list[MinerResultsText | MinerResultsImage]:
        """A copy of the ranked results: ranking sets scores on the objects it is given, so each comparison gets its own."""
        return [result.model_copy(deep=True) for result in self.results]


async def load_task_reads(task_id: str, task_type: TaskType, psql_db: PSQLDB) -> TaskReads:
    """Read a task's rows once, one query per kind of row and none per miner."""
    results = await get_task_results_for_ranking(task_id, psql_db)
    statuses = await get_training_status_for_task(task_id, psql_db)
    losses = await get_per_example_losses_for_task(task_id, psql_db)
    is_quant = task_type in QUANT_TASK_TYPES
    return TaskReads(
        results=results,
        statuses=statuses,
        losses=losses,
        eval_gpus=await get_task_node_eval_gpus(task_id, list(losses), psql_db) if is_quant else {},
        references=await get_quant_sku_references(task_id, psql_db) if is_quant else {},
    )
