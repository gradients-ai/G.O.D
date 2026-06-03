import numpy as np

from core.logging import get_logger
from core.models.task_models import TaskType
from validator.db import constants as db_cst
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import get_all_scores_and_losses_for_task
from validator.db.sql.tasks import get_task
from validator.scoring.models import MinerResultsImage
from validator.scoring.models import MinerResultsText
from validator.scoring.tasks import calculate_miner_ranking_and_scores


logger = get_logger(__name__)


async def get_task_results_for_ranking(task_id: str, psql_db: PSQLDB) -> list[MinerResultsText | MinerResultsImage]:
    """
    Fetch task results from database and convert to MinerResults objects for ranking.
    """
    scores_dicts = await get_all_scores_and_losses_for_task(task_id, psql_db)

    if not scores_dicts:
        logger.warning(f"No scores found for task {task_id}")
        return []

    task_object = await get_task(task_id, psql_db)
    if not task_object:
        logger.warning(f"Could not get task object for task {task_id}")
        return []

    task_type = task_object.task_type

    miner_results = []
    for score_dict in scores_dicts:
        hotkey = score_dict[db_cst.HOTKEY]
        test_loss = score_dict.get(db_cst.TEST_LOSS)

        # Skip invalid results
        if test_loss is None or np.isnan(test_loss):
            continue

        # Create appropriate MinerResults object
        if task_type in [
            TaskType.INSTRUCTTEXTTASK,
            TaskType.CHATTASK,
            TaskType.DPOTASK,
            TaskType.GRPOTASK,
            TaskType.ENVIRONMENTTASK,
        ]:
            miner_result = MinerResultsText(
                hotkey=hotkey,
                test_loss=test_loss,
                synth_loss=test_loss,
                is_finetune=True,  # assume all finetuned
                task_type=task_type,
            )
        else:
            # For image tasks
            miner_result = MinerResultsImage(
                hotkey=hotkey,
                test_loss=test_loss,
                synth_loss=test_loss,
                is_finetune=True,
            )

        miner_results.append(miner_result)

    return miner_results

async def _get_scores_for_task(task_id: str, psql_db: PSQLDB) -> dict[str, float]:
    miner_results = await get_task_results_for_ranking(task_id, psql_db)
    if not miner_results:
        return {}

    ranked_results = calculate_miner_ranking_and_scores(miner_results)
    scores: dict[str, float] = {}
    for result in ranked_results:
        if result.adjusted_loss is None or np.isnan(result.adjusted_loss):
            continue
        scores[result.hotkey] = result.adjusted_loss
    return scores
