import numpy as np

from core.logging import get_logger
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskType
from validator.db import constants as db_cst
from validator.db.database import PSQLDB
from validator.db.sql.quant_references import get_quant_sku_references
from validator.db.sql.submissions_and_scoring import get_all_scores_and_losses_for_task
from validator.db.sql.tasks import get_task
from validator.scoring.models import MinerResultsImage
from validator.scoring.models import MinerResultsText
from validator.scoring.models import QuantSkuReferences
from validator.scoring.quant_ranking import eval_gpu_from_column
from validator.scoring.quant_ranking import reference_on
from validator.scoring.tasks import calculate_miner_ranking_and_scores
from validator.tasks.models import AnyQuantRawTask
from validator.tournament.quant_reference import reference_hotkey_for


logger = get_logger(__name__)


def _quant_scored_fail(hotkey: str, task_type: TaskType, score_dict: dict) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=np.nan,
        synth_loss=np.nan,
        is_finetune=True,
        task_type=task_type,
        quant_failure=score_dict[db_cst.QUANT_FAILURE],
        artifact_bytes=score_dict.get(db_cst.ARTIFACT_BYTES),
        decoder_bytes=score_dict.get(db_cst.DECODER_BYTES),
        passed_bar=score_dict.get(db_cst.PASSED_BAR),
        freeform_penalty=score_dict.get(db_cst.FREEFORM_PENALTY),
        eval_gpu=eval_gpu_from_column(score_dict.get(db_cst.EVAL_GPU)),
    )


def _quant_result(
    task: AnyQuantRawTask, hotkey: str, score_dict: dict, references: QuantSkuReferences
) -> MinerResultsText:
    """A scored quant row, carrying its own reference loss on its own SKU so ranking compares normalised losses."""
    eval_gpu = eval_gpu_from_column(score_dict.get(db_cst.EVAL_GPU))
    reference = reference_on(reference_hotkey_for(task, hotkey), eval_gpu, references)
    test_loss = score_dict[db_cst.TEST_LOSS]
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=test_loss,
        synth_loss=test_loss,
        is_finetune=True,
        task_type=task.task_type,
        artifact_bytes=score_dict.get(db_cst.ARTIFACT_BYTES),
        decoder_bytes=score_dict.get(db_cst.DECODER_BYTES),
        passed_bar=score_dict.get(db_cst.PASSED_BAR),
        freeform_penalty=score_dict.get(db_cst.FREEFORM_PENALTY),
        eval_gpu=eval_gpu,
        reference_loss=reference.loss if reference else None,
    )


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
    references = await get_quant_sku_references(task_id, psql_db) if task_type in QUANT_TASK_TYPES else {}

    miner_results = []
    for score_dict in scores_dicts:
        hotkey = score_dict[db_cst.HOTKEY]
        test_loss = score_dict.get(db_cst.TEST_LOSS)

        # A quant scored fail has no loss but still ranks (last); it is not an error.
        if task_type in QUANT_TASK_TYPES and score_dict.get(db_cst.QUANT_FAILURE):
            miner_results.append(_quant_scored_fail(hotkey, task_type, score_dict))
            continue

        # Skip invalid results
        if test_loss is None or np.isnan(test_loss):
            continue

        # Create appropriate MinerResults object
        if task_type in QUANT_TASK_TYPES:
            miner_result = _quant_result(task_object, hotkey, score_dict, references)
        elif task_type in [
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
