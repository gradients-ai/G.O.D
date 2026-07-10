from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import update_task_node_quality_score_only
from validator.tournament.task_results import get_task_results_for_ranking


logger = get_logger(__name__)


def challenger_beats_boss(boss_loss: float, challenger_loss: float, higher_is_better: bool, margin: float) -> bool:
    """Return True if the challenger beats the boss by at least `margin` on a task.

    The margin is applied additively on the magnitude of the boss score so it stays
    correct for zero/negative scores (GRPO rewards can go negative via the KL penalty):
      higher-is-better: challenger >= boss + abs(boss) * margin
      lower-is-better:  challenger <= boss - abs(boss) * margin
    """
    bar = abs(boss_loss) * margin
    if higher_is_better:
        return challenger_loss >= boss_loss + bar
    return challenger_loss <= boss_loss - bar


async def update_threshold_adjusted_quality_scores_for_task(
    task_id: str,
    winner_hotkey: str,
    threshold_percentage: float,
    psql_db: PSQLDB,
    compared_hotkeys: list[str] | None = None,
) -> None:
    """Persist threshold-adjusted task scores while preserving raw losses."""
    miner_results = await get_task_results_for_ranking(task_id, psql_db)
    if not miner_results:
        logger.warning(f"No valid results for threshold-adjusted scoring on task {task_id}")
        return

    allowed_hotkeys = set(compared_hotkeys) if compared_hotkeys else None
    scored_hotkeys = {result.hotkey for result in miner_results if allowed_hotkeys is None or result.hotkey in allowed_hotkeys}
    if winner_hotkey not in scored_hotkeys:
        logger.warning(
            f"Threshold-adjusted winner {winner_hotkey} not found in valid results for task {task_id}; skipping score update"
        )
        return

    threshold_pct = threshold_percentage * 100
    for result in miner_results:
        if allowed_hotkeys is not None and result.hotkey not in allowed_hotkeys:
            continue

        is_winner = result.hotkey == winner_hotkey
        quality_score = 3.0 if is_winner else 0.0
        score_reason = (
            f"Winner at {threshold_pct:.1f}% boss-round win margin"
            if is_winner
            else f"Lost to winner {winner_hotkey} at {threshold_pct:.1f}% boss-round win margin"
        )
        await update_task_node_quality_score_only(
            task_id=task_id,
            hotkey=result.hotkey,
            quality_score=quality_score,
            score_reason=score_reason,
            psql_db=psql_db,
        )

    logger.info(
        f"Updated threshold-adjusted quality scores for task {task_id}: winner={winner_hotkey}, "
        f"threshold={threshold_pct:.1f}%"
    )
