from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import update_task_node_quality_score_only
from validator.db.sql.tournaments import count_champion_consecutive_wins
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament import constants as t_cst
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentType
from validator.tournament.task_results import _get_scores_for_task
from validator.tournament.task_results import get_task_results_for_ranking


logger = get_logger(__name__)


def get_progressive_threshold(consecutive_wins: int, tournament_type: TournamentType | None = None) -> float:
    """
    Calculate the progressive threshold using exponential decay.
    """
    max_threshold = t_cst.EXPONENTIAL_BASE_THRESHOLD

    if tournament_type and tournament_type == TournamentType.ENVIRONMENT:
        max_threshold = t_cst.EXPONENTIAL_BASE_THRESHOLD_ENVIRONMENT

    current_threshold = max_threshold * (t_cst.EXPONENTIAL_DECAY_RATE ** (consecutive_wins - 1))
    return max(t_cst.EXPONENTIAL_MIN_THRESHOLD, current_threshold)

async def did_contender_beat_boss_on_task(
    task_id: str, contender_hotkey: str, threshold_percentage: float, psql_db: PSQLDB
) -> bool:
    """Return True if contender beats boss on this task by threshold (environment: higher is better)."""
    scores = await _get_scores_for_task(task_id, psql_db)
    contender_score = scores.get(contender_hotkey)
    boss_score = scores.get(EMISSION_BURN_HOTKEY)

    if contender_score is None:
        return False
    if boss_score is None:
        return True

    return contender_score >= boss_score * (1 + threshold_percentage)

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
            f"Threshold-adjusted winner at {threshold_pct:.1f}% progressive threshold"
            if is_winner
            else f"Lost to threshold-adjusted winner {winner_hotkey} at {threshold_pct:.1f}% progressive threshold"
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

async def select_best_contender_by_cumulative_boss_wins(
    tournament: TournamentData,
    candidate_hotkeys: list[str],
    psql_db: PSQLDB,
) -> str | None:
    """Select one contender using cumulative threshold-qualified wins vs boss.

    Uses all completed non-final rounds as the comparison horizon.
    Returns None when no contender has at least one threshold-qualified win.
    """
    if not candidate_hotkeys:
        return None

    boss_hotkey = EMISSION_BURN_HOTKEY
    non_boss_contenders = [h for h in candidate_hotkeys if h != boss_hotkey]
    if not non_boss_contenders:
        return None

    current_champion = tournament.base_winner_hotkey or boss_hotkey
    consecutive_wins = await count_champion_consecutive_wins(psql_db, tournament.tournament_type, current_champion)
    threshold_percentage = get_progressive_threshold(consecutive_wins, tournament.tournament_type)

    all_rounds = await get_tournament_rounds(tournament.tournament_id, psql_db)
    qualifying_rounds = [r for r in all_rounds if not r.is_final_round]
    qualifying_rounds.sort(key=lambda r: r.round_number)
    if not qualifying_rounds:
        logger.info("No completed non-final rounds found for contender selection.")
        return None

    up_to_round_number = qualifying_rounds[-1].round_number

    contender_wins: dict[str, int] = {contender: 0 for contender in non_boss_contenders}
    for contender in non_boss_contenders:
        for round_data in qualifying_rounds:
            round_tasks = await get_tournament_tasks(round_data.round_id, psql_db)
            for task in round_tasks:
                if await did_contender_beat_boss_on_task(task.task_id, contender, threshold_percentage, psql_db):
                    contender_wins[contender] += 1

    best_wins = max(contender_wins.values(), default=0)
    if best_wins <= 0:
        logger.info(
            f"No contender beat boss on any task in non-final rounds up to R{up_to_round_number} by threshold; "
            "returning no contender."
        )
        return None

    best_contenders = [h for h, wins in contender_wins.items() if wins == best_wins]
    if len(best_contenders) == 1:
        logger.info(
            f"Selected contender {best_contenders[0]} with {best_wins} wins "
            f"over boss in R1-R{up_to_round_number}"
        )
        return best_contenders[0]

    tie_break_round = next((r for r in qualifying_rounds if r.round_number == up_to_round_number), None)
    tie_break_scores: dict[str, float] = {}
    if tie_break_round:
        tie_break_tasks = await get_tournament_tasks(tie_break_round.round_id, psql_db)
        for contender in best_contenders:
            best_score = float("-inf")
            found = False
            for task in tie_break_tasks:
                scores = await _get_scores_for_task(task.task_id, psql_db)
                score = scores.get(contender)
                if score is None:
                    continue
                found = True
                best_score = max(best_score, score)
            tie_break_scores[contender] = best_score if found else float("-inf")

    selected = sorted(
        best_contenders,
        key=lambda contender: (tie_break_scores.get(contender, float("-inf")), contender),
        reverse=True,
    )[0]
    logger.info(
        f"Tie on R1-R{up_to_round_number} wins ({best_wins}) between {best_contenders}; "
        f"selected {selected} by round-{up_to_round_number} score / deterministic hotkey tiebreak."
    )
    return selected
