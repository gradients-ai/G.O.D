import statistics

import numpy as np

from core.models.tournament_models import TaskPerformanceDifference
from core.models.tournament_models import TournamentPerformanceData
from core.models.utility_models import TaskType
from validator.core import constants as cts
from validator.core.constants import EMISSION_BURN_HOTKEY
from validator.db.sql.tasks import get_task
from validator.db.sql.tournament_performance import get_boss_round_winner_task_pairs
from validator.db.sql.tournament_performance import get_task_scores_batch
from validator.db.sql.tournament_performance import update_tournament_winning_performance
from validator.db.sql.tournaments import count_champion_consecutive_wins_at_tournament
from validator.db.sql.tournaments import get_final_round_id
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_tasks
from validator.evaluation.scoring import calculate_miner_ranking_and_scores
from validator.tournament.utils import get_progressive_threshold
from validator.tournament.utils import get_task_results_for_ranking
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def calculate_boss_round_performance_differences(tournament_id: str, psql_db) -> list[TaskPerformanceDifference]:
    """Calculate performance differences for all tasks in a boss round."""

    tournament = await get_tournament(tournament_id, psql_db)
    if not tournament:
        logger.error(f"Tournament {tournament_id} not found")
        return []

    current_champion = tournament.base_winner_hotkey or EMISSION_BURN_HOTKEY
    consecutive_wins = await count_champion_consecutive_wins_at_tournament(
        psql_db, tournament.tournament_type, current_champion, tournament_id
    )
    threshold = get_progressive_threshold(consecutive_wins, tournament.tournament_type)

    logger.info(
        f"Calculating boss round performance for tournament {tournament_id}, champion {current_champion} "
        f"with {consecutive_wins} consecutive wins, threshold: {threshold * 100:.1f}%"
    )

    round_id = await get_final_round_id(tournament_id, psql_db)

    round_tasks = await get_tournament_tasks(round_id, psql_db)
    performance_differences = []

    for task in round_tasks:
        task_obj = await get_task(task.task_id, psql_db)
        if not task_obj:
            continue

        miner_results = await get_task_results_for_ranking(task.task_id, psql_db)
        if not miner_results:
            logger.warning(f"No results for task {task.task_id}")
            continue

        ranked_results = calculate_miner_ranking_and_scores(miner_results)

        is_higher_better = task_obj.task_type in [TaskType.GRPOTASK, TaskType.ENVIRONMENTTASK]

        boss_score = None
        challenger_score = None
        challenger_hotkey = None

        if task_obj.task_type == TaskType.ENVIRONMENTTASK:
            valid_participants = [
                (result.hotkey, result.adjusted_loss)
                for result in ranked_results
                if result.adjusted_loss is not None and not np.isnan(result.adjusted_loss)
            ]
            
            valid_participants.sort(key=lambda x: x[1], reverse=True) 

            for hotkey, score in valid_participants:
                if hotkey == EMISSION_BURN_HOTKEY:
                    boss_score = score
                    break
            
            if boss_score is None:
                logger.warning(f"Boss {current_champion} not found in scores for task {task.task_id}")
                continue

            boss_won = tournament.winner_hotkey == EMISSION_BURN_HOTKEY
            
            if boss_won:
                for hotkey, score in valid_participants:
                    if hotkey != EMISSION_BURN_HOTKEY:
                        challenger_score = score
                        challenger_hotkey = hotkey
                        break
            else:
                challenger_hotkey = valid_participants[0][0]
                challenger_score = valid_participants[0][1]
        else:
            for result in ranked_results:
                if result.hotkey == EMISSION_BURN_HOTKEY:
                    boss_score = result.adjusted_loss
                elif challenger_hotkey is None:
                    challenger_hotkey = result.hotkey
                    challenger_score = result.adjusted_loss

        if boss_score is None and challenger_score is None:
            logger.warning(f"Both boss and challenger missing scores for task {task.task_id}")
            continue

        if boss_score is None:
            logger.warning(f"Boss failed evaluation for task {task.task_id} - challenger wins by default")
            performance_differences.append(
                TaskPerformanceDifference(
                    task_id=str(task.task_id),
                    task_type=task_obj.task_type.value,
                    boss_score=None,
                    challenger_score=challenger_score,
                    threshold_used=threshold,
                    performance_difference=None,
                    challenger_won=True,
                )
            )
            continue

        if challenger_score is None:
            logger.warning(f"Challenger failed evaluation for task {task.task_id} - boss wins by default")
            performance_differences.append(
                TaskPerformanceDifference(
                    task_id=str(task.task_id),
                    task_type=task_obj.task_type.value,
                    boss_score=boss_score,
                    challenger_score=None,
                    threshold_used=threshold,
                    performance_difference=None,
                    challenger_won=False,
                )
            )
            continue

        if is_higher_better:
            if boss_score > 0:
                perf_diff = (challenger_score - boss_score) / boss_score
            else:
                perf_diff = 0.0
            challenger_won = challenger_score > boss_score * (1 + threshold)
        else:
            if challenger_score > 0:
                perf_diff = (boss_score - challenger_score) / challenger_score
            else:
                perf_diff = 0.0
            challenger_won = challenger_score < boss_score * (1 - threshold)

        performance_differences.append(
            TaskPerformanceDifference(
                task_id=str(task.task_id),
                task_type=task_obj.task_type.value,
                boss_score=boss_score,
                challenger_score=challenger_score,
                threshold_used=threshold,
                performance_difference=perf_diff,
                challenger_won=challenger_won,
            )
        )

        logger.info(
            f"Task {task.task_id}: Boss={current_champion} "
            f"({boss_score:.6f}), Challenger={challenger_hotkey} ({challenger_score:.6f}), "
            f"Diff={perf_diff * 100:.2f}%, Threshold={threshold * 100:.1f}%, Challenger won: {challenger_won}"
        )

    return performance_differences


async def get_tournament_performance_data(tournament_id: str, psql_db) -> list[TournamentPerformanceData]:
    """Get detailed performance data for tournament vs synthetic comparison."""
    task_pairs = await get_boss_round_winner_task_pairs(tournament_id, psql_db)
    logger.info(f"Found {len(task_pairs)} task pairs for performance comparison")

    if not task_pairs:
        return []

    # Collect all task IDs for batch fetching
    all_task_ids = []
    for task_pair in task_pairs:
        all_task_ids.append(task_pair.tournament_task_id)
        all_task_ids.append(task_pair.synthetic_task_id)

    # Batch fetch all scores
    all_scores = await get_task_scores_batch(all_task_ids, psql_db)

    performance_data = []

    for i, task_pair in enumerate(task_pairs):
        logger.info(
            f"Processing task pair {i + 1}/{len(task_pairs)}: tournament={task_pair.tournament_task_id}, synthetic={task_pair.synthetic_task_id}, winner={task_pair.winner_hotkey}"
        )

        tournament_scores = all_scores.get(task_pair.tournament_task_id, [])
        synthetic_scores = all_scores.get(task_pair.synthetic_task_id, [])
        logger.info(f"Found {len(tournament_scores)} tournament scores and {len(synthetic_scores)} synthetic scores")

        winner_tournament_score = None
        best_synthetic_score = None

        # Check if we need to use EMISSION_BURN_HOTKEY as the winner's placeholder
        winner_hotkey_for_lookup = task_pair.winner_hotkey
        # If the winner is not EMISSION_BURN_HOTKEY but EMISSION_BURN_HOTKEY participated in the task,
        # it means EMISSION_BURN_HOTKEY was acting as the defending champion's proxy
        emission_burn_in_scores = any(score.hotkey == EMISSION_BURN_HOTKEY for score in tournament_scores)
        if task_pair.winner_hotkey != EMISSION_BURN_HOTKEY and emission_burn_in_scores:
            winner_hotkey_for_lookup = EMISSION_BURN_HOTKEY
            logger.info(f"Using EMISSION_BURN_HOTKEY as placeholder for winner {task_pair.winner_hotkey}")

        for score in tournament_scores:
            if score.hotkey == winner_hotkey_for_lookup:
                winner_tournament_score = score.test_loss
                logger.info(f"Winner tournament score for {task_pair.winner_hotkey}: {winner_tournament_score}")
                break

        if synthetic_scores:
            task_type = TaskType(task_pair.task_type)

            if task_type in [TaskType.GRPOTASK, TaskType.ENVIRONMENTTASK]:
                best_synthetic_score = max(score.test_loss for score in synthetic_scores)
                logger.info(f"Best synthetic score (GRPO/Environment - higher is better): {best_synthetic_score}")
            else:
                best_synthetic_score = min(score.test_loss for score in synthetic_scores)
                logger.info(f"Best synthetic score (lower is better): {best_synthetic_score}")

        if winner_tournament_score is not None and best_synthetic_score is not None:
            task_type = TaskType(task_pair.task_type)
            logger.info(f"Task type: {task_type}")

            if task_type in [TaskType.GRPOTASK, TaskType.ENVIRONMENTTASK]:
                if best_synthetic_score > 0:
                    # For GRPO: higher is better, so positive diff means tournament is worse
                    performance_diff = (best_synthetic_score - winner_tournament_score) / best_synthetic_score
                else:
                    performance_diff = 0.0
            else:
                if best_synthetic_score > 0:
                    # For non-GRPO: lower is better, so positive diff means tournament is worse
                    performance_diff = (winner_tournament_score - best_synthetic_score) / best_synthetic_score
                else:
                    performance_diff = 0.0

            performance_data.append(
                TournamentPerformanceData(
                    tournament_task_id=str(task_pair.tournament_task_id),
                    synthetic_task_id=str(task_pair.synthetic_task_id),
                    task_type=task_pair.task_type,
                    tournament_winner_score=winner_tournament_score,
                    best_synthetic_score=best_synthetic_score,
                    performance_difference=performance_diff,
                )
            )

            logger.info(f"Performance difference for task pair {i + 1}: {performance_diff}")
        else:
            if winner_tournament_score is None and best_synthetic_score is not None:
                logger.warning(
                    f"Winner {task_pair.winner_hotkey} has no score in tournament task but synthetic miners do - applying max burn reduction"
                )
                performance_diff = cts.MAX_BURN_REDUCTION / cts.BURN_REDUCTION_RATE

                performance_data.append(
                    TournamentPerformanceData(
                        tournament_task_id=str(task_pair.tournament_task_id),
                        synthetic_task_id=str(task_pair.synthetic_task_id),
                        task_type=task_pair.task_type,
                        tournament_winner_score=0.0,
                        best_synthetic_score=best_synthetic_score,
                        performance_difference=performance_diff,
                    )
                )
            else:
                if winner_tournament_score is None:
                    logger.warning(f"Could not find winner {task_pair.winner_hotkey} score in tournament task for pair {i + 1}")
                if best_synthetic_score is None:
                    logger.warning(f"Could not find any scores in synthetic task for pair {i + 1}")

    return performance_data


async def calculate_performance_difference(tournament_id: str, psql_db) -> float:
    """
    Calculates median performance difference between tournament winner and runner up.
    """
    logger.info(f"=== CALCULATING PERFORMANCE DIFFERENCE FOR TOURNAMENT {tournament_id} ===")

    performance_data: list[TaskPerformanceDifference] = await calculate_boss_round_performance_differences(tournament_id, psql_db)

    if not performance_data:
        logger.info("No task pairs found, returning 0.0 performance difference")
        median_performance_diff = 0.0
    else:
        performance_differences = [
            data.performance_difference for data in performance_data if data.performance_difference is not None
        ]
        median_performance_diff = statistics.median(performance_differences) if performance_differences else 0.0
        logger.info(f"Median performance difference: {median_performance_diff} from {len(performance_differences)} task pairs")

    await update_tournament_winning_performance(tournament_id, median_performance_diff, psql_db)
    logger.info(f"Stored performance difference {median_performance_diff:.4f} in database for tournament {tournament_id}")

    return median_performance_diff
