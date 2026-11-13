import math
import statistics

import numpy as np

import validator.scoring.constants as cts
from core.logging import get_logger
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskType
from validator.db.sql.submissions_and_scoring import get_per_example_losses
from validator.db.sql.tasks import get_task
from validator.db.sql.tournament_performance import get_boss_round_winner_task_pairs
from validator.db.sql.tournament_performance import get_task_scores_batch
from validator.db.sql.tournament_performance import update_tournament_winning_performance
from validator.db.sql.tournaments import get_final_round_id
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_tasks
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.scoring.quant_ranking import QuantReferenceUnavailable
from validator.scoring.tasks import calculate_miner_ranking_and_scores
from validator.tournament import constants as t_cst
from validator.tournament.models import BossTaskAnalytics
from validator.tournament.models import PairedLossComparison
from validator.tournament.models import TaskPerformanceDifference
from validator.tournament.models import TournamentPerformanceData
from validator.tournament.task_results import get_task_results_for_ranking
from validator.tournament.thresholds import challenger_beats_boss
from validator.tournament.thresholds import challenger_takes_paired_task
from validator.tournament.thresholds import challenger_within_freeform_allowed_loss
from validator.tournament.thresholds import challenger_within_paired_allowed_loss
from validator.tournament.thresholds import compare_paired_losses
from validator.tournament.thresholds import quant_challenger_takes_fixed_boss_task
from validator.tournament.thresholds import resolve_freeform_boss_task
from validator.tournament.thresholds import sku_aligned_losses


logger = get_logger(__name__)


def _freeform_analytics(
    boss_result: MinerResultsText | None, challenger_result: MinerResultsText | None
) -> BossTaskAnalytics:
    """Mirror of the freeform crowning decision: the cap, then effective bytes with QUANT_FREEFORM_BOSS_BYTE_MARGIN."""
    verdict = resolve_freeform_boss_task(boss_result, challenger_result, t_cst.QUANT_FREEFORM_BOSS_BYTE_MARGIN)
    challenger_won = verdict.winner_hotkey != EMISSION_BURN_HOTKEY
    return BossTaskAnalytics(
        challenger_won=challenger_won,
        reason=verdict.decided_by,
        within_allowed_loss=None if challenger_won else challenger_within_freeform_allowed_loss(boss_result, challenger_result),
    )


async def _quant_challenger_won_task(
    task_id,
    task_obj,
    challenger_hotkey: str,
    boss_score: float,
    challenger_score: float,
    threshold: float,
    psql_db,
    boss_result: MinerResultsText | None = None,
    challenger_result: MinerResultsText | None = None,
) -> BossTaskAnalytics:
    """Mirror of the crowning decision for a single quant boss-round task.

    Freeform is decided on the cap and effective bytes, as crowning does. Fixed uses the paired per-example
    comparison on vectors aligned onto the boss's SKU; where either side has no vector the boss keeps the task.
    within_allowed_loss comes from the aligned vectors the verdict used, through the function crowning uses.
    """
    if task_obj.task_type == TaskType.QUANTFREEFORMTASK:
        return _freeform_analytics(boss_result, challenger_result)

    boss_losses, boss_fingerprint = await get_per_example_losses(task_id, EMISSION_BURN_HOTKEY, psql_db)
    challenger_losses, challenger_fingerprint = await get_per_example_losses(task_id, challenger_hotkey, psql_db)

    pairable = (
        boss_losses is not None
        and challenger_losses is not None
        and len(boss_losses) == len(challenger_losses)
        and not (
            boss_fingerprint is not None and challenger_fingerprint is not None and boss_fingerprint != challenger_fingerprint
        )
    )
    if pairable:
        try:
            challenger_losses = await sku_aligned_losses(
                task_obj.task_type,
                task_id,
                EMISSION_BURN_HOTKEY,
                challenger_hotkey,
                challenger_losses,
                psql_db,
                task_obj.training_start_point,
            )
        except QuantReferenceUnavailable as e:
            logger.error(f"Task {task_id}: {e} - not counted as a win")
            return BossTaskAnalytics(
                challenger_won=False,
                reason="No same-SKU reference to pair the two quant vectors on",
                within_allowed_loss=False,
            )
        comparison = compare_paired_losses(boss_losses, challenger_losses)
        challenger_won, reason = quant_challenger_takes_fixed_boss_task(comparison, boss_score, challenger_score)
        within = None
        if not challenger_won:
            # A draw is within: the swapped comparison decides nothing either.
            within, _ = challenger_within_paired_allowed_loss(boss_losses, challenger_losses, boss_score, challenger_score)
        return BossTaskAnalytics(
            challenger_won=challenger_won, paired_comparison=comparison, reason=reason, within_allowed_loss=within
        )

    if boss_losses is not None and challenger_losses is not None:
        logger.error(f"Task {task_id}: per-example vectors present but not pairable - not counted as a win")
        return BossTaskAnalytics(
            challenger_won=False,
            reason="Per-example vectors could not be paired - the two models were scored on different data",
            within_allowed_loss=False,
        )

    if (boss_losses is None or challenger_losses is None) and not math.isfinite(boss_score) and math.isfinite(challenger_score):
        # A boss entry that is a scored fail has no vector: a challenger with a valid loss takes the task, as in crowning.
        return BossTaskAnalytics(challenger_won=True, reason="The boss's entry is a scored fail", within_allowed_loss=None)

    # The paired test is part of the rule: without both vectors it cannot be run, so the boss keeps the task.
    logger.error(f"Task {task_id}: no per-example vectors for performance calc - not counted as a win")
    return BossTaskAnalytics(
        challenger_won=False, reason="No per-example losses to pair", within_allowed_loss=False
    )


async def _challenger_won_task(
    task_id,
    task_obj,
    challenger_hotkey: str,
    boss_score: float,
    challenger_score: float,
    threshold: float,
    psql_db,
) -> tuple[bool, PairedLossComparison | None, str | None]:
    """Mirror of the crowning decision for a single lower-is-better boss-round task.

    Uses the paired per-example comparison where both sides have vectors, and the relative margin
    otherwise - the same order of preference as _resolve_boss_round_task_winner in round_results.

    Returns (verdict, comparison, reason). The comparison and reason are what analytics reports;
    both are None when the relative margin decided the task. They are reported separately because
    the verdict is not comparison.challenger_won: the KL scalar gate can overturn a won paired
    comparison, and the reason is the only thing that says so.
    """
    if task_obj.task_type in t_cst.PAIRED_BOSS_ROUND_TASK_TYPES:
        boss_losses, boss_fingerprint = await get_per_example_losses(task_id, EMISSION_BURN_HOTKEY, psql_db)
        challenger_losses, challenger_fingerprint = await get_per_example_losses(task_id, challenger_hotkey, psql_db)

        pairable = (
            boss_losses is not None
            and challenger_losses is not None
            and len(boss_losses) == len(challenger_losses)
            and not (
                boss_fingerprint is not None
                and challenger_fingerprint is not None
                and boss_fingerprint != challenger_fingerprint
            )
        )
        if pairable:
            comparison = compare_paired_losses(boss_losses, challenger_losses)
            challenger_won, reason = challenger_takes_paired_task(comparison, boss_score, challenger_score)
            return challenger_won, comparison, reason

        if boss_losses is not None and challenger_losses is not None:
            # Present but unpairable: crowning does not count these as a win, so report the same
            # to keep the emission projection consistent with the result.
            logger.error(f"Task {task_id}: per-example vectors present but not pairable - not counted as a win")
            return False, None, "Per-example vectors could not be paired - the two models were scored on different data"

        logger.warning(f"Task {task_id}: no per-example vectors for performance calc - using the relative margin")

    return challenger_beats_boss(boss_score, challenger_score, False, threshold), None, None


async def calculate_boss_round_performance_differences(tournament_id: str, psql_db) -> list[TaskPerformanceDifference]:
    """Calculate performance differences for all tasks in a boss round."""

    tournament = await get_tournament(tournament_id, psql_db)
    if not tournament:
        logger.error(f"Tournament {tournament_id} not found")
        return []

    current_champion = tournament.base_winner_hotkey or EMISSION_BURN_HOTKEY
    # Report challenger_won using the same margin crowning uses, so analytics match reality.
    threshold = t_cst.BOSS_ROUND_WIN_MARGIN

    logger.info(
        f"Calculating boss round performance for tournament {tournament_id}, champion {current_champion}, "
        f"threshold: {threshold * 100:.1f}%"
    )

    round_id = await get_final_round_id(tournament_id, psql_db)
    all_round_tasks = await get_tournament_tasks(round_id, psql_db)

    performance_differences = []

    for task in all_round_tasks:
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
        boss_result = None
        challenger_result = None

        if task_obj.task_type == TaskType.ENVIRONMENTTASK:
            valid_participants = [
                (result.hotkey, result.adjusted_loss)
                for result in ranked_results
                if result.adjusted_loss is not None and not np.isnan(result.adjusted_loss)
            ]

            valid_participants.sort(key=lambda x: x[1], reverse=True)

            scores_by_hotkey = {hotkey: score for hotkey, score in valid_participants}
            boss_score = scores_by_hotkey.get(EMISSION_BURN_HOTKEY)

            if boss_score is None:
                logger.warning(f"Boss {current_champion} not found in scores for task {task.task_id}")
                continue

            boss_won = tournament.winner_hotkey == EMISSION_BURN_HOTKEY

            if boss_won:
                # Boss won tournament — pick the best non-boss challenger in this round
                for hotkey, score in valid_participants:
                    if hotkey != EMISSION_BURN_HOTKEY:
                        challenger_score = score
                        challenger_hotkey = hotkey
                        break
            else:
                # Challenger won tournament — look up the actual winner's score in this round
                challenger_hotkey = tournament.winner_hotkey
                challenger_score = scores_by_hotkey.get(challenger_hotkey)
                if challenger_score is None:
                    # Winner didn't participate in this round (e.g. wasn't in R1 yet) — skip
                    logger.info(f"Tournament winner {challenger_hotkey} has no score in task {task.task_id}, skipping")
                    continue
        else:
            for result in ranked_results:
                if result.hotkey == EMISSION_BURN_HOTKEY:
                    boss_score = result.adjusted_loss
                    boss_result = result
                elif challenger_hotkey is None:
                    challenger_hotkey = result.hotkey
                    challenger_score = result.adjusted_loss
                    challenger_result = result

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
                    # A failed score is never within the allowed loss.
                    within_allowed_loss=False if task_obj.task_type in QUANT_TASK_TYPES else None,
                )
            )
            continue

        paired_comparison = None
        decision_reason = None
        within_allowed_loss = None
        if task_obj.task_type == TaskType.ENVIRONMENTTASK:
            # Local import breaks the weights <-> performance_calculator import cycle.
            from validator.scoring.weights import calculate_env_perf_diff_from_win_pct

            num_envs = len(task_obj.environment_names) if task_obj.environment_names else 1
            win_pct = (2 * challenger_score + boss_score - 3 * num_envs) / (3 * num_envs)
            perf_diff = calculate_env_perf_diff_from_win_pct(win_pct)
            challenger_won = challenger_score > boss_score
        elif is_higher_better:
            if boss_score != 0:
                perf_diff = (challenger_score - boss_score) / abs(boss_score)
            else:
                perf_diff = 0.0
            challenger_won = challenger_beats_boss(boss_score, challenger_score, True, threshold)
        else:
            if challenger_score != 0:
                perf_diff = (boss_score - challenger_score) / abs(challenger_score)
            else:
                perf_diff = 0.0
            # perf_diff stays a percentage - it scales emissions and is not a gate. challenger_won
            # is the gate, so it has to be decided the same way crowning decided it or analytics
            # and the emission projection contradict the actual result.
            if task_obj.task_type in QUANT_TASK_TYPES:
                analytics = await _quant_challenger_won_task(
                    task_id=task.task_id,
                    task_obj=task_obj,
                    challenger_hotkey=challenger_hotkey,
                    boss_score=boss_score,
                    challenger_score=challenger_score,
                    threshold=threshold,
                    psql_db=psql_db,
                    boss_result=boss_result,
                    challenger_result=challenger_result,
                )
                challenger_won = analytics.challenger_won
                paired_comparison = analytics.paired_comparison
                decision_reason = analytics.reason
                within_allowed_loss = analytics.within_allowed_loss
            else:
                challenger_won, paired_comparison, decision_reason = await _challenger_won_task(
                    task_id=task.task_id,
                    task_obj=task_obj,
                    challenger_hotkey=challenger_hotkey,
                    boss_score=boss_score,
                    challenger_score=challenger_score,
                    threshold=threshold,
                    psql_db=psql_db,
                )

        # A failed-but-scored entry carries an infinite loss; the ratio is then inf or nan, and it feeds the emission boost.
        # Who won is already decided above, so only the number is dropped.
        recorded_perf_diff = perf_diff if math.isfinite(boss_score) and math.isfinite(challenger_score) else None
        if recorded_perf_diff is None:
            logger.warning(f"Task {task.task_id}: non-finite score (boss={boss_score}, challenger={challenger_score}), no diff")

        performance_differences.append(
            TaskPerformanceDifference(
                task_id=str(task.task_id),
                task_type=task_obj.task_type.value,
                boss_score=boss_score,
                challenger_score=challenger_score,
                threshold_used=threshold,
                performance_difference=recorded_perf_diff,
                challenger_won=challenger_won,
                paired_comparison=paired_comparison,
                decision_reason=decision_reason,
                within_allowed_loss=within_allowed_loss,
            )
        )

        diff_text = "n/a" if recorded_perf_diff is None else f"{recorded_perf_diff * 100:.2f}%"
        logger.info(
            f"Task {task.task_id}: Boss={current_champion} "
            f"({boss_score:.6f}), Challenger={challenger_hotkey} ({challenger_score:.6f}), "
            f"Diff={diff_text}, Threshold={threshold * 100:.1f}%, Challenger won: {challenger_won}"
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
            f"Processing task pair {i + 1}/{len(task_pairs)}: tournament={task_pair.tournament_task_id}, "
            f"synthetic={task_pair.synthetic_task_id}, winner={task_pair.winner_hotkey}"
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
                    f"Winner {task_pair.winner_hotkey} has no score in tournament task but synthetic miners do "
                    "- applying max burn reduction"
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
            data.performance_difference
            for data in performance_data
            if data.performance_difference is not None and math.isfinite(data.performance_difference)
        ]
        median_performance_diff = statistics.median(performance_differences) if performance_differences else 0.0
        if not math.isfinite(median_performance_diff):
            logger.warning(f"Non-finite median performance difference {median_performance_diff}, using 0.0")
            median_performance_diff = 0.0
        logger.info(f"Median performance difference: {median_performance_diff} from {len(performance_differences)} task pairs")

    await update_tournament_winning_performance(tournament_id, median_performance_diff, psql_db)
    logger.info(f"Stored performance difference {median_performance_diff:.4f} in database for tournament {tournament_id}")

    return median_performance_diff
