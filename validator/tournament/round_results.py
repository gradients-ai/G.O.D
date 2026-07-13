from collections import Counter

import numpy as np

from core.constants.environments import EnvironmentName
from core.logging import get_logger
from core.models.task_models import TaskType
from validator.app.config import Config
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import get_task_winner
from validator.db.sql.tasks import get_task
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.db.sql.tournaments import get_training_status_for_task_and_hotkeys
from validator.evaluation.pvp.models import GameOutcome
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.tasks import calculate_miner_ranking_and_scores
from validator.tournament import constants as t_cst
from validator.tournament.models import GroupMatchStanding
from validator.tournament.models import MatchRanking
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentResultsWithWinners
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus
from validator.tournament.task_results import _get_scores_for_task
from validator.tournament.task_results import get_task_results_for_ranking
from validator.tournament.thresholds import challenger_beats_boss
from validator.tournament.thresholds import update_threshold_adjusted_quality_scores_for_task


logger = get_logger(__name__)


def _task_includes_environment(task_object, environment: EnvironmentName) -> bool:
    env_names = getattr(task_object, "environment_names", []) if task_object else []
    for env_name in env_names or []:
        try:
            if EnvironmentName(getattr(env_name, "value", env_name)) == environment:
                return True
        except ValueError:
            continue
    return False


async def determine_env_tournament_winner(
    tournament: TournamentData, _finalists: list[str], _config: Config, psql_db: PSQLDB,
) -> list[str]:
    """Determine environment winner from boss round only.

    Single contender must have no boss-round losses and a strict win on the
    required SWE Infinite final task. If not, boss retains.
    """
    boss_hotkey = EMISSION_BURN_HOTKEY

    all_rounds = await get_tournament_rounds(tournament.tournament_id, psql_db)
    if not all_rounds:
        return [boss_hotkey]

    final_round = next((r for r in all_rounds if r.is_final_round), None)
    if not final_round:
        logger.warning("No final round found for environment tournament; boss wins by default")
        return [boss_hotkey]

    final_tasks = await get_tournament_tasks(final_round.round_id, psql_db)
    if not final_tasks:
        logger.warning("No boss round tasks found; boss wins by default")
        return [boss_hotkey]

    # Identify the single contender from boss round scores
    contender: str | None = None
    for task in final_tasks:
        scores = await _get_scores_for_task(task.task_id, psql_db)
        for hotkey in scores:
            if hotkey != boss_hotkey:
                contender = hotkey
                break
        if contender:
            break

    if not contender:
        logger.info("No contender found in boss round; boss wins by default")
        return [boss_hotkey]

    # Contender must have zero losses and at least one win.
    # Draws are acceptable but any single loss means boss retains.
    wins = 0
    losses = 0
    draws = 0
    saw_required_swe_task = False
    won_required_swe_task = False
    for task in final_tasks:
        task_object = await get_task(task.task_id, psql_db)
        is_swe_task = _task_includes_environment(task_object, EnvironmentName.SWE_INFINITE)
        if is_swe_task:
            saw_required_swe_task = True

        scores = await _get_scores_for_task(task.task_id, psql_db)
        contender_score = scores.get(contender)
        boss_score = scores.get(boss_hotkey)

        if contender_score is None:
            logger.info(f"Contender {contender} has no score on task {task.task_id}; boss retains")
            return [boss_hotkey, contender]

        bs = boss_score if boss_score is not None else 0
        if contender_score > bs:
            outcome = GameOutcome.WIN
            wins += 1
            if is_swe_task:
                won_required_swe_task = True
        elif contender_score < bs:
            outcome = GameOutcome.LOSS
            losses += 1
        else:
            outcome = GameOutcome.DRAW
            draws += 1
        logger.info(
            f"Boss round task {task.task_id}: contender={contender_score:.2f} boss={bs:.2f} -> {outcome.value} "
            f"(running: W={wins} D={draws} L={losses}, required_swe_win={won_required_swe_task})"
        )

    if not saw_required_swe_task:
        logger.warning("Boss retains: final round is missing the required SWE Infinite task")
        return [boss_hotkey, contender]

    if losses == 0 and won_required_swe_task:
        logger.info(
            f"Contender {contender} wins environment tournament: "
            f"W={wins} D={draws} L={losses}, required_swe_win={won_required_swe_task} "
            f"across {len(final_tasks)} tasks"
        )
        return [contender, boss_hotkey]
    else:
        logger.info(
            f"Boss retains: contender W={wins} D={draws} L={losses} "
            f"required_swe_win={won_required_swe_task} across {len(final_tasks)} tasks "
            "(need zero losses and a strict SWE Infinite win)"
        )
        return [boss_hotkey, contender]

def get_real_winner_hotkey(winner_hotkey: str | None, base_winner_hotkey: str | None) -> str | None:
    """
    Get the real hotkey of the tournament winner.

    If winner_hotkey is EMISSION_BURN_HOTKEY (defending champion defended),
    returns base_winner_hotkey (the real defending champion's hotkey).
    Otherwise returns winner_hotkey.

    This is needed because when a defending champion successfully defends,
    winner_hotkey is set to EMISSION_BURN_HOTKEY as a placeholder, and
    base_winner_hotkey contains their actual hotkey.

    Args:
        winner_hotkey: The tournament's winner_hotkey field
        base_winner_hotkey: The tournament's base_winner_hotkey field (defending champion snapshot)

    Returns:
        Real winner's hotkey, or None if no winner
    """
    if not winner_hotkey:
        return None

    if winner_hotkey == EMISSION_BURN_HOTKEY and base_winner_hotkey:
        return base_winner_hotkey

    return winner_hotkey

def get_real_tournament_winner(tournament: TournamentData | TournamentResultsWithWinners | None) -> str | None:
    """
    Get the real tournament winner hotkey, accounting for EMISSION_BURN_HOTKEY.

    When a defending champion wins, winner_hotkey is set to EMISSION_BURN_HOTKEY,
    and the actual winner hotkey is stored in base_winner_hotkey.
    """
    if not tournament:
        return None
    return get_real_winner_hotkey(tournament.winner_hotkey, tournament.base_winner_hotkey)

def did_winner_change(previous_tournament: TournamentData | None, latest_tournament: TournamentData) -> bool:
    """
    Determine if the tournament winner changed between two tournaments.

    Returns True if:
    - No previous tournament exists (first tournament)
    - Latest winner is a real hotkey (not EMISSION_BURN_HOTKEY)

    Returns:
        True if winner should be treated as a new winner, False if defending champion won via placeholder
    """
    if not previous_tournament:
        return True

    # EMISSION_BURN_HOTKEY explicitly marks a defending champion win.
    # Any real hotkey winner should be treated as "new winner" for fresh perf diff calc,
    # even if it's the same hotkey as a previous tournament.
    if latest_tournament.winner_hotkey != EMISSION_BURN_HOTKEY:
        return True

    return False

def determine_boss_round_winner(
    task_winners: list[str],
    boss_hotkey: str,
    tournament_type: TournamentType,
    continuous_sft_winners: list[str] | None = None,
    num_continuous_sft_tasks: int = 0,
) -> str:
    """
    Determine the winner of a boss round based on task results and tournament type.

    Args:
        task_winners: List of hotkeys that won each task in the boss round
        boss_hotkey: The defending champion's hotkey
        tournament_type: Type of tournament (TEXT or IMAGE)
        continuous_sft_winners: Hotkeys that won each *decided* continuous-SFT task.
        num_continuous_sft_tasks: Total continuous-SFT tasks (decided or not). When >0 (text boss
            round), the challenger must win EVERY one to dethrone, on top of the overall threshold —
            so a failed/skipped continuous-SFT task blocks the dethrone. 0 (image) leaves the rule off.

    Returns:
        Hotkey of the boss round winner
    """
    if not task_winners:
        logger.error("No valid task winners found in boss round - all tasks failed to determine winners")
        logger.info(f"Defaulting to boss as winner due to evaluation failures: {boss_hotkey}")
        return boss_hotkey

    # Count wins for each contestant
    win_counts = Counter(task_winners)
    total_tasks = len(task_winners)

    # Find the opponent (non-boss hotkey)
    opponent_hotkey = None
    for hotkey in win_counts.keys():
        if hotkey != boss_hotkey:
            opponent_hotkey = hotkey
            break

    opponent_wins = win_counts.get(opponent_hotkey, 0) if opponent_hotkey else 0

    # Both IMAGE and TEXT tournaments: challenger may lose at most one
    # boss-round task. Each task requires beating the boss by BOSS_ROUND_WIN_MARGIN.
    required_wins = max(1, total_tasks - 1)

    # Continuous-SFT gate: challenger must win EVERY continuous-SFT task; only enforced when >0.
    challenger_continuous_wins = (
        (continuous_sft_winners or []).count(opponent_hotkey) if opponent_hotkey else 0
    )
    continuous_sft_ok = True
    if num_continuous_sft_tasks > 0:
        continuous_sft_ok = challenger_continuous_wins == num_continuous_sft_tasks

    if opponent_hotkey and opponent_wins >= required_wins and continuous_sft_ok:
        logger.info(
            f"{tournament_type.value} tournament: Challenger wins boss round comprehensively: "
            f"{opponent_wins}/{total_tasks} tasks won (required {required_wins})"
            + (
                f", won all {num_continuous_sft_tasks} continuous-SFT tasks"
                if num_continuous_sft_tasks > 0
                else ""
            )
        )
        return opponent_hotkey
    else:
        boss_wins = win_counts.get(boss_hotkey, 0)
        if opponent_hotkey and opponent_wins >= required_wins and not continuous_sft_ok:
            logger.info(
                f"{tournament_type.value} tournament: Boss retains title - challenger won "
                f"{opponent_wins}/{total_tasks} tasks but only {challenger_continuous_wins}/"
                f"{num_continuous_sft_tasks} continuous-SFT tasks (must win ALL to dethrone)"
            )
        elif opponent_hotkey:
            logger.info(
                f"{tournament_type.value} tournament: Boss retains title - challenger won "
                f"{opponent_wins}/{total_tasks} tasks (requires {required_wins}/{total_tasks} to dethrone), "
                f"boss won {boss_wins}/{total_tasks}"
            )
        else:
            logger.info(f"{tournament_type.value} tournament: Boss retains title by default")
        return boss_hotkey

async def get_knockout_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB, config: Config
) -> list[str]:
    """Get winners from knockout round."""
    winners = []

    if not completed_round.is_final_round:
        # Use simple quality score comparison for regular knockout rounds
        for task in round_tasks:
            winner = await get_task_winner(task.task_id, psql_db)
            if winner:
                winners.append(winner)
    else:
        # Boss round. Challenger must beat the boss by BOSS_ROUND_WIN_MARGIN per task.
        boss_hotkey = EMISSION_BURN_HOTKEY
        opponent_hotkey = None
        task_winners = []
        # Decided continuous-SFT winners + total count, fed to determine_boss_round_winner's
        # "challenger must win ALL continuous-SFT tasks" dethrone gate.
        continuous_sft_winners: list[str] = []
        num_continuous_sft_tasks = 0

        def _is_continuous_sft(task_obj) -> bool:
            return task_obj is not None and t_cst.is_continuous_sft_task(task_obj)

        def _award(winner: str | None, task_obj) -> None:
            task_winners.append(winner)
            if winner is not None and _is_continuous_sft(task_obj):
                continuous_sft_winners.append(winner)

        # Get tournament info to determine the current champion and their consecutive wins
        tournament = await get_tournament(completed_round.tournament_id, psql_db)
        if not tournament:
            logger.error(f"Could not find tournament {completed_round.tournament_id}")
            return []

        # Challenger must beat the boss by BOSS_ROUND_WIN_MARGIN on each task to win it.
        threshold_percentage = t_cst.BOSS_ROUND_WIN_MARGIN
        logger.info(f"Boss round using {threshold_percentage * 100:.1f}% win margin per task")

        for task in round_tasks:
            logger.info(f"Processing boss round task {task.task_id}")

            task_object = await get_task(task.task_id, psql_db)

            # Count even undecided ones, so a failed/skipped continuous-SFT task still blocks the dethrone.
            if _is_continuous_sft(task_object):
                num_continuous_sft_tasks += 1

            miner_results = await get_task_results_for_ranking(task.task_id, psql_db)
            if not miner_results:
                logger.warning(f"No valid results for boss round task {task.task_id}. Winner is base contestant.")
                _award(boss_hotkey, task_object)
                continue

            ranked_results = calculate_miner_ranking_and_scores(miner_results)

            boss_loss = None
            opponent_loss = None
            opponent_hotkey = None

            for result in ranked_results:
                if result.hotkey == boss_hotkey:
                    boss_loss = result.adjusted_loss
                else:
                    if opponent_hotkey is None:
                        opponent_hotkey = result.hotkey
                        opponent_loss = result.adjusted_loss

            if boss_loss is None or opponent_loss is None:
                logger.warning(f"Boss round task {task.task_id} missing boss or opponent loss")
                # Check training status to determine winner when evaluation results are missing
                training_statuses = await get_training_status_for_task_and_hotkeys(
                    task.task_id, [boss_hotkey, opponent_hotkey], psql_db
                )

                boss_training_success = training_statuses.get(boss_hotkey) == TrainingStatus.SUCCESS
                opponent_training_success = training_statuses.get(opponent_hotkey) == TrainingStatus.SUCCESS

                if opponent_training_success and not boss_training_success:
                    logger.info(f"Boss training failed, opponent succeeded - opponent wins task {task.task_id}")
                    _award(opponent_hotkey, task_object)
                elif boss_training_success and not opponent_training_success:
                    logger.info(f"Opponent training failed, boss succeeded - boss wins task {task.task_id}")
                    _award(boss_hotkey, task_object)
                elif not boss_training_success and not opponent_training_success:
                    logger.info(f"Both training failed - boss wins by default for task {task.task_id}")
                    _award(boss_hotkey, task_object)
                else:
                    # Both training succeeded but at least one has missing/invalid evaluation results
                    # Check who has valid evaluation results and award to them
                    boss_has_valid_eval = boss_loss is not None
                    opponent_has_valid_eval = opponent_loss is not None

                    if opponent_has_valid_eval and not boss_has_valid_eval:
                        logger.info(f"Boss evaluation failed, opponent succeeded - opponent wins task {task.task_id}")
                        _award(opponent_hotkey, task_object)
                    elif boss_has_valid_eval and not opponent_has_valid_eval:
                        logger.info(f"Opponent evaluation failed, boss succeeded - boss wins task {task.task_id}")
                        _award(boss_hotkey, task_object)
                    else:
                        logger.warning(
                            f"Both evaluation failed or both succeeded but missing results - skipping task {task.task_id}"
                        )
                continue

            logger.info(f"Boss round task {task.task_id}: Boss loss: {boss_loss:.6f}, Opponent loss: {opponent_loss:.6f}")

            higher_is_better = task_object.task_type in (TaskType.GRPOTASK, TaskType.ENVIRONMENTTASK)
            if challenger_beats_boss(boss_loss, opponent_loss, higher_is_better, threshold_percentage):
                task_winner = opponent_hotkey
            else:
                task_winner = boss_hotkey
            _award(task_winner, task_object)
            direction = "higher is better" if higher_is_better else "lower is better"
            winner_label = "opponent" if task_winner == opponent_hotkey else "boss"
            logger.info(
                f"{task_object.task_type} task ({direction}): {winner_label} wins at {threshold_percentage * 100:.1f}% "
                f"margin (boss={boss_loss:.6f}, opponent={opponent_loss:.6f})"
            )

            await update_threshold_adjusted_quality_scores_for_task(
                task_id=task.task_id,
                winner_hotkey=task_winner,
                threshold_percentage=threshold_percentage,
                compared_hotkeys=[boss_hotkey, opponent_hotkey],
                psql_db=psql_db,
            )

        boss_round_winner = determine_boss_round_winner(
            task_winners, boss_hotkey, tournament.tournament_type, continuous_sft_winners, num_continuous_sft_tasks
        )

        winners = [boss_round_winner]

    return winners

async def get_environment_group_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB, config: Config
) -> list[str]:
    """Get winners from environment tournament group rounds.

    For the final round, return all finalists (boss + contender) and defer
    champion decision to determine_env_tournament_winner().
    """
    boss_hotkey = EMISSION_BURN_HOTKEY

    if completed_round.is_final_round:
        if not round_tasks:
            return [boss_hotkey]
        group_id = round_tasks[0].group_id
        if not group_id:
            return [boss_hotkey]
        participants = await get_tournament_group_members(group_id, psql_db)
        participant_hotkeys = [p.hotkey for p in participants]
        if boss_hotkey not in participant_hotkeys:
            participant_hotkeys.append(boss_hotkey)
        return participant_hotkeys

    if not round_tasks:
        logger.warning(f"No tasks found for environment round {completed_round.round_id}")
        return []

    single_group = len(round_tasks) == 1
    all_winners: list[str] = []

    for task in round_tasks:
        group_id = task.group_id
        if not group_id:
            logger.warning(f"No group_id on task {task.task_id}, skipping")
            continue

        participants = await get_tournament_group_members(group_id, psql_db)
        participant_hotkeys = [p.hotkey for p in participants]
        if not participant_hotkeys:
            logger.warning(f"Environment group {group_id} has no participants")
            continue

        miner_results = await get_task_results_for_ranking(task.task_id, psql_db)
        if not miner_results:
            logger.warning(f"No valid results for task {task.task_id}")
            continue

        ranked_results = calculate_miner_ranking_and_scores(miner_results)
        participant_scores: dict[str, float] = {}
        for result in ranked_results:
            if result.adjusted_loss is None or np.isnan(result.adjusted_loss):
                continue
            participant_scores[result.hotkey] = result.adjusted_loss

        if not participant_scores:
            logger.warning(f"Group {group_id} has no valid scores")
            continue

        sorted_participants = sorted(participant_scores.items(), key=lambda x: x[1], reverse=True)
        boss_score = participant_scores.get(boss_hotkey)
        non_boss_sorted = [(hotkey, score) for hotkey, score in sorted_participants if hotkey != boss_hotkey]

        # Boss retains only when down to a single group and boss wins/ties
        if single_group and boss_score is not None and non_boss_sorted:
            top_challenger_score = non_boss_sorted[0][1]
            if boss_score >= top_challenger_score:
                logger.info(
                    f"Environment group {group_id}: boss score {boss_score} >= top challenger {top_challenger_score} "
                    f"— single group, boss retains"
                )
                continue

        # Advance up to ENV_ADVANCE_PER_GROUP but always eliminate at least 1 to guarantee convergence
        top_to_advance = max(1, min(t_cst.ENV_ADVANCE_PER_GROUP, len(non_boss_sorted) - 1))
        if top_to_advance > 0 and len(non_boss_sorted) > top_to_advance:
            cutoff_score = non_boss_sorted[top_to_advance - 1][1]
            group_winners = [h for h, s in non_boss_sorted if s >= cutoff_score]
        else:
            group_winners = [h for h, _ in non_boss_sorted[:top_to_advance]]

        logger.info(f"Environment group {group_id}: advancing {len(group_winners)} winners: {group_winners}")
        all_winners.extend(group_winners)

    logger.info(f"Environment round {completed_round.round_number}: advancing {len(all_winners)} total non-boss winners")
    return all_winners


async def _get_small_tournament_group_winners(round_tasks: list[TournamentTask], psql_db: PSQLDB) -> list[str]:
    """Rank competitors across the multi-match small-tournament group."""
    match_rankings: list[MatchRanking] = []
    match_losses: list[dict[str, float]] = []
    competitors: set[str] = set()
    for task in round_tasks:
        miner_results = await get_task_results_for_ranking(task.task_id, psql_db)
        if not miner_results:
            logger.warning(f"No valid results for small-tournament task {task.task_id}")
            continue

        ranked_results = calculate_miner_ranking_and_scores(miner_results)
        scored = [
            (result.hotkey, result.adjusted_loss)
            for result in ranked_results
            if result.adjusted_loss is not None and not np.isnan(result.adjusted_loss)
        ]
        if not scored:
            logger.warning(f"Small-tournament task {task.task_id} has no valid scores")
            continue

        scored.sort(key=lambda item: item[1])
        competitors.update(hotkey for hotkey, _loss in scored)
        match_rankings.append(MatchRanking(task_id=task.task_id, ranked_hotkeys=[hotkey for hotkey, _loss in scored]))
        match_losses.append({hotkey: loss for hotkey, loss in scored})

    total_matches = len(match_rankings)
    if total_matches == 0 or not competitors:
        return []

    standings: dict[str, GroupMatchStanding] = {}
    for hotkey in competitors:
        total_rank = 0.0
        matches_attended = 0
        summed_loss = 0.0
        for ranking, losses in zip(match_rankings, match_losses, strict=True):
            if hotkey in ranking.ranked_hotkeys:
                total_rank += ranking.ranked_hotkeys.index(hotkey) + 1
                matches_attended += 1
                summed_loss += losses[hotkey]
            else:
                total_rank += len(ranking.ranked_hotkeys) + 1
                summed_loss += float("inf")
        standings[hotkey] = GroupMatchStanding(
            hotkey=hotkey,
            total_rank=total_rank,
            matches_attended=matches_attended,
            total_matches=total_matches,
            summed_loss=summed_loss,
        )

    ordered = sorted(
        standings.values(),
        key=lambda standing: (standing.has_error, standing.average_rank, standing.summed_loss, standing.hotkey),
    )
    winners = [standing.hotkey for standing in ordered[: t_cst.SMALL_TOURNAMENT_ADVANCE]]
    logger.info(
        f"Small-tournament standings "
        f"{[(s.hotkey, round(s.average_rank, 3), round(s.summed_loss, 4), s.has_error) for s in ordered]}; "
        f"advancing top {len(winners)}: {winners}"
    )
    return winners


async def get_group_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB, config: Config = None
) -> list[str]:
    """Get winners from group round based on adjusted loss scores."""

    # Check if this is an environment task
    is_environment = False
    if round_tasks:
        first_task_object = await get_task(round_tasks[0].task_id, psql_db)
        is_environment = first_task_object and first_task_object.task_type == TaskType.ENVIRONMENTTASK

    if is_environment:
        return await get_environment_group_winners(completed_round, round_tasks, psql_db, config)

    distinct_groups = {task.group_id for task in round_tasks}
    if completed_round.round_number == 1 and len(round_tasks) > 1 and len(distinct_groups) == 1:
        return await _get_small_tournament_group_winners(round_tasks, psql_db)

    # Determine how many winners to advance
    if completed_round.is_final_round:
        TOP_WINNERS_TO_ADVANCE = 1
    else:
        TOP_WINNERS_TO_ADVANCE = 8

    all_winners = []

    for task in round_tasks:
        group_id = task.group_id
        task_id = task.task_id

        logger.info(f"Processing group {group_id} in round {completed_round.round_id}")

        participants = await get_tournament_group_members(group_id, psql_db)
        participant_hotkeys = [p.hotkey for p in participants]
        logger.info(f"Group {group_id} and task {task_id} have {len(participant_hotkeys)} participants")

        if not participant_hotkeys:
            logger.warning(f"Group {group_id} has no participants")
            continue

        miner_results = await get_task_results_for_ranking(task_id, psql_db)
        if not miner_results:
            logger.warning(f"No valid results for task {task_id}")
            continue

        ranked_results = calculate_miner_ranking_and_scores(miner_results)

        participant_scores = {}
        for result in ranked_results:
            hotkey = result.hotkey
            adjusted_loss = result.adjusted_loss

            if adjusted_loss is None or np.isnan(adjusted_loss):
                continue

            participant_scores[hotkey] = adjusted_loss

        if not participant_scores:
            logger.warning(f"Group {group_id} has no valid scores - proceeding with no winners")
            continue

        task_object = await get_task(task_id, psql_db)
        higher_is_better = task_object and task_object.task_type in (TaskType.ENVIRONMENTTASK, TaskType.GRPOTASK)

        sorted_participants = sorted(participant_scores.items(), key=lambda x: x[1], reverse=higher_is_better)
        ranking_direction = "descending (higher is better)" if higher_is_better else "ascending (lower is better)"

        logger.info(
            f"Group {group_id} participants sorted by adjusted loss ({ranking_direction}): "
            f"{[(hotkey, f'{loss:.6f}') for hotkey, loss in sorted_participants]}"
        )

        num_to_advance = min(TOP_WINNERS_TO_ADVANCE, len(sorted_participants))
        group_winners = [hotkey for hotkey, _ in sorted_participants[:num_to_advance]]

        logger.info(f"Group {group_id}: Advancing top {num_to_advance} by adjusted loss: {group_winners}")
        all_winners.extend(group_winners)

    return all_winners

async def get_round_winners(completed_round: TournamentRoundData, psql_db: PSQLDB, config: Config) -> list[str]:
    """Get winners from the completed round."""
    round_tasks = await get_tournament_tasks(completed_round.round_id, psql_db)

    if completed_round.round_type == RoundType.KNOCKOUT:
        winners = await get_knockout_winners(completed_round, round_tasks, psql_db, config)
    else:
        winners = await get_group_winners(completed_round, round_tasks, psql_db, config)

    unique_winners = list(dict.fromkeys(winners))
    if len(winners) != len(unique_winners):
        logger.info(f"Removed {len(winners) - len(unique_winners)} duplicate winners from round {completed_round.round_id}")
        logger.info(f"Original winners: {winners}")
        logger.info(f"Unique winners: {unique_winners}")

    return unique_winners
