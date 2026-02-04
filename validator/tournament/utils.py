#!/usr/bin/env python3

import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import aiohttp
import httpx
import numpy as np

from core.models.tournament_models import GpuRequirement
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import TournamentType
from core.models.utility_models import TaskType
from core.models.utility_models import TrainingStatus
from validator.core.config import Config
from validator.core.constants import DEFAULT_PARTICIPANT_COMMIT
from validator.core.constants import DEFAULT_PARTICIPANT_REPO
from validator.core.constants import EMISSION_BURN_HOTKEY
from validator.core.constants import TOURNAMENT_DPO_GPU_MULTIPLIER
from validator.core.constants import TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100
from validator.core.constants import TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100
from validator.core.constants import TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100
from validator.core.constants import TOURNAMENT_GRPO_GPU_MULTIPLIER
from validator.core.models import MinerResultsImage
from validator.core.models import MinerResultsText
from validator.cycle.util_functions import get_model_num_params
from validator.db import constants as db_cst
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import get_all_scores_and_losses_for_task
from validator.db.sql.submissions_and_scoring import get_task_winner
from validator.db.sql.tasks import get_task
from validator.db.sql.tournaments import count_champion_consecutive_wins
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_groups
from validator.db.sql.tournaments import get_tournament_participant
from validator.db.sql.tournaments import get_tournament_tasks
from validator.db.sql.tournaments import get_training_status_for_task_and_hotkeys
from validator.evaluation.scoring import calculate_miner_ranking_and_scores
from validator.tournament import constants as t_cst
from validator.utils.logging import get_logger


logger = get_logger(__name__)


def get_tournament_gpu_requirement(task_type: TaskType, model_params_count: int, model_id: str = None) -> GpuRequirement:
    if task_type == TaskType.IMAGETASK:
        return GpuRequirement.A100
    if not model_params_count and model_id:
        logger.info(f"model_params_count is {model_params_count}, fetching from HuggingFace for model {model_id}")
        try:
            model_params_count = get_model_num_params(model_id)
            logger.info(f"Fetched model_params_count: {model_params_count} for model {model_id}")
        except Exception:
            model_params_count = 0

        if not model_params_count:
            logger.warning(f"Could not determine model size for {model_id}, defaulting to H100_1X")
            return GpuRequirement.H100_1X

    params_b = model_params_count / 1_000_000_000

    if task_type == TaskType.DPOTASK:
        params_b *= TOURNAMENT_DPO_GPU_MULTIPLIER
    elif task_type == TaskType.GRPOTASK:
        params_b *= TOURNAMENT_GRPO_GPU_MULTIPLIER
    elif task_type == TaskType.ENVIRONMENTTASK:
        return GpuRequirement.H100_4X

    if params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_2X_H100:
        return GpuRequirement.H100_1X
    elif params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_4X_H100:
        return GpuRequirement.H100_2X
    elif params_b <= TOURNAMENT_GPU_THRESHOLD_FOR_8X_H100:
        return GpuRequirement.H100_4X
    else:
        return GpuRequirement.H100_8X


def get_progressive_threshold(consecutive_wins: int, tournament_type: TournamentType | None = None) -> float:
    """
    Calculate the progressive threshold using exponential decay.
    """
    max_threshold = t_cst.EXPONENTIAL_BASE_THRESHOLD

    if tournament_type and tournament_type == TournamentType.ENVIRONMENT:
        max_threshold = t_cst.EXPONENTIAL_BASE_THRESHOLD_ENVIRONMENT
        
    current_threshold = max_threshold * (t_cst.EXPONENTIAL_DECAY_RATE ** (consecutive_wins - 1))
    return max(t_cst.EXPONENTIAL_MIN_THRESHOLD, current_threshold)


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
    if not tournament or not tournament.winner_hotkey:
        return None

    winner = tournament.winner_hotkey
    if winner == EMISSION_BURN_HOTKEY:
        winner = tournament.base_winner_hotkey

    return winner


def did_winner_change(previous_tournament: TournamentData | None, latest_tournament: TournamentData) -> bool:
    """
    Determine if the tournament winner changed between two tournaments.

    Returns True if:
    - No previous tournament exists (first tournament)
    - Latest winner is not EMISSION_BURN_HOTKEY and differs from previous winner

    Returns:
        True if winner changed, False if same winner defended
    """
    if not previous_tournament:
        return True

    # If latest winner is not EMISSION_BURN_HOTKEY, a new challenger won
    # If it IS EMISSION_BURN_HOTKEY, the defending champion won
    if (
        previous_tournament.winner_hotkey != latest_tournament.winner_hotkey
        and latest_tournament.winner_hotkey != EMISSION_BURN_HOTKEY
    ):
        return True

    return False


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
        if task_type in [TaskType.INSTRUCTTEXTTASK, TaskType.CHATTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.ENVIRONMENTTASK]:
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


async def get_latest_commit_hash_from_github(repo_url: str) -> str | None:
    """Fetch the latest commit hash from a GitHub repository."""
    # Extract owner/repo from URL: https://github.com/owner/repo
    repo_path = repo_url.split("github.com/")[1].replace(".git", "")
    api_url = f"https://api.github.com/repos/{repo_path}/commits/main"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("sha", "")
                else:
                    logger.error(f"Failed to fetch commit hash from {repo_url}: HTTP {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error fetching commit hash from {repo_url}: {e}")
        return None


async def get_base_contestant(psql_db: PSQLDB, tournament_type: TournamentType, config: Config) -> TournamentParticipant | None:
    """Get a BASE contestant as the last tournament winner."""

    latest_winner = await get_latest_tournament_winner_participant(psql_db, tournament_type, config)
    if latest_winner:
        logger.info(f"Using latest tournament winner as BASE: {latest_winner.hotkey}")

        if latest_winner.backup_repo:
            logger.info(f"Previous winner has backup repo: {latest_winner.backup_repo}")
            commit_hash = await get_latest_commit_hash_from_github(latest_winner.backup_repo)
            if not commit_hash:
                logger.warning(f"Could not fetch commit hash for {latest_winner.backup_repo}, setting to None")

            return TournamentParticipant(
                tournament_id="",
                hotkey=EMISSION_BURN_HOTKEY,
                training_repo=latest_winner.backup_repo,
                training_commit_hash=commit_hash,
            )
        else:
            logger.warning("Could not determine tournament ID for uploaded repo, falling back to original training_repo")
            # Fallback to original training_repo if we can't determine the uploaded repo
            return TournamentParticipant(
                tournament_id="",
                hotkey=EMISSION_BURN_HOTKEY,
                training_repo=latest_winner.training_repo,
                training_commit_hash=latest_winner.training_commit_hash,
            )

    logger.info(
        f"No previous tournament winner found for type {tournament_type.value}, "
        f"using hardcoded base winner: {EMISSION_BURN_HOTKEY}"
    )

    hardcoded_participant = TournamentParticipant(
        tournament_id="",
        hotkey=EMISSION_BURN_HOTKEY,
        training_repo=DEFAULT_PARTICIPANT_REPO,
        training_commit_hash=DEFAULT_PARTICIPANT_COMMIT,
    )

    return hardcoded_participant


async def get_latest_tournament_winner_participant(
    psql_db: PSQLDB, tournament_type: TournamentType, config: Config
) -> TournamentParticipant | None:
    """Get the winner participant from the latest completed tournament of the given type."""
    latest_tournament = await get_latest_completed_tournament(psql_db, tournament_type)
    if not latest_tournament:
        logger.warning(f"No completed tournaments found for type {tournament_type.value}")
        return None

    winner_hotkey = latest_tournament.winner_hotkey
    if not winner_hotkey:
        logger.warning(f"Tournament {latest_tournament.tournament_id} is completed but has no winner_hotkey stored")
        return None

    logger.info(f"Found latest tournament winner: {winner_hotkey}")
    winner_participant = await get_tournament_participant(latest_tournament.tournament_id, winner_hotkey, psql_db)

    # If we can't find the winner's participant record, check if they were the defending champion
    # who entered as EMISSION_BURN_HOTKEY
    if not winner_participant:
        logger.warning(
            f"Could not find participant record for winner {winner_hotkey} in tournament {latest_tournament.tournament_id}"
        )

        # If the winner was the base_winner (defending champion), try to get their record from EMISSION_BURN_HOTKEY
        if winner_hotkey == latest_tournament.base_winner_hotkey:
            logger.info(f"Winner {winner_hotkey} was the defending champion, checking EMISSION_BURN_HOTKEY participant record")
            emission_participant = await get_tournament_participant(
                latest_tournament.tournament_id, EMISSION_BURN_HOTKEY, psql_db
            )
            if emission_participant:
                # Use the EMISSION_BURN_HOTKEY participant's training info but with the actual winner's hotkey
                emission_participant.hotkey = winner_hotkey
                return emission_participant

        # If still no participant record found, return None to use default
        logger.warning(f"No participant record found for winner {winner_hotkey}, will use default")
        return None

    # If the participant is EMISSION_BURN_HOTKEY but we have a real winner, use the real winner's hotkey
    if winner_participant.hotkey == EMISSION_BURN_HOTKEY and latest_tournament.base_winner_hotkey:
        winner_participant.hotkey = latest_tournament.base_winner_hotkey

    return winner_participant


def draw_knockout_bracket(rounds_data, winners_by_round):
    """Draw an ASCII art bracket diagram for knockout tournament progression."""
    logger.info("\nKNOCKOUT BRACKET:")
    logger.info("=" * 60)

    if not rounds_data:
        logger.info("No rounds data available")
        return

    knockout_rounds = [r for r in rounds_data if r.get("type") == RoundType.KNOCKOUT]
    if not knockout_rounds:
        logger.info("No knockout rounds found")
        return

    bracket_lines = []

    for round_num, round_data in enumerate(knockout_rounds):
        participants = round_data.get("participants", [])
        knockout_round_index = None
        for i, r in enumerate(rounds_data):
            if r.get("type") == RoundType.KNOCKOUT and r == round_data:
                knockout_round_index = i
                break

        winners = winners_by_round.get(knockout_round_index, []) if knockout_round_index is not None else []

        if not participants:
            continue

        round_header = f"Round {round_num + 1}"
        if round_data.get("is_final_round"):
            round_header += " 🔥 BOSS ROUND 🔥"
        bracket_lines.append(f"{round_header:>20}")

        for i in range(0, len(participants), 2):
            if i + 1 < len(participants):
                p1 = participants[i]
                p2 = participants[i + 1]

                p1_won = p1 in winners
                p2_won = p2 in winners

                indent = "  " * round_num
                if p1_won:
                    line1 = f"{indent}├─ {p1} ✓"
                else:
                    line1 = f"{indent}├─ {p1}"

                if p2_won:
                    line2 = f"{indent}├─ {p2} ✓"
                else:
                    line2 = f"{indent}├─ {p2}"

                bracket_lines.append(f"{line1:>40}")
                bracket_lines.append(f"{line2:>40}")

                if round_num < len(knockout_rounds) - 1:
                    bracket_lines.append(f"{indent}│")

        bracket_lines.append("")

    for line in bracket_lines:
        logger.info(line)


async def draw_group_stage_table(rounds_data, winners_by_round, psql_db):
    """Draw a table showing group stage results."""
    logger.info("\nGROUP STAGE RESULTS:")
    logger.info("=" * 60)

    group_round = None
    group_round_index = None
    for i, round_data in enumerate(rounds_data):
        if round_data.get("type") == RoundType.GROUP:
            group_round = round_data
            group_round_index = i
            break

    if not group_round:
        logger.info("No group stage found")
        return

    round_id = group_round.get("round_id")
    if not round_id:
        logger.info("No round ID found for group stage")
        return

    group_objs = await get_tournament_groups(round_id, psql_db)
    if not group_objs:
        logger.info("No groups found for group stage")
        return

    winners = winners_by_round.get(group_round_index, []) if group_round_index is not None else []

    logger.info(f"Group Stage: {len(group_objs)} groups")
    logger.info("")

    for group in group_objs:
        group_id = group.group_id
        members = await get_tournament_group_members(group_id, psql_db)
        hotkeys = [m.hotkey for m in members]
        logger.info(f"Group {group_id}:")
        logger.info("-" * 40)
        for i, participant in enumerate(hotkeys):
            if participant in winners:
                logger.info(f"  {i + 1:2d}. {participant} ✓ (ADVANCED)")
            else:
                logger.info(f"  {i + 1:2d}. {participant}")
        logger.info("")


def determine_boss_round_winner(task_winners: list[str], boss_hotkey: str, tournament_type: TournamentType) -> str:
    """
    Determine the winner of a boss round based on task results and tournament type.

    Args:
        task_winners: List of hotkeys that won each task in the boss round
        boss_hotkey: The defending champion's hotkey
        tournament_type: Type of tournament (TEXT or IMAGE)

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

    # Apply different winning requirements based on tournament type
    # Both IMAGE and TEXT tournaments: Challenger must win more than half (majority) of tasks to become new boss
    required_wins = (total_tasks // 2) + 1
    if opponent_hotkey and opponent_wins > total_tasks // 2:
        logger.info(
            f"{tournament_type.value} tournament: Challenger wins boss round with majority: "
            f"{opponent_wins}/{total_tasks} tasks won (required {required_wins})"
        )
        return opponent_hotkey
    else:
        boss_wins = win_counts.get(boss_hotkey, 0)
        if opponent_hotkey:
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
        # Boss round. Progressive threshold system based on consecutive wins.
        boss_hotkey = EMISSION_BURN_HOTKEY
        opponent_hotkey = None
        task_winners = []

        # Get tournament info to determine the current champion and their consecutive wins
        tournament = await get_tournament(completed_round.tournament_id, psql_db)
        if not tournament:
            logger.error(f"Could not find tournament {completed_round.tournament_id}")
            return []

        # Get the current champion (base_winner_hotkey) and count their consecutive wins
        current_champion = tournament.base_winner_hotkey or boss_hotkey
        consecutive_wins = await count_champion_consecutive_wins(psql_db, tournament.tournament_type, current_champion)

        # Calculate the progressive threshold
        threshold_percentage = get_progressive_threshold(consecutive_wins, tournament.tournament_type)
        logger.info(
            f"Champion {current_champion} has {consecutive_wins} consecutive wins, "
            f"using {threshold_percentage * 100:.1f}% threshold"
        )

        for task in round_tasks:
            logger.info(f"Processing boss round task {task.task_id}")

            task_object = await get_task(task.task_id, psql_db)

            miner_results = await get_task_results_for_ranking(task.task_id, psql_db)
            if not miner_results:
                logger.warning(f"No valid results for boss round task {task.task_id}. Winner is base contestant.")
                task_winners.append(boss_hotkey)
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
                    task_winners.append(opponent_hotkey)
                elif boss_training_success and not opponent_training_success:
                    logger.info(f"Opponent training failed, boss succeeded - boss wins task {task.task_id}")
                    task_winners.append(boss_hotkey)
                elif not boss_training_success and not opponent_training_success:
                    logger.info(f"Both training failed - boss wins by default for task {task.task_id}")
                    task_winners.append(boss_hotkey)
                else:
                    # Both training succeeded but at least one has missing/invalid evaluation results
                    # Check who has valid evaluation results and award to them
                    boss_has_valid_eval = boss_loss is not None
                    opponent_has_valid_eval = opponent_loss is not None

                    if opponent_has_valid_eval and not boss_has_valid_eval:
                        logger.info(f"Boss evaluation failed, opponent succeeded - opponent wins task {task.task_id}")
                        task_winners.append(opponent_hotkey)
                    elif boss_has_valid_eval and not opponent_has_valid_eval:
                        logger.info(f"Opponent evaluation failed, boss succeeded - boss wins task {task.task_id}")
                        task_winners.append(boss_hotkey)
                    else:
                        logger.warning(
                            f"Both evaluation failed or both succeeded but missing results - skipping task {task.task_id}"
                        )
                continue

            logger.info(f"Boss round task {task.task_id}: Boss loss: {boss_loss:.6f}, Opponent loss: {opponent_loss:.6f}")

            # Apply progressive threshold system
            boss_multiplier = 1 + threshold_percentage  # For higher-is-better tasks
            boss_divisor = 1 - threshold_percentage  # For lower-is-better tasks

            if task_object.task_type == TaskType.GRPOTASK:
                # For GRPO tasks, higher scores are better
                if boss_loss * boss_multiplier > opponent_loss:
                    task_winners.append(boss_hotkey)
                    logger.info(
                        f"GRPO task: Boss wins (higher is better): {boss_loss:.6f} * "
                        f"{boss_multiplier:.3f} = {boss_loss * boss_multiplier:.6f} > {opponent_loss:.6f}"
                    )
                else:
                    task_winners.append(opponent_hotkey)
                    logger.info(
                        f"GRPO task: Opponent wins (higher is better): {opponent_loss:.6f} >= {boss_loss * boss_multiplier:.6f}"
                    )
            elif task_object.task_type == TaskType.ENVIRONMENTTASK:
                if boss_loss * boss_multiplier > opponent_loss:
                    task_winners.append(boss_hotkey)
                    logger.info(
                        f"Environment task: Boss wins (higher is better): {boss_loss:.6f} * "
                        f"{boss_multiplier:.3f} = {boss_loss * boss_multiplier:.6f} > {opponent_loss:.6f}"
                    )
                else:
                    task_winners.append(opponent_hotkey)
                    logger.info(
                        f"Environment task: Opponent wins (higher is better): {opponent_loss:.6f} >= {boss_loss * boss_multiplier:.6f}"
                    )
            else:
                # For other tasks, lower scores are better
                if boss_loss * boss_divisor < opponent_loss:
                    task_winners.append(boss_hotkey)
                    logger.info(
                        f"{task_object.task_type} task: Boss wins (lower is better): "
                        f"{boss_loss:.6f} * {boss_divisor:.3f} = {boss_loss * boss_divisor:.6f} < {opponent_loss:.6f}"
                    )
                else:
                    task_winners.append(opponent_hotkey)
                    logger.info(
                        f"{task_object.task_type} task: Opponent wins (lower is better): "
                        f"{opponent_loss:.6f} <= {boss_loss * boss_divisor:.6f}"
                    )

        boss_round_winner = determine_boss_round_winner(task_winners, boss_hotkey, tournament.tournament_type)

        winners = [boss_round_winner]

    return winners


async def get_environment_group_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB, config: Config
) -> list[str]:
    """Get winners from environment tournament group round with progressive threshold for boss.
    Returns [winner, second_place] where second_place is the runner-up.
    - If boss wins: 2nd place is highest scoring non-boss (eligible or not)
    - If challenger wins: 2nd place is second highest scoring participant
    """
    boss_hotkey = EMISSION_BURN_HOTKEY

    for task in round_tasks:
        group_id = task.group_id
        task_id = task.task_id

        logger.info(f"Processing environment tournament group {group_id} in round {completed_round.round_id}")

        participants = await get_tournament_group_members(group_id, psql_db)
        participant_hotkeys = [p.hotkey for p in participants]
        logger.info(f"Group {group_id} and task {task_id} have {len(participant_hotkeys)} participants")

        if not participant_hotkeys:
            logger.warning(f"Group {group_id} has no participants, defaulting to boss as winner")
            return [boss_hotkey]

        miner_results = await get_task_results_for_ranking(task_id, psql_db)
        if not miner_results:
            logger.warning(f"No valid results for task {task_id}, defaulting to boss as winner")
            return [boss_hotkey]

        ranked_results = calculate_miner_ranking_and_scores(miner_results)

        participant_scores = {}
        boss_score = None
        
        for result in ranked_results:
            hotkey = result.hotkey
            adjusted_loss = result.adjusted_loss

            if adjusted_loss is None or np.isnan(adjusted_loss):
                continue

            participant_scores[hotkey] = adjusted_loss
            if hotkey == boss_hotkey:
                boss_score = adjusted_loss

        if not participant_scores:
            logger.warning(f"Group {group_id} has no valid scores, defaulting to boss as winner")
            return [boss_hotkey]

        all_sorted_participants = sorted(participant_scores.items(), key=lambda x: x[1], reverse=True)

        # Get tournament info and calculate progressive threshold
        tournament = await get_tournament(completed_round.tournament_id, psql_db)
        if not tournament:
            logger.error(f"Could not find tournament {completed_round.tournament_id}, defaulting to boss as winner")
            return [boss_hotkey]

        current_champion = tournament.base_winner_hotkey or boss_hotkey
        consecutive_wins = await count_champion_consecutive_wins(psql_db, tournament.tournament_type, current_champion)
        threshold_percentage = get_progressive_threshold(consecutive_wins, tournament.tournament_type)
        logger.info(
            f"Environment tournament final round: Champion {current_champion} has {consecutive_wins} consecutive wins, "
            f"using {threshold_percentage * 100:.1f}% threshold"
        )

        eligible_participants = {}
        if boss_score is not None:
            boss_multiplier = 1 + threshold_percentage
            boss_threshold_score = boss_score * boss_multiplier
            
            logger.info(
                f"Boss score: {boss_score:.6f}, Threshold: {threshold_percentage * 100:.1f}%, "
                f"Boss threshold score: {boss_threshold_score:.6f}"
            )
            
            for hotkey, score in participant_scores.items():
                if hotkey == boss_hotkey:
                    eligible_participants[hotkey] = score
                elif score >= boss_threshold_score:
                    eligible_participants[hotkey] = score
                    logger.info(
                        f"Challenger {hotkey} beats boss threshold: {score:.6f} >= {boss_threshold_score:.6f}"
                    )
                else:
                    logger.info(
                        f"Challenger {hotkey} does not beat boss threshold: {score:.6f} < {boss_threshold_score:.6f}"
                    )
            
            if len(eligible_participants) == 1 and boss_hotkey in eligible_participants:
                logger.info(f"No challengers beat boss threshold. Boss {boss_hotkey} wins.")
                # Boss wins - 2nd place is highest scoring non-boss participant
                second_place = None
                for hotkey, score in all_sorted_participants:
                    if hotkey != boss_hotkey:
                        second_place = hotkey
                        break
                if second_place:
                    logger.info(f"2nd place (highest non-boss): {second_place}")
                    return [boss_hotkey, second_place]
                return [boss_hotkey]
        else:
            logger.warning(f"Boss score not found for task {task_id}, proceeding without threshold")
            eligible_participants = participant_scores

        # Sort eligible participants descending for environment tasks (higher is better)
        sorted_participants = sorted(eligible_participants.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(
            f"Group {group_id} participants sorted by adjusted loss (descending, higher is better): "
            f"{[(hotkey, f'{loss:.6f}') for hotkey, loss in sorted_participants]}"
        )

        if sorted_participants:
            winner = sorted_participants[0][0]
            logger.info(f"Group {group_id}: Environment tournament winner: {winner}")
            
            # Determine 2nd place
            second_place = None
            if len(sorted_participants) >= 2:
                # Challenger won - 2nd place is second in eligible list
                second_place = sorted_participants[1][0]
            
            if second_place:
                logger.info(f"Group {group_id}: 2nd place: {second_place}")
                return [winner, second_place]
            return [winner]

    logger.warning(f"No winner found for environment tournament round {completed_round.round_id}, defaulting to boss")
    return [boss_hotkey]


async def get_group_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB, config: Config = None
) -> list[str]:
    """Get winners from group round based on adjusted loss scores (top 8 performers, or 1 for final rounds)."""

    TOP_WINNERS_TO_ADVANCE = 1 if completed_round.is_final_round else 8

    if completed_round.is_final_round:
        task_object = await get_task(round_tasks[0].task_id, psql_db) if round_tasks else None
        if task_object and task_object.task_type == TaskType.ENVIRONMENTTASK:
            return await get_environment_group_winners(completed_round, round_tasks, psql_db, config)

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

        sorted_participants = sorted(participant_scores.items(), key=lambda x: x[1])
        ranking_direction = "ascending (lower is better)"
        
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

    # Ensure unique winners by converting to set and back to list
    unique_winners = list(set(winners))
    if len(winners) != len(unique_winners):
        logger.info(f"Removed {len(winners) - len(unique_winners)} duplicate winners from round {completed_round.round_id}")
        logger.info(f"Original winners: {winners}")
        logger.info(f"Unique winners: {unique_winners}")

    return unique_winners


async def send_to_discord(webhook: str, message: str):
    async with httpx.AsyncClient() as client:
        payload = {"content": message}
        response = await client.post(webhook, json=payload)
        return response


async def notify_tournament_started(tournament_id: str, tournament_type: str, participants: int, discord_url: str):
    try:
        message = (
            f"Tournament Started!\nTournament ID: {tournament_id}\nType: {tournament_type}\n"
            f"Participants: {participants}\nStatus: ACTIVE"
        )
        await send_to_discord(discord_url, message)
    except Exception as e:
        logger.error(f"Failed to send Discord notification for tournament start: {e}")


async def notify_tournament_completed(tournament_id: str, tournament_type: str, winner: str, discord_url: str):
    try:
        message = (
            f"Tournament Completed!\nTournament ID: {tournament_id}\nType: {tournament_type}\nWinner: {winner}\nStatus: COMPLETED"
        )
        await send_to_discord(discord_url, message)
    except Exception as e:
        logger.error(f"Failed to send Discord notification for tournament completion: {e}")


async def notify_organic_task_created(task_id: str, task_type: str, discord_url: str, is_benchmark: bool = False):
    try:
        if is_benchmark:
            message = f"New Benchmark Task Created!\nTask ID: {task_id}\nType: {task_type}"
        else:
            message = f"New Organic Task Created!\nTask ID: {task_id}\nType: {task_type}"
        await send_to_discord(discord_url, message)
    except Exception as e:
        logger.error(f"Failed to send Discord notification for task creation: {e}")


async def validate_repo_obfuscation(repo_url: str) -> bool:
    """
    Validate that a repository is not obfuscated using the obfuscation detection.

    Args:
        repo_url: The repository URL to validate

    Returns:
        bool: True if repo is not obfuscated, False if obfuscated
    """
    try:
        proc = subprocess.run(
            [t_cst.OBFUSCATION_DETECTION_PATH, "--repo", repo_url],
            capture_output=True,
            text=True,
            timeout=30,
        )

        logger.info(f"Obfuscation detection output: {proc.stdout}")

        if proc.returncode == 0:
            logger.info(f"Repo {repo_url} is not obfuscated (exit code 0)")
            return True
        else:
            logger.warning(f"Repo {repo_url} is obfuscated (exit code {proc.returncode})")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Obfuscation detection timed out for repo {repo_url}")
        return False
    except Exception as e:
        logger.error(f"Obfuscation detection failed for repo {repo_url}: {str(e)}")
        return False


async def validate_repo_license(repo_url: str) -> bool:
    """
    Validate that a repository has verbatim LICENSE and NOTICE files matching the current repository.

    Args:
        repo_url: The repository URL to validate

    Returns:
        bool: True if repo has valid LICENSE and NOTICE files, False otherwise
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Cloning repository {repo_url} for license validation")

            clone_proc = subprocess.run(
                ["git", "clone", repo_url, temp_dir],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if clone_proc.returncode != 0:
                logger.error(f"Failed to clone repository {repo_url}: {clone_proc.stderr}")
                return False

            temp_path = Path(temp_dir)
            current_file_path = Path(__file__).resolve()
            repo_root = current_file_path.parent.parent.parent

            expected_license_path = repo_root / "LICENSE.md"
            if not expected_license_path.exists():
                expected_license_path = repo_root / "LICENSE"
                if not expected_license_path.exists():
                    logger.warning(
                        f"Expected LICENSE file not found in validator repository at "
                        f"{repo_root / 'LICENSE.md'} or {repo_root / 'LICENSE'}. "
                        f"Skipping license validation for {repo_url}"
                    )
                    return True

            expected_notice_path = None
            for notice_filename in ["NOTICE", "NOTICE.txt", "notice.txt", "Notice.txt", "notice", "Notice"]:
                potential_path = repo_root / notice_filename
                if potential_path.exists():
                    expected_notice_path = potential_path
                    break

            if not expected_notice_path:
                logger.warning(
                    f"Expected NOTICE file not found in validator repository at {repo_root} "
                    f"(checked NOTICE, NOTICE.txt, notice.txt, Notice.txt, notice, Notice). "
                    f"Skipping license validation for {repo_url}"
                )
                return True

            license_file_path = None
            for license_filename in ["LICENSE.md", "LICENSE", "license.md", "license", "License.md", "License"]:
                potential_path = temp_path / license_filename
                if potential_path.exists():
                    license_file_path = potential_path
                    break

            if not license_file_path:
                logger.warning(
                    f"License file not found in repository {repo_url} "
                    f"(checked LICENSE.md, LICENSE, license.md, license, License.md, License)"
                )
                return False

            license_content = license_file_path.read_text(encoding="utf-8")
            expected_license = expected_license_path.read_text(encoding="utf-8")

            expected_license_normalized = "\n".join(line.rstrip() for line in expected_license.splitlines())
            actual_license_normalized = "\n".join(line.rstrip() for line in license_content.splitlines())

            if expected_license_normalized != actual_license_normalized:
                logger.warning(f"LICENSE file content does not match verbatim for repository {repo_url}")
                return False

            notice_file_path = None
            for notice_filename in ["NOTICE", "NOTICE.txt", "notice.txt", "Notice.txt", "notice", "Notice"]:
                potential_path = temp_path / notice_filename
                if potential_path.exists():
                    notice_file_path = potential_path
                    break

            if not notice_file_path:
                logger.warning(
                    f"NOTICE file not found in repository {repo_url} "
                    f"(checked NOTICE, NOTICE.txt, notice.txt, Notice.txt, notice, Notice)"
                )
                return False

            notice_content = notice_file_path.read_text(encoding="utf-8")
            expected_notice = expected_notice_path.read_text(encoding="utf-8")

            expected_notice_normalized = "\n".join(line.rstrip() for line in expected_notice.splitlines())
            actual_notice_normalized = "\n".join(line.rstrip() for line in notice_content.splitlines())

            if expected_notice_normalized != actual_notice_normalized:
                logger.warning(f"NOTICE file content does not match verbatim for repository {repo_url}")
                return False

            logger.info(f"Repository {repo_url} passed license validation")
            return True

    except subprocess.TimeoutExpired:
        logger.error(f"Repository validation timed out for repo {repo_url}")
        return False
    except Exception as e:
        logger.error(f"Repository validation failed for repo {repo_url}: {str(e)}")
        return False
