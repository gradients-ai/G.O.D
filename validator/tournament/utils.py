#!/usr/bin/env python3

import subprocess
import tempfile
from collections import Counter
from collections import defaultdict
from core.models.pvp_models import GameOutcome
from urllib.parse import urlparse
from pathlib import Path

import aiohttp
import httpx
import numpy as np

from core.models.tournament_models import GitHubOwnerRepo
from core.models.tournament_models import GpuRequirement
from core.models.tournament_models import RespondingNode
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentResultsWithWinners
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentTask
from core.models.tournament_models import TournamentType
from core.models.utility_models import TaskType
from core.models.utility_models import TrainingStatus
from core.utils import build_authenticated_git_url
from core.utils import sanitize_git_text
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
from validator.db.sql.submissions_and_scoring import update_task_node_quality_score_only
from validator.db.sql.tasks import get_task
from validator.db.sql.tournaments import count_champion_consecutive_wins
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_groups
from validator.db.sql.tournaments import get_tournament_pairs
from validator.db.sql.tournaments import get_tournament_participant
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.db.sql.tournaments import get_training_status_for_task_and_hotkeys
from validator.db.sql.tournaments import update_tournament_diff_report
from validator.evaluation.scoring import calculate_miner_ranking_and_scores
from validator.tournament import constants as t_cst
from validator.utils.logging import get_logger
from validator.utils.repo_diff_report import generate_and_upload_repo_diff_report


logger = get_logger(__name__)


def get_tournament_gpu_requirement(task_type: TaskType, model_params_count: int, model_id: str = None) -> GpuRequirement:
    if task_type == TaskType.IMAGETASK:
        return GpuRequirement.H100_1X
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


async def generate_diff_report_for_result(
    tournament: TournamentData,
    challenger_repo: str | None,
    result_summary: str,
    psql_db: PSQLDB,
    challenger_commit_hash: str | None = None,
    challenger_github_token: str | None = None,
) -> str | None:
    if not challenger_repo:
        logger.warning("Challenger repository is missing; skipping repo diff report")
        return None

    previous_boss = await get_tournament_participant(tournament.tournament_id, EMISSION_BURN_HOTKEY, psql_db)
    previous_boss_repo = previous_boss.backup_repo or previous_boss.training_repo if previous_boss else None
    if not previous_boss_repo:
        logger.warning("Previous boss repository is missing; skipping repo diff report")
        return None
    previous_boss_commit_hash = None if previous_boss.backup_repo else previous_boss.training_commit_hash
    previous_boss_github_token = None if previous_boss.backup_repo else previous_boss.github_token

    report_url = await generate_and_upload_repo_diff_report(
        tournament_id=tournament.tournament_id,
        tournament_type=tournament.tournament_type.value,
        challenger_repo_url=challenger_repo,
        previous_boss_repo_url=previous_boss_repo,
        result_summary=result_summary,
        challenger_commit_hash=challenger_commit_hash,
        challenger_github_token=challenger_github_token,
        previous_boss_commit_hash=previous_boss_commit_hash,
        previous_boss_github_token=previous_boss_github_token,
    )
    if report_url:
        await update_tournament_diff_report(tournament.tournament_id, report_url, psql_db)
    return report_url


async def generate_diff_report_and_notify_tournament_completed(
    tournament: TournamentData,
    challenger_repo: str | None,
    result_summary: str,
    winner: str,
    discord_url: str,
    psql_db: PSQLDB,
    challenger_commit_hash: str | None = None,
    challenger_github_token: str | None = None,
) -> None:
    diff_report = None
    try:
        diff_report = await generate_diff_report_for_result(
            tournament,
            challenger_repo,
            result_summary,
            psql_db,
            challenger_commit_hash=challenger_commit_hash,
            challenger_github_token=challenger_github_token,
        )
    except Exception as exc:
        logger.error(f"Failed to generate tournament diff report: {exc}", exc_info=True)

    try:
        await notify_tournament_completed(
            tournament.tournament_id, tournament.tournament_type.value, winner, discord_url, diff_report
        )
    except Exception as exc:
        logger.error(f"Failed to notify tournament completion: {exc}", exc_info=True)


async def _get_final_round_participants(completed_round: TournamentRoundData, psql_db: PSQLDB) -> tuple[str, str]:
    if completed_round.round_type != RoundType.KNOCKOUT:
        raise ValueError(f"Expected a knockout round, got {completed_round.round_type}")

    pairs = await get_tournament_pairs(completed_round.round_id, psql_db)
    if not pairs:
        raise ValueError(f"No pairs found for final round {completed_round.round_id}")

    pair = pairs[0]
    return pair.hotkey1, pair.hotkey2


async def get_challenger_participant_for_retained_boss(
    tournament: TournamentData,
    completed_round: TournamentRoundData,
    winners: list[str],
    psql_db: PSQLDB,
) -> TournamentParticipant | None:
    challenger_hotkey = next((hotkey for hotkey in winners if hotkey != EMISSION_BURN_HOTKEY), None)
    if not challenger_hotkey and completed_round.round_type == RoundType.KNOCKOUT:
        try:
            participant1, participant2 = await _get_final_round_participants(completed_round, psql_db)
            challenger_hotkey = participant2 if participant1 == EMISSION_BURN_HOTKEY else participant1
        except Exception as exc:
            logger.warning(f"Could not determine retained-boss challenger from final round participants: {exc}")

    if not challenger_hotkey:
        logger.warning("Could not determine retained-boss challenger; diff report will not include challenger repo")
        return None

    challenger = await get_tournament_participant(tournament.tournament_id, challenger_hotkey, psql_db)
    if not challenger or not challenger.training_repo:
        logger.warning(f"Challenger {challenger_hotkey} has no training repository in DB")
        return None
    return challenger


def get_progressive_threshold(consecutive_wins: int, tournament_type: TournamentType | None = None) -> float:
    """
    Calculate the progressive threshold using exponential decay.
    """
    max_threshold = t_cst.EXPONENTIAL_BASE_THRESHOLD

    if tournament_type and tournament_type == TournamentType.ENVIRONMENT:
        max_threshold = t_cst.EXPONENTIAL_BASE_THRESHOLD_ENVIRONMENT

    current_threshold = max_threshold * (t_cst.EXPONENTIAL_DECAY_RATE ** (consecutive_wins - 1))
    return max(t_cst.EXPONENTIAL_MIN_THRESHOLD, current_threshold)


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


async def determine_env_tournament_winner(
    tournament: TournamentData, _finalists: list[str], _config: Config, psql_db: PSQLDB,
) -> list[str]:
    """Determine environment winner from boss round only.

    Single contender must beat boss on ALL 3 boss round tasks (no threshold,
    strictly higher score). If not, boss retains.
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
    for task in final_tasks:
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
        elif contender_score < bs:
            outcome = GameOutcome.LOSS
            losses += 1
        else:
            outcome = GameOutcome.DRAW
            draws += 1
        logger.info(
            f"Boss round task {task.task_id}: contender={contender_score:.2f} boss={bs:.2f} -> {outcome.value} "
            f"(running: W={wins} D={draws} L={losses})"
        )

    if losses == 0 and wins > 0:
        logger.info(
            f"Contender {contender} wins environment tournament: "
            f"W={wins} D={draws} L={losses} across {len(final_tasks)} tasks"
        )
        return [contender, boss_hotkey]
    else:
        logger.info(
            f"Boss retains: contender W={wins} D={draws} L={losses} "
            f"across {len(final_tasks)} tasks (need zero losses and at least one win)"
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
                    task_winner = boss_hotkey
                    task_winners.append(task_winner)
                    logger.info(
                        f"GRPO task: Boss wins (higher is better): {boss_loss:.6f} * "
                        f"{boss_multiplier:.3f} = {boss_loss * boss_multiplier:.6f} > {opponent_loss:.6f}"
                    )
                else:
                    task_winner = opponent_hotkey
                    task_winners.append(task_winner)
                    logger.info(
                        f"GRPO task: Opponent wins (higher is better): {opponent_loss:.6f} >= {boss_loss * boss_multiplier:.6f}"
                    )
            elif task_object.task_type == TaskType.ENVIRONMENTTASK:
                if boss_loss * boss_multiplier > opponent_loss:
                    task_winner = boss_hotkey
                    task_winners.append(task_winner)
                    logger.info(
                        f"Environment task: Boss wins (higher is better): {boss_loss:.6f} * "
                        f"{boss_multiplier:.3f} = {boss_loss * boss_multiplier:.6f} > {opponent_loss:.6f}"
                    )
                else:
                    task_winner = opponent_hotkey
                    task_winners.append(task_winner)
                    logger.info(
                        "Environment task: Opponent wins (higher is better): "
                        f"{opponent_loss:.6f} >= {boss_loss * boss_multiplier:.6f}"
                    )
            else:
                # For other tasks, lower scores are better
                if boss_loss * boss_divisor < opponent_loss:
                    task_winner = boss_hotkey
                    task_winners.append(task_winner)
                    logger.info(
                        f"{task_object.task_type} task: Boss wins (lower is better): "
                        f"{boss_loss:.6f} * {boss_divisor:.3f} = {boss_loss * boss_divisor:.6f} < {opponent_loss:.6f}"
                    )
                else:
                    task_winner = opponent_hotkey
                    task_winners.append(task_winner)
                    logger.info(
                        f"{task_object.task_type} task: Opponent wins (lower is better): "
                        f"{opponent_loss:.6f} <= {boss_loss * boss_divisor:.6f}"
                    )

            await update_threshold_adjusted_quality_scores_for_task(
                task_id=task.task_id,
                winner_hotkey=task_winner,
                threshold_percentage=threshold_percentage,
                compared_hotkeys=[boss_hotkey, opponent_hotkey],
                psql_db=psql_db,
            )

        boss_round_winner = determine_boss_round_winner(task_winners, boss_hotkey, tournament.tournament_type)

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


async def notify_tournament_completed(
    tournament_id: str, tournament_type: str, winner: str, discord_url: str, diff_report: str | None = None
):
    try:
        message = (
            f"Tournament Completed!\nTournament ID: {tournament_id}\nType: {tournament_type}\nWinner: {winner}\nStatus: COMPLETED"
        )
        if diff_report:
            message += f"\nDiff Report: {diff_report}"
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


async def validate_repo_obfuscation(
    repo_url: str, commit_hash: str | None = None, github_token: str | None = None
) -> bool:
    """
    Validate that a repository is not obfuscated using the obfuscation detection.

    Args:
        repo_url: The repository URL to validate
        commit_hash: Optional commit hash to validate instead of the default branch

    Returns:
        bool: True if repo is not obfuscated, False if obfuscated
    """
    try:
        clone_url = build_authenticated_git_url(repo_url, github_token)
        cmd = [t_cst.OBFUSCATION_DETECTION_PATH, "--repo", clone_url]
        if commit_hash:
            cmd += ["--commit", commit_hash]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

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


async def validate_repo_license(repo_url: str, github_token: str | None = None) -> bool:
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
            clone_url = build_authenticated_git_url(repo_url, github_token)

            clone_proc = subprocess.run(
                ["git", "clone", clone_url, temp_dir],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if clone_proc.returncode != 0:
                sanitized_stderr = sanitize_git_text(clone_proc.stderr, github_token)
                logger.error(f"Failed to clone repository {repo_url}: {sanitized_stderr}")
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


def parse_github_owner_repo(repo_url: str) -> GitHubOwnerRepo | None:
    path = urlparse(repo_url).path.strip("/")
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] and parts[1]:
        owner, repo_name = parts[0], parts[1].removesuffix(".git")
        return GitHubOwnerRepo(owner=owner, repo=repo_name)
    return None


async def validate_github_tokens(nodes: list[RespondingNode]) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        for node in nodes:
            token = node.training_repo_response.github_token
            if not token:
                continue

            parsed = parse_github_owner_repo(node.training_repo_response.github_repo)
            if not parsed:
                node.training_repo_response.github_token = None
                continue

            try:
                resp = await client.get(
                    f"https://api.github.com/repos/{parsed.owner}/{parsed.repo}",
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                )
                if resp.status_code != 200:
                    logger.warning(
                        f"Token for {node.node.hotkey} does not grant access to "
                        f"{parsed.owner}/{parsed.repo} (HTTP {resp.status_code}) — ignoring token"
                    )
                    node.training_repo_response.github_token = None
            except Exception as e:
                logger.warning(f"Token validation failed for {node.node.hotkey}: {e} — ignoring token")
                node.training_repo_response.github_token = None


def deduplicate_by_github_account(nodes: list[RespondingNode]) -> list[RespondingNode]:
    by_account: defaultdict[str, list[RespondingNode]] = defaultdict(list)
    no_account: list[RespondingNode] = []

    for node in nodes:
        parsed = parse_github_owner_repo(node.training_repo_response.github_repo)
        if parsed:
            by_account[parsed.owner.lower()].append(node)
        else:
            no_account.append(node)

    kept: list[RespondingNode] = list(no_account)
    for account, group in by_account.items():
        if len(group) == 1:
            kept.append(group[0])
            continue

        with_token = [n for n in group if n.training_repo_response.github_token]
        without_token = [n for n in group if not n.training_repo_response.github_token]

        if with_token:
            winner = with_token[0]
            rejected = with_token[1:] + without_token
        else:
            winner = without_token[0]
            rejected = without_token[1:]

        kept.append(winner)
        for r in rejected:
            logger.warning(
                f"Rejecting {r.node.hotkey} — duplicate GitHub account '{account}' "
                f"(kept {winner.node.hotkey})"
            )

    return kept
