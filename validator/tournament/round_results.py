import hashlib
import math
import random
from collections import Counter
from collections.abc import Iterable
from itertools import permutations

import numpy as np

from core.logging import get_logger
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskType
from core.models.task_models import higher_is_better
from validator.app.config import Config
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import get_eligible_hotkeys_for_task
from validator.db.sql.submissions_and_scoring import get_per_example_losses
from validator.db.sql.submissions_and_scoring import get_task_winner
from validator.db.sql.tasks import count_hotkeys_assigned_to_task
from validator.db.sql.tasks import get_nodes_assigned_to_task
from validator.db.sql.tasks import get_task
from validator.db.sql.tournament_seeds import boss_side_seed_failure
from validator.db.sql.tournament_seeds import is_seeded_text_tournament
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_group_members
from validator.db.sql.tournaments import get_tournament_pairs
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import get_tournament_tasks
from validator.db.sql.tournaments import get_training_status_for_task_and_hotkeys
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResults
from validator.scoring.models import MinerResultsText
from validator.scoring.quant_ranking import QuantReferenceUnavailable
from validator.scoring.tasks import calculate_miner_ranking_and_scores
from validator.tournament import constants as t_cst
from validator.tournament.models import BossRoundDrawResolution
from validator.tournament.models import BossRoundTaskVerdict
from validator.tournament.models import ContinuousSftLineageOutcome
from validator.tournament.models import EmptyScoreGroup
from validator.tournament.models import GroupMatchStanding
from validator.tournament.models import MatchRanking
from validator.tournament.models import PairedLossComparison
from validator.tournament.models import RoundOutcome
from validator.tournament.models import RoundType
from validator.tournament.models import T1T2Ranks
from validator.tournament.models import T3ViabilityVerdict
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentResultsWithWinners
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus
from validator.tournament.task_results import get_task_results_for_ranking
from validator.tournament.text_crown import TextHeadToHead
from validator.tournament.text_crown import is_text_big_task
from validator.tournament.text_crown import text_crown_verdict
from validator.tournament.text_crown import text_head_to_head
from validator.tournament.thresholds import challenger_beats_boss
from validator.tournament.thresholds import challenger_takes_paired_task
from validator.tournament.thresholds import challenger_within_freeform_allowed_loss
from validator.tournament.thresholds import challenger_within_paired_allowed_loss
from validator.tournament.thresholds import compare_paired_losses
from validator.tournament.thresholds import paired_head_to_head_winner
from validator.tournament.thresholds import quant_challenger_beats_boss
from validator.tournament.thresholds import quant_challenger_takes_fixed_boss_task
from validator.tournament.thresholds import resolve_freeform_boss_task
from validator.tournament.thresholds import sku_aligned_losses
from validator.tournament.thresholds import update_threshold_adjusted_quality_scores_for_task


logger = get_logger(__name__)


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

async def _resolve_boss_round_task_winner(
    task_id,
    task_object,
    boss_hotkey: str,
    opponent_hotkey: str,
    boss_loss: float,
    opponent_loss: float,
    threshold_percentage: float,
    psql_db: PSQLDB,
) -> BossRoundTaskVerdict:
    """Decide one boss-round task, preferring the paired per-example comparison.

    ``decided_by`` describes what decided it, for the persisted score_reason, and is None when the
    relative margin decided it (the caller's default wording already says that). ``is_draw`` marks
    the task as carrying no result at all, which excludes it from the dethrone tally.

    Instruct and DPO losses are log-likelihoods, where a relative margin is the wrong scale and a
    scalar mean cannot separate a real win from held-out sampling noise. Where both sides have
    per-example vectors those are compared example by example; everything else (GRPO, env, and any
    task evaluated before the evaluators emitted vectors) keeps the relative margin.
    """
    if task_object.task_type in QUANT_TASK_TYPES:
        return await _resolve_quant_boss_round_task_winner(
            task_id, task_object, boss_hotkey, opponent_hotkey, boss_loss, opponent_loss, threshold_percentage, psql_db
        )

    higher_is_better = task_object.task_type in (TaskType.GRPOTASK, TaskType.ENVIRONMENTTASK)

    if task_object.task_type in t_cst.PAIRED_BOSS_ROUND_TASK_TYPES:
        boss_losses, boss_fingerprint = await get_per_example_losses(task_id, boss_hotkey, psql_db)
        opponent_losses, opponent_fingerprint = await get_per_example_losses(task_id, opponent_hotkey, psql_db)

        fingerprints_disagree = (
            boss_fingerprint is not None and opponent_fingerprint is not None and boss_fingerprint != opponent_fingerprint
        )
        if boss_losses is None or opponent_losses is None:
            # Nothing was emitted: pre-existing tasks, the rollout window, or an eval-time vector
            # failure (see EvaluationResultText.per_example_losses_failed). Falling back keeps
            # the round resolvable, but paired significance is unavailable.
            missing = "boss" if boss_losses is None else "challenger"
            logger.warning(
                f"Boss round task {task_id}: no per-example losses for {missing} "
                f"(absent or extraction failed at eval) - falling back to the "
                f"{threshold_percentage * 100:.1f}% margin"
            )
        elif fingerprints_disagree or len(boss_losses) != len(opponent_losses):
            # Vectors exist but cannot be paired. The eval row set is a function of the candidate
            # model's max_position_embeddings, so the two models were scored on different data and
            # no comparison between them is valid. Resolved as not a win rather than falling back
            # to a comparison of two means taken over different example sets.
            logger.error(
                f"Boss round task {task_id}: per-example vectors present but not pairable "
                f"(boss_fingerprint={boss_fingerprint} challenger_fingerprint={opponent_fingerprint} "
                f"boss_n={len(boss_losses)} challenger_n={len(opponent_losses)}) - not counted as a win"
            )
            return BossRoundTaskVerdict(winner_hotkey=boss_hotkey, decided_by="unpairable per-example vectors")
        else:
            comparison = compare_paired_losses(boss_losses, opponent_losses)
            # Logged whether the challenger won or lost: the observed win rate across real boss
            # rounds is the only data that can calibrate BOSS_ROUND_MIN_WIN_RATE.
            logger.info(
                f"BOSS_ROUND_PAIRED task={task_id} type={task_object.task_type} "
                f"challenger_won={comparison.challenger_won} draw={comparison.is_draw} "
                f"win_rate={comparison.win_rate:.4f} win_rate_lb={comparison.win_rate_lower_bound:.4f} "
                f"mean_gap_nats={comparison.mean_gap_nats:.6f} mean_gap_lb={comparison.mean_gap_lower_bound:.6f} "
                f"decided={comparison.n_decided}/{comparison.n_examples} "
                f"required_win_rate={t_cst.BOSS_ROUND_MIN_WIN_RATE} "
                f"required_mean_gap={t_cst.BOSS_ROUND_MIN_MEAN_GAP_NATS} :: {comparison.reason}"
            )
            challenger_won, verdict_reason = challenger_takes_paired_task(comparison, boss_loss, opponent_loss)
            if not challenger_won:
                logger.info(f"Boss round task {task_id}: {verdict_reason}")
            # A draw carries no result. The defender is named so the persisted score row has a
            # holder, but is_draw takes the task out of the dethrone tally entirely and earns the
            # round an extra task in its place - nothing here separated the two models, and that is
            # a fact about the dataset rather than about the challenger.
            if comparison.is_draw:
                return BossRoundTaskVerdict(
                    winner_hotkey=boss_hotkey, decided_by=f"a draw ({verdict_reason})", is_draw=True
                )
            return BossRoundTaskVerdict(
                winner_hotkey=(opponent_hotkey if challenger_won else boss_hotkey),
                decided_by=f"per-example win rate ({verdict_reason})",
            )

    task_winner = (
        opponent_hotkey
        if challenger_beats_boss(boss_loss, opponent_loss, higher_is_better, threshold_percentage)
        else boss_hotkey
    )
    direction = "higher is better" if higher_is_better else "lower is better"
    winner_label = "opponent" if task_winner == opponent_hotkey else "boss"
    logger.info(
        f"{task_object.task_type} task ({direction}): {winner_label} wins at {threshold_percentage * 100:.1f}% "
        f"margin (boss={boss_loss:.6f}, opponent={opponent_loss:.6f})"
    )
    return BossRoundTaskVerdict(winner_hotkey=task_winner)


def _log_paired_comparison(task_id, task_type: TaskType, comparison: PairedLossComparison) -> None:
    logger.info(
        f"BOSS_ROUND_PAIRED task={task_id} type={task_type} "
        f"challenger_won={comparison.challenger_won} draw={comparison.is_draw} "
        f"win_rate={comparison.win_rate:.4f} win_rate_lb={comparison.win_rate_lower_bound:.4f} "
        f"mean_gap_nats={comparison.mean_gap_nats:.6f} mean_gap_lb={comparison.mean_gap_lower_bound:.6f} "
        f"decided={comparison.n_decided}/{comparison.n_examples} "
        f"required_win_rate={comparison.required_win_rate} "
        f"required_mean_gap={comparison.required_mean_gap_nats} :: {comparison.reason}"
    )


def _paired_verdict(
    task_id,
    comparison: PairedLossComparison,
    challenger_won: bool,
    verdict_reason: str,
    boss_hotkey: str,
    opponent_hotkey: str,
    within_allowed_loss: bool = False,
) -> BossRoundTaskVerdict:
    if not challenger_won:
        logger.info(f"Boss round task {task_id}: {verdict_reason}")
    # A draw goes to the defender, who holds the persisted score row; for quant it is simply not a win.
    if comparison.is_draw:
        return BossRoundTaskVerdict(
            winner_hotkey=boss_hotkey,
            decided_by=f"a draw ({verdict_reason})",
            is_draw=True,
            challenger_within_allowed_loss=True,
        )
    return BossRoundTaskVerdict(
        winner_hotkey=(opponent_hotkey if challenger_won else boss_hotkey),
        decided_by=f"per-example win rate ({verdict_reason})",
        challenger_within_allowed_loss=within_allowed_loss and not challenger_won,
    )


async def _resolve_quant_boss_round_task_winner(
    task_id,
    task_object,
    boss_hotkey: str,
    opponent_hotkey: str,
    boss_loss: float,
    opponent_loss: float,
    threshold_percentage: float,
    psql_db: PSQLDB,
) -> BossRoundTaskVerdict:
    """Decide one quant boss-round task on the scalar and per-example losses (the freeform slot is decided separately).

    ``decided_by`` describes what decided it, for the persisted score_reason, and is None when the relative margin
    decided it. ``is_draw`` marks a task nothing separated; it goes to the boss and is not a win for the challenger.

    Fixed quant losses are log-likelihoods, where a relative margin is the wrong scale and a scalar mean cannot
    separate a real win from held-out sampling noise: where both sides have per-example vectors those are compared
    example by example, the challenger's aligned onto the boss's SKU, and the challenger must also be
    QUANT_FIXED_BOSS_MIN_IMPROVEMENT better on the scalar. A fixed task where either side has no vector, or the vectors
    cannot be paired, stays with the boss. Only a task that is not a fixed one is decided on the relative margin.
    """
    if task_object.task_type == TaskType.QUANTFIXEDTASK:
        boss_losses, boss_fingerprint = await get_per_example_losses(task_id, boss_hotkey, psql_db)
        opponent_losses, opponent_fingerprint = await get_per_example_losses(task_id, opponent_hotkey, psql_db)

        fingerprints_disagree = (
            boss_fingerprint is not None and opponent_fingerprint is not None and boss_fingerprint != opponent_fingerprint
        )
        if boss_losses is None or opponent_losses is None:
            # A boss entry that is a scored fail (a non-finite loss) has no vector by nature: a challenger with a valid
            # loss takes the task, as it takes any task the boss failed.
            if not math.isfinite(boss_loss) and math.isfinite(opponent_loss):
                logger.info(f"Boss round task {task_id}: the boss's entry is a scored fail, the challenger takes the task")
                return BossRoundTaskVerdict(winner_hotkey=opponent_hotkey, decided_by="the boss's entry is a scored fail")
            # The paired test is part of the rule: without both vectors it cannot be run, so the boss keeps the task.
            missing = "boss" if boss_losses is None else "challenger"
            logger.error(
                f"Boss round task {task_id}: no per-example losses for {missing} "
                f"(absent or extraction failed at eval) - not counted as a win"
            )
            return BossRoundTaskVerdict(winner_hotkey=boss_hotkey, decided_by="no per-example losses to pair")
        elif fingerprints_disagree or len(boss_losses) != len(opponent_losses):
            logger.error(
                f"Boss round task {task_id}: per-example vectors present but not pairable "
                f"(boss_fingerprint={boss_fingerprint} challenger_fingerprint={opponent_fingerprint} "
                f"boss_n={len(boss_losses)} challenger_n={len(opponent_losses)}) - not counted as a win"
            )
            return BossRoundTaskVerdict(winner_hotkey=boss_hotkey, decided_by="unpairable per-example vectors")
        else:
            # A quant challenger scored on another SKU is compared on the boss's SKU scale; the scalars
            # (adjusted_loss) are already normalised by each side's own SKU reference.
            try:
                opponent_losses = await sku_aligned_losses(
                    task_object.task_type,
                    task_id,
                    boss_hotkey,
                    opponent_hotkey,
                    opponent_losses,
                    psql_db,
                    task_object.training_start_point,
                )
            except QuantReferenceUnavailable as e:
                logger.error(f"Boss round task {task_id}: {e} - not counted as a win")
                return BossRoundTaskVerdict(winner_hotkey=boss_hotkey, decided_by="no same-SKU reference to pair on")
            comparison = compare_paired_losses(boss_losses, opponent_losses)
            _log_paired_comparison(task_id, task_object.task_type, comparison)
            challenger_won, verdict_reason = quant_challenger_takes_fixed_boss_task(comparison, boss_loss, opponent_loss)
            within = False
            if not challenger_won and not comparison.is_draw:
                within, swapped = challenger_within_paired_allowed_loss(boss_losses, opponent_losses, boss_loss, opponent_loss)
                logger.info(
                    f"BOSS_ROUND_ALLOWED_LOSS task={task_id} within={within} boss_example_wins_swapped="
                    f"{swapped.challenger_example_wins} win_rate_lb={swapped.win_rate_lower_bound:.4f} "
                    f"mean_gap_lb={swapped.mean_gap_lower_bound:.6f} :: {swapped.reason}"
                )
            return _paired_verdict(
                task_id, comparison, challenger_won, verdict_reason, boss_hotkey, opponent_hotkey, within
            )

    task_winner = (
        opponent_hotkey
        if quant_challenger_beats_boss(boss_loss, opponent_loss, threshold_percentage)
        else boss_hotkey
    )
    winner_label = "opponent" if task_winner == opponent_hotkey else "boss"
    logger.info(
        f"{task_object.task_type} task (lower is better): {winner_label} wins at {threshold_percentage * 100:.1f}% "
        f"margin (boss={boss_loss:.6f}, opponent={opponent_loss:.6f})"
    )
    return BossRoundTaskVerdict(winner_hotkey=task_winner)


async def _resolve_freeform_boss_round_task(
    task_id: str,
    boss_result: MinerResultsText | None,
    opponent_result: MinerResultsText | None,
    psql_db: PSQLDB,
) -> BossRoundTaskVerdict:
    """Decide a freeform boss task on the bar and the byte margin, and persist the verdict."""
    margin = t_cst.QUANT_FREEFORM_BOSS_BYTE_MARGIN
    verdict = resolve_freeform_boss_task(boss_result, opponent_result, margin)
    if verdict.winner_hotkey == EMISSION_BURN_HOTKEY:
        verdict.challenger_within_allowed_loss = challenger_within_freeform_allowed_loss(boss_result, opponent_result)
    logger.info(
        f"Boss round freeform task {task_id}: {verdict.winner_hotkey} wins on {verdict.decided_by} "
        f"(challenger within allowed loss: {verdict.challenger_within_allowed_loss})"
    )
    await update_threshold_adjusted_quality_scores_for_task(
        task_id=task_id,
        winner_hotkey=verdict.winner_hotkey,
        threshold_percentage=margin,
        compared_hotkeys=[EMISSION_BURN_HOTKEY, opponent_result.hotkey if opponent_result else None],
        psql_db=psql_db,
        basis=verdict.decided_by,
    )
    return verdict


def _quant_boss_round_crowned(opponent_wins: int, built_size: int) -> tuple[bool, str]:
    """The QUANT dethrone rule. Returns (crowned, reason).

    The challenger is crowned iff it wins at least ``built_size - QUANT_BOSS_ROUND_ALLOWED_LOSSES`` tasks (5 of 6).
    ``built_size`` is the size the round was built to, never the number of tasks attached or decided: a task that is
    missing, undecided, drawn or without a result is not a win, and nothing is asked of the tasks it did not win.
    """
    required_wins = max(1, built_size - t_cst.QUANT_BOSS_ROUND_ALLOWED_LOSSES)
    if opponent_wins < required_wins:
        return False, f"challenger won {opponent_wins} of {built_size}, requires {required_wins}"
    return True, f"challenger won {opponent_wins} of {built_size}, requires {required_wins}"


def _determine_quant_boss_round_winner(task_winners: list[str], boss_hotkey: str, built_size: int) -> str:
    """The quant boss round: see _quant_boss_round_crowned. Every task the challenger did not win is the boss's here."""
    win_counts = Counter(task_winners)
    opponent_hotkey = next((hotkey for hotkey in win_counts if hotkey != boss_hotkey), None)
    opponent_wins = win_counts.get(opponent_hotkey, 0) if opponent_hotkey else 0

    crowned, reason = _quant_boss_round_crowned(opponent_wins, built_size)
    label = "Challenger wins the boss round" if crowned else "Boss retains title"
    logger.info(
        f"{TournamentType.QUANT.value} tournament: {label} - {reason} "
        f"({opponent_wins} won of {len(task_winners)} decided, {built_size} built)"
    )
    return opponent_hotkey if crowned and opponent_hotkey else boss_hotkey


def determine_boss_round_winner(
    task_winners: list[str],
    boss_hotkey: str,
    tournament_type: TournamentType,
    continuous_sft_outcomes: list[ContinuousSftLineageOutcome] | None = None,
    expected_task_count: int | None = None,
) -> str:
    """
    Determine the winner of a boss round based on task results and tournament type.

    Args:
        task_winners: List of hotkeys that won each task in the boss round. Drawn tasks are not in
            here - nothing separated the two models on them, so they carry no result.
        boss_hotkey: The defending champion's hotkey
        tournament_type: Type of tournament (TEXT, IMAGE or QUANT)
        continuous_sft_outcomes: How each continuous-SFT lineage in the round went. When non-empty
            (text boss round) the challenger must satisfy EVERY one to dethrone, on top of the
            overall threshold. Satisfied means won it *or* drew it — a draw is the task failing to
            separate the two models rather than the challenger failing to beat the boss, and should
            not cost the crown. A loss, or no result at all, blocks. Empty (image) leaves it off.
        expected_task_count: The size the boss round was built to. Sets the bar when draws left
            fewer decided tasks than that, so an undecidable task never *lowers* what it takes to
            dethrone — a challenger with two dead tasks must sweep the four that worked rather than
            getting in on three. Defaults to however many tasks actually decided.

    Returns:
        Hotkey of the boss round winner
    """
    if not task_winners:
        logger.error("No valid task winners found in boss round - all tasks failed to determine winners")
        logger.info(f"Defaulting to boss as winner due to evaluation failures: {boss_hotkey}")
        return boss_hotkey

    if tournament_type == TournamentType.QUANT:
        # QUANT is always measured against the built size, whatever expected_task_count the caller passed.
        return _determine_quant_boss_round_winner(
            task_winners, boss_hotkey, t_cst.expected_boss_round_task_count(TournamentType.QUANT)
        )

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

    # Both IMAGE and TEXT tournaments: challenger may lose at most one boss-round task. Each task
    # requires beating the boss by BOSS_ROUND_WIN_MARGIN. Measured against the round's built size
    # rather than the decided count so draws cannot slide the bar downward - they take a task out of
    # the numerator, and the challenger has to make it up on the ones that did decide.
    #
    # Clamped to the decided count, because the built size can exceed it once draws outlive their
    # single decider batch: 3 decided against a built size of 5 asked for 4 wins out of 3, which no
    # challenger could ever satisfy - a bar meant to stop draws helping the challenger had turned
    # into draws guaranteeing the boss. Sweeping every task that measured anything is always enough.
    unclamped_required = max(1, max(total_tasks, expected_task_count or 0) - 1)
    required_wins = min(unclamped_required, total_tasks)
    if unclamped_required > total_tasks:
        logger.warning(
            f"Boss round has only {total_tasks} decided task(s) against a built size of "
            f"{expected_task_count}; requiring a sweep of all {total_tasks} rather than an "
            f"unreachable {unclamped_required}. This round measured very little - worth a look at "
            f"whether the tie dead zone is too wide for the datasets being drawn."
        )

    # Continuous-SFT gate: challenger must satisfy EVERY lineage; only enforced when there are any.
    # Held per lineage rather than counted per task so that dropping a drawn lineage can never take
    # the total to zero and switch the strictest rule in the tournament off - a lineage stays in its
    # own gate whatever happened to it, and what varies is only whether it is satisfied.
    outcomes = continuous_sft_outcomes or []
    num_continuous_sft_tasks = len(outcomes)
    challenger_continuous_wins = sum(1 for outcome in outcomes if outcome.satisfied_by(opponent_hotkey))
    continuous_sft_ok = challenger_continuous_wins == num_continuous_sft_tasks
    for outcome in outcomes:
        if not outcome.satisfied_by(opponent_hotkey):
            logger.info(
                f"Continuous-SFT lineage {outcome.lineage} not satisfied: "
                f"winner={outcome.winner_hotkey} is_draw={outcome.is_draw}"
            )

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

async def _resolve_knockout_task_winner(task: TournamentTask, psql_db: PSQLDB) -> str | None:
    """Decide one knockout task, preferring per-example win rate over the mean loss.

    A knockout is a straight head to head between two competitors, so the winner is whichever won
    more of the examples both were scored on. Falls back to the mean-loss ranking when the task is
    not a paired type, when either side has no vector, or when the two vectors cannot be lined up.
    """
    task_object = await get_task(task.task_id, psql_db)
    if task_object is not None and task_object.task_type == TaskType.QUANTFIXEDTASK:
        return await _resolve_quant_knockout_task_winner(task, task_object, psql_db)
    if task_object is None or task_object.task_type not in t_cst.PAIRED_BOSS_ROUND_TASK_TYPES:
        return await get_task_winner(task.task_id, psql_db)

    # Eligibility comes from the PERSISTED quality score, which is what get_task_winner used and
    # what carries SCORE_PENALTY for a submission rejected as a non-finetune. It cannot come from
    # the results themselves: get_task_results_for_ranking hardcodes is_finetune=True, and the
    # scores calculate_miner_ranking_and_scores recomputes award only the rank-1 miner - a score>0
    # filter on those would drop the runner-up of every two-way contest and kill this path.
    eligible = await get_eligible_hotkeys_for_task(task.task_id, psql_db)
    ranked = [
        result
        for result in calculate_miner_ranking_and_scores(await get_task_results_for_ranking(task.task_id, psql_db))
        if result.hotkey in eligible and not np.isnan(result.test_loss)
    ]
    if len(ranked) != 2:
        # Byes, failed evaluations and anything that is not a clean two-way contest.
        return await get_task_winner(task.task_id, psql_db)

    # Sorted, not left in DB row order: paired_head_to_head_winner breaks an exact tie on mean loss
    # in favour of its first argument, so row order would otherwise decide the task.
    ranked.sort(key=lambda result: (result.test_loss, result.hotkey))
    first, second = ranked[0], ranked[1]
    first_losses, first_fingerprint = await get_per_example_losses(task.task_id, first.hotkey, psql_db)
    second_losses, second_fingerprint = await get_per_example_losses(task.task_id, second.hotkey, psql_db)

    pairable = (
        first_losses is not None
        and second_losses is not None
        and len(first_losses) == len(second_losses)
        and not (
            first_fingerprint is not None and second_fingerprint is not None and first_fingerprint != second_fingerprint
        )
    )
    if not pairable:
        logger.info(f"Knockout task {task.task_id}: no usable per-example vectors, ranking on mean loss")
        return await get_task_winner(task.task_id, psql_db)

    winner, description = paired_head_to_head_winner(
        hotkey_a=first.hotkey,
        a_losses=first_losses,
        a_mean_loss=first.test_loss,
        hotkey_b=second.hotkey,
        b_losses=second_losses,
        b_mean_loss=second.test_loss,
    )

    # The vectors are raw cross-entropy, but from round 2 knockout instruct tasks are KL-weighted
    # (task_creator: enable_kl=round_number != 1) and test_loss then carries a per-model penalty
    # the vectors deliberately exclude. Advancing on raw-CE win rate alone would promote a model
    # that is clearly worse on the metric the task actually ranks on, so the sample winner must
    # also not be worse on the ranking scalar. The boss round guards this the same way.
    winner_scalar, loser_scalar = (
        (first.test_loss, second.test_loss) if winner == first.hotkey else (second.test_loss, first.test_loss)
    )
    if winner_scalar > loser_scalar:
        fallback = first.hotkey if first.test_loss <= second.test_loss else second.hotkey
        logger.info(
            f"KNOCKOUT_PAIRED task={task.task_id} sample winner {winner} is worse on the ranking loss "
            f"({winner_scalar:.6f} vs {loser_scalar:.6f}) - advancing {fallback} on mean loss instead"
        )
        # Operands ordered winner-first: this string is persisted as the winner's score_reason.
        winner, description = fallback, f"mean loss ({loser_scalar:.6f} vs {winner_scalar:.6f})"

    logger.info(
        f"KNOCKOUT_PAIRED task={task.task_id} type={task_object.task_type} winner={winner} :: {description} "
        f"(mean losses {first.hotkey}={first.test_loss:.6f} {second.hotkey}={second.test_loss:.6f})"
    )

    # Persist the verdict. Everything that reconstructs a bracket after the fact - audit data,
    # get_task_winners - reads task_nodes.quality_score, which still holds the mean-loss ranking.
    # Without this an auditor recomputing the round disagrees with the validator on exactly the
    # tasks where win rate and mean loss diverge.
    await update_threshold_adjusted_quality_scores_for_task(
        task_id=task.task_id,
        winner_hotkey=winner,
        threshold_percentage=0.0,
        compared_hotkeys=[first.hotkey, second.hotkey],
        psql_db=psql_db,
        basis=f"per-example win rate ({description})",
    )
    return winner


async def _resolve_quant_knockout_task_winner(task: TournamentTask, task_object, psql_db: PSQLDB) -> str | None:
    """Decide one quant knockout task: per-example win rate on the first row's SKU scale, else the mean-loss ranking."""
    # Eligibility comes from the PERSISTED quality score, which is what get_task_winner used and
    # what carries SCORE_PENALTY for a submission rejected as a non-finetune. It cannot come from
    # the results themselves: get_task_results_for_ranking hardcodes is_finetune=True, and the
    # scores calculate_miner_ranking_and_scores recomputes award only the rank-1 miner - a score>0
    # filter on those would drop the runner-up of every two-way contest and kill this path.
    eligible = await get_eligible_hotkeys_for_task(task.task_id, psql_db)
    ranked = [
        result
        for result in calculate_miner_ranking_and_scores(await get_task_results_for_ranking(task.task_id, psql_db))
        if result.hotkey in eligible and not np.isnan(result.test_loss)
    ]
    if len(ranked) != 2:
        # Byes, failed evaluations and anything that is not a clean two-way contest.
        return await get_task_winner(task.task_id, psql_db)

    # Sorted, not left in DB row order: paired_head_to_head_winner breaks an exact tie on mean loss
    # in favour of its first argument, so row order would otherwise decide the task. On adjusted_loss,
    # which is test_loss except on quant tasks, where it is normalised by the row's own SKU reference.
    ranked.sort(key=lambda result: (result.adjusted_loss, result.hotkey))
    first, second = ranked[0], ranked[1]
    first_losses, first_fingerprint = await get_per_example_losses(task.task_id, first.hotkey, psql_db)
    second_losses, second_fingerprint = await get_per_example_losses(task.task_id, second.hotkey, psql_db)

    pairable = (
        first_losses is not None
        and second_losses is not None
        and len(first_losses) == len(second_losses)
        and not (
            first_fingerprint is not None and second_fingerprint is not None and first_fingerprint != second_fingerprint
        )
    )
    if pairable:
        try:
            second_losses = await sku_aligned_losses(
                task_object.task_type,
                task.task_id,
                first.hotkey,
                second.hotkey,
                second_losses,
                psql_db,
                task_object.training_start_point,
            )
        except QuantReferenceUnavailable as e:
            logger.error(f"Knockout task {task.task_id}: {e}")
            pairable = False
    if not pairable:
        logger.info(f"Knockout task {task.task_id}: no usable per-example vectors, ranking on mean loss")
        return await get_task_winner(task.task_id, psql_db)

    winner, description = paired_head_to_head_winner(
        hotkey_a=first.hotkey,
        a_losses=first_losses,
        a_mean_loss=first.adjusted_loss,
        hotkey_b=second.hotkey,
        b_losses=second_losses,
        b_mean_loss=second.adjusted_loss,
    )

    # The vectors are raw cross-entropy, but knockout instruct tasks may be KL-weighted
    # (task_creator: enable_kl=round_number != 1) and test_loss then carries a per-model penalty
    # the vectors deliberately exclude. Advancing on raw-CE win rate alone would promote a model
    # that is clearly worse on the metric the task actually ranks on, so the sample winner must
    # also not be worse on the ranking scalar. The boss round guards this the same way.
    winner_scalar, loser_scalar = (
        (first.adjusted_loss, second.adjusted_loss)
        if winner == first.hotkey
        else (second.adjusted_loss, first.adjusted_loss)
    )
    if winner_scalar > loser_scalar:
        fallback = first.hotkey if first.adjusted_loss <= second.adjusted_loss else second.hotkey
        logger.info(
            f"KNOCKOUT_PAIRED task={task.task_id} sample winner {winner} is worse on the ranking loss "
            f"({winner_scalar:.6f} vs {loser_scalar:.6f}) - advancing {fallback} on mean loss instead"
        )
        # Operands ordered winner-first: this string is persisted as the winner's score_reason.
        winner, description = fallback, f"mean loss ({loser_scalar:.6f} vs {winner_scalar:.6f})"

    logger.info(
        f"KNOCKOUT_PAIRED task={task.task_id} type={task_object.task_type} winner={winner} :: {description} "
        f"(mean losses {first.hotkey}={first.adjusted_loss:.6f} {second.hotkey}={second.adjusted_loss:.6f})"
    )

    # Persist the verdict. Everything that reconstructs a bracket after the fact - audit data,
    # get_task_winners - reads task_nodes.quality_score, which still holds the mean-loss ranking.
    # Without this an auditor recomputing the round disagrees with the validator on exactly the
    # tasks where win rate and mean loss diverge.
    await update_threshold_adjusted_quality_scores_for_task(
        task_id=task.task_id,
        winner_hotkey=winner,
        threshold_percentage=0.0,
        compared_hotkeys=[first.hotkey, second.hotkey],
        psql_db=psql_db,
        basis=f"per-example win rate ({description})",
    )
    return winner


def _resolve_boss_round_draws(
    drawn_task_ids: list[str],
    decider_task_types: list[TaskType],
    round_tasks: list[TournamentTask],
    tournament_type: TournamentType,
) -> BossRoundDrawResolution:
    """Decide whether a boss round's draws earn it replacement tasks.

    One replacement per drawn task, added in a single batch, and only once. The cap needs no state
    of its own: the boss round is built to a fixed size and prep-failure replacement swaps tasks
    without changing the count, so a round holding more tasks than its type calls for is a round
    that has already had its deciders added. A decider that itself draws is therefore just excluded
    and the round resolves on whatever was decided.
    """
    if not drawn_task_ids:
        return BossRoundDrawResolution(reason="No drawn tasks")

    expected = t_cst.expected_boss_round_task_count(tournament_type)
    already_added = len(round_tasks) > expected if expected else False

    if already_added:
        return BossRoundDrawResolution(
            drawn_task_ids=drawn_task_ids,
            decider_task_types=decider_task_types,
            can_add_deciders=False,
            reason=(
                f"{len(drawn_task_ids)} drawn task(s) excluded from the tally; the round already holds "
                f"{len(round_tasks)} tasks against an expected {expected}, so its deciders were already "
                f"added and it resolves on what was decided"
            ),
        )

    return BossRoundDrawResolution(
        drawn_task_ids=drawn_task_ids,
        decider_task_types=decider_task_types,
        can_add_deciders=True,
        reason=(
            f"{len(drawn_task_ids)} boss-round task(s) drew - nothing separated the two models on "
            f"them - so each is excluded from the tally and gets one replacement task: "
            f"{[t.value for t in decider_task_types]}"
        ),
    )


async def _get_quant_boss_round_outcome(
    completed_round: TournamentRoundData,
    round_tasks: list[TournamentTask],
    tournament: TournamentData,
    psql_db: PSQLDB,
) -> RoundOutcome:
    """Resolve a completed quant boss round: six compression tasks, the challenger must win five of them.

    Freeform is decided on the cap and effective bytes, fixed through _resolve_quant_boss_round_task_winner. A draw is
    the boss's and not a win. No task draws replacements: nothing here adds deciders. A task that was never attached or
    never decided is not a win either, and the bar stays the built size (see _quant_boss_round_crowned).
    """
    boss_hotkey = EMISSION_BURN_HOTKEY
    opponent_hotkey = None
    task_winners = []

    # Challenger must beat the boss by BOSS_ROUND_WIN_MARGIN on each relative-margin task to win it.
    threshold_percentage = t_cst.BOSS_ROUND_WIN_MARGIN
    logger.info(f"Boss round using {threshold_percentage * 100:.1f}% win margin per task")

    boss_side_failures = []  # tasks of boss-round seeds that are dead on the boss's side (resolved after the loop)
    for task in round_tasks:
        logger.info(f"Processing boss round task {task.task_id}")

        task_object = await get_task(task.task_id, psql_db)

        seed_role = (
            await boss_side_seed_failure(task.task_id, task_object.status, psql_db) if task_object is not None else None
        )
        if seed_role is not None:
            # The boss's upfront entry is dead (never retrained): the slot is the boss's failure, the challenger takes it.
            logger.error(
                f"Boss round task {task.task_id}: boss-side seed {seed_role.value} failed prep after attach "
                f"(status {task_object.status}); the challenger takes the slot"
            )
            boss_side_failures.append(task_object)
            continue

        miner_results = await get_task_results_for_ranking(task.task_id, psql_db)
        if not miner_results:
            logger.warning(f"No valid results for boss round task {task.task_id}. Winner is base contestant.")
            task_winners.append(boss_hotkey)
            continue

        ranked_results = calculate_miner_ranking_and_scores(miner_results)

        boss_result = None
        opponent_result = None
        for result in ranked_results:
            if result.hotkey == boss_hotkey:
                boss_result = result
            elif opponent_result is None:
                opponent_result = result
        opponent_hotkey = opponent_result.hotkey if opponent_result else None

        # Decided on the bar and bytes, never on loss: a freeform scored fail carries a NaN loss
        # but is still a result, and resolves as a loss for its side.
        if task_object is not None and task_object.task_type == TaskType.QUANTFREEFORMTASK:
            verdict = await _resolve_freeform_boss_round_task(task.task_id, boss_result, opponent_result, psql_db)
            task_winners.append(verdict.winner_hotkey)
            continue

        boss_loss = boss_result.adjusted_loss if boss_result else None
        opponent_loss = opponent_result.adjusted_loss if opponent_result else None

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

        verdict = await _resolve_boss_round_task_winner(
            task_id=task.task_id,
            task_object=task_object,
            boss_hotkey=boss_hotkey,
            opponent_hotkey=opponent_hotkey,
            boss_loss=boss_loss,
            opponent_loss=opponent_loss,
            threshold_percentage=threshold_percentage,
            psql_db=psql_db,
        )
        task_winners.append(verdict.winner_hotkey)

        # Persisted for a draw too: the score row needs a holder, and the reason says it was a draw.
        await update_threshold_adjusted_quality_scores_for_task(
            task_id=task.task_id,
            winner_hotkey=verdict.winner_hotkey,
            threshold_percentage=threshold_percentage,
            compared_hotkeys=[boss_hotkey, opponent_hotkey],
            psql_db=psql_db,
            basis=verdict.decided_by,
        )

    for task_object in boss_side_failures:
        if opponent_hotkey is None:
            assigned = {node.hotkey for node in await get_nodes_assigned_to_task(task_object.task_id, psql_db)}
            opponent_hotkey = next(iter(sorted(assigned - {boss_hotkey})), None)
        if opponent_hotkey is None:
            logger.error(f"Boss round task {task_object.task_id}: no challenger found for a boss-side failure; boss keeps it")
            task_winners.append(boss_hotkey)
            continue
        task_winners.append(opponent_hotkey)
        # Persisted like every other decided task, so the task has a score holder.
        await update_threshold_adjusted_quality_scores_for_task(
            task_id=task_object.task_id,
            winner_hotkey=opponent_hotkey,
            threshold_percentage=threshold_percentage,
            compared_hotkeys=[boss_hotkey, opponent_hotkey],
            psql_db=psql_db,
            basis="boss-side seed failure (the boss's entry never ran)",
        )

    boss_round_winner = determine_boss_round_winner(
        task_winners,
        boss_hotkey,
        tournament.tournament_type,
        expected_task_count=t_cst.expected_boss_round_task_count(tournament.tournament_type) or None,
    )
    return RoundOutcome(winners=[boss_round_winner])


async def _text_task_scores(
    task_id: str, boss_hotkey: str, challenger_hotkey: str, psql_db: PSQLDB
) -> tuple[float | None, float | None]:
    """(boss, challenger) adjusted loss on one task, as the legacy boss round reads them; None for a side with no valid
    result or whose training failed."""
    miner_results = await get_task_results_for_ranking(task_id, psql_db)
    ranked = calculate_miner_ranking_and_scores(miner_results) if miner_results else []
    scores = {result.hotkey: result.adjusted_loss for result in ranked}
    statuses = await get_training_status_for_task_and_hotkeys(task_id, [boss_hotkey, challenger_hotkey], psql_db)

    def side(hotkey: str) -> float | None:
        return None if statuses.get(hotkey) == TrainingStatus.FAILURE else scores.get(hotkey)

    return side(boss_hotkey), side(challenger_hotkey)


async def _text_task_head_to_head(
    task_id: str, label: str, task_type: TaskType, challenger_hotkey: str, psql_db: PSQLDB
) -> TextHeadToHead:
    boss_score, challenger_score = await _text_task_scores(task_id, EMISSION_BURN_HOTKEY, challenger_hotkey, psql_db)
    return text_head_to_head(task_id, label, task_type, boss_score, challenger_score)


async def seeded_text_round_head_to_heads(
    tournament_id: str, challenger_hotkey: str, psql_db: PSQLDB
) -> list[TextHeadToHead]:
    """The challenger's head-to-heads against the boss on the tasks of rounds 1-3 (the seed task each round attached, on
    which both players have a result). Raises when a round or its task is missing: that is a data fault, never a loss."""
    rounds = await get_tournament_rounds(tournament_id, psql_db)
    results: list[TextHeadToHead] = []
    for spec in t_cst.TEXT_ROUND_PLAN:
        played = next((round_data for round_data in rounds if round_data.round_number == spec.round_number), None)
        tasks = await get_tournament_tasks(played.round_id, psql_db) if played else []
        if len(tasks) != 1:
            raise ValueError(
                f"Tournament {tournament_id} round {spec.round_number} holds {len(tasks)} task(s), expected its one seed task"
            )
        task_object = await get_task(tasks[0].task_id, psql_db)
        if task_object is None:
            raise ValueError(f"Tournament {tournament_id} round {spec.round_number} task {tasks[0].task_id} is gone")
        results.append(
            await _text_task_head_to_head(
                tasks[0].task_id, f"round {spec.round_number}", task_object.task_type, challenger_hotkey, psql_db
            )
        )
    return results


async def get_seeded_text_runner_up(tournament_id: str, challenger_hotkey: str, psql_db: PSQLDB) -> str | None:
    """Third place of a seeded text tournament: the next miner after the challenger in round 3's standings, the order
    that picked the round's winner. None when round 3 or another ranked miner is missing."""
    final_miner_round = t_cst.TEXT_ROUND_PLAN[-1].round_number
    rounds = await get_tournament_rounds(tournament_id, psql_db)
    played = next((round_data for round_data in rounds if round_data.round_number == final_miner_round), None)
    if played is None:
        return None
    ordered = await _text_seeded_ordered_hotkeys(await get_tournament_tasks(played.round_id, psql_db), psql_db)
    return next((hotkey for hotkey in ordered if hotkey not in (challenger_hotkey, EMISSION_BURN_HOTKEY)), None)


async def _text_boss_round_challenger(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB
) -> str | None:
    """The boss round's challenger: the non-boss member of its pair, else the non-boss node assigned to its tasks."""
    for pair in await get_tournament_pairs(completed_round.round_id, psql_db):
        challenger = next((hotkey for hotkey in (pair.hotkey1, pair.hotkey2) if hotkey != EMISSION_BURN_HOTKEY), None)
        if challenger:
            return challenger
    for task in round_tasks:
        assigned = {node.hotkey for node in await get_nodes_assigned_to_task(task.task_id, psql_db)}
        challenger = next(iter(sorted(assigned - {EMISSION_BURN_HOTKEY})), None)
        if challenger:
            return challenger
    return None


def _text_boss_round_label(task_object) -> str:
    if t_cst.is_continuous_sft_task(task_object):
        return "boss round continuous"
    if is_text_big_task(task_object.task_type):
        return "boss round big task"
    return f"boss round {task_object.task_type.value}"


async def _get_seeded_text_boss_round_outcome(
    completed_round: TournamentRoundData,
    round_tasks: list[TournamentTask],
    tournament: TournamentData,
    psql_db: PSQLDB,
) -> RoundOutcome:
    """Resolve the boss round of a seeded text tournament on the crown rule (see text_crown).

    The head-to-heads are rounds 1-3 plus every boss-round task present. While the big task is not in the round the
    verdict is "can the challenger still be crowned": if so the outcome is deferred with needs_text_big_task so the caller
    adds the big task, otherwise the boss is retained and the big task is never created. Once the big task is in, all seven
    are decided and the verdict is final. The persisted per-task winners cover the boss-round tasks only: rounds 1-3 keep
    the scores their own rankings stored.
    """
    boss_hotkey = EMISSION_BURN_HOTKEY
    challenger = await _text_boss_round_challenger(completed_round, round_tasks, psql_db)
    if challenger is None:
        logger.error(f"Boss round {completed_round.round_id}: no challenger found; the boss retains")
        return RoundOutcome(winners=[boss_hotkey])

    earlier_rounds = await seeded_text_round_head_to_heads(tournament.tournament_id, challenger, psql_db)
    boss_round: list[tuple[str, TextHeadToHead]] = []
    for task in round_tasks:
        task_object = await get_task(task.task_id, psql_db)
        if task_object is None:
            logger.error(f"Boss round task {task.task_id} is gone; it cannot be decided")
            continue
        boss_round.append(
            (
                task.task_id,
                await _text_task_head_to_head(
                    task.task_id, _text_boss_round_label(task_object), task_object.task_type, challenger, psql_db
                ),
            )
        )

    decided = earlier_rounds + [head_to_head for _, head_to_head in boss_round]
    verdict = text_crown_verdict(decided)
    for head_to_head in decided:
        logger.info(
            f"Text crown {head_to_head.label} task {head_to_head.task_id} ({head_to_head.task_type.value}): "
            f"boss={head_to_head.boss_score} challenger={head_to_head.challenger_score} -> {head_to_head.describe()}"
        )

    big_task_present = any(is_text_big_task(head_to_head.task_type) for _, head_to_head in boss_round)
    stage_one_tasks = sum(1 for _, head_to_head in boss_round if not is_text_big_task(head_to_head.task_type))
    stage_one_expected = 1 + len(t_cst.TEXT_BOSS_ROUND_STAGE_ONE_TASK_TYPES)

    if not big_task_present:
        if verdict.crowned_if_complete and stage_one_tasks == stage_one_expected:
            reason = f"stage 1 scored, the challenger can still be crowned ({verdict.reason}); the big task is needed"
            logger.info(f"Text crown, boss round {completed_round.round_id}: {reason}")
            return RoundOutcome(deferred_reason=reason, needs_text_big_task=True)
        if stage_one_tasks != stage_one_expected:
            logger.error(
                f"Text crown, boss round {completed_round.round_id}: holds {stage_one_tasks} stage-1 task(s), expected "
                f"{stage_one_expected}; the boss retains"
            )
        else:
            logger.info(
                f"Text crown, boss round {completed_round.round_id}: the challenger cannot be crowned ({verdict.reason}); "
                f"the boss retains and the big task is never created"
            )
        winners = [boss_hotkey]
    else:
        winners = [challenger] if verdict.crowned else [boss_hotkey]
        outcome = "challenger crowned" if verdict.crowned else "boss retains"
        incomplete = "" if verdict.complete else f" (only {verdict.decided}/{t_cst.TEXT_CROWN_TASKS} tasks decided)"
        logger.info(f"Text crown, boss round {completed_round.round_id}: {outcome}{incomplete} - {verdict.reason}")

    for task_id, head_to_head in boss_round:
        await update_threshold_adjusted_quality_scores_for_task(
            task_id=task_id,
            winner_hotkey=challenger if head_to_head.is_win else boss_hotkey,
            threshold_percentage=t_cst.TEXT_CROWN_WIN_MARGIN,
            compared_hotkeys=[boss_hotkey, challenger],
            psql_db=psql_db,
            basis=f"the text crown rule ({head_to_head.describe()})",
        )
    return RoundOutcome(winners=winners)


async def get_knockout_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB, config: Config
) -> RoundOutcome:
    """Get winners from knockout round.

    Returns a deferred outcome when a boss round drew tasks and needs replacements added before it
    can be resolved; the caller must add those tasks rather than advance.
    """
    winners = []
    draw_resolution: BossRoundDrawResolution | None = None

    if not completed_round.is_final_round:
        for task in round_tasks:
            winner = await _resolve_knockout_task_winner(task, psql_db)
            if winner:
                winners.append(winner)
    else:
        # Boss round. Challenger must beat the boss by BOSS_ROUND_WIN_MARGIN per task.
        boss_hotkey = EMISSION_BURN_HOTKEY
        opponent_hotkey = None
        task_winners = []
        # How each continuous-SFT lineage went, feeding determine_boss_round_winner's dethrone gate.
        # Keyed by lineage so a lineage can never drop out of its own gate: what varies is whether
        # it ends up satisfied, not whether it is counted.
        continuous_sft_outcomes: dict[str, ContinuousSftLineageOutcome] = {}
        # Tasks that separated nothing. Excluded from every tally below and handed to the caller,
        # which adds one replacement task per draw so the round still resolves on a full set of
        # decided results. See BossRoundDrawResolution.
        drawn_task_ids: list[str] = []
        decider_task_types: list[TaskType] = []

        def _is_continuous_sft(task_obj) -> bool:
            return task_obj is not None and t_cst.is_continuous_sft_task(task_obj)

        def _lineage_outcome(task_obj) -> ContinuousSftLineageOutcome | None:
            """Register this lineage as present in the round, defaulting it to no result.

            Idempotent, so it is safe to call unconditionally per task and again when the task is
            awarded or recorded as a draw.
            """
            if not _is_continuous_sft(task_obj):
                return None
            lineage = t_cst.continuous_sft_lineage_from_ds(task_obj.ds)
            if lineage is None:
                return None
            return continuous_sft_outcomes.setdefault(lineage, ContinuousSftLineageOutcome(lineage=lineage))

        def _award(winner: str | None, task_obj) -> None:
            task_winners.append(winner)
            outcome = _lineage_outcome(task_obj)
            if outcome is not None and winner is not None:
                outcome.winner_hotkey = winner

        def _record_draw(task_id: str, task_obj) -> None:
            """Take a drawn task out of the tally and queue a stand-in for it.

            The stand-in mirrors the drawn task's type so the boss round keeps the mix it was built
            with, with one exception: a drawn continuous-SFT task is stood in for by a plain instruct
            task on a fresh dataset pull, not by another chunk of the same lineage. Two tasks on one
            lineage would share a single sequential train_index and leave the chain's carry-forward
            picking between them, and the lineage does not need a second attempt anyway - a draw
            already satisfies its dethrone gate. The stand-in is only there to top the tally back up.
            """
            drawn_task_ids.append(task_id)
            outcome = _lineage_outcome(task_obj)
            if outcome is not None:
                outcome.is_draw = True
            # Branch on what the task IS, not on whether its lineage parsed. _lineage_outcome
            # returns None for both "not continuous-SFT" and "continuous-SFT whose lineage is no
            # longer recognised" - a state warn_orphaned_continuous_sft_state exists to flag, and
            # one an edit to CONTINUOUS_SFT_LINEAGES between rounds produces. Treating the second as
            # the first queues a CHATTASK decider, which has no creation route, so the round retries
            # a task it can never build every 60 seconds forever.
            if _is_continuous_sft(task_obj):
                decider_task_types.append(TaskType.INSTRUCTTEXTTASK)
            else:
                decider_task_types.append(task_obj.task_type)

        # Get tournament info to determine the current champion and their consecutive wins
        tournament = await get_tournament(completed_round.tournament_id, psql_db)
        if not tournament:
            logger.error(f"Could not find tournament {completed_round.tournament_id}")
            return RoundOutcome()

        # Challenger must beat the boss by BOSS_ROUND_WIN_MARGIN on each task to win it.
        threshold_percentage = t_cst.BOSS_ROUND_WIN_MARGIN
        logger.info(f"Boss round using {threshold_percentage * 100:.1f}% win margin per task")

        # A task with nobody assigned to it never ran, so it holds no evidence about either side.
        # Scoring around it would charge somebody a loss on a task that never happened - and the
        # only way one gets here is a decider that was created but never made it back to PENDING for
        # node assignment (a crash, or a failure partway through creating a batch). Defer instead and
        # let the caller return the round to PENDING, which is what assigns them.
        #
        # Counted from task_nodes rather than get_nodes_assigned_to_task, which joins nodes: that
        # table is emptied and rebuilt for the netuid on every metagraph refresh, so a competitor
        # deregistering mid-tournament would make its tasks read as never assigned. This check would
        # then send the round back to PENDING, where assignment cannot find the missing node either,
        # and the round would stall for good - turning a guard against mis-scoring one task into a
        # wedge on the whole tournament.
        unassigned_task_ids = [
            task.task_id for task in round_tasks if not await count_hotkeys_assigned_to_task(task.task_id, psql_db)
        ]
        if unassigned_task_ids:
            reason = (
                f"{len(unassigned_task_ids)} boss-round task(s) have no assigned nodes and cannot have "
                f"run ({unassigned_task_ids}); returning the round to node assignment rather than "
                f"resolving around them"
            )
            logger.error(f"Boss round {completed_round.round_id}: {reason}")
            return RoundOutcome(deferred_reason=reason, unassigned_task_ids=unassigned_task_ids)

        if tournament.tournament_type == TournamentType.QUANT:
            return await _get_quant_boss_round_outcome(completed_round, round_tasks, tournament, psql_db)

        # A seeded text boss round is decided on its own crown rule and never draws, so it never reaches the draw tally below.
        if tournament.tournament_type == TournamentType.TEXT and await is_seeded_text_tournament(
            tournament.tournament_id, psql_db
        ):
            return await _get_seeded_text_boss_round_outcome(completed_round, round_tasks, tournament, psql_db)

        for task in round_tasks:
            logger.info(f"Processing boss round task {task.task_id}")

            task_object = await get_task(task.task_id, psql_db)

            # Register the lineage before anything can skip the task. Several branches below bail
            # out with a bare `continue` - notably "both evaluation failed" - and if registration
            # rode on the award path those would leave the lineage out of continuous_sft_outcomes
            # entirely, taking the gate's count to zero and switching the strictest rule in the
            # tournament off. A lineage is in its own gate the moment its task exists; all that
            # varies afterwards is whether it ends up satisfied.
            _lineage_outcome(task_object)

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

            verdict = await _resolve_boss_round_task_winner(
                task_id=task.task_id,
                task_object=task_object,
                boss_hotkey=boss_hotkey,
                opponent_hotkey=opponent_hotkey,
                boss_loss=boss_loss,
                opponent_loss=opponent_loss,
                threshold_percentage=threshold_percentage,
                psql_db=psql_db,
            )
            if verdict.is_draw:
                _record_draw(task.task_id, task_object)
            else:
                _award(verdict.winner_hotkey, task_object)

            # Persisted even for a draw: the score row still needs a holder, and the reason says it
            # was a draw. Only the dethrone tally ignores it.
            await update_threshold_adjusted_quality_scores_for_task(
                task_id=task.task_id,
                winner_hotkey=verdict.winner_hotkey,
                threshold_percentage=threshold_percentage,
                compared_hotkeys=[boss_hotkey, opponent_hotkey],
                psql_db=psql_db,
                basis=verdict.decided_by,
            )

        draw_resolution = _resolve_boss_round_draws(
            drawn_task_ids=drawn_task_ids,
            decider_task_types=decider_task_types,
            round_tasks=round_tasks,
            tournament_type=tournament.tournament_type,
        )
        if draw_resolution.needs_deciders:
            logger.info(
                f"Boss round {completed_round.round_id} deferred: {draw_resolution.reason}"
            )
            return RoundOutcome(deferred_reason=draw_resolution.reason, draw_resolution=draw_resolution)

        if draw_resolution.drawn_task_ids:
            logger.info(f"Boss round {completed_round.round_id}: {draw_resolution.reason}")

        # Pinned only when tasks were actually dropped for drawing, which by construction is text
        # only. Everything else keeps deriving the bar from its own resolved task count.
        expected_task_count = (
            t_cst.expected_boss_round_task_count(tournament.tournament_type) if draw_resolution.drawn_task_ids else None
        )
        boss_round_winner = determine_boss_round_winner(
            task_winners,
            boss_hotkey,
            tournament.tournament_type,
            list(continuous_sft_outcomes.values()),
            expected_task_count=expected_task_count,
        )

        winners = [boss_round_winner]

    return RoundOutcome(winners=winners, draw_resolution=draw_resolution)


async def find_groups_with_no_valid_scores(
    completed_round: TournamentRoundData, psql_db: PSQLDB
) -> list[EmptyScoreGroup]:
    """Group tasks that finished with participants but not a single usable score.

    A group can legitimately advance nobody — the boss retains, everyone is eliminated on a
    tie — but it can never legitimately produce *no scores at all*. That means the evaluation
    data was lost or never written, so any winner set derived from this round is wrong: the
    silent `continue` in the winner functions below would just drop the group and advance a
    smaller field. Callers use this to refuse advancement instead.

    Knockout rounds are decided per-pair rather than per-group, so they are not covered here.
    """
    if completed_round.round_type != RoundType.GROUP:
        return []

    round_tasks = await get_tournament_tasks(completed_round.round_id, psql_db)
    empty_groups: list[EmptyScoreGroup] = []

    for task in round_tasks:
        if not task.group_id:
            continue

        participants = await get_tournament_group_members(task.group_id, psql_db)
        if not participants:
            continue

        miner_results = await get_task_results_for_ranking(task.task_id, psql_db)
        ranked_results = calculate_miner_ranking_and_scores(miner_results) if miner_results else []
        has_valid_score = any(
            result.adjusted_loss is not None and not np.isnan(result.adjusted_loss) for result in ranked_results
        )

        if not has_valid_score:
            empty_groups.append(
                EmptyScoreGroup(
                    group_id=task.group_id,
                    task_id=task.task_id,
                    participants=len(participants),
                )
            )

    return empty_groups


async def _small_tournament_ordered_hotkeys(round_tasks: list[TournamentTask], psql_db: PSQLDB) -> list[str]:
    """Full cumulative ranking for a multi-match single-group round (text R1; image R1)."""
    match_rankings: list[MatchRanking] = []
    match_losses: list[dict[str, float]] = []
    competitors: set[str] = set()
    for task in round_tasks:
        miner_results = await get_task_results_for_ranking(task.task_id, psql_db)
        if not miner_results:
            logger.warning(f"No valid results for multi-match group task {task.task_id}")
            continue

        ranked_results = calculate_miner_ranking_and_scores(miner_results)
        scored = [
            (result.hotkey, result.adjusted_loss)
            for result in ranked_results
            if result.adjusted_loss is not None and not np.isnan(result.adjusted_loss)
        ]
        if not scored:
            logger.warning(f"Multi-match group task {task.task_id} has no valid scores")
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
        if hotkey == EMISSION_BURN_HOTKEY:
            continue
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
    logger.info(
        f"Multi-match group standings "
        f"{[(s.hotkey, round(s.average_rank, 3), round(s.summed_loss, 4), s.has_error) for s in ordered]}"
    )
    return [standing.hotkey for standing in ordered]


async def _get_small_tournament_group_winners(round_tasks: list[TournamentTask], psql_db: PSQLDB) -> list[str]:
    """Rank competitors across a multi-match single-group round (text R1; image R1)."""
    ordered = await _small_tournament_ordered_hotkeys(round_tasks, psql_db)
    winners = ordered[: t_cst.SMALL_TOURNAMENT_ADVANCE]
    logger.info(f"advancing top {len(winners)}: {winners}")
    return winners


def take_standings_slots(ordered: list[str], exclude: set[str], slots: int) -> list[str]:
    """Walk a cumulative ranking, skipping excluded hotkeys, until ``slots`` are filled."""
    return [hotkey for hotkey in ordered if hotkey not in exclude][:slots]


async def select_round2_entrants(
    tournament: TournamentData,
    completed_round: TournamentRoundData,
    exclude_hotkeys: set[str],
    psql_db: PSQLDB,
) -> list[str]:
    """R2 challenger field after removing gate-eliminated miners and backfilling.

    Walk the R1 cumulative ranking until SMALL_TOURNAMENT_ADVANCE remain.
    """
    exclude = set(exclude_hotkeys) | {EMISSION_BURN_HOTKEY}
    round_tasks = await get_tournament_tasks(completed_round.round_id, psql_db)
    if tournament.tournament_type == TournamentType.QUANT:
        ordered = await _quant_ordered_hotkeys(round_tasks, psql_db)
        selected = take_standings_slots(ordered, exclude, t_cst.QUANT_ROUND_ONE_ADVANCE)
        logger.info(
            f"R2 backfill for {tournament.tournament_id}: excluded={sorted(exclude_hotkeys)}, selected={selected}"
        )
        return selected
    if tournament.tournament_type == TournamentType.TEXT and await is_seeded_text_tournament(
        tournament.tournament_id, psql_db
    ):
        ordered = await _text_seeded_ordered_hotkeys(round_tasks, psql_db)
        selected = take_standings_slots(ordered, exclude, t_cst.TEXT_ROUND_PLAN[0].advance)
        logger.info(
            f"R2 backfill for {tournament.tournament_id}: excluded={sorted(exclude_hotkeys)}, selected={selected}"
        )
        return selected
    ordered = await _small_tournament_ordered_hotkeys(round_tasks, psql_db)
    selected = take_standings_slots(ordered, exclude, t_cst.SMALL_TOURNAMENT_ADVANCE)
    logger.info(
        f"R2 backfill for {tournament.tournament_id}: excluded={sorted(exclude_hotkeys)}, selected={selected}"
    )
    return selected


def tiebreak_seed(round_id: str) -> int:
    """Per-round shuffle seed: the same on every validator, different for every round."""
    digest = hashlib.sha256(f"{t_cst.R2_TIEBREAK_SEED_SALT}:{round_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def seeded_order(hotkeys: Iterable[str], seed: int) -> dict[str, int]:
    """Shuffle position per hotkey, the last tiebreak so no hotkey string is favoured."""
    shuffled = sorted(hotkeys)
    random.Random(seed).shuffle(shuffled)
    return {hotkey: index for index, hotkey in enumerate(shuffled)}


def _match_order(scored: list[MinerResults], is_higher_better: bool) -> list[MinerResults]:
    """The ranking function's order when it ranked every result, else adjusted_loss in the task's direction.

    Every type reaching this path without a rank (instruct, image) ranks ascending.
    """
    if all(result.rank is not None for result in scored):
        return sorted(scored, key=lambda result: result.rank)
    return sorted(scored, key=lambda result: result.adjusted_loss, reverse=is_higher_better)


def _normalised_gap(metric: float, best: float, is_higher_better: bool) -> float:
    """Non-negative distance from the match's best metric, relative to it when it is positive."""
    if not math.isfinite(metric):
        return float("inf")
    gap = best - metric if is_higher_better else metric - best
    return gap / best if best > 0 else gap


def _match_gaps(metrics: dict[str, float], is_higher_better: bool) -> dict[str, float]:
    finite = [metric for metric in metrics.values() if math.isfinite(metric)]
    if not finite:
        return {hotkey: float("inf") for hotkey in metrics}
    best = max(finite) if is_higher_better else min(finite)
    return {hotkey: _normalised_gap(metric, best, is_higher_better) for hotkey, metric in metrics.items()}


def _standing_log_row(standing: GroupMatchStanding, shuffle_index: int) -> tuple[str, float, float, float, int, bool]:
    return (
        standing.hotkey,
        round(standing.average_rank, 3),
        round(standing.summed_loss, 4),
        round(standing.summed_gap, 4),
        shuffle_index,
        standing.has_error,
    )


async def _small_tournament_standings(
    round_tasks: list[TournamentTask],
    psql_db: PSQLDB,
    cohort: set[str] | None = None,
    recent_round_id: str | None = None,
    seed_round_id: str | None = None,
) -> list[GroupMatchStanding]:
    """Cumulative standings for a single-group round, over one or more matches, best first.

    With a ``cohort``, results of hotkeys outside it are dropped before ranking, so ranks are taken within the
    surviving cohort and a cohort member absent from a match ranks last on it. ``recent_round_id`` marks the
    matches of the most recent round played, whose average rank is the tiebreak after average rank (default: the round
    of the last task, so a single-round call ties on that round); ``seed_round_id`` seeds the final shuffle.
    Ordered by average rank, then most-recent-round rank, then summed normalised gap, then the seeded shuffle.
    """
    match_rankings: list[MatchRanking] = []
    match_losses: list[dict[str, float]] = []
    match_gaps: list[dict[str, float]] = []
    match_is_recent_round: list[bool] = []
    if recent_round_id is None and round_tasks:
        recent_round_id = round_tasks[-1].round_id
    competitors: set[str] = set(cohort or ())
    for task in round_tasks:
        miner_results = await get_task_results_for_ranking(task.task_id, psql_db)
        if cohort is not None:
            miner_results = [result for result in miner_results if result.hotkey in cohort]
        if not miner_results:
            logger.warning(f"No valid results for multi-match group task {task.task_id}")
            continue

        ranked_results = calculate_miner_ranking_and_scores(miner_results)
        scored = [
            result
            for result in ranked_results
            if result.adjusted_loss is not None and not np.isnan(result.adjusted_loss)
        ]
        if not scored:
            logger.warning(f"Multi-match group task {task.task_id} has no valid scores")
            continue

        is_higher_better = higher_is_better(scored[0].task_type)
        ordered_results = _match_order(scored, is_higher_better)
        competitors.update(result.hotkey for result in ordered_results)
        match_rankings.append(
            MatchRanking(task_id=task.task_id, ranked_hotkeys=[result.hotkey for result in ordered_results])
        )
        match_losses.append({result.hotkey: result.adjusted_loss for result in ordered_results})
        metrics = {
            result.hotkey: result.adjusted_loss if result.rank_metric is None else result.rank_metric
            for result in ordered_results
        }
        match_gaps.append(_match_gaps(metrics, is_higher_better))
        match_is_recent_round.append(task.round_id == recent_round_id)

    total_matches = len(match_rankings)
    if total_matches == 0 or not competitors:
        return []

    standings: dict[str, GroupMatchStanding] = {}
    for hotkey in competitors:
        if hotkey == EMISSION_BURN_HOTKEY:
            continue
        total_rank = 0.0
        recent_round_rank = 0.0
        recent_round_matches = 0
        matches_attended = 0
        summed_loss = 0.0
        summed_gap = 0.0
        for ranking, losses, gaps, is_recent_round in zip(
            match_rankings, match_losses, match_gaps, match_is_recent_round, strict=True
        ):
            if hotkey in ranking.ranked_hotkeys:
                rank = ranking.ranked_hotkeys.index(hotkey) + 1
                matches_attended += 1
                summed_loss += losses[hotkey]
                summed_gap += gaps[hotkey]
            else:
                rank = len(ranking.ranked_hotkeys) + 1
                summed_loss += float("inf")
                summed_gap += float("inf")
            total_rank += rank
            if is_recent_round:
                recent_round_rank += rank
                recent_round_matches += 1
        standings[hotkey] = GroupMatchStanding(
            hotkey=hotkey,
            total_rank=total_rank,
            matches_attended=matches_attended,
            total_matches=total_matches,
            summed_loss=summed_loss,
            summed_gap=summed_gap,
            recent_round_rank=recent_round_rank / recent_round_matches if recent_round_matches else 0.0,
        )

    shuffle_index = seeded_order(standings.keys(), tiebreak_seed(seed_round_id or round_tasks[0].round_id))
    ordered = sorted(
        standings.values(),
        key=lambda standing: (
            standing.has_error,
            standing.average_rank,
            standing.recent_round_rank,
            standing.summed_gap,
            shuffle_index[standing.hotkey],
        ),
    )
    logger.info(
        f"Multi-match group standings (hotkey, average_rank, summed_loss, summed_gap, shuffle_index, has_error, "
        f"recent_round_rank) {[(*_standing_log_row(s, shuffle_index[s.hotkey]), round(s.recent_round_rank, 3)) for s in ordered]}"
    )
    return ordered


async def _quant_ordered_hotkeys(
    round_tasks: list[TournamentTask], psql_db: PSQLDB, cohort: set[str] | None = None
) -> list[str]:
    """Full cumulative ranking for a quant single-group round, over one or more matches."""
    standings = await _small_tournament_standings(round_tasks, psql_db, cohort)
    return [standing.hotkey for standing in standings]


async def _cumulative_standings(
    tournament_id: str,
    completed_round: TournamentRoundData,
    psql_db: PSQLDB,
    cohort: set[str] | None = None,
) -> list[GroupMatchStanding]:
    """Standings over every task of rounds 1..completed_round, ranked within ``cohort``.

    The cohort defaults to the completed round's group. Ranks are taken within it, so passing a smaller
    cohort re-ranks the survivors exactly as the final standings will.
    """
    rounds = await get_tournament_rounds(tournament_id, psql_db)
    played = sorted((r for r in rounds if r.round_number <= completed_round.round_number), key=lambda r: r.round_number)
    tasks_by_round = [await get_tournament_tasks(r.round_id, psql_db) for r in played]
    current_tasks = tasks_by_round[-1] if tasks_by_round else []
    group_ids = {task.group_id for task in current_tasks if task.group_id}
    if len(group_ids) != 1:
        return []
    if cohort is None:
        cohort = {member.hotkey for member in await get_tournament_group_members(group_ids.pop(), psql_db)}
    all_tasks = [task for round_tasks in tasks_by_round for task in round_tasks]
    return await _small_tournament_standings(
        all_tasks,
        psql_db,
        cohort,
        recent_round_id=played[-1].round_id,
        seed_round_id=completed_round.round_id,
    )


async def _quant_cumulative_ordered_hotkeys(
    tournament_id: str, completed_round: TournamentRoundData, psql_db: PSQLDB
) -> list[str]:
    """Cohort ranking over every task of rounds 1..completed_round, ranks taken within the completed round's group."""
    standings = await _cumulative_standings(tournament_id, completed_round, psql_db)
    return [standing.hotkey for standing in standings]


def _best_margin(sums: dict[str, int], hotkey: str) -> int:
    """Smallest achievable (own total - best rival total) after a R3 that ranks the cohort 1..n.

    Zero or below means the member can finish top or joint top. R1+R2 rank sums are ``sums``; every
    permutation of R3 ranks is tried, so the best case is exact whatever the tie-break.
    """
    others = [other for other in sums if other != hotkey]
    members = [hotkey, *others]
    margins = []
    for t3_ranks in permutations(range(1, len(members) + 1)):
        totals = {member: sums[member] + rank for member, rank in zip(members, t3_ranks, strict=True)}
        margins.append(totals[hotkey] - min(totals[other] for other in others))
    return min(margins)


def t3_viability_verdicts(ranks: dict[str, T1T2Ranks]) -> list[T3ViabilityVerdict]:
    """Per member: can they still finish top or joint top over R1+R2+R3, given their cohort R1/R2 ranks."""
    sums = {hotkey: pair.t1 + pair.t2 for hotkey, pair in ranks.items()}
    verdicts = []
    for hotkey, pair in ranks.items():
        margin = 0 if len(ranks) == 1 else _best_margin(sums, hotkey)
        verdicts.append(
            T3ViabilityVerdict(
                hotkey=hotkey,
                t1_rank=pair.t1,
                t2_rank=pair.t2,
                viable=margin <= 0,
                best_case_margin=margin,
            )
        )
    return verdicts


def _t1_t2_ranks(standing: GroupMatchStanding) -> T1T2Ranks:
    """R1 and R2 cohort ranks of a two-round standing (one task per round): R2 is the most recent round's rank."""
    t2 = int(round(standing.recent_round_rank))
    return T1T2Ranks(t1=int(round(standing.total_rank)) - t2, t2=t2)


def _log_t3_viability(cohort_size: int, verdicts: list[T3ViabilityVerdict]) -> None:
    for verdict in verdicts:
        outcome = (
            "viable (can still finish top or joint top)"
            if verdict.viable
            else f"DROPPED (best case still {verdict.best_case_margin} behind the best rival)"
        )
        logger.info(
            f"R3 viability, cohort of {cohort_size}: {verdict.hotkey} R1 rank {verdict.t1_rank}, "
            f"R2 rank {verdict.t2_rank}, sum {verdict.t1_rank + verdict.t2_rank}: {outcome}"
        )


async def t3_viable_winners(
    tournament_id: str, completed_round: TournamentRoundData, advancing: list[str], psql_db: PSQLDB
) -> list[str]:
    """Drop advancing miners that can no longer finish top or joint top after R3, to a fixed point.

    Ranks are recomputed within the reduced cohort on every pass, exactly as the final standings will.
    One survivor means R3 is pointless: it is the lone winner and goes straight to the boss round.
    """
    cohort = set(advancing)
    while len(cohort) > 1:
        standings = await _cumulative_standings(tournament_id, completed_round, psql_db, cohort)
        ranks = {standing.hotkey: _t1_t2_ranks(standing) for standing in standings}
        if set(ranks) != cohort:
            logger.warning(f"R3 viability skipped: standings {sorted(ranks)} do not cover cohort {sorted(cohort)}")
            return list(advancing)
        verdicts = t3_viability_verdicts(ranks)
        _log_t3_viability(len(cohort), verdicts)
        viable = {verdict.hotkey for verdict in verdicts if verdict.viable}
        if not viable:
            logger.error("R3 viability left nobody viable, which cannot happen; keeping the cohort")
            break
        if viable == cohort:
            break
        cohort = viable
    kept = [hotkey for hotkey in advancing if hotkey in cohort]
    logger.info(f"R3 viability: {len(kept)} of {len(advancing)} advance to R3: {kept}")
    return kept


async def quant_cumulative_winners(
    tournament_id: str, completed_round: TournamentRoundData, advance: int, psql_db: PSQLDB
) -> list[str]:
    """Top ``advance`` of a quant single-group round by mean rank over rounds 1..N within the surviving cohort."""
    ordered = await _quant_cumulative_ordered_hotkeys(tournament_id, completed_round, psql_db)
    winners = ordered[:advance]
    logger.info(f"advancing top {len(winners)} after round {completed_round.round_number}: {winners}")
    return winners


async def _pre_boss_ordered_hotkeys(
    tournament_id: str, pre_boss: TournamentRoundData, psql_db: PSQLDB
) -> list[str]:
    """Standings that decide third place. When R3 was skipped the boss follows round 2 directly, and the
    ranking is R1+R2 among the miners that advanced from it (the top QUANT_ROUND_TWO_ADVANCE), including any
    dropped as non-viable, re-ranked within that cohort."""
    ordered = await _quant_cumulative_ordered_hotkeys(tournament_id, pre_boss, psql_db)
    if pre_boss.round_number != 2:
        return ordered
    advanced = set(ordered[: t_cst.QUANT_ROUND_TWO_ADVANCE])
    standings = await _cumulative_standings(tournament_id, pre_boss, psql_db, advanced)
    return [standing.hotkey for standing in standings]


async def get_quant_pre_boss_runner_up(
    tournament_id: str,
    final_round: TournamentRoundData,
    challenger_hotkey: str,
    psql_db: PSQLDB,
) -> str | None:
    """Best non-challenger, non-boss miner of the pre-boss quant group round, as 3rd place.

    Uses the same cumulative standings that decided who left the pre-boss round. Returns None when the
    pre-boss round is missing, not a non-final GROUP round, or has no other ranked miner.
    """
    rounds = await get_tournament_rounds(tournament_id, psql_db)
    pre_boss = next((r for r in rounds if r.round_number == final_round.round_number - 1), None)
    if not pre_boss or pre_boss.round_type != RoundType.GROUP or pre_boss.is_final_round:
        return None

    ordered = await _pre_boss_ordered_hotkeys(tournament_id, pre_boss, psql_db)
    return next((hotkey for hotkey in ordered if hotkey not in (challenger_hotkey, EMISSION_BURN_HOTKEY)), None)


async def _get_quant_group_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB
) -> list[str]:
    """Winners of a quant single-group round: round 1 by its own ranking, rounds 2 and 3 by the cumulative standings."""
    if completed_round.round_number == 1:
        ordered = await _quant_ordered_hotkeys(round_tasks, psql_db)
        winners = ordered[: t_cst.QUANT_ROUND_ONE_ADVANCE]
        logger.info(f"advancing top {len(winners)}: {winners}")
        return winners
    if completed_round.round_number == 2:
        advancing = await quant_cumulative_winners(
            completed_round.tournament_id, completed_round, t_cst.QUANT_ROUND_TWO_ADVANCE, psql_db
        )
        return await t3_viable_winners(completed_round.tournament_id, completed_round, advancing, psql_db)
    return await quant_cumulative_winners(
        completed_round.tournament_id, completed_round, t_cst.QUANT_ROUND_THREE_ADVANCE, psql_db
    )



async def _group_members_of_round(round_tasks: list[TournamentTask], psql_db: PSQLDB) -> set[str]:
    """The hotkeys of the single group the round's tasks belong to (the boss is never a member of a seeded text round)."""
    group_ids = {task.group_id for task in round_tasks if task.group_id}
    members: set[str] = set()
    for group_id in group_ids:
        members.update(member.hotkey for member in await get_tournament_group_members(group_id, psql_db))
    return members


async def _text_seeded_ordered_hotkeys(round_tasks: list[TournamentTask], psql_db: PSQLDB) -> list[str]:
    """Standings of a seeded text round from its task alone: the group's members ranked among themselves, so the boss's
    result on the seed task (it trained the same task) never takes a rank."""
    cohort = await _group_members_of_round(round_tasks, psql_db)
    return await _quant_ordered_hotkeys(round_tasks, psql_db, cohort=cohort)


async def _get_text_seeded_group_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB
) -> list[str]:
    """Winners of a seeded text tournament's single-group round: the top TEXT_ROUND_PLAN advance of its members."""
    advance = next(spec.advance for spec in t_cst.TEXT_ROUND_PLAN if spec.round_number == completed_round.round_number)
    ordered = await _text_seeded_ordered_hotkeys(round_tasks, psql_db)
    winners = [hotkey for hotkey in ordered if hotkey != EMISSION_BURN_HOTKEY][:advance]
    logger.info(f"advancing top {len(winners)} of text round {completed_round.round_number}: {winners}")
    return winners


async def get_group_winners(
    completed_round: TournamentRoundData, round_tasks: list[TournamentTask], psql_db: PSQLDB, config: Config = None
) -> list[str]:
    """Get winners from group round based on adjusted loss scores."""

    distinct_groups = {task.group_id for task in round_tasks}
    if len(distinct_groups) == 1 and not completed_round.is_final_round:
        tournament = await get_tournament(completed_round.tournament_id, psql_db)
        if tournament is not None and tournament.tournament_type == TournamentType.QUANT:
            return await _get_quant_group_winners(completed_round, round_tasks, psql_db)
        is_text_miner_round = (
            tournament is not None
            and tournament.tournament_type == TournamentType.TEXT
            and completed_round.round_number in {spec.round_number for spec in t_cst.TEXT_ROUND_PLAN}
        )
        if is_text_miner_round and await is_seeded_text_tournament(completed_round.tournament_id, psql_db):
            return await _get_text_seeded_group_winners(completed_round, round_tasks, psql_db)
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

async def get_round_winners(completed_round: TournamentRoundData, psql_db: PSQLDB, config: Config) -> RoundOutcome:
    """Get winners from the completed round.

    A deferred outcome means the round gained tasks and is not resolvable yet - the caller must add
    them and wait, not advance. Distinguishing that from an empty winner list matters: an empty list
    means the boss retained by default and ends the tournament.
    """
    round_tasks = await get_tournament_tasks(completed_round.round_id, psql_db)

    if completed_round.round_type == RoundType.KNOCKOUT:
        outcome = await get_knockout_winners(completed_round, round_tasks, psql_db, config)
    else:
        outcome = RoundOutcome(winners=await get_group_winners(completed_round, round_tasks, psql_db, config))

    if outcome.is_deferred:
        return outcome

    unique_winners = list(dict.fromkeys(outcome.winners))
    if len(outcome.winners) != len(unique_winners):
        logger.info(
            f"Removed {len(outcome.winners) - len(unique_winners)} duplicate winners from round {completed_round.round_id}"
        )
        logger.info(f"Original winners: {outcome.winners}")
        logger.info(f"Unique winners: {unique_winners}")

    return outcome.model_copy(update={"winners": unique_winners})
