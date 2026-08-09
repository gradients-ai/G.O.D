import numpy as np

import validator.tournament.constants as t_cst
from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import update_task_node_quality_score_only
from validator.tournament.models import PairedLossComparison
from validator.tournament.task_results import get_task_results_for_ranking


logger = get_logger(__name__)


def compare_paired_losses(
    boss_losses: list[float],
    challenger_losses: list[float],
    deadzone_nats: float = t_cst.BOSS_ROUND_TIE_DEADZONE_NATS,
    min_win_rate: float = t_cst.BOSS_ROUND_MIN_WIN_RATE,
    min_mean_gap_nats: float = t_cst.BOSS_ROUND_MIN_MEAN_GAP_NATS,
    min_decided: int = t_cst.BOSS_ROUND_MIN_DECIDED_EXAMPLES,
    confidence: float = t_cst.BOSS_ROUND_BOOTSTRAP_CONFIDENCE,
    resamples: int = t_cst.BOSS_ROUND_BOOTSTRAP_RESAMPLES,
    seed: int = t_cst.BOSS_ROUND_BOOTSTRAP_SEED,
) -> PairedLossComparison:
    """Decide a boss-round task by comparing losses example by example on the same held-out set.

    Lower-is-better log-likelihood losses only (instruct, DPO). Both models must have been scored
    on the identical examples in the identical order — index i is the same example for both.

    The challenger takes the task only if it wins a clear majority of decided examples AND is
    better on average, with both statistics clearing their bar at the one-sided bootstrap bound
    rather than on the point estimate. Ties and tasks with too few decided examples are not wins.
    """
    if len(boss_losses) != len(challenger_losses):
        raise ValueError(
            f"Paired comparison needs equal-length loss vectors, got boss={len(boss_losses)} "
            f"challenger={len(challenger_losses)}"
        )

    boss = np.asarray(boss_losses, dtype=np.float64)
    challenger = np.asarray(challenger_losses, dtype=np.float64)

    # An example is only usable if both sides scored it; a non-finite loss on either side means
    # that example carries no comparison and pairing requires dropping it from both.
    usable = np.isfinite(boss) & np.isfinite(challenger)
    boss, challenger = boss[usable], challenger[usable]
    n_examples = int(boss.size)

    if n_examples == 0:
        return PairedLossComparison(
            n_examples=0,
            n_decided=0,
            challenger_example_wins=0,
            boss_example_wins=0,
            win_rate=0.0,
            win_rate_lower_bound=0.0,
            mean_gap_nats=0.0,
            mean_gap_lower_bound=0.0,
            challenger_won=False,
            reason="No comparable examples between the two vectors",
        )

    # Positive gap = challenger assigned more probability to that example than the boss did.
    gaps = boss - challenger
    challenger_wins = gaps > deadzone_nats
    boss_wins = gaps < -deadzone_nats
    decided = challenger_wins | boss_wins

    n_decided = int(decided.sum())
    n_challenger_wins = int(challenger_wins.sum())
    win_rate = n_challenger_wins / n_decided if n_decided else 0.0
    mean_gap = float(gaps.mean())

    win_rate_lb, mean_gap_lb = _bootstrap_lower_bounds(
        gaps=gaps,
        challenger_wins=challenger_wins,
        decided=decided,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )

    if n_decided < min_decided:
        reason = (
            f"Only {n_decided} decided examples (need {min_decided}) - too few to distinguish the "
            f"two models on this task"
        )
        challenger_won = False
    elif win_rate_lb < min_win_rate:
        reason = (
            f"Challenger win rate {win_rate:.1%} (bound {win_rate_lb:.1%}) below required "
            f"{min_win_rate:.1%} of {n_decided} decided examples"
        )
        challenger_won = False
    elif mean_gap_lb < min_mean_gap_nats:
        reason = (
            f"Challenger mean gap {mean_gap:.4f} nats (bound {mean_gap_lb:.4f}) below required "
            f"{min_mean_gap_nats} nats"
        )
        challenger_won = False
    else:
        reason = (
            f"Challenger won {win_rate:.1%} of {n_decided} decided examples (bound "
            f"{win_rate_lb:.1%}) by {mean_gap:.4f} nats (bound {mean_gap_lb:.4f})"
        )
        challenger_won = True

    return PairedLossComparison(
        n_examples=n_examples,
        n_decided=n_decided,
        challenger_example_wins=n_challenger_wins,
        boss_example_wins=int(boss_wins.sum()),
        win_rate=win_rate,
        win_rate_lower_bound=win_rate_lb,
        mean_gap_nats=mean_gap,
        mean_gap_lower_bound=mean_gap_lb,
        challenger_won=challenger_won,
        reason=reason,
    )


def _bootstrap_lower_bounds(
    gaps: np.ndarray,
    challenger_wins: np.ndarray,
    decided: np.ndarray,
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """One-sided lower bounds on the win rate and mean gap, resampling examples with replacement.

    Resamples the full example set rather than only the decided ones: which examples land in the
    dead zone is itself a property of the sample, so holding it fixed would understate the spread.
    Seeded, so two validators scoring the same boss round reach the same verdict.
    """
    n = gaps.size
    rng = np.random.default_rng(seed)
    win_arr = challenger_wins.astype(np.float64)
    decided_arr = decided.astype(np.float64)

    win_rates = np.empty(resamples, dtype=np.float64)
    mean_gaps = np.empty(resamples, dtype=np.float64)

    # Chunked so the index matrix stays bounded regardless of eval-set size.
    chunk = max(1, min(resamples, 2_000_000 // max(n, 1)))
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        idx = rng.integers(0, n, size=(size, n))
        decided_counts = decided_arr[idx].sum(axis=1)
        win_counts = win_arr[idx].sum(axis=1)
        # A resample that decided nothing is not evidence for the challenger.
        win_rates[start : start + size] = np.divide(
            win_counts, decided_counts, out=np.zeros(size), where=decided_counts > 0
        )
        mean_gaps[start : start + size] = gaps[idx].mean(axis=1)

    percentile = (1.0 - confidence) * 100.0
    return float(np.percentile(win_rates, percentile)), float(np.percentile(mean_gaps, percentile))


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
