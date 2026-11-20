import math

import numpy as np

import validator.tournament.constants as t_cst
from core.constants.environments import TrainingStartPoint
from core.logging import get_logger
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskType
from validator.db.database import PSQLDB
from validator.db.sql.quant_references import get_quant_sku_references
from validator.db.sql.submissions_and_scoring import get_per_example_losses
from validator.db.sql.submissions_and_scoring import get_task_node_eval_gpus
from validator.db.sql.submissions_and_scoring import update_task_node_quality_score_only
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.scoring.quant_ranking import QuantReferenceUnavailable
from validator.scoring.quant_ranking import reference_on
from validator.tournament.models import BossRoundTaskVerdict
from validator.tournament.models import PairedExampleWins
from validator.tournament.models import PairedLossComparison
from validator.tournament.quant_reference import is_own_model_start_point
from validator.tournament.task_results import get_task_results_for_ranking


logger = get_logger(__name__)


def compare_paired_losses(
    boss_losses: list[float],
    challenger_losses: list[float],
    deadzone_nats: float = t_cst.BOSS_ROUND_TIE_DEADZONE_NATS,
    min_win_rate: float = t_cst.BOSS_ROUND_MIN_WIN_RATE,
    min_mean_gap_nats: float = t_cst.BOSS_ROUND_MIN_MEAN_GAP_NATS,
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
            is_draw=False,
            reason="No comparable examples between the two vectors",
            required_win_rate=min_win_rate,
            required_mean_gap_nats=min_mean_gap_nats,
            deadzone_nats=deadzone_nats,
        )

    # Positive gap = challenger assigned more probability to that example than the boss did.
    gaps = boss - challenger
    challenger_wins = gaps > deadzone_nats
    boss_wins = gaps < -deadzone_nats
    decided = challenger_wins | boss_wins

    n_decided = int(decided.sum())
    n_challenger_wins = int(challenger_wins.sum())
    win_rate = n_challenger_wins / n_decided if n_decided else 0.0
    # Mean gap is over every comparable example, ties included: it asks whether the challenger is
    # better overall, and near-identical examples legitimately dilute that. The win rate above uses
    # only decided examples, where the two models actually differ.
    mean_gap = float(gaps.mean())

    # Never a weaker requirement than the relative margin it replaces. A flat floor is looser than
    # 1% of the loss once the loss exceeds 2.0, which would have let a uniformly-but-slightly
    # better challenger take a high-loss task it would have lost under the old rule.
    required_mean_gap = max(min_mean_gap_nats, t_cst.BOSS_ROUND_WIN_MARGIN * abs(float(boss.mean())))

    win_rate_lb, mean_gap_lb = _bootstrap_lower_bounds(
        gaps=gaps,
        challenger_wins=challenger_wins,
        decided=decided,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )

    # A draw needs as much evidence as a difference does. "Everything landed in the dead zone" only
    # says the models are equivalent if there were enough examples that a real gap would have shown
    # up - on a handful of examples a genuinely better model lands inside the dead zone by luck.
    # Below the floor this is the same "cannot tell" case as too few decided, not a finding.
    # No minimum on the decided count. The mean gap is taken over every example, so a handful of
    # decided ones cannot move it: 20 of 800 improved by 0.5 nats averages 0.014 and fails, while
    # 20 improved by 1.5 averages 0.039 and passes. That makes the gate a requirement on total
    # improvement rather than on a count, and it already rejects the saturated case a decided-count
    # floor was meant to catch - tiny decided set AND tiny gaps means a tiny mean gap. A floor on
    # top only threw out tasks where the challenger did move the average decisively.
    is_draw = n_decided == 0
    if is_draw:
        reason = (
            f"Draw - all {n_examples} comparable examples fell inside the {deadzone_nats} nat dead "
            f"zone, so the two models are indistinguishable on this task"
        )
        challenger_won = False
    elif win_rate_lb < min_win_rate:
        reason = (
            f"Challenger win rate {win_rate:.1%} (bound {win_rate_lb:.1%}) below required "
            f"{min_win_rate:.1%} of {n_decided} decided examples"
        )
        challenger_won = False
    elif mean_gap_lb < required_mean_gap:
        reason = (
            f"Challenger mean gap {mean_gap:.4f} nats (bound {mean_gap_lb:.4f}) below required "
            f"{required_mean_gap:.4f} nats"
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
        is_draw=is_draw,
        reason=reason,
        required_win_rate=min_win_rate,
        # The bar actually applied, not the floor it was derived from - on a high-loss task the
        # scaled requirement is the one the challenger had to clear.
        required_mean_gap_nats=required_mean_gap,
        deadzone_nats=deadzone_nats,
    )


def align_onto_sku(losses: list[float], own_reference: list[float], anchor_reference: list[float]) -> list[float]:
    """A per-example loss vector scored on one SKU, expressed on another SKU's scale: x_i * anchor_i / own_i.

    That is the per-example normalisation x_i / ref_i(own SKU), multiplied back by the anchor side's own
    reference ref_i(anchor SKU). The anchor's vector is unchanged by the same step, so the gate compares
    both sides as multiples of their own SKU's reference while its dead zone and mean-gap floor stay in the
    anchor's nats. An example whose own reference is not finite and positive becomes NaN, which the paired
    comparison drops from both sides.
    """
    x = np.asarray(losses, dtype=np.float64)
    own = np.asarray(own_reference, dtype=np.float64)
    anchor = np.asarray(anchor_reference, dtype=np.float64)
    usable = np.isfinite(own) & (own > 0) & np.isfinite(anchor)
    aligned = np.divide(x * anchor, own, out=np.full_like(x, np.nan), where=usable)
    return aligned.tolist()


async def sku_aligned_losses(
    task_type: TaskType,
    task_id: str,
    anchor_hotkey: str,
    other_hotkey: str,
    other_losses: list[float],
    psql_db: PSQLDB,
    start_point: TrainingStartPoint | None = None,
) -> list[float]:
    """other_hotkey's per-example losses on anchor_hotkey's SKU scale, for a paired quant comparison.

    Non-quant vectors, and two quant rows scored on the same SKU of a task-common task (whose references
    cancel exactly), come back unchanged. On an own-model task (start_point, see is_own_model_start_point) each
    row has its own per-hotkey reference, so the references differ by player even on one SKU: the alignment always
    runs, and a missing own reference raises (no fallback to the task-common one). Raises QuantReferenceUnavailable
    when either row has no recorded GPU, either reference is missing, or the two references were not scored on the
    same examples.
    """
    if task_type not in QUANT_TASK_TYPES:
        return other_losses
    gpus = await get_task_node_eval_gpus(task_id, [anchor_hotkey, other_hotkey], psql_db)
    anchor_gpu, other_gpu = gpus.get(anchor_hotkey), gpus.get(other_hotkey)
    if anchor_gpu is None or other_gpu is None:
        raise QuantReferenceUnavailable(f"Task {task_id}: no recorded eval GPU for {anchor_hotkey} or {other_hotkey}")
    own_model = is_own_model_start_point(start_point)
    if not own_model and anchor_gpu.reference_key == other_gpu.reference_key:
        return other_losses

    references = await get_quant_sku_references(task_id, psql_db)
    if own_model:
        anchor_reference = references.get((anchor_hotkey, anchor_gpu.reference_key))
        other_reference = references.get((other_hotkey, other_gpu.reference_key))
    else:
        # Task-common tasks: the references are the '' (task-common) ones.
        anchor_reference, other_reference = reference_on("", anchor_gpu, references), reference_on("", other_gpu, references)
    if anchor_reference is None or other_reference is None:
        raise QuantReferenceUnavailable(
            f"Task {task_id}: no cached reference for {anchor_hotkey} on {anchor_gpu.reference_key} "
            f"or for {other_hotkey} on {other_gpu.reference_key}"
        )
    if (
        anchor_reference.eval_set_fingerprint != other_reference.eval_set_fingerprint
        or len(anchor_reference.per_example_losses) != len(other_losses)
        or len(other_reference.per_example_losses) != len(other_losses)
    ):
        raise QuantReferenceUnavailable(f"Task {task_id}: the per-SKU references do not pair with the loss vectors")
    logger.info(
        f"Task {task_id}: aligning {other_hotkey} ({other_gpu.reference_key}) onto {anchor_hotkey} "
        f"({anchor_gpu.reference_key}) per example"
    )
    return align_onto_sku(other_losses, other_reference.per_example_losses, anchor_reference.per_example_losses)


def count_example_wins(
    a_losses: list[float],
    b_losses: list[float],
    deadzone_nats: float = t_cst.BOSS_ROUND_TIE_DEADZONE_NATS,
) -> tuple[int, int, int]:
    """Split two aligned loss vectors into (a wins, b wins, comparable examples).

    Comparable means both sides produced a finite loss for that example; the remainder of
    n_examples after the two win counts is the dead zone, won by neither.
    """
    a = np.asarray(a_losses, dtype=np.float64)
    b = np.asarray(b_losses, dtype=np.float64)

    usable = np.isfinite(a) & np.isfinite(b)
    gaps = b[usable] - a[usable]  # positive = a assigned more probability to that example
    return int((gaps > deadzone_nats).sum()), int((gaps < -deadzone_nats).sum()), int(gaps.size)


async def summarise_paired_examples(
    task_id: str,
    hotkey_a: str,
    hotkey_b: str,
    psql_db: PSQLDB,
    deadzone_nats: float = t_cst.BOSS_ROUND_TIE_DEADZONE_NATS,
    task_type: TaskType | None = None,
    start_point: TrainingStartPoint | None = None,
) -> PairedExampleWins | None:
    """Report how a two-way task split example by example, or None if it cannot be paired.

    Read-only view for analytics. Returns None on exactly the cases the winner logic falls back to
    the mean loss for - no stored vector on either side, or vectors that do not line up - so a
    consumer that gets None should show the mean losses instead. A quant task_type aligns hotkey_b
    onto hotkey_a's SKU first, as the gates do, and is None without the references to do it.
    """
    a_losses, a_fingerprint = await get_per_example_losses(task_id, hotkey_a, psql_db)
    b_losses, b_fingerprint = await get_per_example_losses(task_id, hotkey_b, psql_db)

    if a_losses is None or b_losses is None or len(a_losses) != len(b_losses):
        return None
    if a_fingerprint is not None and b_fingerprint is not None and a_fingerprint != b_fingerprint:
        return None
    if task_type is not None:
        try:
            b_losses = await sku_aligned_losses(task_type, task_id, hotkey_a, hotkey_b, b_losses, psql_db, start_point)
        except QuantReferenceUnavailable as e:
            logger.warning(f"Paired summary unavailable: {e}")
            return None

    a_wins, b_wins, n_examples = count_example_wins(a_losses, b_losses, deadzone_nats)
    return PairedExampleWins(
        hotkey_a=hotkey_a,
        hotkey_b=hotkey_b,
        hotkey_a_wins=a_wins,
        hotkey_b_wins=b_wins,
        ties=n_examples - a_wins - b_wins,
        n_examples=n_examples,
        deadzone_nats=deadzone_nats,
    )


def challenger_takes_paired_task(
    comparison: PairedLossComparison,
    boss_scalar_loss: float,
    challenger_scalar_loss: float,
) -> tuple[bool, str]:
    """Final verdict for a paired boss-round task. Returns (challenger_won, reason).

    Two conditions. The paired comparison runs on RAW per-example losses; the ranking scalar can
    additionally carry a per-model KL penalty, which is a constant and so cannot be folded into the
    vector without fabricating a 100% win rate. So the challenger must win on the examples AND not
    be worse once its penalty is counted. For non-KL tasks the scalar is the vector mean and the
    second condition is already implied by the first.

    Shared by crowning and by the emission projection: if these two disagreed, analytics would
    report a challenger win on a task the tournament recorded as a loss.
    """
    if comparison.is_draw:
        return False, comparison.reason
    if not comparison.challenger_won:
        return False, comparison.reason
    if challenger_scalar_loss > boss_scalar_loss:
        return False, (
            f"won the paired comparison but is worse on the ranking scalar "
            f"({challenger_scalar_loss:.6f} vs {boss_scalar_loss:.6f})"
        )
    return True, comparison.reason


def paired_head_to_head_winner(
    hotkey_a: str,
    a_losses: list[float],
    a_mean_loss: float,
    hotkey_b: str,
    b_losses: list[float],
    b_mean_loss: float,
    deadzone_nats: float = t_cst.BOSS_ROUND_TIE_DEADZONE_NATS,
) -> tuple[str, str]:
    """Pick the winner of a knockout task on per-example win rate. Returns (winner, description).

    Same pairing as the boss round and for the same reasons, but a knockout must always advance
    someone, so this is a bare majority of decided examples with no significance requirement: a
    confidence bound has nothing to add when one of the two has to go through either way. Equal
    wins, or nothing decided, falls back to the lower mean loss.
    """
    a_wins, b_wins, _ = count_example_wins(a_losses, b_losses, deadzone_nats)
    decided = a_wins + b_wins

    if a_wins == b_wins:
        winner = hotkey_a if a_mean_loss <= b_mean_loss else hotkey_b
        return winner, (
            f"tied on {a_wins}/{decided} decided examples, broken on mean loss "
            f"({a_mean_loss:.6f} vs {b_mean_loss:.6f})"
        )

    winner = hotkey_a if a_wins > b_wins else hotkey_b
    win_rate = max(a_wins, b_wins) / decided
    return winner, f"won {max(a_wins, b_wins)}/{decided} decided examples ({win_rate:.1%})"


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
    # The bar is shortened by a relative tolerance so a challenger exactly `margin` better counts despite float rounding.
    bar = abs(boss_loss) * margin * (1 - t_cst.CROWN_BOUNDARY_RELATIVE_TOLERANCE)
    if higher_is_better:
        return challenger_loss >= boss_loss + bar
    return challenger_loss <= boss_loss - bar


def quant_challenger_beats_boss(boss_loss: float, challenger_loss: float, margin: float) -> bool:
    """challenger_beats_boss for a lower-is-better quant task, whose scored fails carry a non-finite loss.

    A non-finite score (a scored fail is inf, a missing one nan) is never compared arithmetically, since inf - inf is
    nan and would make every comparison false: a non-finite challenger never beats the boss, and a finite challenger
    always beats a non-finite boss.
    """
    if not math.isfinite(challenger_loss):
        return False
    if not math.isfinite(boss_loss):
        return True
    return challenger_beats_boss(boss_loss, challenger_loss, False, margin)


def quant_challenger_takes_paired_task(
    comparison: PairedLossComparison,
    boss_scalar_loss: float,
    challenger_scalar_loss: float,
) -> tuple[bool, str]:
    """challenger_takes_paired_task for a quant task: the ranking scalars may be non-finite (a scored fail).

    A non-finite challenger scalar never wins; a non-finite boss scalar cannot be worse than the challenger's.
    """
    if comparison.is_draw or not comparison.challenger_won:
        return False, comparison.reason
    if not math.isfinite(challenger_scalar_loss):
        return False, "won the paired comparison but its ranking scalar is not finite"
    if not math.isfinite(boss_scalar_loss):
        return True, comparison.reason
    return challenger_takes_paired_task(comparison, boss_scalar_loss, challenger_scalar_loss)


def quant_challenger_takes_fixed_boss_task(
    comparison: PairedLossComparison,
    boss_scalar_loss: float,
    challenger_scalar_loss: float,
) -> tuple[bool, str]:
    """The fixed quant boss-round task: the paired per-example test AND a scalar loss QUANT_FIXED_BOSS_MIN_IMPROVEMENT better.

    A paired win that misses the scalar improvement goes to the boss, and the reason says by how much it missed.
    Shared by crowning and by the emission projection.
    """
    won, reason = quant_challenger_takes_paired_task(comparison, boss_scalar_loss, challenger_scalar_loss)
    if not won:
        return False, reason
    minimum = t_cst.QUANT_FIXED_BOSS_MIN_IMPROVEMENT
    if quant_challenger_beats_boss(boss_scalar_loss, challenger_scalar_loss, minimum):
        return True, reason
    improvement = (boss_scalar_loss - challenger_scalar_loss) / abs(boss_scalar_loss)
    return False, (
        f"won the paired comparison but the scalar loss improved only {improvement * 100:.2f}%, "
        f"below the {minimum * 100:g}% minimum ({challenger_scalar_loss:.6f} vs {boss_scalar_loss:.6f})"
    )


def _under_freeform_cap(result: MinerResultsText | None) -> bool:
    return result is not None and result.quant_failure is None and result.passed_bar is True


def resolve_freeform_boss_task(
    boss: MinerResultsText | None, opponent: MinerResultsText | None, margin: float
) -> BossRoundTaskVerdict:
    """Decide a freeform boss task on the cap, then on effective bytes. Never a draw.

    The challenger takes it only by being under the cap and either the boss not being, or having at
    most (1 - margin) x the boss's effective bytes (total bytes x the loss penalty). The margin is on
    effective size. Every other case - both over the cap, the challenger over it, a scored fail, a missing
    byte count - stays with the boss.
    """
    if not _under_freeform_cap(opponent):
        return BossRoundTaskVerdict(winner_hotkey=EMISSION_BURN_HOTKEY, decided_by="cap: challenger did not pass")
    if not _under_freeform_cap(boss):
        return BossRoundTaskVerdict(winner_hotkey=opponent.hotkey, decided_by="cap: boss failed, challenger passed")
    if opponent.effective_bytes is None or boss.effective_bytes is None:
        return BossRoundTaskVerdict(winner_hotkey=EMISSION_BURN_HOTKEY, decided_by="bytes: missing byte count")
    bar = (1 - margin) * boss.effective_bytes * (1 + t_cst.CROWN_BOUNDARY_RELATIVE_TOLERANCE)
    challenger_won = opponent.effective_bytes <= bar
    return BossRoundTaskVerdict(
        winner_hotkey=opponent.hotkey if challenger_won else EMISSION_BURN_HOTKEY,
        decided_by=(
            f"effective bytes {opponent.effective_bytes:.0f} vs {boss.effective_bytes:.0f} ({margin * 100:g}% margin)"
        ),
    )


def challenger_within_freeform_allowed_loss(
    boss: MinerResultsText | None,
    challenger: MinerResultsText | None,
    margin: float = t_cst.BOSS_ROUND_FREEFORM_ALLOWED_LOSS_WITHIN,
) -> bool:
    """Whether a challenger that lost a freeform boss task lost within the allowed loss.

    Within means under the cap with effective bytes at most (1 + margin) x the boss's, inclusive. A scored fail,
    a missing or non-finite byte count on either side, or a missing result is not within.
    """
    if not _under_freeform_cap(challenger) or boss is None:
        return False
    boss_bytes, challenger_bytes = boss.effective_bytes, challenger.effective_bytes
    if boss_bytes is None or challenger_bytes is None or not math.isfinite(boss_bytes) or not math.isfinite(challenger_bytes):
        return False
    return challenger_bytes <= (1 + margin) * boss_bytes


def compare_paired_losses_swapped(boss_losses: list[float], challenger_losses: list[float]) -> PairedLossComparison:
    """The paired gate with the sides swapped: does the BOSS clear the bar against the challenger?

    Same function and same default constants as the forward call (dead zone, win rate, mean-gap floor,
    bootstrap, seed), on the same aligned vectors; comparison.challenger_won then means the boss won.
    """
    return compare_paired_losses(boss_losses=challenger_losses, challenger_losses=boss_losses)


def challenger_within_paired_allowed_loss(
    boss_losses: list[float],
    challenger_losses: list[float],
    boss_scalar_loss: float,
    challenger_scalar_loss: float,
) -> tuple[bool, PairedLossComparison]:
    """Whether a challenger that lost a fixed task (or any other paired-loss task) lost within the allowed loss.

    Within unless the boss would have taken the task from the challenger under the same paired bar with the
    sides swapped, scalar gate included. A draw in the swapped comparison (nothing decided) is within. The vectors
    are the ones the forward verdict used: the challenger's already aligned onto the boss's SKU scale. Returns
    the verdict and the swapped comparison for logging; callers fail closed (not within) when they cannot pair.
    """
    swapped = compare_paired_losses_swapped(boss_losses, challenger_losses)
    boss_takes_it, _ = quant_challenger_takes_paired_task(swapped, challenger_scalar_loss, boss_scalar_loss)
    return not boss_takes_it, swapped


async def update_threshold_adjusted_quality_scores_for_task(
    task_id: str,
    winner_hotkey: str,
    threshold_percentage: float,
    psql_db: PSQLDB,
    compared_hotkeys: list[str] | None = None,
    basis: str | None = None,
) -> None:
    """Persist threshold-adjusted task scores while preserving raw losses.

    ``basis`` describes what decided the task, for the stored score_reason. Callers that decide on
    something other than the win margin pass their own, so the persisted reason matches reality.
    """
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
    decided_by = basis if basis is not None else f"{threshold_pct:.1f}% boss-round win margin"
    for result in miner_results:
        if allowed_hotkeys is not None and result.hotkey not in allowed_hotkeys:
            continue

        is_winner = result.hotkey == winner_hotkey
        quality_score = 3.0 if is_winner else 0.0
        score_reason = (
            f"Winner at {decided_by}" if is_winner else f"Lost to winner {winner_hotkey} at {decided_by}"
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
        f"decided by {decided_by}"
    )
