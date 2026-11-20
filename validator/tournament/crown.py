"""The crown rule of a tournament, the same for text, image and compression: seven head-to-heads between the challenger and
the boss.

Pure functions and models only. The seven tasks are the task of each of rounds 1-3 and the four tasks of the boss round.
Each head-to-head is a win or a non-win; a non-win is within the smaller allowance (strictly better than the boss), within the
larger allowance (level with the boss, or worse by at most CROWN_LARGER_ALLOWANCE) or beyond both (worse by more). Only what
a win is depends on the task (text_head_to_head, image_head_to_head, freeform_head_to_head, fixed_head_to_head); the non-win
bands are the same (non_win_outcome). The challenger is crowned when it has at most CROWN_MAX_NON_WINS non-wins, every non-win
is within the larger allowance and, with two non-wins, at least one is within the smaller allowance (crown_verdict). With all
seven decided that is: six wins and a seventh within the larger allowance, or five wins, one non-win within the smaller
allowance and the other within the larger one.
"""

import math
from enum import Enum

from pydantic import BaseModel
from pydantic import ConfigDict

from core.models.task_models import TaskType
from core.models.task_models import higher_is_better
from validator.scoring.models import MinerResultsText
from validator.tournament import constants as t_cst
from validator.tournament.models import PairedLossComparison
from validator.tournament.thresholds import challenger_beats_boss


class HeadToHeadOutcome(str, Enum):
    """How one task went for the challenger against the boss."""

    WIN = "win"
    WITHIN_SMALLER = "within_smaller"  # a non-win strictly better than the boss
    WITHIN_LARGER = "within_larger"  # a non-win level with the boss or worse by at most CROWN_LARGER_ALLOWANCE
    BEYOND = "beyond"  # a non-win worse by more than CROWN_LARGER_ALLOWANCE: it rules the crown out


class HeadToHead(BaseModel):
    """One task of the crown rule, decided.

    ``boss_score`` and ``challenger_score`` are the compared scalars: the adjusted loss (higher is better for GRPO) on a text
    or fixed task, the weighted loss on an image task, the effective bytes on a freeform task; None for a side with no valid
    result. ``deficit`` is how far the challenger trails the boss relative to the boss's value (0.0 when level, ahead or a
    win, infinite when it cannot be measured); None only on a result that has no comparison to measure. ``improvement`` is
    how much better the challenger is, as a fraction of the boss's value (negative when behind); None when it cannot be
    measured.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    label: str
    task_type: TaskType
    outcome: HeadToHeadOutcome
    boss_score: float | None = None
    challenger_score: float | None = None
    deficit: float | None = None
    improvement: float | None = None
    reason: str = ""

    @property
    def is_win(self) -> bool:
        return self.outcome == HeadToHeadOutcome.WIN

    def describe(self) -> str:
        measured = "" if self.improvement is None else f", challenger {self.improvement * 100:+.3f}% vs the boss"
        deficit = "" if self.deficit is None or self.is_win else f", deficit {_percent(self.deficit)}"
        return f"{self.outcome.value}{deficit}{measured}: {self.reason}"


class CrownVerdict(BaseModel):
    """What the crown rule says about the head-to-heads decided so far.

    ``crowned_if_complete`` evaluates the rule on ``decided`` tasks only, counting every task still to come as a win: it is
    the "can still be crowned" answer while tasks are missing and the final answer once all of them are in. ``crowned`` is
    the final answer: the rule holds and exactly CROWN_TASKS tasks were decided. More than CROWN_TASKS decided is not a task
    set the rule is defined on, so it never crowns.
    """

    model_config = ConfigDict(extra="forbid")

    crowned_if_complete: bool
    non_wins: int
    decided: int
    reason: str

    @property
    def can_still_be_crowned(self) -> bool:
        return self.crowned_if_complete

    @property
    def complete(self) -> bool:
        return self.decided == t_cst.CROWN_TASKS

    @property
    def crowned(self) -> bool:
        return self.crowned_if_complete and self.complete


def is_valid_score(score: float | None) -> bool:
    return score is not None and math.isfinite(score)


def is_text_big_task(task_type: TaskType) -> bool:
    """The big task of a seeded text boss round is its only instruct task: the continuous seed is a chat task and the
    others are GRPO and DPO."""
    return task_type == TaskType.INSTRUCTTEXTTASK


def _percent(value: float) -> str:
    return "unbounded" if math.isinf(value) else f"{value * 100:.3f}%"


def _within(value: float, allowance: float) -> bool:
    """At most the allowance; a value exactly on it counts despite float rounding (an infinite value never does)."""
    return value <= allowance * (1 + t_cst.CROWN_BOUNDARY_RELATIVE_TOLERANCE)


def non_win_outcome(deficit: float, strictly_better: bool) -> HeadToHeadOutcome:
    """The band of a non-win, shared by text, fixed and freeform: within the smaller allowance when the challenger is strictly
    better than the boss (no tolerance), else within the larger one when its deficit is at most CROWN_LARGER_ALLOWANCE,
    else beyond both."""
    if strictly_better:
        return HeadToHeadOutcome.WITHIN_SMALLER
    if _within(deficit, t_cst.CROWN_LARGER_ALLOWANCE):
        return HeadToHeadOutcome.WITHIN_LARGER
    return HeadToHeadOutcome.BEYOND


def _relative_deficit(boss_value: float, challenger_value: float, task_is_higher_better: bool) -> float:
    """How far the challenger is behind the boss relative to the boss's magnitude; 0.0 when it is level or ahead."""
    behind = boss_value - challenger_value if task_is_higher_better else challenger_value - boss_value
    if behind <= 0:
        return 0.0
    magnitude = abs(boss_value)
    return behind / magnitude if magnitude > 0 else math.inf


def _improvement(boss_value: float, challenger_value: float, task_is_higher_better: bool) -> float | None:
    magnitude = abs(boss_value)
    if magnitude == 0:
        return None
    gain = challenger_value - boss_value if task_is_higher_better else boss_value - challenger_value
    return gain / magnitude


def is_unpairable(paired: PairedLossComparison | None) -> bool:
    """Whether a task has no paired comparison to decide on: no comparison at all, or one in which no example was scored
    finitely by both sides (n_examples 0)."""
    return paired is None or paired.n_examples == 0


def text_needs_paired_test(task_type: TaskType) -> bool:
    """Text task types whose win against the boss also needs the paired per-example test (instruct, chat/continuous, DPO)."""
    return task_type in t_cst.PAIRED_BOSS_ROUND_TASK_TYPES


def scalar_head_to_head(
    task_id: str,
    label: str,
    task_type: TaskType,
    boss_score: float | None,
    challenger_score: float | None,
    win_margin: float,
    paired: PairedLossComparison | None = None,
) -> HeadToHead:
    """Decide one task on its scalar score, with the paired per-example test where the task type has one.

    A challenger with no valid score: beyond both allowances, whatever the boss has. A boss with no valid score while the
    challenger has one: a win. Otherwise the deficit is how far the challenger's score trails the boss's, relative to the
    boss's magnitude (0 when level or ahead), and the task is a win when the score is at least ``win_margin`` better
    (strictly better against a boss score of exactly 0) and, for a type with per-example losses (text_needs_paired_test),
    ``paired`` is a comparison the challenger won. ``paired`` None for such a type, or a comparison with no comparable
    example (is_unpairable), means the per-example vectors were missing or could not be paired: the boss keeps the task. A
    non-win is banded on the score alone (non_win_outcome): within the smaller allowance when the score is strictly better than
    the boss's, within the larger one when level or worse by at most CROWN_LARGER_ALLOWANCE, beyond both otherwise. A type
    without per-example losses (GRPO, image) is decided on the scalar alone.
    """
    fields = dict(
        task_id=str(task_id), label=label, task_type=task_type, boss_score=boss_score, challenger_score=challenger_score
    )
    if not is_valid_score(challenger_score):
        return HeadToHead(
            **fields, outcome=HeadToHeadOutcome.BEYOND, deficit=math.inf, reason="no valid challenger result"
        )
    if not is_valid_score(boss_score):
        return HeadToHead(
            **fields, outcome=HeadToHeadOutcome.WIN, deficit=0.0, reason="no valid boss result, the challenger takes it"
        )

    task_is_higher_better = higher_is_better(task_type)
    strictly_better = challenger_score > boss_score if task_is_higher_better else challenger_score < boss_score
    margin_met = challenger_beats_boss(boss_score, challenger_score, task_is_higher_better, win_margin) and (
        boss_score != 0 or strictly_better
    )
    deficit = _relative_deficit(boss_score, challenger_score, task_is_higher_better)
    improvement = _improvement(boss_score, challenger_score, task_is_higher_better)
    margin = f"{win_margin * 100:g}%"

    if text_needs_paired_test(task_type):
        if is_unpairable(paired):
            is_win, reason = False, "no pairable per-example losses, the boss keeps the task"
        elif paired.is_draw or not paired.challenger_won:
            is_win, reason = False, f"paired test not passed ({paired.reason})"
        elif not margin_met:
            is_win, reason = False, f"paired test passed but the score is not {margin} better"
        else:
            is_win, reason = True, f"paired test passed and the score is at least {margin} better"
    else:
        is_win = margin_met
        reason = f"score at least {margin} better" if is_win else f"score not {margin} better"

    if is_win:
        return HeadToHead(**fields, outcome=HeadToHeadOutcome.WIN, deficit=0.0, improvement=improvement, reason=reason)
    outcome = non_win_outcome(deficit, strictly_better)
    return HeadToHead(**fields, outcome=outcome, deficit=deficit, improvement=improvement, reason=reason)


def text_head_to_head(
    task_id: str,
    label: str,
    task_type: TaskType,
    boss_score: float | None,
    challenger_score: float | None,
    paired: PairedLossComparison | None = None,
) -> HeadToHead:
    """Decide one text task: scalar_head_to_head with TEXT_CROWN_WIN_MARGIN."""
    return scalar_head_to_head(
        task_id, label, task_type, boss_score, challenger_score, t_cst.TEXT_CROWN_WIN_MARGIN, paired=paired
    )


def image_head_to_head(task_id: str, label: str, boss_score: float | None, challenger_score: float | None) -> HeadToHead:
    """Decide one image task on its weighted loss (lower is better) with IMAGE_CROWN_WIN_MARGIN and no paired test."""
    return scalar_head_to_head(task_id, label, TaskType.IMAGETASK, boss_score, challenger_score, t_cst.IMAGE_CROWN_WIN_MARGIN)


def valid_freeform_bytes(result: MinerResultsText | None) -> float | None:
    """The effective bytes of a valid freeform entry (under the cap, no scored fail, a finite positive byte count), else None."""
    if result is None or result.quant_failure is not None or result.passed_bar is not True:
        return None
    effective = result.effective_bytes
    if effective is None or not math.isfinite(effective) or effective <= 0:
        return None
    return effective


def freeform_head_to_head(
    task_id: str, label: str, boss: MinerResultsText | None, challenger: MinerResultsText | None
) -> HeadToHead:
    """Decide one freeform quant task on effective bytes.

    A challenger with no valid entry (valid_freeform_bytes): beyond both allowances. A boss with no valid entry (over the
    cap, a scored fail, no result, no byte count) while the challenger has one: a win. Otherwise a win when the challenger
    has at least QUANT_FREEFORM_BOSS_BYTE_MARGIN fewer effective bytes than the boss; any other result is banded on the
    effective bytes alone (non_win_outcome): within the smaller allowance with strictly fewer bytes, within the larger one with
    equal bytes or at most CROWN_LARGER_ALLOWANCE more, beyond both with more. The deficit is how many more bytes the
    challenger has, relative to the boss's (0 with fewer or equal).
    """
    boss_bytes, challenger_bytes = valid_freeform_bytes(boss), valid_freeform_bytes(challenger)
    fields = dict(
        task_id=str(task_id),
        label=label,
        task_type=TaskType.QUANTFREEFORMTASK,
        boss_score=boss_bytes,
        challenger_score=challenger_bytes,
    )
    if challenger_bytes is None:
        return HeadToHead(**fields, outcome=HeadToHeadOutcome.BEYOND, deficit=math.inf, reason="no valid challenger entry")
    if boss_bytes is None:
        return HeadToHead(
            **fields, outcome=HeadToHeadOutcome.WIN, deficit=0.0, reason="no valid boss entry, the challenger takes it"
        )
    margin = t_cst.QUANT_FREEFORM_BOSS_BYTE_MARGIN
    improvement = (boss_bytes - challenger_bytes) / boss_bytes
    bar = (1 - margin) * boss_bytes * (1 + t_cst.CROWN_BOUNDARY_RELATIVE_TOLERANCE)
    if challenger_bytes <= bar:
        return HeadToHead(
            **fields,
            outcome=HeadToHeadOutcome.WIN,
            deficit=0.0,
            improvement=improvement,
            reason=f"{_percent(improvement)} fewer effective bytes, at least the {margin * 100:g}% win margin",
        )
    strictly_fewer = challenger_bytes < boss_bytes
    reason = (
        f"{_percent(improvement)} fewer effective bytes, under the {margin * 100:g}% win margin"
        if strictly_fewer
        else "no fewer effective bytes than the boss"
    )
    deficit = _relative_deficit(boss_bytes, challenger_bytes, False)
    return HeadToHead(
        **fields, outcome=non_win_outcome(deficit, strictly_fewer), deficit=deficit, improvement=improvement, reason=reason
    )


def fixed_head_to_head(
    task_id: str,
    label: str,
    boss_loss: float | None,
    challenger_loss: float | None,
    paired: PairedLossComparison | None,
    boss_takes_paired_test: bool,
) -> HeadToHead:
    """Decide one fixed quant task: the paired per-example test and the scalar loss.

    A challenger with no valid loss (none, or a scored fail): beyond both allowances. A boss whose entry is a scored fail or
    absent while the challenger has a valid loss: a win, with or without vectors. With both valid, a win when the challenger
    passes the paired test and its scalar loss is at least QUANT_FIXED_BOSS_MIN_IMPROVEMENT better. Every other result is a
    non-win banded on the scalar loss alone (non_win_outcome): within the smaller allowance with a strictly lower loss, within
    the larger one with an equal loss or at most CROWN_LARGER_ALLOWANCE higher, beyond both with a higher one. ``paired`` None
    (a side has no per-example losses, they cannot be paired, or no same-SKU reference aligns them), a comparison with no
    comparable example (is_unpairable), a draw, a paired test lost without the boss taking the task, and the boss taking it
    on the paired test with the sides swapped (``boss_takes_paired_test``) differ only in the reason.
    """
    fields = dict(
        task_id=str(task_id),
        label=label,
        task_type=TaskType.QUANTFIXEDTASK,
        boss_score=boss_loss,
        challenger_score=challenger_loss,
    )
    if not is_valid_score(challenger_loss):
        return HeadToHead(
            **fields, outcome=HeadToHeadOutcome.BEYOND, deficit=math.inf, reason="no valid challenger entry"
        )
    if not is_valid_score(boss_loss):
        return HeadToHead(
            **fields,
            outcome=HeadToHeadOutcome.WIN,
            deficit=0.0,
            reason="the boss's entry is a scored fail, the challenger takes it",
        )
    improvement = _improvement(boss_loss, challenger_loss, False)
    minimum = t_cst.QUANT_FIXED_BOSS_MIN_IMPROVEMENT
    if is_unpairable(paired):
        reason = "no pairable per-example losses, the boss keeps the task"
    elif paired.challenger_won and not paired.is_draw:
        if challenger_beats_boss(boss_loss, challenger_loss, False, minimum) and (boss_loss != 0 or challenger_loss < 0):
            return HeadToHead(
                **fields,
                outcome=HeadToHeadOutcome.WIN,
                deficit=0.0,
                improvement=improvement,
                reason=f"paired test passed and the scalar loss is at least {minimum * 100:g}% better",
            )
        reason = f"paired test passed but the scalar loss is not {minimum * 100:g}% better"
    elif boss_takes_paired_test:
        reason = "the boss takes it on the paired test"
    elif paired.is_draw:
        reason = f"a draw on the paired test ({paired.reason})"
    else:
        reason = f"paired test not passed ({paired.reason})"
    deficit = _relative_deficit(boss_loss, challenger_loss, False)
    outcome = non_win_outcome(deficit, challenger_loss < boss_loss)
    return HeadToHead(**fields, outcome=outcome, deficit=deficit, improvement=improvement, reason=reason)


def crown_verdict(results: list[HeadToHead]) -> CrownVerdict:
    """The crown rule on the head-to-heads decided so far (every undecided task counts as a win).

    Crowned iff at most CROWN_MAX_NON_WINS non-wins, none beyond the larger allowance and, with two non-wins, at least one
    within the smaller allowance. More than CROWN_TASKS head-to-heads is a task set the rule is not defined on: never crowned.
    """
    non_wins = [result for result in results if not result.is_win]
    count = len(non_wins)
    decided = len(results)
    beyond = [result for result in non_wins if result.outcome == HeadToHeadOutcome.BEYOND]
    smaller = [result for result in non_wins if result.outcome == HeadToHeadOutcome.WITHIN_SMALLER]

    if decided > t_cst.CROWN_TASKS:
        crowned, reason = False, f"{decided} tasks decided, the crown rule is defined on exactly {t_cst.CROWN_TASKS}"
    elif count > t_cst.CROWN_MAX_NON_WINS:
        crowned, reason = False, f"{count} non-wins of {decided} decided, at most {t_cst.CROWN_MAX_NON_WINS} allowed"
    elif beyond:
        crowned = False
        reason = f"{', '.join(result.label for result in beyond)} beyond the larger allowance"
    elif count == 2 and not smaller:
        crowned = False
        reason = f"two non-wins ({', '.join(result.label for result in non_wins)}) and neither within the smaller allowance"
    else:
        crowned = True
        reason = f"{decided - count} wins and {count} non-win(s) of {decided} decided" + (
            f" ({', '.join(f'{result.label} {result.outcome.value}' for result in non_wins)})" if non_wins else ""
        )
    return CrownVerdict(crowned_if_complete=crowned, non_wins=count, decided=decided, reason=reason)
