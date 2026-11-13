"""The crown rule of a seeded text tournament: seven head-to-heads between the challenger and the boss.

Pure functions and models only. The seven tasks are the tasks of rounds 1-3 and the four tasks of the boss round. Each is
a win (the challenger beats the boss by TEXT_CROWN_WIN_MARGIN) or a non-win with a deficit: how far the challenger trails
the boss, relative to the size of the boss's score. The challenger is crowned when it has few enough non-wins and every
deficit is small enough (see text_crown_verdict).
"""

import math

from pydantic import BaseModel
from pydantic import ConfigDict

from core.models.task_models import TaskType
from core.models.task_models import higher_is_better
from validator.tournament import constants as t_cst
from validator.tournament.thresholds import challenger_beats_boss


def is_valid_score(score: float | None) -> bool:
    return score is not None and math.isfinite(score)


def is_text_big_task(task_type: TaskType) -> bool:
    """The big task of a seeded text boss round is its only instruct task: the continuous seed is a chat task and the
    others are GRPO and DPO."""
    return task_type == TaskType.INSTRUCTTEXTTASK


class TextHeadToHead(BaseModel):
    """One task of the crown rule, decided.

    ``deficit`` is 0.0 for a win and may be infinite for a non-win (a side has no valid result, or the boss's score is 0
    and the challenger is worse). ``improvement`` is how much better the challenger is than the boss as a fraction of the
    boss's score magnitude (negative when it is behind); None when it cannot be measured.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    label: str
    task_type: TaskType
    boss_score: float | None = None
    challenger_score: float | None = None
    is_win: bool
    deficit: float
    improvement: float | None = None

    def describe(self) -> str:
        if not is_valid_score(self.challenger_score):
            return "no valid challenger result"
        if not is_valid_score(self.boss_score):
            return "no valid boss result, the challenger takes it"
        measured = "" if self.improvement is None else f", challenger {self.improvement * 100:+.3f}% vs the boss"
        if self.is_win:
            return f"win{measured}"
        deficit = "unbounded" if math.isinf(self.deficit) else f"{self.deficit * 100:.3f}%"
        return f"non-win, deficit {deficit}{measured}"


class TextCrownVerdict(BaseModel):
    """What the crown rule says about the head-to-heads decided so far.

    ``crowned_if_complete`` evaluates the rule on ``decided`` tasks only, counting every task still to come as a win: it
    is the "can still be crowned" test while tasks are missing and the final answer once all of them are in. ``crowned``
    is the final answer: the rule holds and all TEXT_CROWN_TASKS tasks were decided.
    """

    model_config = ConfigDict(extra="forbid")

    crowned_if_complete: bool
    non_wins: int
    decided: int
    reason: str

    @property
    def complete(self) -> bool:
        return self.decided >= t_cst.TEXT_CROWN_TASKS

    @property
    def crowned(self) -> bool:
        return self.crowned_if_complete and self.complete


def _deficit(boss_score: float, challenger_score: float, task_is_higher_better: bool) -> float:
    """How far the challenger is behind the boss relative to the boss's score magnitude; 0.0 when it is level or ahead."""
    behind = boss_score - challenger_score if task_is_higher_better else challenger_score - boss_score
    if behind <= 0:
        return 0.0
    magnitude = abs(boss_score)
    return behind / magnitude if magnitude > 0 else math.inf


def _improvement(boss_score: float, challenger_score: float, task_is_higher_better: bool) -> float | None:
    magnitude = abs(boss_score)
    if magnitude == 0:
        return None
    gain = challenger_score - boss_score if task_is_higher_better else boss_score - challenger_score
    return gain / magnitude


def text_head_to_head(
    task_id: str,
    label: str,
    task_type: TaskType,
    boss_score: float | None,
    challenger_score: float | None,
) -> TextHeadToHead:
    """Decide one task from the two scores (the adjusted loss of each side; higher is better for GRPO, lower otherwise).

    A challenger with no valid score has a non-win with an unbounded deficit, whatever the boss has. A boss with no valid
    score while the challenger has one loses the task. Against a boss score of exactly 0 the margin is 0 as well, so a win
    there needs the challenger to be strictly better, not level.
    """
    fields = dict(
        task_id=str(task_id), label=label, task_type=task_type, boss_score=boss_score, challenger_score=challenger_score
    )
    if not is_valid_score(challenger_score):
        return TextHeadToHead(**fields, is_win=False, deficit=math.inf)
    if not is_valid_score(boss_score):
        return TextHeadToHead(**fields, is_win=True, deficit=0.0)

    task_is_higher_better = higher_is_better(task_type)
    strictly_better = challenger_score > boss_score if task_is_higher_better else challenger_score < boss_score
    if challenger_beats_boss(boss_score, challenger_score, task_is_higher_better, t_cst.TEXT_CROWN_WIN_MARGIN) and (
        boss_score != 0 or strictly_better
    ):
        deficit = 0.0
        is_win = True
    else:
        deficit = _deficit(boss_score, challenger_score, task_is_higher_better)
        is_win = False
    return TextHeadToHead(
        **fields,
        is_win=is_win,
        deficit=deficit,
        improvement=_improvement(boss_score, challenger_score, task_is_higher_better),
    )


def _percent(deficit: float) -> str:
    return "unbounded" if math.isinf(deficit) else f"{deficit * 100:.3f}%"


def text_crown_verdict(results: list[TextHeadToHead]) -> TextCrownVerdict:
    """The crown rule on the head-to-heads decided so far (every undecided task counts as a win).

    The challenger is crowned iff it has at most TEXT_CROWN_MAX_NON_WINS non-wins, no deficit above
    TEXT_CROWN_MAX_DEFICIT, and, with exactly two non-wins, at least one deficit at most
    TEXT_CROWN_SECOND_NON_WIN_MAX_DEFICIT. With all seven tasks that is: six wins and a seventh lost by at most 0.25%, or
    five wins and two losses of at most 0.1% and 0.25%.
    """
    non_wins = [result for result in results if not result.is_win]
    count = len(non_wins)
    decided = len(results)
    deficits = sorted(result.deficit for result in non_wins)
    # A deficit exactly at a limit is within it; the slack only absorbs float rounding (an infinite deficit stays refused).
    slack = 1 + t_cst.CROWN_BOUNDARY_RELATIVE_TOLERANCE
    max_deficit = t_cst.TEXT_CROWN_MAX_DEFICIT * slack
    second_max_deficit = t_cst.TEXT_CROWN_SECOND_NON_WIN_MAX_DEFICIT * slack

    if count > t_cst.TEXT_CROWN_MAX_NON_WINS:
        crowned, reason = False, f"{count} non-wins of {decided} decided, at most {t_cst.TEXT_CROWN_MAX_NON_WINS} allowed"
    elif deficits and deficits[-1] > max_deficit:
        crowned = False
        reason = f"a deficit of {_percent(deficits[-1])} is above the {t_cst.TEXT_CROWN_MAX_DEFICIT * 100:.3f}% allowed"
    elif count == 2 and deficits[0] > second_max_deficit:
        crowned = False
        reason = (
            f"two non-wins and neither deficit ({_percent(deficits[0])}, {_percent(deficits[1])}) is within "
            f"{t_cst.TEXT_CROWN_SECOND_NON_WIN_MAX_DEFICIT * 100:.3f}%"
        )
    else:
        crowned = True
        reason = (
            f"{decided - count} wins and {count} non-win(s) of {decided} decided"
            + (f" (deficits {', '.join(_percent(d) for d in deficits)})" if deficits else "")
        )
    return TextCrownVerdict(crowned_if_complete=crowned, non_wins=count, decided=decided, reason=reason)
