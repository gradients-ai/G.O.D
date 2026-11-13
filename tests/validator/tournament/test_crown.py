"""The shared crown rule (crown.py) and each task kind's win and allowances, as pure functions.

Text: the paired per-example test plus a 1% better score on instruct, chat and DPO; GRPO on the score alone; non-wins
within 0.1% (tight) or 0.25% (looser). Freeform: 3% fewer effective bytes; at most 1% more (tight) or 2% more (looser).
Fixed: the paired test plus a 1% better scalar; a draw on the paired test is within both allowances, a task the boss takes
on it is beyond them.
"""

import math

import pytest

from core.models.task_models import TaskType
from validator.scoring.models import MinerResultsText
from validator.tournament import constants as t_cst
from validator.tournament.crown import HeadToHead
from validator.tournament.crown import HeadToHeadOutcome
from validator.tournament.crown import crown_verdict
from validator.tournament.crown import fixed_head_to_head
from validator.tournament.crown import freeform_head_to_head
from validator.tournament.crown import text_head_to_head
from validator.tournament.models import PairedLossComparison


WIN, TIGHT, LOOSER, BEYOND = (
    HeadToHeadOutcome.WIN,
    HeadToHeadOutcome.WITHIN_TIGHT,
    HeadToHeadOutcome.WITHIN_LOOSER,
    HeadToHeadOutcome.BEYOND,
)


def _result(label: str, outcome: HeadToHeadOutcome) -> HeadToHead:
    return HeadToHead(task_id=label, label=label, task_type=TaskType.INSTRUCTTEXTTASK, outcome=outcome)


def _verdict(outcomes: list[HeadToHeadOutcome]):
    return crown_verdict([_result(f"task {i}", outcome) for i, outcome in enumerate(outcomes)])


def _paired(challenger_won: bool = True, is_draw: bool = False) -> PairedLossComparison:
    return PairedLossComparison(
        n_examples=100, n_decided=0 if is_draw else 100, challenger_example_wins=90 if challenger_won else 10,
        boss_example_wins=10 if challenger_won else 90, win_rate=0.9 if challenger_won else 0.1,
        win_rate_lower_bound=0.8 if challenger_won else 0.05, mean_gap_nats=0.1, mean_gap_lower_bound=0.05,
        challenger_won=challenger_won and not is_draw, is_draw=is_draw, reason="scripted", required_win_rate=0.6,
        required_mean_gap_nats=0.01,
    )  # fmt: skip


PASSED, FAILED, DRAWN = _paired(True), _paired(False), _paired(False, is_draw=True)
# What compare_paired_losses returns when no example was scored finitely by both sides.
NOTHING_COMPARABLE = _paired(False).model_copy(
    update={"n_examples": 0, "n_decided": 0, "challenger_example_wins": 0, "boss_example_wins": 0, "win_rate": 0.0}
)


class TestTheSharedVerdict:
    @pytest.mark.parametrize(
        "name, outcomes, crowned",
        [
            ("seven wins", [WIN] * 7, True),
            ("six wins, the seventh within the looser allowance", [WIN] * 6 + [LOOSER], True),
            ("six wins, the seventh within the tight allowance", [WIN] * 6 + [TIGHT], True),
            ("six wins, the seventh beyond", [WIN] * 6 + [BEYOND], False),
            ("five wins, one tight and one looser", [WIN] * 5 + [TIGHT, LOOSER], True),
            ("five wins, two tight", [WIN] * 5 + [TIGHT, TIGHT], True),
            ("five wins, two looser only", [WIN] * 5 + [LOOSER, LOOSER], False),
            ("five wins, one tight and one beyond", [WIN] * 5 + [TIGHT, BEYOND], False),
            ("four wins, three tight", [WIN] * 4 + [TIGHT] * 3, False),
        ],
    )
    def test_the_table(self, name, outcomes, crowned):
        assert _verdict(outcomes).crowned is crowned, name

    @pytest.mark.parametrize(
        "outcomes, possible",
        [
            ([WIN] * 3, True),
            ([WIN, TIGHT, LOOSER], True),
            ([WIN, LOOSER, LOOSER], False),
            ([WIN, WIN, BEYOND], False),
            ([TIGHT, TIGHT, TIGHT], False),
        ],
    )
    def test_undecided_tasks_count_as_wins(self, outcomes, possible):
        verdict = _verdict(outcomes)

        assert verdict.can_still_be_crowned is possible and verdict.crowned_if_complete is possible
        assert not verdict.complete and not verdict.crowned

    def test_more_than_seven_head_to_heads_never_crown(self):
        verdict = _verdict([WIN] * 9)

        assert not verdict.crowned and not verdict.crowned_if_complete and not verdict.complete
        assert "exactly 7" in verdict.reason

    def test_the_constants(self):
        assert (t_cst.CROWN_TASKS, t_cst.CROWN_MAX_NON_WINS) == (7, 2)
        assert (t_cst.TEXT_CROWN_WIN_MARGIN, t_cst.TEXT_CROWN_TIGHT_ALLOWANCE, t_cst.TEXT_CROWN_LOOSER_ALLOWANCE) == (
            0.01,
            0.001,
            0.0025,
        )
        assert t_cst.QUANT_FREEFORM_BOSS_BYTE_MARGIN == 0.03
        assert (t_cst.QUANT_CROWN_FREEFORM_TIGHT_ALLOWANCE, t_cst.QUANT_CROWN_FREEFORM_LOOSER_ALLOWANCE) == (0.01, 0.02)
        assert t_cst.QUANT_FIXED_BOSS_MIN_IMPROVEMENT == 0.01


class TestTextAgainstTheBoss:
    @pytest.mark.parametrize("task_type", [TaskType.INSTRUCTTEXTTASK, TaskType.CHATTASK, TaskType.DPOTASK])
    def test_a_win_needs_the_paired_test_and_one_percent(self, task_type):
        assert text_head_to_head("t", "x", task_type, 1.0, 0.99, PASSED).is_win
        short = text_head_to_head("t", "x", task_type, 1.0, 0.991, PASSED)
        assert short.outcome == TIGHT and short.deficit == 0.0
        failed = text_head_to_head("t", "x", task_type, 1.0, 0.9, FAILED)
        assert failed.outcome == TIGHT and failed.deficit == 0.0  # ahead by 10% but the paired test failed
        drawn = text_head_to_head("t", "x", task_type, 1.0, 0.9, DRAWN)
        assert drawn.outcome == TIGHT

    @pytest.mark.parametrize("task_type", [TaskType.INSTRUCTTEXTTASK, TaskType.CHATTASK, TaskType.DPOTASK])
    def test_missing_vectors_leave_the_task_with_the_boss_with_its_deficit(self, task_type):
        ahead = text_head_to_head("t", "x", task_type, 1.0, 0.9, None)
        behind = text_head_to_head("t", "x", task_type, 1.0, 1.002, None)

        assert ahead.outcome == TIGHT and ahead.deficit == 0.0
        assert behind.outcome == LOOSER and behind.deficit == pytest.approx(0.002)

    def test_no_comparable_example_is_treated_like_missing_vectors(self):
        nothing = text_head_to_head("t", "x", TaskType.DPOTASK, 1.0, 1.002, NOTHING_COMPARABLE)
        assert nothing.outcome == LOOSER and nothing.deficit == pytest.approx(0.002)
        assert "no pairable" in nothing.reason

    @pytest.mark.parametrize("boss", [None, math.nan, math.inf])
    def test_a_boss_without_a_valid_score_loses_to_a_valid_challenger_without_vectors(self, boss):
        assert text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, boss, 1.0, None).is_win

    def test_grpo_is_the_score_alone(self):
        assert text_head_to_head("t", "g", TaskType.GRPOTASK, 1.0, 1.01, None).is_win
        assert text_head_to_head("t", "g", TaskType.GRPOTASK, 1.0, 1.01, FAILED).is_win
        assert not text_head_to_head("t", "g", TaskType.GRPOTASK, 1.0, 1.0099, None).is_win

    @pytest.mark.parametrize(
        "challenger, outcome",
        [
            (1.001, TIGHT),  # exactly 0.1% behind
            (1.001 * (1 + 1e-6), LOOSER),
            (1.0025, LOOSER),  # exactly 0.25% behind
            (1.0025 * (1 + 1e-6), BEYOND),
        ],
    )
    def test_the_deficit_boundaries(self, challenger, outcome):
        assert text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, 1.0, challenger, PASSED).outcome == outcome

    def test_no_valid_challenger_is_beyond(self):
        result = text_head_to_head("t", "x", TaskType.DPOTASK, 1.0, None, None)

        assert result.outcome == BEYOND and math.isinf(result.deficit)


def _freeform(total_bytes: float | None, passed_bar: bool = True, quant_failure: str | None = None) -> MinerResultsText:
    return MinerResultsText(
        hotkey="h", test_loss=1.0, synth_loss=1.0, is_finetune=True, task_type=TaskType.QUANTFREEFORMTASK,
        artifact_bytes=total_bytes, decoder_bytes=None if total_bytes is None else 0, passed_bar=passed_bar,
        quant_failure=quant_failure,
    )  # fmt: skip


class TestFreeformAgainstTheBoss:
    @pytest.mark.parametrize(
        "challenger_bytes, outcome",
        [
            (97_000, WIN),  # exactly 3% fewer
            (97_001, TIGHT),  # fewer, by less than 3%
            (100_000, TIGHT),
            (101_000, TIGHT),  # exactly 1% more
            (101_001, LOOSER),
            (102_000, LOOSER),  # exactly 2% more
            (102_001, BEYOND),
        ],
    )
    def test_the_byte_boundaries(self, challenger_bytes, outcome):
        result = freeform_head_to_head("t", "x", _freeform(100_000), _freeform(challenger_bytes))

        assert result.outcome == outcome

    @pytest.mark.parametrize(
        "challenger", [None, _freeform(90_000, passed_bar=False), _freeform(90_000, quant_failure="bad"), _freeform(None)]
    )
    def test_an_invalid_challenger_entry_is_beyond(self, challenger):
        assert freeform_head_to_head("t", "x", _freeform(100_000), challenger).outcome == BEYOND

    @pytest.mark.parametrize(
        "boss", [None, _freeform(90_000, passed_bar=False), _freeform(90_000, quant_failure="bad"), _freeform(None)]
    )
    def test_an_invalid_boss_entry_loses_to_a_valid_challenger(self, boss):
        assert freeform_head_to_head("t", "x", boss, _freeform(200_000)).is_win
        assert freeform_head_to_head("t", "x", boss, None).outcome == BEYOND


class TestFixedAgainstTheBoss:
    def test_a_win_needs_the_paired_test_and_one_percent(self):
        assert fixed_head_to_head("t", "x", 1.0, 0.99, PASSED, False).is_win
        assert not fixed_head_to_head("t", "x", 1.0, 0.99 * (1 + 1e-6), PASSED, False).is_win

    def test_a_paired_win_without_the_scalar_minimum_is_a_draw(self):
        assert fixed_head_to_head("t", "x", 1.0, 0.995, PASSED, False).outcome == TIGHT

    def test_a_draw_on_the_paired_test_is_within_both_allowances(self):
        assert fixed_head_to_head("t", "x", 1.0, 1.0, DRAWN, False).outcome == TIGHT
        assert fixed_head_to_head("t", "x", 1.0, 1.01, FAILED, False).outcome == TIGHT

    def test_a_task_the_boss_takes_on_the_paired_test_is_beyond(self):
        assert fixed_head_to_head("t", "x", 1.0, 1.1, FAILED, True).outcome == BEYOND

    def test_missing_vectors_leave_the_task_with_the_boss(self):
        assert fixed_head_to_head("t", "x", 1.0, 0.5, None, False).outcome == BEYOND

    def test_no_comparable_example_leaves_the_task_with_the_boss_like_missing_vectors(self):
        result = fixed_head_to_head("t", "x", 1.0, 0.5, NOTHING_COMPARABLE, False)
        assert result.outcome == BEYOND and "no pairable" in result.reason

    @pytest.mark.parametrize("boss", [None, math.inf, math.nan])
    def test_a_scored_fail_boss_loses_to_a_valid_challenger_even_without_vectors(self, boss):
        assert fixed_head_to_head("t", "x", boss, 1.0, None, False).is_win

    @pytest.mark.parametrize("challenger", [None, math.inf, math.nan])
    def test_an_invalid_challenger_is_beyond(self, challenger):
        assert fixed_head_to_head("t", "x", 1.0, challenger, PASSED, False).outcome == BEYOND
