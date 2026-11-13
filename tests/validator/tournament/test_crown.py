"""The shared crown rule (crown.py) and each task kind's win and allowances, as pure functions.

Win: text is the paired per-example test plus a 1% better score on instruct, chat and DPO, GRPO the score alone; freeform
is 3% fewer effective bytes; fixed is the paired test plus a 1% better scalar. Every other result is banded on the value
alone, the same for the three kinds: strictly better than the boss is within the smaller allowance, level or at most 1%
worse is within the larger allowance, more than 1% worse is beyond.
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


WIN, SMALLER, LARGER, BEYOND = (
    HeadToHeadOutcome.WIN,
    HeadToHeadOutcome.WITHIN_SMALLER,
    HeadToHeadOutcome.WITHIN_LARGER,
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
            ("six wins, the seventh within the larger allowance", [WIN] * 6 + [LARGER], True),
            ("six wins, the seventh within the smaller allowance", [WIN] * 6 + [SMALLER], True),
            ("six wins, the seventh beyond", [WIN] * 6 + [BEYOND], False),
            ("five wins, one within the smaller and one within the larger allowance", [WIN] * 5 + [SMALLER, LARGER], True),
            ("five wins, two within the smaller allowance", [WIN] * 5 + [SMALLER, SMALLER], True),
            ("five wins, two within the larger allowance only", [WIN] * 5 + [LARGER, LARGER], False),
            ("five wins, one within the smaller allowance and one beyond", [WIN] * 5 + [SMALLER, BEYOND], False),
            ("four wins, three within the smaller allowance", [WIN] * 4 + [SMALLER] * 3, False),
        ],
    )
    def test_the_table(self, name, outcomes, crowned):
        assert _verdict(outcomes).crowned is crowned, name

    @pytest.mark.parametrize(
        "outcomes, possible",
        [
            ([WIN] * 3, True),
            ([WIN, SMALLER, LARGER], True),
            ([WIN, LARGER, LARGER], False),
            ([WIN, WIN, BEYOND], False),
            ([SMALLER, SMALLER, SMALLER], False),
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
        assert t_cst.TEXT_CROWN_WIN_MARGIN == 0.01
        assert t_cst.CROWN_LARGER_ALLOWANCE == 0.01
        assert t_cst.QUANT_FREEFORM_BOSS_BYTE_MARGIN == 0.03
        assert t_cst.QUANT_FIXED_BOSS_MIN_IMPROVEMENT == 0.01


class TestTextAgainstTheBoss:
    @pytest.mark.parametrize("task_type", [TaskType.INSTRUCTTEXTTASK, TaskType.CHATTASK, TaskType.DPOTASK])
    def test_a_win_needs_the_paired_test_and_one_percent(self, task_type):
        assert text_head_to_head("t", "x", task_type, 1.0, 0.99, PASSED).is_win
        short = text_head_to_head("t", "x", task_type, 1.0, 0.991, PASSED)
        assert short.outcome == SMALLER and short.deficit == 0.0
        failed = text_head_to_head("t", "x", task_type, 1.0, 0.9, FAILED)
        assert failed.outcome == SMALLER and failed.deficit == 0.0  # ahead by 10% but the paired test failed
        drawn = text_head_to_head("t", "x", task_type, 1.0, 0.9, DRAWN)
        assert drawn.outcome == SMALLER

    @pytest.mark.parametrize("task_type", [TaskType.INSTRUCTTEXTTASK, TaskType.CHATTASK, TaskType.DPOTASK])
    def test_missing_vectors_leave_the_task_with_the_boss_with_its_deficit(self, task_type):
        ahead = text_head_to_head("t", "x", task_type, 1.0, 0.9, None)
        behind = text_head_to_head("t", "x", task_type, 1.0, 1.002, None)

        assert ahead.outcome == SMALLER and ahead.deficit == 0.0
        assert behind.outcome == LARGER and behind.deficit == pytest.approx(0.002)

    def test_no_comparable_example_is_treated_like_missing_vectors(self):
        nothing = text_head_to_head("t", "x", TaskType.DPOTASK, 1.0, 1.002, NOTHING_COMPARABLE)
        assert nothing.outcome == LARGER and nothing.deficit == pytest.approx(0.002)
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
            (0.995, SMALLER),  # strictly better, short of the win margin
            (1.0 - 1e-12, SMALLER),
            (1.0, LARGER),  # level
            (1.0 + 1e-12, LARGER),
            (1.01, LARGER),  # exactly 1% worse
            (1.01 * (1 + 1e-6), BEYOND),
        ],
    )
    def test_the_non_win_bands(self, challenger, outcome):
        assert text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, 1.0, challenger, PASSED).outcome == outcome

    @pytest.mark.parametrize(
        "challenger, outcome",
        [(1.005, SMALLER), (1.0, LARGER), (0.99, LARGER), (0.99 * (1 - 1e-6), BEYOND)],
    )
    def test_grpo_bands_run_the_other_way(self, challenger, outcome):
        # Higher is better: a slightly higher score is within the smaller allowance, exactly 1% lower is within the
        # larger allowance, more than that is beyond.
        assert text_head_to_head("t", "g", TaskType.GRPOTASK, 1.0, challenger, None).outcome == outcome

    @pytest.mark.parametrize("task_type", [TaskType.INSTRUCTTEXTTASK, TaskType.CHATTASK, TaskType.DPOTASK])
    @pytest.mark.parametrize("challenger, outcome", [(0.5, SMALLER), (1.0, LARGER), (1.01, LARGER), (1.02, BEYOND)])
    def test_unpairable_text_is_banded_on_the_score(self, task_type, challenger, outcome):
        for paired in (None, NOTHING_COMPARABLE):
            result = text_head_to_head("t", "x", task_type, 1.0, challenger, paired)
            assert result.outcome == outcome and "no pairable" in result.reason

    @pytest.mark.parametrize("boss", [1.15, 2.3, 0.7, 123.456789])
    def test_float_awkward_one_percent_worse_is_larger(self, boss):
        worse = boss * 1.01
        assert text_head_to_head("t", "x", TaskType.INSTRUCTTEXTTASK, boss, worse, PASSED).outcome == LARGER

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
            (96_000, WIN),
            (97_001, SMALLER),  # fewer, by just under 3%
            (99_999, SMALLER),  # one byte fewer
            (100_000, LARGER),  # equal bytes
            (101_000, LARGER),  # exactly 1% more
            (101_001, BEYOND),  # just over 1% more
            (150_000, BEYOND),
        ],
    )
    def test_the_byte_boundaries(self, challenger_bytes, outcome):
        result = freeform_head_to_head("t", "x", _freeform(100_000), _freeform(challenger_bytes))

        assert result.outcome == outcome

    @pytest.mark.parametrize("boss_bytes", [1_000_000_000, 123_456_700, 987_654_300, 999_999_900])
    @pytest.mark.parametrize(
        "percent, extra_bytes, outcome",
        [(97, 0, WIN), (97, 1000, SMALLER), (101, 0, LARGER), (101, 1000, BEYOND)],
    )
    def test_float_rounding_does_not_move_the_boundaries(self, boss_bytes, percent, extra_bytes, outcome):
        # Whole-byte counts exactly on 97% and 101% of the boss (where (1 - 0.03) * boss is not exact in binary), and a
        # count 1000 bytes past each.
        challenger_bytes = boss_bytes * percent // 100 + extra_bytes
        result = freeform_head_to_head("t", "x", _freeform(boss_bytes), _freeform(challenger_bytes))

        assert result.outcome == outcome

    def test_the_deficit_is_how_many_more_bytes_relative_to_the_boss(self):
        assert freeform_head_to_head("t", "x", _freeform(100_000), _freeform(97_000)).deficit == 0.0
        assert freeform_head_to_head("t", "x", _freeform(100_000), _freeform(98_000)).deficit == 0.0
        assert freeform_head_to_head("t", "x", _freeform(100_000), _freeform(100_000)).deficit == 0.0
        worse = freeform_head_to_head("t", "x", _freeform(100_000), _freeform(110_000))
        assert worse.deficit == pytest.approx(0.1) and "no fewer" in worse.reason
        fewer = freeform_head_to_head("t", "x", _freeform(100_000), _freeform(98_000))
        assert "2.000% fewer" in fewer.reason

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

    @pytest.mark.parametrize(
        "challenger, paired, boss_takes, outcome",
        [
            (0.99, PASSED, False, WIN),  # exactly 1% lower, paired test passed
            (0.99, FAILED, False, SMALLER),  # 1% lower without the paired test
            (0.99, DRAWN, False, SMALLER),
            (0.995, PASSED, False, SMALLER),  # paired test passed, under 1% lower
            (0.999, FAILED, True, SMALLER),  # slightly lower with the boss taking the paired test
            (1.0, DRAWN, False, LARGER),  # exactly equal
            (1.0, PASSED, False, LARGER),
            (1.01, FAILED, False, LARGER),  # exactly 1% higher
            (1.01, FAILED, True, LARGER),
            (1.01 * (1 + 1e-6), FAILED, False, BEYOND),  # just over 1% higher
            (1.1, FAILED, True, BEYOND),
        ],
    )
    def test_the_bands(self, challenger, paired, boss_takes, outcome):
        assert fixed_head_to_head("t", "x", 1.0, challenger, paired, boss_takes).outcome == outcome

    def test_the_reason_says_which_case_it_was(self):
        assert "not 1% better" in fixed_head_to_head("t", "x", 1.0, 0.995, PASSED, False).reason
        assert "boss takes it on the paired test" in fixed_head_to_head("t", "x", 1.0, 1.1, FAILED, True).reason
        assert "a draw" in fixed_head_to_head("t", "x", 1.0, 1.0, DRAWN, False).reason
        assert "not passed" in fixed_head_to_head("t", "x", 1.0, 1.0, FAILED, False).reason

    def test_the_deficit_is_how_much_worse_relative_to_the_boss(self):
        assert fixed_head_to_head("t", "x", 2.0, 1.0, PASSED, False).deficit == 0.0
        assert fixed_head_to_head("t", "x", 2.0, 2.0, DRAWN, False).deficit == 0.0
        assert fixed_head_to_head("t", "x", 2.0, 2.02, FAILED, False).deficit == pytest.approx(0.01)
        assert math.isinf(fixed_head_to_head("t", "x", 1.0, None, PASSED, False).deficit)

    @pytest.mark.parametrize("paired", [None, NOTHING_COMPARABLE])
    @pytest.mark.parametrize("challenger, outcome", [(0.5, SMALLER), (1.0, LARGER), (1.01, LARGER), (1.02, BEYOND)])
    def test_unpairable_fixed_is_banded_on_the_scalar_loss(self, paired, challenger, outcome):
        result = fixed_head_to_head("t", "x", 1.0, challenger, paired, False)
        assert result.outcome == outcome and "no pairable" in result.reason

    @pytest.mark.parametrize("boss", [None, math.inf, math.nan])
    def test_a_scored_fail_boss_loses_to_a_valid_challenger_even_without_vectors(self, boss):
        assert fixed_head_to_head("t", "x", boss, 1.0, None, False).is_win

    @pytest.mark.parametrize("challenger", [None, math.inf, math.nan])
    def test_an_invalid_challenger_is_beyond(self, challenger):
        assert fixed_head_to_head("t", "x", 1.0, challenger, PASSED, False).outcome == BEYOND
