"""The QUANT dethrone rule: the boss round is six compression tasks and the challenger must win five of them.

Freeform needs 3% fewer effective bytes; fixed needs the paired per-example test and a 1% better scalar loss. Nothing is
asked of the sixth task, and the bar is always the built size: a task that is missing, undecided, drawn or without a
result is simply not a win.

Also pins TEXT and IMAGE to the rule they have always had: quant must not leak into determine_boss_round_winner for them.
"""

import itertools
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.tournament.round_results as round_results
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.tournament import constants as t_cst
from validator.tournament.models import BossRoundTaskVerdict
from validator.tournament.models import ContinuousSftLineageOutcome
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus
from validator.tournament.round_results import _quant_boss_round_crowned
from validator.tournament.round_results import determine_boss_round_winner


BOSS = EMISSION_BURN_HOTKEY
CHALLENGER = "5Challenger"
FREEFORM_ID = "11111111-2222-3333-4444-000000000002"
FIXED, FREEFORM = TaskType.QUANTFIXEDTASK, TaskType.QUANTFREEFORMTASK
SUCCESS, FAILURE = TrainingStatus.SUCCESS, TrainingStatus.FAILURE
BUILT = 6


def crown(task_winners: list[str], expected_task_count: int | None = BUILT) -> str:
    return determine_boss_round_winner(task_winners, BOSS, TournamentType.QUANT, expected_task_count=expected_task_count)


def winners(challenger_wins: int, boss_wins: int) -> list[str]:
    return [CHALLENGER] * challenger_wins + [BOSS] * boss_wins


# --- determine_boss_round_winner, QUANT -----------------------------------------------------------------------------


def test_the_quant_boss_round_is_six_tasks_and_five_must_be_won():
    assert t_cst.expected_boss_round_task_count(TournamentType.QUANT) == t_cst.FINAL_ROUND_QUANT_TASKS == 6
    assert t_cst.QUANT_BOSS_ROUND_ALLOWED_LOSSES == 1
    assert t_cst.FINAL_ROUND_QUANT_TASKS - t_cst.QUANT_BOSS_ROUND_ALLOWED_LOSSES == 5


@pytest.mark.parametrize(
    "challenger_wins,boss_wins,crowned",
    [
        (6, 0, True),
        (5, 1, True),  # the sixth, lost or drawn or without a result, is not looked at
        (4, 2, False),
        (3, 3, False),
        (1, 5, False),
        (0, 6, False),
        # Five tasks attached (a seed failed and was never attached): all five must be won.
        (5, 0, True),
        (4, 1, False),
        # Four or fewer cannot reach five wins.
        (4, 0, False),
        (3, 0, False),
        (1, 0, False),
    ],
)
def test_crown_table_against_the_built_size(challenger_wins, boss_wins, crowned):
    expected = CHALLENGER if crowned else BOSS
    assert crown(winners(challenger_wins, boss_wins)) == expected


def test_the_bar_is_the_built_size_whatever_the_caller_passes():
    """Four decided tasks won outright do not crown, even if a caller passes the decided count as the expected size."""
    assert crown(winners(4, 0), expected_task_count=4) == BOSS
    assert crown(winners(4, 0), expected_task_count=None) == BOSS
    assert crown(winners(5, 0), expected_task_count=5) == CHALLENGER


def test_no_decided_tasks_keeps_the_boss():
    assert crown([]) == BOSS


def test_the_boss_only_ever_wins_by_the_challenger_missing_five():
    assert crown(winners(0, 6)) == BOSS and crown([BOSS]) == BOSS


@pytest.mark.parametrize(
    "opponent_wins,built_size,crowned",
    [(6, 6, True), (5, 6, True), (4, 6, False), (0, 6, False)],
)
def test_crown_rule_function_table(opponent_wins, built_size, crowned):
    assert _quant_boss_round_crowned(opponent_wins, built_size)[0] is crowned


def test_the_crown_reason_names_the_wins_and_the_bar():
    crowned, reason = _quant_boss_round_crowned(4, 6)
    assert not crowned and "4 of 6" in reason and "requires 5" in reason


# --- per-task resolution of the freeform slot -----------------------------------------------------------------------


def _freeform(
    hotkey: str, total_bytes: int | None, passed_bar: bool = True, quant_failure: str | None = None
) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=float("nan") if quant_failure else 1.0,
        synth_loss=1.0,
        is_finetune=True,
        task_type=FREEFORM,
        artifact_bytes=total_bytes,
        decoder_bytes=None if total_bytes is None else 0,
        passed_bar=passed_bar,
        quant_failure=quant_failure,
    )


async def _freeform_verdict(monkeypatch, challenger_bytes: int, boss_bytes: int = 2000) -> BossRoundTaskVerdict:
    monkeypatch.setattr(round_results, "update_threshold_adjusted_quality_scores_for_task", AsyncMock())
    return await round_results._resolve_freeform_boss_round_task(
        FREEFORM_ID, _freeform(BOSS, boss_bytes), _freeform(CHALLENGER, challenger_bytes), psql_db=None
    )


@pytest.mark.asyncio
async def test_freeform_win_needs_three_percent_fewer_effective_bytes(monkeypatch):
    assert t_cst.QUANT_FREEFORM_BOSS_BYTE_MARGIN == 0.03
    assert (await _freeform_verdict(monkeypatch, 1942)).winner_hotkey == BOSS  # 2.9% fewer: not a win
    assert (await _freeform_verdict(monkeypatch, 1938)).winner_hotkey == CHALLENGER  # 3.1% fewer: a win


@pytest.mark.asyncio
async def test_freeform_within_one_percent_worse_is_a_narrow_loss(monkeypatch):
    assert t_cst.BOSS_ROUND_FREEFORM_ALLOWED_LOSS_WITHIN == 0.01
    narrow = await _freeform_verdict(monkeypatch, 2020)
    assert narrow.winner_hotkey == BOSS and narrow.challenger_within_allowed_loss is True
    just_outside = await _freeform_verdict(monkeypatch, 2021)
    assert just_outside.winner_hotkey == BOSS and just_outside.challenger_within_allowed_loss is False
    equal = await _freeform_verdict(monkeypatch, 2000)
    assert equal.challenger_within_allowed_loss is True


@pytest.mark.asyncio
async def test_freeform_challenger_over_the_cap_is_a_clear_loss(monkeypatch):
    monkeypatch.setattr(round_results, "update_threshold_adjusted_quality_scores_for_task", AsyncMock())
    verdict = await round_results._resolve_freeform_boss_round_task(
        FREEFORM_ID, _freeform(BOSS, 2000), _freeform(CHALLENGER, 10, passed_bar=False), psql_db=None
    )
    assert verdict.winner_hotkey == BOSS and verdict.challenger_within_allowed_loss is False


# --- the whole six-task boss round through get_round_winners -------------------------------------------------------------


WIN, CLEAR_LOSS, NARROW_LOSS, DRAW, BOSS_ONLY, NOBODY, BOSS_FAILED = (
    "win", "clear_loss", "narrow_loss", "draw", "boss_only", "nobody", "boss_failed",
)
# The six roles of the round: freeform and fixed for the round-2, round-3 and continuous source models.
ROLES = (FREEFORM, FIXED, FREEFORM, FIXED, FREEFORM, FIXED)
TASK_IDS = [f"11111111-2222-3333-4444-{i:012d}" for i in range(1, 7)]


def _fixed_result(hotkey: str, loss: float | None) -> MinerResultsText:
    return MinerResultsText(hotkey=hotkey, test_loss=1.0, synth_loss=1.0, is_finetune=True, task_type=FIXED, adjusted_loss=loss)


class Round:
    """A quant boss round with scripted per-task evidence. outcomes[i] is how role i went, or None when it was never attached."""

    def __init__(self, monkeypatch, outcomes: list[str | None]):
        self.monkeypatch = monkeypatch
        self.outcomes = outcomes
        self.verdicts: dict[str, BossRoundTaskVerdict] = {}
        self.results: dict[str, list] = {}
        self.statuses: dict[str, dict] = {}
        self.task_types: dict[str, TaskType] = {}
        self.attached: list[TournamentTask] = []
        for task_id, task_type, outcome in zip(TASK_IDS, ROLES, outcomes):
            if outcome is None:
                continue
            self.attached.append(TournamentTask(tournament_id="t", round_id="r", task_id=task_id))
            self.task_types[task_id] = task_type
            self._script(task_id, task_type, outcome)

    def _script(self, task_id: str, task_type: TaskType, outcome: str) -> None:
        both_ok = {BOSS: SUCCESS, CHALLENGER: SUCCESS}
        self.statuses[task_id] = both_ok
        if task_type == FREEFORM:
            bytes_by_outcome = {WIN: 1900, CLEAR_LOSS: 2100, NARROW_LOSS: 2010, DRAW: 2000}
            if outcome in bytes_by_outcome:
                self.results[task_id] = [_freeform(BOSS, 2000), _freeform(CHALLENGER, bytes_by_outcome[outcome])]
            elif outcome == BOSS_ONLY:
                self.results[task_id] = [_freeform(BOSS, 2000)]
            elif outcome == BOSS_FAILED:
                self.results[task_id] = [_freeform(CHALLENGER, 1900)]
                self.statuses[task_id] = {BOSS: FAILURE, CHALLENGER: SUCCESS}
            else:
                self.results[task_id] = []
            return
        if outcome in (WIN, CLEAR_LOSS, NARROW_LOSS, DRAW):
            self.results[task_id] = [_fixed_result(BOSS, 1.0), _fixed_result(CHALLENGER, 0.9)]
            self.verdicts[task_id] = {
                WIN: BossRoundTaskVerdict(winner_hotkey=CHALLENGER, decided_by="per-example win rate"),
                CLEAR_LOSS: BossRoundTaskVerdict(winner_hotkey=BOSS, decided_by="per-example win rate"),
                NARROW_LOSS: BossRoundTaskVerdict(
                    winner_hotkey=BOSS, decided_by="per-example win rate", challenger_within_allowed_loss=True
                ),
                DRAW: BossRoundTaskVerdict(
                    winner_hotkey=BOSS, decided_by="a draw", is_draw=True, challenger_within_allowed_loss=True
                ),
            }[outcome]
        elif outcome == BOSS_ONLY:
            self.results[task_id] = [_fixed_result(BOSS, 1.0)]
        elif outcome == BOSS_FAILED:
            self.results[task_id] = [_fixed_result(CHALLENGER, 0.9)]
            self.statuses[task_id] = {BOSS: FAILURE, CHALLENGER: SUCCESS}
        else:
            self.results[task_id] = []

    async def run(self):
        mp = self.monkeypatch

        async def fake_task(task_id, psql_db):
            return SimpleNamespace(
                task_type=self.task_types[str(task_id)], ds="d", training_start_point=None, status=TaskStatus.SUCCESS
            )

        async def fake_results(task_id, psql_db):
            return self.results[str(task_id)]

        async def fake_fixed(**kwargs):
            return self.verdicts[str(kwargs["task_id"])]

        async def no_seed_failure(*args, **kwargs):
            return None

        tournament = TournamentData(tournament_id="t", tournament_type=TournamentType.QUANT)
        mp.setattr(round_results, "get_tournament_tasks", AsyncMock(return_value=self.attached))
        mp.setattr(round_results, "get_tournament", AsyncMock(return_value=tournament))
        mp.setattr(round_results, "count_hotkeys_assigned_to_task", AsyncMock(return_value=2))
        mp.setattr(round_results, "get_task", fake_task)
        mp.setattr(round_results, "boss_side_seed_failure", no_seed_failure)
        mp.setattr(round_results, "get_task_results_for_ranking", fake_results)
        mp.setattr(round_results, "calculate_miner_ranking_and_scores", lambda ranked: ranked)
        mp.setattr(round_results, "_resolve_boss_round_task_winner", fake_fixed)
        mp.setattr(round_results, "update_threshold_adjusted_quality_scores_for_task", AsyncMock())
        mp.setattr(
            round_results,
            "get_training_status_for_task_and_hotkeys",
            AsyncMock(side_effect=lambda task_id, hotkeys, db: self.statuses[str(task_id)]),
        )
        completed = TournamentRoundData(
            round_id="r", tournament_id="t", round_number=4, round_type=RoundType.KNOCKOUT, is_final_round=True
        )
        outcome = await round_results.get_round_winners(completed, psql_db=None, config=None)
        # A quant boss round never adds replacement tasks, drawn or not.
        assert outcome.deferred_reason is None and outcome.draw_resolution is None
        return outcome.winners[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcomes,crowned",
    [
        ([WIN] * 6, True),
        # Five wins and a sixth that went any way at all.
        ([WIN] * 5 + [CLEAR_LOSS], True),
        ([WIN] * 5 + [NARROW_LOSS], True),
        ([WIN] * 5 + [DRAW], True),
        ([WIN] * 5 + [BOSS_ONLY], True),
        ([WIN] * 5 + [NOBODY], True),
        ([CLEAR_LOSS] + [WIN] * 5, True),
        ([WIN, WIN, WIN, WIN, CLEAR_LOSS, WIN], True),
        # Four wins, however narrow the two losses.
        ([WIN] * 4 + [NARROW_LOSS, NARROW_LOSS], False),
        ([WIN] * 4 + [DRAW, DRAW], False),
        ([WIN] * 4 + [CLEAR_LOSS, NARROW_LOSS], False),
        ([WIN] * 4 + [BOSS_ONLY, NOBODY], False),
        ([WIN] * 3 + [CLEAR_LOSS] * 3, False),
        ([CLEAR_LOSS] * 6, False),
        ([NOBODY] * 6, False),
        # Five attached: all five must be won.
        ([WIN] * 5 + [None], True),
        ([None] + [WIN] * 5, True),
        ([WIN] * 4 + [CLEAR_LOSS, None], False),
        ([WIN] * 4 + [DRAW, None], False),
        ([WIN] * 4 + [NARROW_LOSS, None], False),
        # Four attached cannot reach five.
        ([WIN] * 4 + [None, None], False),
        ([None, None] + [WIN] * 4, False),
        ([WIN, None, WIN, None, WIN, WIN], False),
        # Nothing decided: the boss.
        ([None] * 6, False),
    ],
)
async def test_the_six_task_round_through_get_round_winners(monkeypatch, outcomes, crowned):
    winner = await Round(monkeypatch, outcomes).run()
    assert winner == (CHALLENGER if crowned else BOSS)


@pytest.mark.asyncio
async def test_boss_entries_failed_on_every_task_crowns_the_challenger(monkeypatch):
    assert await Round(monkeypatch, [BOSS_FAILED] * 6).run() == CHALLENGER


@pytest.mark.asyncio
async def test_one_failed_boss_entry_counts_as_a_challenger_win(monkeypatch):
    assert await Round(monkeypatch, [WIN] * 4 + [BOSS_FAILED, CLEAR_LOSS]).run() == CHALLENGER
    assert await Round(monkeypatch, [WIN] * 3 + [BOSS_FAILED, CLEAR_LOSS, CLEAR_LOSS]).run() == BOSS


@pytest.mark.asyncio
async def test_a_task_neither_side_has_a_result_on_is_not_a_win(monkeypatch):
    assert await Round(monkeypatch, [WIN] * 4 + [NOBODY, NOBODY]).run() == BOSS


@pytest.mark.asyncio
async def test_challenger_entry_failed_on_two_tasks_does_not_crown(monkeypatch):
    rnd = Round(monkeypatch, [WIN] * 4 + [BOSS_ONLY, BOSS_ONLY])
    rnd.statuses[TASK_IDS[4]] = {BOSS: SUCCESS, CHALLENGER: FAILURE}
    rnd.statuses[TASK_IDS[5]] = {BOSS: SUCCESS, CHALLENGER: FAILURE}
    assert await rnd.run() == BOSS


# --- TEXT and IMAGE keep their existing rule -----------------------------------------------------------------


def _main_rule(task_winners, boss, outcomes, expected_task_count):
    """The text and image boss-round rule, restated as an oracle."""
    if not task_winners:
        return boss
    wins = Counter(task_winners)
    total = len(task_winners)
    opponent = next((hotkey for hotkey in wins if hotkey != boss), None)
    opponent_wins = wins.get(opponent, 0) if opponent else 0
    required = min(max(1, max(total, expected_task_count or 0) - 1), total)
    continuous_ok = all(outcome.satisfied_by(opponent) for outcome in outcomes)
    return opponent if opponent and opponent_wins >= required and continuous_ok else boss


@pytest.mark.parametrize("tournament_type", [TournamentType.TEXT, TournamentType.IMAGE])
def test_text_and_image_follow_mains_rule_exactly(tournament_type):
    won = ContinuousSftLineageOutcome(lineage="a", winner_hotkey=CHALLENGER)
    lost_lineage = ContinuousSftLineageOutcome(lineage="b", winner_hotkey=BOSS)
    drawn = ContinuousSftLineageOutcome(lineage="c", is_draw=True)
    outcome_sets = [[], [won], [won, lost_lineage], [drawn, won], [ContinuousSftLineageOutcome(lineage="d")]]
    checked = 0
    for size in range(0, 7):
        for winners in itertools.product([BOSS, CHALLENGER], repeat=size):
            for outcomes in outcome_sets:
                for expected in (None, 2, 4, 6):
                    actual = determine_boss_round_winner(
                        list(winners), BOSS, tournament_type, outcomes, expected_task_count=expected
                    )
                    assert actual == _main_rule(list(winners), BOSS, outcomes, expected), (winners, outcomes, expected)
                    checked += 1
    assert checked > 1000
