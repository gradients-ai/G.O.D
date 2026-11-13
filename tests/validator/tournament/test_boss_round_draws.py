"""Quant boss-round draws are decided misses, and the dethrone bar stays pinned to the built size.

A draw is zero decided examples: every held-out example landed inside the tie dead zone, so nothing
separated the two models. The boss holds the task and it is not a win for the challenger, like any other
miss: the challenger needs five wins of the six the round is built with. There are no replacement tasks. A task
skipped for lack of evidence is the same: it is simply not a win, and the pinned bar stays at six.
"""

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from validator.tasks.models import QuantFixedRawTask
from validator.tournament import constants as t_cst
from validator.tournament import round_results
from validator.tournament.models import BossRoundTaskVerdict
from validator.tournament.models import RoundStatus
from validator.tournament.models import RoundType
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.round_results import determine_boss_round_winner
from validator.tournament.thresholds import compare_paired_losses


async def _async(value):
    return value


async def _async_none(**_kwargs):
    return None


BOSS = "boss_hotkey"
CHALLENGER = "challenger_hotkey"
QUANT_BOSS_ROUND_SIZE = t_cst.FINAL_ROUND_QUANT_TASKS  # six


def _round_tasks(n: int) -> list[TournamentTask]:
    return [
        TournamentTask(
            tournament_id="tourn_1",
            round_id="tourn_1_round_004",
            task_id=f"00000000-0000-0000-0000-{i:012d}",
            pair_id="tourn_1_round_004_pair_001",
        )
        for i in range(n)
    ]


def _decide(winners, expected=QUANT_BOSS_ROUND_SIZE):
    return determine_boss_round_winner(winners, BOSS, TournamentType.QUANT, expected_task_count=expected)


def test_all_examples_inside_dead_zone_is_a_draw():
    """The trigger condition: no example separated the two models by more than the dead zone."""
    boss = np.random.default_rng(7).uniform(0.02, 0.03, 800)
    challenger = boss - (t_cst.BOSS_ROUND_TIE_DEADZONE_NATS / 2)

    result = compare_paired_losses(list(boss), list(challenger))

    assert result.n_decided == 0
    assert result.is_draw is True
    assert result.challenger_won is False


def test_a_skipped_task_never_lowers_the_dethrone_bar():
    """Four decided tasks of a built six: the challenger cannot get in on four, however clean.

    Excluding a skipped task takes it out of the numerator; it must not also shrink the denominator.
    """
    assert _decide([CHALLENGER] * 4) == BOSS
    assert _decide([CHALLENGER] * 4, expected=4) == BOSS
    assert _decide([CHALLENGER] * 5) == CHALLENGER


def test_bar_is_never_clamped_down_to_what_decided():
    """One decided task against a built size of six cannot be crowned, won or not."""
    assert _decide([CHALLENGER]) == BOSS
    assert _decide([BOSS]) == BOSS


def test_a_draw_is_a_miss_and_a_second_miss_ends_the_challenge():
    """A drawn task is the boss's like any other: the sixth may be a miss, the fifth may not."""
    assert _decide([CHALLENGER] * 5 + [BOSS]) == CHALLENGER
    assert _decide([CHALLENGER] * 4 + [BOSS, BOSS]) == BOSS


@pytest.mark.parametrize(
    "tournament_type,expected",
    [
        (TournamentType.QUANT, 6),
        (TournamentType.TEXT, t_cst.FINAL_ROUND_TEXT_TASKS),
        # Image keeps deriving the bar from its own count, so it gets 0.
        (TournamentType.IMAGE, 0),
    ],
)
def test_expected_boss_round_task_count(tournament_type, expected):
    assert t_cst.expected_boss_round_task_count(tournament_type) == expected


def test_freeform_can_never_draw():
    """Freeform is decided on effective bytes, never on the paired gate: only fixed can draw."""
    assert TaskType.QUANTFREEFORMTASK not in t_cst.PAIRED_BOSS_ROUND_TASK_TYPES


def _final_round() -> TournamentRoundData:
    return TournamentRoundData(
        round_id="tourn_1_round_004",
        tournament_id="tourn_1",
        round_number=4,
        round_type=RoundType.KNOCKOUT,
        is_final_round=True,
        status=RoundStatus.PENDING,
    )


@pytest.mark.asyncio
async def test_orphaned_task_defers_the_round_instead_of_scoring_it(monkeypatch):
    """A boss-round task with no assigned nodes never ran, so it must not be scored against anyone.

    One gets here when a task is created but the round never makes it back to PENDING - a crash, or a
    failure partway through creating a batch. The challenger is not charged a loss for it;
    the round is deferred and sent back for node assignment.
    """
    round_tasks = _round_tasks(QUANT_BOSS_ROUND_SIZE + 1)
    orphan_id = round_tasks[-1].task_id

    async def fake_get_tournament(_tournament_id, _psql_db):
        return TournamentData(tournament_id="tourn_1", tournament_type=TournamentType.QUANT)

    async def fake_count_hotkeys_assigned_to_task(task_id, _psql_db):
        return 0 if task_id == orphan_id else 2

    monkeypatch.setattr(round_results, "get_tournament", fake_get_tournament)
    monkeypatch.setattr(round_results, "count_hotkeys_assigned_to_task", fake_count_hotkeys_assigned_to_task)

    outcome = await round_results.get_knockout_winners(_final_round(), round_tasks, psql_db=None, config=None)

    assert outcome.is_deferred is True
    assert outcome.unassigned_task_ids == [orphan_id]
    assert outcome.winners == []


@pytest.mark.asyncio
async def test_a_drawn_fixed_quant_task_is_a_miss_not_a_replacement(monkeypatch):
    """A drawn task goes to the boss and is not a win; the round resolves, nothing is queued."""
    rows = _round_tasks(QUANT_BOSS_ROUND_SIZE)
    fixed = QuantFixedRawTask(
        task_id=rows[0].task_id,
        model_id="org/dense-3b",
        ds="ds",
        dataset_type=InstructTextDatasetType(field_instruction="instruction", field_output="output"),
        quant_spec=QuantFormat.W3A16_ASYM_G128.spec(),
        status="success",
        is_organic=False,
        account_id=uuid4(),
        hours_to_complete=1.0,
        created_at=datetime(2026, 1, 1),
    )
    others = [
        SimpleNamespace(task_id=row.task_id, task_type=TaskType.QUANTFIXEDTASK, status=TaskStatus.SUCCESS) for row in rows[1:]
    ]
    tasks = {rows[0].task_id: fixed, **{task.task_id: task for task in others}}

    async def _tournament(_tournament_id, _psql_db):
        return TournamentData(tournament_id="tourn_1", tournament_type=TournamentType.QUANT)

    async def _resolve(**kwargs):
        if kwargs["task_id"] == rows[0].task_id:
            return BossRoundTaskVerdict(
                winner_hotkey=BOSS, decided_by="a draw", is_draw=True, challenger_within_allowed_loss=True
            )
        return BossRoundTaskVerdict(winner_hotkey=CHALLENGER, decided_by="won")

    monkeypatch.setattr(round_results, "EMISSION_BURN_HOTKEY", BOSS)
    monkeypatch.setattr(round_results, "get_tournament", _tournament)
    monkeypatch.setattr(round_results, "count_hotkeys_assigned_to_task", lambda _tid, _db: _async(2))
    monkeypatch.setattr(round_results, "get_task", lambda tid, _db: _async(tasks[tid]))
    monkeypatch.setattr(
        round_results,
        "get_task_results_for_ranking",
        lambda _tid, _db: _async([SimpleNamespace(hotkey=BOSS), SimpleNamespace(hotkey=CHALLENGER)]),
    )
    monkeypatch.setattr(
        round_results,
        "calculate_miner_ranking_and_scores",
        lambda ranked: [SimpleNamespace(hotkey=r.hotkey, adjusted_loss=0.5 if r.hotkey == BOSS else 0.4) for r in ranked],
    )
    monkeypatch.setattr(round_results, "_resolve_boss_round_task_winner", _resolve)
    monkeypatch.setattr(round_results, "update_threshold_adjusted_quality_scores_for_task", _async_none)
    calls = []
    real = round_results.determine_boss_round_winner
    monkeypatch.setattr(round_results, "determine_boss_round_winner", lambda *a, **k: calls.append(k) or real(*a, **k))

    outcome = await round_results.get_knockout_winners(_final_round(), rows, psql_db=None, config=None)

    assert outcome.is_deferred is False
    assert outcome.draw_resolution is None
    assert outcome.winners == [CHALLENGER]  # the five other tasks are won; the draw is the one that is not
    assert calls[0]["expected_task_count"] == QUANT_BOSS_ROUND_SIZE
