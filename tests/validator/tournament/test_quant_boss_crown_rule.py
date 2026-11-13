"""The QUANT crown: seven head-to-heads against the boss, rounds 1-3 and the four-task boss round, on the shared rule.

Freeform wins with 3% fewer effective bytes, fixed with the paired per-example test plus a 1% better scalar loss; any other
result is banded on the value alone: strictly better is within the smaller allowance, level or at most 1% worse is
within the larger allowance, more is beyond. At most two non-wins, all within the larger allowance, and with two of them one within the smaller allowance (crown.py). Driven through
get_round_winners with the database faked: rounds 1-3 hold one seed task each, the boss round four.

Also pins TEXT and IMAGE to the rule they have always had in determine_boss_round_winner.
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
from validator.tournament.models import ContinuousSftLineageOutcome
from validator.tournament.models import RoundType
from validator.tournament.models import SeedRole
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentTask
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus
from validator.tournament.round_results import determine_boss_round_winner


BOSS = EMISSION_BURN_HOTKEY
CHALLENGER = "5Challenger"
FIXED, FREEFORM = TaskType.QUANTFIXEDTASK, TaskType.QUANTFREEFORMTASK
SUCCESS, FAILURE = TrainingStatus.SUCCESS, TrainingStatus.FAILURE
TID = "t"

# How a task went for the challenger.
(
    WIN, SMALLER, LARGER, BEYOND, NO_VECTORS, NO_VECTORS_LARGER, NO_VECTORS_BEYOND, BOSS_FAILED, BOSS_FAILED_NO_VECTORS, NOBODY
) = (
    "win", "smaller", "larger", "beyond", "no_vectors", "no_vectors_larger", "no_vectors_beyond", "boss_failed",
    "boss_failed_no_vectors", "nobody",
)  # fmt: skip
NO_VECTOR_OUTCOMES = (NO_VECTORS, NO_VECTORS_LARGER, NO_VECTORS_BEYOND, BOSS_FAILED_NO_VECTORS)
# The seven tasks: rounds 1-3 (freeform, fixed, freeform), then the boss round (R2 model freeform, R3 model fixed,
# continuous model freeform and fixed).
KINDS = (FREEFORM, FIXED, FREEFORM, FREEFORM, FIXED, FREEFORM, FIXED)
BOSS_ROUND_ROLES = (
    SeedRole.BOSS_R2_FREEFORM,
    SeedRole.BOSS_R3_FIXED,
    SeedRole.BOSS_CONTINUOUS_FREEFORM,
    SeedRole.BOSS_CONTINUOUS_FIXED,
)
TASK_IDS = [f"11111111-2222-3333-4444-{i:012d}" for i in range(1, 8)]
ROWS = 200


def test_the_quant_boss_round_is_four_tasks():
    assert t_cst.FINAL_ROUND_QUANT_TASKS == 4
    assert t_cst.FINAL_ROUND_QUANT_TASK_DISTRIBUTION == {FREEFORM: 2, FIXED: 2}


def _freeform(hotkey: str, total_bytes: int, passed_bar: bool = True) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey, test_loss=1.0, synth_loss=1.0, is_finetune=True, task_type=FREEFORM, artifact_bytes=total_bytes,
        decoder_bytes=0, passed_bar=passed_bar,
    )  # fmt: skip


def _fixed(hotkey: str, loss: float) -> MinerResultsText:
    return MinerResultsText(
        hotkey=hotkey, test_loss=loss, synth_loss=loss, is_finetune=True, task_type=FIXED, adjusted_loss=loss
    )


class Tournament:
    """A quant tournament after its boss round: outcomes[i] is how task i went (rounds 1-3, then the boss round), or None
    for a boss-round task that was never attached."""

    def __init__(self, monkeypatch, outcomes: list[str | None]):
        self.mp = monkeypatch
        self.results: dict[str, list] = {}
        self.vectors: dict[tuple[str, str], list[float]] = {}
        self.statuses: dict[str, dict] = {}
        self.kinds: dict[str, TaskType] = {}
        self.round_tasks: dict[str, list[TournamentTask]] = {}
        self.seed_roles: dict[str, SeedRole] = {}
        self.boss_side_failures: set[str] = set()
        for index, (task_id, kind, outcome) in enumerate(zip(TASK_IDS, KINDS, outcomes)):
            if outcome is None:
                continue
            round_id = f"r{index + 1}" if index < 3 else "boss"
            registered = TournamentTask(tournament_id=TID, round_id=round_id, task_id=task_id)
            self.round_tasks.setdefault(round_id, []).append(registered)
            self.kinds[task_id] = kind
            if index >= 3:
                self.seed_roles[task_id] = BOSS_ROUND_ROLES[index - 3]
            self._script(task_id, kind, outcome)

    def _script(self, task_id: str, kind: TaskType, outcome: str) -> None:
        self.statuses[task_id] = {BOSS: SUCCESS, CHALLENGER: SUCCESS}
        if outcome in (BOSS_FAILED, BOSS_FAILED_NO_VECTORS):
            self.statuses[task_id] = {BOSS: FAILURE, CHALLENGER: SUCCESS}
        if outcome == NOBODY:  # e.g. a seed that failed prep: nobody has a result on it
            self.results[task_id] = []
            return
        if kind == FREEFORM:
            challenger_bytes = {WIN: 96_000, SMALLER: 99_000, LARGER: 100_000, BEYOND: 101_001}.get(outcome, 96_000)
            self.results[task_id] = [_freeform(BOSS, 100_000), _freeform(CHALLENGER, challenger_bytes)]
            return
        challenger_loss = {
            WIN: 0.9, SMALLER: 0.995, LARGER: 1.0, BEYOND: 1.1, NO_VECTORS: 0.5, NO_VECTORS_LARGER: 1.0, NO_VECTORS_BEYOND: 1.1,
        }.get(outcome, 0.9)  # fmt: skip
        self.results[task_id] = [_fixed(BOSS, 1.0), _fixed(CHALLENGER, challenger_loss)]
        if outcome in (BOSS_FAILED, BOSS_FAILED_NO_VECTORS):
            self.results[task_id] = [_fixed(BOSS, float("inf")), _fixed(CHALLENGER, challenger_loss)]
        if outcome not in NO_VECTOR_OUTCOMES:
            self.vectors[(task_id, BOSS)] = [1.0] * ROWS
            self.vectors[(task_id, CHALLENGER)] = [challenger_loss] * ROWS

    async def run(self) -> str:
        mp = self.mp
        rounds = [
            TournamentRoundData(round_id=f"r{n}", tournament_id=TID, round_number=n, round_type=RoundType.GROUP)
            for n in (1, 2, 3)
        ]
        final = TournamentRoundData(
            round_id="boss", tournament_id=TID, round_number=4, round_type=RoundType.KNOCKOUT, is_final_round=True
        )

        async def task_of(task_id, _db):
            return SimpleNamespace(
                task_type=self.kinds[str(task_id)], ds="d", training_start_point=None, status=TaskStatus.SUCCESS
            )

        async def results(task_id, _db):
            return self.results[str(task_id)]

        async def losses(task_id, hotkey, _db):
            vector = self.vectors.get((str(task_id), hotkey))
            return (vector, "fingerprint") if vector is not None else (None, None)

        async def aligned(task_type, task_id, anchor, other, other_losses, _db, start_point=None):
            return other_losses

        async def seed_of(task_id, _db):
            role = self.seed_roles.get(str(task_id))
            return SimpleNamespace(role=role) if role else None

        mp.setattr(
            round_results,
            "get_tournament_tasks",
            AsyncMock(side_effect=lambda round_id, _db: self.round_tasks.get(round_id, [])),
        )
        mp.setattr(round_results, "get_tournament_rounds", AsyncMock(return_value=rounds + [final]))
        tournament = TournamentData(tournament_id=TID, tournament_type=TournamentType.QUANT)
        mp.setattr(round_results, "get_tournament", AsyncMock(return_value=tournament))
        pair = SimpleNamespace(hotkey1=CHALLENGER, hotkey2=BOSS)
        mp.setattr(round_results, "get_tournament_pairs", AsyncMock(return_value=[pair]))
        mp.setattr(round_results, "count_hotkeys_assigned_to_task", AsyncMock(return_value=2))
        mp.setattr(round_results, "get_task", task_of)
        mp.setattr(
            round_results,
            "boss_side_seed_failure",
            AsyncMock(
                side_effect=lambda task_id, _status, _db: (
                    self.seed_roles[task_id] if task_id in self.boss_side_failures else None
                )
            ),
        )
        mp.setattr(round_results, "get_seed_task_by_task_id", seed_of)
        mp.setattr(round_results, "get_task_results_for_ranking", results)
        mp.setattr(round_results, "calculate_miner_ranking_and_scores", lambda ranked: ranked)
        mp.setattr(round_results, "get_per_example_losses", losses)
        mp.setattr(round_results, "sku_aligned_losses", aligned)
        self.persisted = AsyncMock()
        mp.setattr(round_results, "update_threshold_adjusted_quality_scores_for_task", self.persisted)
        mp.setattr(
            round_results,
            "get_training_status_for_task_and_hotkeys",
            AsyncMock(side_effect=lambda task_id, hotkeys, _db: self.statuses[str(task_id)]),
        )
        outcome = await round_results.get_round_winners(final, psql_db=None, config=None)
        assert outcome.deferred_reason is None and outcome.draw_resolution is None
        return outcome.winners[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcomes, crowned",
    [
        ([WIN] * 7, True),
        # Six wins: the seventh within the larger allowance (freeform equal bytes, fixed an equal loss).
        ([WIN] * 6 + [SMALLER], True),
        ([WIN] * 5 + [LARGER, WIN], True),
        ([WIN] * 6 + [BEYOND], False),
        ([LARGER] + [WIN] * 6, True),
        ([BEYOND] + [WIN] * 6, False),
        # Five wins: one within the smaller allowance and one within the larger.
        ([SMALLER, WIN, LARGER] + [WIN] * 4, True),
        ([LARGER, WIN, LARGER] + [WIN] * 4, False),
        ([WIN, SMALLER, WIN, LARGER] + [WIN] * 3, True),  # the fixed draw is the smaller one
        ([WIN] * 4 + [SMALLER, WIN, SMALLER], True),
        ([WIN] * 4 + [SMALLER, SMALLER, SMALLER], False),
        # A fixed task worse than the boss is beyond, vectors or not; missing vectors are banded on the scalar loss.
        ([WIN] * 4 + [BEYOND, WIN, WIN], False),
        ([WIN] * 4 + [NO_VECTORS_BEYOND, WIN, WIN], False),
        ([WIN] * 4 + [NO_VECTORS, WIN, WIN], True),
        ([WIN] * 4 + [NO_VECTORS_LARGER, WIN, WIN], True),
        ([WIN] * 4 + [NO_VECTORS_LARGER, WIN, LARGER], False),
        ([WIN] * 4 + [NO_VECTORS, WIN, LARGER], True),
        # The boss's fixed entry is a scored fail: the challenger takes it, with or without vectors.
        ([WIN] * 4 + [BOSS_FAILED_NO_VECTORS, WIN, BOSS_FAILED], True),
        # A boss-round task never attached is undecided: no crown.
        ([WIN] * 6 + [None], False),
    ],
)
async def test_the_seven_head_to_heads_through_get_round_winners(monkeypatch, outcomes, crowned):
    winner = await Tournament(monkeypatch, outcomes).run()
    assert winner == (CHALLENGER if crowned else BOSS)


@pytest.mark.asyncio
async def test_only_the_boss_round_tasks_are_persisted(monkeypatch):
    tournament = Tournament(monkeypatch, [WIN] * 6 + [BEYOND])
    await tournament.run()

    persisted = [call.kwargs["task_id"] for call in tournament.persisted.call_args_list]
    assert persisted == TASK_IDS[3:]
    assert tournament.persisted.call_args_list[-1].kwargs["winner_hotkey"] == BOSS


@pytest.mark.asyncio
async def test_a_boss_side_seed_failure_is_the_challengers(monkeypatch):
    tournament = Tournament(monkeypatch, [WIN] * 6 + [BEYOND])
    tournament.boss_side_failures.add(TASK_IDS[6])

    assert await tournament.run() == CHALLENGER
    persisted = tournament.persisted.call_args_list[-1].kwargs
    assert persisted["winner_hotkey"] == CHALLENGER and persisted["compared_hotkeys"] == [BOSS, CHALLENGER]
    assert "boss-side seed failure" in persisted["basis"]


@pytest.mark.asyncio
async def test_a_task_the_round_plays_for_a_failed_boss_round_seed_is_won_by_a_valid_challenger_entry(monkeypatch):
    tournament = Tournament(monkeypatch, [WIN] * 7)
    own_task = TASK_IDS[5]  # a freeform slot: the boss was never assigned, so it has no row and no result
    tournament.seed_roles.pop(own_task)
    tournament.statuses[own_task] = {CHALLENGER: SUCCESS}
    tournament.results[own_task] = [_freeform(CHALLENGER, 150_000)]

    assert await tournament.run() == CHALLENGER


def _extra_boss_round_task(tournament: "Tournament", index: int) -> None:
    task_id = f"21111111-2222-3333-4444-{index:012d}"
    tournament.round_tasks["boss"].append(TournamentTask(tournament_id=TID, round_id="boss", task_id=task_id))
    tournament.kinds[task_id] = FREEFORM
    tournament.seed_roles[task_id] = SeedRole.BOSS_R3_FREEFORM
    tournament._script(task_id, FREEFORM, WIN)


@pytest.mark.asyncio
async def test_a_boss_round_of_more_than_four_tasks_never_crowns(monkeypatch):
    tournament = Tournament(monkeypatch, [WIN] * 7)
    _extra_boss_round_task(tournament, 1)
    _extra_boss_round_task(tournament, 2)

    assert await tournament.run() == BOSS


@pytest.mark.asyncio
async def test_seven_tasks_of_another_shape_never_crown(monkeypatch):
    """Round 3 holds no task and the boss round five: seven decided, but not rounds 1-3 and the four boss-round tasks."""
    tournament = Tournament(monkeypatch, [WIN] * 7)
    tournament.round_tasks["r3"] = []
    _extra_boss_round_task(tournament, 1)

    assert await tournament.run() == BOSS


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
