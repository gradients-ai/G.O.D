"""The per-miner boss comparisons of a round task read the database a fixed number of times, however many miners it has."""

from unittest.mock import AsyncMock
from unittest.mock import patch
from uuid import uuid4

import pytest

from core.models.task_models import TaskType
from validator.db.sql.submissions_and_scoring import StoredLosses
from validator.endpoints import tournament_analytics
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.scoring.models import MinerResultsText
from validator.tournament import round_results
from validator.tournament import task_reads
from validator.tournament.crown import HeadToHead
from validator.tournament.crown import HeadToHeadOutcome
from validator.tournament.models import BossComparison
from validator.tournament.models import RoundStatus
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType


TASK_ID = str(uuid4())
EXAMPLES = 40


def tournament(tournament_type: TournamentType = TournamentType.TEXT) -> TournamentData:
    return TournamentData(
        tournament_id="t1", tournament_type=tournament_type, status=TournamentStatus.ACTIVE, base_winner_hotkey=None
    )


def hotkeys(miners: int) -> list[str]:
    return [f"5Miner{i:02d}" for i in range(miners)]


class Reads:
    """The four readers of a task, counting how often each is asked."""

    def __init__(self, miners: int):
        everyone = [EMISSION_BURN_HOTKEY, *hotkeys(miners)]
        self.results = [
            MinerResultsText(
                hotkey=hk, test_loss=1.0 - 0.001 * i, synth_loss=1.0, is_finetune=True, task_type=TaskType.INSTRUCTTEXTTASK
            )
            for i, hk in enumerate(everyone)
        ]
        self.losses = {hk: StoredLosses(losses=[1.0 - 0.001 * i] * EXAMPLES, fingerprint="f") for i, hk in enumerate(everyone)}
        self.mocks = {
            "get_task_results_for_ranking": AsyncMock(return_value=self.results),
            "get_training_status_for_task": AsyncMock(return_value={}),
            "get_per_example_losses_for_task": AsyncMock(return_value=self.losses),
            "get_task_node_eval_gpus": AsyncMock(return_value={}),
            "get_quant_sku_references": AsyncMock(return_value={}),
        }

    def __enter__(self):
        self.patches = [patch.object(task_reads, name, mock) for name, mock in self.mocks.items()]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()

    def calls(self) -> int:
        return sum(mock.await_count for mock in self.mocks.values())


def task_object() -> object:
    return type("Task", (), {"task_type": TaskType.INSTRUCTTEXTTASK})()


@pytest.mark.asyncio
async def test_database_reads_do_not_grow_with_the_number_of_miners():
    counts = {}
    for miners in (2, 12):
        db = AsyncMock()
        with Reads(miners) as reads:
            compared = await round_results.round_task_boss_comparisons(
                tournament(), TASK_ID, task_object(), hotkeys(miners), db
            )
        counts[miners] = reads.calls()
        assert len(compared) == miners
        # Every reader at most once, and nothing else touches the connection.
        assert all(mock.await_count <= 1 for mock in reads.mocks.values())
        assert db.connection.await_count == 0
    assert counts[2] == counts[12]
    # The ranked results, the statuses and the loss vectors: one read each.
    assert counts[12] == 3


@pytest.mark.asyncio
async def test_the_comparison_carries_the_paired_detail_of_each_miner():
    with Reads(3):
        compared = await round_results.round_task_boss_comparisons(tournament(), TASK_ID, task_object(), hotkeys(3), AsyncMock())
    assert all(c.paired is not None and c.paired.n_examples == EXAMPLES for c in compared)
    assert all(c.outcome in ("win", "within_smaller", "within_larger", "beyond") for c in compared)


@pytest.mark.asyncio
async def test_a_compression_task_reads_the_gpus_and_references_once_and_runs_in_memory():
    quant = AsyncMock(
        side_effect=lambda task_id, label, task_object, hotkey, db, reads: HeadToHead(
            task_id=task_id, label=label, task_type=TaskType.QUANTFIXEDTASK, outcome=HeadToHeadOutcome.WIN
        )
    )
    quant_task = type("Task", (), {"task_type": TaskType.QUANTFIXEDTASK})()
    with Reads(4) as reads, patch.object(round_results, "quant_task_head_to_head", quant):
        compared = await round_results.round_task_boss_comparisons(
            tournament(TournamentType.QUANT), TASK_ID, quant_task, hotkeys(4), AsyncMock()
        )
    assert [c.outcome for c in compared] == ["win"] * 4
    assert reads.mocks["get_task_node_eval_gpus"].await_count == 1
    assert reads.mocks["get_quant_sku_references"].await_count == 1
    assert all(call.args[-1] is not None for call in quant.await_args_list)


@pytest.mark.asyncio
async def test_a_miner_that_cannot_be_decided_is_left_out_and_one_failure_does_not_drop_the_round():
    async def head_to_head(tournament_type, task_id, label, task_type, hotkey, db, reads):
        if hotkey == "5Miner00":
            raise round_results.CrownEvaluationFault([task_id])
        if hotkey == "5Miner01":
            raise RuntimeError("boom")
        return HeadToHead(task_id=task_id, label=label, task_type=task_type, outcome=HeadToHeadOutcome.BEYOND)

    with Reads(3), patch.object(round_results, "_task_head_to_head", head_to_head):
        compared = await round_results.round_task_boss_comparisons(tournament(), TASK_ID, task_object(), hotkeys(3), AsyncMock())
    assert compared == [BossComparison(hotkey="5Miner02", outcome="beyond")]


class Round:
    def __init__(self, status: RoundStatus, number: int = 1, final: bool = False):
        self.status, self.round_number, self.is_final_round, self.round_id = status, number, final, "r"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "round_data",
    [Round(RoundStatus.ACTIVE), Round(RoundStatus.PENDING), Round(RoundStatus.COMPLETED, final=True), Round(RoundStatus.COMPLETED, number=9)],
)
async def test_an_unfinished_final_or_unplanned_round_has_no_comparisons(round_data):
    compute = AsyncMock(return_value=[BossComparison(hotkey="5A", outcome="win")])
    tasks = [type("T", (), {"task_id": TASK_ID})()]
    with patch.object(tournament_analytics, "round_task_boss_comparisons", compute), patch.object(
        tournament_analytics, "plays_round_plan", AsyncMock(return_value=True)
    ):
        got = await tournament_analytics._get_boss_comparisons(tournament(), round_data, tasks, object(), [], AsyncMock())
    assert got is None
    compute.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_finished_round_is_computed_once_and_then_served_from_the_caches():
    tournament_analytics._boss_comparisons_cache.clear()
    stored = {}
    redis = AsyncMock()
    redis.get = AsyncMock(side_effect=lambda k: stored.get(k))
    redis.set = AsyncMock(side_effect=lambda k, v, ex=None: stored.__setitem__(k, v))
    compute = AsyncMock(return_value=[BossComparison(hotkey="5A", outcome="win")])
    tasks = [type("T", (), {"task_id": TASK_ID})()]
    scores = [{"hotkey": "5A"}]
    args = (tournament(), Round(RoundStatus.COMPLETED), tasks, object(), scores, AsyncMock(), redis)
    with patch.object(tournament_analytics, "round_task_boss_comparisons", compute), patch.object(
        tournament_analytics, "plays_round_plan", AsyncMock(return_value=True)
    ):
        first = await tournament_analytics._get_boss_comparisons(*args)
        second = await tournament_analytics._get_boss_comparisons(*args)
        # Another process: nothing in memory, the stored answer in Redis.
        tournament_analytics._boss_comparisons_cache.clear()
        third = await tournament_analytics._get_boss_comparisons(*args)
    assert first == second == third == [BossComparison(hotkey="5A", outcome="win")]
    assert compute.await_count == 1
