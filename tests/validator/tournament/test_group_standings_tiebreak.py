"""Multi-match group standings: average rank, then summed normalised gap, then a per-round seeded shuffle."""

import pytest

import validator.tournament.round_results as rr
from core.models.task_models import TaskType
from validator.scoring.models import MinerResultsText
from validator.tournament.models import GroupMatchStanding
from validator.tournament.models import TournamentTask


ROUND_ID = "round-1"
TASK_IDS = [f"{i}1111111-2222-3333-4444-555555555555" for i in range(4)]
INF = float("inf")


def _result(hotkey, loss, rank=None, rank_metric=None, task_type=TaskType.INSTRUCTTEXTTASK):
    return MinerResultsText(
        hotkey=hotkey,
        test_loss=loss,
        synth_loss=loss,
        is_finetune=True,
        task_type=task_type,
        adjusted_loss=loss,
        rank=rank,
        rank_metric=rank_metric,
    )


def _tasks(count: int, round_id: str = ROUND_ID) -> list[TournamentTask]:
    return [TournamentTask(tournament_id="t", round_id=round_id, task_id=task_id) for task_id in TASK_IDS[:count]]


@pytest.fixture
def wire(monkeypatch):
    """Serve `results_by_task[i]` for task i, already ranked; return the list every built standing lands in."""
    standings: list[GroupMatchStanding] = []

    def _wire(results_by_task):
        async def fake_results(task_id, psql_db):
            return results_by_task[TASK_IDS.index(task_id)]

        def recording_standing(**fields):
            standing = GroupMatchStanding(**fields)
            standings.append(standing)
            return standing

        monkeypatch.setattr(rr, "get_task_results_for_ranking", fake_results)
        monkeypatch.setattr(rr, "calculate_miner_ranking_and_scores", lambda results: results)
        monkeypatch.setattr(rr, "GroupMatchStanding", recording_standing)
        return standings

    return _wire


def _latin_square(hotkeys):
    """Every hotkey takes every place once, so average rank and summed gap tie for all of them."""
    return [
        [_result(hotkey, float((place + shift) % len(hotkeys) + 1)) for place, hotkey in enumerate(hotkeys)]
        for shift in range(len(hotkeys))
    ]


def _expected_shuffle(hotkeys, round_id):
    index = rr.seeded_order(hotkeys, rr.tiebreak_seed(round_id))
    return sorted(hotkeys, key=index.__getitem__)


@pytest.mark.asyncio
async def test_equal_average_rank_goes_to_the_smaller_gap(wire):
    # A wins task 0 narrowly and loses task 1 badly; B is the reverse. Both average rank 1.5.
    wire([[_result("A", 1.0), _result("B", 1.1)], [_result("B", 1.0), _result("A", 1.5)]])

    ordered = await rr._quant_ordered_hotkeys(_tasks(2), psql_db=None)

    assert ordered == ["B", "A"]


@pytest.mark.asyncio
async def test_gap_is_normalised_per_task(wire):
    # Raw loss sums favour B (2.2 + 100.0 < 2.0 + 105.0); relative to each task's best A is closer.
    standings = wire([[_result("A", 2.0), _result("B", 2.2)], [_result("B", 100.0), _result("A", 105.0)]])

    ordered = await rr._quant_ordered_hotkeys(_tasks(2), psql_db=None)

    by_hotkey = {standing.hotkey: standing for standing in standings}
    assert by_hotkey["A"].summed_gap == pytest.approx(0.05)
    assert by_hotkey["B"].summed_gap == pytest.approx(0.1)
    assert ordered == ["A", "B"]


@pytest.mark.asyncio
async def test_full_tie_follows_the_round_seeded_shuffle(wire):
    hotkeys = ["5A", "5B", "5C", "5D"]
    wire(_latin_square(hotkeys))

    first = await rr._quant_ordered_hotkeys(_tasks(4), psql_db=None)
    second = await rr._quant_ordered_hotkeys(_tasks(4), psql_db=None)

    assert first == second == _expected_shuffle(hotkeys, ROUND_ID)


@pytest.mark.asyncio
async def test_full_tie_order_changes_with_the_round(wire):
    hotkeys = ["5A", "5B", "5C", "5D"]
    wire(_latin_square(hotkeys))

    orders = {tuple(await rr._quant_ordered_hotkeys(_tasks(4, f"round-{i}"), psql_db=None)) for i in range(8)}

    assert len(orders) > 1


@pytest.mark.asyncio
async def test_lexically_last_hotkey_can_win_a_tie(wire):
    hotkeys = ["5A", "5B", "5C", "5Z"]
    wire(_latin_square(hotkeys))

    winners = {(await rr._quant_ordered_hotkeys(_tasks(4, f"round-{i}"), psql_db=None))[0] for i in range(16)}

    assert "5Z" in winners


@pytest.mark.asyncio
async def test_missed_match_contributes_infinite_gap_and_an_error(wire):
    standings = wire([[_result("A", 1.0), _result("B", 2.0)], [_result("B", 1.0)]])

    ordered = await rr._quant_ordered_hotkeys(_tasks(2), psql_db=None)

    by_hotkey = {standing.hotkey: standing for standing in standings}
    assert by_hotkey["A"].has_error and by_hotkey["A"].summed_gap == INF
    assert by_hotkey["B"].summed_gap == pytest.approx(1.0)
    assert ordered == ["B", "A"]


@pytest.mark.asyncio
async def test_rank_from_the_ranking_function_beats_loss_order(wire):
    # Freeform-style: ranked by effective bytes, so rank disagrees with loss and rank_metric is the effective byte count.
    freeform = TaskType.QUANTFREEFORMTASK
    standings = wire(
        [
            [
                _result("A", 3.0, rank=1, rank_metric=100.0, task_type=freeform),
                _result("B", 1.0, rank=2, rank_metric=150.0, task_type=freeform),
                _result("C", 2.0, rank=3, rank_metric=200.0, task_type=freeform),
            ]
        ]
    )

    ordered = await rr._quant_ordered_hotkeys(_tasks(1), psql_db=None)

    by_hotkey = {standing.hotkey: standing for standing in standings}
    assert ordered == ["A", "B", "C"]
    assert by_hotkey["B"].summed_gap == pytest.approx(0.5)
    assert by_hotkey["C"].summed_gap == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_freeform_effective_bytes_feed_the_match_gap(wire):
    freeform = TaskType.QUANTFREEFORMTASK
    standings = wire(
        [
            [
                _result("A", 1.0, rank=1, rank_metric=1000.0, task_type=freeform),
                _result("B", 1.0, rank=2, rank_metric=1030.0, task_type=freeform),
            ]
        ]
    )

    await rr._quant_ordered_hotkeys(_tasks(1), psql_db=None)

    by_hotkey = {standing.hotkey: standing for standing in standings}
    assert by_hotkey["A"].summed_gap == pytest.approx(0.0)
    assert by_hotkey["B"].summed_gap == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_without_ranks_the_order_is_adjusted_loss_ascending(wire):
    wire([[_result("A", 3.0), _result("B", 1.0), _result("C", 2.0)]])

    ordered = await rr._quant_ordered_hotkeys(_tasks(1), psql_db=None)

    assert ordered == ["B", "C", "A"]


@pytest.mark.asyncio
async def test_partial_ranks_fall_back_to_adjusted_loss(wire):
    wire([[_result("A", 3.0, rank=1), _result("B", 1.0), _result("C", 2.0, rank=2)]])

    ordered = await rr._quant_ordered_hotkeys(_tasks(1), psql_db=None)

    assert ordered == ["B", "C", "A"]


def test_scored_failures_contribute_infinite_gap():
    assert rr._match_gaps({"A": 1.0, "B": INF}, is_higher_better=False) == {"A": 0.0, "B": INF}


def test_all_failed_match_gives_every_competitor_infinite_gap():
    assert rr._match_gaps({"A": INF, "B": INF}, is_higher_better=False) == {"A": INF, "B": INF}


def test_perfect_best_uses_absolute_gap():
    assert rr._match_gaps({"A": 0.0, "B": 0.25}, is_higher_better=False) == {"A": 0.0, "B": 0.25}


def test_higher_is_better_gap_is_measured_down_from_the_max():
    gaps = rr._match_gaps({"A": 4.0, "B": 3.0, "C": -1.0}, is_higher_better=True)

    assert gaps == {"A": 0.0, "B": 0.25, "C": 1.25}
    assert all(gap >= 0 for gap in gaps.values())


def test_tiebreak_seed_is_deterministic_and_per_round():
    assert rr.tiebreak_seed("round-1") == rr.tiebreak_seed("round-1")
    assert rr.tiebreak_seed("round-1") != rr.tiebreak_seed("round-2")
    assert 0 <= rr.tiebreak_seed("round-1") < 2**64


def test_seeded_order_ignores_input_order():
    seed = rr.tiebreak_seed(ROUND_ID)

    assert rr.seeded_order(["5C", "5A", "5B"], seed) == rr.seeded_order(["5A", "5B", "5C"], seed)
    assert sorted(rr.seeded_order(["5C", "5A", "5B"], seed).values()) == [0, 1, 2]
