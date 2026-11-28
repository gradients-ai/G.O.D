"""Contenders for the emission balancer: challengers ranked strictly above the boss in round 1."""

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

import validator.scoring.emission_balance as eb
from core.models.tournament_models import TournamentType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import RoundRanks
from validator.tournament.models import TournamentRoundData
from validator.tournament.round_results import count_round_one_above_boss


BOSS = EMISSION_BURN_HOTKEY


@pytest.fixture(autouse=True)
def _empty_cache():
    eb._CONTENDERS_BY_TOURNAMENT.clear()
    yield
    eb._CONTENDERS_BY_TOURNAMENT.clear()


def _round(number: int, final: bool = False) -> TournamentRoundData:
    return TournamentRoundData(
        round_id=f"r{number}", tournament_id="t", round_number=number, round_type="group", is_final_round=final
    )


async def _above_boss(ranks: dict[str, int] | None, rounds: list[TournamentRoundData] | None = None) -> int | None:
    rounds = [_round(1), _round(2)] if rounds is None else rounds
    ranked = RoundRanks(round_number=1, round_id="r1", task_ids=[], ranks=ranks or {})
    with (
        patch("validator.tournament.round_results.get_tournament_rounds", AsyncMock(return_value=rounds)),
        patch("validator.tournament.round_results._round_ranks_with_boss", AsyncMock(return_value=ranked)) as ranker,
    ):
        result = await count_round_one_above_boss("t", None)
    if rounds and any(r.round_number == 1 for r in rounds):
        assert ranker.await_args.args[0].round_number == 1
    return result


async def test_only_challengers_strictly_above_the_boss_count():
    assert await _above_boss({"a": 1, "b": 2, BOSS: 3, "c": 3, "d": 5}) == 2


async def test_a_boss_with_no_round_one_result_ranks_last_so_every_valid_entry_counts():
    # The boss shares the last rank with the entries that have no valid result.
    assert await _above_boss({"a": 1, "b": 2, "c": 3, "failed": 4, BOSS: 4}) == 3


async def test_a_boss_on_top_leaves_no_contenders():
    assert await _above_boss({BOSS: 1, "a": 2, "b": 3}) == 0


async def test_no_round_one_is_none():
    assert await _above_boss({"a": 1, BOSS: 2}, rounds=[_round(2)]) is None


async def _counts(per_tournament: dict[str, int | None | Exception], completers: int = 0) -> list[int]:
    async def above_boss(tournament_id, psql_db):
        value = per_tournament[tournament_id]
        if isinstance(value, Exception):
            raise value
        return value

    with (
        patch.object(eb, "get_recent_completed_tournament_ids", AsyncMock(return_value=list(per_tournament))),
        patch.object(eb, "count_round_one_above_boss", above_boss),
        patch.object(eb, "count_round_one_completers", AsyncMock(return_value=completers)),
    ):
        return await eb.get_recent_contender_counts(None, TournamentType.TEXT)


async def test_the_window_is_the_last_four_with_enough_contenders():
    assert await _counts({"t6": 9, "t5": 2, "t4": 8, "t3": 7, "t2": 0, "t1": 6, "t0": 5}) == [9, 8, 7, 6]


async def test_an_unavailable_ranking_falls_back_to_round_one_completers():
    assert await _counts({"t1": RuntimeError("no standings"), "t0": None}, completers=4) == [4, 4]


async def test_a_computed_count_is_cached_and_a_fallback_is_not():
    calls = AsyncMock(return_value=6)
    with (
        patch.object(eb, "count_round_one_above_boss", calls),
        patch.object(eb, "count_round_one_completers", AsyncMock(return_value=3)),
    ):
        assert await eb.tournament_contenders(None, "t") == 6
        assert await eb.tournament_contenders(None, "t") == 6
    assert calls.await_count == 1

    failing = AsyncMock(side_effect=RuntimeError("down"))
    with (
        patch.object(eb, "count_round_one_above_boss", failing),
        patch.object(eb, "count_round_one_completers", AsyncMock(return_value=3)),
    ):
        assert await eb.tournament_contenders(None, "u") == 3
        assert await eb.tournament_contenders(None, "u") == 3
    assert failing.await_count == 2
