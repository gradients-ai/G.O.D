"""Regression tests for tournament placement persistence.

Tournament completion computes a 3rd-place hotkey and passes it into
``update_tournament_placements``. That helper must accept the extra argument and
write ``final_position = 3``; omitting it crashes the active-tournament loop.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from validator.db.sql.tournaments import update_tournament_placements


def _mock_psql_db():
    connection = AsyncMock()

    @asynccontextmanager
    async def transaction():
        yield None

    connection.transaction = MagicMock(side_effect=lambda: transaction())

    @asynccontextmanager
    async def acquire():
        yield connection

    psql_db = MagicMock()
    psql_db.connection = AsyncMock(return_value=acquire())
    return psql_db, connection


class TestUpdateTournamentPlacements:
    async def test_accepts_third_place_and_writes_position_three(self):
        psql_db, connection = _mock_psql_db()

        await update_tournament_placements(
            "tourn_test",
            "winner_hk",
            "second_hk",
            "third_hk",
            psql_db,
        )

        assert connection.execute.await_count == 2
        _winner_sql, tournament_id, winner_hotkey = connection.execute.await_args_list[0].args
        assert tournament_id == "tourn_test"
        assert winner_hotkey == "winner_hk"

        placement_sql, _, winner_arg, second_arg, third_arg = connection.execute.await_args_list[1].args
        assert winner_arg == "winner_hk"
        assert second_arg == "second_hk"
        assert third_arg == "third_hk"
        assert "THEN 3" in placement_sql

    async def test_omitted_third_place_still_persists_top_two(self):
        psql_db, connection = _mock_psql_db()

        await update_tournament_placements(
            "tourn_test",
            "winner_hk",
            "second_hk",
            None,
            psql_db,
        )

        placement_sql, _, _, second_arg, third_arg = connection.execute.await_args_list[1].args
        assert second_arg == "second_hk"
        assert third_arg is None
        assert "THEN 3" in placement_sql
