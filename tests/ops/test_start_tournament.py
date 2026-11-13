from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from ops.tools.tournament import start_tournament
from validator.tournament.models import TournamentType


STARTABLE_TYPES = [t for t in TournamentType if t != TournamentType.ENVIRONMENT]


@pytest.mark.parametrize("tournament_type", STARTABLE_TYPES)
def test_parses_every_tournament_type(tournament_type):
    assert start_tournament.parse_args([tournament_type.value]).tournament_type is tournament_type


@pytest.mark.parametrize("argv", [[], ["bogus"], ["environment"]])
def test_requires_a_known_tournament_type(argv):
    with pytest.raises(SystemExit):
        start_tournament.parse_args(argv)


@pytest.mark.parametrize("tournament_type", STARTABLE_TYPES)
async def test_creates_a_tournament_of_the_requested_type(monkeypatch, tournament_type):
    config = MagicMock()
    config.psql_db.connect = AsyncMock()
    config.psql_db.close = AsyncMock()
    monkeypatch.setattr(start_tournament, "load_config", MagicMock(return_value=config))
    create = AsyncMock(return_value="tourn_x")
    monkeypatch.setattr(start_tournament, "create_basic_tournament", create)

    await start_tournament.main(tournament_type)

    create.assert_awaited_once_with(tournament_type, config.psql_db, config)
    config.psql_db.close.assert_awaited_once()
