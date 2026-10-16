from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from validator.endpoints.tournament_analytics import get_next_tournament_dates
from validator.tournament import constants as t_cst
from validator.tournament.models import TournamentType
from validator.tournament.tournament_manager import check_and_start_tournament


@pytest.mark.asyncio
async def test_check_and_start_skips_environment_when_paused():
    assert t_cst.ENVIRONMENT_TOURNAMENTS_ENABLED is False

    psql_db = MagicMock()
    config = MagicMock()

    with patch(
        "validator.tournament.tournament_manager.get_active_tournament",
        new_callable=AsyncMock,
    ) as get_active:
        await check_and_start_tournament(TournamentType.ENVIRONMENT, psql_db, config)
        get_active.assert_not_awaited()


@pytest.mark.asyncio
async def test_next_dates_reports_environment_paused_when_idle():
    assert t_cst.ENVIRONMENT_TOURNAMENTS_ENABLED is False

    config = MagicMock()
    config.psql_db = MagicMock()

    with (
        patch(
            "validator.endpoints.tournament_analytics.tournament_sql.get_active_tournament",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "validator.endpoints.tournament_analytics.tournament_sql.get_tournaments_with_status",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "validator.endpoints.tournament_analytics.tournament_sql.get_latest_tournament_with_created_at",
            new_callable=AsyncMock,
            side_effect=[
                (None, None),  # text
                (None, None),  # image
            ],
        ),
        patch(
            "validator.endpoints.tournament_analytics._calculate_next_tournament_start_time",
            side_effect=lambda *_args, **_kwargs: MagicMock(),
        ),
    ):
        response = await get_next_tournament_dates(config)

    assert response.environment.tournament_status == "paused"
    assert response.environment.next_start_date is None
    assert response.text.tournament_status == "waiting"
    assert response.image.tournament_status == "waiting"
