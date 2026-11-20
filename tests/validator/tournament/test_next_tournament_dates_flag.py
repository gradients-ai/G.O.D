"""get_next_tournament_dates always advertises a start date for text, image and quant."""

from datetime import datetime
from datetime import timezone
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

import validator.endpoints.tournament_analytics as analytics


@pytest.fixture
def config(monkeypatch):
    sql = MagicMock()
    sql.get_active_tournament = AsyncMock(return_value=None)
    sql.get_tournaments_with_status = AsyncMock(return_value=[])
    sql.get_latest_tournament_with_created_at = AsyncMock(return_value=(None, datetime.now(timezone.utc)))
    monkeypatch.setattr(analytics, "tournament_sql", sql)
    return MagicMock()


@pytest.mark.asyncio
async def test_all_three_types_are_advertised(config):
    result = await analytics.get_next_tournament_dates(config)

    assert result.quant is not None and result.quant.tournament_type.value == "quant"
    assert result.quant.next_start_date is not None
    assert result.text is not None and result.text.next_start_date is not None
    assert result.image is not None and result.image.next_start_date is not None
