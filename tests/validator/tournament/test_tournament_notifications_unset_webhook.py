"""Tournament start/completion announcements are a quiet no-op when no Discord webhook is configured."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from validator.tournament import notifications


@pytest.mark.parametrize("webhook", [None, ""])
async def test_no_webhook_sends_nothing_and_logs_no_error(monkeypatch, webhook):
    send = AsyncMock()
    error = MagicMock()
    monkeypatch.setattr(notifications, "send_to_discord", send)
    monkeypatch.setattr(notifications.logger, "error", error)

    await notifications.notify_tournament_started("tourn_1", "text", 12, webhook)
    await notifications.notify_tournament_completed("tourn_1", "text", "hk_1", webhook)

    send.assert_not_awaited()
    error.assert_not_called()


async def test_a_configured_webhook_still_announces_the_start(monkeypatch):
    send = AsyncMock()
    monkeypatch.setattr(notifications, "send_to_discord", send)

    await notifications.notify_tournament_started("tourn_1", "text", 12, "https://discord/webhook")

    send.assert_awaited_once()
    assert "tourn_1" in send.await_args.args[1]
