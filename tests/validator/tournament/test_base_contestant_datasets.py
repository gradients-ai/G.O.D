from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from core.models.tournament_models import TournamentType
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.participants import get_base_contestant


MODULE = "validator.tournament.participants"
ENVIRONMENT = TournamentType.ENVIRONMENT


def _winner(**kwargs):
    return SimpleNamespace(
        tournament_id=kwargs.get("tournament_id", "tourn_prev"),
        hotkey=kwargs.get("hotkey", "5WinnerHotkey"),
        training_repo=kwargs.get("training_repo"),
        backup_repo=kwargs.get("backup_repo"),
        training_commit_hash=kwargs.get("training_commit_hash"),
        requested_datasets=kwargs.get("requested_datasets"),
    )


class TestGetBaseContestant:
    @pytest.mark.asyncio
    async def test_propagates_previous_winner_datasets(self):
        latest_winner = _winner(
            backup_repo="https://github.com/gradients-opensource/god-environment-position-1.git",
            training_commit_hash="abc123",
            requested_datasets=[
                "gradients-io-tournaments/pvp-tool-calling-sft",
                "gradients-io-tournaments/SWE-ZERO-12M-trajectories-filtered",
            ],
        )

        with (
            patch(f"{MODULE}.get_latest_tournament_winner_participant", new_callable=AsyncMock) as winner_lookup,
            patch(f"{MODULE}.get_latest_commit_hash_from_github", new_callable=AsyncMock) as commit_lookup,
        ):
            winner_lookup.return_value = latest_winner
            commit_lookup.return_value = "deadbeef"

            base = await get_base_contestant(AsyncMock(), ENVIRONMENT, AsyncMock())

        assert base is not None
        assert base.hotkey == EMISSION_BURN_HOTKEY
        assert base.training_repo == latest_winner.backup_repo
        assert base.requested_datasets == [
            "gradients-io-tournaments/pvp-tool-calling-sft",
            "gradients-io-tournaments/SWE-ZERO-12M-trajectories-filtered",
        ]

    @pytest.mark.asyncio
    async def test_no_datasets_when_previous_winner_has_none(self):
        latest_winner = _winner(
            backup_repo="https://github.com/gradients-opensource/god-environment-position-1.git",
            training_commit_hash="abc123",
        )

        with (
            patch(f"{MODULE}.get_latest_tournament_winner_participant", new_callable=AsyncMock) as winner_lookup,
            patch(f"{MODULE}.get_latest_commit_hash_from_github", new_callable=AsyncMock) as commit_lookup,
        ):
            winner_lookup.return_value = latest_winner
            commit_lookup.return_value = "deadbeef"

            base = await get_base_contestant(AsyncMock(), ENVIRONMENT, AsyncMock())

        assert base is not None
        assert base.requested_datasets is None
