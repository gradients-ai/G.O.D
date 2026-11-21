"""The winner's repository is uploaded only once its tournament is marked completed, never when the final round ends."""

from unittest.mock import AsyncMock

import pytest

import validator.tournament.tournament_manager as tm
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType


TID = "tourn_completed"


def _completed(winner: str | None = EMISSION_BURN_HOTKEY) -> TournamentData:
    return TournamentData(
        tournament_id=TID,
        tournament_type=TournamentType.TEXT,
        status=TournamentStatus.COMPLETED,
        winner_hotkey=winner,
    )


def _winner(backup_repo: str | None = None, training_repo: str | None = "https://github.com/a/b") -> TournamentParticipant:
    return TournamentParticipant(
        tournament_id=TID, hotkey=EMISSION_BURN_HOTKEY, training_repo=training_repo, backup_repo=backup_repo
    )


@pytest.fixture
def seams(monkeypatch):
    tm._winner_upload_attempted_at.clear()
    upload = AsyncMock(return_value="https://github.com/org/backup")
    monkeypatch.setattr(tm, "upload_participant_repository", upload)
    monkeypatch.setattr(tm, "get_latest_completed_tournament", AsyncMock(return_value=_completed()))
    monkeypatch.setattr(tm, "get_tournament_participant", AsyncMock(return_value=_winner()))
    return upload


async def test_a_completed_tournament_uploads_its_winner(seams):
    await tm.upload_completed_tournament_winner(TournamentType.TEXT, None, None)

    seams.assert_awaited_once()
    assert seams.await_args.args[:4] == (TID, TournamentType.TEXT, EMISSION_BURN_HOTKEY, 1)


@pytest.mark.parametrize(
    "tournament, winner",
    [
        pytest.param(None, _winner(), id="no completed tournament"),
        pytest.param(_completed(winner=None), _winner(), id="no winner set"),
        pytest.param(_completed(), _winner(backup_repo="https://github.com/org/backup"), id="already uploaded"),
        pytest.param(_completed(), _winner(training_repo=None), id="no training repo"),
    ],
)
async def test_nothing_to_upload_uploads_nothing(seams, monkeypatch, tournament, winner):
    monkeypatch.setattr(tm, "get_latest_completed_tournament", AsyncMock(return_value=tournament))
    monkeypatch.setattr(tm, "get_tournament_participant", AsyncMock(return_value=winner))

    await tm.upload_completed_tournament_winner(TournamentType.TEXT, None, None)

    seams.assert_not_awaited()


async def test_a_failed_upload_is_not_retried_before_the_interval(seams):
    seams.return_value = None

    await tm.upload_completed_tournament_winner(TournamentType.TEXT, None, None)
    await tm.upload_completed_tournament_winner(TournamentType.TEXT, None, None)

    seams.assert_awaited_once()


async def test_a_failing_lookup_never_raises(seams, monkeypatch):
    monkeypatch.setattr(tm, "get_latest_completed_tournament", AsyncMock(side_effect=RuntimeError("db down")))

    await tm.upload_completed_tournament_winner(TournamentType.TEXT, None, None)

    seams.assert_not_awaited()


async def test_the_scheduler_uploads_before_deciding_on_a_new_tournament(seams, monkeypatch):
    monkeypatch.setattr(tm, "get_active_tournament", AsyncMock(return_value=_completed()))

    await tm.check_and_start_tournament(TournamentType.TEXT, None, None)

    seams.assert_awaited_once()
