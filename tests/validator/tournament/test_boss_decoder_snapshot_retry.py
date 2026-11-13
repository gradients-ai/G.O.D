"""Quant boss decoder snapshot: never leaves a tournament without its boss row, retries transient failures only,
and keeps repairing while the tournament is pending."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament import tournament_manager as tm
from validator.tournament.decoder_snapshot import DecoderMissing
from validator.tournament.decoder_snapshot import DecoderSnapshot
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentType


COMMIT = "c" * 40
REPO = "https://github.com/boss/repo"
TID = "tourn_boss"


def _snapshot() -> DecoderSnapshot:
    return DecoderSnapshot(
        repo_url=REPO, commit_hash=COMMIT, tar_sha256="t" * 64, xz_bytes=10, xz_sha256="x" * 64,
        xz_version="xz 5.4", tree_sha="a" * 40, object_name="quant-decoders/obj.tar.xz",
    )


def _boss(snapshot: bool = False, tournament_id: str = TID) -> TournamentParticipant:
    boss = TournamentParticipant(
        tournament_id=tournament_id, hotkey=EMISSION_BURN_HOTKEY, training_repo=REPO, training_commit_hash=COMMIT
    )
    return tm._with_decoder_snapshot(boss, _snapshot() if snapshot else None)


@pytest.fixture(autouse=True)
def _clean_state():
    def clear():
        tm._BOSS_DECODER_DETERMINISTIC.clear()
        tm._BOSS_DECODER_NEXT_REPAIR_AT.clear()
        tm._BOSS_DECODER_ALERTED.clear()

    clear()
    yield
    clear()


@pytest.fixture
def snap(monkeypatch):
    mock = AsyncMock(return_value=_snapshot())
    sleep = AsyncMock()
    monkeypatch.setattr(tm, "snapshot_decoder", mock)
    monkeypatch.setattr(tm.asyncio, "sleep", sleep)
    monkeypatch.setattr(tm.service_cst, "BUCKET_NAME", "bucket")
    mock.sleep = sleep
    return mock


@pytest.fixture
def basic(monkeypatch):
    add = AsyncMock()
    monkeypatch.setattr(tm, "get_base_contestant", AsyncMock(return_value=_boss(tournament_id="")))
    monkeypatch.setattr(tm, "get_latest_tournament_winner_participant", AsyncMock(return_value=None))
    monkeypatch.setattr(tm, "create_tournament", AsyncMock())
    monkeypatch.setattr(tm, "add_tournament_participants", add)
    return add


def _config() -> SimpleNamespace:
    return SimpleNamespace(psql_db=None, discord_url="https://discord.example/hook")


@pytest.mark.parametrize("error", [OSError("scratch dir"), RuntimeError("clone timed out"), ValueError("tar")])
async def test_unexpected_snapshot_errors_never_propagate_and_the_boss_is_still_added(snap, basic, error):
    snap.side_effect = error

    await tm.create_basic_tournament(TournamentType.QUANT, psql_db=None, config=_config())

    boss = basic.await_args.args[0][0]
    assert boss.hotkey == EMISSION_BURN_HOTKEY and not boss.has_decoder_snapshot


async def test_transient_failure_is_retried_in_place_and_succeeds(snap, basic):
    snap.side_effect = [RuntimeError("clone timed out"), _snapshot()]

    await tm.create_basic_tournament(TournamentType.QUANT, psql_db=None, config=_config())

    assert snap.await_count == 2
    snap.sleep.assert_awaited_once_with(tm.t_cst.BOSS_DECODER_SNAPSHOT_RETRY_SLEEP_SECONDS)
    assert basic.await_args.args[0][0].has_decoder_snapshot


async def test_transient_failure_is_bounded(snap, basic):
    snap.side_effect = OSError("minio down")

    await tm.create_basic_tournament(TournamentType.QUANT, psql_db=None, config=_config())

    assert snap.await_count == tm.t_cst.BOSS_DECODER_SNAPSHOT_ATTEMPTS


async def test_deterministic_failure_is_not_retried_and_is_remembered(snap, basic):
    snap.side_effect = DecoderMissing("No decoder/ directory")

    await tm.create_basic_tournament(TournamentType.QUANT, psql_db=None, config=_config())

    assert snap.await_count == 1
    snap.sleep.assert_not_awaited()
    assert basic.await_args.args[0][0].hotkey == EMISSION_BURN_HOTKEY
    assert len(tm._BOSS_DECODER_DETERMINISTIC) == 1


async def test_boss_without_pinned_commit_is_deterministic(snap):
    boss = TournamentParticipant(tournament_id=TID, hotkey=EMISSION_BURN_HOTKEY, training_repo=REPO, training_commit_hash=None)

    result = await tm._with_boss_decoder_snapshot(boss)

    assert result == boss
    snap.assert_not_awaited()
    assert TID in tm._BOSS_DECODER_DETERMINISTIC


async def test_text_and_image_never_snapshot_the_boss(snap, basic):
    for tournament_type in (TournamentType.TEXT, TournamentType.IMAGE):
        await tm.create_basic_tournament(tournament_type, psql_db=None, config=_config())

    snap.assert_not_awaited()


@pytest.fixture
def repair(monkeypatch):
    state = SimpleNamespace(
        participants=[], add=AsyncMock(), set_snapshot=AsyncMock(), discord=AsyncMock(), now=[1000.0],
        base=AsyncMock(return_value=_boss(tournament_id="")),
    )
    monkeypatch.setattr(tm, "get_tournament_participants", AsyncMock(side_effect=lambda *_: list(state.participants)))
    monkeypatch.setattr(tm, "get_base_contestant", state.base)
    monkeypatch.setattr(tm, "add_tournament_participants", state.add)
    monkeypatch.setattr(tm, "set_tournament_participant_decoder_snapshot", state.set_snapshot)
    monkeypatch.setattr(tm, "send_to_discord", state.discord)
    monkeypatch.setattr(tm.time, "monotonic", lambda: state.now[0])
    return state


async def test_pending_quant_without_a_boss_row_gets_one(snap, repair):
    await tm._repair_quant_boss_participant(TID, _config())

    added = repair.add.await_args.args[0][0]
    assert (added.tournament_id, added.hotkey) == (TID, EMISSION_BURN_HOTKEY)
    assert added.has_decoder_snapshot


async def test_pending_repair_persists_a_recovered_snapshot_then_stops(snap, repair):
    repair.participants = [_boss()]

    await tm._repair_quant_boss_participant(TID, _config())

    repair.set_snapshot.assert_awaited_once()
    assert repair.set_snapshot.await_args.args[0].has_decoder_snapshot
    repair.add.assert_not_awaited()

    repair.participants = [_boss(snapshot=True)]
    snap.reset_mock()
    await tm._repair_quant_boss_participant(TID, _config())
    snap.assert_not_awaited()


async def test_repair_is_throttled_by_the_cooldown_and_alerts_once(snap, repair):
    snap.side_effect = OSError("minio down")
    repair.participants = [_boss()]

    await tm._repair_quant_boss_participant(TID, _config())
    await tm._repair_quant_boss_participant(TID, _config())
    assert snap.await_count == 1  # second call is inside the cooldown
    repair.now[0] += tm.t_cst.BOSS_DECODER_REPAIR_COOLDOWN_SECONDS + 1
    await tm._repair_quant_boss_participant(TID, _config())

    assert snap.await_count == 2
    assert repair.discord.await_count == 1
    assert "transient" in repair.discord.await_args.kwargs["message"]
    repair.set_snapshot.assert_not_awaited()


async def test_deterministic_boss_is_not_retried_by_the_repair_and_alerts_once(snap, repair):
    snap.side_effect = DecoderMissing("No decoder/ directory")
    repair.participants = [_boss()]

    for _ in range(3):
        await tm._repair_quant_boss_participant(TID, _config())
        repair.now[0] += tm.t_cst.BOSS_DECODER_REPAIR_COOLDOWN_SECONDS + 1

    assert snap.await_count == 1
    assert repair.discord.await_count == 1
    assert "Not retrying" in repair.discord.await_args.kwargs["message"]


async def test_repair_never_raises(snap, repair, monkeypatch):
    monkeypatch.setattr(tm, "get_tournament_participants", AsyncMock(side_effect=RuntimeError("db down")))

    await tm._repair_quant_boss_participant(TID, _config())
