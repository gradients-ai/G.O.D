"""A pending tournament short of miners must not block the others, and one raising tournament must not stop the rest.

populate_tournament_participants is a single pass: below the minimum it returns 0 and the 60s pending loop moves on.
The miner ping pass is spaced out per tournament; the persisted-count read is not. process_active_tournaments isolates
each tournament in its own try/except.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fiber.chain.models import Node

from core.models.payload_models import TrainingRepoResponse
from validator.tournament import tournament_manager as tm
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType


COMMIT = "c" * 40


class _StopLoop(BaseException):
    """Raised from the patched sleep to end the endless cycle after one pass (not an Exception, so nothing swallows it)."""


def _tournament(tournament_id: str, tournament_type: TournamentType, status=TournamentStatus.PENDING) -> TournamentData:
    return TournamentData(tournament_id=tournament_id, tournament_type=tournament_type, status=status)


def _node(index: int) -> Node:
    return Node(
        hotkey=f"5Miner{index}", coldkey=f"5Cold{index}", node_id=index, incentive=0.0, netuid=56, alpha_stake=0.0,
        tao_stake=0.0, stake=0.0, vtrust=0.0, last_updated=0.0, ip=f"10.0.0.{index}", ip_type=4, port=8091,
    )


@pytest.fixture(autouse=True)
def _clean_state():
    for state in (tm._PARTICIPANT_NEXT_PASS_AT,):
        state.clear()
    tm._SHORT_TOURNAMENTS_ALERTED.clear()
    yield
    tm._PARTICIPANT_NEXT_PASS_AT.clear()
    tm._SHORT_TOURNAMENTS_ALERTED.clear()


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []

    async def fake_sleep(seconds):
        calls.append(seconds)
        raise _StopLoop

    monkeypatch.setattr(tm.asyncio, "sleep", fake_sleep)
    return calls


@pytest.fixture
def enrolment(monkeypatch):
    """Fake everything one enrolment pass touches. `persisted` holds the hotkeys already enrolled per tournament."""
    state = SimpleNamespace(
        types={},
        persisted={},
        enrollable=set(),
        pings=AsyncMock(),
        discord=AsyncMock(),
    )

    async def get_tournament(tournament_id, _db):
        return _tournament(tournament_id, state.types[tournament_id])

    async def get_participants(tournament_id, _db):
        hotkeys = state.persisted.setdefault(tournament_id, [])
        return [TournamentParticipant(tournament_id=tournament_id, hotkey=h) for h in hotkeys]

    async def ping(node, _config, _type):
        await state.pings(node.hotkey)
        return TrainingRepoResponse(github_repo=f"https://github.com/{node.hotkey}/repo", commit_hash=COMMIT)

    async def enroll(participant, _coldkey, _fee, _db):
        if participant.hotkey in state.enrollable:
            state.persisted[participant.tournament_id].append(participant.hotkey)
            return True
        return False

    monkeypatch.setattr(tm.t_cst, "MIN_MINERS_FOR_TOURN", 2)
    monkeypatch.setattr(tm, "get_tournament", get_tournament)
    monkeypatch.setattr(tm, "get_tournament_participants", get_participants)
    monkeypatch.setattr(tm, "get_all_nodes", AsyncMock(return_value=[_node(1), _node(2)]))
    monkeypatch.setattr(tm, "get_node_by_hotkey", AsyncMock(return_value=None))
    monkeypatch.setattr(tm, "_get_miner_training_repo", ping)
    monkeypatch.setattr(tm, "validate_github_tokens", AsyncMock())
    monkeypatch.setattr(tm, "validate_repo_obfuscation", AsyncMock(return_value=True))
    monkeypatch.setattr(tm, "validate_repo_license", AsyncMock(return_value=True))
    monkeypatch.setattr(tm, "_nodes_with_decoder_snapshots", AsyncMock(side_effect=lambda nodes: nodes))
    monkeypatch.setattr(tm, "enroll_tournament_participant_with_fee", enroll)
    monkeypatch.setattr(tm, "send_to_discord", state.discord)
    return state


def _config(**extra) -> SimpleNamespace:
    return SimpleNamespace(psql_db=None, discord_url="https://discord.example/hook", **extra)


async def test_short_quant_returns_zero_without_sleeping_and_alerts_once(enrolment, sleeps):
    enrolment.types["q"] = TournamentType.QUANT
    config = _config()

    assert await tm.populate_tournament_participants("q", config, None) == 0
    assert await tm.populate_tournament_participants("q", config, None) == 0

    assert sleeps == []
    assert enrolment.discord.await_count == 1
    message = enrolment.discord.await_args.kwargs["message"]
    assert "quant" in message and "0 enrolled" in message and "needs 2" in message and "not blocked" in message


async def test_thin_quant_does_not_block_text_in_the_same_pending_pass(enrolment, sleeps, monkeypatch):
    enrolment.types.update({"q": TournamentType.QUANT, "t": TournamentType.TEXT})
    enrolment.enrollable = {"5Miner1", "5Miner2"}
    # Quant gets nobody: its pings find nobody enrollable only for it, so route by tournament.
    real_enroll = tm.enroll_tournament_participant_with_fee

    async def enroll(participant, *args):
        return participant.tournament_id == "t" and await real_enroll(participant, *args)

    monkeypatch.setattr(tm, "enroll_tournament_participant_with_fee", enroll)
    monkeypatch.setattr(tm, "_repair_quant_boss_participant", AsyncMock())
    pending = [_tournament("q", TournamentType.QUANT), _tournament("t", TournamentType.TEXT)]
    monkeypatch.setattr(tm, "get_tournaments_with_status", AsyncMock(return_value=pending))
    activate = AsyncMock(return_value=True)
    monkeypatch.setattr(tm, "activate_pending_tournament", activate)
    monkeypatch.setattr(tm, "notify_tournament_started", AsyncMock())

    with pytest.raises(_StopLoop):
        await tm.process_pending_tournaments(_config())

    assert sleeps == [tm.t_cst.TOURNAMENT_PENDING_CYCLE_INTERVAL]
    activate.assert_awaited_once_with("t", None)


async def test_ping_pass_is_spaced_out_but_the_persisted_count_check_still_runs(enrolment, monkeypatch):
    enrolment.types["q"] = TournamentType.QUANT
    now = [1000.0]
    monkeypatch.setattr(tm.time, "monotonic", lambda: now[0])
    config = _config()

    await tm.populate_tournament_participants("q", config, None)
    assert enrolment.pings.await_count == 2  # first call pings both nodes immediately

    now[0] += tm.t_cst.TOURNAMENT_PARTICIPANT_RETRY_INTERVAL_SECONDS - 1
    assert await tm.populate_tournament_participants("q", config, None) == 0
    assert enrolment.pings.await_count == 2  # inside the interval: no new ping pass

    # Miners got enrolled elsewhere meanwhile: the cheap persisted check sees them without any ping.
    enrolment.persisted["q"] = ["5MinerA", "5MinerB"]
    assert await tm.populate_tournament_participants("q", config, None) == 2
    assert enrolment.pings.await_count == 2
    assert "q" not in tm._PARTICIPANT_NEXT_PASS_AT

    enrolment.persisted["q"] = []
    await tm.populate_tournament_participants("q", config, None)
    assert enrolment.pings.await_count == 4  # entry was cleared, so a new pass runs at once

    now[0] += tm.t_cst.TOURNAMENT_PARTICIPANT_RETRY_INTERVAL_SECONDS + 1
    await tm.populate_tournament_participants("q", config, None)
    assert enrolment.pings.await_count == 6  # interval elapsed


async def test_pass_that_reaches_the_minimum_activates_and_returns_the_count(enrolment):
    enrolment.types["t"] = TournamentType.TEXT
    enrolment.enrollable = {"5Miner1", "5Miner2"}

    assert await tm.populate_tournament_participants("t", _config(), None) == 2
    assert enrolment.discord.await_count == 0


async def test_one_raising_pending_tournament_does_not_stop_the_next(monkeypatch, sleeps):
    monkeypatch.setattr(
        tm, "get_tournaments_with_status",
        AsyncMock(return_value=[_tournament("bad", TournamentType.TEXT), _tournament("good", TournamentType.IMAGE)]),
    )

    async def populate(tournament_id, *_):
        if tournament_id == "bad":
            raise RuntimeError("boom")
        return 4

    monkeypatch.setattr(tm, "populate_tournament_participants", populate)
    activate = AsyncMock(return_value=True)
    monkeypatch.setattr(tm, "activate_pending_tournament", activate)
    monkeypatch.setattr(tm, "notify_tournament_started", AsyncMock())

    with pytest.raises(_StopLoop):
        await tm.process_pending_tournaments(_config())

    activate.assert_awaited_once_with("good", None)


async def test_one_raising_active_tournament_does_not_stop_the_next(monkeypatch, sleeps):
    monkeypatch.setattr(
        tm, "get_tournaments_with_status",
        AsyncMock(return_value=[
            _tournament("bad", TournamentType.QUANT, TournamentStatus.ACTIVE),
            _tournament("good", TournamentType.TEXT, TournamentStatus.ACTIVE),
        ]),
    )
    monkeypatch.setattr(tm, "_alert_if_near_weekly_deadline", AsyncMock())

    async def seed_tasks(tournament_id, _config):
        if tournament_id == "bad":
            raise RuntimeError("seed boom")

    monkeypatch.setattr(tm, "process_tournament_seed_tasks", seed_tasks)
    monkeypatch.setattr(tm, "seed_halt_reason", AsyncMock(return_value=None))
    monkeypatch.setattr(tm, "get_tournament_rounds", AsyncMock(return_value=[]))
    create_first_round = AsyncMock()
    monkeypatch.setattr(tm, "create_first_round_for_active_tournament", create_first_round)

    with pytest.raises(_StopLoop):
        await tm.process_active_tournaments(_config())

    create_first_round.assert_awaited_once()
    assert create_first_round.await_args.args[0] == "good"
