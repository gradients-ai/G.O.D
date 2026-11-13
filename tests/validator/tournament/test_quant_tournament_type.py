"""Quant is a tournament type of its own: scheduling, the boss lookup that seeds it from text, and the enrolment fee."""

from datetime import datetime
from datetime import timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fiber.chain.models import Node

from core.models.payload_models import TrainingRepoResponse
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament import constants as t_cst
from validator.tournament import participants
from validator.tournament import tournament_manager as tm
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentParticipant
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType


COMMIT = "c" * 40
IN_SLOT = datetime(2026, 11, 9, 9, 30, tzinfo=timezone.utc)  # Monday 09:30 UTC, inside quant's (0, 9) slot
OUT_OF_SLOT = datetime(2026, 11, 11, 15, 0, tzinfo=timezone.utc)  # Wednesday afternoon
IMAGE_SLOT = (t_cst.TOURNAMENT_SCHEDULE_IMAGE_DAY_OF_WEEK, t_cst.TOURNAMENT_SCHEDULE_IMAGE_HOUR)
CHAMPION = "5TextChampion"


def _freeze(monkeypatch, now: datetime) -> None:
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(tm, "datetime", FrozenDatetime)


class _StopLoop(Exception):
    pass


@pytest.fixture
def scheduling(monkeypatch):
    mocks = SimpleNamespace(create=AsyncMock(return_value="tourn_new"))
    monkeypatch.setattr(tm, "create_basic_tournament", mocks.create)
    monkeypatch.setattr(tm, "get_active_tournament", AsyncMock(return_value=None))
    monkeypatch.setattr(tm, "get_tournaments_with_status", AsyncMock(return_value=[]))
    monkeypatch.setattr(tm, "get_latest_tournament_with_created_at", AsyncMock(return_value=(None, None)))
    return mocks


async def _one_scheduler_pass(monkeypatch) -> list[TournamentType]:
    """The tournament types one pass of the scheduler loop checks, in order."""
    check = AsyncMock()

    async def stop(_seconds):
        raise _StopLoop

    monkeypatch.setattr(tm, "check_and_start_tournament", check)
    monkeypatch.setattr(tm.asyncio, "sleep", stop)
    with pytest.raises(_StopLoop):
        await tm.process_tournament_scheduling(SimpleNamespace(psql_db=None))
    return [call.args[0] for call in check.await_args_list]


class TestSchedulerLoop:
    async def test_every_pass_schedules_text_image_and_quant(self, monkeypatch):
        assert await _one_scheduler_pass(monkeypatch) == [TournamentType.TEXT, TournamentType.IMAGE, TournamentType.QUANT]


class TestFirstTournamentTiming:
    def test_quant_schedule_is_monday_nine(self):
        assert tm._get_tournament_schedule(TournamentType.QUANT) == (0, 9)
        assert tm._get_tournament_schedule(TournamentType.QUANT) == (
            t_cst.TOURNAMENT_SCHEDULE_QUANT_DAY_OF_WEEK,
            t_cst.TOURNAMENT_SCHEDULE_QUANT_HOUR,
        )

    @pytest.mark.parametrize("now", [IN_SLOT, OUT_OF_SLOT])
    @pytest.mark.parametrize("tournament_type", [TournamentType.TEXT, TournamentType.IMAGE, TournamentType.QUANT])
    async def test_first_tournament_of_any_type_is_immediate(self, scheduling, monkeypatch, tournament_type, now):
        _freeze(monkeypatch, now)

        await tm.check_and_start_tournament(tournament_type, None, SimpleNamespace())

        scheduling.create.assert_awaited_once()
        assert scheduling.create.await_args.args[0] == tournament_type

    async def test_later_quant_tournaments_follow_the_ordinary_slot_rule(self, scheduling, monkeypatch):
        previous = TournamentData(
            tournament_id="tourn_q1", tournament_type=TournamentType.QUANT, status=TournamentStatus.COMPLETED
        )
        monkeypatch.setattr(
            tm,
            "get_latest_tournament_with_created_at",
            AsyncMock(return_value=(previous, datetime(2026, 11, 2, 9, tzinfo=timezone.utc))),
        )
        monkeypatch.setattr(
            tm, "get_tournament_completion_time", AsyncMock(return_value=datetime(2026, 11, 6, 9, tzinfo=timezone.utc))
        )

        _freeze(monkeypatch, OUT_OF_SLOT)
        await tm.check_and_start_tournament(TournamentType.QUANT, None, SimpleNamespace())
        scheduling.create.assert_not_awaited()

        _freeze(monkeypatch, IN_SLOT)
        await tm.check_and_start_tournament(TournamentType.QUANT, None, SimpleNamespace())
        scheduling.create.assert_awaited_once()

    def test_text_and_image_slots_are_unchanged(self):
        assert tm._get_tournament_schedule(TournamentType.TEXT) == (
            t_cst.TOURNAMENT_SCHEDULE_TEXT_DAY_OF_WEEK,
            t_cst.TOURNAMENT_SCHEDULE_TEXT_HOUR,
        )
        assert tm._get_tournament_schedule(TournamentType.IMAGE) == IMAGE_SLOT


def _completed(
    tournament_id: str, tournament_type: TournamentType, winner: str, base_winner: str | None = None
) -> TournamentData:
    return TournamentData(
        tournament_id=tournament_id,
        tournament_type=tournament_type,
        status=TournamentStatus.COMPLETED,
        winner_hotkey=winner,
        base_winner_hotkey=base_winner,
    )


@pytest.fixture
def history(monkeypatch):
    """Completed tournaments by type; the participant table is keyed by (tournament_id, hotkey)."""
    state = SimpleNamespace(completed={}, participants={}, asked_types=[])

    async def latest_completed(_psql_db, tournament_type):
        state.asked_types.append(tournament_type)
        return state.completed.get(tournament_type)

    async def participant(tournament_id, hotkey, _psql_db):
        return state.participants.get((tournament_id, hotkey))

    monkeypatch.setattr(participants, "get_latest_completed_tournament", latest_completed)
    monkeypatch.setattr(participants, "get_tournament_participant", participant)
    monkeypatch.setattr(participants, "get_latest_commit_hash_from_github", AsyncMock(return_value=COMMIT))
    return state


def _participant(tournament_id: str, hotkey: str, repo: str, backup_repo: str | None = None) -> TournamentParticipant:
    return TournamentParticipant(
        tournament_id=tournament_id, hotkey=hotkey, training_repo=repo, training_commit_hash=COMMIT, backup_repo=backup_repo
    )


class TestBossLookup:
    async def test_quant_without_history_takes_the_latest_text_champion(self, history):
        history.completed[TournamentType.TEXT] = _completed("tourn_text", TournamentType.TEXT, CHAMPION)
        history.participants[("tourn_text", CHAMPION)] = _participant("tourn_text", CHAMPION, "https://github.com/c/repo")

        winner = await participants.get_latest_tournament_winner_participant(None, TournamentType.QUANT, SimpleNamespace())

        assert winner is not None and winner.hotkey == CHAMPION
        assert history.asked_types == [TournamentType.QUANT, TournamentType.TEXT]

    async def test_text_champion_defended_by_the_boss_placeholder_keeps_its_identity(self, history):
        history.completed[TournamentType.TEXT] = _completed(
            "tourn_text", TournamentType.TEXT, EMISSION_BURN_HOTKEY, base_winner=CHAMPION
        )
        history.participants[("tourn_text", EMISSION_BURN_HOTKEY)] = _participant(
            "tourn_text", EMISSION_BURN_HOTKEY, "https://github.com/boss/repo"
        )

        winner = await participants.get_latest_tournament_winner_participant(None, TournamentType.QUANT, SimpleNamespace())

        assert winner is not None
        assert winner.hotkey == CHAMPION
        assert winner.training_repo == "https://github.com/boss/repo"

    async def test_quant_with_a_completed_quant_tournament_never_looks_at_text(self, history):
        history.completed[TournamentType.QUANT] = _completed("tourn_quant", TournamentType.QUANT, "5QuantChampion")
        history.completed[TournamentType.TEXT] = _completed("tourn_text", TournamentType.TEXT, CHAMPION)
        history.participants[("tourn_quant", "5QuantChampion")] = _participant(
            "tourn_quant", "5QuantChampion", "https://github.com/q/repo"
        )
        history.participants[("tourn_text", CHAMPION)] = _participant("tourn_text", CHAMPION, "https://github.com/c/repo")

        winner = await participants.get_latest_tournament_winner_participant(None, TournamentType.QUANT, SimpleNamespace())

        assert winner is not None and winner.hotkey == "5QuantChampion"
        assert history.asked_types == [TournamentType.QUANT]

    async def test_a_quant_tournament_without_a_stored_winner_does_not_fall_back_to_text(self, history):
        history.completed[TournamentType.QUANT] = _completed("tourn_quant", TournamentType.QUANT, "")
        history.completed[TournamentType.TEXT] = _completed("tourn_text", TournamentType.TEXT, CHAMPION)

        winner = await participants.get_latest_tournament_winner_participant(None, TournamentType.QUANT, SimpleNamespace())

        assert winner is None

    async def test_text_without_history_does_not_use_quant(self, history):
        history.completed[TournamentType.QUANT] = _completed("tourn_quant", TournamentType.QUANT, "5QuantChampion")

        winner = await participants.get_latest_tournament_winner_participant(None, TournamentType.TEXT, SimpleNamespace())

        assert winner is None
        assert history.asked_types == [TournamentType.TEXT]

    async def test_boss_participant_is_the_text_champions_uploaded_repo(self, history):
        history.completed[TournamentType.TEXT] = _completed("tourn_text", TournamentType.TEXT, CHAMPION)
        history.participants[("tourn_text", CHAMPION)] = _participant(
            "tourn_text", CHAMPION, "https://github.com/c/repo", backup_repo="https://github.com/god/text-position-1"
        )

        boss = await participants.get_base_contestant(None, TournamentType.QUANT, SimpleNamespace())

        assert boss is not None
        assert boss.hotkey == EMISSION_BURN_HOTKEY
        assert boss.training_repo == "https://github.com/god/text-position-1"

    async def test_quant_with_no_history_anywhere_uses_the_default_repo(self, history):
        boss = await participants.get_base_contestant(None, TournamentType.QUANT, SimpleNamespace())

        assert boss is not None
        assert boss.training_repo == t_cst.DEFAULT_PARTICIPANT_REPO

    async def test_new_quant_tournament_snapshots_the_text_champions_identity(self, history, monkeypatch):
        history.completed[TournamentType.TEXT] = _completed(
            "tourn_text", TournamentType.TEXT, EMISSION_BURN_HOTKEY, base_winner=CHAMPION
        )
        history.participants[("tourn_text", EMISSION_BURN_HOTKEY)] = _participant(
            "tourn_text", EMISSION_BURN_HOTKEY, "https://github.com/boss/repo"
        )
        create = AsyncMock()
        monkeypatch.setattr(tm, "create_tournament", create)
        monkeypatch.setattr(tm, "add_tournament_participants", AsyncMock())
        monkeypatch.setattr(tm, "_with_boss_decoder_snapshot", AsyncMock(side_effect=lambda participant: participant))

        await tm.create_basic_tournament(TournamentType.QUANT, psql_db=None, config=SimpleNamespace())

        assert create.await_args.args[0].base_winner_hotkey == CHAMPION


def _node(index: int) -> Node:
    return Node(
        hotkey=f"5Miner{index}",
        coldkey=f"5Cold{index}",
        node_id=index,
        incentive=0.0,
        netuid=56,
        alpha_stake=0.0,
        tao_stake=0.0,
        stake=0.0,
        vtrust=0.0,
        last_updated=0.0,
        ip=f"10.0.0.{index}",
        ip_type=4,
        port=8091,
    )


@pytest.fixture
def enrolment(monkeypatch):
    nodes = [_node(1)]
    mocks = SimpleNamespace(
        tournament_type=TournamentType.QUANT,
        enroll=AsyncMock(return_value=True),
        pinged_types=[],
    )

    async def get_tournament(tournament_id, _psql_db):
        return TournamentData(tournament_id=tournament_id, tournament_type=mocks.tournament_type, status=TournamentStatus.PENDING)

    async def ping(node, _config, tournament_type):
        mocks.pinged_types.append(tournament_type)
        return TrainingRepoResponse(github_repo=f"https://github.com/{node.hotkey}/repo", commit_hash=COMMIT)

    persisted = [[], [TournamentParticipant(tournament_id="tourn_fee", hotkey="5Persisted")]]
    monkeypatch.setattr(tm.t_cst, "MIN_MINERS_FOR_TOURN", 1)
    monkeypatch.setattr(tm, "get_tournament", get_tournament)
    monkeypatch.setattr(tm, "get_tournament_participants", AsyncMock(side_effect=persisted))
    monkeypatch.setattr(tm, "get_all_nodes", AsyncMock(return_value=nodes))
    monkeypatch.setattr(tm, "_get_miner_training_repo", ping)
    monkeypatch.setattr(tm, "validate_github_tokens", AsyncMock())
    monkeypatch.setattr(tm, "validate_repo_obfuscation", AsyncMock(return_value=True))
    monkeypatch.setattr(tm, "validate_repo_license", AsyncMock(return_value=True))
    monkeypatch.setattr(tm, "_nodes_with_decoder_snapshots", AsyncMock(side_effect=lambda responding: responding))
    monkeypatch.setattr(tm, "enroll_tournament_participant_with_fee", mocks.enroll)
    return mocks


class TestEnrolmentFee:
    @pytest.mark.parametrize(
        ("tournament_type", "fee_rao"),
        [
            (TournamentType.QUANT, 600_000_000),
            (TournamentType.TEXT, 700_000_000),
            (TournamentType.IMAGE, 400_000_000),
        ],
    )
    async def test_fee_per_type(self, enrolment, tournament_type, fee_rao):
        enrolment.tournament_type = tournament_type

        await tm.populate_tournament_participants("tourn_fee", SimpleNamespace(), psql_db=None)

        assert enrolment.enroll.await_args.args[2] == fee_rao
        assert enrolment.pinged_types == [tournament_type]

    def test_quant_fee_constant_is_point_six_tao(self):
        assert t_cst.TOURNAMENT_QUANT_PARTICIPATION_FEE_RAO == 600_000_000

    @pytest.mark.parametrize(
        ("tournament_type", "pinned_commit"),
        [(TournamentType.QUANT, COMMIT), (TournamentType.TEXT, None), (TournamentType.IMAGE, None)],
    )
    async def test_licence_is_checked_at_the_pinned_commit_for_quant_only(
        self, enrolment, monkeypatch, tournament_type, pinned_commit
    ):
        license_check = AsyncMock(return_value=True)
        monkeypatch.setattr(tm, "validate_repo_license", license_check)
        enrolment.tournament_type = tournament_type

        await tm.populate_tournament_participants("tourn_fee", SimpleNamespace(), psql_db=None)

        assert license_check.await_args.args[2] == pinned_commit

    @pytest.mark.parametrize(
        ("tournament_type", "snapshotted"),
        [(TournamentType.QUANT, True), (TournamentType.TEXT, False), (TournamentType.IMAGE, False)],
    )
    async def test_only_quant_entrants_snapshot_decoders(self, enrolment, monkeypatch, tournament_type, snapshotted):
        snapshots = AsyncMock(side_effect=lambda responding: responding)
        monkeypatch.setattr(tm, "_nodes_with_decoder_snapshots", snapshots)
        enrolment.tournament_type = tournament_type

        await tm.populate_tournament_participants("tourn_fee", SimpleNamespace(), psql_db=None)

        assert snapshots.await_count == (1 if snapshotted else 0)
