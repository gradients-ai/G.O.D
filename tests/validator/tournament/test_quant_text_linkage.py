"""A quant tournament's tasks are created FROM the text tournament's boss tasks, as new tasks with a persisted link.

The two tournaments never share a task row, a result, a GPU reservation or a status. Checked here, in memory (the faking style
of the neighbouring tests) and against the SQL text and the migration:
  1. every slot's quant task is a different task from its text task, with its own rows;
  2. a quant_task_sources row names the text tournament, role and task, and after release the exact repo;
  3. a quant task is released only once the text boss's model has landed;
  4. creation is idempotent across restarts and a duplicate for a slot is refused and deleted;
  5. a failed or missing source neither halts a tournament nor creates a duplicate;
  6. the text side does not depend on any quant task.
"""

import inspect
import re
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import UUID
from uuid import uuid4

import pytest
from datasets import Dataset
from huggingface_hub.errors import RepositoryNotFoundError
from tenacity import wait_none

import validator.db.constants as db_cst
import validator.tournament.tournament_manager as tm
from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import FileFormat
from core.models.quant_models import QuantFormat
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from tests.validator.tournament import test_seed_tasks as seed_fixtures
from tests.validator.tournament.quant_source_fakes import FakeQuantSourceStore
from tests.validator.tournament.test_seed_adapter_merge import FULL_MODEL_FILES
from tests.validator.tournament.test_seed_adapter_merge import MERGED
from tests.validator.tournament.test_seed_adapter_merge import World as MergeWorld
from tests.validator.tournament.test_seed_tasks import PINNED
from tests.validator.tournament.test_seed_tasks import _instruct
from tests.validator.tournament.test_seed_tasks import env  # noqa: F401  (fixture)
from validator.db.sql import quant_sources as sources_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tasks.datasets import constants as data_cst
from validator.tasks.datasets import preparation
from validator.tasks.models import ChatRawTask
from validator.tasks.synthetics import scheduler
from validator.tournament import chaining
from validator.tournament import constants as t_cst
from validator.tournament import orchestrator
from validator.tournament import quant_sources
from validator.tournament import seed_tasks
from validator.tournament import task_creator
from validator.tournament.models import ANY_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import BOSS_ROUND_SEED_ROLES
from validator.tournament.models import QUANT_ROUND_SEED_ROLES
from validator.tournament.models import QUANT_SLOT_BY_SEED_ROLE
from validator.tournament.models import QUANT_SLOT_SPECS
from validator.tournament.models import RETIRED_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import TEXT_SEED_ROLES
from validator.tournament.models import ChunkScheme
from validator.tournament.models import ContinuousSftTaskChunk
from validator.tournament.models import Group
from validator.tournament.models import GroupRound
from validator.tournament.models import QuantSourceSlot
from validator.tournament.models import QuantTaskSource
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus


QUANT_TID = seed_fixtures.TID  # the quant tournament
TEXT_TID = "text_tourn_1"  # the text tournament it is bound to
CHALLENGER = "5Challenger"
CHUNK_INDEX = 7
REAL_SEED_CREATION_ORDER = seed_tasks.SEED_CREATION_ORDER
REAL_REPOINT = seed_tasks.repoint_chained_quant_task
MIGRATION = Path(__file__).parents[3] / "validator" / "db" / "migrations" / "20261113140000_quant_task_sources.sql"
ROUND_SLOTS = (QuantSourceSlot.R1, QuantSourceSlot.R2, QuantSourceSlot.R3)
BOSS_SLOTS = tuple(slot for slot in QuantSourceSlot if slot not in ROUND_SLOTS)
# The slots a quant tournament creates a seed for: rounds 1-3 and the four boss-round compressions.
CREATED_SLOTS = (
    QuantSourceSlot.R1,
    QuantSourceSlot.R2,
    QuantSourceSlot.R3,
    QuantSourceSlot.BOSS_R2_FREEFORM,
    QuantSourceSlot.BOSS_R3_FIXED,
    QuantSourceSlot.BOSS_CONTINUOUS_FREEFORM,
    QuantSourceSlot.BOSS_CONTINUOUS_FIXED,
)
SUCCESS = TrainingStatus.SUCCESS.value
FAILURE = TrainingStatus.FAILURE.value
TRAINING = TrainingStatus.TRAINING.value


class _Cancelled(BaseException):
    """The process stopping: not an Exception, so no `except Exception` in the code under test sees it."""


class HubDown(Exception):
    pass


# --------------------------------------------------------------------------------------------------------- the world


def _chat_task(**overrides) -> ChatRawTask:
    fields = dict(
        task_id=uuid4(),
        model_id="org/carried-winner",
        ds=t_cst.continuous_sft_ds("qwen3-8b", "chunk-label-3"),  # a name with another number: the index is not read from it
        status=TaskStatus.TRAINING,
        is_organic=False,
        account_id=uuid4(),
        hours_to_complete=2.5,
        created_at=datetime(2026, 1, 1),
        training_data="s3://bucket/continuous_train.json",
        test_data="s3://bucket/continuous_test.json",
        file_format=FileFormat.S3,
        training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
        model_params_count=8_000_000_000,
    )
    fields.update(overrides)
    return ChatRawTask(**fields)


def _held_quant(task_type, upstream, *, quant_format=None, **overrides):
    """A held chained quant task as the creation functions build it (a namespace, as in the neighbouring tests)."""
    fields = dict(
        task_id=uuid4(),
        task_type=task_type,
        model_id=upstream.model_id,
        ds=upstream.ds,
        status=TaskStatus.AWAITING_UPSTREAM,
        upstream_task_id=upstream.task_id,
        training_start_point=TrainingStartPoint.DEFAULT,
        quant_spec=quant_format.spec() if quant_format else None,
        training_data=None,
        test_data=None,
        calibration_data=None,
        hours_to_complete=1.0,
        termination_at=None,
        model_params_count=upstream.model_params_count,
        baseline_stats=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class QuantWorld:
    """A bound quant tournament (QUANT_TID) over an active text tournament (TEXT_TID) that has its four boss seeds.

    Persistence is in memory (the seed store of test_seed_tasks plus the quant source store); the hub is a dict of repo ->
    "full" or "adapter" (absent: not found); the text boss's training rows are `training`.
    """

    def __init__(self, env, monkeypatch) -> None:  # noqa: F811
        self.env = env
        self.mp = monkeypatch
        self.config = env.config
        self.sources = FakeQuantSourceStore().install(monkeypatch)
        self.created: list = []
        self.continuous_calls: list = []
        self.registered: list[UUID] = []
        self.registrations: list = []
        self.updates: list = []
        self.training: dict[str, str] = {}
        self.hub: dict[str, str] = {}
        self.hub_errors: set[str] = set()
        self.alerts = env.alerts
        self.discord = AsyncMock()
        self.chunk_fetches: list[int] = []
        self.fixed_seen: list = []

        monkeypatch.setattr(seed_tasks, "SEED_CREATION_ORDER", REAL_SEED_CREATION_ORDER)
        monkeypatch.setattr(seed_tasks, "repoint_chained_quant_task", REAL_REPOINT)
        monkeypatch.setattr(seed_tasks, "create_chained_quant_task", self._chained)
        monkeypatch.setattr(task_creator, "create_chained_quant_task", self._chained)
        monkeypatch.setattr(seed_tasks, "create_chained_continuous_quant_task", self._continuous)
        monkeypatch.setattr(
            seed_tasks,
            "get_continuous_sft_task_chunk",
            AsyncMock(
                side_effect=lambda task_id, _db: ContinuousSftTaskChunk(
                    task_id=task_id, lineage="qwen3-8b", train_index=CHUNK_INDEX, chunk_scheme=ChunkScheme.V2
                )
            ),
        )
        monkeypatch.setattr(task_sql, "get_task", self._get_task)
        monkeypatch.setattr(task_sql, "delete_task", self._delete_task)
        monkeypatch.setattr(task_sql, "update_task", self._update_task)
        monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", self._register)
        monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", self._existing_in_round)
        monkeypatch.setattr(task_creator, "_get_existing_tasks", self._existing_tasks)
        monkeypatch.setattr(task_creator.tournament_sql, "get_tournament_task_by_task_id", self._registered_row)
        monkeypatch.setattr(task_creator.seeds_sql, "mark_seed_attached", seeds_sql.mark_seed_attached)

        # The text tournament and its four seeds.
        self.text_tasks: dict[SeedRole, object] = {}
        self.text_rows: dict[SeedRole, TournamentSeedTask] = {}
        for role in sorted(TEXT_SEED_ROLES, key=lambda r: r.value):
            self.add_text_seed(role)
        self.tournaments = [
            TournamentData(tournament_id=TEXT_TID, tournament_type=TournamentType.TEXT, status=TournamentStatus.ACTIVE)
        ]
        monkeypatch.setattr(
            tournament_sql, "get_tournaments_with_status", AsyncMock(side_effect=lambda status, _db: self.tournaments)
        )

        self.completed: list[TournamentData] = []  # text tournaments completed within the binding window (newest first)
        monkeypatch.setattr(
            tournament_sql, "get_completed_tournaments_since", AsyncMock(side_effect=lambda _type, _since, _db: self.completed)
        )

        # The quant tournament itself is running (no winner), so its boss-round seeds are still replaced.
        running_quant = TournamentData(
            tournament_id=QUANT_TID, tournament_type=TournamentType.QUANT, status=TournamentStatus.ACTIVE
        )

        async def get_tournament(tournament_id, _db):
            known = [*self.tournaments, *self.completed, running_quant]
            return next((t for t in known if t.tournament_id == tournament_id), None)

        monkeypatch.setattr(tournament_sql, "get_tournament", get_tournament)
        monkeypatch.setattr(quant_sources, "_WAITING_SINCE", {})
        monkeypatch.setattr(quant_sources, "_WAIT_ALERTED", set())
        monkeypatch.setattr(quant_sources, "send_to_discord", self.discord)
        self.sources.bind(QUANT_TID, TEXT_TID)

        # The release seams.
        monkeypatch.setattr(
            orchestrator,
            "_training_status_of",
            AsyncMock(side_effect=lambda task_id, hotkey, _c: self.training.get(str(task_id))),
        )
        monkeypatch.setattr(orchestrator, "_is_full_model_repo", self._is_full_model)
        monkeypatch.setattr(seed_tasks, "is_adapter_repo", self._is_adapter)
        monkeypatch.setattr(chaining, "chained_input_repo", self._input_repo)
        monkeypatch.setattr(orchestrator, "_is_non_boss_tournament_task", AsyncMock(side_effect=self._is_round_task))
        monkeypatch.setattr(orchestrator, "_fetch_continuous_quant_chunk", self._fetch_chunk)

    # --- text side

    def add_text_seed(self, role: SeedRole, attempt: int = 1) -> TournamentSeedTask:
        if role == SeedRole.TEXT_CONTINUOUS:
            task = _chat_task()
        else:
            task = _instruct(
                model_id=f"org/base-{role.value}",
                ds=f"org/ds-{role.value}",
                status=TaskStatus.TRAINING,
                training_data="s3://bucket/seed_train.json",
                test_data="s3://bucket/seed_test.json",
                calibration_data="s3://bucket/seed_calibration.json",
            )
        row = TournamentSeedTask(
            tournament_id=TEXT_TID, role=role, attempt=attempt, task_id=task.task_id, status=SeedStatus.TRAINING
        )
        self.env.tasks[task.task_id] = task
        self.env.store.rows.append(row)
        self.text_tasks[role], self.text_rows[role] = task, row
        self.training[str(task.task_id)] = TRAINING
        return row

    def text_task(self, role: SeedRole):
        return self.text_tasks[role]

    def land(self, role: SeedRole, repo_state: str = "full") -> str:
        """The text boss's run on the role's text task succeeded and left a repo of the given kind on the hub."""
        task = self.text_task(role)
        self.training[str(task.task_id)] = SUCCESS
        repo = self.raw_repo(task.task_id)
        self.hub[repo] = repo_state
        return repo

    def raw_repo(self, task_id) -> str:
        return f"hub/tournament-{TEXT_TID}-{task_id}-boss"

    async def _input_repo(self, upstream_task_id, hotkey, _db):
        row = next((r for r in self.env.store.rows if r.task_id == UUID(str(upstream_task_id))), None)
        if row is not None and row.merged_output_repo:
            return row.merged_output_repo
        return self.raw_repo(upstream_task_id)

    async def _is_full_model(self, repo):
        if repo in self.hub_errors:
            raise HubDown(repo)
        return self.hub.get(repo) == "full"

    async def _is_adapter(self, repo):
        if repo in self.hub_errors:
            raise HubDown(repo)
        if repo not in self.hub:
            raise RepositoryNotFoundError(f"{repo} not found")
        return self.hub[repo] == "adapter"

    # --- creation seams

    async def _chained(self, config, task_type, upstream, *, quant_format, boss=True):
        task = _held_quant(task_type, upstream, quant_format=quant_format, hours_to_complete=2.0 if boss else 1.0)
        self.created.append((task, upstream, quant_format, boss))
        self.env.tasks[task.task_id] = task
        return task

    async def _continuous(self, config, task_type, upstream, *, quant_format, train_index, tournament_id):
        task = _held_quant(
            task_type,
            upstream,
            quant_format=quant_format,
            ds=t_cst.continuous_quant_ds(train_index, tournament_id, "chunk-label"),
            training_data="https://minio/train-at-creation",
            test_data="https://minio/test-at-creation",
        )
        self.continuous_calls.append((task, upstream, quant_format, train_index, tournament_id))
        self.env.tasks[task.task_id] = task
        return task

    async def _get_task(self, task_id, _db, connection=None):
        return self.env.tasks.get(UUID(str(task_id)))

    async def _delete_task(self, task_id, _db):
        self.env.tasks.pop(UUID(str(task_id)), None)
        await self.sources.delete_task(task_id, _db)

    async def _update_task(self, task, _db):
        self.updates.append(task.task_id)

    async def _register(self, task, tournament_id, round_id, config, group_id=None, pair_id=None):
        self.registered.append(task.task_id)
        self.registrations.append((task.task_id, tournament_id, round_id, group_id))

    async def _existing_in_round(self, round_id, config, group_id=None, pair_id=None):
        return [SimpleNamespace(task_id=task_id) for task_id in self.registered]

    async def _existing_tasks(self, tournament_tasks, config):
        return [self.env.tasks[t.task_id] for t in tournament_tasks]

    async def _registered_row(self, task_id, _db):
        return SimpleNamespace(task_id=task_id) if UUID(str(task_id)) in self.registered else None

    async def _fetch_chunk(self, config, train_index):
        self.chunk_fetches.append(train_index)
        return SimpleNamespace(
            train_url=f"https://minio/train-fresh-{len(self.chunk_fetches)}", test_url="https://minio/test-fresh"
        )

    async def _is_round_task(self, task, config):
        return await seeds_sql.get_seed_task_by_task_id(task.task_id, config.psql_db) is None

    # --- helpers

    async def create_seeds(self) -> dict[SeedRole, TournamentSeedTask]:
        seeds = await seed_tasks.create_seed_tasks(QUANT_TID, self.config, TournamentType.QUANT)
        return {seed.role: seed for seed in seeds}

    def quant_seed(self, role: SeedRole) -> TournamentSeedTask:
        return next(
            r for r in self.env.store.rows if r.tournament_id == QUANT_TID and r.role == role and r.status != SeedStatus.REPLACED
        )

    def quant_task(self, role: SeedRole):
        return self.env.tasks[self.quant_seed(role).task_id]

    async def create_round_task(self, round_number: int):
        round_data = GroupRound(
            round_id=f"{QUANT_TID}_round_{round_number:03d}", round_number=round_number, groups=[Group(member_ids=["a", "b"])]
        )
        # The round attaches the round seed the boss trains: the seed is created first, as the first round's creation does.
        await self.create_seeds()
        (task,) = await task_creator._create_seed_input_round_task(round_data, QUANT_TID, self.config)
        self.env.assigned.setdefault(str(task.task_id), [CHALLENGER])
        return task

    def text_snapshot(self) -> list:
        """Everything of the text tournament that a quant task must never touch."""
        return [
            [row.model_dump() for row in self.env.store.rows if row.tournament_id == TEXT_TID],
            {str(task_id): task.model_dump() for task_id, task in ((t.task_id, t) for t in self.text_tasks.values())},
        ]


@pytest.fixture
def qw(env, monkeypatch):  # noqa: F811
    return QuantWorld(env, monkeypatch)


def _source(qw: QuantWorld, slot: QuantSourceSlot) -> QuantTaskSource:
    source = qw.sources.live(QUANT_TID, slot)
    assert source is not None, f"no live source row for {slot.value}"
    return source


# ------------------------------------------------------------------------------------ the slot table and the constants


class TestSlotTable:
    def test_the_nine_slots_and_their_sources(self):
        assert [slot.value for slot in QuantSourceSlot] == [
            "r1",
            "r2",
            "r3",
            "boss_r2_freeform",
            "boss_r2_fixed",
            "boss_r3_freeform",
            "boss_r3_fixed",
            "boss_continuous_freeform",
            "boss_continuous_fixed",
        ]
        assert {slot: spec.text_role for slot, spec in QUANT_SLOT_SPECS.items()} == {
            QuantSourceSlot.R1: SeedRole.TEXT_R1,
            QuantSourceSlot.R2: SeedRole.TEXT_R2,
            QuantSourceSlot.R3: SeedRole.TEXT_R3,
            QuantSourceSlot.BOSS_R2_FREEFORM: SeedRole.TEXT_R2,
            QuantSourceSlot.BOSS_R2_FIXED: SeedRole.TEXT_R2,
            QuantSourceSlot.BOSS_R3_FREEFORM: SeedRole.TEXT_R3,
            QuantSourceSlot.BOSS_R3_FIXED: SeedRole.TEXT_R3,
            QuantSourceSlot.BOSS_CONTINUOUS_FREEFORM: SeedRole.TEXT_CONTINUOUS,
            QuantSourceSlot.BOSS_CONTINUOUS_FIXED: SeedRole.TEXT_CONTINUOUS,
        }

    def test_the_round_plan_follows_the_slot_table(self):
        for spec in t_cst.QUANT_ROUND_PLAN:
            slot_spec = QUANT_SLOT_SPECS[spec.slot]
            assert (spec.task_type, spec.input_role) == (slot_spec.task_type, slot_spec.text_role)
            assert slot_spec.quant_seed_role in QUANT_ROUND_SEED_ROLES and spec.input_role in TEXT_SEED_ROLES
        assert [spec.slot for spec in t_cst.QUANT_ROUND_PLAN] == list(ROUND_SLOTS)

    def test_every_slot_has_a_seed_role_and_seven_are_created(self):
        assert set(QUANT_SLOT_BY_SEED_ROLE) == ANY_BOSS_ROUND_SEED_ROLES - RETIRED_BOSS_ROUND_SEED_ROLES | QUANT_ROUND_SEED_ROLES
        assert set(seed_tasks.SEED_CREATION_ORDER) == BOSS_ROUND_SEED_ROLES | QUANT_ROUND_SEED_ROLES
        assert [QUANT_SLOT_BY_SEED_ROLE[role] for role in seed_tasks.SEED_CREATION_ORDER] == list(CREATED_SLOTS)
        assert BOSS_ROUND_SEED_ROLES == {
            SeedRole.BOSS_R2_FREEFORM,
            SeedRole.BOSS_R3_FIXED,
            SeedRole.BOSS_CONTINUOUS_FREEFORM,
            SeedRole.BOSS_CONTINUOUS_FIXED,
        }
        for role, slot in QUANT_SLOT_BY_SEED_ROLE.items():
            assert QUANT_SLOT_SPECS[slot].quant_seed_role == role
            if role in QUANT_ROUND_SEED_ROLES:
                continue
            assert slot.value == role.value
            assert QUANT_SLOT_SPECS[slot].task_type == (
                TaskType.QUANTFREEFORMTASK if role.value.endswith("freeform") else TaskType.QUANTFIXEDTASK
            )

    def test_the_old_roles_are_kept_for_old_rows_but_not_created(self):
        old = {SeedRole.INPUT_SMALL, SeedRole.INPUT_LARGE, SeedRole.BOSS_INSTRUCT, SeedRole.BOSS_CONTINUOUS_INSTRUCT}
        assert not old & set(seed_tasks.SEED_CREATION_ORDER)
        assert {SeedRole.BOSS_CHAINED_FREEFORM, SeedRole.BOSS_CHAINED_FIXED} <= ANY_BOSS_ROUND_SEED_ROLES
        assert not {SeedRole.BOSS_CHAINED_FREEFORM, SeedRole.BOSS_CHAINED_FIXED} & BOSS_ROUND_SEED_ROLES

    def test_the_fixed_formats_of_the_boss_round_are_two_and_three_bit_and_drawn_per_tournament_and_role(self):
        assert set(t_cst.QUANT_BOSS_ROUND_FIXED_FORMATS) == {QuantFormat.W3A16_ASYM_G128, QuantFormat.W2A16_ASYM_G64}
        fixed_roles = [r for r in seed_tasks.SEED_CREATION_ORDER if r.value.endswith("fixed")]
        assert len(fixed_roles) == 2
        for role in fixed_roles:
            assert seed_tasks._boss_fixed_format("t1", role) == seed_tasks._boss_fixed_format("t1", role)
            assert seed_tasks._boss_fixed_format("t1", role) in t_cst.QUANT_BOSS_ROUND_FIXED_FORMATS
        drawn = {tuple(seed_tasks._boss_fixed_format(f"t{i}", role) for role in fixed_roles) for i in range(40)}
        assert len({fmt for combo in drawn for fmt in combo}) == 2  # both formats occur
        assert any(len(set(combo)) == 2 for combo in drawn)  # the two tasks of one tournament can differ


# ------------------------------------------------------------------------------------------------------ the SQL module


class _Conn:
    def __init__(self, fetchrow=(), fetch=(), fetchval=()):
        self.calls: list[tuple[str, str, tuple]] = []
        self._fetchrow, self._fetch, self._fetchval = list(fetchrow), list(fetch), list(fetchval)

    async def execute(self, query, *args):
        self.calls.append(("execute", query, args))

    async def fetchrow(self, query, *args):
        self.calls.append(("fetchrow", query, args))
        return self._fetchrow.pop(0) if self._fetchrow else None

    async def fetch(self, query, *args):
        self.calls.append(("fetch", query, args))
        return self._fetch.pop(0) if self._fetch else []

    async def fetchval(self, query, *args):
        self.calls.append(("fetchval", query, args))
        return self._fetchval.pop(0) if self._fetchval else None

    def transaction(self):
        class _Tx:
            async def __aenter__(self_inner):
                return None

            async def __aexit__(self_inner, *exc):
                return False

        return _Tx()


class _Db:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    async def connection(self):
        conn = self.conn

        class _Ctx:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


def _flat(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _task_source_row(**overrides) -> dict:
    row = dict(
        quant_task_id=uuid4(),
        quant_tournament_id=QUANT_TID,
        slot="r2",
        source_tournament_id=TEXT_TID,
        source_role="text_r2",
        source_task_id=uuid4(),
        source_model_repo=None,
        source_is_fallback=False,
        superseded=False,
        created_at=None,
        updated_at=None,
    )
    row.update(overrides)
    return row


def _new_source(**overrides) -> QuantTaskSource:
    fields = dict(
        quant_task_id=uuid4(),
        quant_tournament_id=QUANT_TID,
        slot=QuantSourceSlot.R2,
        source_tournament_id=TEXT_TID,
        source_role=SeedRole.TEXT_R2,
        source_task_id=uuid4(),
    )
    fields.update(overrides)
    return QuantTaskSource(**fields)


class TestSql:
    async def test_bind_inserts_with_do_nothing_then_reads_the_binding_back(self):
        conn = _Conn(fetchrow=[{"quant_tournament_id": QUANT_TID, "source_tournament_id": TEXT_TID, "created_at": None}])

        bound = await sources_sql.bind_quant_tournament(QUANT_TID, TEXT_TID, _Db(conn))

        (kind1, insert, args1), (kind2, read, args2) = conn.calls
        assert kind1 == "execute" and "INSERT INTO quant_tournament_sources" in insert and "ON CONFLICT DO NOTHING" in insert
        assert args1 == (QUANT_TID, TEXT_TID)
        assert kind2 == "fetchrow" and "FROM quant_tournament_sources" in read and "WHERE quant_tournament_id = $1" in read
        assert args2 == (QUANT_TID,) and bound.source_tournament_id == TEXT_TID

    async def test_a_quant_tournament_whose_text_tournament_was_taken_reads_back_nothing(self):
        assert await sources_sql.bind_quant_tournament(QUANT_TID, TEXT_TID, _Db(_Conn())) is None

    async def test_the_binding_and_the_taken_text_tournaments_are_read_by_their_own_keys(self):
        conn = _Conn(fetchrow=[None], fetch=[[{"source_tournament_id": TEXT_TID}]])
        db = _Db(conn)

        assert await sources_sql.get_quant_tournament_source(QUANT_TID, db) is None
        assert await sources_sql.get_bound_source_tournament_ids(db) == {TEXT_TID}
        assert conn.calls[0][2] == (QUANT_TID,) and "quant_tournament_sources" in conn.calls[1][1]

    async def test_inserting_a_task_source_names_the_partial_unique_index_and_returns_the_row(self):
        source = _new_source()
        conn = _Conn(fetchrow=[_task_source_row(quant_task_id=source.quant_task_id)])

        result = await sources_sql.insert_quant_task_source(source, _Db(conn))

        ((kind, insert, args),) = conn.calls
        flat = _flat(insert)
        assert kind == "fetchrow" and "INSERT INTO quant_task_sources" in flat
        assert "ON CONFLICT (quant_tournament_id, slot) WHERE NOT superseded DO NOTHING RETURNING" in flat
        assert args == (
            source.quant_task_id,
            QUANT_TID,
            "r2",
            TEXT_TID,
            "text_r2",
            source.source_task_id,
            None,
            False,
            False,
        )
        assert result.inserted is True and result.source.quant_task_id == source.quant_task_id

    async def test_a_slot_conflict_returns_the_existing_live_row_instead_of_succeeding_with_a_second_task(self):
        existing = _task_source_row()
        conn = _Conn(fetchrow=[None, existing])  # the insert returns nothing; the live row is read back

        result = await sources_sql.insert_quant_task_source(_new_source(), _Db(conn))

        assert result.inserted is False and result.source.quant_task_id == existing["quant_task_id"]
        read = _flat(conn.calls[1][1])
        assert "NOT superseded" in read and conn.calls[1][2] == (QUANT_TID, "r2")

    async def test_replacing_supersedes_the_old_row_in_the_same_transaction_before_the_insert(self):
        old_id = uuid4()
        source = _new_source()
        conn = _Conn(fetchrow=[_task_source_row(quant_task_id=source.quant_task_id)])

        await sources_sql.insert_quant_task_source(source, _Db(conn), replacing_task_id=old_id)

        (first, second) = conn.calls
        assert first[0] == "execute" and "SET superseded = TRUE" in first[1] and first[2] == (old_id,)
        assert second[0] == "fetchrow" and "INSERT INTO quant_task_sources" in second[1]

    async def test_a_conflict_with_no_live_row_to_read_back_is_an_error(self):
        with pytest.raises(RuntimeError, match="could not record its source"):
            await sources_sql.insert_quant_task_source(_new_source(), _Db(_Conn(fetchrow=[None, None])))

    async def test_live_row_by_slot_by_task_and_the_listing(self):
        row = _task_source_row()
        conn = _Conn(fetchrow=[row, row], fetch=[[row]])
        db = _Db(conn)

        assert (await sources_sql.get_live_quant_task_source(QUANT_TID, QuantSourceSlot.R2, db)).slot == QuantSourceSlot.R2
        assert (await sources_sql.get_quant_task_source(row["quant_task_id"], db)).quant_task_id == row["quant_task_id"]
        assert len(await sources_sql.list_quant_task_sources(QUANT_TID, db)) == 1
        live, by_task, listing = (_flat(call[1]) for call in conn.calls)
        assert "NOT superseded" in live and conn.calls[0][2] == (QUANT_TID, "r2")
        assert "WHERE quant_task_id = $1" in by_task and "ORDER BY" in listing and conn.calls[2][2] == (QUANT_TID,)

    async def test_release_sets_the_exact_repo_and_the_fallback_flag(self):
        conn = _Conn()
        task_id = uuid4()

        await sources_sql.set_quant_task_source_model(task_id, "org/model", True, _Db(conn))

        ((kind, query, args),) = conn.calls
        assert kind == "execute" and "SET source_model_repo = $2, source_is_fallback = $3" in query
        assert "WHERE quant_task_id = $1" in query and args == (task_id, "org/model", True)

    async def test_marking_superseded(self):
        conn = _Conn()
        task_id = uuid4()

        await sources_sql.mark_quant_task_source_superseded(task_id, _Db(conn))

        assert "SET superseded = TRUE" in conn.calls[0][1] and conn.calls[0][2] == (task_id,)

    async def test_repointing_moves_the_source_task_only_while_the_task_holds_no_model(self):
        conn = _Conn()
        task_id, new_source = uuid4(), uuid4()

        await sources_sql.repoint_quant_task_source(task_id, new_source, _Db(conn))

        ((_, query, args),) = conn.calls
        assert "SET source_task_id = $2" in query and "source_model_repo IS NULL" in query and args == (task_id, new_source)


class TestMigration:
    def setup_method(self):
        self.sql = MIGRATION.read_text()
        self.up, self.down = self.sql.split("-- migrate:down")

    def test_the_slot_check_lists_exactly_the_nine_slots(self):
        check = re.search(r"slot TEXT NOT NULL CHECK \(slot IN \((.*?)\)\)", self.up, re.S)
        assert check is not None
        assert set(re.findall(r"'([a-z0-9_]+)'", check.group(1))) == {slot.value for slot in QuantSourceSlot}

    def test_at_most_one_live_task_per_slot_is_enforced_by_a_partial_unique_index(self):
        flat = _flat(self.up)
        assert re.search(
            r"CREATE UNIQUE INDEX IF NOT EXISTS \w+ ON quant_task_sources \(quant_tournament_id, slot\) WHERE NOT superseded;",
            flat,
        )
        assert "CREATE INDEX IF NOT EXISTS idx_quant_task_sources_source_task_id ON quant_task_sources (source_task_id);" in flat

    def test_one_text_tournament_feeds_at_most_one_quant_tournament(self):
        flat = _flat(self.up)
        assert "quant_tournament_id TEXT PRIMARY KEY REFERENCES tournaments(tournament_id)" in flat
        assert "source_tournament_id TEXT NOT NULL UNIQUE REFERENCES tournaments(tournament_id)" in flat

    def test_the_task_source_columns_and_keys(self):
        flat = _flat(self.up)
        for column in (
            "quant_task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE",
            "quant_tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id)",
            "source_tournament_id TEXT NOT NULL REFERENCES tournaments(tournament_id)",
            "source_role TEXT NOT NULL",
            "source_task_id UUID NOT NULL REFERENCES tasks(task_id)",
            "source_model_repo TEXT",
            "source_is_fallback BOOLEAN NOT NULL DEFAULT FALSE",
            "superseded BOOLEAN NOT NULL DEFAULT FALSE",
            "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
            "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
        ):
            assert column in flat, column

    def test_the_header_explains_the_columns_and_down_drops_both_tables(self):
        for word in ("source_task_id", "source_model_repo", "source_is_fallback", "superseded"):
            assert re.search(rf"^--\s+{word}\b", self.up, re.M)
        assert "DROP TABLE IF EXISTS quant_task_sources;" in self.down
        assert "DROP TABLE IF EXISTS quant_tournament_sources;" in self.down

    def test_the_table_and_column_names_are_the_constants_the_sql_module_uses(self):
        assert (
            db_cst.QUANT_TOURNAMENT_SOURCES_TABLE == "quant_tournament_sources"
            and db_cst.QUANT_TASK_SOURCES_TABLE == "quant_task_sources"
        )
        for name in (
            db_cst.QUANT_TASK_ID,
            db_cst.QUANT_TOURNAMENT_ID,
            db_cst.QUANT_SOURCE_SLOT,
            db_cst.QUANT_SOURCE_TOURNAMENT_ID,
            db_cst.QUANT_SOURCE_ROLE,
            db_cst.QUANT_SOURCE_TASK_ID,
            db_cst.QUANT_SOURCE_MODEL_REPO,
            db_cst.QUANT_SOURCE_IS_FALLBACK,
            db_cst.QUANT_SOURCE_SUPERSEDED,
        ):
            assert re.search(rf"\b{name}\b", self.up)


# ------------------------------------------------------------------------------------------------------- binding


class TestBinding:
    async def test_picks_the_newest_active_text_tournament_that_has_all_four_seeds_and_stores_it(self, qw):
        qw.sources.bindings.clear()
        qw.tournaments[:] = [
            TournamentData(
                tournament_id="text_new_without_seeds", tournament_type=TournamentType.TEXT, status=TournamentStatus.ACTIVE
            ),
            TournamentData(tournament_id="image_x", tournament_type=TournamentType.IMAGE, status=TournamentStatus.ACTIVE),
            TournamentData(tournament_id=TEXT_TID, tournament_type=TournamentType.TEXT, status=TournamentStatus.ACTIVE),
        ]

        binding = await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config)

        assert binding.source_tournament_id == TEXT_TID and qw.sources.bindings[QUANT_TID] == binding

    async def test_a_text_tournament_with_a_role_missing_is_not_eligible(self, qw):
        qw.sources.bindings.clear()
        qw.env.store.rows[:] = [r for r in qw.env.store.rows if r.role != SeedRole.TEXT_R3]

        assert await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config) is None
        assert qw.sources.bindings == {}

    async def test_the_binding_is_stable_and_a_second_quant_tournament_cannot_bind_the_same_text_tournament(self, qw):
        qw.sources.bindings.clear()

        first = await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config)
        again = await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config)
        other = await quant_sources.bind_quant_source_tournament("quant_other", qw.config)

        assert first == again and first.source_tournament_id == TEXT_TID
        assert other is None and set(qw.sources.bindings) == {QUANT_TID}

    async def test_a_second_text_tournament_feeds_the_second_quant_tournament(self, qw):
        qw.sources.bindings.clear()
        qw.tournaments.insert(
            0, TournamentData(tournament_id="text_2", tournament_type=TournamentType.TEXT, status=TournamentStatus.ACTIVE)
        )
        for role in TEXT_SEED_ROLES:
            qw.env.store.rows.append(
                TournamentSeedTask(tournament_id="text_2", role=role, task_id=uuid4(), status=SeedStatus.TRAINING)
            )

        first = await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config)
        second = await quant_sources.bind_quant_source_tournament("quant_other", qw.config)

        assert (first.source_tournament_id, second.source_tournament_id) == ("text_2", TEXT_TID)

    async def test_without_an_eligible_text_tournament_it_waits_quietly(self, qw, caplog):
        qw.sources.bindings.clear()
        qw.tournaments.clear()

        for _ in range(3):
            assert await quant_sources.bind_quant_source_tournament(QUANT_TID, qw.config) is None

        assert qw.sources.bindings == {} and qw.alerts.await_count == 0

    @pytest.fixture
    def first_round(self, qw, monkeypatch):
        monkeypatch.setattr(tm, "get_tournament_rounds", AsyncMock(return_value=[]))
        monkeypatch.setattr(
            tm,
            "get_tournament",
            AsyncMock(return_value=TournamentData(tournament_id=QUANT_TID, tournament_type=TournamentType.QUANT)),
        )
        monkeypatch.setattr(
            tm,
            "get_tournament_participants",
            AsyncMock(return_value=[SimpleNamespace(hotkey=EMISSION_BURN_HOTKEY), SimpleNamespace(hotkey="miner_a")]),
        )
        monkeypatch.setattr(tm, "get_node_by_hotkey", AsyncMock(return_value=SimpleNamespace(hotkey="miner_a")))
        monkeypatch.setattr(t_cst, "TOURN_DEDUP_ENABLED", False)
        create = AsyncMock()
        monkeypatch.setattr(tm, "_create_first_round", create)
        return create

    async def test_an_unbound_quant_tournament_creates_no_seed_no_round_and_does_not_halt(self, qw, first_round):
        qw.sources.bindings.clear()
        qw.tournaments.clear()

        created = await tm.create_first_round_for_active_tournament(QUANT_TID, qw.config, qw.config.psql_db)

        assert created is False and first_round.await_count == 0
        assert qw.env.store.rows == [r for r in qw.env.store.rows if r.tournament_id == TEXT_TID]  # no quant seed row
        assert qw.created == [] and qw.continuous_calls == [] and qw.sources.sources == []
        assert await seed_tasks.seed_halt_reason(QUANT_TID, qw.config.psql_db) is None
        await seed_tasks.process_tournament_seed_tasks(QUANT_TID, qw.config)  # no seed rows: left alone
        assert qw.alerts.await_count == 0

    async def test_a_bound_quant_tournament_binds_first_creates_its_seven_seeds_and_the_round(self, qw, first_round):
        qw.sources.bindings.clear()

        created = await tm.create_first_round_for_active_tournament(QUANT_TID, qw.config, qw.config.psql_db)

        assert created is True and first_round.await_count == 1
        assert qw.sources.bindings[QUANT_TID].source_tournament_id == TEXT_TID
        assert {r.role for r in qw.env.store.rows if r.tournament_id == QUANT_TID} == set(seed_tasks.SEED_CREATION_ORDER)
        assert {s.slot for s in qw.sources.sources} == set(CREATED_SLOTS)

    async def test_first_round_creation_twice_leaves_one_task_and_one_live_source_per_slot(self, qw, first_round):
        await tm.create_first_round_for_active_tournament(QUANT_TID, qw.config, qw.config.psql_db)
        tasks_after_first = len(qw.env.tasks)
        await tm.create_first_round_for_active_tournament(QUANT_TID, qw.config, qw.config.psql_db)

        assert len(qw.env.tasks) == tasks_after_first
        assert len(qw.created) + len(qw.continuous_calls) == 7
        assert sorted(s.slot.value for s in qw.sources.sources) == sorted(slot.value for slot in CREATED_SLOTS)
        assert len([r for r in qw.env.store.rows if r.tournament_id == QUANT_TID]) == 7


# ------------------------------------------------------------------------------- the miner rounds on the text seeds


class TestRoundTasks:
    @pytest.mark.parametrize("round_number", [1, 2, 3])
    async def test_a_round_attaches_its_quant_seed_chained_on_the_text_seed_task(self, qw, round_number):
        slot = QuantSourceSlot(f"r{round_number}")
        text_role = QUANT_SLOT_SPECS[slot].text_role

        task = await qw.create_round_task(round_number)

        seed = qw.quant_seed(QUANT_SLOT_SPECS[slot].quant_seed_role)
        text_task = qw.text_task(text_role)
        assert task.task_id == seed.task_id and seed.tournament_id == QUANT_TID
        assert seed.attached_round_id == f"{QUANT_TID}_round_{round_number:03d}"  # its finalisation waits for the round
        assert task.upstream_task_id == text_task.task_id and task.status == TaskStatus.AWAITING_UPSTREAM
        source = _source(qw, slot)
        assert (source.quant_task_id, source.source_tournament_id, source.source_role, source.source_task_id) == (
            task.task_id,
            TEXT_TID,
            text_role,
            text_task.task_id,
        )
        assert qw.registered == [task.task_id]
        assert qw.env.assigned[str(task.task_id)] == [EMISSION_BURN_HOTKEY]  # the boss trains it with its pinned code
        (_, _, boss) = [(t, u, b) for t, u, _, b in qw.created if t is task][0]
        assert boss is False  # the non-boss budget of a miner round

    async def test_the_round_resolves_the_source_from_the_bound_text_tournament_not_the_quant_one(self, qw):
        await qw.create_round_task(1)

        assert _source(qw, QuantSourceSlot.R1).source_tournament_id == TEXT_TID
        assert {r.tournament_id for r in qw.env.store.rows if r.role in TEXT_SEED_ROLES} == {TEXT_TID}

    async def test_a_round_whose_text_role_has_no_live_seed_stays_pending(self, qw):
        for row in qw.env.store.rows:
            if row.role == SeedRole.TEXT_R2:
                row.status = SeedStatus.REPLACED

        with pytest.raises(task_creator.MissingBossRoundSeeds) as error:
            await qw.create_round_task(2)

        assert error.value.roles == ["quant_r2"] and qw.registered == []
        assert QuantSourceSlot.R2 not in {s.slot for s in qw.sources.sources}

    async def test_a_round_never_creates_a_task_while_unbound(self, qw):
        qw.sources.bindings.clear()
        qw.tournaments.clear()

        with pytest.raises(task_creator.MissingBossRoundSeeds):
            await qw.create_round_task(1)

        assert qw.created == []

    async def test_a_restart_with_the_round_task_registered_creates_nothing(self, qw):
        first = await qw.create_round_task(1)
        made = len(qw.created)
        again = await qw.create_round_task(1)

        assert again.task_id == first.task_id and len(qw.created) == made and qw.registered == [first.task_id]


def _fail_quant_role(qw: QuantWorld, role: SeedRole) -> None:
    """The role is out of replacements: its latest row is `failed` and no seed of it is live."""
    qw.env.store.rows.append(
        TournamentSeedTask(tournament_id=QUANT_TID, role=role, attempt=3, task_id=uuid4(), status=SeedStatus.FAILED)
    )


async def _round_task_without_seed(qw: QuantWorld, round_number: int):
    round_data = GroupRound(
        round_id=f"{QUANT_TID}_round_{round_number:03d}", round_number=round_number, groups=[Group(member_ids=["a", "b"])]
    )
    (task,) = await task_creator._create_seed_input_round_task(round_data, QUANT_TID, qw.config)
    return task


class TestRoundTasksOfAFailedRole:
    """A round whose quant seed role is `failed` plays a task of its own, which the boss has no entry on."""

    @pytest.mark.parametrize("round_number", [1, 2, 3])
    async def test_the_round_creates_its_own_task_with_no_boss(self, qw, round_number):
        slot = QuantSourceSlot(f"r{round_number}")
        _fail_quant_role(qw, QUANT_SLOT_SPECS[slot].quant_seed_role)

        task = await _round_task_without_seed(qw, round_number)

        text_task = qw.text_task(QUANT_SLOT_SPECS[slot].text_role)
        assert task.upstream_task_id == text_task.task_id and task.status == TaskStatus.AWAITING_UPSTREAM
        assert _source(qw, slot).quant_task_id == task.task_id and qw.registered == [task.task_id]
        assert str(task.task_id) not in qw.env.assigned  # no boss assignment
        assert await seeds_sql.get_seed_task_by_task_id(task.task_id, qw.config.psql_db) is None
        (_, _, boss) = [(t, u, b) for t, u, _, b in qw.created if t is task][0]
        assert boss is False

    async def test_a_live_source_row_of_the_failed_seed_is_superseded(self, qw):
        await qw.create_seeds()
        failed = qw.quant_seed(SeedRole.QUANT_R1)
        failed.status = SeedStatus.FAILED

        task = await _round_task_without_seed(qw, 1)

        assert task.task_id != failed.task_id and _source(qw, QuantSourceSlot.R1).quant_task_id == task.task_id
        assert next(s for s in qw.sources.sources if s.quant_task_id == failed.task_id).superseded

    async def test_a_crash_between_creation_and_registration_registers_the_existing_task_and_creates_no_other(
        self, qw, monkeypatch
    ):
        _fail_quant_role(qw, SeedRole.QUANT_R2)
        real_register = qw._register
        crash = AsyncMock(side_effect=_Cancelled)
        monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", crash)
        with pytest.raises(_Cancelled):
            await _round_task_without_seed(qw, 2)
        assert len(qw.created) == 1 and qw.registered == []
        orphan = _source(qw, QuantSourceSlot.R2).quant_task_id

        monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", real_register)
        task = await _round_task_without_seed(qw, 2)

        assert task.task_id == orphan and len(qw.created) == 1  # no second task
        assert qw.registered == [orphan]
        assert [s.slot for s in qw.sources.sources] == [QuantSourceSlot.R2] and not qw.sources.deleted

    async def test_a_concurrent_pass_that_took_the_slot_first_leaves_its_task_and_the_duplicate_is_deleted(self, qw, monkeypatch):
        _fail_quant_role(qw, SeedRole.QUANT_R1)
        text_task = qw.text_task(SeedRole.TEXT_R1)
        winner = _held_quant(TaskType.QUANTFREEFORMTASK, text_task)
        qw.env.tasks[winner.task_id] = winner
        real_insert = qw.sources.insert_quant_task_source

        async def racing_insert(source, db, replacing_task_id=None):
            # The other pass records its task right before ours: the slot is taken when ours is inserted.
            qw.sources.sources.append(
                QuantTaskSource(
                    quant_task_id=winner.task_id,
                    quant_tournament_id=QUANT_TID,
                    slot=QuantSourceSlot.R1,
                    source_tournament_id=TEXT_TID,
                    source_role=SeedRole.TEXT_R1,
                    source_task_id=text_task.task_id,
                )
            )
            return await real_insert(source, db, replacing_task_id)

        monkeypatch.setattr(sources_sql, "insert_quant_task_source", racing_insert)

        task = await _round_task_without_seed(qw, 1)

        loser = qw.created[0][0]
        assert task.task_id == winner.task_id and loser.task_id != winner.task_id
        assert qw.sources.deleted == [loser.task_id] and loser.task_id not in qw.env.tasks
        assert [s.quant_task_id for s in qw.sources.sources] == [winner.task_id]

    async def test_a_source_row_that_cannot_be_recorded_deletes_the_task_and_raises(self, qw, monkeypatch):
        _fail_quant_role(qw, SeedRole.QUANT_R3)
        monkeypatch.setattr(sources_sql, "insert_quant_task_source", AsyncMock(side_effect=ConnectionError("db down")))

        with pytest.raises(ConnectionError):
            await _round_task_without_seed(qw, 3)

        created = qw.created[0][0]
        assert qw.sources.deleted == [created.task_id] and created.task_id not in qw.env.tasks and qw.registered == []


class TestReplacingARoundTask:
    async def test_replacing_a_round_task_moves_the_slot_to_the_new_task(self, qw, monkeypatch):
        original = await qw.create_round_task(2)
        new_task = _held_quant(original.task_type, qw.text_task(SeedRole.TEXT_R2))
        qw.env.tasks[new_task.task_id] = new_task
        monkeypatch.setattr(task_creator, "create_new_task_of_same_type", AsyncMock(return_value=new_task))
        monkeypatch.setattr(task_creator.task_sql, "get_nodes_assigned_to_task", AsyncMock(return_value=[]))
        monkeypatch.setattr(task_creator.task_sql, "detach_task_from_tournament", AsyncMock())
        monkeypatch.setattr(task_creator, "_repoint_chained_dependents", AsyncMock())

        await task_creator.replace_tournament_task(
            str(original.task_id), QUANT_TID, f"{QUANT_TID}_round_002", f"{QUANT_TID}_round_002_group_001", None, qw.config
        )

        live = _source(qw, QuantSourceSlot.R2)
        old = next(s for s in qw.sources.sources if s.quant_task_id == original.task_id)
        assert live.quant_task_id == new_task.task_id and old.superseded
        assert live.source_task_id == old.source_task_id and live.source_model_repo is None


class _ReplaceHarness:
    """The seams replace_tournament_task touches, in memory: node assignment (a second assignment of a node fails, as the
    table's key would), expected repos, detaching, and the creation of the replacement task."""

    def __init__(self, qw: QuantWorld, monkeypatch, original) -> None:
        self.qw = qw
        self.original = original
        self.round_id = f"{QUANT_TID}_round_002"
        self.group_id = f"{self.round_id}_group_001"
        self.nodes: dict[str, list] = {str(original.task_id): [SimpleNamespace(hotkey="hk1"), SimpleNamespace(hotkey="hk2")]}
        self.expected: dict[tuple[str, str], str] = {
            (str(original.task_id), "hk1"): "repo-1",
            (str(original.task_id), "hk2"): "repo-2",
        }
        self.created: list = []
        self.detached: list[str] = []
        self.detach_error: type[BaseException] | None = None
        self.assign_error_after: int | None = None
        self.assign_calls = 0
        monkeypatch.setattr(task_creator, "create_new_task_of_same_type", self._create)
        monkeypatch.setattr(task_sql, "get_nodes_assigned_to_task", self._nodes_of)
        monkeypatch.setattr(task_sql, "assign_node_to_task", self._assign)
        monkeypatch.setattr(task_sql, "get_expected_repo_name", self._expected_of)
        monkeypatch.setattr(task_sql, "set_expected_repo_name", self._set_expected)
        monkeypatch.setattr(task_sql, "detach_task_from_tournament", self._detach)
        monkeypatch.setattr(task_creator, "_repoint_chained_dependents", AsyncMock())
        monkeypatch.setattr(task_creator, "_copy_own_model_state", AsyncMock())

    async def _create(self, original, config, model_id_override=None):
        task = _held_quant(self.original.task_type, self.qw.text_task(SeedRole.TEXT_R2))
        self.qw.env.tasks[task.task_id] = task
        self.created.append(task)
        return task

    async def _nodes_of(self, task_id, _db):
        return list(self.nodes.get(str(task_id), []))

    async def _assign(self, task_id, node, _db):
        if self.assign_error_after is not None and self.assign_calls >= self.assign_error_after:
            raise _Cancelled
        self.assign_calls += 1
        held = self.nodes.setdefault(str(task_id), [])
        assert node.hotkey not in {n.hotkey for n in held}, "a node is assigned twice"
        held.append(node)

    async def _expected_of(self, task_id, hotkey, _db):
        return self.expected.get((str(task_id), hotkey))

    async def _set_expected(self, task_id, node, _db, name):
        self.expected[(str(task_id), node.hotkey)] = name

    async def _detach(self, task_id, tournament_id, _db):
        if self.detach_error is not None:
            raise self.detach_error
        self.detached.append(str(task_id))
        self.qw.registered[:] = [t for t in self.qw.registered if str(t) != str(task_id)]

    async def replace(self):
        return await task_creator.replace_tournament_task(
            str(self.original.task_id), QUANT_TID, self.round_id, self.group_id, None, self.qw.config
        )

    def assert_completed(self) -> None:
        """Exactly one live task for the slot, registered once, with the nodes, the original detached, no third task."""
        (task,) = self.created
        live = _source(self.qw, QuantSourceSlot.R2)
        assert live.quant_task_id == task.task_id
        assert [
            s.quant_task_id for s in self.qw.sources.sources if not s.superseded and s.slot == QuantSourceSlot.R2
        ] == [task.task_id]
        assert self.qw.registered == [task.task_id]
        assert {n.hotkey for n in self.nodes[str(task.task_id)]} == {"hk1", "hk2"}
        assert len(self.nodes[str(task.task_id)]) == 2
        assert (self.expected[(str(task.task_id), "hk1")], self.expected[(str(task.task_id), "hk2")]) == ("repo-1", "repo-2")
        assert self.detached == [str(self.original.task_id)]
        assert not self.qw.sources.deleted


class TestResumingATaskReplacement:
    async def _harness(self, qw, monkeypatch):
        original = await qw.create_round_task(2)
        return _ReplaceHarness(qw, monkeypatch, original)

    async def test_a_replacement_runs_through_once(self, qw, monkeypatch):
        harness = await self._harness(qw, monkeypatch)

        new_id = await harness.replace()

        assert new_id == harness.created[0].task_id
        harness.assert_completed()

    async def test_a_stop_after_the_source_row_moved_resumes_with_the_same_task(self, qw, monkeypatch):
        harness = await self._harness(qw, monkeypatch)
        real_register = qw._register
        monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock(side_effect=_Cancelled))
        with pytest.raises(_Cancelled):
            await harness.replace()
        (second,) = harness.created
        assert _source(qw, QuantSourceSlot.R2).quant_task_id == second.task_id  # the slot moved
        assert harness.original.task_id in qw.registered and second.task_id not in qw.registered
        monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", real_register)

        new_id = await harness.replace()  # the next pass: the original is still registered and failed

        assert new_id == second.task_id
        harness.assert_completed()

    async def test_a_stop_after_registration_before_the_detach_resumes_without_registering_twice(self, qw, monkeypatch):
        harness = await self._harness(qw, monkeypatch)
        harness.detach_error = _Cancelled
        with pytest.raises(_Cancelled):
            await harness.replace()
        (second,) = harness.created
        assert second.task_id in qw.registered and len(harness.nodes[str(second.task_id)]) == 2
        harness.detach_error = None

        new_id = await harness.replace()

        assert new_id == second.task_id
        harness.assert_completed()

    async def test_a_stop_in_the_middle_of_the_node_assignment_assigns_only_the_missing_nodes(self, qw, monkeypatch):
        harness = await self._harness(qw, monkeypatch)
        harness.assign_error_after = 1
        with pytest.raises(_Cancelled):
            await harness.replace()
        (second,) = harness.created
        assert [n.hotkey for n in harness.nodes[str(second.task_id)]] == ["hk1"]
        harness.assign_error_after = None

        await harness.replace()

        harness.assert_completed()

    async def test_a_registration_that_raises_resumes_the_same_way(self, qw, monkeypatch):
        harness = await self._harness(qw, monkeypatch)
        real_register = qw._register
        monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", AsyncMock(side_effect=RuntimeError("db down")))
        with pytest.raises(RuntimeError, match="db down"):
            await harness.replace()
        monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", real_register)

        await harness.replace()

        harness.assert_completed()

    async def test_a_concurrent_pass_that_took_the_slot_is_adopted_and_the_duplicate_deleted(self, qw, monkeypatch):
        harness = await self._harness(qw, monkeypatch)
        rival = _held_quant(harness.original.task_type, qw.text_task(SeedRole.TEXT_R2))
        qw.env.tasks[rival.task_id] = rival
        real_insert = qw.sources.insert_quant_task_source

        async def racing_insert(source, db, replacing_task_id=None):
            # Another pass moves the slot to its own task right before ours is inserted.
            await qw.sources.mark_quant_task_source_superseded(harness.original.task_id, db)
            qw.sources.sources.append(source.model_copy(update={"quant_task_id": rival.task_id, "superseded": False}))
            return await real_insert(source, db, replacing_task_id)

        monkeypatch.setattr(sources_sql, "insert_quant_task_source", racing_insert)

        new_id = await harness.replace()

        (duplicate,) = harness.created
        assert new_id == rival.task_id and duplicate.task_id not in qw.env.tasks
        assert qw.sources.deleted == [duplicate.task_id]
        assert _source(qw, QuantSourceSlot.R2).quant_task_id == rival.task_id and qw.registered == [rival.task_id]
        assert {n.hotkey for n in harness.nodes[str(rival.task_id)]} == {"hk1", "hk2"}
        assert harness.detached == [str(harness.original.task_id)]


# ------------------------------------------------------------------------ the quant seeds: rounds 1-3 and the boss round


class TestCompressionSeeds:
    async def test_the_seven_seeds_are_new_held_chained_tasks_on_the_text_seed_task_of_their_source_role(self, qw):
        seeds = await qw.create_seeds()

        assert list(seeds) == list(seed_tasks.SEED_CREATION_ORDER) and len(qw.sources.sources) == 7
        for role, seed in seeds.items():
            slot = QUANT_SLOT_BY_SEED_ROLE[role]
            text_task = qw.text_task(QUANT_SLOT_SPECS[slot].text_role)
            task = qw.env.tasks[seed.task_id]
            assert task.task_id != text_task.task_id and task.upstream_task_id == text_task.task_id
            assert task.status == TaskStatus.AWAITING_UPSTREAM and task.task_type == QUANT_SLOT_SPECS[slot].task_type
            source = _source(qw, slot)
            assert (source.quant_task_id, source.source_tournament_id, source.source_task_id) == (
                task.task_id,
                TEXT_TID,
                text_task.task_id,
            )
            assert source.source_role == QUANT_SLOT_SPECS[slot].text_role

    async def test_the_quant_seeds_belong_to_the_quant_tournament_and_the_text_rows_are_untouched(self, qw):
        before = qw.text_snapshot()

        await qw.create_seeds()

        assert qw.text_snapshot() == before
        quant_rows = [r for r in qw.env.store.rows if r.tournament_id == QUANT_TID]
        assert len(quant_rows) == 7 and all(
            (r.boss_repo, r.boss_commit) == (PINNED.training_repo, PINNED.training_commit_hash) for r in quant_rows
        )

    async def test_the_round_one_to_three_models_compress_through_the_ordinary_chained_path(self, qw):
        seeds = await qw.create_seeds()

        assert len(qw.created) == 5
        by_task = {task.task_id: (upstream, fmt, boss) for task, upstream, fmt, boss in qw.created}
        for role in (SeedRole.QUANT_R1, SeedRole.QUANT_R2, SeedRole.QUANT_R3):
            upstream, fmt, boss = by_task[seeds[role].task_id]
            spec = next(plan for plan in t_cst.QUANT_ROUND_PLAN if plan.slot == QUANT_SLOT_BY_SEED_ROLE[role])
            assert boss is False and upstream.task_id == qw.text_task(spec.input_role).task_id  # the miner-round budget
            assert fmt == spec.quant_format  # R2 is the pinned W2A16-asym-g64, R1 and R3 are freeform
        for role in (SeedRole.BOSS_R2_FREEFORM, SeedRole.BOSS_R3_FIXED):
            upstream, fmt, boss = by_task[seeds[role].task_id]
            assert boss is True
            assert (fmt is not None) == (role == SeedRole.BOSS_R3_FIXED)
            assert fmt is None or fmt in t_cst.QUANT_BOSS_ROUND_FIXED_FORMATS
        assert by_task[seeds[SeedRole.BOSS_R2_FREEFORM].task_id][0].task_id == qw.text_task(SeedRole.TEXT_R2).task_id
        assert by_task[seeds[SeedRole.BOSS_R3_FIXED].task_id][0].task_id == qw.text_task(SeedRole.TEXT_R3).task_id

    async def test_the_continuous_seeds_take_the_chunk_index_from_the_text_tasks_chunk_record(self, qw):
        await qw.create_seeds()

        assert len(qw.continuous_calls) == 2
        text_task = qw.text_task(SeedRole.TEXT_CONTINUOUS)
        for task, upstream, fmt, train_index, tournament_id in qw.continuous_calls:
            assert upstream is text_task and train_index == CHUNK_INDEX and tournament_id == QUANT_TID
            assert task.upstream_task_id == text_task.task_id
        assert "3" in text_task.ds and str(CHUNK_INDEX) not in text_task.ds  # the name could not have given the index
        source = inspect.getsource(seed_tasks.draw_compression_task)
        assert "get_continuous_sft_task_chunk" in source and "continuous_quant_train_index" not in source

    async def test_a_boss_seed_waits_for_a_text_role_with_no_live_seed_and_the_others_are_created(self, qw):
        for row in qw.env.store.rows:
            if row.role == SeedRole.TEXT_R3:
                row.status = SeedStatus.REPLACED

        seeds = await qw.create_seeds()

        assert set(seeds) == set(seed_tasks.SEED_CREATION_ORDER) - {SeedRole.QUANT_R3, SeedRole.BOSS_R3_FIXED}
        assert {s.slot for s in qw.sources.sources} == set(CREATED_SLOTS) - {QuantSourceSlot.R3, QuantSourceSlot.BOSS_R3_FIXED}

    async def test_the_seeds_are_assigned_to_the_boss_and_the_quant_tournaments_code_pins_them(self, qw):
        for seed in (await qw.create_seeds()).values():
            assert qw.env.assigned[str(seed.task_id)] == [EMISSION_BURN_HOTKEY]
            assert qw.env.expected[str(seed.task_id)] == f"tournament-{QUANT_TID}-{seed.task_id}-{EMISSION_BURN_HOTKEY[:8]}"
            assert seed.tournament_id == QUANT_TID  # the quant tournament's participant row, never the text tournament's

    async def test_creation_twice_makes_one_task_and_one_live_source_per_slot(self, qw):
        first = await qw.create_seeds()
        second = await qw.create_seeds()

        assert {r: s.task_id for r, s in first.items()} == {r: s.task_id for r, s in second.items()}
        assert len(qw.created) + len(qw.continuous_calls) == 7
        live = sorted(s.slot.value for s in qw.sources.sources if not s.superseded)
        assert live == sorted(slot.value for slot in CREATED_SLOTS)

    async def test_a_crash_after_the_task_and_source_but_before_the_seed_row_adopts_the_task(self, qw, monkeypatch):
        real_insert = seeds_sql.insert_seed_task
        calls = {"n": 0}

        async def crashing_insert(seed, db):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _Cancelled
            return await real_insert(seed, db)

        monkeypatch.setattr(seeds_sql, "insert_seed_task", crashing_insert)
        with pytest.raises(_Cancelled):
            await qw.create_seeds()
        orphan = _source(qw, QuantSourceSlot.R1).quant_task_id
        made_before = len(qw.created) + len(qw.continuous_calls)
        assert made_before == 1 and not [r for r in qw.env.store.rows if r.tournament_id == QUANT_TID]

        seeds = await qw.create_seeds()

        assert seeds[SeedRole.QUANT_R1].task_id == orphan
        assert len(qw.created) + len(qw.continuous_calls) == 7  # the orphan plus six new, not eight
        assert sorted(s.slot.value for s in qw.sources.sources) == sorted(slot.value for slot in CREATED_SLOTS)
        assert not qw.sources.deleted

    async def test_a_failing_seed_row_insert_deletes_the_task_and_its_source(self, qw, monkeypatch):
        monkeypatch.setattr(seeds_sql, "insert_seed_task", AsyncMock(side_effect=ConnectionError("db down")))

        assert await qw.create_seeds() == {}

        made = [t for t, *_ in qw.created] + [c[0] for c in qw.continuous_calls]
        assert len(made) == 7 and sorted(qw.sources.deleted) == sorted(t.task_id for t in made)
        assert qw.sources.sources == [] and not any(t.task_id in qw.env.tasks for t in made)

    async def test_a_slot_taken_by_a_concurrent_pass_deletes_the_duplicate_and_creates_no_seed(self, qw, monkeypatch):
        text_task = qw.text_task(SeedRole.TEXT_R2)
        winner = _held_quant(TaskType.QUANTFREEFORMTASK, text_task)
        qw.env.tasks[winner.task_id] = winner
        qw.sources.sources.append(
            QuantTaskSource(
                quant_task_id=winner.task_id,
                quant_tournament_id=QUANT_TID,
                slot=QuantSourceSlot.BOSS_R2_FREEFORM,
                source_tournament_id=TEXT_TID,
                source_role=SeedRole.TEXT_R2,
                source_task_id=text_task.task_id,
            )
        )
        # The live row has a seed row of ANOTHER attempt: this attempt must not make a second task.
        qw.env.store.rows.append(
            TournamentSeedTask(tournament_id=QUANT_TID, role=SeedRole.BOSS_R2_FREEFORM, attempt=5, task_id=winner.task_id)
        )

        seed = await seed_tasks._create_seed(SeedRole.BOSS_R2_FREEFORM, QUANT_TID, 1, {}, qw.config)

        assert seed is None and qw.created == [] and qw.sources.deleted == []
        assert [s.quant_task_id for s in qw.sources.sources] == [winner.task_id]

    async def test_nothing_is_created_while_the_quant_tournament_is_unbound(self, qw):
        qw.sources.bindings.clear()
        qw.tournaments.clear()

        assert await qw.create_seeds() == {}
        assert qw.created == [] and qw.continuous_calls == [] and qw.sources.sources == []


class TestCompressionSeedLifecycle:
    async def _quant_world(self, qw):
        await qw.create_seeds()
        return qw

    async def test_a_boss_failure_is_the_bosss_failed_entry_not_a_redraw_and_nothing_halts(self, qw, monkeypatch):
        await qw.create_seeds()
        seed = qw.quant_seed(SeedRole.BOSS_R3_FIXED)
        task = qw.quant_task(SeedRole.BOSS_R3_FIXED)
        task.status = TaskStatus.TRAINING
        monkeypatch.setattr(
            tournament_sql,
            "get_training_status_for_task_and_hotkeys",
            AsyncMock(return_value={EMISSION_BURN_HOTKEY: FAILURE}),
        )
        before = len(qw.env.tasks)

        await seed_tasks.process_tournament_seed_tasks(QUANT_TID, qw.config)

        assert qw.quant_seed(SeedRole.BOSS_R3_FIXED).status == SeedStatus.TRAINED and seed.attempt == 1
        assert len(qw.env.tasks) == before and await seed_tasks.seed_halt_reason(QUANT_TID, qw.config.psql_db) is None

    async def test_a_prep_failure_is_a_charged_replacement_that_moves_the_slot_to_a_new_task_on_the_same_text_seed(self, qw):
        await qw.create_seeds()
        old = qw.quant_seed(SeedRole.BOSS_R2_FREEFORM)
        old_task = qw.env.tasks[old.task_id]
        old_task.status = TaskStatus.PREP_TASK_FAILURE

        new = await seed_tasks.replace_seed(old, "prep failed", qw.config)

        assert new.attempt == 2 and new.task_id != old.task_id
        live = _source(qw, QuantSourceSlot.BOSS_R2_FREEFORM)
        superseded = next(s for s in qw.sources.sources if s.quant_task_id == old.task_id)
        assert live.quant_task_id == new.task_id and superseded.superseded
        assert live.source_task_id == superseded.source_task_id == qw.text_task(SeedRole.TEXT_R2).task_id
        assert [r.status for r in qw.env.store.rows if r.role == SeedRole.BOSS_R2_FREEFORM] == [
            SeedStatus.REPLACED,
            SeedStatus.TRAINING,
        ]

    async def test_a_role_out_of_replacements_fails_without_halting_and_alerts_once(self, qw):
        await qw.create_seeds()
        role = SeedRole.BOSS_R2_FREEFORM
        for attempt in range(1, t_cst.SEED_MAX_REPLACEMENTS + 1):
            seed = qw.quant_seed(role)
            qw.env.tasks[seed.task_id].status = TaskStatus.FAILURE
            await seed_tasks.replace_seed(seed, "failed", qw.config)
        last = qw.quant_seed(role)
        qw.env.tasks[last.task_id].status = TaskStatus.FAILURE
        qw.alerts.reset_mock()

        assert await seed_tasks.replace_seed(last, "failed again", qw.config) is None
        await seed_tasks.replace_seed(last, "failed again", qw.config)

        latest = {r.role: r for r in await seeds_sql.get_latest_seed_tasks(QUANT_TID, None)}
        assert latest[role].status == SeedStatus.FAILED
        assert await seed_tasks.seed_halt_reason(QUANT_TID, None) is None
        assert qw.alerts.await_count == 1 and "not halted" in qw.alerts.await_args.kwargs["message"]

    async def test_a_text_role_out_of_replacements_does_not_halt_its_tournament(self, qw):
        for row in qw.env.store.rows:
            if row.role == SeedRole.TEXT_R1:
                row.status = SeedStatus.FAILED

        assert await seed_tasks.seed_halt_reason(TEXT_TID, None) is None


# ------------------------------------------------------------------------------------------------------- release


class TestRelease:
    async def release(self, qw, task):
        await orchestrator._release_chained_task(task, qw.config)

    async def _round_task(self, qw, round_number=1):
        task = await qw.create_round_task(round_number)
        return task

    async def test_held_while_the_text_boss_is_training(self, qw):
        task = await self._round_task(qw)

        await self.release(qw, task)

        assert task.status == TaskStatus.AWAITING_UPSTREAM and _source(qw, QuantSourceSlot.R1).source_model_repo is None
        assert qw.updates == [] and qw.alerts.await_count == 0

    async def test_held_while_the_output_is_an_adapter_awaiting_its_merge(self, qw):
        task = await self._round_task(qw)
        qw.land(SeedRole.TEXT_R1, "adapter")

        await self.release(qw, task)

        assert task.status == TaskStatus.AWAITING_UPSTREAM and _source(qw, QuantSourceSlot.R1).source_model_repo is None
        assert qw.alerts.await_count == 0

    async def test_released_with_the_merged_repo_once_it_is_stored(self, qw):
        task = await self._round_task(qw)
        raw = qw.land(SeedRole.TEXT_R1, "adapter")
        qw.hub[MERGED] = "full"
        for row in qw.env.store.rows:
            if row.task_id == qw.text_task(SeedRole.TEXT_R1).task_id:
                row.merged_output_repo = MERGED

        await self.release(qw, task)

        assert raw != MERGED and task.status == TaskStatus.PENDING and task.model_id == MERGED
        source = _source(qw, QuantSourceSlot.R1)
        assert (source.source_model_repo, source.source_is_fallback) == (MERGED, False)

    async def test_released_with_the_raw_repo_for_a_full_model_output(self, qw):
        task = await self._round_task(qw)
        raw = qw.land(SeedRole.TEXT_R1, "full")

        await self.release(qw, task)

        assert task.status == TaskStatus.PENDING and task.model_id == raw
        source = _source(qw, QuantSourceSlot.R1)
        assert (source.source_model_repo, source.source_is_fallback) == (raw, False) and qw.alerts.await_count == 0

    async def test_the_upstream_data_is_copied_for_a_pinned_task_and_the_hours_are_the_non_boss_budget(self, qw):
        task = await self._round_task(qw)
        qw.land(SeedRole.TEXT_R1)

        await self.release(qw, task)

        upstream = qw.text_task(SeedRole.TEXT_R1)
        assert (task.training_data, task.test_data, task.calibration_data) == (
            upstream.training_data,
            upstream.test_data,
            upstream.calibration_data,
        )
        assert task.hours_to_complete == scheduler.quant_training_hours(
            task.task_type, upstream.model_params_count, boss=False, baseline_stats=upstream.baseline_stats
        )

    async def test_the_size_the_text_tasks_prep_recorded_reaches_the_task_at_release(self, qw):
        task = await self._round_task(qw)
        task.model_params_count = 0  # created before the text task was prepped
        qw.text_task(SeedRole.TEXT_R1).model_params_count = 1_700_000_000
        qw.land(SeedRole.TEXT_R1)

        await self.release(qw, task)

        assert task.model_params_count == 1_700_000_000

    async def test_a_hub_error_holds_and_raises_for_a_retry(self, qw):
        task = await self._round_task(qw)
        raw = qw.land(SeedRole.TEXT_R1)
        qw.hub_errors.add(raw)

        with pytest.raises(HubDown):
            await self.release(qw, task)

        assert task.status == TaskStatus.AWAITING_UPSTREAM and _source(qw, QuantSourceSlot.R1).source_model_repo is None

    async def test_held_until_a_node_is_assigned(self, qw):
        task = await self._round_task(qw)
        qw.land(SeedRole.TEXT_R1)
        qw.env.assigned[str(task.task_id)] = []

        await self.release(qw, task)

        assert task.status == TaskStatus.AWAITING_UPSTREAM

    async def test_releasing_twice_is_impossible_the_pass_only_picks_held_tasks(self, qw):
        task = await self._round_task(qw)
        qw.land(SeedRole.TEXT_R1)
        await self.release(qw, task)
        assert task.status == TaskStatus.PENDING

        held = await task_sql.get_tasks_with_status(TaskStatus.AWAITING_UPSTREAM, None)

        assert task not in held  # process_awaiting_upstream_tasks releases only what it reads as AWAITING_UPSTREAM
        source = inspect.getsource(orchestrator.process_awaiting_upstream_tasks)
        assert "TaskStatus.AWAITING_UPSTREAM" in source

    @pytest.mark.parametrize("role", sorted(BOSS_ROUND_SEED_ROLES, key=lambda r: r.value))
    async def test_every_boss_slot_is_released_on_the_exact_repo_and_the_source_row_names_it(self, qw, role):
        await qw.create_seeds()
        slot = QUANT_SLOT_BY_SEED_ROLE[role]
        text_role = QUANT_SLOT_SPECS[slot].text_role
        task = qw.quant_task(role)
        raw = qw.land(text_role)

        await self.release(qw, task)

        assert task.status == TaskStatus.PENDING and task.model_id == raw
        source = _source(qw, slot)
        assert (source.source_model_repo, source.source_is_fallback, source.source_role) == (raw, False, text_role)
        assert source.source_tournament_id == TEXT_TID and source.source_task_id == qw.text_task(text_role).task_id

    @pytest.mark.parametrize("round_number", [1, 2, 3])
    async def test_every_miner_round_slot_is_released_on_the_exact_repo_and_the_source_row_names_it(self, qw, round_number):
        slot = QuantSourceSlot(f"r{round_number}")
        text_role = QUANT_SLOT_SPECS[slot].text_role
        task = await self._round_task(qw, round_number)
        raw = qw.land(text_role)

        await self.release(qw, task)

        source = _source(qw, slot)
        assert task.model_id == raw and (source.source_model_repo, source.source_role) == (raw, text_role)

    async def test_a_boss_compression_is_released_with_the_boss_hours_and_still_assigned_to_the_boss(self, qw):
        await qw.create_seeds()
        task = qw.quant_task(SeedRole.BOSS_R2_FREEFORM)
        qw.land(SeedRole.TEXT_R2)

        await self.release(qw, task)

        upstream = qw.text_task(SeedRole.TEXT_R2)
        assert task.hours_to_complete == scheduler.quant_training_hours(
            task.task_type, upstream.model_params_count, boss=True, baseline_stats=upstream.baseline_stats
        )
        assert qw.env.assigned[str(task.task_id)] == [EMISSION_BURN_HOTKEY]


class TestReleaseLeavesACalibrationSource:
    async def test_a_released_task_copies_the_text_tasks_calibration_rows_and_still_carves_when_it_has_none(self, qw):
        task = await qw.create_round_task(1)
        text_task = qw.text_task(SeedRole.TEXT_R1)
        qw.land(SeedRole.TEXT_R1, "full")
        task.carve_calibration = False  # a task created with the flag off

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING
        assert task.calibration_data == text_task.calibration_data == "s3://bucket/seed_calibration.json"
        assert task.carve_calibration is True
        assert qw.alerts.await_count == 0

    async def test_a_text_task_that_published_no_calibration_rows_is_alerted_and_the_task_carves_its_own(self, qw, caplog):
        task = await qw.create_round_task(1)
        text_task = qw.text_task(SeedRole.TEXT_R1)
        text_task.calibration_data = None
        qw.land(SeedRole.TEXT_R1, "full")

        with caplog.at_level("ERROR"):
            await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and task.calibration_data is None and task.carve_calibration is True
        assert qw.alerts.await_count == 1
        message = qw.alerts.await_args.kwargs["message"]
        assert str(text_task.task_id) in message and "published no calibration rows" in message
        assert "published no calibration rows" in caplog.text


class TestReleaseOfTheContinuousCompressions:
    async def test_the_continuous_task_keeps_its_chunk_data_gets_fresh_urls_and_is_released_on_the_models_repo(self, qw):
        await qw.create_seeds()
        task = qw.quant_task(SeedRole.BOSS_CONTINUOUS_FIXED)
        ds_before = task.ds
        raw = qw.land(SeedRole.TEXT_CONTINUOUS)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and task.model_id == raw
        assert task.ds == ds_before and t_cst.continuous_quant_train_index(task.ds) == CHUNK_INDEX
        assert qw.chunk_fetches == [CHUNK_INDEX] and task.training_data.startswith("https://minio/train-fresh")
        assert task.training_start_point == TrainingStartPoint.DEFAULT  # a shared input: both players compress the one model
        assert task.calibration_data is None  # nothing was copied from the text task, which has no such split
        assert task.carve_calibration is True  # its own prep carves from the chunk
        assert qw.alerts.await_count == 0

    async def test_a_failed_chunk_refresh_holds_the_task(self, qw):
        await qw.create_seeds()
        task = qw.quant_task(SeedRole.BOSS_CONTINUOUS_FREEFORM)
        qw.land(SeedRole.TEXT_CONTINUOUS)
        qw.mp.setattr(
            orchestrator, "_fetch_continuous_quant_chunk", AsyncMock(side_effect=ConnectionError("content service down"))
        )

        await orchestrator._release_chained_task(task, qw.config)

        assert (
            task.status == TaskStatus.AWAITING_UPSTREAM
            and _source(qw, QuantSourceSlot.BOSS_CONTINUOUS_FREEFORM).source_model_repo is None
        )

    async def test_the_continuous_adapter_output_holds_until_its_merge_is_stored(self, qw):
        await qw.create_seeds()
        task = qw.quant_task(SeedRole.BOSS_CONTINUOUS_FREEFORM)
        qw.land(SeedRole.TEXT_CONTINUOUS, "adapter")

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.AWAITING_UPSTREAM and qw.alerts.await_count == 0

    async def test_hours_come_out_of_the_normal_boss_round_quant_budget_after_prep(self, qw, monkeypatch):
        await qw.create_seeds()
        task = qw.quant_task(SeedRole.BOSS_CONTINUOUS_FREEFORM)
        seen = []
        monkeypatch.setattr(orchestrator, "quant_hours_after_prep", lambda t, stats, boss: seen.append(boss) or 3.0)

        hours = await orchestrator._hours_after_prep(task, qw.config)

        assert hours == 3.0 and seen == [True]  # a boss-round budget, from the task's own prep stats
        assert (
            data_cst.QUANT_BOSS_HOURS_MULTIPLIER[TaskType.QUANTFREEFORMTASK]
            == 1.0
            == data_cst.QUANT_BOSS_HOURS_MULTIPLIER[TaskType.QUANTFIXEDTASK]
        )


# --------------------------------------------------------------------------- the real continuous creation function


class _CreationSeams:
    def __init__(self, monkeypatch) -> None:
        self.content_calls: list = []

        async def add_task(task, psql_db):
            task.task_id = uuid4()
            return task

        async def content_service(endpoint, keypair, params):
            self.content_calls.append(params)
            return {"train_s3_url": "https://minio/train?sig=1", "test_s3_url": "https://minio/test?sig=2", "ds": "label-x"}

        monkeypatch.setattr(scheduler, "add_task", add_task)
        monkeypatch.setattr(scheduler, "call_content_service", content_service)
        monkeypatch.setattr(scheduler.create_chained_continuous_quant_task.retry, "wait", wait_none())


class TestCreateChainedContinuousQuantTask:
    @pytest.fixture
    def seams(self, monkeypatch):
        return _CreationSeams(monkeypatch)

    @pytest.mark.parametrize("task_type", sorted(QUANT_TASK_TYPES, key=lambda t: t.value))
    async def test_the_task_carries_the_same_chunk_and_is_held_as_a_shared_input(self, seams, task_type):
        upstream = _chat_task(model_params_count=0)
        fmt = QuantFormat.W2A16_ASYM_G64 if task_type == TaskType.QUANTFIXEDTASK else None

        task = await scheduler.create_chained_continuous_quant_task(
            MagicMock(), task_type, upstream, quant_format=fmt, train_index=CHUNK_INDEX, tournament_id=QUANT_TID
        )

        assert seams.content_calls == [{"train_index": CHUNK_INDEX}]
        assert task.ds == t_cst.continuous_quant_ds(CHUNK_INDEX, QUANT_TID, "label-x")
        assert (task.status, task.upstream_task_id, task.model_id) == (
            TaskStatus.AWAITING_UPSTREAM,
            upstream.task_id,
            upstream.model_id,
        )
        assert task.training_start_point == TrainingStartPoint.DEFAULT and task.file_format == FileFormat.S3
        assert (task.training_data, task.test_data) == ("https://minio/train?sig=1", "https://minio/test?sig=2")
        assert task.carve_calibration is True
        assert task.dataset_type == scheduler._continuous_quant_dataset_type()
        assert task.hours_to_complete == scheduler.quant_flat_hours(task_type, 0, boss=True) > 0
        assert (getattr(task, "quant_spec", None) == fmt.spec()) if fmt else not hasattr(task, "quant_spec")

    async def test_a_recreated_copy_keeps_the_chunk_and_gets_fresh_urls(self, seams):
        upstream = _chat_task()
        task = await scheduler.create_chained_continuous_quant_task(
            MagicMock(), TaskType.QUANTFREEFORMTASK, upstream, quant_format=None, train_index=CHUNK_INDEX, tournament_id=QUANT_TID
        )
        task.training_data = task.test_data = "tokenised-after-prep"
        seams.content_calls.clear()

        fresh = await scheduler.recreate_chained_quant_task(MagicMock(), task)

        assert fresh.ds == task.ds and fresh.upstream_task_id == upstream.task_id and fresh.status == TaskStatus.AWAITING_UPSTREAM
        assert fresh.training_data == "https://minio/train?sig=1" and seams.content_calls == [{"train_index": CHUNK_INDEX}]
        assert fresh.training_start_point == TrainingStartPoint.DEFAULT and fresh.task_id != task.task_id

    async def test_prep_takes_the_whole_chunk_not_a_half_of_it(self, seams):
        rows = 47_500
        chunk = Dataset.from_dict({"id": list(range(rows))})
        task = await scheduler.create_chained_continuous_quant_task(
            MagicMock(),
            TaskType.QUANTFIXEDTASK,
            _chat_task(),
            quant_format=QuantFormat.W2A16_ASYM_G64,
            train_index=CHUNK_INDEX,
            tournament_id=QUANT_TID,
        )

        assert t_cst.continuous_rows_part(task) is None
        assert preparation.select_continuous_rows(task, chunk) is chunk
        # A compression on the own-upstream model of a continuous chunk is split into disjoint shares.
        own = SimpleNamespace(
            task_id=uuid4(), task_type=task.task_type, ds=task.ds, training_start_point=TrainingStartPoint.OWN_UPSTREAM_MODEL
        )
        assert t_cst.continuous_rows_part(own)[0] == t_cst.ContinuousRowsPart.COMPRESSION

    async def test_repointing_a_continuous_task_moves_only_the_upstream_and_keeps_the_data(self, seams):
        old_upstream, new_upstream = _chat_task(), _chat_task(model_id="org/other-carried", model_params_count=4_000_000_000)
        task = await scheduler.create_chained_continuous_quant_task(
            MagicMock(),
            TaskType.QUANTFREEFORMTASK,
            old_upstream,
            quant_format=None,
            train_index=CHUNK_INDEX,
            tournament_id=QUANT_TID,
        )
        ds, urls = task.ds, (task.training_data, task.test_data)

        scheduler.repoint_chained_quant_task(task, new_upstream, pinned_data=False, upstream_train_index=CHUNK_INDEX)

        assert task.upstream_task_id == new_upstream.task_id and task.model_id == "org/other-carried"
        assert task.model_params_count == 4_000_000_000 and (task.ds, (task.training_data, task.test_data)) == (ds, urls)

    async def test_repointing_onto_another_chunk_is_refused(self, seams):
        upstream = _chat_task()
        task = await scheduler.create_chained_continuous_quant_task(
            MagicMock(), TaskType.QUANTFREEFORMTASK, upstream, quant_format=None, train_index=CHUNK_INDEX, tournament_id=QUANT_TID
        )

        with pytest.raises(ValueError, match="not re-pointing"):
            scheduler.repoint_chained_quant_task(task, _chat_task(), pinned_data=False, upstream_train_index=CHUNK_INDEX + 1)

        assert task.upstream_task_id == upstream.task_id


# ----------------------------------------------------------------------- a failed or missing source, the re-point


class TestFailedOrMissingSource:
    async def test_a_text_boss_failed_entry_releases_on_the_fallback_model_flagged_with_one_alert_and_no_halt(self, qw):
        task = await qw.create_round_task(1)
        text_task = qw.text_task(SeedRole.TEXT_R1)
        qw.training[str(text_task.task_id)] = FAILURE
        qw.text_rows[SeedRole.TEXT_R1].status = SeedStatus.TRAINED  # final: the seed loop recorded the boss's failed entry
        qw.hub[text_task.model_id] = "full"

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and task.model_id == text_task.model_id
        source = _source(qw, QuantSourceSlot.R1)
        assert (source.source_model_repo, source.source_is_fallback) == (text_task.model_id, True)
        assert qw.alerts.await_count == 1 and "fallback" in qw.alerts.await_args.kwargs["message"]
        assert (
            await seed_tasks.seed_halt_reason(TEXT_TID, None) is None
            and await seed_tasks.seed_halt_reason(QUANT_TID, None) is None
        )
        assert qw.text_rows[SeedRole.TEXT_R1].status == SeedStatus.TRAINED  # the text seed itself is untouched

    async def test_a_boss_failure_that_may_still_be_replaced_holds(self, qw):
        task = await qw.create_round_task(1)
        text_task = qw.text_task(SeedRole.TEXT_R1)
        qw.training[str(text_task.task_id)] = FAILURE  # the seed row is still `training`: the seed loop may replace it
        qw.hub[text_task.model_id] = "full"

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.AWAITING_UPSTREAM and qw.alerts.await_count == 0

    async def test_a_succeeded_run_with_a_missing_output_takes_the_fallback(self, qw):
        task = await qw.create_round_task(2)
        text_task = qw.text_task(SeedRole.TEXT_R2)
        qw.training[str(text_task.task_id)] = SUCCESS  # no repo on the hub
        qw.hub[text_task.model_id] = "full"
        qw.text_rows[SeedRole.TEXT_R2].status = SeedStatus.TRAINED  # the seed loop has seen the run (old enough: no timestamp)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.model_id == text_task.model_id and _source(qw, QuantSourceSlot.R2).source_is_fallback is True

    async def test_a_succeeded_run_with_a_non_model_output_takes_the_fallback(self, qw):
        task = await qw.create_round_task(2)
        text_task = qw.text_task(SeedRole.TEXT_R2)
        qw.land(SeedRole.TEXT_R2, "empty")
        qw.hub[text_task.model_id] = "full"
        qw.text_rows[SeedRole.TEXT_R2].status = SeedStatus.TRAINED  # the seed loop has seen the run (old enough: no timestamp)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.model_id == text_task.model_id and _source(qw, QuantSourceSlot.R2).source_is_fallback is True

    async def test_a_succeeded_run_with_a_missing_output_holds_while_the_row_is_still_training(self, qw):
        task = await qw.create_round_task(2)
        text_task = qw.text_task(SeedRole.TEXT_R2)
        qw.training[str(text_task.task_id)] = SUCCESS  # no repo on the hub; the seed row is still `training`
        qw.hub[text_task.model_id] = "full"

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.AWAITING_UPSTREAM and qw.alerts.await_count == 0

    async def test_the_continuous_fallback_uses_the_lineage_seed_model_when_the_carried_base_is_an_adapter(self, qw):
        await qw.create_seeds()
        task = qw.quant_task(SeedRole.BOSS_CONTINUOUS_FIXED)
        text_task = qw.text_task(SeedRole.TEXT_CONTINUOUS)
        qw.training[str(text_task.task_id)] = FAILURE
        qw.text_rows[SeedRole.TEXT_CONTINUOUS].status = SeedStatus.TRAINED
        qw.hub[text_task.model_id] = "adapter"  # the carried previous winner
        seed_model = t_cst.continuous_sft_seed_repo_for_ds(text_task.ds)
        qw.hub[seed_model] = "full"

        await orchestrator._release_chained_task(task, qw.config)

        assert task.model_id == seed_model and seed_model == "Qwen/Qwen3-8B-Base"
        source = _source(qw, QuantSourceSlot.BOSS_CONTINUOUS_FIXED)
        assert (source.source_model_repo, source.source_is_fallback) == (seed_model, True)

    async def test_the_continuous_fallback_prefers_the_base_when_it_is_a_full_model(self, qw):
        await qw.create_seeds()
        task = qw.quant_task(SeedRole.BOSS_CONTINUOUS_FIXED)
        text_task = qw.text_task(SeedRole.TEXT_CONTINUOUS)
        qw.training[str(text_task.task_id)] = FAILURE
        qw.text_rows[SeedRole.TEXT_CONTINUOUS].status = SeedStatus.TRAINED
        qw.hub[text_task.model_id] = "full"

        await orchestrator._release_chained_task(task, qw.config)

        assert task.model_id == text_task.model_id

    async def test_with_no_full_model_fallback_the_task_holds_and_alerts_once(self, qw):
        task = await qw.create_round_task(1)
        text_task = qw.text_task(SeedRole.TEXT_R1)
        qw.training[str(text_task.task_id)] = FAILURE
        qw.text_rows[SeedRole.TEXT_R1].status = SeedStatus.TRAINED
        qw.hub[text_task.model_id] = "adapter"

        for _ in range(3):
            await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.AWAITING_UPSTREAM and qw.alerts.await_count == 1
        assert await seed_tasks.seed_halt_reason(TEXT_TID, None) is None

    async def test_a_pending_text_seed_replacement_holds_then_the_task_is_repointed_and_released_on_the_replacement(self, qw):
        task = await qw.create_round_task(2)
        old_text = qw.text_task(SeedRole.TEXT_R2)
        qw.training[str(old_text.task_id)] = TRAINING
        await orchestrator._release_chained_task(task, qw.config)
        assert task.status == TaskStatus.AWAITING_UPSTREAM and task.upstream_task_id == old_text.task_id

        old_text.status = TaskStatus.PREP_TASK_FAILURE.value
        new_seed = await seed_tasks.replace_seed(qw.text_rows[SeedRole.TEXT_R2], "prep", qw.config, prep_rejected=True)

        new_text = qw.env.tasks[new_seed.task_id]
        assert new_text.task_id != old_text.task_id and task.upstream_task_id == new_text.task_id
        assert task.model_id == new_text.model_id and task.ds == new_text.ds  # base and data follow the replacement
        source = _source(qw, QuantSourceSlot.R2)
        assert source.source_task_id == new_text.task_id and source.quant_task_id == task.task_id

        new_text.training_data, new_text.test_data = "s3://bucket/new_train.json", "s3://bucket/new_test.json"
        qw.training[str(new_text.task_id)] = SUCCESS
        qw.hub[qw.raw_repo(new_text.task_id)] = "full"
        qw.text_rows[SeedRole.TEXT_R2] = new_seed
        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and task.model_id == qw.raw_repo(new_text.task_id)
        assert _source(qw, QuantSourceSlot.R2).source_model_repo == task.model_id

    async def test_a_task_still_on_a_replaced_seed_after_a_restart_follows_the_live_seed(self, qw):
        task = await qw.create_round_task(1)
        old_text = qw.text_task(SeedRole.TEXT_R1)
        qw.text_rows[SeedRole.TEXT_R1].status = SeedStatus.REPLACED  # replaced, but the re-point never ran
        await orchestrator._release_chained_task(task, qw.config)
        assert task.upstream_task_id == old_text.task_id  # no live successor yet: held

        successor = qw.add_text_seed(SeedRole.TEXT_R1, attempt=2)
        qw.env.store.rows[:] = [r for r in qw.env.store.rows if r is not qw.env.store.rows[-1]] + [successor]
        await orchestrator._release_chained_task(task, qw.config)

        assert task.upstream_task_id == successor.task_id and _source(qw, QuantSourceSlot.R1).source_task_id == successor.task_id

    async def test_a_released_task_is_never_moved_by_a_later_text_replacement(self, qw):
        task = await qw.create_round_task(1)
        raw = qw.land(SeedRole.TEXT_R1)
        await orchestrator._release_chained_task(task, qw.config)
        old_text = qw.text_task(SeedRole.TEXT_R1)
        old_text.status = TaskStatus.PREP_TASK_FAILURE.value

        await seed_tasks.replace_seed(qw.text_rows[SeedRole.TEXT_R1], "prep", qw.config, prep_rejected=True)

        assert task.status == TaskStatus.PENDING and task.upstream_task_id == old_text.task_id and task.model_id == raw
        assert _source(qw, QuantSourceSlot.R1).source_task_id == old_text.task_id

    async def test_a_continuous_task_is_repointed_only_to_the_same_chunk(self, qw):
        await qw.create_seeds()
        task = qw.quant_task(SeedRole.BOSS_CONTINUOUS_FREEFORM)
        old_text = qw.text_task(SeedRole.TEXT_CONTINUOUS)
        new_text = _chat_task()
        qw.env.tasks[new_text.task_id] = new_text
        ds_before = task.ds

        assert await seed_tasks.repoint_task_onto_seed(task, old_text, new_text, qw.config) is True  # same chunk 7 on both
        assert task.upstream_task_id == new_text.task_id and task.ds == ds_before
        assert _source(qw, QuantSourceSlot.BOSS_CONTINUOUS_FREEFORM).source_task_id == new_text.task_id

        other = _chat_task()
        qw.env.tasks[other.task_id] = other
        qw.mp.setattr(
            seed_tasks,
            "get_continuous_sft_task_chunk",
            AsyncMock(
                return_value=ContinuousSftTaskChunk(
                    task_id=other.task_id, lineage="qwen3-8b", train_index=CHUNK_INDEX + 1, chunk_scheme=ChunkScheme.V2
                )
            ),
        )
        qw.alerts.reset_mock()
        assert await seed_tasks.repoint_task_onto_seed(task, new_text, other, qw.config) is False
        assert task.upstream_task_id == new_text.task_id and qw.alerts.await_count == 1


class TestTextTournamentOver:
    """Once the text tournament is over nothing merges, replaces or trains on its seeds, so the state of the seed is final
    whatever its row says."""

    @pytest.mark.parametrize("status", [TournamentStatus.COMPLETED, TournamentStatus.CANCELLED])
    async def test_an_adapter_output_whose_merge_is_not_stored_takes_the_fallback(self, qw, status):
        task = await qw.create_round_task(1)
        text_task = qw.text_task(SeedRole.TEXT_R1)
        qw.land(SeedRole.TEXT_R1, "adapter")
        qw.hub[text_task.model_id] = "full"
        qw.tournaments[0].status = status

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and task.model_id == text_task.model_id
        assert _source(qw, QuantSourceSlot.R1).source_is_fallback is True and qw.alerts.await_count == 1

    async def test_a_stored_merge_is_used(self, qw):
        task = await qw.create_round_task(1)
        qw.land(SeedRole.TEXT_R1, "adapter")
        qw.hub[MERGED] = "full"
        qw.text_rows[SeedRole.TEXT_R1].merged_output_repo = MERGED
        qw.tournaments[0].status = TournamentStatus.COMPLETED

        await orchestrator._release_chained_task(task, qw.config)

        assert task.model_id == MERGED and _source(qw, QuantSourceSlot.R1).source_is_fallback is False
        assert qw.alerts.await_count == 0

    async def test_a_full_model_output_is_used(self, qw):
        task = await qw.create_round_task(1)
        raw = qw.land(SeedRole.TEXT_R1, "full")
        qw.tournaments[0].status = TournamentStatus.COMPLETED

        await orchestrator._release_chained_task(task, qw.config)

        assert task.model_id == raw and _source(qw, QuantSourceSlot.R1).source_is_fallback is False

    async def test_a_failed_run_on_a_seed_row_never_moved_to_a_final_status_takes_the_fallback(self, qw):
        task = await qw.create_round_task(1)
        text_task = qw.text_task(SeedRole.TEXT_R1)
        qw.training[str(text_task.task_id)] = FAILURE  # the row is still `training`
        qw.hub[text_task.model_id] = "full"
        qw.tournaments[0].status = TournamentStatus.COMPLETED

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and _source(qw, QuantSourceSlot.R1).source_is_fallback is True

    @pytest.mark.parametrize("boss_status", [TRAINING, None])
    async def test_a_boss_that_never_finished_takes_the_fallback(self, qw, boss_status):
        task = await qw.create_round_task(1)
        text_task = qw.text_task(SeedRole.TEXT_R1)
        qw.training[str(text_task.task_id)] = boss_status
        qw.hub[text_task.model_id] = "full"
        qw.tournaments[0].status = TournamentStatus.CANCELLED

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and task.model_id == text_task.model_id
        assert _source(qw, QuantSourceSlot.R1).source_is_fallback is True and qw.alerts.await_count == 1

    async def test_the_grace_period_does_not_apply(self, qw):
        task = await qw.create_round_task(1)
        text_task = qw.text_task(SeedRole.TEXT_R1)
        qw.land(SeedRole.TEXT_R1, "empty")
        qw.hub[text_task.model_id] = "full"
        qw.text_rows[SeedRole.TEXT_R1].updated_at = datetime.now(timezone.utc)
        qw.tournaments[0].status = TournamentStatus.COMPLETED

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and _source(qw, QuantSourceSlot.R1).source_is_fallback is True

    @pytest.mark.parametrize("status", [TournamentStatus.ACTIVE, TournamentStatus.PENDING])
    async def test_while_the_text_tournament_runs_everything_still_holds(self, qw, status):
        qw.tournaments[0].status = status
        text_task = qw.text_task(SeedRole.TEXT_R1)
        qw.hub[text_task.model_id] = "full"
        held = []
        for case in ("adapter awaiting merge", "failed on a replaceable row", "still training"):
            task = await qw.create_round_task(1) if not held else held[0]
            if case == "adapter awaiting merge":
                qw.land(SeedRole.TEXT_R1, "adapter")
            elif case == "failed on a replaceable row":
                qw.training[str(text_task.task_id)] = FAILURE
            else:
                qw.training[str(text_task.task_id)] = TRAINING
            held.append(task)

            await orchestrator._release_chained_task(task, qw.config)

            assert task.status == TaskStatus.AWAITING_UPSTREAM, case
        assert qw.alerts.await_count == 0 and _source(qw, QuantSourceSlot.R1).source_model_repo is None


class TestFallbackGrace:
    """A hub lookup right after the boss finished must not make the fallback permanent."""

    GRACE = t_cst.QUANT_SOURCE_FALLBACK_GRACE_SECONDS

    async def _setup(self, qw, updated_ago: float | None, output: str = "empty"):
        task = await qw.create_round_task(2)
        text_task = qw.text_task(SeedRole.TEXT_R2)
        qw.land(SeedRole.TEXT_R2, output)
        qw.hub[text_task.model_id] = "full"
        row = qw.text_rows[SeedRole.TEXT_R2]
        row.status = SeedStatus.TRAINED
        row.updated_at = None if updated_ago is None else datetime.now(timezone.utc) - timedelta(seconds=updated_ago)
        return task, text_task

    def test_the_grace_is_thirty_minutes(self):
        assert t_cst.QUANT_SOURCE_FALLBACK_GRACE_SECONDS == 30 * 60

    @pytest.mark.parametrize("output", ["empty", "adapter"])
    async def test_before_the_grace_the_task_holds_without_an_alert(self, qw, output):
        task, _ = await self._setup(qw, self.GRACE - 60, output)
        if output == "adapter":
            qw.text_rows[SeedRole.TEXT_R2].merged_output_repo = MERGED  # merged, but the merged repo is not on the hub yet

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.AWAITING_UPSTREAM and qw.alerts.await_count == 0
        assert _source(qw, QuantSourceSlot.R2).source_model_repo is None

    async def test_after_the_grace_the_task_takes_the_fallback(self, qw):
        task, text_task = await self._setup(qw, self.GRACE + 60)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.model_id == text_task.model_id and _source(qw, QuantSourceSlot.R2).source_is_fallback is True

    async def test_a_row_without_a_timestamp_counts_as_old_enough(self, qw):
        task, text_task = await self._setup(qw, None)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.model_id == text_task.model_id

    async def test_a_naive_timestamp_is_read_as_utc(self, qw):
        task, text_task = await self._setup(qw, self.GRACE + 60)
        row = qw.text_rows[SeedRole.TEXT_R2]
        row.updated_at = row.updated_at.replace(tzinfo=None)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.model_id == text_task.model_id

    async def test_a_usable_output_is_never_held_by_the_grace(self, qw):
        task, _ = await self._setup(qw, 10, output="full")

        await orchestrator._release_chained_task(task, qw.config)

        assert task.status == TaskStatus.PENDING and _source(qw, QuantSourceSlot.R2).source_is_fallback is False

    async def test_a_failed_run_on_a_final_seed_takes_the_fallback_at_once(self, qw):
        task = await qw.create_round_task(2)
        text_task = qw.text_task(SeedRole.TEXT_R2)
        qw.training[str(text_task.task_id)] = FAILURE
        qw.hub[text_task.model_id] = "full"
        row = qw.text_rows[SeedRole.TEXT_R2]
        row.status, row.updated_at = SeedStatus.TRAINED, datetime.now(timezone.utc)

        await orchestrator._release_chained_task(task, qw.config)

        assert task.model_id == text_task.model_id and _source(qw, QuantSourceSlot.R2).source_is_fallback is True

    def test_the_seed_row_timestamp_is_bumped_by_a_status_change_and_by_the_merge_record(self):
        for function in (seeds_sql.update_seed_status, seeds_sql.set_seed_merged_output_repo):
            assert "{cst.UPDATED_AT} = NOW()" in inspect.getsource(function)


# ------------------------------------------------------------------------ the adapter merge of the continuous seed


class TestContinuousAdapterMerge:
    async def test_a_continuous_chat_seed_whose_boss_output_is_an_adapter_is_merged_and_the_repo_stored(self, monkeypatch):
        world = MergeWorld(monkeypatch, role=SeedRole.TEXT_CONTINUOUS, tournament_type=TournamentType.TEXT)
        world.task = _chat_task(training_data="s3://bucket/prepped_train.json")
        world.task.task_id = world.seed.task_id

        await world.merge_cycle()

        assert world.seed.merged_output_repo == MERGED
        kwargs = world.dispatch.await_args.kwargs
        assert kwargs["task_type"] == TaskType.CHATTASK and kwargs["model_id"] == world.raw_repo
        assert kwargs["training_data_url"] == "s3://bucket/prepped_train.json" and kwargs["augmentation_config"] is None
        world.replace.assert_not_awaited()

    async def test_a_continuous_full_model_output_is_left_alone(self, monkeypatch):
        world = MergeWorld(monkeypatch, role=SeedRole.TEXT_CONTINUOUS, tournament_type=TournamentType.TEXT)
        world.task = _chat_task(training_data="s3://bucket/prepped_train.json")
        world.hub[world.raw_repo] = FULL_MODEL_FILES

        await world.merge_cycle()

        assert world.seed.merged_output_repo is None
        world.dispatch.assert_not_awaited()


# ---------------------------------------------------------------------------------------- independence of rows


class TestIndependence:
    async def test_the_release_and_creation_paths_write_nothing_of_the_text_tournament(self, qw, monkeypatch):
        await qw.create_round_task(1)
        before = qw.text_snapshot()

        text_task_ids = {str(t.task_id) for t in qw.text_tasks.values()}

        def forbidden(name):
            async def refuse(*args, **kwargs):
                # A round marks its own quant seed attached: that is a quant row. No text seed row may be written.
                if name == "mark_seed_attached" and str(args[0]) not in text_task_ids:
                    return None
                raise AssertionError(f"{name} must not be called by a quant task's creation or release")

            return refuse

        for name in (
            "update_seed_status",
            "mark_seed_replaced",
            "mark_seed_attached",
            "set_seed_merged_output_repo",
            "insert_seed_task",
        ):
            if name != "insert_seed_task":
                monkeypatch.setattr(seeds_sql, name, forbidden(name))
        await qw.create_round_task(2)
        qw.land(SeedRole.TEXT_R1)
        qw.land(SeedRole.TEXT_R2)
        for task in list(qw.env.tasks.values()):
            if getattr(task, "status", None) == TaskStatus.AWAITING_UPSTREAM and task.task_type in QUANT_TASK_TYPES:
                await orchestrator._release_chained_task(task, qw.config)

        assert qw.text_snapshot() == before
        written = set(qw.updates)
        assert written and not written & {t.task_id for t in qw.text_tasks.values()}

    async def test_the_assignments_expected_repos_and_training_rows_are_keyed_by_the_quant_task_id(self, qw, monkeypatch):
        seeds = await qw.create_seeds()
        text_ids = {str(t.task_id) for t in qw.text_tasks.values()}
        quant_ids = {str(seed.task_id) for seed in seeds.values()}
        assert quant_ids.isdisjoint(text_ids)
        assert (
            set(qw.env.assigned) == quant_ids and set(qw.env.expected) == quant_ids
        )  # nodes and expected repos of the quant tasks only

        # The challenger's training rows of an attached quant task are keyed by the quant task id too.
        added = []

        async def add_pairs(assignments, _db):
            added.extend(assignments)

        task = qw.quant_task(SeedRole.BOSS_R2_FREEFORM)
        task.status = TaskStatus.TRAINING
        qw.env.assigned[str(task.task_id)] = [EMISSION_BURN_HOTKEY, CHALLENGER]
        monkeypatch.setattr(tournament_sql, "add_tournament_task_hotkey_pairs_for_training", add_pairs)
        monkeypatch.setattr(
            tournament_sql, "get_training_status_for_task", AsyncMock(return_value={EMISSION_BURN_HOTKEY: SUCCESS})
        )
        monkeypatch.setattr(tournament_sql, "get_tournament_id_by_task_id", AsyncMock(return_value=QUANT_TID))
        task.created_at = datetime(2026, 1, 1)

        await orchestrator._add_challenger_rows(SeedRole.BOSS_R2_FREEFORM, task, qw.config)

        assert [a.task_id for a in added] == [str(task.task_id)] and [a.hotkey for a in added] == [CHALLENGER]
        assert str(task.task_id) not in text_ids

    def test_the_creation_and_release_code_names_no_text_task_write(self):
        for function in (
            task_creator._create_seed_input_round_task,
            seed_tasks._acquire_compression_task,
            seed_tasks.draw_compression_task,
            orchestrator._release_on_text_boss_model,
            orchestrator._text_boss_model,
            quant_sources.record_new_quant_task,
        ):
            source = inspect.getsource(function)
            writes = ["update_seed_status", "mark_seed_replaced", "set_seed_merged_output_repo"]
            if function is not task_creator._create_seed_input_round_task:  # it marks its own quant round seed attached
                writes.append("mark_seed_attached")
            for write in writes:
                assert write not in source, f"{function.__name__} calls {write}"

    def test_the_link_is_never_derived_from_a_repo_or_ds_name(self):
        for module in (quant_sources, sources_sql):
            source = inspect.getsource(module)
            assert (
                "continuous_quant_train_index" not in source
                and ".split(" not in source
                and "ds" not in re.findall(r"\bds\b", source)
            )
        assert "continuous_quant_train_index" not in inspect.getsource(orchestrator._text_boss_model)


# -------------------------------------------------------------------------------- the text side is unaffected


class TestTextSideIsUnaffected:
    async def test_a_failed_quant_task_chained_on_a_text_seed_leaves_the_text_seed_and_round_untouched(self, qw, monkeypatch):
        quant_task = await qw.create_round_task(1)
        text_task = qw.text_task(SeedRole.TEXT_R1)
        seed = qw.text_rows[SeedRole.TEXT_R1]
        qw.training[str(text_task.task_id)] = SUCCESS
        monkeypatch.setattr(
            tournament_sql,
            "get_training_status_for_task_and_hotkeys",
            AsyncMock(side_effect=lambda task_id, hotkeys, _db: {EMISSION_BURN_HOTKEY: qw.training.get(str(task_id))}),
        )
        monkeypatch.setattr(tm, "_majority_failure_blocks_completion", AsyncMock(return_value=False))
        text_task.status = TaskStatus.SUCCESS.value
        tournament_task = SimpleNamespace(task_id=text_task.task_id, tournament_id=TEXT_TID, round_id="text_round_001")

        status_before = await seed_tasks._evaluate_seed(seed, text_task, qw.config)
        completion_before = await tm.is_tourn_task_completed(tournament_task, text_task, qw.config)

        quant_task.status = TaskStatus.FAILURE  # the quant task dies, with its own training rows failing too
        qw.training[str(quant_task.task_id)] = FAILURE
        status_after = await seed_tasks._evaluate_seed(seed, text_task, qw.config)
        completion_after = await tm.is_tourn_task_completed(tournament_task, text_task, qw.config)
        await seed_tasks.process_tournament_seed_tasks(TEXT_TID, qw.config)

        assert status_before == status_after == (SeedStatus.TRAINED, None)  # the boss trained it; the round has not finished
        assert completion_before == completion_after == (True, "Task completed successfully")
        assert text_task.status == TaskStatus.SUCCESS.value and qw.text_rows[SeedRole.TEXT_R1].attempt == 1
        assert await seed_tasks.seed_halt_reason(TEXT_TID, None) is None

    async def test_the_text_seed_evaluation_does_not_read_any_quant_state(self):
        for function in (seed_tasks._evaluate_seed, seed_tasks._evaluate_text_seed):
            source = inspect.getsource(function)
            assert "quant_sources" not in source and "quant_task" not in source

    async def test_a_text_seed_replacement_is_not_blocked_by_a_quant_task_chained_on_it(self, qw):
        await qw.create_round_task(3)
        old_text = qw.text_task(SeedRole.TEXT_R3)
        old_text.status = TaskStatus.PREP_TASK_FAILURE.value

        new = await seed_tasks.replace_seed(qw.text_rows[SeedRole.TEXT_R3], "prep", qw.config, prep_rejected=True)

        assert new is not None and new.attempt == 2 and new.tournament_id == TEXT_TID
