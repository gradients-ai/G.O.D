"""Seed tasks: creation of the five roles, the boss pin, task filters, hours routing, trainer priority, the
boss-round hold, and the replacement ladder with its halt."""

import inspect
import random
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import UUID
from uuid import uuid4

import pytest

import validator.db.constants as db_cst
from core.constants.quant import QUANT_MODEL_SIZE_B
from core.models.quant_models import LAUNCH_QUANT_FORMATS
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.db.sql import submissions_and_scoring as submissions_and_scoring_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tasks.models import InstructTextRawTask
from validator.tasks.synthetics.scheduler import draw_quant_format
from validator.tournament import constants as t_cst
from validator.tournament import orchestrator
from validator.tournament import seed_tasks
from validator.tournament import task_creator
from validator.tournament import tournament_manager
from validator.tournament.models import ANY_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import INPUT_SEED_ROLES
from validator.tournament.models import TEXT_ROUND_SEED_ROLES
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentTaskTraining
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingRepoInfo
from validator.tournament.models import TrainingStatus


TID = "tourn_seed_test"
# The five seed roles of a role-seeded quant tournament. Quant tournaments are seeded with six compressions of the text boss's
# models instead; the fixture below creates the role-seeded kind so that its rows are still evaluated, replaced and halted.
LEGACY_SEED_CREATION_ORDER = (
    SeedRole.INPUT_SMALL,
    SeedRole.INPUT_LARGE,
    SeedRole.BOSS_INSTRUCT,
    SeedRole.BOSS_CHAINED_FREEFORM,
    SeedRole.BOSS_CHAINED_FIXED,
)
LEGACY_BOSS_ROUND_SEED_ROLES = frozenset({SeedRole.BOSS_CHAINED_FREEFORM, SeedRole.BOSS_CHAINED_FIXED})
BOSS_REPO = "https://github.com/org/boss-backup"
BOSS_COMMIT = "a" * 40
PINNED = TrainingRepoInfo(
    training_repo=BOSS_REPO, training_commit_hash=BOSS_COMMIT, github_token=None, requested_datasets=None
)


class FakeSeedStore:
    """In-memory stand-in for validator.db.sql.tournament_seeds."""

    def __init__(self) -> None:
        self.rows: list[TournamentSeedTask] = []
        self.attached: set[UUID] = set()

    async def insert_seed_task(self, seed, _db):
        self.rows.append(seed.model_copy())
        return seed

    async def get_latest_seed_tasks(self, tournament_id, _db):
        latest = {}
        for row in sorted(self.rows, key=lambda r: r.attempt):
            if row.tournament_id == tournament_id:
                latest[row.role] = row
        return sorted(latest.values(), key=lambda r: r.role.value)

    async def get_active_seed_tasks(self, tournament_id, db):
        live = {SeedStatus.TRAINING, SeedStatus.TRAINED, SeedStatus.READY}
        return [row for row in await self.get_latest_seed_tasks(tournament_id, db) if row.status in live]

    async def get_failed_seed_tasks(self, tournament_id, db):
        return [row for row in await self.get_latest_seed_tasks(tournament_id, db) if row.status == SeedStatus.FAILED]

    async def update_seed_status(self, tournament_id, role, attempt, status, _db):
        for row in self.rows:
            if (row.tournament_id, row.role, row.attempt) == (tournament_id, role, attempt):
                row.status = status

    async def get_seed_task_history(self, tournament_id, _db):
        return sorted((r for r in self.rows if r.tournament_id == tournament_id), key=lambda r: (r.role.value, r.attempt))

    async def mark_seed_replaced(self, tournament_id, role, attempt, prep_rejected, _db):
        for row in self.rows:
            if (row.tournament_id, row.role, row.attempt) == (tournament_id, role, attempt):
                row.status, row.prep_rejected = SeedStatus.REPLACED, prep_rejected

    async def get_seed_task_by_task_id(self, task_id, _db):
        return next((row for row in self.rows if str(row.task_id) == str(task_id)), None)

    async def get_seed_roles_by_task_ids(self, task_ids, _db):
        return {str(row.task_id): row.role for row in self.rows if str(row.task_id) in set(task_ids)}

    async def is_task_attached_to_round(self, task_id, _db):
        return UUID(str(task_id)) in self.attached

    async def mark_seed_attached(self, task_id, round_id, _db):
        for row in self.rows:
            if row.task_id == UUID(str(task_id)) and row.attached_round_id is None:
                row.attached_round_id = round_id

    async def role_ever_attached(self, tournament_id, role, _db):
        return any(
            row.tournament_id == tournament_id
            and row.role == role
            and (row.attached_round_id is not None or row.task_id in self.attached)
            for row in self.rows
        )


def _instruct(**overrides) -> InstructTextRawTask:
    fields = dict(
        task_id=uuid4(),
        model_id="org/base",
        ds="org/ds",
        field_system="sys",
        field_instruction="instruction",
        field_input="input",
        field_output="output",
        format="{instruction} {input}",
        no_input_format="{instruction}",
        status=TaskStatus.PENDING.value,
        is_organic=False,
        account_id=uuid4(),
        hours_to_complete=6.0,
        created_at=datetime(2026, 1, 1),
        model_params_count=3_000_000_000,
    )
    fields.update(overrides)
    return InstructTextRawTask(**fields)


def _quant(task_type, **overrides):
    fields = dict(
        task_id=uuid4(),
        task_type=task_type,
        model_id="org/base",
        ds="org/ds",
        status=TaskStatus.PENDING,
        upstream_task_id=None,
        training_start_point=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def env(monkeypatch):
    store = FakeSeedStore()
    tasks: dict = {}
    assigned: dict = {}
    ns = SimpleNamespace(
        store=store,
        tasks=tasks,
        assigned=assigned,
        pool_calls=[],
        alerts=AsyncMock(),
        repoint=MagicMock(),
    )

    def register(task):
        tasks[task.task_id] = task
        return task

    async def make_instruct(config, models, datasets, **kwargs):
        ns.instruct_calls.append((models, kwargs))
        return register(_instruct())

    async def make_chained(config, task_type, upstream, *, quant_format, boss=True):
        ns.chained_calls.append((task_type, upstream, quant_format, boss))
        return register(_quant(task_type, upstream_task_id=upstream.task_id, status=TaskStatus.AWAITING_UPSTREAM))

    ns.instruct_calls, ns.chained_calls = [], []

    def text_models(_keypair, low, high):
        ns.pool_calls.append((low, high))
        return f"pool-{low}-{high}"

    for name in (
        "insert_seed_task",
        "get_latest_seed_tasks",
        "get_active_seed_tasks",
        "get_failed_seed_tasks",
        "update_seed_status",
        "get_seed_task_history",
        "mark_seed_replaced",
        "get_seed_task_by_task_id",
        "get_seed_roles_by_task_ids",
        "is_task_attached_to_round",
        "mark_seed_attached",
        "role_ever_attached",
    ):
        monkeypatch.setattr(seeds_sql, name, getattr(store, name))
    monkeypatch.setattr(seed_tasks, "SEED_CREATION_ORDER", LEGACY_SEED_CREATION_ORDER)
    monkeypatch.setattr(seed_tasks, "_get_text_models", text_models)
    monkeypatch.setattr(seed_tasks, "_get_instruct_text_datasets", MagicMock(return_value="datasets"))
    monkeypatch.setattr(seed_tasks, "filter_quant_compatible", lambda pool: pool)
    monkeypatch.setattr(seed_tasks, "create_synthetic_instruct_text_task", make_instruct)
    monkeypatch.setattr(seed_tasks, "create_chained_quant_task", make_chained)
    monkeypatch.setattr(seed_tasks, "repoint_chained_quant_task", ns.repoint)
    monkeypatch.setattr(seed_tasks, "send_to_discord", ns.alerts)
    monkeypatch.setattr(seed_tasks, "_SEED_HALT_ALERTED", set())
    monkeypatch.setattr(seed_tasks, "_SEED_ISSUE_ALERTED", set())
    monkeypatch.setattr(seed_tasks, "get_node_by_hotkey", AsyncMock(return_value=SimpleNamespace(hotkey=EMISSION_BURN_HOTKEY)))
    monkeypatch.setattr(tournament_sql, "get_tournament_training_repo_and_commit", AsyncMock(return_value=PINNED))

    async def get_task(task_id, _db, connection=None):
        return tasks.get(task_id)

    async def nodes_assigned(task_id, _db):
        return [SimpleNamespace(hotkey=hotkey) for hotkey in assigned.get(str(task_id), [])]

    async def assign(task_id, node, _db):
        assigned.setdefault(str(task_id), []).append(node.hotkey)

    ns.expected = {}

    async def set_repo(task_id, node, _db, name):
        ns.expected[str(task_id)] = name

    async def with_status(status, _db, **kwargs):
        return [task for task in tasks.values() if task.status == status]

    monkeypatch.setattr(task_sql, "get_task", get_task)
    monkeypatch.setattr(task_sql, "get_nodes_assigned_to_task", nodes_assigned)
    monkeypatch.setattr(task_sql, "assign_node_to_task", assign)
    monkeypatch.setattr(task_sql, "set_expected_repo_name", set_repo)
    monkeypatch.setattr(task_sql, "update_task", AsyncMock())
    monkeypatch.setattr(task_sql, "get_tasks_with_status", with_status)
    ns.config = SimpleNamespace(psql_db=None, keypair=None, discord_url="http://discord")
    return ns


def _live(env) -> dict[SeedRole, TournamentSeedTask]:
    return {row.role: row for row in env.store.rows if row.status != SeedStatus.REPLACED}


class TestCreation:
    async def test_creates_the_five_roles_in_order_and_nothing_else(self, env):
        seeds = await seed_tasks.create_seed_tasks(TID, env.config)

        assert [seed.role for seed in seeds] == list(seed_tasks.SEED_CREATION_ORDER)
        assert list(seed_tasks.SEED_CREATION_ORDER) == [
            SeedRole.INPUT_SMALL,
            SeedRole.INPUT_LARGE,
            SeedRole.BOSS_INSTRUCT,
            SeedRole.BOSS_CHAINED_FREEFORM,
            SeedRole.BOSS_CHAINED_FIXED,
        ]
        assert {row.role for row in env.store.rows} == INPUT_SEED_ROLES | {SeedRole.BOSS_INSTRUCT} | LEGACY_BOSS_ROUND_SEED_ROLES
        assert len(env.store.rows) == 5 and len(env.instruct_calls) == 3 and len(env.chained_calls) == 2
        assert all(seed.status == SeedStatus.TRAINING and seed.attempt == 1 for seed in seeds)

    async def test_instruct_seeds_use_their_bands_and_flags(self, env):
        await seed_tasks.create_seed_tasks(TID, env.config)

        assert env.pool_calls == [t_cst.SEED_INPUT_SMALL_SIZE_B, t_cst.SEED_INPUT_LARGE_SIZE_B, tuple(QUANT_MODEL_SIZE_B)]
        small, large, boss = env.instruct_calls
        for _, kwargs in (small, large):
            assert kwargs == dict(enable_kl=False, allow_augmentation=True, carve_calibration=True)
        # The boss input carves a calibration set too: both compression seeds inherit it at release and publish it.
        assert boss[1] == dict(enable_kl=False, allow_augmentation=False, carve_calibration=True)

    async def test_no_seed_task_enables_kl(self, env):
        await seed_tasks.create_seed_tasks(TID, env.config)

        assert env.instruct_calls and all(kwargs["enable_kl"] is False for _, kwargs in env.instruct_calls)

    async def test_large_band_falls_back_to_small_with_one_constant(self, env, monkeypatch):
        monkeypatch.setattr(t_cst, "SEED_INPUT_LARGE_FALLS_BACK_TO_SMALL", True)
        await seed_tasks.create_seed_tasks(TID, env.config)

        assert env.pool_calls[:2] == [t_cst.SEED_INPUT_SMALL_SIZE_B, t_cst.SEED_INPUT_SMALL_SIZE_B]

    async def test_compression_seeds_are_chained_on_the_boss_instruct_seed_with_a_seeded_format(self, env):
        seeds = {seed.role: seed for seed in await seed_tasks.create_seed_tasks(TID, env.config)}

        by_type = {task_type: (upstream, fmt, boss) for task_type, upstream, fmt, boss in env.chained_calls}
        assert set(by_type) == {TaskType.QUANTFREEFORMTASK, TaskType.QUANTFIXEDTASK}
        assert by_type[TaskType.QUANTFREEFORMTASK][1] is None
        assert all(str(upstream.task_id) == str(seeds[SeedRole.BOSS_INSTRUCT].task_id) for upstream, _, _ in by_type.values())
        expected = draw_quant_format(random.Random(f"{TID}:seed:quant_format:{TaskType.QUANTFIXEDTASK.value}"))
        assert by_type[TaskType.QUANTFIXEDTASK][1] == expected
        for role in LEGACY_BOSS_ROUND_SEED_ROLES:
            task = env.tasks[seeds[role].task_id]
            assert task.status == TaskStatus.AWAITING_UPSTREAM
            assert str(task.upstream_task_id) == str(seeds[SeedRole.BOSS_INSTRUCT].task_id)

    async def test_both_players_share_one_hours_budget_on_a_compression_seed(self, env):
        await seed_tasks.create_seed_tasks(TID, env.config)

        assert all(boss is True for *_, boss in env.chained_calls)

    async def test_the_fixed_format_is_a_launch_format_deterministic_per_tournament(self, env):
        await seed_tasks.create_seed_tasks(TID, env.config)

        chained = next(fmt for task_type, _, fmt, _ in env.chained_calls if task_type == TaskType.QUANTFIXEDTASK)
        assert chained in LAUNCH_QUANT_FORMATS
        assert seed_tasks._seed_quant_format("t7", TaskType.QUANTFIXEDTASK) == seed_tasks._seed_quant_format(
            "t7", TaskType.QUANTFIXEDTASK
        )

    async def test_no_continuous_chunk_is_drawn(self, env):
        await seed_tasks.create_seed_tasks(TID, env.config)

        assert not hasattr(seed_tasks, "_tournament_chunk")
        assert not hasattr(seed_tasks, "_next_continuous_quant_index")

    async def test_every_seed_is_assigned_to_the_boss_with_its_expected_repo_and_pin(self, env):
        for seed in await seed_tasks.create_seed_tasks(TID, env.config):
            assert env.assigned[str(seed.task_id)] == [EMISSION_BURN_HOTKEY]
            assert env.expected[str(seed.task_id)] == f"tournament-{TID}-{seed.task_id}-{EMISSION_BURN_HOTKEY[:8]}"
            assert (seed.boss_repo, seed.boss_commit) == (BOSS_REPO, BOSS_COMMIT)

    async def test_idempotent_per_role(self, env):
        first = await seed_tasks.create_seed_tasks(TID, env.config)
        second = await seed_tasks.create_seed_tasks(TID, env.config)

        assert [s.task_id for s in first] == [s.task_id for s in second]
        assert len(env.store.rows) == 5 and len(env.instruct_calls) == 3 and len(env.chained_calls) == 2

    async def test_a_missing_role_is_filled_on_retry_and_existing_ones_are_kept(self, env):
        await seed_tasks.create_seed_tasks(TID, env.config)
        dropped = env.store.rows.pop()  # the fixed compression
        env.assigned.pop(str(dropped.task_id))

        seeds = await seed_tasks.create_seed_tasks(TID, env.config)

        assert len(seeds) == 5 and len(env.chained_calls) == 3 and len(env.instruct_calls) == 3


class TestOnlyTheQuantTournamentIsSeeded:
    async def test_seeds_are_processed_for_active_quant_and_text_tournaments_only(self, env, monkeypatch):
        tournaments = [
            SimpleNamespace(tournament_id=f"{kind.value}-1", tournament_type=kind)
            for kind in (TournamentType.TEXT, TournamentType.IMAGE, TournamentType.QUANT)
        ]
        monkeypatch.setattr(tournament_sql, "get_tournaments_with_status", AsyncMock(return_value=tournaments))
        process = AsyncMock()
        monkeypatch.setattr(seed_tasks, "process_tournament_seed_tasks", process)

        await seed_tasks.process_seed_tasks(env.config)

        assert [call.args for call in process.await_args_list] == [("text-1", env.config), ("quant-1", env.config)]

    def test_the_legacy_text_and_image_task_creation_never_reads_seeds(self):
        # create_text_tournament_tasks is not listed: it asks whether the tournament is seeded and attaches the seed if so.
        for creator in (
            task_creator._create_new_text_boss_round_tasks,
            task_creator.create_image_tournament_tasks,
            task_creator._create_new_image_boss_round_tasks,
        ):
            assert "seeds_sql" not in inspect.getsource(creator)
        # replace_tournament_task asks the one question "is this a seeded text tournament" (a text tournament without seed
        # rows answers no and keeps the legacy redraw); it reads no other seed state.
        used = set(re.findall(r"seeds_sql\.(\w+)", inspect.getsource(task_creator.replace_tournament_task)))
        assert used == {"is_seeded_text_tournament"}


class TestPin:
    async def test_unset_commit_is_pinned_from_head(self, env, monkeypatch):
        unpinned = PINNED._replace(training_commit_hash=None)
        monkeypatch.setattr(tournament_sql, "get_tournament_training_repo_and_commit", AsyncMock(return_value=unpinned))
        monkeypatch.setattr(seed_tasks, "get_latest_commit_hash_from_github", AsyncMock(return_value="b" * 40))
        persist = AsyncMock()
        monkeypatch.setattr(tournament_sql, "set_tournament_participant_training_commit", persist)

        info = await seed_tasks._boss_training_info(TID, env.config)

        assert info.training_commit_hash == "b" * 40
        persist.assert_awaited_once_with(TID, EMISSION_BURN_HOTKEY, "b" * 40, None)

    async def test_boss_row_returns_its_pinned_commit(self):
        row = {
            db_cst.TRAINING_REPO: BOSS_REPO,
            db_cst.TRAINING_COMMIT_HASH: BOSS_COMMIT,
            db_cst.GITHUB_TOKEN: None,
            db_cst.BACKUP_REPO: None,
            db_cst.REQUESTED_DATASETS: None,
        }
        db = _FakeDb(fetchrow=[row])

        info = await tournament_sql.get_tournament_training_repo_and_commit(EMISSION_BURN_HOTKEY, TID, db)

        assert (info.training_repo, info.training_commit_hash) == (BOSS_REPO, BOSS_COMMIT)


class _FakeConn:
    def __init__(self, queries, fetchrow, fetch=()):
        self.queries, self._fetchrow, self._fetch = queries, list(fetchrow), list(fetch)

    async def fetch(self, query, *args):
        self.queries.append(query)
        return self._fetch.pop(0) if self._fetch else []

    async def fetchrow(self, query, *args):
        self.queries.append(query)
        return self._fetchrow.pop(0) if self._fetchrow else None

    async def fetchval(self, query, *args):
        self.queries.append(query)
        return self._fetchrow.pop(0) if self._fetchrow else None


class _FakeDb:
    def __init__(self, fetchrow=(), fetch=()):
        self.queries: list[str] = []
        self._fetchrow = fetchrow
        self._fetch = fetch

    async def connection(self):
        conn = _FakeConn(self.queries, self._fetchrow, self._fetch)

        class _Ctx:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


class TestTournamentFilters:
    @pytest.mark.parametrize("mode", ["only", "exclude"])
    async def test_task_list_filter_covers_the_seed_table(self, mode):
        db = _FakeDb()

        await task_sql.get_tasks_with_status(TaskStatus.LOOKING_FOR_NODES, db, tournament_filter=mode)

        query = db.queries[0]
        assert "tournament_seed_tasks" in query and "tournament_tasks" in query
        assert ("NOT IN (" if mode == "exclude" else "AND task_id IN (") in query

    async def test_tournament_id_lookup_falls_through_to_the_seed_table(self):
        db = _FakeDb(fetchrow=[None, None, {db_cst.TOURNAMENT_ID: TID}])

        assert await tournament_sql.get_tournament_id_by_task_id(str(uuid4()), db) == TID
        assert "tournament_seed_tasks" in db.queries[-1]


class TestHoursRouting:
    @pytest.mark.parametrize(
        ("role", "non_boss", "boss"),
        [(role, role in INPUT_SEED_ROLES | TEXT_ROUND_SEED_ROLES, role in ANY_BOSS_ROUND_SEED_ROLES) for role in SeedRole],
    )
    async def test_seed_roles_route_hours_without_a_round(self, env, role, non_boss, boss):
        task = _instruct()
        env.store.rows.append(TournamentSeedTask(tournament_id=TID, role=role, task_id=task.task_id))

        assert await orchestrator._is_non_boss_tournament_task(task, env.config) is non_boss
        assert await orchestrator._is_boss_round_task(task, env.config) is boss


class TestTrainerPriority:
    @pytest.fixture
    def scheduling(self, env, monkeypatch):
        order: list = []
        gpu_lookups: list = []

        async def create_request(task, hotkey, *args):
            order.append(task.task_id)
            return SimpleNamespace()

        async def check_gpus(config, required, excluded_trainers=frozenset()):
            gpu_lookups.append(1)
            if ns.no_gpus:
                return None
            if ns.free_gpus is not None:
                if required.value > ns.free_gpus:
                    return None
                ns.free_gpus -= required.value
            return ("1.2.3.4", list(range(required.value)), "NVLINK")

        def requirement(task):
            return SimpleNamespace(value=ns.needs.get(task.task_id, 1))

        # free_gpus None = unlimited; needs maps task_id -> GPUs the row requires (default 1)
        ns = SimpleNamespace(order=order, gpu_lookups=gpu_lookups, no_gpus=False, free_gpus=None, needs={})
        monkeypatch.setattr(orchestrator, "get_tournament_id_by_task_id", AsyncMock(return_value=TID))
        monkeypatch.setattr(orchestrator, "_update_all_trainers_gpu_availability", AsyncMock())
        monkeypatch.setattr(orchestrator, "_reference_ready", AsyncMock(return_value=True))
        monkeypatch.setattr(orchestrator, "get_task_gpu_requirement", MagicMock(side_effect=lambda task: requirement(task)))
        monkeypatch.setattr(orchestrator, "_check_suitable_gpus", check_gpus)
        monkeypatch.setattr(orchestrator, "_create_training_request", create_request)
        monkeypatch.setattr(orchestrator, "start_training_task", AsyncMock(return_value=True))
        monkeypatch.setattr(orchestrator.tournament_sql, "update_tournament_task_training_status", AsyncMock())
        monkeypatch.setattr(orchestrator.tournament_sql, "update_gpu_availability", AsyncMock())
        monkeypatch.setattr(orchestrator.gpu_cost_sql, "start_cost_run", AsyncMock())
        monkeypatch.setattr(orchestrator.asyncio, "sleep", AsyncMock())
        ns.env = env
        return ns

    def _rows(self, env, roles):
        rows = []
        for role in roles:
            task = _instruct()
            if role is not None:
                env.store.rows.append(TournamentSeedTask(tournament_id=TID, role=role, task_id=task.task_id))
            rows.append(
                TournamentTaskTraining(
                    task=task,
                    hotkey="5Miner" if role is None else EMISSION_BURN_HOTKEY,
                    training_status=TrainingStatus.PENDING,
                    n_training_attempts=0,
                    created_at=datetime(2026, 1, 1),
                    updated_at=datetime(2026, 1, 1),
                    training_repo="https://github.com/x/r",
                    training_commit_hash="c" * 40,
                )
            )
        return rows

    def test_rank_order_for_every_role(self):
        rank = seed_tasks.training_rank
        ordered = [
            rank(SeedRole.INPUT_SMALL),
            rank(SeedRole.INPUT_LARGE),
            rank(None),
            rank(SeedRole.BOSS_INSTRUCT),
        ]
        assert ordered == [0, 1, 2, 3]
        for role in (SeedRole.BOSS_CHAINED_FREEFORM, SeedRole.BOSS_CHAINED_FIXED):
            assert rank(role) == seed_tasks.DEFERRABLE_BOSS_SEED_TRAINING_RANK == 4

    def test_the_boss_entry_check_is_the_boss_hotkey_at_a_boss_rank_only(self):
        is_entry = seed_tasks.is_boss_entry_row
        burn, challenger = EMISSION_BURN_HOTKEY, "5Challenger"
        for role in (SeedRole.BOSS_INSTRUCT, SeedRole.BOSS_CHAINED_FREEFORM, SeedRole.BOSS_CHAINED_FIXED):
            assert is_entry(seed_tasks.training_rank(role), burn) and not is_entry(seed_tasks.training_rank(role), challenger)
        for role in (SeedRole.INPUT_SMALL, SeedRole.INPUT_LARGE, None):  # incl. an organic task trained under the burn hotkey
            assert not is_entry(seed_tasks.training_rank(role), burn)
        assert not hasattr(seed_tasks, "row_training_rank")

    async def test_activation_dispatches_i1_i4_b_in_one_cycle_in_order(self, scheduling):
        env = scheduling.env
        scheduling.free_gpus = 3
        rows = self._rows(env, [SeedRole.BOSS_INSTRUCT, SeedRole.INPUT_LARGE, SeedRole.INPUT_SMALL])
        i1, i4, b = rows[2].task.task_id, rows[1].task.task_id, rows[0].task.task_id

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        assert scheduling.order == [i1, i4, b]
        assert not rows

    async def test_the_boss_input_and_a_compression_start_at_activation_when_gpus_are_free(self, scheduling):
        env = scheduling.env
        rows = self._rows(env, [SeedRole.BOSS_CHAINED_FIXED, SeedRole.BOSS_INSTRUCT, SeedRole.INPUT_SMALL])
        i1, mba, compression = rows[2].task.task_id, rows[1].task.task_id, rows[0].task.task_id

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        assert scheduling.order == [i1, mba, compression]

    async def test_one_free_gpu_goes_to_i1_only(self, scheduling):
        env = scheduling.env
        scheduling.free_gpus = 1
        rows = self._rows(env, [SeedRole.BOSS_INSTRUCT, SeedRole.INPUT_LARGE, SeedRole.INPUT_SMALL])
        i1 = rows[2].task.task_id

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        assert scheduling.order == [i1]

    @pytest.mark.parametrize("role", [SeedRole.BOSS_INSTRUCT, SeedRole.BOSS_CHAINED_FIXED])
    async def test_a_boss_entry_yields_to_a_starved_miner_row(self, scheduling, role):
        env = scheduling.env
        scheduling.free_gpus = 1
        rows = self._rows(env, [role, None])
        miner = rows[1].task.task_id
        scheduling.needs[miner] = 2  # the miner row is short of GPUs

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        assert scheduling.order == []  # the entry did not take the free GPU the miner row could not use

    async def test_a_miner_row_goes_before_mba(self, scheduling):
        env = scheduling.env
        scheduling.free_gpus = 1
        rows = self._rows(env, [SeedRole.BOSS_INSTRUCT, None])
        mba, miner = rows[0].task.task_id, rows[1].task.task_id

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        assert scheduling.order == [miner]
        assert mba not in scheduling.order

    async def test_a_starved_boss_input_does_not_stop_the_compression_when_it_fits(self, scheduling):
        env = scheduling.env
        scheduling.free_gpus = 2
        rows = self._rows(env, [SeedRole.BOSS_CHAINED_FIXED, SeedRole.BOSS_INSTRUCT, SeedRole.INPUT_SMALL])
        compression, mba = rows[0].task.task_id, rows[1].task.task_id
        scheduling.needs[mba] = 4  # the boss instruct task is starved: no trainer has 4 free

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        assert scheduling.order[1:] == [compression]  # after the small input seed; the boss task fell through, none skipped

    async def test_starved_i1_does_not_stop_i4_taking_the_free_gpu(self, scheduling):
        env = scheduling.env
        scheduling.free_gpus = 1
        rows = self._rows(env, [SeedRole.INPUT_LARGE, SeedRole.INPUT_SMALL, None])
        i4, i1 = rows[0].task.task_id, rows[1].task.task_id
        scheduling.needs[i1] = 2

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        assert scheduling.order == [i4]
        assert not rows  # the small input seed and the miner row were starved and fell through, nothing blocked

    async def test_order_is_i1_i4_miner_rows_mba_compressions(self, scheduling):
        env = scheduling.env
        roles_in = [
            SeedRole.BOSS_CHAINED_FIXED, None, SeedRole.BOSS_INSTRUCT,
            SeedRole.INPUT_LARGE, SeedRole.BOSS_CHAINED_FREEFORM, SeedRole.INPUT_SMALL, None,
        ]
        rows = self._rows(env, roles_in)
        by_id = {row.task.task_id: role for row, role in zip(rows, roles_in)}

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        roles = [by_id[task_id] for task_id in scheduling.order]
        assert roles[:5] == [SeedRole.INPUT_SMALL, SeedRole.INPUT_LARGE, None, None, SeedRole.BOSS_INSTRUCT]
        assert set(roles[5:]) == {SeedRole.BOSS_CHAINED_FREEFORM, SeedRole.BOSS_CHAINED_FIXED}

    async def test_boss_entries_are_skipped_when_a_miner_row_found_no_gpus_but_challenger_rows_are_not(self, scheduling):
        env = scheduling.env
        scheduling.no_gpus = True
        rows = self._rows(env, [SeedRole.BOSS_CHAINED_FIXED, SeedRole.BOSS_INSTRUCT, None, SeedRole.INPUT_SMALL])
        challenger_row = self._rows(env, [SeedRole.BOSS_INSTRUCT])[0].model_copy(update={"hotkey": "5Challenger"})
        rows.append(challenger_row)

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        # the small input seed and the miner row looked; the two boss entries did not; the challenger's boss instruct row
        # (rank 3) did, after them.
        assert scheduling.gpu_lookups == [1, 1, 1]

    async def test_a_starved_miner_row_defers_a_boss_compression_but_never_a_challenger_row(self, scheduling):
        env = scheduling.env
        scheduling.free_gpus = 2
        rows = self._rows(env, [SeedRole.BOSS_CHAINED_FIXED, None, SeedRole.BOSS_CHAINED_FIXED])
        boss_row, miner = rows[0].task.task_id, rows[1].task.task_id
        challenger_row = rows[2].model_copy(update={"hotkey": "5Challenger"})  # same rank as the boss's compression row
        rows[2] = challenger_row
        scheduling.needs[miner] = 3  # the miner row is short of GPUs

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        assert scheduling.order == [challenger_row.task.task_id]  # the challenger ran; the boss's own entry yielded
        assert boss_row not in scheduling.order

    @pytest.mark.parametrize("role", [SeedRole.BOSS_INSTRUCT, SeedRole.BOSS_CHAINED_FIXED])
    async def test_a_challenger_row_ties_with_the_boss_row_of_the_same_task_and_goes_oldest_first(self, scheduling, role):
        env = scheduling.env
        scheduling.free_gpus = 1
        old_boss = self._rows(env, [role])[0].model_copy(update={"created_at": datetime(2026, 1, 1)})
        new_challenger = self._rows(env, [role])[0]
        new_challenger = new_challenger.model_copy(update={"hotkey": "5Challenger", "created_at": datetime(2026, 1, 2)})
        rows = [new_challenger, old_boss]  # the query returns newest first

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        assert scheduling.order == [old_boss.task.task_id]

        # The same with the roles of old and new swapped: the older row (now the challenger's) goes first.
        env2_rows = [
            old_boss.model_copy(update={"created_at": datetime(2026, 1, 3)}),
            new_challenger.model_copy(update={"created_at": datetime(2026, 1, 1)}),
        ]
        scheduling.order.clear()
        scheduling.free_gpus = 1
        await orchestrator.schedule_tasks_for_training(env2_rows, env.config)
        assert scheduling.order == [new_challenger.task.task_id]

    async def test_challenger_mba_does_not_wait_behind_a_boss_compression(self, scheduling):
        """Rank is the task's: the challenger's boss instruct row (rank 3) goes before the boss's compressions (4)."""
        env = scheduling.env
        scheduling.free_gpus = 1
        compression, other_compression = self._rows(env, [SeedRole.BOSS_CHAINED_FREEFORM, SeedRole.BOSS_CHAINED_FIXED])
        challenger_mba = self._rows(env, [SeedRole.BOSS_INSTRUCT])[0].model_copy(update={"hotkey": "5Challenger"})
        rows = [compression, other_compression, challenger_mba]

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        assert scheduling.order == [challenger_mba.task.task_id]

    async def test_an_unfinished_mba_goes_ahead_of_the_challengers_compression_rows(self, scheduling):
        """Once the boss round exists there are no miner rows: the boss instruct task still pending goes before the challenger's
        compression rows (rank 3 against 4), whoever's row it is."""
        env = scheduling.env
        scheduling.free_gpus = 1
        boss_mba = self._rows(env, [SeedRole.BOSS_INSTRUCT])[0]
        challenger_compression = self._rows(env, [SeedRole.BOSS_CHAINED_FIXED])[0]
        challenger_compression = challenger_compression.model_copy(update={"hotkey": "5Challenger"})
        rows = [challenger_compression, boss_mba]

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        assert scheduling.order == [boss_mba.task.task_id]

    async def test_boss_round_seeds_alone_still_train(self, scheduling):
        env = scheduling.env
        rows = self._rows(env, [SeedRole.BOSS_INSTRUCT, SeedRole.BOSS_CHAINED_FREEFORM])

        await orchestrator.schedule_tasks_for_training(rows, env.config)

        assert len(scheduling.order) == 2

class TestBossRoundHold:
    @pytest.fixture
    def hold(self, env, monkeypatch):
        statuses = {}
        losses = []
        monkeypatch.setattr(tournament_sql, "get_training_status_for_task", AsyncMock(side_effect=lambda *_: statuses))
        monkeypatch.setattr(submissions_and_scoring_sql, "get_task_node_losses", AsyncMock(side_effect=lambda *_: losses))
        return SimpleNamespace(env=env, statuses=statuses, losses=losses)

    def _seed(self, env, role, hotkeys, attached=True):
        task = _instruct()
        env.store.rows.append(TournamentSeedTask(tournament_id=TID, role=role, task_id=task.task_id))
        env.assigned[str(task.task_id)] = hotkeys
        if attached:
            env.store.attached.add(task.task_id)
        return task.task_id

    async def test_non_seed_input_seeds_and_the_boss_input_are_never_held(self, hold):
        env = hold.env
        input_id = self._seed(env, SeedRole.INPUT_SMALL, [EMISSION_BURN_HOTKEY], attached=False)
        boss_input_id = self._seed(env, SeedRole.BOSS_INSTRUCT, [EMISSION_BURN_HOTKEY], attached=False)

        assert await seeds_sql.boss_round_seed_ready_to_finalise(input_id, None)
        assert await seeds_sql.boss_round_seed_ready_to_finalise(boss_input_id, None)
        assert await seeds_sql.boss_round_seed_ready_to_finalise(uuid4(), None)

    async def test_unattached_boss_round_seed_is_held(self, hold):
        task_id = self._seed(hold.env, SeedRole.BOSS_CHAINED_FREEFORM, [EMISSION_BURN_HOTKEY], attached=False)
        hold.statuses[EMISSION_BURN_HOTKEY] = "success"

        assert not await seeds_sql.boss_round_seed_ready_to_finalise(task_id, None)

    async def test_attached_but_no_challenger_is_held(self, hold):
        task_id = self._seed(hold.env, SeedRole.BOSS_CHAINED_FREEFORM, [EMISSION_BURN_HOTKEY])
        hold.statuses[EMISSION_BURN_HOTKEY] = "success"

        assert not await seeds_sql.boss_round_seed_ready_to_finalise(task_id, None)

    async def test_challenger_still_training_is_held(self, hold):
        task_id = self._seed(hold.env, SeedRole.BOSS_CHAINED_FREEFORM, [EMISSION_BURN_HOTKEY, "5Chal"])
        hold.statuses.update({EMISSION_BURN_HOTKEY: "success", "5Chal": "training"})

        assert not await seeds_sql.boss_round_seed_ready_to_finalise(task_id, None)

    async def test_released_once_every_player_is_terminal(self, hold):
        task_id = self._seed(hold.env, SeedRole.BOSS_CHAINED_FREEFORM, [EMISSION_BURN_HOTKEY, "5Chal"])
        hold.statuses.update({EMISSION_BURN_HOTKEY: "success", "5Chal": "failure"})

        assert await seeds_sql.boss_round_seed_ready_to_finalise(task_id, None)

    async def test_persisted_quant_failure_counts_as_terminal(self, hold):
        task_id = self._seed(hold.env, SeedRole.BOSS_CHAINED_FIXED, [EMISSION_BURN_HOTKEY, "5Chal"])
        hold.statuses.update({EMISSION_BURN_HOTKEY: "success"})
        hold.losses.append({db_cst.HOTKEY: "5Chal", db_cst.QUANT_FAILURE: "upstream instruct run failed"})

        assert await seeds_sql.boss_round_seed_ready_to_finalise(task_id, None)


class TestStatusAndReplacement:
    @pytest.fixture
    def outcome(self, env, monkeypatch):
        ns = SimpleNamespace(boss_status={}, usable=[EMISSION_BURN_HOTKEY], rejection=None)
        monkeypatch.setattr(
            tournament_sql, "get_training_status_for_task_and_hotkeys", AsyncMock(side_effect=lambda *_: ns.boss_status)
        )
        monkeypatch.setattr(seed_tasks.chaining, "shared_input_candidates", AsyncMock(side_effect=lambda *_: ns.usable))
        monkeypatch.setattr(seed_tasks, "_boss_output_rejection", AsyncMock(side_effect=lambda *_: ns.rejection))
        monkeypatch.setattr(seed_tasks, "boss_output_awaits_merge", AsyncMock(return_value=False))
        ns.env = env
        return ns

    async def _setup(self, env):
        seeds = {seed.role: seed for seed in await seed_tasks.create_seed_tasks(TID, env.config)}
        return seeds

    async def test_input_seed_success_with_a_full_model_is_ready(self, outcome):
        env = outcome.env
        seeds = await self._setup(env)
        env.tasks[seeds[SeedRole.INPUT_SMALL].task_id].status = TaskStatus.SUCCESS

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        assert _live(env)[SeedRole.INPUT_SMALL].status == SeedStatus.READY

    async def test_boss_round_seed_is_trained_after_the_boss_run_and_stays_held(self, outcome):
        env = outcome.env
        seeds = await self._setup(env)
        outcome.boss_status = {EMISSION_BURN_HOTKEY: "success"}

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        live = _live(env)
        assert live[SeedRole.BOSS_CHAINED_FIXED].status == SeedStatus.TRAINED
        assert live[SeedRole.BOSS_CHAINED_FREEFORM].status == SeedStatus.TRAINED
        assert seeds[SeedRole.BOSS_CHAINED_FIXED].task_id == live[SeedRole.BOSS_CHAINED_FIXED].task_id

    async def test_the_boss_input_is_evaluated_like_an_input_seed(self, outcome):
        env = outcome.env
        seeds = await self._setup(env)
        outcome.boss_status = {EMISSION_BURN_HOTKEY: "success"}

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)
        assert _live(env)[SeedRole.BOSS_INSTRUCT].status == SeedStatus.TRAINING  # not usable until its task is SUCCESS

        env.tasks[seeds[SeedRole.BOSS_INSTRUCT].task_id].status = TaskStatus.SUCCESS
        await seed_tasks.process_tournament_seed_tasks(TID, env.config)
        assert _live(env)[SeedRole.BOSS_INSTRUCT].status == SeedStatus.READY

    async def test_a_boss_input_whose_output_is_not_a_full_model_is_replaced_and_its_compressions_re_created(self, outcome):
        env = outcome.env
        seeds = await self._setup(env)
        env.tasks[seeds[SeedRole.BOSS_INSTRUCT].task_id].status = TaskStatus.SUCCESS
        outcome.rejection = "org/out is not a full model: adapter"

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        live = _live(env)
        assert live[SeedRole.BOSS_INSTRUCT].attempt == 2
        assert all(live[role].attempt == 2 for role in LEGACY_BOSS_ROUND_SEED_ROLES)

    @pytest.mark.parametrize("db_status", [TaskStatus.PREP_TASK_FAILURE.value, TaskStatus.FAILURE.value])
    async def test_a_seed_that_failed_with_the_db_status_string_is_replaced(self, outcome, db_status):
        """RawTask.status is a plain str when read from the DB, not a TaskStatus: the replacement path must not need .value."""
        env = outcome.env
        seeds = await self._setup(env)
        seed = seeds[SeedRole.INPUT_LARGE]
        old_task = env.tasks[seed.task_id]
        old_task.status = db_status
        assert type(old_task.status) is str

        assert await seed_tasks._evaluate_seed(seed, old_task, env.config) == (None, f"task ended {db_status}")

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        live = _live(env)[SeedRole.INPUT_LARGE]
        assert live.attempt == seed.attempt + 1 and live.task_id != seed.task_id

    async def test_failed_input_seed_is_replaced_once_and_dependents_are_repointed(self, outcome):
        env = outcome.env
        seeds = await self._setup(env)
        old = env.tasks[seeds[SeedRole.INPUT_SMALL].task_id]
        old.status = TaskStatus.PREP_TASK_FAILURE
        dependent = _quant(
            TaskType.QUANTFREEFORMTASK, upstream_task_id=old.task_id, status=TaskStatus.AWAITING_UPSTREAM, ds=old.ds
        )
        env.tasks[dependent.task_id] = dependent

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        rows = [row for row in env.store.rows if row.role == SeedRole.INPUT_SMALL]
        assert [(row.attempt, row.status) for row in rows] == [(1, SeedStatus.REPLACED), (2, SeedStatus.TRAINING)]
        assert rows[1].task_id != rows[0].task_id
        assert env.assigned[str(rows[1].task_id)] == [EMISSION_BURN_HOTKEY]
        env.repoint.assert_called_once()
        assert env.repoint.call_args.args[0] is dependent
        assert env.repoint.call_args.args[1].task_id == rows[1].task_id

    async def test_boss_run_failure_replaces_the_seed(self, outcome):
        env = outcome.env
        await self._setup(env)
        outcome.boss_status = {EMISSION_BURN_HOTKEY: "failure"}

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        # A compression's own boss failure is not redrawn; attempt 2 is the re-creation by its replaced upstream.
        assert {row.attempt for row in env.store.rows if row.role == SeedRole.BOSS_CHAINED_FREEFORM} == {1, 2}

    async def test_no_finetune_replaces_an_input_seed(self, outcome):
        env = outcome.env
        seeds = await self._setup(env)
        env.tasks[seeds[SeedRole.INPUT_LARGE].task_id].status = TaskStatus.SUCCESS
        outcome.usable = []

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        assert _live(env)[SeedRole.INPUT_LARGE].attempt == 2

    async def test_replacing_the_boss_instruct_recreates_both_chained_seeds_on_the_new_one(self, outcome):
        env = outcome.env
        seeds = await self._setup(env)
        env.tasks[seeds[SeedRole.BOSS_INSTRUCT].task_id].status = TaskStatus.FAILURE
        old_chained = [seeds[SeedRole.BOSS_CHAINED_FREEFORM].task_id, seeds[SeedRole.BOSS_CHAINED_FIXED].task_id]

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        live = _live(env)
        new_instruct = live[SeedRole.BOSS_INSTRUCT]
        assert new_instruct.attempt == 2
        for role in (SeedRole.BOSS_CHAINED_FREEFORM, SeedRole.BOSS_CHAINED_FIXED):
            assert live[role].attempt == 2 and live[role].task_id not in old_chained
        first_fixed = next(fmt for task_type, _, fmt, _ in env.chained_calls if task_type == TaskType.QUANTFIXEDTASK)
        assert env.chained_calls[-1][2] == first_fixed  # the re-created fixed slot redraws the same format
        new_upstreams = {str(upstream.task_id) for _, upstream, _, _ in env.chained_calls[2:]}
        assert new_upstreams == {str(new_instruct.task_id)}
        for task_id in old_chained:
            assert env.tasks[task_id].status == TaskStatus.FAILURE  # the held originals can never be released
        env.repoint.assert_not_called()  # chained boss seeds are re-created, not repointed

    async def test_third_failure_marks_the_row_failed_alerts_and_halts(self, outcome):
        env = outcome.env
        await self._setup(env)
        assert await seed_tasks.seed_halt_reason(TID, None) is None

        for _ in range(t_cst.SEED_MAX_REPLACEMENTS + 1):
            current = _live(env)[SeedRole.INPUT_SMALL]
            env.tasks[current.task_id].status = TaskStatus.FAILURE
            await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        rows = [row for row in env.store.rows if row.role == SeedRole.INPUT_SMALL]
        assert [row.status for row in rows] == [SeedStatus.REPLACED, SeedStatus.REPLACED, SeedStatus.FAILED]
        assert "input_small" in await seed_tasks.seed_halt_reason(TID, None)
        env.alerts.assert_awaited_once()
        assert TID in env.alerts.await_args.kwargs["message"]

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)  # a further pass does not re-alert or redraw
        env.alerts.assert_awaited_once()
        assert len(rows) == 3


class TestBossRoundSeedsAreNeverReplacedLate:
    """The hard rule: nothing boss-side trains after R3, so a boss-round seed is never redrawn once attached (or once
    the boss round exists), and a lost chained seed is retried only before the boss round."""

    @pytest.fixture
    def outcome(self, env, monkeypatch):
        ns = SimpleNamespace(boss_status={}, env=env)
        monkeypatch.setattr(
            tournament_sql, "get_training_status_for_task_and_hotkeys", AsyncMock(side_effect=lambda *_: ns.boss_status)
        )
        monkeypatch.setattr(seed_tasks.chaining, "shared_input_candidates", AsyncMock(return_value=[EMISSION_BURN_HOTKEY]))
        monkeypatch.setattr(seed_tasks, "_boss_output_rejection", AsyncMock(return_value=None))
        return ns

    async def _seeds(self, env):
        return {seed.role: seed for seed in await seed_tasks.create_seed_tasks(TID, env.config)}

    def _assert_nothing_drawn(self, env, seeds, snapshot):
        assert len(env.tasks) == snapshot
        assert {(row.role, row.attempt) for row in env.store.rows} == {(role, 1) for role in seeds}
        assert all(row.status != SeedStatus.REPLACED for row in env.store.rows)
        assert all(hotkeys == [EMISSION_BURN_HOTKEY] for hotkeys in env.assigned.values())  # no training row to add either

    @pytest.mark.parametrize("dead_status", [TaskStatus.PREP_TASK_FAILURE, TaskStatus.FAILURE])
    async def test_a_post_attach_failure_draws_no_third_task_and_alerts(self, outcome, dead_status):
        env = outcome.env
        seeds = await self._seeds(env)
        seed = seeds[SeedRole.BOSS_CHAINED_FIXED]
        env.store.attached.add(seed.task_id)
        env.tasks[seed.task_id].status = dead_status
        snapshot = len(env.tasks)

        for _ in range(2):
            await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        self._assert_nothing_drawn(env, seeds, snapshot)
        env.alerts.assert_awaited_once()  # once per process, not once per cycle
        assert "not replaced" in env.alerts.await_args.kwargs["message"]

    async def test_a_detached_original_still_counts_as_attached_through_the_durable_marker(self, outcome):
        env = outcome.env
        seeds = await self._seeds(env)
        seed = seeds[SeedRole.BOSS_CHAINED_FREEFORM]
        await env.store.mark_seed_attached(seed.task_id, "tourn_round_004", None)  # the tournament_tasks row is gone
        env.tasks[seed.task_id].status = TaskStatus.PREP_TASK_FAILURE
        snapshot = len(env.tasks)

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        self._assert_nothing_drawn(env, seeds, snapshot)

    async def test_no_attached_boss_round_role_is_replaced(self, outcome):
        env = outcome.env
        seeds = await self._seeds(env)
        for role in LEGACY_BOSS_ROUND_SEED_ROLES:
            env.store.attached.add(seeds[role].task_id)
            env.tasks[seeds[role].task_id].status = TaskStatus.PREP_TASK_FAILURE
        snapshot = len(env.tasks)

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        self._assert_nothing_drawn(env, seeds, snapshot)

    async def test_an_input_seed_is_still_replaced_with_the_boss_round_present(self, outcome):
        env = outcome.env
        seeds = await self._seeds(env)
        env.tasks[seeds[SeedRole.INPUT_SMALL].task_id].status = TaskStatus.PREP_TASK_FAILURE

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        assert _live(env)[SeedRole.INPUT_SMALL].attempt == 2

    async def test_a_lost_chained_seed_is_marked_replaced_and_retried_on_the_new_instruct(self, outcome, monkeypatch):
        env = outcome.env
        seeds = await self._seeds(env)
        old_chained = {role: seeds[role] for role in seed_tasks.CHAINED_SEED_TASK_TYPES}
        real = seed_tasks.create_chained_quant_task
        broken = {"value": True}

        async def flaky(config, task_type, upstream, *, quant_format, boss=True):
            if broken["value"]:
                raise RuntimeError("pool is empty")
            return await real(config, task_type, upstream, quant_format=quant_format, boss=boss)

        monkeypatch.setattr(seed_tasks, "create_chained_quant_task", flaky)
        env.tasks[seeds[SeedRole.BOSS_INSTRUCT].task_id].status = TaskStatus.PREP_TASK_FAILURE

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        new_instruct = _live(env)[SeedRole.BOSS_INSTRUCT]
        assert new_instruct.attempt == 2
        for role, old in old_chained.items():
            assert [row.status for row in env.store.rows if row.role == role] == [SeedStatus.REPLACED]  # no live row left
            assert env.tasks[old.task_id].status == TaskStatus.FAILURE  # the held original can never be released
        assert env.store.rows and SeedRole.BOSS_CHAINED_FIXED not in _live(env)

        broken["value"] = False
        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        live = _live(env)
        for role, old in old_chained.items():
            assert live[role].attempt == 2 and live[role].task_id != old.task_id
            assert str(env.tasks[live[role].task_id].upstream_task_id) == str(new_instruct.task_id)

    async def test_an_unattached_failed_boss_round_seed_is_replaced_though_the_boss_round_row_exists(self, outcome):
        env = outcome.env
        seeds = await self._seeds(env)
        env.tasks[seeds[SeedRole.BOSS_CHAINED_FIXED].task_id].status = TaskStatus.PREP_TASK_FAILURE  # nothing attached yet

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        assert _live(env)[SeedRole.BOSS_CHAINED_FIXED].attempt == 2

    async def test_a_missing_role_is_drawn_while_the_boss_round_has_attached_nothing(self, outcome):
        env = outcome.env
        seeds = await self._seeds(env)
        env.store.rows = [row for row in env.store.rows if row.role != SeedRole.BOSS_CHAINED_FIXED]
        before = len(env.tasks)

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        assert len(env.tasks) == before + 1
        assert _live(env)[SeedRole.BOSS_CHAINED_FIXED].attempt == 1 and seeds

    async def test_a_missing_role_whose_seed_was_attached_is_never_redrawn(self, outcome):
        env = outcome.env
        seeds = await self._seeds(env)
        env.store.attached.add(seeds[SeedRole.BOSS_CHAINED_FIXED].task_id)
        env.store.rows = [
            row.model_copy(update={"status": SeedStatus.REPLACED}) if row.role == SeedRole.BOSS_CHAINED_FIXED else row
            for row in env.store.rows
        ]
        before = len(env.tasks)

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        assert len(env.tasks) == before
        assert SeedRole.BOSS_CHAINED_FIXED not in _live(env)

    async def test_the_retry_respects_the_replacement_cap_and_halts(self, outcome):
        env = outcome.env
        seeds = await self._seeds(env)
        for row in env.store.rows:
            if row.role == SeedRole.BOSS_CHAINED_FIXED:
                row.status = SeedStatus.REPLACED
                row.attempt = t_cst.SEED_MAX_REPLACEMENTS + 1
                earlier = [
                    row.model_copy(update={"attempt": attempt, "task_id": uuid4()})
                    for attempt in range(1, t_cst.SEED_MAX_REPLACEMENTS + 1)
                ]
        env.store.rows.extend(earlier)
        before = len(env.tasks)

        for _ in range(2):
            await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        assert len(env.tasks) == before and seeds
        failed = [row for row in env.store.rows if row.role == SeedRole.BOSS_CHAINED_FIXED]
        assert {row.attempt: row.status for row in failed} == {
            1: SeedStatus.REPLACED,
            2: SeedStatus.REPLACED,
            3: SeedStatus.FAILED,
        }
        env.alerts.assert_awaited_once()
        assert "HALTED" in env.alerts.await_args.kwargs["message"]


class TestBossCompressionEntryFailure:
    """A boss-side training failure of a compression seed is the boss's failed entry: no redraw, no halt. The boss's code
    cannot be changed by a redraw (e.g. a boss with no quant code)."""

    @pytest.fixture
    def outcome(self, env, monkeypatch):
        ns = SimpleNamespace(env=env, failed_task=None, replace=AsyncMock())
        monkeypatch.setattr(
            tournament_sql,
            "get_training_status_for_task_and_hotkeys",
            AsyncMock(
                side_effect=lambda task_id, *_: (
                    {EMISSION_BURN_HOTKEY: TrainingStatus.FAILURE.value} if task_id == str(ns.failed_task) else {}
                )
            ),
        )
        return ns

    @pytest.mark.parametrize(
        "role",
        [
            SeedRole.BOSS_CHAINED_FREEFORM,
            SeedRole.BOSS_CHAINED_FIXED,
        ],
    )
    @pytest.mark.parametrize("task_status", [TaskStatus.TRAINING.value, TaskStatus.FAILURE.value])
    async def test_it_is_recorded_not_redrawn_and_nothing_halts(self, outcome, role, task_status, monkeypatch):
        env = outcome.env
        await seed_tasks.create_seed_tasks(TID, env.config)
        seed = _live(env)[role]
        outcome.failed_task = seed.task_id
        env.tasks[seed.task_id].status = task_status
        monkeypatch.setattr(seed_tasks, "replace_seed", outcome.replace)

        for _ in range(3):
            await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        outcome.replace.assert_not_called()
        live = _live(env)[role]
        assert (live.task_id, live.attempt, live.status) == (seed.task_id, seed.attempt, SeedStatus.TRAINED)
        assert await seed_tasks.seed_halt_reason(TID, env.config.psql_db) is None

    @pytest.mark.parametrize("role", [SeedRole.BOSS_CHAINED_FIXED, SeedRole.BOSS_CHAINED_FREEFORM])
    async def test_a_prep_failure_of_a_compression_seed_keeps_its_path(self, outcome, role):
        env = outcome.env
        await seed_tasks.create_seed_tasks(TID, env.config)
        seed = _live(env)[role]
        env.tasks[seed.task_id].status = TaskStatus.PREP_TASK_FAILURE.value

        assert await seed_tasks._evaluate_seed(seed, env.tasks[seed.task_id], env.config) == (
            None,
            f"task ended {TaskStatus.PREP_TASK_FAILURE.value}",
        )

    @pytest.mark.parametrize("role", [SeedRole.BOSS_INSTRUCT])
    async def test_a_failed_boss_input_run_still_redraws(self, outcome, role):
        env = outcome.env
        await seed_tasks.create_seed_tasks(TID, env.config)
        seed = _live(env)[role]
        outcome.failed_task = seed.task_id

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        live = _live(env)[role]
        assert live.attempt == seed.attempt + 1 and live.task_id != seed.task_id


class TestTournamentStartedBeforeSeeding:
    """A tournament with no seed rows is never seeded retroactively."""

    async def test_no_seed_is_created_or_trained_for_it(self, env, caplog):
        with caplog.at_level("INFO"):
            for _ in range(2):
                await seed_tasks.process_tournament_seed_tasks("legacy-tournament", env.config)

        assert env.store.rows == []
        assert not env.pool_calls and not env.tasks
        assert sum("not seeding it" in record.message for record in caplog.records) == 1

    async def test_a_seeded_tournament_still_retries_a_missing_role(self, env):
        await seed_tasks.create_seed_tasks(TID, env.config)
        env.store.rows = [row for row in env.store.rows if row.role != SeedRole.BOSS_CHAINED_FIXED]

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        assert SeedRole.BOSS_CHAINED_FIXED in {seed.role for seed in await seeds_sql.get_active_seed_tasks(TID, None)}


class TestBossRoundWithAMissingRole:
    """The boss round row exists before attach; a role with no live seed must not wedge it silently."""

    async def _with_missing_role(self, env):
        seeds = {seed.role: seed for seed in await seed_tasks.create_seed_tasks(TID, env.config)}
        env.store.rows = [row for row in env.store.rows if row.role != SeedRole.BOSS_CHAINED_FIXED]
        return seeds

    def _attach_stubs(self, env, monkeypatch):
        registered = []

        async def register(task, tournament_id, round_id, config, pair_id=None):
            registered.append(task.task_id)
            env.store.attached.add(task.task_id)

        monkeypatch.setattr(task_creator, "_get_existing_tasks_by_identifier", AsyncMock(return_value=[]))
        monkeypatch.setattr(task_creator, "_create_and_register_tournament_task", register)
        return registered

    async def test_attach_raises_the_typed_error_naming_the_missing_roles(self, env, monkeypatch):
        await self._with_missing_role(env)
        self._attach_stubs(env, monkeypatch)
        live = {seed.role: seed for seed in await seeds_sql.get_active_seed_tasks(TID, None)}

        with pytest.raises(task_creator.MissingBossRoundSeeds) as raised:
            await task_creator._attach_boss_round_seeds(live, TID, "round_4", "round_4_pair_001", env.config)

        assert raised.value.tournament_id == TID and raised.value.roles == [SeedRole.BOSS_CHAINED_FIXED.value]

    async def test_the_missing_role_is_drawn_with_the_final_round_row_present_and_the_next_attach_succeeds(
        self, env, monkeypatch
    ):
        await self._with_missing_role(env)
        registered = self._attach_stubs(env, monkeypatch)

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)  # the final round row exists, nothing attached

        live = {seed.role: seed for seed in await seeds_sql.get_active_seed_tasks(TID, None)}
        assert SeedRole.BOSS_CHAINED_FIXED in live
        tasks = await task_creator._attach_boss_round_seeds(live, TID, "round_4", "round_4_pair_001", env.config)
        assert len(tasks) == len(LEGACY_BOSS_ROUND_SEED_ROLES) == len(registered)
        assert all(row.attached_round_id == "round_4" for row in env.store.rows if row.role in LEGACY_BOSS_ROUND_SEED_ROLES)

    async def test_an_attached_role_is_never_redrawn(self, env, monkeypatch):
        seeds = await self._with_missing_role(env)
        env.store.rows.append(
            TournamentSeedTask(
                tournament_id=TID, role=SeedRole.BOSS_CHAINED_FIXED, attempt=1, task_id=uuid4(), status=SeedStatus.REPLACED,
                attached_round_id="round_4",
            )
        )
        before = len(env.tasks)

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        assert len(env.tasks) == before and seeds
        assert SeedRole.BOSS_CHAINED_FIXED not in _live(env)

    async def test_the_attach_error_alerts_once_naming_tournament_and_roles(self, env):
        error = task_creator.MissingBossRoundSeeds(TID, [SeedRole.BOSS_CHAINED_FIXED.value])

        for _ in range(3):
            await tournament_manager._alert_missing_boss_round_seeds(error, "round_4", env.config)

        env.alerts.assert_awaited_once()
        message = env.alerts.await_args.kwargs["message"]
        assert TID in message and "boss_chained_fixed" in message and "PENDING" in message


class TestOneAlertPerBossSideFailure:
    async def test_the_replacement_guard_and_the_completion_check_share_one_discord_message(self, env, monkeypatch):
        seeds = {seed.role: seed for seed in await seed_tasks.create_seed_tasks(TID, env.config)}
        seed = seeds[SeedRole.BOSS_CHAINED_FIXED]
        env.store.attached.add(seed.task_id)
        env.tasks[seed.task_id].status = TaskStatus.PREP_TASK_FAILURE
        monkeypatch.setattr(
            tournament_manager.seeds_sql, "boss_side_seed_failure", AsyncMock(return_value=SeedRole.BOSS_CHAINED_FIXED)
        )

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)
        completed, _ = await tournament_manager.is_tourn_task_completed(
            SimpleNamespace(tournament_id=TID, round_id="round_4", task_id=seed.task_id),
            SimpleNamespace(status=TaskStatus.PREP_TASK_FAILURE.value, task_id=seed.task_id, task_type=TaskType.QUANTFIXEDTASK),
            env.config,
            final_round=True,
        )

        assert completed is True
        env.alerts.assert_awaited_once()


class TestDrawFailureAlert:
    async def test_a_seed_draw_failure_alerts_once_per_role_not_every_cycle(self, env, monkeypatch):
        broken = AsyncMock(side_effect=RuntimeError("no 4-8B model"))
        monkeypatch.setattr(seed_tasks, "create_synthetic_instruct_text_task", broken)

        for _ in range(3):
            created = await seed_tasks.create_seed_tasks(TID, env.config)

        assert {seed.role for seed in created}.isdisjoint({SeedRole.INPUT_SMALL, SeedRole.INPUT_LARGE, SeedRole.BOSS_INSTRUCT})
        env.alerts.assert_awaited()
        messages = [call.kwargs["message"] for call in env.alerts.await_args_list]
        assert len(messages) == len(set(messages))  # one alert per role, none repeated across the three cycles
        assert any("input_small" in message and "no 4-8B model" in message for message in messages)
        assert any("boss_instruct" in message for message in messages)

    async def test_a_replacement_draw_failure_alerts(self, env, monkeypatch):
        seeds = {seed.role: seed for seed in await seed_tasks.create_seed_tasks(TID, env.config)}
        monkeypatch.setattr(seed_tasks, "_draw_seed_task", AsyncMock(return_value=None))
        env.alerts.reset_mock()

        for _ in range(2):
            assert await seed_tasks.replace_seed(seeds[SeedRole.INPUT_SMALL], "task ended failure", env.config) is None

        env.alerts.assert_awaited_once()
        assert "could not draw a replacement for input_small" in env.alerts.await_args.kwargs["message"]


class TestBossSideSeedFailure:
    """seeds_sql.boss_side_seed_failure: which dead boss-round seed tasks are the boss's failure."""

    async def _seed(self, env, role):
        seeds = {seed.role: seed for seed in await seed_tasks.create_seed_tasks(TID, env.config)}
        return seeds[role].task_id

    @staticmethod
    def _db(boss_row, challenger_row=None, quant_failed=()):
        rows = [{db_cst.HOTKEY: EMISSION_BURN_HOTKEY, db_cst.TRAINING_STATUS: boss_row}] if boss_row else []
        if challenger_row:
            rows.append({db_cst.HOTKEY: "5Challenger", db_cst.TRAINING_STATUS: challenger_row})
        return _FakeDb(fetch=[rows, [{db_cst.HOTKEY: hotkey} for hotkey in quant_failed]])

    async def test_prep_failure_of_any_boss_round_seed_is_boss_side_when_the_challenger_is_usable(self, env):
        for role in LEGACY_BOSS_ROUND_SEED_ROLES:
            task_id = await self._seed(env, role)
            for boss_row in (None, "failure"):
                db = self._db(boss_row, "success")
                assert await seeds_sql.boss_side_seed_failure(task_id, TaskStatus.PREP_TASK_FAILURE, db) == role

    @pytest.mark.parametrize(
        ("boss_row", "challenger_row", "quant_failed"),
        [
            (None, None, []),  # nobody ran anything: not the challenger's slot
            ("failure", None, []),
            (None, "failure", []),
            (None, "success", ["5Challenger"]),  # the challenger has a persisted quant_failure
        ],
    )
    async def test_prep_failure_without_a_usable_challenger_is_not_credited(self, env, boss_row, challenger_row, quant_failed):
        for role in LEGACY_BOSS_ROUND_SEED_ROLES:
            task_id = await self._seed(env, role)
            db = self._db(boss_row, challenger_row, quant_failed)
            assert await seeds_sql.boss_side_seed_failure(task_id, TaskStatus.PREP_TASK_FAILURE, db) is None

    async def test_prep_failure_after_the_boss_trained_is_not_the_boss_failure(self, env):
        task_id = await self._seed(env, SeedRole.BOSS_CHAINED_FIXED)
        db = self._db("success", "success")

        assert await seeds_sql.boss_side_seed_failure(task_id, TaskStatus.PREP_TASK_FAILURE, db) is None

    async def test_input_seeds_and_non_seeds_are_never_boss_side(self, env):
        task_id = await self._seed(env, SeedRole.INPUT_SMALL)
        assert await seeds_sql.boss_side_seed_failure(task_id, TaskStatus.PREP_TASK_FAILURE, None) is None
        assert await seeds_sql.boss_side_seed_failure(uuid4(), TaskStatus.PREP_TASK_FAILURE, None) is None

    async def test_live_statuses_are_not_failures(self, env):
        task_id = await self._seed(env, SeedRole.BOSS_CHAINED_FIXED)
        assert await seeds_sql.boss_side_seed_failure(task_id, TaskStatus.TRAINING, None) is None

    @pytest.mark.parametrize(
        ("boss_row", "challenger_row", "quant_failed", "expected"),
        [
            ("failure", "success", [], True),
            ("failure", "failure", [], False),  # both failed: the ordinary rule, the boss wins by default
            ("failure", "success", ["5Challenger"], False),  # the challenger has a persisted quant_failure
            ("failure", None, [], False),  # the challenger has no usable result
            ("success", "success", [], False),
            (None, "success", [], False),
        ],
    )
    async def test_failure_is_boss_side_only_when_the_challenger_side_is_intact(
        self, env, boss_row, challenger_row, quant_failed, expected
    ):
        task_id = await self._seed(env, SeedRole.BOSS_CHAINED_FIXED)
        rows = [{db_cst.HOTKEY: EMISSION_BURN_HOTKEY, db_cst.TRAINING_STATUS: boss_row}] if boss_row else []
        if challenger_row:
            rows.append({db_cst.HOTKEY: "5Challenger", db_cst.TRAINING_STATUS: challenger_row})
        db = _FakeDb(fetch=[rows, [{db_cst.HOTKEY: hotkey} for hotkey in quant_failed]])

        role = await seeds_sql.boss_side_seed_failure(task_id, TaskStatus.FAILURE, db)

        assert (role == SeedRole.BOSS_CHAINED_FIXED) is expected


class TestPrepRejections:
    """A prep-stage dataset rejection redraws without consuming SEED_MAX_REPLACEMENTS (prod's path has no cap), bounded by
    SEED_MAX_PREP_REJECTIONS; real failures keep the cap."""

    @pytest.fixture
    def outcome(self, env, monkeypatch):
        ns = SimpleNamespace(env=env)
        monkeypatch.setattr(tournament_sql, "get_training_status_for_task_and_hotkeys", AsyncMock(return_value={}))
        return ns

    @staticmethod
    def _rows(env, role):
        return [row for row in env.store.rows if row.role == role]

    @staticmethod
    async def _fail(env, role, status):
        current = _live(env)[role]
        env.tasks[current.task_id].status = status
        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

    @staticmethod
    def _failed_run(task_id, failures, identity="task"):
        return {
            "task_id": task_id,
            "category": "prep",
            "prep_failure_count": failures,
            "should_notify_prep_failure": False,
            "metadata": {"prep_identity": identity},
        }

    async def test_a_deterministic_model_prep_error_gives_up_and_redraws_uncharged(self, outcome, monkeypatch):
        env = outcome.env
        await seed_tasks.create_seed_tasks(TID, env.config)
        seed = _live(env)[SeedRole.INPUT_LARGE]
        task = env.tasks[seed.task_id]
        task.status = TaskStatus.AWAITING_MODEL_PREP.value
        runs = [self._failed_run(task.task_id, failures) for failures in range(1, t_cst.SEED_MAX_PREP_ATTEMPTS + 1)]
        monkeypatch.setattr(orchestrator.gpu_cost_sql, "finish_cost_run", AsyncMock(side_effect=runs))

        for expected in [TaskStatus.AWAITING_MODEL_PREP.value] * (t_cst.SEED_MAX_PREP_ATTEMPTS - 1):
            await orchestrator._finish_prep_cost_run("prep:key", False, env.config)
            assert task.status == expected
        await orchestrator._finish_prep_cost_run("prep:key", False, env.config)
        assert task.status == TaskStatus.PREP_TASK_FAILURE

        await seed_tasks.process_tournament_seed_tasks(TID, env.config)

        rows = self._rows(env, SeedRole.INPUT_LARGE)
        old_row = next(row for row in rows if row.attempt == seed.attempt)
        assert old_row.status == SeedStatus.REPLACED and old_row.prep_rejected is True
        live = _live(env)[SeedRole.INPUT_LARGE]
        assert live.attempt == seed.attempt + 1 and live.task_id != seed.task_id

    async def test_a_non_seed_task_keeps_retrying_model_prep(self, outcome, monkeypatch):
        env = outcome.env
        task = _instruct(status=TaskStatus.AWAITING_MODEL_PREP.value)
        env.tasks[task.task_id] = task
        monkeypatch.setattr(
            orchestrator.gpu_cost_sql, "finish_cost_run", AsyncMock(return_value=self._failed_run(task.task_id, 50))
        )

        await orchestrator._finish_prep_cost_run("prep:key", False, env.config)

        assert task.status == TaskStatus.AWAITING_MODEL_PREP.value

    async def test_a_per_miner_prep_failure_does_not_give_up_on_the_seed(self, outcome, monkeypatch):
        env = outcome.env
        await seed_tasks.create_seed_tasks(TID, env.config)
        task = env.tasks[_live(env)[SeedRole.BOSS_CHAINED_FREEFORM].task_id]
        task.status = TaskStatus.AWAITING_MODEL_PREP.value
        run = self._failed_run(task.task_id, 3, identity="miner:5Challenger")
        monkeypatch.setattr(orchestrator.gpu_cost_sql, "finish_cost_run", AsyncMock(return_value=run))

        await orchestrator._finish_prep_cost_run("prep:key", False, env.config)

        assert task.status == TaskStatus.AWAITING_MODEL_PREP.value

    async def test_three_prep_rejections_leave_the_replacements_available_for_real_failures(self, outcome):
        env = outcome.env
        await seed_tasks.create_seed_tasks(TID, env.config)
        role = SeedRole.INPUT_SMALL

        for _ in range(3):
            await self._fail(env, role, TaskStatus.PREP_TASK_FAILURE.value)
        assert [(row.attempt, row.status, row.prep_rejected) for row in self._rows(env, role)] == [
            (1, SeedStatus.REPLACED, True),
            (2, SeedStatus.REPLACED, True),
            (3, SeedStatus.REPLACED, True),
            (4, SeedStatus.TRAINING, False),
        ]

        for _ in range(t_cst.SEED_MAX_REPLACEMENTS):  # the cap of real failures is intact
            await self._fail(env, role, TaskStatus.FAILURE.value)
        assert _live(env)[role].attempt == 4 + t_cst.SEED_MAX_REPLACEMENTS
        env.alerts.assert_not_awaited()

        await self._fail(env, role, TaskStatus.FAILURE.value)  # one more real failure halts
        assert self._rows(env, role)[-1].status == SeedStatus.FAILED
        env.alerts.assert_awaited_once()
        assert "HALTED" in env.alerts.await_args.kwargs["message"]

    async def test_real_training_failures_still_halt_after_two_replacements(self, outcome):
        env = outcome.env
        await seed_tasks.create_seed_tasks(TID, env.config)

        for _ in range(t_cst.SEED_MAX_REPLACEMENTS + 1):
            await self._fail(env, SeedRole.BOSS_INSTRUCT, TaskStatus.FAILURE.value)

        rows = self._rows(env, SeedRole.BOSS_INSTRUCT)
        assert [row.status for row in rows] == [SeedStatus.REPLACED, SeedStatus.REPLACED, SeedStatus.FAILED]
        assert not any(row.prep_rejected for row in rows)
        env.alerts.assert_awaited_once()

    async def test_the_prep_rejection_limit_halts_with_one_alert(self, outcome):
        env = outcome.env
        await seed_tasks.create_seed_tasks(TID, env.config)
        role = SeedRole.INPUT_LARGE

        for _ in range(t_cst.SEED_MAX_PREP_REJECTIONS - 1):
            await self._fail(env, role, TaskStatus.PREP_TASK_FAILURE.value)
        env.alerts.assert_not_awaited()  # normal redraws never alert
        assert _live(env)[role].attempt == t_cst.SEED_MAX_PREP_REJECTIONS

        await self._fail(env, role, TaskStatus.PREP_TASK_FAILURE.value)
        await seed_tasks.process_tournament_seed_tasks(TID, env.config)  # further passes do not re-alert or redraw

        rows = self._rows(env, role)
        assert len(rows) == t_cst.SEED_MAX_PREP_REJECTIONS and rows[-1].status == SeedStatus.FAILED
        assert "input_large" in await seed_tasks.seed_halt_reason(TID, None)
        env.alerts.assert_awaited_once()
        assert "prep-stage dataset rejections" in env.alerts.await_args.kwargs["message"]

    async def test_a_redraw_excludes_datasets_rejected_earlier_in_the_tournament_on_any_role(self, outcome, monkeypatch):
        env = outcome.env

        async def all_datasets(_keypair):
            for name in ("ds-a", "ds-b", "ds-c"):
                yield SimpleNamespace(dataset_id=name)

        async def draw(config, models, datasets, **kwargs):
            task = _instruct(ds=(await anext(datasets)).dataset_id)
            env.tasks[task.task_id] = task
            return task

        monkeypatch.setattr(seed_tasks, "_get_instruct_text_datasets", all_datasets)
        monkeypatch.setattr(seed_tasks, "create_synthetic_instruct_text_task", draw)
        await seed_tasks.create_seed_tasks(TID, env.config)
        env.tasks[_live(env)[SeedRole.INPUT_SMALL].task_id].ds = "ds-a"
        env.tasks[_live(env)[SeedRole.INPUT_LARGE].task_id].ds = "ds-b"

        await self._fail(env, SeedRole.INPUT_SMALL, TaskStatus.PREP_TASK_FAILURE.value)
        assert env.tasks[_live(env)[SeedRole.INPUT_SMALL].task_id].ds == "ds-b"  # not the rejected ds-a

        await self._fail(env, SeedRole.INPUT_LARGE, TaskStatus.PREP_TASK_FAILURE.value)
        assert env.tasks[_live(env)[SeedRole.INPUT_LARGE].task_id].ds == "ds-c"  # not its own ds-b, nor small's ds-a

    async def test_a_compression_prep_rejection_counts_against_the_normal_cap(self, outcome):
        env = outcome.env
        await seed_tasks.create_seed_tasks(TID, env.config)
        role = SeedRole.BOSS_CHAINED_FIXED
        assert not seed_tasks.is_prep_rejection(_live(env)[role], _instruct(status=TaskStatus.PREP_TASK_FAILURE.value))

        for _ in range(t_cst.SEED_MAX_REPLACEMENTS + 1):
            await self._fail(env, role, TaskStatus.PREP_TASK_FAILURE.value)

        rows = self._rows(env, role)
        assert [row.status for row in rows] == [SeedStatus.REPLACED, SeedStatus.REPLACED, SeedStatus.FAILED]
        assert not any(row.prep_rejected for row in rows)
        assert len({env.tasks[row.task_id].ds for row in rows}) == 1  # the instruct seed's data throughout: a redraw cannot help
        env.alerts.assert_awaited_once()

    @pytest.mark.parametrize("status", [TaskStatus.PREP_TASK_FAILURE.value, TaskStatus.FAILURE.value])
    async def test_a_replaced_boss_instruct_retires_its_chained_seeds_without_charging_them(self, outcome, status):
        env = outcome.env
        await seed_tasks.create_seed_tasks(TID, env.config)
        chained = (SeedRole.BOSS_CHAINED_FREEFORM, SeedRole.BOSS_CHAINED_FIXED)

        for _ in range(2):
            await self._fail(env, SeedRole.BOSS_INSTRUCT, status)
        for role in chained:
            rows = self._rows(env, role)
            assert [(row.attempt, row.status, row.prep_rejected) for row in rows[:2]] == [
                (1, SeedStatus.REPLACED, True),
                (2, SeedStatus.REPLACED, True),
            ]
            assert rows[2].attempt == 3 and rows[2].status == SeedStatus.TRAINING

        for _ in range(t_cst.SEED_MAX_REPLACEMENTS):  # the chained roles keep their own full cap
            await self._fail(env, SeedRole.BOSS_CHAINED_FIXED, TaskStatus.FAILURE.value)
        assert _live(env)[SeedRole.BOSS_CHAINED_FIXED].attempt == 3 + t_cst.SEED_MAX_REPLACEMENTS

    async def test_a_prep_rejected_dataset_is_read_back_from_the_history(self, outcome):
        env = outcome.env
        await seed_tasks.create_seed_tasks(TID, env.config)
        await self._fail(env, SeedRole.BOSS_INSTRUCT, TaskStatus.PREP_TASK_FAILURE.value)

        excluded = await seed_tasks._excluded_datasets(TID, frozenset({"ds-extra"}), env.config)

        assert excluded == {"org/ds", "ds-extra"}


class TestPrepRejectedColumn:
    def test_the_migration_declares_the_column(self):
        migration = (Path(seed_tasks.__file__).parents[1] / "db/migrations/20261109120000_tournament_seed_tasks.sql").read_text()

        assert "prep_rejected BOOLEAN NOT NULL DEFAULT FALSE" in migration

    async def test_insert_and_select_carry_the_column(self):
        recorded = []

        class _Conn(_FakeConn):
            async def execute(self, query, *args):
                recorded.append((query, args))

        row = dict(tournament_id=TID, role="input_small", attempt=2, task_id=uuid4(), status="training", prep_rejected=True)
        db = _FakeDb(fetchrow=[row])
        conn = _Conn(db.queries, [row])

        class _Ctx:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *exc):
                return False

        async def connection():
            return _Ctx()

        db.connection = connection
        seed = TournamentSeedTask(
            tournament_id=TID, role=SeedRole.INPUT_SMALL, attempt=2, task_id=row["task_id"], prep_rejected=True
        )

        stored = await seeds_sql.insert_seed_task(seed, db)

        assert stored.prep_rejected is True
        assert db_cst.SEED_PREP_REJECTED in recorded[0][0] and recorded[0][1][-1] is True
        assert db_cst.SEED_PREP_REJECTED in db.queries[0]


class TestCreateSeedFailures:
    async def test_boss_info_failure_creates_no_task(self, env, monkeypatch):
        monkeypatch.setattr(seed_tasks, "_boss_training_info", AsyncMock(side_effect=RuntimeError("db down")))

        seed = await seed_tasks._create_seed(SeedRole.INPUT_SMALL, TID, 1, {}, env.config)

        assert seed is None
        assert env.instruct_calls == [] and env.tasks == {} and env.store.rows == []
        env.alerts.assert_awaited_once()

    async def test_insert_failure_deletes_the_task_and_alerts(self, env, monkeypatch):
        monkeypatch.setattr(seeds_sql, "insert_seed_task", AsyncMock(side_effect=RuntimeError("insert failed")))
        delete = AsyncMock()
        monkeypatch.setattr(task_sql, "delete_task", delete)

        seed = await seed_tasks._create_seed(SeedRole.INPUT_SMALL, TID, 1, {}, env.config)

        assert seed is None
        (created,) = env.tasks.values()
        delete.assert_awaited_once_with(created.task_id, env.config.psql_db)
        env.alerts.assert_awaited_once()
        assert env.store.rows == [] and env.assigned == {}

    async def test_failed_cleanup_still_alerts_and_returns_none(self, env, monkeypatch):
        monkeypatch.setattr(seeds_sql, "insert_seed_task", AsyncMock(side_effect=RuntimeError("insert failed")))
        monkeypatch.setattr(task_sql, "delete_task", AsyncMock(side_effect=RuntimeError("delete failed")))

        assert await seed_tasks._create_seed(SeedRole.INPUT_SMALL, TID, 1, {}, env.config) is None
        env.alerts.assert_awaited_once()

    async def test_happy_path_keeps_the_task_and_records_the_boss(self, env, monkeypatch):
        delete = AsyncMock()
        monkeypatch.setattr(task_sql, "delete_task", delete)

        seed = await seed_tasks._create_seed(SeedRole.INPUT_SMALL, TID, 1, {}, env.config)

        assert seed is not None and seed.status == SeedStatus.TRAINING
        assert (seed.boss_repo, seed.boss_commit) == (BOSS_REPO, BOSS_COMMIT)
        assert str(seed.task_id) in env.assigned
        delete.assert_not_awaited()
        env.alerts.assert_not_awaited()
