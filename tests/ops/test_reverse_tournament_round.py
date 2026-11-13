"""reverse_tournament_round: refuses while a bound quant tournament depends on the round, keeps seed tasks (and the boss's rows),
deletes the other tasks. Drives the real function against an in-memory connection that applies the statements it is sent."""

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID
from uuid import uuid4

import pytest

from core.models.task_models import TaskStatus
from ops.tools.tournament import reverse_tournament_round as tool
from validator.db.sql import quant_sources as quant_sources_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import QuantSourceSlot
from validator.tournament.models import QuantTaskSource
from validator.tournament.models import SeedRole
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TrainingStatus


TOURNAMENT_ID = "tourn_text"
PREVIOUS_ROUND_ID = "tourn_text_round_001"
ROUND_ID = "tourn_text_round_002"
MINER_A = "5MinerA"
MINER_B = "5MinerB"


class ForeignKeyViolation(Exception):
    pass


class FakeTransaction:
    def __init__(self, db: "FakeDB") -> None:
        self.db = db

    async def __aenter__(self):
        self.db.transactions += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.db.rolled_back += exc_type is not None
        return False


class FakeConnection:
    """Applies the statements the tool uses to in-memory tables; any statement is recorded, in order."""

    def __init__(self, db: "FakeDB") -> None:
        self.db = db

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self.db)

    async def execute(self, sql: str, *args):
        self.db.statements.append((" ".join(sql.split()), args))
        db = self.db
        if "DELETE FROM tasks" in sql:
            (task_id,) = args
            if any(source.source_task_id == task_id for source in db.quant_sources):
                raise ForeignKeyViolation(f"quant_task_sources still names task {task_id}")
            db.tasks.discard(task_id)
            for table in (db.tournament_tasks, db.trainings, db.task_nodes, db.submissions):
                table[:] = [row for row in table if row["task_id"] != task_id]
            db.seeds[:] = [seed for seed in db.seeds if seed.task_id != task_id]
        elif "DELETE FROM tournament_rounds" in sql:
            db.rounds_deleted.append(args[0])

    async def fetch(self, sql: str, *args):
        self.db.statements.append((" ".join(sql.split()), args))
        db = self.db
        sql = " ".join(sql.split())
        if sql.startswith("DELETE FROM tournament_tasks"):
            task_id, round_id = args
            return self._remove(db.tournament_tasks, lambda row: row["task_id"] == task_id and row["round_id"] == round_id)
        if sql.startswith("DELETE FROM tournament_task_hotkey_trainings"):
            task_id, boss = args
            return self._remove(db.trainings, lambda row: row["task_id"] == task_id and row["hotkey"] != boss)
        if sql.startswith("DELETE FROM submissions"):
            task_id, _netuid, boss = args
            return self._remove(db.submissions, lambda row: row["task_id"] == task_id and row["hotkey"] != boss)
        if sql.startswith("DELETE FROM task_nodes"):
            task_id, _netuid, boss = args
            return self._remove(db.task_nodes, lambda row: row["task_id"] == task_id and row["hotkey"] != boss)
        if sql.startswith("DELETE FROM tournament_task_failure_reviews"):
            return []
        if sql.startswith("UPDATE tournament_seed_tasks"):
            task_id, round_id = args
            hit = [seed for seed in db.seeds if seed.task_id == task_id and seed.attached_round_id == round_id]
            for seed in hit:
                seed.attached_round_id = None
            return [(seed.task_id,) for seed in hit]
        raise AssertionError(f"unexpected statement: {sql}")

    @staticmethod
    def _remove(table: list[dict], matches) -> list[tuple]:
        gone = [row for row in table if matches(row)]
        table[:] = [row for row in table if not matches(row)]
        return [(row.get("hotkey", row["task_id"]),) for row in gone]


class FakeConnectionContext:
    def __init__(self, db: "FakeDB") -> None:
        self.db = db

    async def __aenter__(self) -> FakeConnection:
        self.db.opened += 1
        return FakeConnection(self.db)

    async def __aexit__(self, *exc):
        return False


class FakeDB:
    """The psql_db the tool is given. Its reads are served by the patched sql helpers; its connection is the writes' only path."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple]] = []
        self.opened = 0
        self.transactions = 0
        self.rolled_back = 0
        self.rounds_deleted: list[str] = []
        self.tasks: set[UUID] = set()
        self.tournament_tasks: list[dict] = []
        self.trainings: list[dict] = []
        self.task_nodes: list[dict] = []
        self.submissions: list[dict] = []
        self.seeds: list[TournamentSeedTask] = []
        self.quant_sources: list[QuantTaskSource] = []
        self.task_status: dict[UUID, TaskStatus] = {}

    async def connection(self) -> FakeConnectionContext:
        return FakeConnectionContext(self)

    def writes(self) -> list[str]:
        return [sql for sql, _ in self.statements if sql.split()[0] in ("DELETE", "UPDATE", "INSERT")]

    def add_round_task(
        self, task_id: UUID, status: TaskStatus = TaskStatus.TRAINING, boss_status: str | None = "success"
    ) -> None:
        """A task registered in the round, the boss and two miners on it."""
        self.tasks.add(task_id)
        self.task_status[task_id] = status
        self.tournament_tasks.append({"task_id": task_id, "round_id": ROUND_ID})
        for hotkey in (EMISSION_BURN_HOTKEY, MINER_A, MINER_B):
            self.task_nodes.append({"task_id": task_id, "hotkey": hotkey})
            self.submissions.append({"task_id": task_id, "hotkey": hotkey})
        self.trainings.append({"task_id": task_id, "hotkey": EMISSION_BURN_HOTKEY, "status": boss_status})
        self.trainings.extend({"task_id": task_id, "hotkey": hotkey, "status": "success"} for hotkey in (MINER_A, MINER_B))

    def add_seed(self, task_id: UUID, role: SeedRole = SeedRole.TEXT_R2, attached: str | None = ROUND_ID) -> TournamentSeedTask:
        seed = TournamentSeedTask(tournament_id=TOURNAMENT_ID, role=role, task_id=task_id, attached_round_id=attached)
        self.seeds.append(seed)
        return seed

    def rows_of(self, table: list[dict], task_id: UUID) -> set[str]:
        return {row["hotkey"] for row in table if row["task_id"] == task_id}


def quant_source(source_task_id: UUID, quant_tournament_id: str = "tourn_quant", superseded: bool = False) -> QuantTaskSource:
    return QuantTaskSource(
        quant_task_id=uuid4(),
        quant_tournament_id=quant_tournament_id,
        slot=QuantSourceSlot.R2,
        source_tournament_id=TOURNAMENT_ID,
        source_role=SeedRole.TEXT_R2,
        source_task_id=source_task_id,
        superseded=superseded,
    )


@pytest.fixture
def db(monkeypatch) -> FakeDB:
    fake = FakeDB()
    rounds = [SimpleNamespace(round_id=PREVIOUS_ROUND_ID, round_number=1), SimpleNamespace(round_id=ROUND_ID, round_number=2)]

    async def get_tournament(_tournament_id, _db):
        return SimpleNamespace(status=TournamentStatus.ACTIVE)

    async def get_tournament_rounds(_tournament_id, _db):
        return rounds

    async def get_tournament_tasks(round_id, _db):
        return [SimpleNamespace(task_id=row["task_id"]) for row in fake.tournament_tasks if row["round_id"] == round_id]

    async def get_training_status(task_id, hotkeys, _db):
        rows = [row for row in fake.trainings if str(row["task_id"]) == task_id and row["hotkey"] in hotkeys]
        return {row["hotkey"]: row["status"] for row in rows}

    async def get_task(task_id, _db):
        return SimpleNamespace(status=fake.task_status[task_id])

    async def get_seed(task_id, _db):
        return next((seed for seed in fake.seeds if seed.task_id == task_id), None)

    async def live_sources(task_id, _db):
        return [row for row in fake.quant_sources if row.source_task_id == task_id and not row.superseded]

    monkeypatch.setattr(tournament_sql, "get_tournament", get_tournament)
    monkeypatch.setattr(tournament_sql, "get_tournament_rounds", get_tournament_rounds)
    monkeypatch.setattr(tournament_sql, "get_tournament_tasks", get_tournament_tasks)
    monkeypatch.setattr(tournament_sql, "get_training_status_for_task_and_hotkeys", get_training_status)
    monkeypatch.setattr(task_sql, "get_task", get_task)
    monkeypatch.setattr(seeds_sql, "get_seed_task_by_task_id", get_seed)
    monkeypatch.setattr(quant_sources_sql, "list_live_quant_sources_naming_task", live_sources)
    return fake


@pytest.fixture
def log(monkeypatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr(tool, "logger", mock)
    return mock


def messages(mock: MagicMock, level: str) -> str:
    return "\n".join(str(call.args[0]) for call in getattr(mock, level).call_args_list)


async def test_refuses_while_a_live_quant_task_names_a_round_task(db, log) -> None:
    seed_task = uuid4()
    db.add_round_task(seed_task)
    db.add_seed(seed_task)
    source = quant_source(seed_task, "tourn_quant_live")
    db.quant_sources.append(source)

    assert await tool.reverse_tournament_round(TOURNAMENT_ID, ROUND_ID, db) is False

    assert db.writes() == [] and db.opened == 0 and db.transactions == 0
    error = messages(log, "error")
    assert ROUND_ID in error and str(seed_task) in error and "tourn_quant_live" in error and str(source.quant_task_id) in error
    assert db.rounds_deleted == [] and seed_task in db.tasks and len(db.tournament_tasks) == 1


async def test_refuses_for_a_non_seed_task_too(db, log) -> None:
    task = uuid4()
    db.add_round_task(task)
    db.quant_sources.append(quant_source(task))

    assert await tool.reverse_tournament_round(TOURNAMENT_ID, ROUND_ID, db) is False
    assert db.writes() == [] and task in db.tasks


async def test_a_superseded_source_row_does_not_refuse(db, log) -> None:
    seed_task = uuid4()
    db.add_round_task(seed_task)
    db.add_seed(seed_task)
    db.quant_sources.append(quant_source(seed_task, superseded=True))

    assert await tool.reverse_tournament_round(TOURNAMENT_ID, ROUND_ID, db) is True
    assert seed_task in db.tasks


async def test_a_seed_task_is_kept_and_only_the_miners_rows_go(db, log) -> None:
    seed_task = uuid4()
    db.add_round_task(seed_task)
    seed = db.add_seed(seed_task)

    assert await tool.reverse_tournament_round(TOURNAMENT_ID, ROUND_ID, db) is True

    assert not any(sql.startswith("DELETE FROM tasks") for sql in db.writes())
    assert seed_task in db.tasks
    assert db.tournament_tasks == []
    assert db.rows_of(db.trainings, seed_task) == {EMISSION_BURN_HOTKEY}
    assert db.rows_of(db.task_nodes, seed_task) == {EMISSION_BURN_HOTKEY}
    assert db.rows_of(db.submissions, seed_task) == {EMISSION_BURN_HOTKEY}
    assert seed.attached_round_id is None
    assert db.rounds_deleted == [ROUND_ID] and db.transactions == 1 and db.rolled_back == 0
    info = messages(log, "info")
    assert str(seed_task) in info and "kept" in info and MINER_A in info and MINER_B in info and "cleared" in info
    assert not log.warning.called  # the task is still TRAINING: nothing to warn about


async def test_a_seed_past_training_is_not_reset_and_the_operator_is_warned(db, log) -> None:
    seed_task = uuid4()
    db.add_round_task(seed_task, status=TaskStatus.SUCCESS)
    db.add_seed(seed_task)

    assert await tool.reverse_tournament_round(TOURNAMENT_ID, ROUND_ID, db) is True

    assert db.task_status[seed_task] == TaskStatus.SUCCESS
    assert not any("UPDATE tasks" in sql for sql in db.writes())
    warning = messages(log, "warning")
    assert str(seed_task) in warning and "success" in warning and "did NOT reset" in warning


async def test_a_seed_whose_boss_run_failed_keeps_its_marker(db, log) -> None:
    seed_task = uuid4()
    db.add_round_task(seed_task, boss_status=TrainingStatus.FAILURE.value)
    seed = db.add_seed(seed_task)

    assert await tool.reverse_tournament_round(TOURNAMENT_ID, ROUND_ID, db) is True

    assert seed.attached_round_id == ROUND_ID
    assert db.rows_of(db.trainings, seed_task) == {EMISSION_BURN_HOTKEY}
    assert "kept: the task or the boss's run failed" in messages(log, "info")


async def test_a_quant_boss_round_seed_keeps_its_marker(db, log) -> None:
    seed_task = uuid4()
    db.add_round_task(seed_task)
    seed = db.add_seed(seed_task, role=SeedRole.BOSS_R3_FREEFORM)

    assert await tool.reverse_tournament_round(TOURNAMENT_ID, ROUND_ID, db) is True

    assert seed.attached_round_id == ROUND_ID and seed_task in db.tasks
    assert db.rows_of(db.task_nodes, seed_task) == {EMISSION_BURN_HOTKEY}
    assert "boss-round compression" in messages(log, "info")


async def test_a_non_seed_task_is_still_deleted(db, log) -> None:
    task = uuid4()
    db.add_round_task(task)

    assert await tool.reverse_tournament_round(TOURNAMENT_ID, ROUND_ID, db) is True

    assert task not in db.tasks
    assert any(sql.startswith("DELETE FROM tasks") for sql in db.writes())
    assert db.rows_of(db.task_nodes, task) == set() and db.rounds_deleted == [ROUND_ID]


async def test_a_round_with_a_seed_and_a_plain_task_handles_each(db, log) -> None:
    seed_task, plain_task = uuid4(), uuid4()
    db.add_round_task(seed_task)
    db.add_round_task(plain_task)
    db.add_seed(seed_task)

    assert await tool.reverse_tournament_round(TOURNAMENT_ID, ROUND_ID, db) is True

    assert seed_task in db.tasks and plain_task not in db.tasks
    assert db.rows_of(db.task_nodes, seed_task) == {EMISSION_BURN_HOTKEY}


async def test_every_write_goes_through_the_transaction_connection(db, log) -> None:
    task = uuid4()
    db.add_round_task(task)

    assert await tool.reverse_tournament_round(TOURNAMENT_ID, ROUND_ID, db) is True

    # The previous round's reactivation is a write too, and it is on the one connection of the one transaction.
    assert any(sql.startswith("UPDATE tournament_rounds") for sql in db.writes())
    assert db.opened == 1 and db.transactions == 1
