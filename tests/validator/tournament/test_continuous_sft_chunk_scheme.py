"""Continuous-SFT chunk scheme: the one-time cursor conversion and per-task chunk records.

The real Postgres is not available here. Two small in-memory stand-ins cover it:
  - MigrationSim interprets the statements of the migration file itself (up and down), so the
    SQL text is what is tested, and any statement it does not recognise fails the test;
  - FakeContinuousDb plays continuous_sft_state and continuous_sft_task_chunks behind the real
    functions in validator/db/sql/continuous_sft.py (it applies next_train_index_after where the SQL
    CASE does the same arithmetic). The SQL the real advance function sends is asserted separately.
"""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import UUID
from uuid import uuid4

import pytest
from tenacity import stop_after_attempt

import validator.db.constants as cst
from validator.db.sql import continuous_sft as csft
from validator.tasks.synthetics import scheduler
from validator.tournament import constants as t_cst
from validator.tournament.models import ChunkScheme
from validator.tournament.models import ContinuousSftState
from validator.tournament.models import ContinuousSftTaskChunk


LINEAGE = "qwen3-8b"
OTHER_LINEAGE = "alt-lineage"
SEED = "Qwen/Qwen3-8B-Base"
MIGRATION = (
    Path(__file__).resolve().parents[3] / "validator" / "db" / "migrations" / "20261113160000_continuous_sft_chunk_scheme.sql"
)


# ------------------------------------------------------------------------------------------------
# Migration: interpret the file's own statements
# ------------------------------------------------------------------------------------------------
def _statements(section: str) -> list[str]:
    """The section's SQL statements, comments stripped and whitespace collapsed."""
    text = "\n".join(line for line in section.splitlines() if not line.strip().startswith("--"))
    return [" ".join(stmt.split()) for stmt in text.split(";") if stmt.strip()]


def _sections() -> tuple[list[str], list[str]]:
    text = MIGRATION.read_text()
    assert "-- migrate:up" in text and "-- migrate:down" in text
    up, down = text.split("-- migrate:up", 1)[1].split("-- migrate:down")
    return _statements(up), _statements(down)


class MigrationSim:
    """Just enough of Postgres to run this migration against continuous_sft_state rows."""

    def __init__(self, rows: dict[str, dict]) -> None:
        self.rows = rows  # lineage -> {"train_index": int, ["chunk_scheme": str]}
        self.has_column = any("chunk_scheme" in row for row in rows.values())
        self.column_default = "v2" if self.has_column else None
        self.chunks_table_exists = False

    def insert_row(self, lineage: str, train_index: int = 0) -> None:
        """A new row created through the table's defaults."""
        row = {"train_index": train_index}
        if self.has_column:
            row["chunk_scheme"] = self.column_default
        self.rows[lineage] = row

    def run(self, statements: list[str]) -> None:
        for stmt in statements:
            self._run_one(stmt)

    def _run_one(self, stmt: str) -> None:
        if re.fullmatch(
            r"ALTER TABLE continuous_sft_state ADD COLUMN IF NOT EXISTS chunk_scheme TEXT NOT NULL DEFAULT 'v1' "
            r"CHECK \(chunk_scheme IN \('v1', ?'v2'\)\)",
            stmt,
        ):
            if not self.has_column:
                self.has_column = True
                self.column_default = "v1"
                for row in self.rows.values():
                    row["chunk_scheme"] = "v1"
        elif re.fullmatch(
            r"UPDATE continuous_sft_state SET train_index = train_index \* 2, chunk_scheme = 'v2' "
            r"WHERE chunk_scheme = 'v1'",
            stmt,
        ):
            assert self.has_column
            for row in self.rows.values():
                if row["chunk_scheme"] == "v1":
                    row["train_index"] *= 2
                    row["chunk_scheme"] = "v2"
        elif stmt == "ALTER TABLE continuous_sft_state ALTER COLUMN chunk_scheme SET DEFAULT 'v2'":
            assert self.has_column
            self.column_default = "v2"
        elif stmt.startswith("CREATE TABLE IF NOT EXISTS continuous_sft_task_chunks"):
            self.chunks_table_exists = True
        elif stmt == "DROP TABLE IF EXISTS continuous_sft_task_chunks":
            self.chunks_table_exists = False
        elif re.fullmatch(
            r"UPDATE continuous_sft_state SET train_index = \(train_index \+ 1\) / 2 WHERE chunk_scheme = 'v2'", stmt
        ):
            assert self.has_column
            for row in self.rows.values():
                if row["chunk_scheme"] == "v2":
                    row["train_index"] = (row["train_index"] + 1) // 2
        elif stmt == "ALTER TABLE continuous_sft_state DROP COLUMN IF EXISTS chunk_scheme":
            self.has_column = False
            self.column_default = None
            for row in self.rows.values():
                row.pop("chunk_scheme", None)
        else:
            raise AssertionError(f"migration statement not understood by the simulator: {stmt}")


class TestMigration:
    def test_up_doubles_every_row_once(self):
        up, _ = _sections()
        sim = MigrationSim({"a": {"train_index": 5}, "b": {"train_index": 0}, "c": {"train_index": 17}})
        sim.run(up)
        assert {k: v["train_index"] for k, v in sim.rows.items()} == {"a": 10, "b": 0, "c": 34}
        assert all(row["chunk_scheme"] == "v2" for row in sim.rows.values())
        assert sim.chunks_table_exists

    def test_second_application_is_a_no_op(self):
        up, _ = _sections()
        sim = MigrationSim({"a": {"train_index": 5}})
        sim.run(up)
        sim.run(up)
        assert sim.rows == {"a": {"train_index": 10, "chunk_scheme": "v2"}}

    def test_rows_created_afterwards_default_to_the_new_scheme(self):
        up, _ = _sections()
        sim = MigrationSim({"a": {"train_index": 5}})
        sim.run(up)
        sim.insert_row("new")
        assert sim.rows["new"] == {"train_index": 0, "chunk_scheme": "v2"}
        sim.run(up)  # a repeat must not double the new row either
        assert sim.rows["new"]["train_index"] == 0

    def test_default_flips_only_after_the_doubling(self):
        up, _ = _sections()
        flat = [s for s in up if "train_index * 2" in s or "SET DEFAULT" in s or "ADD COLUMN" in s]
        assert ["ADD COLUMN" in flat[0], "train_index * 2" in flat[1], "SET DEFAULT" in flat[2]] == [True] * 3

    @pytest.mark.parametrize("cursor, expected", [(10, 5), (11, 6), (0, 0), (1, 1)])
    def test_down_converts_back_rounding_up(self, cursor, expected):
        up, down = _sections()
        sim = MigrationSim({"a": {"train_index": 5}})
        sim.run(up)
        sim.rows["a"]["train_index"] = cursor  # the cursor after some v2 tasks have run
        sim.run(down)
        assert sim.rows["a"] == {"train_index": expected}

    def test_down_drops_the_table_and_the_column(self):
        up, down = _sections()
        sim = MigrationSim({"a": {"train_index": 5}})
        sim.run(up)
        sim.run(down)
        assert not sim.chunks_table_exists
        assert not sim.has_column
        assert all("chunk_scheme" not in row for row in sim.rows.values())

    def test_chunk_table_cascades_from_tasks(self):
        up, _ = _sections()
        create = next(s for s in up if s.startswith("CREATE TABLE IF NOT EXISTS continuous_sft_task_chunks"))
        assert "task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE" in create
        assert "chunk_scheme TEXT NOT NULL CHECK (chunk_scheme IN ('v1', 'v2'))" in create


# ------------------------------------------------------------------------------------------------
# Fake db behind the real continuous_sft functions
# ------------------------------------------------------------------------------------------------
class FakeContinuousDb:
    """In-memory continuous_sft_state + continuous_sft_task_chunks behind the PSQLDB interface."""

    def __init__(self) -> None:
        self.state: dict[str, dict] = {}
        self.chunks: dict[UUID, ContinuousSftTaskChunk] = {}
        self.statements: list[tuple[str, tuple]] = []
        self.fail_chunk_insert = False

    async def connection(self):
        db = self

        class _Conn:
            def transaction(self):
                tx = MagicMock()
                tx.__aenter__ = AsyncMock(return_value=None)
                tx.__aexit__ = AsyncMock(return_value=False)
                return tx

            async def fetchrow(self, sql, *args):
                return db._fetchrow(sql, args)

            async def execute(self, sql, *args):
                return db._execute(sql, args)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        return _Conn()

    def _fetchrow(self, sql: str, args: tuple):
        self.statements.append((sql, args))
        head = " ".join(sql.split())
        if head.startswith(f"SELECT {cst.CONTINUOUS_SFT_LINEAGE}") and cst.CONTINUOUS_SFT_TASK_CHUNKS_TABLE in head:
            chunk = self.chunks.get(args[0])
            if chunk is None:
                return None
            return {
                cst.CONTINUOUS_SFT_LINEAGE: chunk.lineage,
                cst.CONTINUOUS_SFT_TRAIN_INDEX: chunk.train_index,
                cst.CONTINUOUS_SFT_CHUNK_SCHEME: chunk.chunk_scheme.value,
            }
        if head.startswith("SELECT") and cst.CONTINUOUS_SFT_STATE_TABLE in head:
            row = self.state.get(args[0])
            if row is None:
                return None
            return {
                cst.CONTINUOUS_SFT_TRAIN_INDEX: row["train_index"],
                cst.CONTINUOUS_SFT_CHUNK_SCHEME: row["chunk_scheme"],
                cst.CONTINUOUS_SFT_LAST_WINNER_REPO: row["winner"],
                cst.CONTINUOUS_SFT_UPDATED_AT: None,
            }
        if head.startswith(f"INSERT INTO {cst.CONTINUOUS_SFT_STATE_TABLE}"):
            return self._advance(*args)
        raise AssertionError(f"unexpected fetchrow: {head}")

    def _execute(self, sql: str, args: tuple):
        self.statements.append((sql, args))
        head = " ".join(sql.split())
        if head.startswith(f"INSERT INTO {cst.CONTINUOUS_SFT_TASK_CHUNKS_TABLE}"):
            if self.fail_chunk_insert:
                raise RuntimeError("chunk insert failed")
            task_id, lineage, train_index, scheme = args
            self.chunks.setdefault(
                task_id,
                ContinuousSftTaskChunk(
                    task_id=task_id, lineage=lineage, train_index=train_index, chunk_scheme=ChunkScheme(scheme)
                ),
            )
            return
        raise AssertionError(f"unexpected execute: {head}")

    def _advance(self, lineage, winner, source_round_id, task_index, task_scheme):
        chunk = None
        if task_scheme is not None:
            chunk = ContinuousSftTaskChunk(
                task_id=uuid4(), lineage=lineage, train_index=task_index, chunk_scheme=ChunkScheme(task_scheme)
            )
        row = self.state.get(lineage)
        if row is None:
            first = csft.next_train_index_after(chunk, 0)
            self.state[lineage] = {"train_index": first, "chunk_scheme": "v2", "winner": winner, "round": source_round_id}
        elif row["round"] == source_round_id:
            return None
        else:
            row["train_index"] = csft.next_train_index_after(chunk, row["train_index"])
            row["chunk_scheme"] = "v2"
            row["winner"] = winner if winner is not None else row["winner"]
            row["round"] = source_round_id
        out = self.state[lineage]
        return {cst.CONTINUOUS_SFT_TRAIN_INDEX: out["train_index"], cst.CONTINUOUS_SFT_LAST_WINNER_REPO: out["winner"]}

    # -- test helpers -----------------------------------------------------------------------------
    def seed(self, lineage: str, train_index: int, scheme: str, winner: str | None = "org/w0") -> None:
        self.state[lineage] = {"train_index": train_index, "chunk_scheme": scheme, "winner": winner, "round": None}

    def apply_migration(self) -> None:
        """Run the migration file's up statements over the state rows (previous-scheme rows have no column)."""
        rows = {
            k: {"train_index": v["train_index"], **({} if v["chunk_scheme"] == "v1" else {"chunk_scheme": v["chunk_scheme"]})}
            for k, v in self.state.items()
        }
        up, _ = _sections()
        MigrationSim(rows).run(up)
        for lineage, row in rows.items():
            self.state[lineage]["train_index"] = row["train_index"]
            self.state[lineage]["chunk_scheme"] = row["chunk_scheme"]


def _chunk(task_id: UUID, index: int, scheme: ChunkScheme = ChunkScheme.V2, lineage: str = LINEAGE):
    return ContinuousSftTaskChunk(task_id=task_id, lineage=lineage, train_index=index, chunk_scheme=scheme)


async def _assign_task_id(task, psql_db):
    task.task_id = uuid4()
    return task


def _wire_creation(monkeypatch, db: FakeContinuousDb):
    """Real create_continuous_sft_task over the fake db; content service and add_task faked."""
    served: list[int] = []

    async def content_service(endpoint, keypair, params):
        served.append(params["train_index"])
        return {"train_s3_url": "https://s3/train", "test_s3_url": "https://s3/test", "ds": f"chunk-{params['train_index']}"}

    monkeypatch.setattr(scheduler, "call_content_service", content_service)
    monkeypatch.setattr(scheduler, "add_task", AsyncMock(side_effect=_assign_task_id))
    return SimpleNamespace(psql_db=db, keypair=MagicMock()), served


# ------------------------------------------------------------------------------------------------
# next_train_index_after
# ------------------------------------------------------------------------------------------------
class TestNextTrainIndex:
    def test_v2_task_leaves_next_chunk(self):
        assert csft.next_train_index_after(_chunk(uuid4(), 10, ChunkScheme.V2), 10) == 11

    def test_v1_task_consumes_two_chunks(self):
        assert csft.next_train_index_after(_chunk(uuid4(), 5, ChunkScheme.V1), 10) == 12

    def test_no_record_is_current_plus_two(self):
        assert csft.next_train_index_after(None, 10) == 12

    def test_never_moves_backwards(self):
        assert csft.next_train_index_after(_chunk(uuid4(), 3, ChunkScheme.V2), 10) == 10
        assert csft.next_train_index_after(_chunk(uuid4(), 1, ChunkScheme.V1), 10) == 10


class TestAdvanceSql:
    """What the real function sends; the CASE arithmetic itself needs Postgres (see test_continuous_sft_state)."""

    async def test_single_guarded_statement_mirrors_next_train_index_after(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "v2")
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(_chunk(task_id, 10), db)
        db.statements.clear()
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "r1", db, task_id=task_id)

        inserts = [(sql, args) for sql, args in db.statements if "INSERT INTO continuous_sft_state" in sql]
        assert len(inserts) == 1
        sql, args = inserts[0]
        flat = " ".join(sql.split()).replace("( ", "(")
        assert args == (LINEAGE, "org/a", "r1", 10, "v2")
        assert "IS DISTINCT FROM $3" in flat
        assert "GREATEST(continuous_sft_state.train_index," in flat
        assert "WHEN 'v2' THEN $4::int + 1" in flat
        assert "WHEN 'v1' THEN 2 * $4::int + 2" in flat
        assert "ELSE continuous_sft_state.train_index + 2" in flat
        assert "chunk_scheme = 'v2'" in flat
        assert "COALESCE($2, continuous_sft_state.last_winner_repo)" in flat

    async def test_no_record_passes_null_index_and_scheme(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "v2")
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "r1", db, task_id=uuid4())
        _, args = [(s, a) for s, a in db.statements if "INSERT INTO continuous_sft_state" in s][0]
        assert args == (LINEAGE, "org/a", "r1", None, None)


# ------------------------------------------------------------------------------------------------
# The switch, with tasks in flight
# ------------------------------------------------------------------------------------------------
class TestTaskInFlightAcrossTheSwitch:
    async def test_task_without_record_moves_cursor_to_12(self):
        # Created at previous-scheme index 5 before the migration (no record); the migration then set the cursor to 10.
        db = FakeContinuousDb()
        db.seed(LINEAGE, 5, "v1")
        db.apply_migration()
        assert db.state[LINEAGE]["train_index"] == 10
        await csft.advance_continuous_sft_state(LINEAGE, "org/w", "r1", db, task_id=uuid4())
        assert db.state[LINEAGE]["train_index"] == 12  # never 11: the second chunk of the pair is consumed

    async def test_recorded_previous_scheme_task_moves_cursor_to_12(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 5, "v1")
        db.apply_migration()
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(_chunk(task_id, 5, ChunkScheme.V1), db)
        await csft.advance_continuous_sft_state(LINEAGE, "org/w", "r1", db, task_id=task_id)
        assert db.state[LINEAGE]["train_index"] == 12

    async def test_new_scheme_task_created_at_10_moves_cursor_to_11(self, monkeypatch):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 5, "v1")
        config, served = _wire_creation(monkeypatch, db)
        db.apply_migration()

        task = await scheduler.create_continuous_sft_task(config, LINEAGE, SEED)
        assert served == [10]
        assert await csft.get_continuous_sft_task_chunk(task.task_id, db) == _chunk(task.task_id, 10)
        await csft.advance_continuous_sft_state(LINEAGE, "org/w", "r1", db, task_id=task.task_id)
        assert db.state[LINEAGE]["train_index"] == 11

    async def test_cursor_walks_past_both_chunks_after_the_switch(self, monkeypatch):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 5, "v1")
        config, served = _wire_creation(monkeypatch, db)
        db.apply_migration()
        for round_number in range(1, 5):
            task = await scheduler.create_continuous_sft_task(config, LINEAGE, SEED)
            await csft.advance_continuous_sft_state(LINEAGE, "org/w", f"r{round_number}", db, task_id=task.task_id)
        # Previous-scheme chunk 5 = chunks 10 and 11; chunk 6 = 12 and 13.
        assert served == [10, 11, 12, 13]
        assert (await csft.get_continuous_sft_state(LINEAGE, db)).train_index == 14

    def test_migration_applied_twice_doubles_once(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 5, "v1")
        db.apply_migration()
        db.apply_migration()
        assert db.state[LINEAGE]["train_index"] == 10


# ------------------------------------------------------------------------------------------------
# Advance rules
# ------------------------------------------------------------------------------------------------
class TestAdvanceRules:
    async def test_same_round_twice_changes_nothing(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "v2")
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(_chunk(task_id, 10), db)
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "r1", db, task_id=task_id)
        await csft.advance_continuous_sft_state(LINEAGE, "org/b", "r1", db, task_id=task_id)
        assert db.state[LINEAGE]["train_index"] == 11
        assert db.state[LINEAGE]["winner"] == "org/a"

    async def test_stale_low_task_index_never_moves_cursor_back(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 20, "v2")
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(_chunk(task_id, 3), db)
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "r1", db, task_id=task_id)
        assert db.state[LINEAGE]["train_index"] == 20

    async def test_week_without_winner_still_advances_and_keeps_repo(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "v2", winner="org/prev")
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(_chunk(task_id, 10), db)
        await csft.advance_continuous_sft_state(LINEAGE, None, "r1", db, task_id=task_id)
        assert db.state[LINEAGE]["train_index"] == 11
        assert db.state[LINEAGE]["winner"] == "org/prev"

    async def test_lineages_advance_independently(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "v2")
        db.seed(OTHER_LINEAGE, 40, "v2", winner="org/other")
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(_chunk(task_id, 10), db)
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "round-x", db, task_id=task_id)
        assert db.state[LINEAGE]["train_index"] == 11
        assert db.state[OTHER_LINEAGE]["train_index"] == 40

    async def test_first_insert_without_row(self):
        db = FakeContinuousDb()
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "r1", db)
        assert db.state[LINEAGE]["train_index"] == 2  # no record: next_train_index_after(None, 0)
        assert db.state[LINEAGE]["chunk_scheme"] == "v2"

    async def test_first_insert_with_new_scheme_record(self):
        db = FakeContinuousDb()
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(_chunk(task_id, 0), db)
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "r1", db, task_id=task_id)
        assert db.state[LINEAGE]["train_index"] == 1


# ------------------------------------------------------------------------------------------------
# State and task creation
# ------------------------------------------------------------------------------------------------
class TestStateAndCreation:
    def test_defaults_are_the_new_scheme(self):
        assert ContinuousSftState(lineage=LINEAGE).chunk_scheme == ChunkScheme.V2
        assert t_cst.CONTINUOUS_SFT_CHUNK_SCHEME == ChunkScheme.V2

    async def test_get_state_returns_the_stored_scheme(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "v2")
        assert (await csft.get_continuous_sft_state(LINEAGE, db)).chunk_scheme == ChunkScheme.V2

    async def test_new_lineage_starts_at_zero_in_the_new_scheme(self, monkeypatch):
        db = FakeContinuousDb()
        config, served = _wire_creation(monkeypatch, db)
        task = await scheduler.create_continuous_sft_task(config, LINEAGE, SEED)
        assert served == [0]
        assert db.chunks[task.task_id] == _chunk(task.task_id, 0)

    async def test_deletes_the_task_when_recording_fails(self, monkeypatch):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "v2")
        db.fail_chunk_insert = True
        config, _ = _wire_creation(monkeypatch, db)
        delete_task = AsyncMock()
        monkeypatch.setattr(scheduler, "delete_task", delete_task)
        # One attempt only: the creator's retry wrapper (which re-raises the original error) would otherwise sleep.
        single_attempt = scheduler.create_continuous_sft_task.retry_with(stop=stop_after_attempt(1))
        with pytest.raises(RuntimeError, match="chunk insert failed"):
            await single_attempt(config, LINEAGE, SEED)
        created = [call.args[0].task_id for call in scheduler.add_task.call_args_list]
        assert len(created) == 1
        assert [call.args[0] for call in delete_task.await_args_list] == created
        assert db.chunks == {}
