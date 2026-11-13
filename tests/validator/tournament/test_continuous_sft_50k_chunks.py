"""Continuous-SFT chunk scheme switch.

The real Postgres is not available here, so a small in-memory fake plays the continuous_sft_state row
and the continuous_sft_task_chunks table behind the real functions in validator/db/sql/continuous_sft.py
(the fake applies next_train_index_after where the SQL CASE does the same arithmetic). The SQL text and
parameters the real advance function issues are asserted separately.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import UUID
from uuid import uuid4

import pytest

import validator.db.constants as cst
from core.constants.environments import TrainingStartPoint
from core.models.task_models import TaskType
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
    Path(__file__).resolve().parents[3] / "validator" / "db" / "migrations" / "20261113120000_continuous_sft_50k_chunks.sql"
)


class FakeContinuousDb:
    """In-memory continuous_sft_state + continuous_sft_task_chunks behind the PSQLDB interface."""

    def __init__(self) -> None:
        self.state: dict[str, dict] = {}
        self.chunks: dict[UUID, ContinuousSftTaskChunk] = {}
        self.statements: list[tuple[str, tuple]] = []
        self.fail_chunk_insert = False

    # -- PSQLDB surface -------------------------------------------------------------------------
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

    # -- statement dispatch ---------------------------------------------------------------------
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
            # First insert: a recorded task leaves its own next index, otherwise the cursor starts at 1.
            first = csft.next_train_index_after(chunk, 0) if chunk else 1
            self.state[lineage] = {
                "train_index": first,
                "chunk_scheme": "50k",
                "winner": winner,
                "round": source_round_id,
            }
        elif row["round"] == source_round_id:
            return None
        else:
            row["train_index"] = csft.next_train_index_after(chunk, row["train_index"])
            row["chunk_scheme"] = "50k"
            row["winner"] = winner if winner is not None else row["winner"]
            row["round"] = source_round_id
        out = self.state[lineage]
        return {cst.CONTINUOUS_SFT_TRAIN_INDEX: out["train_index"], cst.CONTINUOUS_SFT_LAST_WINNER_REPO: out["winner"]}

    # -- test helpers ---------------------------------------------------------------------------
    def seed(self, lineage: str, train_index: int, scheme: str, winner: str | None = "org/w0") -> None:
        self.state[lineage] = {"train_index": train_index, "chunk_scheme": scheme, "winner": winner, "round": None}

    def apply_migration(self) -> None:
        """What the migration's UPDATE does; the WHERE makes a repeat a no-op."""
        for row in self.state.values():
            if row["chunk_scheme"] == "100k":
                row["train_index"] *= 2
                row["chunk_scheme"] = "50k"


async def _assign_task_id(task, psql_db):
    """What add_task does to the task it is given: assign its id."""
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
    config = SimpleNamespace(psql_db=db, keypair=MagicMock())
    return config, served


async def _create(config, lineage=LINEAGE):
    return await scheduler.create_continuous_sft_task(config, lineage, SEED)


# ------------------------------------------------------------------------------------------------
# Migration file
# ------------------------------------------------------------------------------------------------
class TestMigrationFile:
    def _up(self) -> str:
        text = MIGRATION.read_text()
        assert "-- migrate:up" in text and "-- migrate:down" in text
        return text.split("-- migrate:down")[0]

    def test_doubles_once_guarded_by_scheme(self):
        up = self._up()
        assert up.count("train_index * 2") == 1
        assert "SET train_index = train_index * 2, chunk_scheme = '50k' WHERE chunk_scheme = '100k'" in " ".join(up.split())

    def test_adds_column_then_flips_default(self):
        flat = " ".join(self._up().split())
        add = "ADD COLUMN IF NOT EXISTS chunk_scheme TEXT NOT NULL DEFAULT '100k' CHECK (chunk_scheme IN ('100k', '50k'))"
        flip = "ALTER COLUMN chunk_scheme SET DEFAULT '50k'"
        assert add in flat and flip in flat
        assert flat.index(add) < flat.index("train_index * 2") < flat.index(flip)

    def test_creates_task_chunks_table(self):
        flat = " ".join(self._up().split())
        assert "CREATE TABLE IF NOT EXISTS continuous_sft_task_chunks" in flat
        assert "task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE" in flat
        assert "chunk_scheme TEXT NOT NULL CHECK (chunk_scheme IN ('100k', '50k'))" in flat

    def test_header_explains_the_deploy_window(self):
        header = MIGRATION.read_text().split("ALTER TABLE")[0]
        assert "IS the continuous-SFT chunk scheme switch" in header
        assert "between two tournaments" in header

    def test_down_reverses(self):
        down = " ".join(MIGRATION.read_text().split("-- migrate:down")[1].split())
        assert "DROP TABLE IF EXISTS continuous_sft_task_chunks" in down
        assert "SET train_index = train_index / 2 WHERE chunk_scheme = '50k'" in down
        assert "DROP COLUMN IF EXISTS chunk_scheme" in down


# ------------------------------------------------------------------------------------------------
# next_train_index_after
# ------------------------------------------------------------------------------------------------
def _chunk(index: int, scheme: ChunkScheme) -> ContinuousSftTaskChunk:
    return ContinuousSftTaskChunk(task_id=uuid4(), lineage=LINEAGE, train_index=index, chunk_scheme=scheme)


class TestNextTrainIndex:
    def test_fifty_k_task_leaves_next_chunk(self):
        assert csft.next_train_index_after(_chunk(10, ChunkScheme.FIFTY_K), 10) == 11

    def test_hundred_k_task_consumes_both_halves(self):
        assert csft.next_train_index_after(_chunk(5, ChunkScheme.HUNDRED_K), 10) == 12

    def test_no_record_is_current_plus_two(self):
        assert csft.next_train_index_after(None, 10) == 12

    def test_never_moves_backwards(self):
        assert csft.next_train_index_after(_chunk(3, ChunkScheme.FIFTY_K), 10) == 10
        assert csft.next_train_index_after(_chunk(1, ChunkScheme.HUNDRED_K), 10) == 10


# ------------------------------------------------------------------------------------------------
# Switch and advance flows
# ------------------------------------------------------------------------------------------------
class TestSwitchBetweenTournaments:
    async def test_cursor_doubles_then_walks_halves(self, monkeypatch):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 5, "100k")
        config, served = _wire_creation(monkeypatch, db)

        db.apply_migration()
        assert (await csft.get_continuous_sft_state(LINEAGE, db)).train_index == 10

        for round_number in range(1, 5):
            task = await _create(config)
            await csft.advance_continuous_sft_state(LINEAGE, "org/w", f"r{round_number}", db, task_id=task.task_id)

        # Previous-scheme chunk 5 = chunks 10 and 11; chunk 6 = 12 and 13.
        assert served == [10, 11, 12, 13]
        assert [i % 2 for i in served] == [0, 1, 0, 1]
        assert (await csft.get_continuous_sft_state(LINEAGE, db)).train_index == 14

    async def test_first_task_after_switch_is_recorded_in_50k(self, monkeypatch):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 5, "100k")
        config, _ = _wire_creation(monkeypatch, db)
        db.apply_migration()

        task = await _create(config)
        record = await csft.get_continuous_sft_task_chunk(task.task_id, db)
        assert record == ContinuousSftTaskChunk(
            task_id=task.task_id, lineage=LINEAGE, train_index=10, chunk_scheme=ChunkScheme.FIFTY_K
        )
        await csft.advance_continuous_sft_state(LINEAGE, "org/w", "r1", db, task_id=task.task_id)
        assert db.state[LINEAGE]["train_index"] == 11

    def test_migration_twice_doubles_once(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 5, "100k")
        db.apply_migration()
        db.apply_migration()
        assert db.state[LINEAGE]["train_index"] == 10


class TestTaskInFlightAcrossTheSwitch:
    async def test_task_without_record_moves_cursor_to_12(self):
        # Created at previous-scheme index 5 before the migration (no record); the migration then set the cursor to 10.
        db = FakeContinuousDb()
        db.seed(LINEAGE, 5, "100k")
        db.apply_migration()
        await csft.advance_continuous_sft_state(LINEAGE, "org/w", "r1", db, task_id=uuid4())
        assert db.state[LINEAGE]["train_index"] == 12  # never 11: the second half of chunk 5 is consumed

    async def test_task_with_explicit_100k_record_moves_cursor_to_12(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 5, "100k")
        db.apply_migration()
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(
            ContinuousSftTaskChunk(task_id=task_id, lineage=LINEAGE, train_index=5, chunk_scheme=ChunkScheme.HUNDRED_K), db
        )
        await csft.advance_continuous_sft_state(LINEAGE, "org/w", "r1", db, task_id=task_id)
        assert db.state[LINEAGE]["train_index"] == 12


class TestAdvanceRules:
    async def test_same_round_twice_changes_nothing(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "50k")
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(_chunk_for(task_id, 10), db)
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "r1", db, task_id=task_id)
        await csft.advance_continuous_sft_state(LINEAGE, "org/b", "r1", db, task_id=task_id)
        assert db.state[LINEAGE]["train_index"] == 11
        assert db.state[LINEAGE]["winner"] == "org/a"

    async def test_stale_low_task_index_never_moves_cursor_back(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 20, "50k")
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(_chunk_for(task_id, 3), db)
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "r1", db, task_id=task_id)
        assert db.state[LINEAGE]["train_index"] == 20

    async def test_lineages_advance_independently(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "50k")
        db.seed(OTHER_LINEAGE, 40, "50k", winner="org/other")
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(_chunk_for(task_id, 10), db)
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "round-x", db, task_id=task_id)
        assert db.state[LINEAGE]["train_index"] == 11
        assert db.state[OTHER_LINEAGE]["train_index"] == 40
        other_task = uuid4()
        await csft.record_continuous_sft_task_chunk(
            ContinuousSftTaskChunk(task_id=other_task, lineage=OTHER_LINEAGE, train_index=40, chunk_scheme=ChunkScheme.FIFTY_K),
            db,
        )
        await csft.advance_continuous_sft_state(OTHER_LINEAGE, "org/b", "round-x", db, task_id=other_task)
        assert db.state[OTHER_LINEAGE]["train_index"] == 41
        assert db.state[LINEAGE]["train_index"] == 11

    async def test_week_without_winner_still_advances_and_keeps_repo(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "50k", winner="org/prev")
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(_chunk_for(task_id, 10), db)
        await csft.advance_continuous_sft_state(LINEAGE, None, "r1", db, task_id=task_id)
        assert db.state[LINEAGE]["train_index"] == 11
        assert db.state[LINEAGE]["winner"] == "org/prev"

    async def test_first_insert_without_row(self):
        db = FakeContinuousDb()
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "r1", db)
        assert db.state[LINEAGE]["train_index"] == 1
        assert db.state[LINEAGE]["chunk_scheme"] == "50k"


def _chunk_for(task_id: UUID, index: int) -> ContinuousSftTaskChunk:
    return ContinuousSftTaskChunk(task_id=task_id, lineage=LINEAGE, train_index=index, chunk_scheme=ChunkScheme.FIFTY_K)


class TestAdvanceSql:
    """What the real function sends; the CASE arithmetic itself needs Postgres (see test_continuous_sft_state)."""

    async def test_single_guarded_statement_with_params(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "50k")
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(_chunk_for(task_id, 10), db)
        db.statements.clear()
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "r1", db, task_id=task_id)

        inserts = [(sql, args) for sql, args in db.statements if "INSERT INTO continuous_sft_state" in sql]
        assert len(inserts) == 1
        sql, args = inserts[0]
        flat = " ".join(sql.split()).replace("( ", "(")
        assert args == (LINEAGE, "org/a", "r1", 10, "50k")
        assert "IS DISTINCT FROM $3" in flat
        assert "GREATEST(continuous_sft_state.train_index," in flat
        assert "WHEN '50k' THEN $4::int + 1" in flat
        assert "WHEN '100k' THEN 2 * $4::int + 2" in flat
        assert "ELSE continuous_sft_state.train_index + 2" in flat
        assert "chunk_scheme = '50k'" in flat
        assert "COALESCE($2, continuous_sft_state.last_winner_repo)" in flat

    async def test_no_record_passes_null_index_and_scheme(self):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "50k")
        await csft.advance_continuous_sft_state(LINEAGE, "org/a", "r1", db, task_id=uuid4())
        _, args = [(s, a) for s, a in db.statements if "INSERT INTO continuous_sft_state" in s][0]
        assert args == (LINEAGE, "org/a", "r1", None, None)


# ------------------------------------------------------------------------------------------------
# create_continuous_sft_task
# ------------------------------------------------------------------------------------------------
class TestCreateTask:
    async def test_refuses_a_100k_scheme_state(self, monkeypatch):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 5, "100k")
        config, served = _wire_creation(monkeypatch, db)
        with pytest.raises(RuntimeError, match="20261113120000_continuous_sft_50k_chunks"):
            await _create(config)
        assert served == []  # never sent a previous-scheme index to the content service
        scheduler.add_task.assert_not_called()

    async def test_new_lineage_starts_at_zero_in_50k(self, monkeypatch):
        db = FakeContinuousDb()
        config, served = _wire_creation(monkeypatch, db)
        task = await _create(config)
        assert served == [0]
        assert db.chunks[task.task_id].chunk_scheme == ChunkScheme.FIFTY_K

    async def test_deletes_the_task_when_recording_fails(self, monkeypatch):
        db = FakeContinuousDb()
        db.seed(LINEAGE, 10, "50k")
        db.fail_chunk_insert = True
        config, _ = _wire_creation(monkeypatch, db)
        delete_task = AsyncMock()
        monkeypatch.setattr(scheduler.task_sql, "delete_task", delete_task)
        with pytest.raises(RuntimeError, match="chunk insert failed"):
            await _create(config)
        # The creator retries on failure; every task it created along the way is deleted again.
        created_ids = [call.args[0].task_id for call in scheduler.add_task.call_args_list]
        assert created_ids and [call.args[0] for call in delete_task.await_args_list] == created_ids
        assert db.chunks == {}


def test_state_default_is_50k_scheme():
    assert ContinuousSftState(lineage=LINEAGE).chunk_scheme == ChunkScheme.FIFTY_K
    assert t_cst.CONTINUOUS_SFT_CHUNK_SCHEME == ChunkScheme.FIFTY_K


# ------------------------------------------------------------------------------------------------
# Hours
# ------------------------------------------------------------------------------------------------
def _stats(total_tokens: int, num_records: int, tokens_per_sec: float | None):
    throughput = SimpleNamespace(tokens_per_sec=tokens_per_sec) if tokens_per_sec else None
    return SimpleNamespace(dataset=SimpleNamespace(total_tokens=total_tokens, num_records=num_records), throughput=throughput)


class TestPinnedHours:
    def _hours(self, stats, max_hours=None, current_hours=4.0):
        return scheduler.compute_hours_from_baseline_stats(
            current_hours,
            stats,
            TaskType.CHATTASK,
            model_id=SEED,
            model_params_count=8_600_000_000,
            training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
            ds=t_cst.continuous_sft_ds(LINEAGE, "chunk-00010"),
            max_hours=max_hours,
        )

    def test_constant_is_two_and_a_half(self):
        assert t_cst.CONTINUOUS_SFT_TRAINING_HOURS == 2.5

    def test_no_stats(self):
        assert self._hours(None) == 2.5

    def test_stats_that_would_compute_far_more(self):
        assert self._hours(_stats(50_000_000_000, 5_000_000, 500.0)) == 2.5

    def test_stats_that_would_compute_far_less(self):
        assert self._hours(_stats(1_000, 10, 1_000_000.0), current_hours=30.0) == 2.5

    @pytest.mark.parametrize("max_hours", [1.0, 2.0, 3.0, 10.0, 48.0])
    def test_max_hours_does_not_change_it(self, max_hours):
        assert self._hours(_stats(200_000_000, 100_000, 8_000.0), max_hours=max_hours) == 2.5

    def test_start_point_alone_or_ds_alone_identifies_it(self):
        stats = _stats(200_000_000, 100_000, 8_000.0)
        by_start_point = scheduler.compute_hours_from_baseline_stats(
            9.0,
            stats,
            TaskType.CHATTASK,
            model_params_count=8_600_000_000,
            training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
        )
        by_ds = scheduler.compute_hours_from_baseline_stats(
            9.0, stats, TaskType.CHATTASK, model_params_count=8_600_000_000, ds=t_cst.continuous_sft_ds(LINEAGE, "x")
        )
        assert by_start_point == by_ds == 2.5

    def test_plain_instruct_task_is_unchanged(self):
        stats = _stats(200_000_000, 100_000, 8_000.0)
        expected = scheduler.compute_training_hours(
            max(200_000_000, 100_000 * scheduler.data_cst.EFFECTIVE_MIN_TOKENS_PER_ROW),
            8_600_000_000,
            TaskType.INSTRUCTTEXTTASK,
            8_000.0,
            TrainingStartPoint.DEFAULT,
            max_hours=None,
            gpu_count_multiplier=t_cst.gpu_count_multiplier(TaskType.INSTRUCTTEXTTASK, None),
        )
        got = scheduler.compute_hours_from_baseline_stats(
            4.0, stats, TaskType.INSTRUCTTEXTTASK, model_id="org/model", model_params_count=8_600_000_000
        )
        assert got == expected
