"""continuous_sft_state read/advance: mock-branch tests (run anywhere) + real-Postgres semantics.

The mock tests cover the Python-side branches (missing-row default, row mapping, idempotent-skip
return, orphan warning). The real-DB tests cover the SQL semantics that mocks CANNOT prove and that
are the scariest failure modes of the whole feature: the idempotency guard (no double +1 on
reprocess), COALESCE winner-preserve on an empty week, and the first-advance value (2 with no chunk record). They are gated on
RUN_CONTINUOUS_SFT_DB_TESTS=1 (+ a wipeable DATABASE_URL) so a normal `pytest` never touches a DB.
"""

import logging
import os
from uuid import uuid4

import pytest
import pytest_asyncio

import validator.db.constants as cst
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from validator.db.database import PSQLDB
from validator.db.sql import continuous_sft as csft
from validator.tournament.models import ChunkScheme
from validator.tournament.models import ContinuousSftState
from validator.tournament.models import ContinuousSftTaskChunk


# --------------------------------------------------------------------------------------------------
# Mock-branch tests (no DB): assert the Python control flow, not the SQL semantics.
# --------------------------------------------------------------------------------------------------
def _mock_psql(*, fetchrow=None, fetch=None):
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow)
    conn.fetch = AsyncMock(return_value=fetch or [])
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=None)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    psql = MagicMock()
    psql.connection = AsyncMock(return_value=cm)
    return psql, conn


class TestStateMockBranches:
    async def test_get_state_missing_row_defaults_to_seed(self):
        psql, _ = _mock_psql(fetchrow=None)
        state = await csft.get_continuous_sft_state("qwen", psql)
        assert state == ContinuousSftState(lineage="qwen", train_index=0, last_winner_repo=None)

    async def test_get_state_maps_row_fields(self):
        row = {
            cst.CONTINUOUS_SFT_TRAIN_INDEX: 7,
            cst.CONTINUOUS_SFT_CHUNK_SCHEME: "50k",
            cst.CONTINUOUS_SFT_LAST_WINNER_REPO: "org/x",
            cst.CONTINUOUS_SFT_UPDATED_AT: None,
        }
        psql, _ = _mock_psql(fetchrow=row)
        state = await csft.get_continuous_sft_state("qwen", psql)
        assert (state.lineage, state.train_index, state.last_winner_repo) == ("qwen", 7, "org/x")
        assert state.chunk_scheme == ChunkScheme.FIFTY_K

    async def test_advance_idempotent_skip_returns_without_raising(self):
        # RETURNING produced no row (same source_round_id) -> function no-ops, does not raise.
        psql, conn = _mock_psql(fetchrow=None)
        result = await csft.advance_continuous_sft_state("qwen", "org/x", "round-1", psql)
        assert result is None
        assert conn.fetchrow.await_args.args[1:] == ("qwen", "org/x", "round-1", None, None)

    async def test_advance_success_passes_params_in_order(self):
        psql, conn = _mock_psql(
            fetchrow={cst.CONTINUOUS_SFT_TRAIN_INDEX: 8, cst.CONTINUOUS_SFT_LAST_WINNER_REPO: "org/x"}
        )
        await csft.advance_continuous_sft_state("qwen", "org/x", "round-9", psql)
        assert conn.fetchrow.await_args.args[1:] == ("qwen", "org/x", "round-9", None, None)

    async def test_warn_orphaned_warns_only_unknown_lineage(self):
        rows = [
            {cst.CONTINUOUS_SFT_LINEAGE: "qwen", cst.CONTINUOUS_SFT_TRAIN_INDEX: 3, cst.CONTINUOUS_SFT_LAST_WINNER_REPO: "r"},
            {cst.CONTINUOUS_SFT_LINEAGE: "old_slug", cst.CONTINUOUS_SFT_TRAIN_INDEX: 9, cst.CONTINUOUS_SFT_LAST_WINNER_REPO: "r2"},
        ]
        psql, _ = _mock_psql(fetch=rows)

        captured = []
        handler = logging.Handler()
        handler.emit = lambda record: captured.append(record.getMessage())
        # Attach to the module's own logger object (fiber-configured, propagate=False), not by name.
        prev_level = csft.logger.level
        csft.logger.addHandler(handler)
        csft.logger.setLevel(logging.WARNING)
        try:
            await csft.warn_orphaned_continuous_sft_state({"alt", "qwen"}, psql)
        finally:
            csft.logger.removeHandler(handler)
            csft.logger.setLevel(prev_level)

        joined = "\n".join(captured)
        assert "old_slug" in joined
        assert "qwen" not in joined  # the known lineage is not flagged


# --------------------------------------------------------------------------------------------------
# Real-Postgres semantics — the SQL that mocks cannot exercise. Run on the node:
#   RUN_CONTINUOUS_SFT_DB_TESTS=1 DATABASE_URL=... pytest tests/validator/db/test_continuous_sft_state.py
# --------------------------------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS continuous_sft_state (
    lineage TEXT PRIMARY KEY,
    train_index INT NOT NULL DEFAULT 0,
    last_winner_repo TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE continuous_sft_state ADD COLUMN IF NOT EXISTS last_source_round_id TEXT;
ALTER TABLE continuous_sft_state ADD COLUMN IF NOT EXISTS chunk_scheme TEXT NOT NULL DEFAULT '50k';
CREATE TABLE IF NOT EXISTS continuous_sft_task_chunks (
    task_id UUID PRIMARY KEY,
    lineage TEXT NOT NULL,
    train_index INT NOT NULL,
    chunk_scheme TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


@pytest_asyncio.fixture
async def cst_db():
    if not os.getenv("RUN_CONTINUOUS_SFT_DB_TESTS"):
        pytest.skip("set RUN_CONTINUOUS_SFT_DB_TESTS=1 (+ a wipeable DATABASE_URL) to run continuous_sft DB tests")
    db = PSQLDB()
    await db.connect()
    async with await db.connection() as conn:
        await conn.execute(_SCHEMA)
        await conn.execute("TRUNCATE continuous_sft_state, continuous_sft_task_chunks")
    yield db
    async with await db.connection() as conn:
        await conn.execute("TRUNCATE continuous_sft_state, continuous_sft_task_chunks")


class TestStateRealDb:
    async def test_first_advance_from_empty_without_record_sets_index_two(self, cst_db):
        await csft.advance_continuous_sft_state("qwen", "org/a", "r1", cst_db)
        state = await csft.get_continuous_sft_state("qwen", cst_db)
        assert state.train_index == 2
        assert state.last_winner_repo == "org/a"

    async def test_double_advance_same_round_is_idempotent(self, cst_db):
        # The scariest guard: reprocessing the same round must NOT double-increment or overwrite.
        await csft.advance_continuous_sft_state("qwen", "org/a", "r1", cst_db)
        await csft.advance_continuous_sft_state("qwen", "org/b", "r1", cst_db)  # same round id
        state = await csft.get_continuous_sft_state("qwen", cst_db)
        assert state.train_index == 2
        assert state.last_winner_repo == "org/a"

    async def test_new_round_increments(self, cst_db):
        await csft.advance_continuous_sft_state("qwen", "org/a", "r1", cst_db)
        await csft.advance_continuous_sft_state("qwen", "org/b", "r2", cst_db)
        state = await csft.get_continuous_sft_state("qwen", cst_db)
        assert state.train_index == 4  # no task record: first insert 2, then +2
        assert state.last_winner_repo == "org/b"

    async def test_none_winner_preserves_prior_via_coalesce(self, cst_db):
        # An empty/failed week (winner=None) advances the cursor but keeps the accumulated base.
        await csft.advance_continuous_sft_state("qwen", "org/a", "r1", cst_db)
        await csft.advance_continuous_sft_state("qwen", None, "r2", cst_db)
        state = await csft.get_continuous_sft_state("qwen", cst_db)
        assert state.train_index == 4  # no task record: first insert 2, then +2
        assert state.last_winner_repo == "org/a"

    async def test_lineages_advance_independently(self, cst_db):
        # Same round id across lineages is fine (the guard is per-row); each tracks its own cursor.
        await csft.advance_continuous_sft_state("alt", "org/q", "r1", cst_db)
        await csft.advance_continuous_sft_state("qwen", "org/w", "r1", cst_db)
        alt = await csft.get_continuous_sft_state("alt", cst_db)
        qwen = await csft.get_continuous_sft_state("qwen", cst_db)
        assert (alt.train_index, alt.last_winner_repo) == (2, "org/q")
        assert (qwen.train_index, qwen.last_winner_repo) == (2, "org/w")

    async def _advance_with_record(self, db, index, scheme, round_id="r-rec", *, seed_cursor=None):
        task_id = uuid4()
        await csft.record_continuous_sft_task_chunk(
            ContinuousSftTaskChunk(task_id=task_id, lineage="qwen", train_index=index, chunk_scheme=scheme), db
        )
        if seed_cursor is not None:
            async with await db.connection() as conn:
                await conn.execute(
                    "INSERT INTO continuous_sft_state (lineage, train_index, last_source_round_id) VALUES ('qwen', $1, 'seed')",
                    seed_cursor,
                )
        await csft.advance_continuous_sft_state("qwen", "org/a", round_id, db, task_id=task_id)
        return (await csft.get_continuous_sft_state("qwen", db)).train_index

    async def test_new_scheme_record_moves_cursor_past_its_chunk(self, cst_db):
        assert await self._advance_with_record(cst_db, 10, ChunkScheme.FIFTY_K, seed_cursor=10) == 11

    async def test_previous_scheme_record_consumes_both_halves(self, cst_db):
        assert await self._advance_with_record(cst_db, 5, ChunkScheme.HUNDRED_K, seed_cursor=10) == 12

    async def test_stale_new_scheme_record_never_moves_cursor_back(self, cst_db):
        assert await self._advance_with_record(cst_db, 3, ChunkScheme.FIFTY_K, seed_cursor=20) == 20

    async def test_first_insert_with_new_scheme_record(self, cst_db):
        assert await self._advance_with_record(cst_db, 0, ChunkScheme.FIFTY_K) == 1
