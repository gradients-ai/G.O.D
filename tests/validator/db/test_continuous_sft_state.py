"""continuous_sft_state read/advance: mock-branch tests (run anywhere) + real-Postgres semantics.

The mock tests cover the Python-side branches (missing-row default, row mapping, idempotent-skip
return, orphan warning). The real-DB tests cover the SQL semantics that mocks CANNOT prove and that
are the scariest failure modes of the whole feature: the idempotency guard (no double +1 on
reprocess), COALESCE winner-preserve on an empty week, and first-advance == 1. They are gated on
RUN_CONTINUOUS_SFT_DB_TESTS=1 (+ a wipeable DATABASE_URL) so a normal `pytest` never touches a DB.
"""

import logging
import os

import pytest
import pytest_asyncio

import validator.db.constants as cst
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from validator.db.database import PSQLDB
from validator.db.sql import continuous_sft as csft
from validator.tournament.models import ContinuousSftState


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
            cst.CONTINUOUS_SFT_LAST_WINNER_REPO: "org/x",
            cst.CONTINUOUS_SFT_UPDATED_AT: None,
        }
        psql, _ = _mock_psql(fetchrow=row)
        state = await csft.get_continuous_sft_state("qwen", psql)
        assert (state.lineage, state.train_index, state.last_winner_repo) == ("qwen", 7, "org/x")

    async def test_advance_idempotent_skip_returns_without_raising(self):
        # RETURNING produced no row (same source_round_id) -> function no-ops, does not raise.
        psql, conn = _mock_psql(fetchrow=None)
        result = await csft.advance_continuous_sft_state("qwen", "org/x", "round-1", psql)
        assert result is None
        assert conn.fetchrow.await_args.args[1:] == ("qwen", "org/x", "round-1")

    async def test_advance_success_passes_params_in_order(self):
        psql, conn = _mock_psql(
            fetchrow={cst.CONTINUOUS_SFT_TRAIN_INDEX: 8, cst.CONTINUOUS_SFT_LAST_WINNER_REPO: "org/x"}
        )
        await csft.advance_continuous_sft_state("qwen", "org/x", "round-9", psql)
        assert conn.fetchrow.await_args.args[1:] == ("qwen", "org/x", "round-9")

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
            await csft.warn_orphaned_continuous_sft_state({"quasar", "qwen"}, psql)
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
"""


@pytest_asyncio.fixture
async def cst_db():
    if not os.getenv("RUN_CONTINUOUS_SFT_DB_TESTS"):
        pytest.skip("set RUN_CONTINUOUS_SFT_DB_TESTS=1 (+ a wipeable DATABASE_URL) to run continuous_sft DB tests")
    db = PSQLDB()
    await db.connect()
    async with await db.connection() as conn:
        await conn.execute(_SCHEMA)
        await conn.execute("TRUNCATE continuous_sft_state")
    yield db
    async with await db.connection() as conn:
        await conn.execute("TRUNCATE continuous_sft_state")


class TestStateRealDb:
    async def test_first_advance_from_empty_sets_index_one(self, cst_db):
        await csft.advance_continuous_sft_state("qwen", "org/a", "r1", cst_db)
        state = await csft.get_continuous_sft_state("qwen", cst_db)
        assert state.train_index == 1
        assert state.last_winner_repo == "org/a"

    async def test_double_advance_same_round_is_idempotent(self, cst_db):
        # The scariest guard: reprocessing the same round must NOT double-increment or overwrite.
        await csft.advance_continuous_sft_state("qwen", "org/a", "r1", cst_db)
        await csft.advance_continuous_sft_state("qwen", "org/b", "r1", cst_db)  # same round id
        state = await csft.get_continuous_sft_state("qwen", cst_db)
        assert state.train_index == 1
        assert state.last_winner_repo == "org/a"

    async def test_new_round_increments(self, cst_db):
        await csft.advance_continuous_sft_state("qwen", "org/a", "r1", cst_db)
        await csft.advance_continuous_sft_state("qwen", "org/b", "r2", cst_db)
        state = await csft.get_continuous_sft_state("qwen", cst_db)
        assert state.train_index == 2
        assert state.last_winner_repo == "org/b"

    async def test_none_winner_preserves_prior_via_coalesce(self, cst_db):
        # An empty/failed week (winner=None) advances the cursor but keeps the accumulated base.
        await csft.advance_continuous_sft_state("qwen", "org/a", "r1", cst_db)
        await csft.advance_continuous_sft_state("qwen", None, "r2", cst_db)
        state = await csft.get_continuous_sft_state("qwen", cst_db)
        assert state.train_index == 2
        assert state.last_winner_repo == "org/a"

    async def test_lineages_advance_independently(self, cst_db):
        # Same round id across lineages is fine (the guard is per-row); each tracks its own cursor.
        await csft.advance_continuous_sft_state("quasar", "org/q", "r1", cst_db)
        await csft.advance_continuous_sft_state("qwen", "org/w", "r1", cst_db)
        quasar = await csft.get_continuous_sft_state("quasar", cst_db)
        qwen = await csft.get_continuous_sft_state("qwen", cst_db)
        assert (quasar.train_index, quasar.last_winner_repo) == (1, "org/q")
        assert (qwen.train_index, qwen.last_winner_repo) == (1, "org/w")
