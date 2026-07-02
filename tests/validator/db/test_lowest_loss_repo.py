"""get_lowest_loss_repo_for_task — the query that picks the carried-forward winner.

Real-Postgres tests (gated on RUN_CONTINUOUS_SFT_DB_TESTS): seed submissions + task_nodes and
prove the guards actually filter. The quality_score > 0 guard is the ONLY thing stopping a
penalized cheat / non-finetune with a low raw test_loss from becoming the permanent lineage base,
so it gets first-class coverage here alongside the NaN/NULL-loss and netuid guards.
"""

import os
import uuid

import pytest
import pytest_asyncio

from validator.db.database import PSQLDB
from validator.db.sql.tasks import NETUID
from validator.db.sql.tasks import get_lowest_loss_repo_for_task


FOREIGN_NETUID = NETUID + 1


@pytest_asyncio.fixture
async def winner_db():
    if not os.getenv("RUN_CONTINUOUS_SFT_DB_TESTS"):
        pytest.skip("set RUN_CONTINUOUS_SFT_DB_TESTS=1 (+ a wipeable DATABASE_URL) to run continuous_sft DB tests")
    db = PSQLDB()
    await db.connect()
    task_ids: list[uuid.UUID] = []
    yield db, task_ids
    async with await db.connection() as conn:
        for tid in task_ids:
            await conn.execute("DELETE FROM task_nodes WHERE task_id=$1", tid)
            await conn.execute("DELETE FROM submissions WHERE task_id=$1", tid)  # FK child of tasks
            await conn.execute("DELETE FROM tasks WHERE task_id=$1", tid)


async def _new_task(db, task_ids) -> uuid.UUID:
    tid = uuid.uuid4()
    async with await db.connection() as conn:
        await conn.execute(
            "INSERT INTO tasks (task_id, model_id, ds, status, account_id, task_type) VALUES ($1,$2,$3,$4,$5,$6)",
            tid,
            "base/model",
            "continuous-sft:qwen:x",
            "success",
            uuid.uuid4(),
            "InstructTextTask",
        )
    task_ids.append(tid)
    return tid


async def _seed_miner(db, task_id, hotkey, repo, *, netuid=NETUID, quality_score=1.0, test_loss=0.5):
    async with await db.connection() as conn:
        await conn.execute(
            "INSERT INTO submissions (task_id, hotkey, netuid, repo) VALUES ($1,$2,$3,$4)",
            task_id,
            hotkey,
            netuid,
            repo,
        )
        await conn.execute(
            "INSERT INTO task_nodes (task_id, hotkey, netuid, quality_score, test_loss) VALUES ($1,$2,$3,$4,$5)",
            task_id,
            hotkey,
            netuid,
            quality_score,
            test_loss,
        )


class TestLowestLossRepo:
    async def test_picks_strictly_min_test_loss(self, winner_db):
        db, ids = winner_db
        tid = await _new_task(db, ids)
        await _seed_miner(db, tid, "hkA", "org/A", test_loss=0.5)
        await _seed_miner(db, tid, "hkB", "org/B", test_loss=0.3)
        await _seed_miner(db, tid, "hkC", "org/C", test_loss=0.9)
        assert await get_lowest_loss_repo_for_task(tid, db) == "org/B"

    async def test_excludes_nonpositive_quality_score(self, winner_db):
        # The cheat guard: a penalized (quality_score=0) repo with the LOWEST raw loss is not carried.
        db, ids = winner_db
        tid = await _new_task(db, ids)
        await _seed_miner(db, tid, "hkCheat", "org/cheat", test_loss=0.1, quality_score=0.0)
        await _seed_miner(db, tid, "hkGood", "org/good", test_loss=0.4, quality_score=1.0)
        assert await get_lowest_loss_repo_for_task(tid, db) == "org/good"

    async def test_excludes_null_quality_score(self, winner_db):
        db, ids = winner_db
        tid = await _new_task(db, ids)
        await _seed_miner(db, tid, "hkNull", "org/null", test_loss=0.1, quality_score=None)
        await _seed_miner(db, tid, "hkGood", "org/good", test_loss=0.4, quality_score=1.0)
        assert await get_lowest_loss_repo_for_task(tid, db) == "org/good"

    async def test_excludes_null_test_loss(self, winner_db):
        # A failed/NaN eval (persisted as NULL) never wins even though it would sort lowest.
        db, ids = winner_db
        tid = await _new_task(db, ids)
        await _seed_miner(db, tid, "hkNoLoss", "org/noloss", test_loss=None, quality_score=1.0)
        await _seed_miner(db, tid, "hkGood", "org/good", test_loss=0.4, quality_score=1.0)
        assert await get_lowest_loss_repo_for_task(tid, db) == "org/good"

    async def test_respects_netuid(self, winner_db):
        db, ids = winner_db
        tid = await _new_task(db, ids)
        await _seed_miner(db, tid, "hkForeign", "org/foreign", test_loss=0.01, netuid=FOREIGN_NETUID)
        await _seed_miner(db, tid, "hkOurs", "org/ours", test_loss=0.5, netuid=NETUID)
        assert await get_lowest_loss_repo_for_task(tid, db) == "org/ours"

    async def test_returns_none_when_no_eligible(self, winner_db):
        db, ids = winner_db
        tid = await _new_task(db, ids)
        await _seed_miner(db, tid, "hkPenalized", "org/x", test_loss=0.1, quality_score=0.0)
        assert await get_lowest_loss_repo_for_task(tid, db) is None
