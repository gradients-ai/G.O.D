"""Trainer GPU reservations against a real Postgres: a job's end releases exactly the GPUs it still holds, and the
reconciliation with the trainer's report frees idle GPUs without freeing one another job holds.

Gated on RUN_GPU_RESERVATION_DB_TESTS=1 (+ a wipeable DATABASE_URL) so a normal `pytest` never touches a DB:
    RUN_GPU_RESERVATION_DB_TESTS=1 DATABASE_URL=... pytest tests/validator/db/test_gpu_reservations.py
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from core.models.trainer_contract_models import GPUInfo
from core.models.trainer_contract_models import GPUType
from validator.db.database import PSQLDB
from validator.db.sql import tournaments as tournament_sql
from validator.tournament import constants as t_cst
from validator.tournament import orchestrator
from validator.tournament.gpu_reservations import prep_job_key
from validator.tournament.gpu_reservations import training_job_key
from validator.tournament.models import TrainingStatus


TRAINER = "10.0.0.1"
OTHER_TRAINER = "10.0.0.2"
TASK = "00000000-0000-0000-0000-000000000001"
HOTKEY = "hk-a"
OTHER_HOTKEY = "hk-b"
GRACE = t_cst.GPU_RESERVATION_GRACE_PERIOD_SECONDS

MIGRATIONS = Path(__file__).resolve().parents[3] / "validator" / "db" / "migrations"
# The trainers_gpus migrations, in order (the GPU cost triggers need other tables and are left out).
TRAINERS_GPUS_MIGRATIONS = [
    "20250625131001_add_trainers_gpus_table.sql",
    "20260907120000_add_gpu_interconnect_to_trainers_gpus.sql",
    "20261020040000_add_nvlink_peers_to_trainers_gpus.sql",
    "20261124120000_add_gpu_reservation_owner.sql",
]
TRAININGS_SCHEMA = """
CREATE TABLE tournament_task_hotkey_trainings (
    task_id UUID NOT NULL,
    hotkey TEXT NOT NULL,
    training_status TEXT NOT NULL,
    n_training_attempts INT NOT NULL DEFAULT 0,
    trainer_ip TEXT,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, hotkey)
);
"""


def _migrate_up(name: str) -> str:
    sql = (MIGRATIONS / name).read_text()
    return sql.split("-- migrate:up", 1)[1].split("-- migrate:down", 1)[0]


@pytest_asyncio.fixture
async def db():
    if not os.getenv("RUN_GPU_RESERVATION_DB_TESTS"):
        pytest.skip("set RUN_GPU_RESERVATION_DB_TESTS=1 (+ a wipeable DATABASE_URL) to run the GPU reservation DB tests")
    database = PSQLDB()
    await database.connect()
    async with await database.connection() as conn:
        await conn.execute("DROP TABLE IF EXISTS trainers_gpus, tournament_task_hotkey_trainings")
        for name in TRAINERS_GPUS_MIGRATIONS:
            await conn.execute(_migrate_up(name))
        await conn.execute(TRAININGS_SCHEMA)
        for trainer_ip in (TRAINER, OTHER_TRAINER):
            for gpu_id in range(4):
                await conn.execute(
                    "INSERT INTO trainers_gpus (trainer_ip, gpu_id, gpu_type, vram_gb, used_until)"
                    " VALUES ($1, $2, 'H100', 80, NULL)",
                    trainer_ip,
                    gpu_id,
                )
        for hotkey in (HOTKEY, OTHER_HOTKEY):
            await conn.execute(
                "INSERT INTO tournament_task_hotkey_trainings (task_id, hotkey, training_status) VALUES ($1, $2, 'pending')",
                TASK,
                hotkey,
            )
    yield database
    async with await database.connection() as conn:
        await conn.execute("DROP TABLE IF EXISTS trainers_gpus, tournament_task_hotkey_trainings")
    await database.close()


async def _reserved(db: PSQLDB, trainer_ip: str = TRAINER) -> dict[int, str | None]:
    """GPU id -> holder of a live reservation (None for a hold without a job key), for the reserved GPUs only."""
    async with await db.connection() as conn:
        rows = await conn.fetch(
            "SELECT gpu_id, reserved_by FROM trainers_gpus WHERE trainer_ip = $1 AND used_until > now() ORDER BY gpu_id",
            trainer_ip,
        )
    return {row["gpu_id"]: row["reserved_by"] for row in rows}


async def _age_reservations(db: PSQLDB, seconds: int) -> None:
    async with await db.connection() as conn:
        await conn.execute(
            "UPDATE trainers_gpus SET reserved_at = reserved_at - ($1::double precision * INTERVAL '1 second')",
            float(seconds),
        )


def _report(busy: set[int]) -> list[GPUInfo]:
    return [GPUInfo(gpu_id=i, gpu_type=GPUType.H100, vram_gb=80, available=i not in busy) for i in range(4)]


class TestJobEndReleasesItsGpus:
    async def test_a_model_prep_finishing_early_frees_its_gpus_at_once(self, db):
        key = prep_job_key(TASK)
        await tournament_sql.reserve_gpus(TRAINER, [0, 1], t_cst.MODEL_PREP_GPU_RESERVE_HOURS, key, db)
        assert await _reserved(db) == {0: key, 1: key}

        assert await tournament_sql.release_gpus(key, db) == [0, 1]
        assert await _reserved(db) == {}

    @pytest.mark.parametrize("end", [TrainingStatus.PENDING, TrainingStatus.FAILURE, TrainingStatus.SUCCESS])
    async def test_a_training_run_ending_frees_its_gpus_with_its_status(self, db, end):
        await tournament_sql.reserve_gpus(TRAINER, [2, 3], 1.25, training_job_key(TASK, HOTKEY), db)
        await tournament_sql.update_tournament_task_training_status(TASK, HOTKEY, TrainingStatus.TRAINING, db, TRAINER)
        assert await _reserved(db) == {2: training_job_key(TASK, HOTKEY), 3: training_job_key(TASK, HOTKEY)}

        await tournament_sql.update_tournament_task_training_status(TASK, HOTKEY, end, db, count_attempt=True)

        assert await _reserved(db) == {}

    async def test_moving_a_run_to_training_keeps_its_gpus(self, db):
        await tournament_sql.reserve_gpus(TRAINER, [0], 1.0, training_job_key(TASK, HOTKEY), db)
        await tournament_sql.update_tournament_task_training_status(TASK, HOTKEY, TrainingStatus.TRAINING, db, TRAINER)
        assert await _reserved(db) == {0: training_job_key(TASK, HOTKEY)}

    async def test_a_gpu_another_job_took_since_is_not_freed(self, db):
        mine, theirs = training_job_key(TASK, HOTKEY), training_job_key(TASK, OTHER_HOTKEY)
        await tournament_sql.reserve_gpus(TRAINER, [0, 1], 1.0, mine, db)
        # GPU 1 was freed meanwhile and another run reserved it.
        await tournament_sql.reserve_gpus(TRAINER, [1], 1.0, theirs, db)

        await tournament_sql.update_tournament_task_training_status(TASK, HOTKEY, TrainingStatus.FAILURE, db)

        assert await _reserved(db) == {1: theirs}

    async def test_a_release_scoped_to_one_trainer_leaves_the_jobs_gpus_on_another(self, db):
        key = prep_job_key(TASK, HOTKEY)
        await tournament_sql.reserve_gpus(OTHER_TRAINER, [0], 1.0, key, db)

        assert await tournament_sql.release_gpus(key, db, TRAINER) == []
        assert await _reserved(db, OTHER_TRAINER) == {0: key}


class TestReconciliation:
    async def _reconcile(self, db, monkeypatch, report_for):
        monkeypatch.setattr(orchestrator, "fetch_trainer_gpus", AsyncMock(side_effect=report_for))
        config = type("Config", (), {"psql_db": db})()
        await orchestrator._update_all_trainers_gpu_availability(config)

    async def test_an_ended_job_is_freed_after_the_grace_period_but_a_running_one_is_kept(self, db, monkeypatch):
        ended, running = training_job_key(TASK, HOTKEY), training_job_key(TASK, OTHER_HOTKEY)
        await tournament_sql.reserve_gpus(TRAINER, [0, 1], 1.0, ended, db)
        await tournament_sql.reserve_gpus(TRAINER, [2], 1.0, running, db)
        report = {TRAINER: _report(busy={2}), OTHER_TRAINER: _report(busy=set())}

        await self._reconcile(db, monkeypatch, lambda ip: report[ip])
        assert await _reserved(db) == {0: ended, 1: ended, 2: running}  # within the grace period

        await _age_reservations(db, GRACE + 60)
        await self._reconcile(db, monkeypatch, lambda ip: report[ip])
        assert await _reserved(db) == {2: running}

    async def test_its_own_holds_do_not_restart_the_grace_period(self, db, monkeypatch):
        key = training_job_key(TASK, HOTKEY)
        await tournament_sql.reserve_gpus(TRAINER, [0], 1.0, key, db)
        await _age_reservations(db, GRACE + 60)
        # The run outlives its reservation: the trainer still reports GPU 0 busy.
        async with await db.connection() as conn:
            await conn.execute("UPDATE trainers_gpus SET used_until = now() - INTERVAL '1 minute' WHERE gpu_id = 0")
        await self._reconcile(db, monkeypatch, lambda ip: _report(busy={0} if ip == TRAINER else set()))
        assert await _reserved(db) == {0: None}

        # The run ends: the hold goes with the next report, not GRACE later.
        await self._reconcile(db, monkeypatch, lambda ip: _report(busy=set()))
        assert await _reserved(db) == {}

    async def test_a_busy_gpu_without_a_reservation_is_held_and_a_live_reservation_is_not_overwritten(self, db, monkeypatch):
        key = training_job_key(TASK, HOTKEY)
        await tournament_sql.reserve_gpus(TRAINER, [1], 1.0, key, db)

        await self._reconcile(db, monkeypatch, lambda ip: _report(busy={0, 1} if ip == TRAINER else set()))

        assert await _reserved(db) == {0: None, 1: key}
        async with await db.connection() as conn:
            hours = await conn.fetchval(
                "SELECT EXTRACT(EPOCH FROM used_until - now()) / 3600 FROM trainers_gpus WHERE trainer_ip = $1 AND gpu_id = 1",
                TRAINER,
            )
        assert hours < 1.01  # not stretched to the untracked hold

    async def test_a_trainer_restart_frees_what_it_no_longer_runs(self, db, monkeypatch):
        # Before the restart: a prep on 0-1 (its orchestrator process gone), a run on 2, and GPU 3 held as busy.
        await tournament_sql.reserve_gpus(TRAINER, [0, 1], 1.0, prep_job_key(TASK), db)
        await tournament_sql.reserve_gpus(TRAINER, [2], 1.0, training_job_key(TASK, HOTKEY), db)
        await self._reconcile(db, monkeypatch, lambda ip: _report(busy={0, 1, 2, 3} if ip == TRAINER else set()))
        assert await _reserved(db) == {0: prep_job_key(TASK), 1: prep_job_key(TASK), 2: training_job_key(TASK, HOTKEY), 3: None}

        # The trainer is down for a while, then comes back with its stale jobs marked failed and every GPU idle.
        def unreachable(ip):
            raise httpx.ConnectError("down")

        await self._reconcile(db, monkeypatch, unreachable)
        assert set(await _reserved(db)) == {0, 1, 2, 3}

        await _age_reservations(db, GRACE + 60)
        await self._reconcile(db, monkeypatch, lambda ip: _report(busy=set()))
        assert await _reserved(db) == {}

    async def test_an_unreachable_trainer_keeps_the_job_keys(self, db, monkeypatch):
        key = training_job_key(TASK, HOTKEY)
        await tournament_sql.reserve_gpus(TRAINER, [0], 1.0, key, db)

        def unreachable(ip):
            raise httpx.ConnectError("down")

        await self._reconcile(db, monkeypatch, unreachable)
        assert set(await _reserved(db)) == {0, 1, 2, 3}

        # The run ends meanwhile: its own GPU is released, the trainer's other GPUs stay held.
        await tournament_sql.update_tournament_task_training_status(TASK, HOTKEY, TrainingStatus.PENDING, db)
        assert await _reserved(db) == {1: None, 2: None, 3: None}


class TestRegistration:
    async def test_registering_a_trainer_takes_its_report_and_drops_job_keys(self, db, monkeypatch):
        monkeypatch.setattr(tournament_sql.gpu_costs, "reconcile_trainer_capacity", AsyncMock())
        await tournament_sql.reserve_gpus(TRAINER, [0, 1], 1.0, prep_job_key(TASK), db)

        await tournament_sql.add_trainer_gpus(TRAINER, _report(busy={1}), db)

        assert await _reserved(db) == {1: None}
