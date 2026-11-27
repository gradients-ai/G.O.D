"""The orchestrator releases a job's GPU reservation when it sees the job end, and reconciles the reservations with what
each trainer reports. The SQL semantics (owner-keyed release, grace period, no clobbering) are covered against a real
Postgres in tests/validator/db/test_gpu_reservations.py."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from core.models.payload_models import ModelPrepJob
from core.models.payload_models import ModelPrepResponse
from core.models.task_models import TaskStatus
from core.models.trainer_contract_models import GPUInfo
from core.models.trainer_contract_models import GPUType
from validator.tournament import orchestrator
from validator.tournament.models import TrainerInfo


CONFIG = SimpleNamespace(psql_db=None)
TASK_ID = "00000000-0000-0000-0000-000000000001"


def _trainer(ip: str) -> TrainerInfo:
    return TrainerInfo(trainer_ip=ip, gpus=[])


class _PrepStatus:
    """httpx.AsyncClient stand-in answering the model-prep status endpoint per trainer."""

    def __init__(self, jobs: dict[str, ModelPrepJob | None]):
        self.jobs = jobs

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None):
        ip = url.split("//", 1)[1].split(":", 1)[0]
        job = self.jobs.get(ip)
        request = httpx.Request("GET", url)
        if job is None:
            return httpx.Response(404, request=request)
        return httpx.Response(200, json=job.model_dump(mode="json"), request=request)


def _prep_job(status: TaskStatus, gpu_ids=(0, 1)) -> ModelPrepJob:
    return ModelPrepJob(
        task_id=TASK_ID,
        model_id="org/model",
        gpu_ids=list(gpu_ids),
        status=status,
        result=ModelPrepResponse(augmented_model_id="org/aug") if status == TaskStatus.SUCCESS else None,
    )


@pytest.fixture
def prep_world(monkeypatch):
    world = SimpleNamespace(release=AsyncMock(return_value=[]))
    monkeypatch.setattr(
        orchestrator.tournament_sql, "get_trainers", AsyncMock(return_value=[_trainer("10.0.0.1"), _trainer("10.0.0.2")])
    )
    monkeypatch.setattr(orchestrator.tournament_sql, "release_gpus", world.release)
    monkeypatch.setattr(orchestrator, "_finish_prep_cost_run", AsyncMock())
    monkeypatch.setattr(orchestrator, "_apply_measured_training_hours", AsyncMock(return_value=False))
    monkeypatch.setattr(orchestrator.task_sql, "update_task", AsyncMock())

    def answer(jobs):
        monkeypatch.setattr(orchestrator.httpx, "AsyncClient", _PrepStatus(jobs))

    world.answer = answer
    return world


def _task():
    return SimpleNamespace(task_id=TASK_ID, augmented_model_id=None, baseline_stats=None, status=None)


class TestAPrepRecoveredAfterARestart:
    @pytest.mark.parametrize("status", [TaskStatus.SUCCESS, TaskStatus.FAILURE])
    async def test_an_ended_prep_releases_its_gpus_on_its_trainer(self, prep_world, status):
        prep_world.answer({"10.0.0.2": _prep_job(status)})

        await orchestrator._recover_model_prep_from_trainer(_task(), CONFIG)

        prep_world.release.assert_awaited_once_with(f"prep:{TASK_ID}:task", None, "10.0.0.2")

    async def test_a_running_prep_keeps_its_gpus(self, prep_world):
        prep_world.answer({"10.0.0.1": _prep_job(TaskStatus.TRAINING)})

        assert await orchestrator._recover_model_prep_from_trainer(_task(), CONFIG) is True

        prep_world.release.assert_not_awaited()

    async def test_an_ended_miner_prep_releases_its_gpus_and_a_running_one_keeps_them(self, prep_world, monkeypatch):
        monkeypatch.setattr(orchestrator.task_sql, "set_miner_baseline_stats", AsyncMock())
        ended = _prep_job(TaskStatus.FAILURE)
        running = _prep_job(TaskStatus.TRAINING)

        class PerHotkey(_PrepStatus):
            async def get(self, url, params=None):
                job = {"hk-ended": ended, "hk-running": running}[params["hotkey"]]
                ip = url.split("//", 1)[1].split(":", 1)[0]
                if ip != "10.0.0.1":
                    return httpx.Response(404, request=httpx.Request("GET", url))
                return httpx.Response(200, json=job.model_dump(mode="json"), request=httpx.Request("GET", url))

        monkeypatch.setattr(orchestrator.httpx, "AsyncClient", PerHotkey({}))

        handled = await orchestrator._recover_miner_preps_from_trainers(
            _task(), [("hk-ended", "org/a"), ("hk-running", "org/b")], CONFIG
        )

        assert handled == {"hk-running"}
        prep_world.release.assert_awaited_once_with(f"prep:{TASK_ID}:miner:hk-ended", None, "10.0.0.1")


class TestTheReconciliation:
    @pytest.fixture
    def sql(self, monkeypatch):
        sql = SimpleNamespace(
            free=AsyncMock(return_value=[]),
            hold=AsyncMock(return_value=[]),
            unreachable=AsyncMock(),
        )
        monkeypatch.setattr(orchestrator.tournament_sql, "get_trainers", AsyncMock(return_value=[_trainer("10.0.0.1")]))
        monkeypatch.setattr(orchestrator.tournament_sql, "free_gpus_reported_idle", sql.free)
        monkeypatch.setattr(orchestrator.tournament_sql, "hold_gpus_reported_busy", sql.hold)
        monkeypatch.setattr(orchestrator.tournament_sql, "hold_unreachable_trainer_gpus", sql.unreachable)
        return sql

    async def test_idle_gpus_are_offered_for_freeing_and_busy_ones_for_holding(self, sql, monkeypatch):
        report = [GPUInfo(gpu_id=i, gpu_type=GPUType.H100, vram_gb=80, available=i in (0, 3)) for i in range(4)]
        monkeypatch.setattr(orchestrator, "fetch_trainer_gpus", AsyncMock(return_value=report))

        await orchestrator._update_all_trainers_gpu_availability(CONFIG)

        sql.free.assert_awaited_once_with("10.0.0.1", [0, 3], orchestrator.cst.GPU_RESERVATION_GRACE_PERIOD_SECONDS, None)
        sql.hold.assert_awaited_once_with("10.0.0.1", [1, 2], orchestrator.cst.UNTRACKED_GPU_HOLD_HOURS, None)
        sql.unreachable.assert_not_awaited()

    async def test_an_unreachable_trainer_keeps_its_gpus_held(self, sql, monkeypatch):
        monkeypatch.setattr(orchestrator, "fetch_trainer_gpus", AsyncMock(side_effect=httpx.ConnectError("down")))

        await orchestrator._update_all_trainers_gpu_availability(CONFIG)

        sql.unreachable.assert_awaited_once_with("10.0.0.1", orchestrator.cst.UNTRACKED_GPU_HOLD_HOURS, None)
        sql.free.assert_not_awaited()
