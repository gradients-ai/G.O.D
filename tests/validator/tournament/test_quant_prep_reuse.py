"""One quant prep per (model, dataset): a quant task copies a sibling's baseline_stats and sizes its own hours."""

import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.models.dataset_models import InstructTextDatasetType
from core.models.model_prep_models import InstructBaselineStats
from core.models.model_prep_models import InstructDatasetStats
from core.models.model_prep_models import InstructTrainingDynamics
from core.models.model_prep_models import SeqLengthDistribution
from core.models.model_prep_models import ThroughputStats
from core.models.model_prep_models import WeightStats
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.db.sql import tournaments as tournament_sql
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tournament import orchestrator


PARAMS = int(3.82e9)
CONFIG = SimpleNamespace(psql_db=object())


def _stats(near_duplicate_rate: float = 0.0) -> InstructBaselineStats:
    seq_dist = SeqLengthDistribution(mean=1041.1, p50=350, p95=900, p99=1200, max=2000)
    return InstructBaselineStats(
        dataset=InstructDatasetStats(
            total_tokens=int(40e6),
            num_records=100_000,
            seq_length_distribution=seq_dist,
            near_duplicate_rate=near_duplicate_rate,
            vocab_size=150_000,
            prompt_tokens=int(20e6),
            completion_tokens=int(20e6),
            completion_length_distribution=seq_dist,
        ),
        weights=WeightStats(by_group={}),
        training=InstructTrainingDynamics(init_loss=2.0, activation_rms={}, output_entropy=3.0, masked_completion_loss=2.5),
        throughput=ThroughputStats(tokens_per_sec=8_700.0, seq_len=1024, micro_batch_size=8, n_gpus=1, gpu_name="H100"),
    )


def _task(task_type: TaskType, **overrides):
    fields = dict(
        is_organic=False,
        task_id=uuid.uuid4(),
        status=TaskStatus.AWAITING_MODEL_PREP,
        model_id="org/model",
        ds="org/dataset",
        account_id=uuid.uuid4(),
        hours_to_complete=4.0,
        created_at=datetime(2026, 11, 1),
        model_params_count=PARAMS,
        dataset_type=InstructTextDatasetType(field_instruction="instruction", field_output="output"),
    )
    fields.update(overrides)
    if task_type == TaskType.QUANTFIXEDTASK:
        return QuantFixedRawTask(**fields, quant_spec=QuantFormat.W4A16_SYM_G128.spec())
    return QuantFreeformRawTask(**fields)


@pytest.fixture
def sql(monkeypatch):
    fakes = SimpleNamespace(
        quant_stats=AsyncMock(return_value=None),
        quant_ids=AsyncMock(return_value=[]),
        env_stats=AsyncMock(return_value=None),
        env_ids=AsyncMock(return_value=[]),
        miners=AsyncMock(return_value=[]),
        update_task=AsyncMock(),
    )
    monkeypatch.setattr(tournament_sql, "get_quant_sibling_baseline_stats", fakes.quant_stats)
    monkeypatch.setattr(tournament_sql, "get_quant_prep_sibling_task_ids", fakes.quant_ids)
    monkeypatch.setattr(tournament_sql, "get_sibling_env_baseline_stats", fakes.env_stats)
    monkeypatch.setattr(tournament_sql, "get_matching_sibling_task_ids", fakes.env_ids)
    monkeypatch.setattr(orchestrator.task_sql, "get_miners_needing_baseline_stats", fakes.miners)
    monkeypatch.setattr(orchestrator.task_sql, "update_task", fakes.update_task)
    monkeypatch.setattr(orchestrator, "_is_boss_round_task", AsyncMock(return_value=False))
    monkeypatch.setattr(orchestrator, "_model_prep_in_progress", set())
    return fakes


@pytest.mark.asyncio
async def test_sibling_stats_are_copied_and_hours_applied(sql):
    sql.quant_stats.return_value = _stats().model_dump()
    task = _task(TaskType.QUANTFREEFORMTASK)

    assert await orchestrator._try_reuse_sibling_model_prep(task, CONFIG) is True

    assert isinstance(task.baseline_stats, InstructBaselineStats)
    assert task.baseline_stats.dataset.num_records == 100_000
    assert task.status == TaskStatus.LOOKING_FOR_NODES
    assert task.hours_to_complete == 2.0
    assert task.termination_at is not None
    sql.update_task.assert_awaited_once()
    sql.quant_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_freeform_and_fixed_get_their_own_hours_from_the_same_stats(sql):
    sql.quant_stats.return_value = _stats()
    freeform = _task(TaskType.QUANTFREEFORMTASK)
    fixed = _task(TaskType.QUANTFIXEDTASK)

    await orchestrator._try_reuse_sibling_model_prep(freeform, CONFIG)
    await orchestrator._try_reuse_sibling_model_prep(fixed, CONFIG)

    assert freeform.hours_to_complete == 2.0
    assert fixed.hours_to_complete == 1.25
    assert freeform.hours_to_complete != fixed.hours_to_complete


@pytest.mark.asyncio
async def test_sibling_in_progress_without_stats_waits(sql):
    sibling = str(uuid.uuid4())
    sql.quant_ids.return_value = [sibling]
    orchestrator._model_prep_in_progress.add(sibling)
    task = _task(TaskType.QUANTFIXEDTASK)

    assert await orchestrator._try_reuse_sibling_model_prep(task, CONFIG) is True

    assert task.status == TaskStatus.AWAITING_MODEL_PREP
    assert task.baseline_stats is None
    assert task.hours_to_complete == 4.0
    sql.update_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_sibling_not_in_progress_does_not_wait(sql):
    sql.quant_ids.return_value = [str(uuid.uuid4())]
    task = _task(TaskType.QUANTFIXEDTASK)

    assert await orchestrator._try_reuse_sibling_model_prep(task, CONFIG) is False
    sql.update_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_sibling_dispatches_normally(sql):
    task = _task(TaskType.QUANTFREEFORMTASK)

    assert await orchestrator._try_reuse_sibling_model_prep(task, CONFIG) is False
    assert task.status == TaskStatus.AWAITING_MODEL_PREP


@pytest.mark.asyncio
async def test_augmented_quant_task_is_not_reused(sql):
    task = _task(TaskType.QUANTFREEFORMTASK)
    task.augmentation_config = object()

    assert await orchestrator._try_reuse_sibling_model_prep(task, CONFIG) is False
    sql.quant_stats.assert_not_awaited()
    sql.quant_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_env_task_keeps_the_env_path(sql):
    sql.env_stats.return_value = {"task_type": "env"}
    task = SimpleNamespace(
        task_id=uuid.uuid4(), task_type=TaskType.ENVIRONMENTTASK, model_id="org/model", augmentation_config=None,
        baseline_stats=None, status=TaskStatus.AWAITING_MODEL_PREP,
    )

    assert await orchestrator._try_reuse_sibling_model_prep(task, CONFIG) is True

    sql.miners.assert_awaited_once()
    sql.env_stats.assert_awaited_once()
    assert task.baseline_stats == {"task_type": "env"}
    assert task.status == TaskStatus.LOOKING_FOR_NODES
    sql.quant_stats.assert_not_awaited()
    sql.quant_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_env_task_waits_on_in_progress_sibling(sql):
    sibling = str(uuid.uuid4())
    sql.env_ids.return_value = [sibling]
    orchestrator._model_prep_in_progress.add(sibling)
    task = SimpleNamespace(task_id=uuid.uuid4(), task_type=TaskType.ENVIRONMENTTASK, model_id="m", augmentation_config=None)

    assert await orchestrator._try_reuse_sibling_model_prep(task, CONFIG) is True
    sql.quant_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_instruct_task_is_never_reused(sql):
    task = SimpleNamespace(task_id=uuid.uuid4(), task_type=TaskType.INSTRUCTTEXTTASK, model_id="m", augmentation_config=None)

    assert await orchestrator._try_reuse_sibling_model_prep(task, CONFIG) is False
    for fake in (sql.quant_stats, sql.quant_ids, sql.env_stats, sql.env_ids, sql.miners, sql.update_task):
        fake.assert_not_awaited()


@pytest.mark.asyncio
async def test_near_duplicate_stats_fail_the_task(sql):
    sql.quant_stats.return_value = _stats(near_duplicate_rate=0.99)
    task = _task(TaskType.QUANTFIXEDTASK)

    assert await orchestrator._try_reuse_sibling_model_prep(task, CONFIG) is True

    assert task.status == TaskStatus.PREP_TASK_FAILURE
    assert task.hours_to_complete == 4.0
    assert task.termination_at is None
    sql.update_task.assert_awaited_once()


class _FakeConnection:
    def __init__(self, row=None, rows=()):
        self.row, self.rows, self.calls = row, list(rows), []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def fetchrow(self, query, *args):
        self.calls.append((query, args))
        return self.row

    async def fetch(self, query, *args):
        self.calls.append((query, args))
        return self.rows


class _FakeDb:
    def __init__(self, conn):
        self.conn = conn

    async def connection(self):
        return self.conn


def _assert_sibling_filters(query: str, args: tuple, task_id: str):
    flat = " ".join(query.split())
    assert args[0] == task_id
    assert sorted(args[1]) == sorted([TaskType.QUANTFIXEDTASK.value, TaskType.QUANTFREEFORMTASK.value])
    assert "tt.task_id != tt_self.task_id" in flat
    assert "tt.tournament_id = tt_self.tournament_id" in flat
    assert "WHERE tt_self.task_id = $1" in flat
    assert "split_part(t.training_data, '?', 1) = split_part(t_self.training_data, '?', 1)" in flat
    assert "split_part(t_self.training_data, '?', 1) != ''" in flat
    assert "t.model_id = t_self.model_id" in flat
    assert "t.augmentation_config IS NULL" in flat
    assert "t.task_type::text = ANY($2::text[])" in flat


@pytest.mark.asyncio
async def test_sql_quant_sibling_stats_query():
    conn = _FakeConnection(row={"baseline_stats": '{"task_type": "instruct"}'})
    result = await tournament_sql.get_quant_sibling_baseline_stats("tid", _FakeDb(conn))

    assert result == {"task_type": "instruct"}
    (query, args), = conn.calls
    _assert_sibling_filters(query, args, "tid")
    flat = " ".join(query.split())
    assert "t.baseline_stats IS NOT NULL" in flat
    assert len(args) == 2
    assert "ORDER BY t.created_at DESC LIMIT 1" in flat


@pytest.mark.asyncio
async def test_sql_quant_sibling_stats_none_without_row():
    assert await tournament_sql.get_quant_sibling_baseline_stats("tid", _FakeDb(_FakeConnection())) is None


@pytest.mark.asyncio
async def test_sql_quant_sibling_ids_query():
    conn = _FakeConnection(rows=[{"task_id": "a"}, {"task_id": "b"}])
    result = await tournament_sql.get_quant_prep_sibling_task_ids("tid", _FakeDb(conn))

    assert result == ["a", "b"]
    (query, args), = conn.calls
    _assert_sibling_filters(query, args, "tid")
    assert "baseline_stats" not in query


@pytest.mark.asyncio
async def test_lookup_failure_dispatches_a_prep_instead_of_stalling_the_cycle(sql):
    sql.quant_stats.side_effect = RuntimeError("db down")
    task = _task(TaskType.QUANTFREEFORMTASK)

    assert await orchestrator._try_reuse_sibling_model_prep(task, CONFIG) is False
    assert task.status == TaskStatus.AWAITING_MODEL_PREP
