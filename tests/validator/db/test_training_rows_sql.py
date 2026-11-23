"""Training-row loaders: scoped to the tasks they are asked for, and tolerant of a stored task that fails validation."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from pydantic import ValidationError

import validator.db.constants as cst
from core.constants.network import NETUID
from core.models.dataset_models import InstructTextDatasetType
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.db.sql import tournaments as tournament_sql
from validator.tasks.models import QuantFreeformRawTask
from validator.tournament.models import TrainingStatus


TRAINER_IP = "10.0.0.1"


class _Strict(BaseModel):
    value: int


def _validation_error() -> ValidationError:
    try:
        _Strict(value="not an int")
    except ValidationError as e:
        return e
    raise AssertionError("expected a validation error")


def _task(task_id: uuid.UUID) -> QuantFreeformRawTask:
    return QuantFreeformRawTask(
        is_organic=False,
        task_id=task_id,
        status="training",
        model_id="org/model",
        ds="org/dataset",
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        training_data="s3://bucket/train.json",
        created_at=datetime(2026, 10, 19),
        dataset_type=InstructTextDatasetType(field_instruction="instruction", field_output="output"),
    )


def _training_row(task_id: uuid.UUID, hotkey: str) -> dict:
    return {
        cst.TASK_ID: task_id,
        cst.HOTKEY: hotkey,
        cst.TRAINING_STATUS: TrainingStatus.SUCCESS.value,
        cst.N_TRAINING_ATTEMPTS: 1,
        cst.CREATED_AT: datetime(2026, 10, 19),
        cst.UPDATED_AT: datetime(2026, 10, 19),
        cst.PRIORITY: 1,
        cst.TRAINING_REPO: "https://github.com/miner/repo",
        cst.TRAINING_COMMIT_HASH: "abc",
        cst.GITHUB_TOKEN: None,
        cst.REQUESTED_DATASETS: None,
        cst.TRAINER_IP: TRAINER_IP,
    }


def _mock_psql(rows: list[dict]):
    conn = MagicMock()
    conn.fetch = AsyncMock(return_value=rows)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    psql = MagicMock()
    psql.connection = AsyncMock(return_value=cm)
    return psql, conn


@pytest.fixture
def task_loader(monkeypatch):
    """get_tasks_by_ids over a set of loadable tasks; any id in `unparsable` fails validation, as a stale row does."""
    unparsable: set[uuid.UUID] = set()

    async def load(task_ids, _psql_db, _connection=None):
        if unparsable & set(task_ids):
            raise _validation_error()
        return [_task(task_id) for task_id in task_ids]

    loader = AsyncMock(side_effect=load)
    loader.unparsable = unparsable
    monkeypatch.setattr(tournament_sql.task_sql, "get_tasks_by_ids", loader)
    return loader


async def test_training_rows_for_task_load_only_that_task(task_loader):
    task_id = uuid.uuid4()
    task_loader.unparsable.add(uuid.uuid4())
    psql, conn = _mock_psql([_training_row(task_id, "a"), _training_row(task_id, "b")])

    trainings = await tournament_sql.get_training_rows_for_task(str(task_id), TrainingStatus.SUCCESS, psql)

    query, *args = conn.fetch.await_args.args
    assert f"{cst.TASK_ID} = $1" in query and f"{cst.TRAINING_STATUS} = $2" in query
    assert args == [str(task_id), TrainingStatus.SUCCESS]
    assert [(t.hotkey, t.task.task_id, t.trainer_ip) for t in trainings] == [(h, task_id, TRAINER_IP) for h in ("a", "b")]
    assert [call.args[0] for call in task_loader.await_args_list] == [[task_id]]


async def test_unsettled_freeform_decode_query_is_scoped_to_unfinished_freeform_tasks(task_loader):
    task_id = uuid.uuid4()
    psql, conn = _mock_psql([_training_row(task_id, "a")])

    trainings = await tournament_sql.get_unsettled_freeform_decode_trainings(psql)

    query, *args = conn.fetch.await_args.args
    assert args == [
        TrainingStatus.SUCCESS,
        TaskType.QUANTFREEFORMTASK.value,
        [TaskStatus.TRAINING.value, TaskStatus.PREEVALUATION.value, TaskStatus.EVALUATING.value],
        NETUID,
    ]
    assert "NOT EXISTS" in query
    assert f"tn.{cst.DECODER_SHA256} IS NOT NULL OR tn.{cst.QUANT_FAILURE} IS NOT NULL" in query
    assert [t.hotkey for t in trainings] == ["a"]


async def test_no_rows_loads_no_tasks(task_loader):
    psql, _ = _mock_psql([])

    assert await tournament_sql.get_unsettled_freeform_decode_trainings(psql) == []
    assert await tournament_sql.get_training_rows_for_task(str(uuid.uuid4()), TrainingStatus.SUCCESS, psql) == []
    task_loader.assert_not_awaited()


async def test_training_tasks_skip_a_task_that_fails_validation(task_loader):
    good, bad = uuid.uuid4(), uuid.uuid4()
    task_loader.unparsable.add(bad)
    psql, _ = _mock_psql([_training_row(good, "a"), _training_row(bad, "b"), _training_row(good, "c")])

    trainings = await tournament_sql.get_tournament_training_tasks(psql, TrainingStatus.SUCCESS)

    assert [(t.hotkey, t.task.task_id) for t in trainings] == [("a", good), ("c", good)]


async def test_training_tasks_do_not_swallow_other_errors(monkeypatch):
    monkeypatch.setattr(tournament_sql.task_sql, "get_tasks_by_ids", AsyncMock(side_effect=ConnectionError("db gone")))
    psql, _ = _mock_psql([_training_row(uuid.uuid4(), "a")])

    with pytest.raises(ConnectionError):
        await tournament_sql.get_tournament_training_tasks(psql, TrainingStatus.SUCCESS)
