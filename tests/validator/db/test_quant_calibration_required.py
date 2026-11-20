"""A compression task past data prep is never stored without its calibration set: the Python guard and the SQL CHECK."""

import re
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantFormat
from core.models.task_models import QUANT_TASK_TYPES
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.db.sql import tasks as task_sql
from validator.tasks.models import QUANT_CALIBRATION_REQUIRED_STATUSES
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tasks.models import has_published_calibration


MIGRATION = Path(__file__).parents[3] / "validator/db/migrations/20261114120000_quant_calibration_required.sql"
REQUIRED = sorted(QUANT_CALIBRATION_REQUIRED_STATUSES, key=lambda status: status.value)
ALLOWED_WITHOUT_CALIBRATION = [
    TaskStatus.PENDING,
    TaskStatus.AWAITING_UPSTREAM,
    TaskStatus.PREPARING_DATA,
    TaskStatus.PREP_TASK_FAILURE,
    TaskStatus.PREEVALUATION,
    TaskStatus.EVALUATING,
    TaskStatus.SUCCESS,
    TaskStatus.FAILURE,
]
CALIBRATION = dict(calibration_data="https://minio/cal.jsonl", calibration_fingerprint="fp", calibration_n_examples=120)


def _fields(status: TaskStatus, **overrides) -> dict:
    return dict(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=status,
        model_id="org/base",
        ds="org/data",
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 14),
        **overrides,
    )


def _quant(task_type, status: TaskStatus, **overrides):
    dataset_type = InstructTextDatasetType(field_instruction="q", field_output="a")
    if task_type == TaskType.QUANTFIXEDTASK:
        spec = QuantFormat.W4A16_SYM_G128.spec()
        return QuantFixedRawTask(**_fields(status, **overrides), dataset_type=dataset_type, quant_spec=spec)
    return QuantFreeformRawTask(**_fields(status, **overrides), dataset_type=dataset_type)


def _instruct(status: TaskStatus) -> InstructTextRawTask:
    return InstructTextRawTask(**_fields(status), field_instruction="q", field_output="a")


def _psql_that_stores():
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value={"task_id": uuid.uuid4()})
    conn.execute = AsyncMock(return_value="UPDATE 1")
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


@pytest.mark.parametrize("task_type", sorted(QUANT_TASK_TYPES, key=lambda t: t.value))
@pytest.mark.parametrize("status", REQUIRED)
async def test_add_and_update_reject_a_quant_task_without_calibration_in_a_required_status(task_type, status, monkeypatch):
    psql, conn = _psql_that_stores()
    get_task = AsyncMock()
    monkeypatch.setattr(task_sql, "get_task", get_task)
    task = _quant(task_type, status)

    with pytest.raises(ValueError, match="no published calibration set"):
        await task_sql.add_task(task, psql)
    with pytest.raises(ValueError, match="no published calibration set"):
        await task_sql.update_task(task, psql)

    psql.connection.assert_not_awaited()
    get_task.assert_not_awaited()


@pytest.mark.parametrize("count", [0, -1])
async def test_a_non_positive_example_count_is_rejected_too(count):
    task = _quant(TaskType.QUANTFREEFORMTASK, TaskStatus.LOOKING_FOR_NODES, **{**CALIBRATION, "calibration_n_examples": count})

    assert not has_published_calibration(task)
    with pytest.raises(ValueError, match="no published calibration set"):
        await task_sql.add_task(task, MagicMock())


def test_the_required_statuses_stop_at_training():
    assert TaskStatus.TRAINING in QUANT_CALIBRATION_REQUIRED_STATUSES
    assert not {TaskStatus.PREEVALUATION, TaskStatus.EVALUATING} & QUANT_CALIBRATION_REQUIRED_STATUSES


@pytest.mark.parametrize("missing", ["calibration_data", "calibration_fingerprint", "calibration_n_examples"])
async def test_a_partial_calibration_is_rejected_too(missing):
    task = _quant(TaskType.QUANTFREEFORMTASK, TaskStatus.LOOKING_FOR_NODES, **{**CALIBRATION, missing: None})

    with pytest.raises(ValueError, match="no published calibration set"):
        await task_sql.add_task(task, MagicMock())


@pytest.mark.parametrize("status", ALLOWED_WITHOUT_CALIBRATION)
async def test_add_accepts_a_quant_task_without_calibration_outside_the_required_statuses(status):
    psql, _ = _psql_that_stores()

    stored = await task_sql.add_task(_quant(TaskType.QUANTFREEFORMTASK, status), psql)

    assert stored.status == status


@pytest.mark.parametrize("status", REQUIRED)
async def test_add_accepts_a_quant_task_with_its_calibration_in_every_required_status(status):
    psql, _ = _psql_that_stores()

    stored = await task_sql.add_task(_quant(TaskType.QUANTFREEFORMTASK, status, **CALIBRATION), psql)

    assert stored.status == status


@pytest.mark.parametrize("status", [*REQUIRED, *ALLOWED_WITHOUT_CALIBRATION])
async def test_add_accepts_a_non_quant_task_in_any_status(status):
    psql, _ = _psql_that_stores()

    stored = await task_sql.add_task(_instruct(status), psql)

    assert stored.status == status


@pytest.mark.parametrize("status", ALLOWED_WITHOUT_CALIBRATION)
async def test_update_passes_the_guard_outside_the_required_statuses(status, monkeypatch):
    monkeypatch.setattr(task_sql, "get_task", AsyncMock(return_value=None))

    with pytest.raises(ValueError, match="not found in the database"):  # past the guard, into the lookup
        await task_sql.update_task(_quant(TaskType.QUANTFREEFORMTASK, status), MagicMock())


async def test_update_passes_the_guard_for_a_calibrated_quant_task_and_a_non_quant_task(monkeypatch):
    monkeypatch.setattr(task_sql, "get_task", AsyncMock(return_value=None))

    for task in (_quant(TaskType.QUANTFREEFORMTASK, TaskStatus.TRAINING, **CALIBRATION), _instruct(TaskStatus.TRAINING)):
        with pytest.raises(ValueError, match="not found in the database"):
            await task_sql.update_task(task, MagicMock())


def test_the_migration_status_list_is_the_required_status_list():
    sql = MIGRATION.read_text()
    up = sql.split("-- migrate:down")[0]
    add = up[up.index("ADD CONSTRAINT tasks_quant_calibration_required") :]
    not_in = re.findall(r"status NOT IN \(([^)]*)\)", add)
    assert len(not_in) == 1
    listed = {value for value in re.findall(r"'([^']+)'", not_in[0])}
    assert listed == {status.value for status in QUANT_CALIBRATION_REQUIRED_STATUSES}


def test_the_listing_block_selects_the_same_statuses_and_types_as_the_constraint():
    up = MIGRATION.read_text().split("-- migrate:down")[0]
    constraint = up.split("ALTER TABLE tasks DROP CONSTRAINT")[1]
    status_lists = [set(re.findall(r"'([^']+)'", m)) for m in re.findall(r"status (?:NOT )?IN \(([^)]*)\)", up)]
    type_lists = [set(re.findall(r"'([^']+)'", m)) for m in re.findall(r"task_type(?:::text)? (?:NOT )?IN \(([^)]*)\)", up)]

    assert len(status_lists) == 2 and status_lists[0] == status_lists[1]
    assert len(type_lists) == 2 and type_lists[0] == type_lists[1] == {t.value for t in QUANT_TASK_TYPES}
    assert "NOT VALID" in constraint and "UPDATE tasks" not in up and "DELETE" not in up
    assert "DROP CONSTRAINT IF EXISTS tasks_quant_calibration_required" in MIGRATION.read_text().split("-- migrate:down")[1]
