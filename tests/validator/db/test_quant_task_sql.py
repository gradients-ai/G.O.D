"""quant_tasks persistence: insert dispatch and row -> QuantFixed/QuantFreeform task mapping (mocked DB)."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import validator.db.constants as cst
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantFormat
from core.models.quant_models import QuantFreeformPenalty
from core.models.task_models import TaskType
from validator.db.sql import tasks as task_sql
from validator.scoring.constants import QUANT_BAR_CAP
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tasks.models import freeform_penalty_policy


TASK_ID = uuid.uuid4()
UPSTREAM_TASK_ID = uuid.uuid4()
SPEC = QuantFormat.W4A8_FP8_G128.spec()
INSTRUCT_DATASET_TYPE = {"field_instruction": "instruction", "field_output": "output"}
CHAT_DATASET_TYPE = {"chat_column": "messages", "chat_role_field": "role", "chat_content_field": "content"}


def _mock_psql(*, fetchrow=None, fetch=None, execute="UPDATE 1"):
    conn = MagicMock()
    conn.fetchrow = AsyncMock(side_effect=fetchrow) if isinstance(fetchrow, list) else AsyncMock(return_value=fetchrow)
    conn.fetch = AsyncMock(return_value=fetch or [])
    conn.execute = AsyncMock(return_value=execute)
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


def _base_fields(task_type: TaskType) -> dict:
    return {
        cst.TASK_ID: TASK_ID,
        cst.IS_ORGANIC: False,
        cst.STATUS: "pending",
        cst.MODEL_ID: "org/model",
        cst.DS: "org/dataset",
        cst.ACCOUNT_ID: uuid.uuid4(),
        cst.HOURS_TO_COMPLETE: 2.0,
        cst.CREATED_AT: datetime(2026, 10, 19),
        cst.TASK_TYPE: task_type.value,
    }


def _quant_columns(
    dataset_type: dict,
    quant_spec: dict | None,
    bar_cap: float | None,
    penalty_knee: float | None = None,
    penalty_curvature: float | None = None,
) -> dict:
    return {
        cst.DATASET_TYPE: dataset_type,
        cst.FILE_FORMAT: FileFormat.HF.value,
        cst.QUANT_SPEC: quant_spec,
        cst.BAR_CAP: bar_cap,
        cst.PENALTY_KNEE: penalty_knee,
        cst.PENALTY_CURVATURE: penalty_curvature,
        cst.UPSTREAM_TASK_ID: UPSTREAM_TASK_ID,
    }


def _fixed_row(dataset_type: dict = INSTRUCT_DATASET_TYPE) -> dict:
    return {**_base_fields(TaskType.QUANTFIXEDTASK), **_quant_columns(dataset_type, SPEC.model_dump(), None)}


def _freeform_row() -> dict:
    return {**_base_fields(TaskType.QUANTFREEFORMTASK), **_quant_columns(INSTRUCT_DATASET_TYPE, None, 0.03)}


def _raw_task_fields() -> dict:
    fields = _base_fields(TaskType.QUANTFIXEDTASK)
    del fields[cst.TASK_ID], fields[cst.TASK_TYPE]
    return {**fields, cst.DATASET_TYPE: InstructTextDatasetType(**INSTRUCT_DATASET_TYPE)}


def _quant_insert_args(conn) -> tuple:
    calls = [c.args for c in conn.execute.await_args_list if f"INSERT INTO {cst.QUANT_TASKS_TABLE}" in c.args[0]]
    assert len(calls) == 1
    return calls[0][1:]


async def test_add_fixed_task_inserts_spec_into_quant_tasks():
    psql, conn = _mock_psql(fetchrow={cst.TASK_ID: TASK_ID})
    task = await task_sql.add_task(QuantFixedRawTask(**_raw_task_fields(), quant_spec=SPEC), psql)

    task_id, dataset_type, file_format, quant_spec, bar_cap, *_ = _quant_insert_args(conn)
    assert task.task_id == task_id == TASK_ID
    assert dataset_type == InstructTextDatasetType(**INSTRUCT_DATASET_TYPE).model_dump()
    assert file_format == FileFormat.HF
    assert quant_spec == SPEC.model_dump()
    assert bar_cap is None


async def test_add_freeform_task_inserts_bar_cap_without_spec():
    psql, conn = _mock_psql(fetchrow={cst.TASK_ID: TASK_ID})
    await task_sql.add_task(QuantFreeformRawTask(**_raw_task_fields()), psql)

    _, _, _, quant_spec, bar_cap, *_ = _quant_insert_args(conn)
    assert quant_spec is None
    assert bar_cap == QUANT_BAR_CAP


async def test_add_freeform_task_inserts_the_pinned_penalty():
    psql, conn = _mock_psql(fetchrow={cst.TASK_ID: TASK_ID})
    await task_sql.add_task(QuantFreeformRawTask(**_raw_task_fields()), psql)

    _, _, _, _, bar_cap, penalty_knee, penalty_curvature, *_ = _quant_insert_args(conn)
    assert (bar_cap, penalty_knee, penalty_curvature) == (0.08, 0.04, 2500.0)


async def test_add_fixed_task_inserts_no_penalty():
    psql, conn = _mock_psql(fetchrow={cst.TASK_ID: TASK_ID})
    await task_sql.add_task(QuantFixedRawTask(**_raw_task_fields(), quant_spec=SPEC), psql)

    _, _, _, _, _, penalty_knee, penalty_curvature, *_ = _quant_insert_args(conn)
    assert penalty_knee is None and penalty_curvature is None


async def test_get_freeform_task_keeps_pinned_penalty():
    row = {**_freeform_row(), cst.PENALTY_KNEE: 0.03, cst.PENALTY_CURVATURE: 900.0}
    psql, _ = _mock_psql(fetchrow=[_base_fields(TaskType.QUANTFREEFORMTASK), row])
    task = await task_sql.get_task(TASK_ID, psql)

    assert (task.penalty_knee, task.penalty_curvature) == (0.03, 900.0)


async def test_get_legacy_freeform_task_is_the_cliff_at_its_bar_cap():
    psql, _ = _mock_psql(fetchrow=[_base_fields(TaskType.QUANTFREEFORMTASK), _freeform_row()])
    task = await task_sql.get_task(TASK_ID, psql)

    assert task.penalty_knee is None and task.penalty_curvature is None
    assert freeform_penalty_policy(task) == QuantFreeformPenalty.cliff(1.03)


async def test_get_fixed_task_builds_spec_and_instruct_dataset_type():
    psql, _ = _mock_psql(fetchrow=[_base_fields(TaskType.QUANTFIXEDTASK), _fixed_row()])
    task = await task_sql.get_task(TASK_ID, psql)

    assert type(task) is QuantFixedRawTask
    assert task.quant_spec == SPEC
    assert type(task.dataset_type) is InstructTextDatasetType
    assert task.upstream_task_id == UPSTREAM_TASK_ID


async def test_get_fixed_task_with_chat_dataset_type():
    psql, _ = _mock_psql(fetchrow=[_base_fields(TaskType.QUANTFIXEDTASK), _fixed_row(CHAT_DATASET_TYPE)])
    task = await task_sql.get_task(TASK_ID, psql)

    assert type(task.dataset_type) is ChatTemplateDatasetType
    assert task.dataset_type.chat_column == "messages"


async def test_get_freeform_task_keeps_pinned_bar_cap():
    psql, _ = _mock_psql(fetchrow=[_base_fields(TaskType.QUANTFREEFORMTASK), _freeform_row()])
    task = await task_sql.get_task(TASK_ID, psql)

    assert type(task) is QuantFreeformRawTask
    assert task.bar_cap == 0.03


async def test_create_task_from_data_builds_both_quant_types():
    fixed = await task_sql._create_task_from_data(TaskType.QUANTFIXEDTASK.value, _fixed_row(), TASK_ID, MagicMock(), MagicMock())
    freeform = await task_sql._create_task_from_data(
        TaskType.QUANTFREEFORMTASK.value, _freeform_row(), TASK_ID, MagicMock(), MagicMock()
    )

    assert type(fixed) is QuantFixedRawTask and fixed.quant_spec == SPEC
    assert type(freeform) is QuantFreeformRawTask and freeform.bar_cap == 0.03


async def test_add_task_inserts_upstream():
    psql, conn = _mock_psql(fetchrow={cst.TASK_ID: TASK_ID})
    task = QuantFreeformRawTask(**_raw_task_fields(), upstream_task_id=UPSTREAM_TASK_ID)
    await task_sql.add_task(task, psql)

    query = next(c.args[0] for c in conn.execute.await_args_list if f"INSERT INTO {cst.QUANT_TASKS_TABLE}" in c.args[0])
    columns = [c.strip() for c in query.split("(", 1)[1].split(")", 1)[0].split(",")]
    values = dict(zip(columns, _quant_insert_args(conn)))
    assert "reference_hotkey" not in columns
    # The per-SKU references live in quant_sku_references, never on the task row.
    assert not {"reference_loss", "reference_per_example_losses", "reference_fingerprint"} & set(columns)
    assert values[cst.UPSTREAM_TASK_ID] == UPSTREAM_TASK_ID
