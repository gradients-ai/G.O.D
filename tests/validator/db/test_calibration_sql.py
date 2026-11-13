"""Calibration persistence (mocked DB): the per-dataset row record, the per-(task, hotkey) reference, the task columns."""

import math
import re
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import validator.db.constants as cst
from core.models.quant_models import QuantEvalGpu
from validator.db.sql import calibration as calibration_sql
from validator.db.sql import quant_calibration_references as references_sql
from validator.db.sql import tasks as task_sql
from validator.scoring.models import QuantCalibrationReferenceRecord
from validator.tasks.models import InstructTextRawTask


TASK_ID = uuid.uuid4()
GPU = QuantEvalGpu(name="NVIDIA H100 PCIe", sm_count=114, sku="H100 PCIe")


def _mock_psql(*, fetchrow=None, fetch=None):
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow)
    conn.fetch = AsyncMock(return_value=fetch or [])
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.executemany = AsyncMock(return_value=None)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    psql = MagicMock()
    psql.connection = AsyncMock(return_value=cm)
    return psql, conn


async def test_recorded_hashes_are_read_per_dataset_and_a_task_can_exclude_its_own():
    psql, conn = _mock_psql(fetch=[{cst.ROW_HASH: "a"}, {cst.ROW_HASH: "b"}])

    hashes = await calibration_sql.get_calibration_row_hashes("org/ds", psql, exclude_task_id=TASK_ID)

    query, *args = conn.fetch.await_args.args
    assert hashes == {"a", "b"} and args == ["org/ds", TASK_ID]
    assert "IS DISTINCT FROM" in query


async def test_no_database_reads_nothing():
    assert await calibration_sql.get_calibration_row_hashes("org/ds", None) == set()


async def test_record_inserts_each_hash_once_and_never_overwrites():
    psql, conn = _mock_psql()

    await calibration_sql.record_calibration_rows("org/ds", ["a", "b"], TASK_ID, psql)

    query, rows = conn.executemany.await_args.args
    assert f"ON CONFLICT ({cst.DS}, {cst.ROW_HASH}) DO NOTHING" in query
    assert rows == [("org/ds", "a", TASK_ID), ("org/ds", "b", TASK_ID)]


async def test_record_without_a_database_or_rows_is_skipped():
    psql, conn = _mock_psql()

    await calibration_sql.record_calibration_rows("org/ds", ["a"], TASK_ID, None)
    await calibration_sql.record_calibration_rows("org/ds", [], TASK_ID, psql)

    conn.executemany.assert_not_called()


def _record(loss: float = 1.5, losses=(1.5, math.nan)) -> QuantCalibrationReferenceRecord:
    return QuantCalibrationReferenceRecord(
        task_id=TASK_ID, hotkey="miner", model="org/own", loss=loss, per_example_losses=list(losses),
        eval_set_fingerprint="fp", eval_gpu=GPU,
    )  # fmt: skip


def _row(record: QuantCalibrationReferenceRecord) -> dict:
    return {
        cst.TASK_ID: record.task_id,
        cst.HOTKEY: record.hotkey,
        cst.REFERENCE_MODEL: record.model,
        cst.LOSS: record.loss,
        cst.PER_EXAMPLE_LOSSES: [None if math.isnan(x) else x for x in record.per_example_losses],
        cst.EVAL_SET_FINGERPRINT: record.eval_set_fingerprint,
        cst.EVAL_GPU: record.eval_gpu.model_dump(),
        cst.MEASURED_AT: datetime(2026, 11, 7),
    }


async def test_reference_add_is_first_writer_wins_and_unscored_examples_go_in_as_null():
    stored = _record(1.5)
    psql, conn = _mock_psql(fetchrow=_row(stored))

    returned = await references_sql.add_quant_calibration_reference(_record(1.6), psql)

    query, *args = conn.execute.await_args.args
    assert f"ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}) DO NOTHING" in query
    assert args == [TASK_ID, "miner", "org/own", 1.6, [1.5, None], "fp", GPU.model_dump()]
    assert returned.loss == 1.5 and math.isnan(returned.per_example_losses[1])


async def test_references_are_read_by_hotkey():
    psql, conn = _mock_psql(fetch=[_row(_record())])

    references = await references_sql.get_quant_calibration_references(TASK_ID, psql)

    assert list(references) == ["miner"] and references["miner"].sku == "H100 PCIe"


async def test_add_task_binds_every_placeholder_including_the_calibration_columns():
    conn = MagicMock()
    conn.fetchrow = AsyncMock(return_value={cst.TASK_ID: TASK_ID})
    task = InstructTextRawTask(
        is_organic=False, status="pending", model_id="m", ds="ds", account_id=uuid.uuid4(), hours_to_complete=1.0,
        created_at=datetime(2026, 11, 7), field_instruction="q", field_output="a",
        calibration_data="https://c", calibration_fingerprint="fp", calibration_n_examples=3, carve_calibration=True,
    )  # fmt: skip

    await task_sql._insert_base_task(conn, task)

    query, *args = conn.fetchrow.await_args.args
    assert max(int(n) for n in re.findall(r"\$(\d+)", query)) == len(args)
    for column in (cst.CALIBRATION_DATA, cst.CALIBRATION_FINGERPRINT, cst.CALIBRATION_N_EXAMPLES, cst.CARVE_CALIBRATION):
        assert column in query
    assert args[-4:] == ["https://c", "fp", 3, True]


def test_the_migration_adds_every_column_the_model_and_the_sql_use():
    migration = (Path(__file__).parents[3] / "validator/db/migrations/20261107120000_quant_calibration_set.sql").read_text()
    for column in (cst.CALIBRATION_DATA, cst.CALIBRATION_FINGERPRINT, cst.CALIBRATION_N_EXAMPLES, cst.CARVE_CALIBRATION):
        assert f"ALTER TABLE tasks ADD COLUMN IF NOT EXISTS {column}" in migration
    assert cst.DATASET_CALIBRATION_ROWS_TABLE in migration and cst.QUANT_CALIBRATION_REFERENCES_TABLE in migration
    assert migration.index("-- migrate:up") < migration.index("-- migrate:down")
