"""The per-(task, hotkey) bf16 loss on the calibration rows published to a freeform miner (quant_calibration_references)."""

from uuid import UUID

from asyncpg.connection import Connection

import validator.db.constants as cst
from core.models.quant_models import QuantEvalGpu
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import json_safe_losses
from validator.db.sql.submissions_and_scoring import losses_from_column
from validator.scoring.models import QuantCalibrationReferenceRecord


_COLUMNS = (
    f"{cst.TASK_ID}, {cst.HOTKEY}, {cst.REFERENCE_MODEL}, {cst.LOSS}, {cst.PER_EXAMPLE_LOSSES}, "
    f"{cst.EVAL_SET_FINGERPRINT}, {cst.EVAL_GPU}, {cst.MEASURED_AT}"
)


def _record_from_row(row) -> QuantCalibrationReferenceRecord:
    return QuantCalibrationReferenceRecord(
        task_id=row[cst.TASK_ID],
        hotkey=row[cst.HOTKEY],
        model=row[cst.REFERENCE_MODEL],
        loss=row[cst.LOSS],
        per_example_losses=losses_from_column(row[cst.PER_EXAMPLE_LOSSES]),
        eval_set_fingerprint=row[cst.EVAL_SET_FINGERPRINT],
        eval_gpu=QuantEvalGpu.model_validate(row[cst.EVAL_GPU]),
        measured_at=row[cst.MEASURED_AT],
    )


async def add_quant_calibration_reference(
    reference: QuantCalibrationReferenceRecord, psql_db: PSQLDB
) -> QuantCalibrationReferenceRecord:
    """Insert the reference, or replace the stored (task, hotkey) row when it was measured on another example set; a stored row
    on the same example set is kept (first writer wins). Returns the stored row."""
    async with await psql_db.connection() as connection:
        connection: Connection
        # per_example_losses and eval_gpu bind directly: jsonb codec; an unscored example goes in as null.
        await connection.execute(
            f"""
            INSERT INTO {cst.QUANT_CALIBRATION_REFERENCES_TABLE} AS stored
            ({cst.TASK_ID}, {cst.HOTKEY}, {cst.REFERENCE_MODEL}, {cst.LOSS}, {cst.PER_EXAMPLE_LOSSES},
             {cst.EVAL_SET_FINGERPRINT}, {cst.EVAL_GPU})
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::jsonb)
            ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}) DO UPDATE SET
                {cst.REFERENCE_MODEL} = EXCLUDED.{cst.REFERENCE_MODEL},
                {cst.LOSS} = EXCLUDED.{cst.LOSS},
                {cst.PER_EXAMPLE_LOSSES} = EXCLUDED.{cst.PER_EXAMPLE_LOSSES},
                {cst.EVAL_SET_FINGERPRINT} = EXCLUDED.{cst.EVAL_SET_FINGERPRINT},
                {cst.EVAL_GPU} = EXCLUDED.{cst.EVAL_GPU},
                {cst.MEASURED_AT} = NOW()
            WHERE stored.{cst.EVAL_SET_FINGERPRINT} IS DISTINCT FROM EXCLUDED.{cst.EVAL_SET_FINGERPRINT}
            """,
            reference.task_id,
            reference.hotkey,
            reference.model,
            reference.loss,
            json_safe_losses(reference.per_example_losses),
            reference.eval_set_fingerprint,
            reference.eval_gpu.model_dump(),
        )
        row = await connection.fetchrow(
            f"SELECT {_COLUMNS} FROM {cst.QUANT_CALIBRATION_REFERENCES_TABLE} WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2",
            reference.task_id,
            reference.hotkey,
        )
    return _record_from_row(row)


async def get_quant_calibration_references(task_id: UUID, psql_db: PSQLDB) -> dict[str, QuantCalibrationReferenceRecord]:
    """The task's stored calibration references by hotkey; empty before any was measured."""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"SELECT {_COLUMNS} FROM {cst.QUANT_CALIBRATION_REFERENCES_TABLE} WHERE {cst.TASK_ID} = $1",
            task_id,
        )
    return {record.hotkey: record for record in map(_record_from_row, rows)}
