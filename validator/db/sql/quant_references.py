"""The per-(task, hotkey, SKU) bf16 reference cache every quant loss is normalised by (quant_sku_references)."""

from uuid import UUID

from asyncpg.connection import Connection

import validator.db.constants as cst
from core.models.quant_models import QuantEvalGpu
from validator.db.database import PSQLDB
from validator.db.sql.submissions_and_scoring import json_safe_losses
from validator.db.sql.submissions_and_scoring import losses_from_column
from validator.scoring.models import QuantSkuReference
from validator.scoring.models import QuantSkuReferences


_COLUMNS = (
    f"{cst.TASK_ID}, {cst.HOTKEY}, {cst.SKU}, {cst.REFERENCE_MODEL}, {cst.LOSS}, {cst.PER_EXAMPLE_LOSSES}, "
    f"{cst.EVAL_SET_FINGERPRINT}, {cst.EVAL_GPU}, {cst.MEASURED_AT}"
)


def _reference_from_row(row) -> QuantSkuReference:
    return QuantSkuReference(
        task_id=row[cst.TASK_ID],
        hotkey=row[cst.HOTKEY],
        sku=row[cst.SKU],
        model=row[cst.REFERENCE_MODEL],
        loss=row[cst.LOSS],
        per_example_losses=losses_from_column(row[cst.PER_EXAMPLE_LOSSES]),
        eval_set_fingerprint=row[cst.EVAL_SET_FINGERPRINT],
        eval_gpu=QuantEvalGpu.model_validate(row[cst.EVAL_GPU]),
        measured_at=row[cst.MEASURED_AT],
    )


async def add_quant_sku_reference(reference: QuantSkuReference, psql_db: PSQLDB) -> QuantSkuReference:
    """Insert the reference unless its (task, hotkey, SKU) is already cached; returns the stored row.

    First writer wins: two deployments on one SKU can both measure before either is cached, and the caller
    compares what it measured with what is stored rather than overwriting it.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        # per_example_losses and eval_gpu bind directly: jsonb codec (see set_task_node_losses). An example with
        # no supervised token is NaN in the scorer's vector and must go in as null, as for a submission.
        await connection.execute(
            f"""
            INSERT INTO {cst.QUANT_SKU_REFERENCES_TABLE}
            ({cst.TASK_ID}, {cst.HOTKEY}, {cst.SKU}, {cst.REFERENCE_MODEL}, {cst.LOSS}, {cst.PER_EXAMPLE_LOSSES},
             {cst.EVAL_SET_FINGERPRINT}, {cst.EVAL_GPU})
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8::jsonb)
            ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.SKU}) DO NOTHING
            """,
            reference.task_id,
            reference.hotkey,
            reference.sku,
            reference.model,
            reference.loss,
            json_safe_losses(reference.per_example_losses),
            reference.eval_set_fingerprint,
            reference.eval_gpu.model_dump(),
        )
        row = await connection.fetchrow(
            f"""
            SELECT {_COLUMNS} FROM {cst.QUANT_SKU_REFERENCES_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2 AND {cst.SKU} = $3
            """,
            reference.task_id,
            reference.hotkey,
            reference.sku,
        )
    return _reference_from_row(row)


async def get_quant_sku_references(task_id: UUID, psql_db: PSQLDB) -> QuantSkuReferences:
    """The task's cached references by (hotkey, reference_key); empty before any deployment measured one."""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"SELECT {_COLUMNS} FROM {cst.QUANT_SKU_REFERENCES_TABLE} WHERE {cst.TASK_ID} = $1",
            task_id,
        )
    references = [_reference_from_row(row) for row in rows]
    return {reference.key: reference for reference in references}
