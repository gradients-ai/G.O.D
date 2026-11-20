"""The rows already carved into a calibration set, per dataset (dataset_calibration_rows).

A later raw split of the same dataset never puts these rows in its test set, so a calibration row is never scored.
"""

from uuid import UUID

from asyncpg.connection import Connection

import validator.db.constants as cst
from core.logging import get_logger
from validator.db.database import PSQLDB


logger = get_logger(__name__)


async def get_calibration_row_hashes(ds: str, psql_db: PSQLDB | None, exclude_task_id: UUID | None = None) -> set[str]:
    """Row hashes recorded for ds; empty with no database. exclude_task_id skips that task's own rows, so a re-prep
    of the task that carved them reproduces its own split."""
    if psql_db is None:
        return set()
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT {cst.ROW_HASH} FROM {cst.DATASET_CALIBRATION_ROWS_TABLE}
            WHERE {cst.DS} = $1 AND {cst.TASK_ID} IS DISTINCT FROM $2
            """,
            ds,
            exclude_task_id,
        )
    return {row[cst.ROW_HASH] for row in rows}


async def record_calibration_rows(ds: str, hashes: list[str], task_id: UUID, psql_db: PSQLDB | None) -> None:
    """Record hashes as calibration rows of ds, once each; skipped (and logged) with no database."""
    if not hashes:
        return
    if psql_db is None:
        logger.warning(f"No database: not recording {len(hashes)} calibration rows of {ds}")
        return
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.executemany(
            f"""
            INSERT INTO {cst.DATASET_CALIBRATION_ROWS_TABLE} ({cst.DS}, {cst.ROW_HASH}, {cst.TASK_ID})
            VALUES ($1, $2, $3)
            ON CONFLICT ({cst.DS}, {cst.ROW_HASH}) DO NOTHING
            """,
            [(ds, row_hash, task_id) for row_hash in hashes],
        )
