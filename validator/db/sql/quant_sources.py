"""SQL for quant_tournament_sources and quant_task_sources: which text tournament feeds a quant tournament, and which text
seed task and boss model each quant task came from (see validator/tournament/quant_sources.py)."""

from uuid import UUID

from asyncpg.connection import Connection
from pydantic import BaseModel

import validator.db.constants as cst
from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.tournament.models import QuantSourceSlot
from validator.tournament.models import QuantTaskSource
from validator.tournament.models import QuantTournamentSource


logger = get_logger(__name__)

_TOURNAMENT_SOURCE_COLUMNS = f"{cst.QUANT_TOURNAMENT_ID}, {cst.QUANT_SOURCE_TOURNAMENT_ID}, {cst.CREATED_AT}"
_TASK_SOURCE_COLUMNS = (
    f"{cst.QUANT_TASK_ID}, {cst.QUANT_TOURNAMENT_ID}, {cst.QUANT_SOURCE_SLOT}, {cst.QUANT_SOURCE_TOURNAMENT_ID}, "
    f"{cst.QUANT_SOURCE_ROLE}, {cst.QUANT_SOURCE_TASK_ID}, {cst.QUANT_SOURCE_MODEL_REPO}, {cst.QUANT_SOURCE_IS_FALLBACK}, "
    f"{cst.QUANT_SOURCE_SUPERSEDED}, {cst.CREATED_AT}, {cst.UPDATED_AT}"
)


class QuantTaskSourceInsert(BaseModel):
    """The result of recording a task source: the live row of the slot and whether it is the row just written (False: the
    slot already had a live task, so the caller must discard the task it just created)."""

    source: QuantTaskSource
    inserted: bool


def _tournament_source_from_row(row) -> QuantTournamentSource:
    return QuantTournamentSource(**dict(row))


def _task_source_from_row(row) -> QuantTaskSource:
    return QuantTaskSource(**dict(row))


async def bind_quant_tournament(
    quant_tournament_id: str, source_tournament_id: str, psql_db: PSQLDB
) -> QuantTournamentSource | None:
    """Bind a quant tournament to a text tournament, then read the binding back so two passes (or two processes) agree.

    The insert does nothing when the quant tournament is already bound, or when the text tournament feeds another quant
    tournament (the unique source). The row read back is the quant tournament's own: None when it has none, i.e. the text
    tournament was taken by another quant tournament.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            f"""
            INSERT INTO {cst.QUANT_TOURNAMENT_SOURCES_TABLE} ({cst.QUANT_TOURNAMENT_ID}, {cst.QUANT_SOURCE_TOURNAMENT_ID})
            VALUES ($1, $2)
            ON CONFLICT DO NOTHING
            """,
            quant_tournament_id,
            source_tournament_id,
        )
        row = await connection.fetchrow(
            f"""
            SELECT {_TOURNAMENT_SOURCE_COLUMNS} FROM {cst.QUANT_TOURNAMENT_SOURCES_TABLE}
            WHERE {cst.QUANT_TOURNAMENT_ID} = $1
            """,
            quant_tournament_id,
        )
    return _tournament_source_from_row(row) if row else None


async def get_quant_tournament_source(quant_tournament_id: str, psql_db: PSQLDB) -> QuantTournamentSource | None:
    """The text tournament a quant tournament is bound to, or None while it is not bound."""
    async with await psql_db.connection() as connection:
        connection: Connection
        row = await connection.fetchrow(
            f"""
            SELECT {_TOURNAMENT_SOURCE_COLUMNS} FROM {cst.QUANT_TOURNAMENT_SOURCES_TABLE}
            WHERE {cst.QUANT_TOURNAMENT_ID} = $1
            """,
            quant_tournament_id,
        )
    return _tournament_source_from_row(row) if row else None


async def get_bound_source_tournament_ids(psql_db: PSQLDB) -> set[str]:
    """The text tournaments that already feed a quant tournament."""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(f"SELECT {cst.QUANT_SOURCE_TOURNAMENT_ID} FROM {cst.QUANT_TOURNAMENT_SOURCES_TABLE}")
    return {row[cst.QUANT_SOURCE_TOURNAMENT_ID] for row in rows}


async def insert_quant_task_source(
    source: QuantTaskSource, psql_db: PSQLDB, replacing_task_id: UUID | None = None
) -> QuantTaskSourceInsert:
    """Record where a new quant task came from.

    The partial unique index allows one live row per (quant tournament, slot), so a second task for a slot never succeeds
    silently: on a conflict the existing live row is returned with inserted=False and the caller deletes the task it just
    created. replacing_task_id is the task this one replaces; its row is marked superseded in the same transaction, before
    the insert, so the slot is free for the new row.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            if replacing_task_id is not None:
                await connection.execute(
                    f"""
                    UPDATE {cst.QUANT_TASK_SOURCES_TABLE}
                    SET {cst.QUANT_SOURCE_SUPERSEDED} = TRUE, {cst.UPDATED_AT} = NOW()
                    WHERE {cst.QUANT_TASK_ID} = $1
                    """,
                    replacing_task_id,
                )
            inserted = await connection.fetchrow(
                f"""
                INSERT INTO {cst.QUANT_TASK_SOURCES_TABLE}
                    ({cst.QUANT_TASK_ID}, {cst.QUANT_TOURNAMENT_ID}, {cst.QUANT_SOURCE_SLOT}, {cst.QUANT_SOURCE_TOURNAMENT_ID},
                     {cst.QUANT_SOURCE_ROLE}, {cst.QUANT_SOURCE_TASK_ID}, {cst.QUANT_SOURCE_MODEL_REPO},
                     {cst.QUANT_SOURCE_IS_FALLBACK}, {cst.QUANT_SOURCE_SUPERSEDED})
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT ({cst.QUANT_TOURNAMENT_ID}, {cst.QUANT_SOURCE_SLOT}) WHERE NOT {cst.QUANT_SOURCE_SUPERSEDED}
                DO NOTHING
                RETURNING {_TASK_SOURCE_COLUMNS}
                """,
                source.quant_task_id,
                source.quant_tournament_id,
                source.slot.value,
                source.source_tournament_id,
                source.source_role.value,
                source.source_task_id,
                source.source_model_repo,
                source.source_is_fallback,
                source.superseded,
            )
            if inserted is not None:
                return QuantTaskSourceInsert(source=_task_source_from_row(inserted), inserted=True)
            existing = await connection.fetchrow(
                f"""
                SELECT {_TASK_SOURCE_COLUMNS} FROM {cst.QUANT_TASK_SOURCES_TABLE}
                WHERE {cst.QUANT_TOURNAMENT_ID} = $1 AND {cst.QUANT_SOURCE_SLOT} = $2 AND NOT {cst.QUANT_SOURCE_SUPERSEDED}
                """,
                source.quant_tournament_id,
                source.slot.value,
            )
    if existing is None:
        raise RuntimeError(
            f"Quant task {source.quant_task_id}: could not record its source for slot {source.slot.value} of "
            f"{source.quant_tournament_id}"
        )
    return QuantTaskSourceInsert(source=_task_source_from_row(existing), inserted=False)


async def get_live_quant_task_source(quant_tournament_id: str, slot: QuantSourceSlot, psql_db: PSQLDB) -> QuantTaskSource | None:
    """The live (not superseded) source row of a slot of a quant tournament, or None."""
    async with await psql_db.connection() as connection:
        connection: Connection
        row = await connection.fetchrow(
            f"""
            SELECT {_TASK_SOURCE_COLUMNS} FROM {cst.QUANT_TASK_SOURCES_TABLE}
            WHERE {cst.QUANT_TOURNAMENT_ID} = $1 AND {cst.QUANT_SOURCE_SLOT} = $2 AND NOT {cst.QUANT_SOURCE_SUPERSEDED}
            """,
            quant_tournament_id,
            slot.value,
        )
    return _task_source_from_row(row) if row else None


async def list_live_quant_sources_naming_task(source_task_id: UUID | str, psql_db: PSQLDB) -> list[QuantTaskSource]:
    """The live (not superseded) source rows whose source_task_id is this task: the quant tasks that depend on a text seed."""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT {_TASK_SOURCE_COLUMNS} FROM {cst.QUANT_TASK_SOURCES_TABLE}
            WHERE {cst.QUANT_SOURCE_TASK_ID} = $1 AND NOT {cst.QUANT_SOURCE_SUPERSEDED}
            ORDER BY {cst.QUANT_TOURNAMENT_ID}, {cst.QUANT_SOURCE_SLOT}
            """,
            UUID(str(source_task_id)),
        )
    return [_task_source_from_row(row) for row in rows]


async def get_quant_task_source(quant_task_id: UUID | str, psql_db: PSQLDB) -> QuantTaskSource | None:
    """The source row of a quant task (live or superseded), or None when the task is not a task of a bound quant tournament."""
    async with await psql_db.connection() as connection:
        connection: Connection
        row = await connection.fetchrow(
            f"SELECT {_TASK_SOURCE_COLUMNS} FROM {cst.QUANT_TASK_SOURCES_TABLE} WHERE {cst.QUANT_TASK_ID} = $1",
            UUID(str(quant_task_id)),
        )
    return _task_source_from_row(row) if row else None


async def list_quant_task_sources(quant_tournament_id: str, psql_db: PSQLDB) -> list[QuantTaskSource]:
    """Every source row of a quant tournament, superseded ones included, oldest first."""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT {_TASK_SOURCE_COLUMNS} FROM {cst.QUANT_TASK_SOURCES_TABLE}
            WHERE {cst.QUANT_TOURNAMENT_ID} = $1
            ORDER BY {cst.CREATED_AT}, {cst.QUANT_SOURCE_SLOT}
            """,
            quant_tournament_id,
        )
    return [_task_source_from_row(row) for row in rows]


async def set_quant_task_source_model(quant_task_id: UUID | str, repo: str, is_fallback: bool, psql_db: PSQLDB) -> None:
    """Record the exact hub repo a quant task was released on, and whether it is the text task's base model (a fallback)."""
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            f"""
            UPDATE {cst.QUANT_TASK_SOURCES_TABLE}
            SET {cst.QUANT_SOURCE_MODEL_REPO} = $2, {cst.QUANT_SOURCE_IS_FALLBACK} = $3, {cst.UPDATED_AT} = NOW()
            WHERE {cst.QUANT_TASK_ID} = $1
            """,
            UUID(str(quant_task_id)),
            repo,
            is_fallback,
        )


async def mark_quant_task_source_superseded(quant_task_id: UUID | str, psql_db: PSQLDB) -> None:
    """Mark a quant task's source row superseded: another task took its slot."""
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            f"""
            UPDATE {cst.QUANT_TASK_SOURCES_TABLE}
            SET {cst.QUANT_SOURCE_SUPERSEDED} = TRUE, {cst.UPDATED_AT} = NOW()
            WHERE {cst.QUANT_TASK_ID} = $1
            """,
            UUID(str(quant_task_id)),
        )


async def repoint_quant_task_source(quant_task_id: UUID | str, new_source_task_id: UUID | str, psql_db: PSQLDB) -> None:
    """Move a held quant task's source from a replaced text seed task to its replacement (only while the task holds no
    model yet: a released task keeps the model it was released on)."""
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            f"""
            UPDATE {cst.QUANT_TASK_SOURCES_TABLE}
            SET {cst.QUANT_SOURCE_TASK_ID} = $2, {cst.UPDATED_AT} = NOW()
            WHERE {cst.QUANT_TASK_ID} = $1 AND {cst.QUANT_SOURCE_MODEL_REPO} IS NULL
            """,
            UUID(str(quant_task_id)),
            UUID(str(new_source_task_id)),
        )
