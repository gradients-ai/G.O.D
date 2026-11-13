from uuid import UUID

from asyncpg.connection import Connection

import validator.db.constants as cst
from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.tournament.models import ChunkScheme
from validator.tournament.models import ContinuousSftState
from validator.tournament.models import ContinuousSftTaskChunk


logger = get_logger(__name__)


async def get_continuous_sft_state(lineage: str, psql_db: PSQLDB) -> ContinuousSftState:
    """Read a lineage's continuous-SFT state row, tolerating a missing (not-yet-created) row."""
    async with await psql_db.connection() as connection:
        connection: Connection
        row = await connection.fetchrow(
            f"""
            SELECT {cst.CONTINUOUS_SFT_TRAIN_INDEX}, {cst.CONTINUOUS_SFT_CHUNK_SCHEME},
                   {cst.CONTINUOUS_SFT_LAST_WINNER_REPO}, {cst.CONTINUOUS_SFT_UPDATED_AT}
            FROM {cst.CONTINUOUS_SFT_STATE_TABLE}
            WHERE {cst.CONTINUOUS_SFT_LINEAGE} = $1
            """,
            lineage,
        )
    if row is None:
        # No row = lineage's first tournament, OR a slug rename/removal (silent reset to seed) - log it.
        logger.info(
            f"continuous_sft_state[{lineage}] has no row; starting from seed (train_index=0) - "
            f"expected on first run, otherwise check the lineage slug wasn't renamed"
        )
        return ContinuousSftState(lineage=lineage, train_index=0, last_winner_repo=None)
    return ContinuousSftState(
        lineage=lineage,
        train_index=row[cst.CONTINUOUS_SFT_TRAIN_INDEX],
        chunk_scheme=ChunkScheme(row[cst.CONTINUOUS_SFT_CHUNK_SCHEME]),
        last_winner_repo=row[cst.CONTINUOUS_SFT_LAST_WINNER_REPO],
        updated_at=row[cst.CONTINUOUS_SFT_UPDATED_AT],
    )


async def warn_orphaned_continuous_sft_state(known_lineages: set[str], psql_db: PSQLDB) -> None:
    """WARN on continuous_sft_state rows whose lineage isn't in CONTINUOUS_SFT_LINEAGES - a slug
    rename/removal strands the accumulated row while the new slug restarts from seed."""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT {cst.CONTINUOUS_SFT_LINEAGE}, {cst.CONTINUOUS_SFT_TRAIN_INDEX}, {cst.CONTINUOUS_SFT_LAST_WINNER_REPO}
            FROM {cst.CONTINUOUS_SFT_STATE_TABLE}
            """
        )
    for row in rows:
        lineage = row[cst.CONTINUOUS_SFT_LINEAGE]
        if lineage not in known_lineages:
            logger.warning(
                f"Orphaned continuous_sft_state row: lineage={lineage!r} "
                f"(train_index={row[cst.CONTINUOUS_SFT_TRAIN_INDEX]}, "
                f"last_winner_repo={row[cst.CONTINUOUS_SFT_LAST_WINNER_REPO]}) is not in "
                f"CONTINUOUS_SFT_LINEAGES — its accumulated chain is stranded (slug renamed/removed?)"
            )


async def record_continuous_sft_task_chunk(chunk: ContinuousSftTaskChunk, psql_db: PSQLDB) -> None:
    """Record the cursor value and chunk scheme a continuous-SFT task was created with (once per task)."""
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            f"""
            INSERT INTO {cst.CONTINUOUS_SFT_TASK_CHUNKS_TABLE}
                ({cst.TASK_ID}, {cst.CONTINUOUS_SFT_LINEAGE}, {cst.CONTINUOUS_SFT_TRAIN_INDEX},
                 {cst.CONTINUOUS_SFT_CHUNK_SCHEME})
            VALUES ($1, $2, $3, $4)
            ON CONFLICT ({cst.TASK_ID}) DO NOTHING
            """,
            chunk.task_id,
            chunk.lineage,
            chunk.train_index,
            chunk.chunk_scheme.value,
        )


async def get_continuous_sft_task_chunk(task_id: UUID, psql_db: PSQLDB) -> ContinuousSftTaskChunk | None:
    """The chunk record of a continuous-SFT task; None for a task created before chunk records existed."""
    async with await psql_db.connection() as connection:
        connection: Connection
        row = await connection.fetchrow(
            f"""
            SELECT {cst.CONTINUOUS_SFT_LINEAGE}, {cst.CONTINUOUS_SFT_TRAIN_INDEX}, {cst.CONTINUOUS_SFT_CHUNK_SCHEME}
            FROM {cst.CONTINUOUS_SFT_TASK_CHUNKS_TABLE}
            WHERE {cst.TASK_ID} = $1
            """,
            task_id,
        )
    if row is None:
        return None
    return ContinuousSftTaskChunk(
        task_id=task_id,
        lineage=row[cst.CONTINUOUS_SFT_LINEAGE],
        train_index=row[cst.CONTINUOUS_SFT_TRAIN_INDEX],
        chunk_scheme=ChunkScheme(row[cst.CONTINUOUS_SFT_CHUNK_SCHEME]),
    )


def next_train_index_after(chunk: ContinuousSftTaskChunk | None, current_train_index: int) -> int:
    """The cursor (in v2 chunk units) a completed task leaves behind; never below the current cursor.

    V2 task: the next chunk, index + 1. V1 task: its chunk k is the V2 chunks
    2k and 2k+1, so both are consumed and the next unread one is 2k + 2. No record means the task was
    created before the scheme switch (a V1 task whose index was not recorded): the migration
    doubled the cursor the task was created from, so the current value is 2k and the next unread
    chunk is current + 2.
    """
    if chunk is None:
        following = current_train_index + 2
    elif chunk.chunk_scheme == ChunkScheme.V2:
        following = chunk.train_index + 1
    else:
        following = 2 * chunk.train_index + 2
    return max(current_train_index, following)


async def advance_continuous_sft_state(
    lineage: str, winner_repo: str | None, source_round_id: str, psql_db: PSQLDB, task_id: UUID | None = None
) -> None:
    """Advance a lineage's train cursor and record its winner (upsert; creates the row on first run).

    The cursor moves past the chunk the task trained on, in v2 chunk units (see next_train_index_after, which
    the SQL CASE mirrors so the arithmetic runs on the current row inside the one statement), and never
    backwards. task_id selects the task's chunk record; without one the cursor moves by two, as for a task
    created before the scheme switch. winner_repo=None is preserved via COALESCE (a failed/empty week doesn't
    discard the chain). Idempotent per source_round_id (the ON CONFLICT WHERE no-ops a repeat), so a crash
    between carry-forward and the winner_hotkey guard can't double-advance train_index.
    """
    chunk = await get_continuous_sft_task_chunk(task_id, psql_db) if task_id is not None else None
    task_index = chunk.train_index if chunk else None
    task_scheme = chunk.chunk_scheme.value if chunk else None
    state_table = cst.CONTINUOUS_SFT_STATE_TABLE
    train_index = cst.CONTINUOUS_SFT_TRAIN_INDEX
    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            row = await connection.fetchrow(
                f"""
                INSERT INTO {state_table}
                    ({cst.CONTINUOUS_SFT_LINEAGE}, {train_index}, {cst.CONTINUOUS_SFT_CHUNK_SCHEME},
                     {cst.CONTINUOUS_SFT_LAST_WINNER_REPO}, {cst.CONTINUOUS_SFT_LAST_SOURCE_ROUND_ID})
                VALUES ($1,
                        CASE $5::text WHEN '{ChunkScheme.V2.value}' THEN $4::int + 1
                                      WHEN '{ChunkScheme.V1.value}' THEN 2 * $4::int + 2
                                      ELSE 2 END,
                        '{ChunkScheme.V2.value}', $2, $3)
                ON CONFLICT ({cst.CONTINUOUS_SFT_LINEAGE}) DO UPDATE
                SET {train_index} = GREATEST(
                        {state_table}.{train_index},
                        CASE $5::text WHEN '{ChunkScheme.V2.value}' THEN $4::int + 1
                                      WHEN '{ChunkScheme.V1.value}' THEN 2 * $4::int + 2
                                      ELSE {state_table}.{train_index} + 2 END),
                    {cst.CONTINUOUS_SFT_CHUNK_SCHEME} = '{ChunkScheme.V2.value}',
                    {cst.CONTINUOUS_SFT_LAST_WINNER_REPO} =
                        COALESCE($2, {state_table}.{cst.CONTINUOUS_SFT_LAST_WINNER_REPO}),
                    {cst.CONTINUOUS_SFT_LAST_SOURCE_ROUND_ID} = $3,
                    {cst.CONTINUOUS_SFT_UPDATED_AT} = CURRENT_TIMESTAMP
                WHERE {state_table}.{cst.CONTINUOUS_SFT_LAST_SOURCE_ROUND_ID}
                      IS DISTINCT FROM $3
                RETURNING {train_index}, {cst.CONTINUOUS_SFT_LAST_WINNER_REPO}
                """,
                lineage,
                winner_repo,
                source_round_id,
                task_index,
                task_scheme,
            )
    if row is None:
        logger.info(
            f"continuous_sft_state[{lineage}] already advanced for round {source_round_id}; "
            f"skipping (idempotent no-op)"
        )
        return
    logger.info(
        f"Advanced continuous_sft_state[{lineage}] -> train_index={row[cst.CONTINUOUS_SFT_TRAIN_INDEX]}, "
        f"last_winner_repo={row[cst.CONTINUOUS_SFT_LAST_WINNER_REPO]}"
    )
