from asyncpg.connection import Connection

import validator.db.constants as cst
from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.tournament.models import ContinuousSftState


logger = get_logger(__name__)


async def get_continuous_sft_state(lineage: str, psql_db: PSQLDB) -> ContinuousSftState:
    """Read a lineage's continuous-SFT state row, tolerating a missing (not-yet-created) row."""
    async with await psql_db.connection() as connection:
        connection: Connection
        row = await connection.fetchrow(
            f"""
            SELECT {cst.CONTINUOUS_SFT_TRAIN_INDEX}, {cst.CONTINUOUS_SFT_LAST_WINNER_REPO}, {cst.CONTINUOUS_SFT_UPDATED_AT}
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


async def advance_continuous_sft_state(
    lineage: str, winner_repo: str | None, source_round_id: str, psql_db: PSQLDB
) -> None:
    """Advance a lineage's train cursor and record its winner (upsert; creates the row on first run).

    winner_repo=None is preserved via COALESCE (a failed/empty week doesn't discard the chain).
    Idempotent per source_round_id (the ON CONFLICT WHERE no-ops a repeat), so a crash between
    carry-forward and the winner_hotkey guard can't double-advance train_index.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            row = await connection.fetchrow(
                f"""
                INSERT INTO {cst.CONTINUOUS_SFT_STATE_TABLE}
                    ({cst.CONTINUOUS_SFT_LINEAGE}, {cst.CONTINUOUS_SFT_TRAIN_INDEX},
                     {cst.CONTINUOUS_SFT_LAST_WINNER_REPO}, {cst.CONTINUOUS_SFT_LAST_SOURCE_ROUND_ID})
                VALUES ($1, 1, $2, $3)
                ON CONFLICT ({cst.CONTINUOUS_SFT_LINEAGE}) DO UPDATE
                SET {cst.CONTINUOUS_SFT_TRAIN_INDEX} =
                        {cst.CONTINUOUS_SFT_STATE_TABLE}.{cst.CONTINUOUS_SFT_TRAIN_INDEX} + 1,
                    {cst.CONTINUOUS_SFT_LAST_WINNER_REPO} =
                        COALESCE($2, {cst.CONTINUOUS_SFT_STATE_TABLE}.{cst.CONTINUOUS_SFT_LAST_WINNER_REPO}),
                    {cst.CONTINUOUS_SFT_LAST_SOURCE_ROUND_ID} = $3,
                    {cst.CONTINUOUS_SFT_UPDATED_AT} = CURRENT_TIMESTAMP
                WHERE {cst.CONTINUOUS_SFT_STATE_TABLE}.{cst.CONTINUOUS_SFT_LAST_SOURCE_ROUND_ID}
                      IS DISTINCT FROM $3
                RETURNING {cst.CONTINUOUS_SFT_TRAIN_INDEX}, {cst.CONTINUOUS_SFT_LAST_WINNER_REPO}
                """,
                lineage,
                winner_repo,
                source_round_id,
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
