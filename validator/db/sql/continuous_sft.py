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
        # First run for this lineage: rows are created lazily on completion (advance upserts).
        return ContinuousSftState(lineage=lineage, train_index=0, last_winner_repo=None)
    return ContinuousSftState(
        lineage=lineage,
        train_index=row[cst.CONTINUOUS_SFT_TRAIN_INDEX],
        last_winner_repo=row[cst.CONTINUOUS_SFT_LAST_WINNER_REPO],
        updated_at=row[cst.CONTINUOUS_SFT_UPDATED_AT],
    )


async def advance_continuous_sft_state(lineage: str, winner_repo: str | None, psql_db: PSQLDB) -> None:
    """Record a lineage's just-finished continuous-SFT winner and advance its train cursor.

    train_index -> train_index + 1 (monotonic; the content service does the chunk wrap, so we
    never mod here and stay agnostic to the chunk count).
    last_winner_repo -> winner_repo when set; when winner_repo is None (a failed/empty week
    with no scored submission) the prior lineage is PRESERVED via COALESCE rather than reset to
    the seed, so one bad week does not discard the accumulated chain. The row is upserted so the
    first completion for a lineage creates it.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            row = await connection.fetchrow(
                f"""
                INSERT INTO {cst.CONTINUOUS_SFT_STATE_TABLE}
                    ({cst.CONTINUOUS_SFT_LINEAGE}, {cst.CONTINUOUS_SFT_TRAIN_INDEX}, {cst.CONTINUOUS_SFT_LAST_WINNER_REPO})
                VALUES ($1, 1, $2)
                ON CONFLICT ({cst.CONTINUOUS_SFT_LINEAGE}) DO UPDATE
                SET {cst.CONTINUOUS_SFT_TRAIN_INDEX} =
                        {cst.CONTINUOUS_SFT_STATE_TABLE}.{cst.CONTINUOUS_SFT_TRAIN_INDEX} + 1,
                    {cst.CONTINUOUS_SFT_LAST_WINNER_REPO} =
                        COALESCE($2, {cst.CONTINUOUS_SFT_STATE_TABLE}.{cst.CONTINUOUS_SFT_LAST_WINNER_REPO}),
                    {cst.CONTINUOUS_SFT_UPDATED_AT} = CURRENT_TIMESTAMP
                RETURNING {cst.CONTINUOUS_SFT_TRAIN_INDEX}, {cst.CONTINUOUS_SFT_LAST_WINNER_REPO}
                """,
                lineage,
                winner_repo,
            )
    logger.info(
        f"Advanced continuous_sft_state[{lineage}] -> train_index={row[cst.CONTINUOUS_SFT_TRAIN_INDEX]}, "
        f"last_winner_repo={row[cst.CONTINUOUS_SFT_LAST_WINNER_REPO]}"
    )
