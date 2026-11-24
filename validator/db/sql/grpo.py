from uuid import UUID

import validator.db.constants as cst
from core.logging import get_logger
from core.models.reward_models import RewardFunction
from validator.db.database import PSQLDB


logger = get_logger(__name__)


async def get_generic_reward_functions_from_db(psql_db: PSQLDB, num_rewards: int) -> list[RewardFunction]:
    """Get random generic reward functions from the database.

    Args:
        psql_db: Database connection
        num_rewards: Number of random reward functions to retrieve

    Returns:
        List of generic RewardFunction objects
    """
    query = f"""
        SELECT {cst.REWARD_ID}, {cst.FUNC_HASH}, {cst.REWARD_FUNC}, {cst.IS_GENERIC}, {cst.IS_MANUAL}
        FROM {cst.REWARD_FUNCTIONS_TABLE}
        WHERE {cst.IS_GENERIC} = true
        AND {cst.IS_MANUAL} = true
        ORDER BY RANDOM()
        LIMIT $1
    """

    async with await psql_db.connection() as conn:
        rows = await conn.fetch(query, num_rewards)
        return [
            RewardFunction(
                reward_id=str(row[cst.REWARD_ID]),
                reward_func=row[cst.REWARD_FUNC],
                func_hash=row[cst.FUNC_HASH],
                is_generic=row[cst.IS_GENERIC],
                is_manual=row[cst.IS_MANUAL],
                reward_weight=1.0,
            )
            for row in rows
        ]


async def get_reward_function_by_id(psql_db: PSQLDB, reward_id: UUID) -> RewardFunction | None:
    """Get a reward function by its ID.

    Args:
        psql_db: Database connection
        reward_id: UUID of the reward function
    """
    query = f"""
        SELECT {cst.REWARD_ID}, {cst.FUNC_HASH}, {cst.REWARD_FUNC}, {cst.IS_GENERIC}
        FROM {cst.REWARD_FUNCTIONS_TABLE}
        WHERE {cst.REWARD_ID} = $1
    """

    async with await psql_db.connection() as conn:
        row = await conn.fetchrow(query, reward_id)
        if row is None:
            return None
        return RewardFunction(
            reward_id=str(row[cst.REWARD_ID]),
            reward_func=row[cst.REWARD_FUNC],
            func_hash=row[cst.FUNC_HASH],
            is_generic=row[cst.IS_GENERIC],
            reward_weight=1.0,
        )
