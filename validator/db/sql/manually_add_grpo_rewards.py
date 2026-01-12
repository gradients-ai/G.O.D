import hashlib
import inspect
from typing import Callable

import asyncpg

import validator.db.constants as cst
from validator.utils.logging import get_logger


logger = get_logger(__name__)


async def manually_store_reward_functions(
    connection_string: str,
    reward_functions: list[Callable],
    is_generic: bool = True
):
    """Manually store reward functions in the database."""

    pool = await asyncpg.create_pool(connection_string)

    query = f"""
        INSERT INTO {cst.REWARD_FUNCTIONS_TABLE}
        ({cst.REWARD_FUNC}, {cst.FUNC_HASH}, {cst.IS_GENERIC}, {cst.IS_MANUAL})
        VALUES ($1, $2, $3, $4)
        ON CONFLICT DO NOTHING
    """

    try:
        async with pool.acquire() as conn:
            for func in reward_functions:
                reward_func = inspect.getsource(func)
                func_hash = hashlib.sha256(reward_func.encode()).hexdigest()
                await conn.execute(
                    query,
                    reward_func,
                    func_hash,
                    is_generic,
                    True,
                )
                logger.info(f"Stored/Skipped function: {func.__name__}")
    finally:
        await pool.close()


async def update_is_manual_for_reward_functions(
    connection_string: str,
    is_manual: bool,
    func_names: list[str] | None
):
    """Update the is_manual field for reward functions.

    Args:
        connection_string: Database connection string
        is_manual: Boolean value to set for is_manual field
    """

    pool = await asyncpg.create_pool(connection_string)

    try:
        async with pool.acquire() as conn:
            for func_name in func_names:
                query = f"""
                    UPDATE {cst.REWARD_FUNCTIONS_TABLE}
                    SET {cst.IS_MANUAL} = $1
                    WHERE {cst.REWARD_FUNC} LIKE $2
                """
                pattern = f"%def {func_name}%"
                result = await conn.execute(query, is_manual, pattern)
                logger.info(f"Updated is_manual={is_manual} for function: {func_name}")
    finally:
        await pool.close()


if __name__ == "__main__":
    pass
