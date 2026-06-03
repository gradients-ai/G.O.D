import hashlib
import re
from uuid import UUID

import validator.db.constants as cst
from core.logging import get_logger
from core.models.reward_models import RewardFunction
from validator.db.database import PSQLDB


logger = get_logger(__name__)


async def get_all_reward_functions(psql_db: PSQLDB) -> list[RewardFunction]:
    """Get all GRPO reward functions from the database.

    Returns:
        List of RewardFunction objects containing reward function information
    """
    query = f"""
        SELECT {cst.REWARD_ID}, {cst.REWARD_FUNC}, {cst.FUNC_HASH}, {cst.IS_GENERIC}
        FROM {cst.REWARD_FUNCTIONS_TABLE}
        WHERE {cst.IS_MANUAL} = true
        ORDER BY {cst.CREATED_AT} DESC
    """

    async with await psql_db.connection() as connection:
        rows = await connection.fetch(query)
        return [
            RewardFunction(
                reward_id=str(row[cst.REWARD_ID]),
                reward_func=row[cst.REWARD_FUNC],
                func_hash=row[cst.FUNC_HASH],
                is_generic=row[cst.IS_GENERIC],
                reward_weight=1.0,  # Default weight for listing
            )
            for row in rows
        ]


async def add_reward_function(
    psql_db: PSQLDB, name: str, description: str, code: str, reward_weight: float | None = None
) -> UUID:
    """Add a new reward function to the database.

    Args:
        psql_db: Database connection
        name: Function name
        description: Function description
        code: Complete function code or just the function body
        reward_weight: Optional weight for the function

    Returns:
        UUID of the created reward function
    """
    # Check if code already contains a function definition
    has_function_def = re.match(r"^\s*def\s+\w+\s*\(", code, re.MULTILINE)

    if has_function_def:
        # Check if it already has a docstring
        has_docstring = re.search(r'def\s+\w+\s*\([^)]*\):\s*\n\s*(""".*?"""|\'\'\'.*?\'\'\')', code, re.DOTALL)

        if has_docstring:
            # Use code as is - it has both function def and docstring
            reward_func = code
        else:
            # Add docstring after function definition
            lines = code.split("\n")
            for i, line in enumerate(lines):
                if re.match(r"^\s*def\s+\w+\s*\([^)]*\):\s*$", line):
                    # Insert docstring after the def line
                    indent = len(line) - len(line.lstrip())
                    lines.insert(i + 1, f'{" " * (indent + 4)}"""{description}"""')
                    break
            reward_func = "\n".join(lines)
    else:
        # Build the complete function code with docstring
        # Indent the code body properly
        indented_code = "\n".join(f"    {line}" for line in code.strip().split("\n"))
        reward_func = f'def {name}(completions, **kwargs):\n    """{description}"""\n{indented_code}'

    func_hash = hashlib.sha256(reward_func.encode()).hexdigest()

    query = f"""
        INSERT INTO {cst.REWARD_FUNCTIONS_TABLE}
        ({cst.REWARD_FUNC}, {cst.FUNC_HASH}, {cst.IS_GENERIC}, {cst.IS_MANUAL})
        VALUES ($1, $2, $3, $4)
        ON CONFLICT ({cst.FUNC_HASH}) DO UPDATE
        SET {cst.REWARD_FUNC} = EXCLUDED.{cst.REWARD_FUNC}
        RETURNING {cst.REWARD_ID}
    """

    async with await psql_db.connection() as connection:
        reward_id = await connection.fetchval(
            query,
            reward_func,
            func_hash,
            False,  # is_generic is always False as mentioned
            True,  # is_manual is True for manually added functions
        )
        logger.info(f"Added reward function {name} with ID {reward_id}")
        return reward_id


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
