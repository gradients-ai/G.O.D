import json
import os
import socket
from typing import Any

import asyncpg
from asyncpg import Pool
from asyncpg.pool import PoolAcquireContext
from dotenv import load_dotenv

import validator.db.constants as cst
from validator.utils.logging import get_logger


logger = get_logger(__name__)


load_dotenv()


# FIXME: Just use the DSN instead of all this logic to construct it?
def _get_connection_string_from_env() -> str:
    username = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    database = os.getenv("POSTGRES_DB")
    postgres_url = os.getenv("DATABASE_URL")

    if postgres_url:
        return postgres_url

    # FIXME: make these proper conditions with exceptions. Sometimes `assert` can be ignored.
    assert username is not None
    assert password is not None
    assert host is not None
    assert port is not None
    assert database is not None

    port = int(port)

    if not all([username, password, host, port, database]):
        raise ValueError(
            "All of POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_PORT, and POSTGRES_HOST must be set",
            f"But i got: username; {username}, password; *****, host; {host}, port; {port}, database; {database}",
        )

    return _get_connection_string(username, password, host, port, database)


def _get_connection_string(username: str, password: str, host: str, port: int, database: str) -> str:
    return f"postgresql://{username}:{password}@{host}:{port}/{database}"


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register JSON codec so asyncpg deserializes JSONB columns to Python dicts/lists."""
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    await conn.set_type_codec(
        "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


class PSQLDB:
    def __init__(self, from_env: bool = True, connection_string: str | None = None):
        self.from_env = from_env
        self.connection_string = connection_string
        self.pool: Pool | None = None

    async def connect(self) -> None:
        if self.pool is None:
            if self.from_env:
                self.connection_string = _get_connection_string_from_env()
            elif self.connection_string is None:
                raise ValueError("Either from_env must be True or connection_string must be set")

            logger.debug(f"Connecting to {self.connection_string}....")
            try:
                self.pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=cst.MIN_POOL_SIZE,
                    max_size=cst.MAX_POOL_SIZE,
                    command_timeout=cst.COMMAND_TIMEOUT,
                    timeout=cst.TIMEOUT,
                    max_queries=cst.MAX_QUERIES,
                    init=_init_connection,
                )
                if self.pool is None:
                    raise ConnectionError("Failed to create connection pool")
                else:
                    logger.debug("Connection pool created successfully")
            except asyncpg.exceptions.PostgresError as e:
                raise ConnectionError(f"PostgreSQL error: {str(e)}") from e
            except socket.gaierror as e:
                raise ConnectionError(
                    f"DNS resolution failed: {str(e)}. Check your host name. connection_string: {self.connection_string}"
                ) from e
            except Exception as e:
                raise ConnectionError(f"Unexpected error while connecting: {str(e)}") from e

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def fetchall(self, query: str, *args: Any) -> list[dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("Database connection not established. Call connect() first.")
        async with self.pool.acquire() as connection:
            try:
                rows = await connection.fetch(query, *args)
                return [dict(row) for row in rows]

            except asyncpg.exceptions.PostgresError as e:
                logger.error(f"PostgreSQL error in fetch_all: {str(e)}")
                logger.error(f"Query: {query}")
                raise

    async def connection(self) -> PoolAcquireContext:
        if not self.pool:
            raise RuntimeError("Database connection not established. Call connect() first.")
        return self.pool.acquire(timeout=cst.TIMEOUT)
