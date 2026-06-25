"""Database bootstrap for the monitor.

Loads DB credentials from `.vali.env` at the repo root (same convention as the
other scripts in this repo) and hands back a connected `PSQLDB`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[3]


class MissingDatabaseConfig(RuntimeError):
    pass


def _has_db_config() -> bool:
    if os.getenv("DATABASE_URL"):
        return True
    required = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB")
    return all(os.getenv(key) for key in required)


def load_environment() -> None:
    """Load DB env vars from `.vali.env` / `.env` at the repo root if present.

    `PSQLDB` reads `DATABASE_URL` (or the `POSTGRES_*` vars) from the environment,
    so this just makes sure they are populated before we connect.
    """
    for candidate in (".vali.env", ".env"):
        env_path = REPO_ROOT / candidate
        if env_path.exists():
            load_dotenv(env_path, override=False)


async def connect():
    """Return a connected `PSQLDB`. Caller is responsible for `close()`."""
    load_environment()
    if not _has_db_config():
        raise MissingDatabaseConfig(
            "No database credentials found. Run from the G.O.D root with a `.vali.env` "
            "(or `.env`) containing DATABASE_URL=postgresql://... (or the POSTGRES_* vars)."
        )
    # Imported lazily so `sys.path` is set up before the heavy validator imports run.
    from validator.db.database import PSQLDB

    psql_db = PSQLDB()
    await psql_db.connect()
    return psql_db
