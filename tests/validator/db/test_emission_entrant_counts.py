"""The emission balancer's entrant history: the last N completed tournaments of a type, the latest included.

Runs the real SQL of get_recent_completed_entrant_counts against a throwaway local Postgres.
"""

import random
import shutil
import subprocess
import tempfile
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import asyncpg
import pytest

from core.models.tournament_models import TournamentType
from validator.db.sql.tournaments import get_recent_completed_entrant_counts


class _Cluster:
    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="pg_entrant_counts_"))
        self.port = 55000 + random.randint(0, 900)
        subprocess.run(["initdb", "-D", str(self.dir / "data"), "-A", "trust", "-U", "postgres"], check=True, capture_output=True)
        subprocess.run(
            [
                "pg_ctl", "-D", str(self.dir / "data"), "-w", "start", "-l", str(self.dir / "log"),
                "-o", f"-c listen_addresses='' -k {self.dir} -p {self.port}",
            ],  # fmt: skip
            check=True,
            capture_output=True,
        )

    async def connect(self) -> asyncpg.Connection:
        return await asyncpg.connect(host=str(self.dir), port=self.port, user="postgres", database="postgres")

    def stop(self) -> None:
        subprocess.run(["pg_ctl", "-D", str(self.dir / "data"), "-m", "immediate", "stop"], capture_output=True)
        shutil.rmtree(self.dir, ignore_errors=True)


@pytest.fixture(scope="module")
def cluster():
    if not all(shutil.which(binary) for binary in ("initdb", "pg_ctl")):
        pytest.skip("no local postgres binaries")
    cluster = _Cluster()
    yield cluster
    cluster.stop()


class _Db:
    def __init__(self, connection: asyncpg.Connection) -> None:
        self.connection_obj = connection

    async def connection(self):
        outer = self

        class _Ctx:
            async def __aenter__(self):
                return outer.connection_obj

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


_SCHEMA = """
DROP TABLE IF EXISTS tournaments, tournament_participants;
CREATE TABLE tournaments (tournament_id text PRIMARY KEY, tournament_type text, status text, created_at timestamptz);
CREATE TABLE tournament_participants (tournament_id text, hotkey text);
"""

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


async def _seed(connection: asyncpg.Connection, tournaments: list[tuple[str, str, int]], tournament_type: str = "text") -> None:
    """Insert tournaments oldest first as (status, id, entrants); each is one day newer than the previous."""
    for day, (status, tournament_id, entrants) in enumerate(tournaments):
        await connection.execute(
            "INSERT INTO tournaments VALUES ($1, $2, $3, $4)", tournament_id, tournament_type, status, _T0 + timedelta(days=day)
        )
        for i in range(entrants):
            await connection.execute("INSERT INTO tournament_participants VALUES ($1, $2)", tournament_id, f"hk{i}")


@pytest.fixture
async def connection(cluster):
    connection = await cluster.connect()
    await connection.execute(_SCHEMA)
    yield connection
    await connection.close()


async def _counts(connection, tournament_type: TournamentType = TournamentType.TEXT) -> list[int]:
    return await get_recent_completed_entrant_counts(_Db(connection), tournament_type, limit=4, min_entrants=3)


async def test_the_latest_completed_tournament_is_counted(connection):
    await _seed(connection, [("completed", "a", 5), ("completed", "b", 7)])

    assert await _counts(connection) == [7, 5]


async def test_a_first_completed_tournament_counts_on_its_own(connection):
    await _seed(connection, [("completed", "q1", 6)], tournament_type="quant")

    assert await _counts(connection, TournamentType.QUANT) == [6]
    assert await _counts(connection, TournamentType.TEXT) == []


async def test_pending_active_and_cancelled_tournaments_are_excluded_by_status(connection):
    await _seed(
        connection,
        [("completed", "a", 5), ("cancelled", "b", 9), ("completed", "c", 6), ("active", "d", 8), ("pending", "e", 4)],
    )

    assert await _counts(connection) == [6, 5]


async def test_the_window_is_the_last_four_completed(connection):
    await _seed(connection, [("completed", f"t{i}", 3 + i) for i in range(6)] + [("active", "live", 20)])

    assert await _counts(connection) == [8, 7, 6, 5]


async def test_tournaments_below_the_entrant_minimum_are_skipped_and_older_ones_fill_the_window(connection):
    await _seed(
        connection,
        [("completed", tournament_id, entrants) for tournament_id, entrants in zip("abcdef", [3, 4, 2, 5, 0, 6])],
    )

    assert await _counts(connection) == [6, 5, 4, 3]
