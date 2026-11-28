"""The emission balancer's tournament history, its round-1 completer fallback and the champion's defence count.

Runs the real SQL of get_recent_completed_tournament_ids, count_round_one_completers and count_champion_defences on a
throwaway local Postgres.
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

import validator.scoring.constants as scoring_cst
from core.models.tournament_models import TournamentType
from validator.db.sql.tournaments import count_champion_defences
from validator.db.sql.tournaments import count_round_one_completers
from validator.db.sql.tournaments import get_recent_completed_tournament_ids


class _Cluster:
    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="pg_contender_counts_"))
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
DROP TABLE IF EXISTS tournaments, tournament_participants, tournament_rounds, tournament_tasks,
    tournament_task_hotkey_trainings, task_nodes;
CREATE TABLE tournaments (
    tournament_id text PRIMARY KEY, tournament_type text, status text, created_at timestamptz,
    winner_hotkey text, base_winner_hotkey text
);
CREATE TABLE tournament_participants (tournament_id text, hotkey text);
CREATE TABLE tournament_rounds (round_id text PRIMARY KEY, tournament_id text, round_number int);
CREATE TABLE tournament_tasks (tournament_id text, round_id text, task_id text);
CREATE TABLE tournament_task_hotkey_trainings (task_id text, hotkey text, training_status text);
CREATE TABLE task_nodes (task_id text, hotkey text, test_loss double precision);
"""

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
BURN = scoring_cst.EMISSION_BURN_HOTKEY


async def _tournament(
    connection: asyncpg.Connection,
    tournament_id: str,
    day: int,
    status: str = "completed",
    tournament_type: str = "text",
    winner: str | None = None,
    base_winner: str | None = None,
) -> None:
    await connection.execute(
        "INSERT INTO tournaments VALUES ($1, $2, $3, $4, $5, $6)",
        tournament_id, tournament_type, status, _T0 + timedelta(days=day), winner, base_winner,
    )  # fmt: skip


async def _round(connection: asyncpg.Connection, tournament_id: str, round_number: int, tasks: dict[str, list]) -> None:
    """One round with its tasks; each task lists (hotkey, training_status, test_loss) rows."""
    round_id = f"{tournament_id}_r{round_number}"
    await connection.execute("INSERT INTO tournament_rounds VALUES ($1, $2, $3)", round_id, tournament_id, round_number)
    for task_id, rows in tasks.items():
        await connection.execute("INSERT INTO tournament_tasks VALUES ($1, $2, $3)", tournament_id, round_id, task_id)
        for hotkey, status, loss in rows:
            await connection.execute("INSERT INTO tournament_task_hotkey_trainings VALUES ($1, $2, $3)", task_id, hotkey, status)
            if loss is not None:
                await connection.execute("INSERT INTO task_nodes VALUES ($1, $2, $3)", task_id, hotkey, loss)


@pytest.fixture
async def connection(cluster):
    connection = await cluster.connect()
    await connection.execute(_SCHEMA)
    yield connection
    await connection.close()


async def _ids(connection, tournament_type: TournamentType = TournamentType.TEXT, limit: int = 12) -> list[str]:
    return await get_recent_completed_tournament_ids(_Db(connection), tournament_type, limit=limit)


async def _completers(connection, tournament_id: str) -> int:
    return await count_round_one_completers(_Db(connection), tournament_id)


async def test_only_round_one_completers_count(connection):
    await _tournament(connection, "t", 0)
    for i in range(9):
        await connection.execute("INSERT INTO tournament_participants VALUES ('t', $1)", f"signup{i}")
    await _round(
        connection,
        "t",
        1,
        {
            "a": [("ok1", "success", 1.2), ("crashed", "failure", None), ("no_eval", "success", None)],
            "b": [("ok2", "success", 0.9), ("nan", "success", float("nan")), ("inf", "success", float("inf"))],
            "c": [("ok3", "success", 1.1), ("ok1", "success", 1.0)],
        },
    )
    await _round(connection, "t", 2, {"d": [("r2_only", "success", 1.0), ("r2_also", "success", 1.0)]})

    assert await _completers(connection, "t") == 3


async def test_a_miner_that_fails_one_task_but_completes_another_counts(connection):
    await _tournament(connection, "t", 0)
    await _round(
        connection,
        "t",
        1,
        {"a": [("x", "failure", None), ("y", "success", 1.0)], "b": [("x", "success", 1.0), ("z", "success", 1.0)]},
    )

    assert await _completers(connection, "t") == 3


async def test_a_tournament_without_round_one_has_no_completers(connection):
    await _tournament(connection, "t", 0)

    assert await _completers(connection, "t") == 0


async def test_ids_are_completed_tournaments_of_the_type_newest_first(connection):
    await _tournament(connection, "a", 0)
    await _tournament(connection, "b", 1, status="cancelled")
    await _tournament(connection, "c", 2)
    await _tournament(connection, "d", 3, status="active")
    await _tournament(connection, "e", 4, status="pending")
    await _tournament(connection, "q", 5, tournament_type="quant")

    assert await _ids(connection) == ["c", "a"]
    assert await _ids(connection, TournamentType.QUANT) == ["q"]


async def test_ids_are_limited(connection):
    for i in range(6):
        await _tournament(connection, f"t{i}", i)

    assert await _ids(connection, limit=4) == ["t5", "t4", "t3", "t2"]


async def _defences(connection, tournament_id: str) -> int:
    return await count_champion_defences(_Db(connection), tournament_id)


async def test_the_crown_tournament_is_zero_defences(connection):
    await _tournament(connection, "old", 0, winner="boss_a", base_winner="x")
    await _tournament(connection, "crown", 1, winner="boss_b", base_winner="boss_a")

    assert await _defences(connection, "crown") == 0


async def test_each_kept_crown_is_one_more_defence(connection):
    await _tournament(connection, "crown", 0, winner="boss", base_winner="prev")
    await _tournament(connection, "d1", 1, winner=BURN, base_winner="boss")
    await _tournament(connection, "d2", 2, winner=BURN, base_winner="boss")
    await _tournament(connection, "d3", 3, winner="boss", base_winner="boss")

    assert await _defences(connection, "d1") == 1
    assert await _defences(connection, "d3") == 3


async def test_cancelled_tournaments_and_other_types_do_not_break_the_reign(connection):
    await _tournament(connection, "crown", 0, winner="boss", base_winner="prev")
    await _tournament(connection, "cancelled", 1, status="cancelled", base_winner="boss")
    await _tournament(connection, "other_type", 2, tournament_type="image", winner="someone_else")
    await _tournament(connection, "d1", 3, winner=BURN, base_winner="boss")

    assert await _defences(connection, "d1") == 1


async def test_a_reign_counts_from_its_own_crown_not_an_earlier_one(connection):
    await _tournament(connection, "first_reign", 0, winner="boss", base_winner="prev")
    await _tournament(connection, "first_defence", 1, winner=BURN, base_winner="boss")
    await _tournament(connection, "dethroned", 2, winner="usurper", base_winner="boss")
    await _tournament(connection, "back", 3, winner="boss", base_winner="usurper")
    await _tournament(connection, "again", 4, winner=BURN, base_winner="boss")

    assert await _defences(connection, "back") == 0
    assert await _defences(connection, "again") == 1


async def test_later_tournaments_do_not_count_toward_an_earlier_one(connection):
    await _tournament(connection, "crown", 0, winner="boss", base_winner="prev")
    await _tournament(connection, "d1", 1, winner=BURN, base_winner="boss")
    await _tournament(connection, "d2", 2, winner=BURN, base_winner="boss")

    assert await _defences(connection, "d1") == 1


async def test_an_unknown_champion_or_tournament_is_zero(connection):
    await _tournament(connection, "no_champion", 0, winner=BURN, base_winner=None)

    assert await _defences(connection, "no_champion") == 0
    assert await _defences(connection, "missing") == 0
