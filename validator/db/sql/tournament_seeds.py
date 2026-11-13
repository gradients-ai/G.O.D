"""SQL for tournament_seed_tasks: the tasks the boss trains upfront (see validator/tournament/seed_tasks.py)."""

from uuid import UUID

from asyncpg.connection import Connection

import validator.db.constants as cst
from core.logging import get_logger
from core.models.task_models import TaskStatus
from validator.db.database import PSQLDB
from validator.db.sql import submissions_and_scoring as submissions_and_scoring_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournaments as tournament_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import ANY_BOSS_COMPRESSION_SEED_ROLES
from validator.tournament.models import ANY_BOSS_ROUND_SEED_ROLES
from validator.tournament.models import ATTACHED_SEED_ROLES
from validator.tournament.models import SWAPPED_SEED_ROLES
from validator.tournament.models import TEXT_SEED_ROLES
from validator.tournament.models import RoundStatus
from validator.tournament.models import SeedRole
from validator.tournament.models import SeedStatus
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TournamentType
from validator.tournament.models import TrainingStatus


logger = get_logger(__name__)

_TERMINAL_TRAINING_STATUSES = frozenset({TrainingStatus.SUCCESS.value, TrainingStatus.FAILURE.value})

_SEED_COLUMNS = (
    f"{cst.TOURNAMENT_ID}, {cst.SEED_ROLE}, {cst.SEED_ATTEMPT}, {cst.TASK_ID}, {cst.SEED_STATUS}, "
    f"{cst.SEED_BOSS_REPO}, {cst.SEED_BOSS_COMMIT}, {cst.SEED_ATTACHED_ROUND_ID}, {cst.SEED_PREP_REJECTED}, "
    f"{cst.SEED_MERGED_OUTPUT_REPO}, {cst.CREATED_AT}, {cst.UPDATED_AT}"
)
# The live attempts are never replaced or failed: a role's current row is its highest attempt.
_LATEST_PER_ROLE = f"DISTINCT ON ({cst.SEED_ROLE})"


def _seed_from_row(row) -> TournamentSeedTask:
    return TournamentSeedTask(**dict(row))


async def insert_seed_task(seed: TournamentSeedTask, psql_db: PSQLDB) -> TournamentSeedTask:
    """Insert one seed attempt; an existing (tournament, role, attempt) row is returned unchanged."""
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            f"""
            INSERT INTO {cst.TOURNAMENT_SEED_TASKS_TABLE}
                ({cst.TOURNAMENT_ID}, {cst.SEED_ROLE}, {cst.SEED_ATTEMPT}, {cst.TASK_ID}, {cst.SEED_STATUS},
                 {cst.SEED_BOSS_REPO}, {cst.SEED_BOSS_COMMIT}, {cst.SEED_PREP_REJECTED})
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT ({cst.TOURNAMENT_ID}, {cst.SEED_ROLE}, {cst.SEED_ATTEMPT}) DO NOTHING
            """,
            seed.tournament_id,
            seed.role.value,
            seed.attempt,
            seed.task_id,
            seed.status.value,
            seed.boss_repo,
            seed.boss_commit,
            seed.prep_rejected,
        )
        row = await connection.fetchrow(
            f"""
            SELECT {_SEED_COLUMNS} FROM {cst.TOURNAMENT_SEED_TASKS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.SEED_ROLE} = $2 AND {cst.SEED_ATTEMPT} = $3
            """,
            seed.tournament_id,
            seed.role.value,
            seed.attempt,
        )
    return _seed_from_row(row)


async def get_latest_seed_tasks(tournament_id: str, psql_db: PSQLDB) -> list[TournamentSeedTask]:
    """The latest attempt of every role of a tournament, whatever its status."""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT {_LATEST_PER_ROLE} {_SEED_COLUMNS}
            FROM {cst.TOURNAMENT_SEED_TASKS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
            ORDER BY {cst.SEED_ROLE}, {cst.SEED_ATTEMPT} DESC
            """,
            tournament_id,
        )
    return [_seed_from_row(row) for row in rows]


async def get_seed_task_history(tournament_id: str, psql_db: PSQLDB) -> list[TournamentSeedTask]:
    """Every attempt of every role of a tournament (the replacement counters and the rejected datasets read it)."""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT {_SEED_COLUMNS} FROM {cst.TOURNAMENT_SEED_TASKS_TABLE}
            WHERE {cst.TOURNAMENT_ID} = $1
            ORDER BY {cst.SEED_ROLE}, {cst.SEED_ATTEMPT}
            """,
            tournament_id,
        )
    return [_seed_from_row(row) for row in rows]


async def get_seed_task_by_task_id(task_id: UUID | str, psql_db: PSQLDB) -> TournamentSeedTask | None:
    """The seed row (role and tournament) a task belongs to, or None when it is not a seed."""
    async with await psql_db.connection() as connection:
        connection: Connection
        row = await connection.fetchrow(
            f"SELECT {_SEED_COLUMNS} FROM {cst.TOURNAMENT_SEED_TASKS_TABLE} WHERE {cst.TASK_ID} = $1",
            UUID(str(task_id)),
        )
    return _seed_from_row(row) if row else None


async def get_seed_roles_by_task_ids(task_ids: list[str], psql_db: PSQLDB) -> dict[str, SeedRole]:
    """Role of every given task that is a seed (one batched query); non-seed tasks are absent from the result."""
    if not task_ids:
        return {}
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT {cst.TASK_ID}, {cst.SEED_ROLE}
            FROM {cst.TOURNAMENT_SEED_TASKS_TABLE}
            WHERE {cst.TASK_ID} = ANY($1::uuid[])
            """,
            [UUID(str(task_id)) for task_id in task_ids],
        )
    return {str(row[cst.TASK_ID]): SeedRole(row[cst.SEED_ROLE]) for row in rows}


async def update_seed_status(tournament_id: str, role: SeedRole, attempt: int, status: SeedStatus, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            f"""
            UPDATE {cst.TOURNAMENT_SEED_TASKS_TABLE}
            SET {cst.SEED_STATUS} = $4, {cst.UPDATED_AT} = NOW()
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.SEED_ROLE} = $2 AND {cst.SEED_ATTEMPT} = $3
            """,
            tournament_id,
            role.value,
            attempt,
            status.value,
        )
    logger.info(f"Seed {tournament_id}/{role.value} attempt {attempt} -> {status.value}")


async def set_seed_merged_output_repo(task_id: UUID | str, repo: str, psql_db: PSQLDB) -> None:
    """Record the full model the boss's adapter output on a seed was merged into (the seed's output from now on)."""
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            f"""
            UPDATE {cst.TOURNAMENT_SEED_TASKS_TABLE}
            SET {cst.SEED_MERGED_OUTPUT_REPO} = $2, {cst.UPDATED_AT} = NOW()
            WHERE {cst.TASK_ID} = $1
            """,
            UUID(str(task_id)),
            repo,
        )
    logger.info(f"Seed task {task_id}: boss output merged into {repo}")


async def mark_seed_replaced(tournament_id: str, role: SeedRole, attempt: int, prep_rejected: bool, psql_db: PSQLDB) -> None:
    """Mark a row replaced, recording whether the replacement is uncharged (a prep rejection; see TournamentSeedTask)."""
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            f"""
            UPDATE {cst.TOURNAMENT_SEED_TASKS_TABLE}
            SET {cst.SEED_STATUS} = $4, {cst.SEED_PREP_REJECTED} = $5, {cst.UPDATED_AT} = NOW()
            WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.SEED_ROLE} = $2 AND {cst.SEED_ATTEMPT} = $3
            """,
            tournament_id,
            role.value,
            attempt,
            SeedStatus.REPLACED.value,
            prep_rejected,
        )
    logger.info(f"Seed {tournament_id}/{role.value} attempt {attempt} -> replaced (prep_rejected={prep_rejected})")


async def mark_seed_attached(task_id: UUID | str, round_id: str, psql_db: PSQLDB) -> None:
    """Record, durably and once, that the boss round attached this seed task (idempotent: the first round id stays)."""
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            f"""
            UPDATE {cst.TOURNAMENT_SEED_TASKS_TABLE}
            SET {cst.SEED_ATTACHED_ROUND_ID} = $2, {cst.UPDATED_AT} = NOW()
            WHERE {cst.TASK_ID} = $1 AND {cst.SEED_ATTACHED_ROUND_ID} IS NULL
            """,
            UUID(str(task_id)),
            round_id,
        )


async def role_ever_attached(tournament_id: str, role: SeedRole, psql_db: PSQLDB) -> bool:
    """Whether any attempt of the role was attached to the boss round: the durable marker, or a tournament_tasks row."""
    async with await psql_db.connection() as connection:
        connection: Connection
        found = await connection.fetchval(
            f"""
            SELECT 1 FROM {cst.TOURNAMENT_SEED_TASKS_TABLE} s
            WHERE s.{cst.TOURNAMENT_ID} = $1 AND s.{cst.SEED_ROLE} = $2
              AND (s.{cst.SEED_ATTACHED_ROUND_ID} IS NOT NULL
                   OR EXISTS (SELECT 1 FROM {cst.TOURNAMENT_TASKS_TABLE} t WHERE t.{cst.TASK_ID} = s.{cst.TASK_ID}))
            LIMIT 1
            """,
            tournament_id,
            role.value,
        )
    return found is not None


async def boss_side_seed_failure(task_id: UUID | str, task_status: TaskStatus | str, psql_db: PSQLDB) -> SeedRole | None:
    """The role when a boss-round seed task is dead on the BOSS's side only, else None.

    Dead on the boss's side: the task failed prep after the boss round attached it (PREP_TASK_FAILURE) while the boss's
    own training row is not SUCCESS (a boss that trained successfully was not the cause), or it ended FAILURE with the
    boss's own training row FAILURE. In both cases the challenger's side must be intact: a non-burn hotkey with a
    SUCCESS training row and no persisted quant_failure. The boss's entries are trained upfront and never retrained, so
    such a slot is not replaced: the round counts it as the boss's failure (the challenger takes the task). A slot
    where nobody produced a usable result is never credited to the challenger (that would crown it on no evidence);
    None is returned and the ordinary rule holds: no valid results, the boss keeps the task by default.
    """
    status = TaskStatus(task_status)
    if status not in (TaskStatus.PREP_TASK_FAILURE, TaskStatus.FAILURE):
        return None
    seed = await get_seed_task_by_task_id(task_id, psql_db)
    if seed is None or seed.role not in ANY_BOSS_ROUND_SEED_ROLES:
        return None
    async with await psql_db.connection() as connection:
        connection: Connection
        training_rows = await connection.fetch(
            f"""
            SELECT {cst.HOTKEY}, {cst.TRAINING_STATUS} FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            WHERE {cst.TASK_ID} = $1
            """,
            str(task_id),
        )
        quant_failure_rows = await connection.fetch(
            f"SELECT {cst.HOTKEY} FROM {cst.TASK_NODES_TABLE} WHERE {cst.TASK_ID} = $1 AND {cst.QUANT_FAILURE} IS NOT NULL",
            UUID(str(task_id)),
        )
    trained = {row[cst.HOTKEY]: row[cst.TRAINING_STATUS] for row in training_rows}
    quant_failed = {row[cst.HOTKEY] for row in quant_failure_rows}
    if status == TaskStatus.PREP_TASK_FAILURE:
        if trained.get(EMISSION_BURN_HOTKEY) == TrainingStatus.SUCCESS.value:
            return None
    elif trained.get(EMISSION_BURN_HOTKEY) != TrainingStatus.FAILURE.value:
        return None
    challenger_usable = any(
        hotkey != EMISSION_BURN_HOTKEY and training == TrainingStatus.SUCCESS.value and hotkey not in quant_failed
        for hotkey, training in trained.items()
    )
    return seed.role if challenger_usable else None


async def boss_compression_entry_failed(task_id: UUID | str, psql_db: PSQLDB) -> bool:
    """Whether a task is one of the boss's compression entries (freeform or fixed) whose boss run failed."""
    seed = await get_seed_task_by_task_id(task_id, psql_db)
    if seed is None or seed.role not in ANY_BOSS_COMPRESSION_SEED_ROLES:
        return False
    async with await psql_db.connection() as connection:
        connection: Connection
        training_status = await connection.fetchval(
            f"""
            SELECT {cst.TRAINING_STATUS} FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
            """,
            str(task_id),
            EMISSION_BURN_HOTKEY,
        )
    return training_status == TrainingStatus.FAILURE.value


async def get_active_seed_tasks(tournament_id: str, psql_db: PSQLDB) -> list[TournamentSeedTask]:
    """Rows still being worked on: training, trained or ready (not replaced, not failed)."""
    active = {SeedStatus.TRAINING, SeedStatus.TRAINED, SeedStatus.READY}
    return [seed for seed in await get_latest_seed_tasks(tournament_id, psql_db) if seed.status in active]


async def get_failed_seed_tasks(tournament_id: str, psql_db: PSQLDB) -> list[TournamentSeedTask]:
    """Roles whose latest attempt is `failed`: out of replacements."""
    return [seed for seed in await get_latest_seed_tasks(tournament_id, psql_db) if seed.status == SeedStatus.FAILED]


async def is_task_attached_to_round(task_id: UUID | str, psql_db: PSQLDB) -> bool:
    """Whether the task has a tournament_tasks row (a boss-round seed gets one when the boss round attaches it)."""
    async with await psql_db.connection() as connection:
        connection: Connection
        found = await connection.fetchval(
            f"SELECT 1 FROM {cst.TOURNAMENT_TASKS_TABLE} WHERE {cst.TASK_ID} = $1 LIMIT 1",
            str(task_id),
        )
    return found is not None


# The round whose completion releases an input seed: input_small feeds R1 and R2, input_large feeds R3. A boss-round seed
# is released only with its tournament.
_INPUT_SEED_RELEASING_ROUND = {SeedRole.INPUT_SMALL: 2, SeedRole.INPUT_LARGE: 3}


async def is_seeded_text_tournament(tournament_id: str, psql_db: PSQLDB) -> bool:
    """Whether a tournament is a text tournament with seed rows. A text tournament without seed rows is unseeded
    and runs the plain text flow."""
    tournament = await tournament_sql.get_tournament(tournament_id, psql_db)
    if tournament is None or tournament.tournament_type != TournamentType.TEXT:
        return False
    return bool(await get_seed_task_history(tournament_id, psql_db))


def _text_seed_hidden_condition() -> str:
    """The hidden-or-not condition for the text seed roles (replaces the generic tournament release and role arms).

    A quant task chained on a text seed is trained and scored on the text task's own data and test split, so what hides the
    text task is the quant tournament that uses it, not only the text round that attaches it:
      * the text seed task is hidden while no round has attached it (unless its text tournament is over), and while any
        live quant_task_sources row names it as its source and that quant tournament is not completed or cancelled;
      * a quant task chained on it (via_quant_task) is hidden until its own quant tournament (owner status: its source row,
        else its own seed row, else its tournament_tasks row; with none of them, the text tournament) is completed or
        cancelled. The text round's completion releases nothing.
    """
    over = f"('{TournamentStatus.COMPLETED.value}', '{TournamentStatus.CANCELLED.value}')"
    roles = ", ".join(f"'{role.value}'" for role in sorted(TEXT_SEED_ROLES, key=lambda role: role.value))
    return (
        f"h.{cst.SEED_ROLE} IN ({roles}) THEN "
        f"CASE WHEN h.via_quant_task THEN COALESCE(o.{cst.TOURNAMENT_STATUS}, t.{cst.TOURNAMENT_STATUS}) NOT IN {over} "
        f"ELSE (h.{cst.SEED_ATTACHED_ROUND_ID} IS NULL AND t.{cst.TOURNAMENT_STATUS} NOT IN {over}) OR EXISTS ("
        f"SELECT 1 FROM {cst.QUANT_TASK_SOURCES_TABLE} qs "
        f"JOIN {cst.TOURNAMENTS_TABLE} qt_t ON qt_t.{cst.TOURNAMENT_ID} = qs.{cst.QUANT_TOURNAMENT_ID} "
        f"WHERE qs.{cst.QUANT_SOURCE_TASK_ID} = h.{cst.TASK_ID} AND NOT qs.{cst.QUANT_SOURCE_SUPERSEDED} "
        f"AND qt_t.{cst.TOURNAMENT_STATUS} NOT IN {over}) END"
    )


async def get_hidden_seed_task_ids(task_ids: list[str], psql_db: PSQLDB) -> set[str]:
    """The given tasks whose data must stay hidden on public and auditing endpoints (one batched query).

    A seed task, and a miner round's task chained on it (it carries the seed's split), stays hidden until the last round
    using the seed completes: input_small until round 2 of its tournament, input_large until round 3, a boss-round seed
    (the boss round's input and its two compressions) until the tournament. A completed or cancelled tournament releases
    all of its seeds, except the text seed roles, which follow _text_seed_hidden_condition: a text seed is hidden until a
    round attaches it and while a live quant task source of a running quant tournament names it; a quant task chained on it
    is hidden until its own quant tournament is completed or cancelled. Other tasks are never returned.
    """
    if not task_ids:
        return set()
    releasing_round_cases = " ".join(
        f"WHEN '{role.value}' THEN NOT EXISTS ("
        f"SELECT 1 FROM {cst.TOURNAMENT_ROUNDS_TABLE} r WHERE r.{cst.TOURNAMENT_ID} = h.{cst.TOURNAMENT_ID} "
        f"AND r.{cst.ROUND_NUMBER} = {round_number} AND r.{cst.ROUND_STATUS} = '{RoundStatus.COMPLETED.value}')"
        for role, round_number in _INPUT_SEED_RELEASING_ROUND.items()
    )
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            WITH requested AS (SELECT unnest($1::uuid[]) AS {cst.TASK_ID}),
            seeded AS (
                SELECT q.{cst.TASK_ID}, s.{cst.SEED_ROLE}, s.{cst.TOURNAMENT_ID}, s.{cst.SEED_ATTACHED_ROUND_ID},
                       FALSE AS via_quant_task, NULL::text AS owner_tournament_id
                FROM requested q JOIN {cst.TOURNAMENT_SEED_TASKS_TABLE} s ON s.{cst.TASK_ID} = q.{cst.TASK_ID}
                UNION
                SELECT q.{cst.TASK_ID}, s.{cst.SEED_ROLE}, s.{cst.TOURNAMENT_ID}, s.{cst.SEED_ATTACHED_ROUND_ID},
                       TRUE AS via_quant_task,
                       COALESCE(qs.{cst.QUANT_TOURNAMENT_ID}, own.{cst.TOURNAMENT_ID}, tt.{cst.TOURNAMENT_ID})
                           AS owner_tournament_id
                FROM requested q
                JOIN {cst.QUANT_TASKS_TABLE} qt ON qt.{cst.TASK_ID} = q.{cst.TASK_ID}
                JOIN {cst.TOURNAMENT_SEED_TASKS_TABLE} s ON s.{cst.TASK_ID} = qt.{cst.UPSTREAM_TASK_ID}
                LEFT JOIN {cst.QUANT_TASK_SOURCES_TABLE} qs ON qs.{cst.QUANT_TASK_ID} = q.{cst.TASK_ID}
                LEFT JOIN {cst.TOURNAMENT_SEED_TASKS_TABLE} own ON own.{cst.TASK_ID} = q.{cst.TASK_ID}
                LEFT JOIN {cst.TOURNAMENT_TASKS_TABLE} tt ON tt.{cst.TASK_ID} = q.{cst.TASK_ID}
            )
            SELECT DISTINCT h.{cst.TASK_ID}
            FROM seeded h
            JOIN {cst.TOURNAMENTS_TABLE} t ON t.{cst.TOURNAMENT_ID} = h.{cst.TOURNAMENT_ID}
            LEFT JOIN {cst.TOURNAMENTS_TABLE} o ON o.{cst.TOURNAMENT_ID} = h.owner_tournament_id
            WHERE CASE WHEN {_text_seed_hidden_condition()}
                ELSE t.{cst.TOURNAMENT_STATUS} NOT IN ('{TournamentStatus.COMPLETED.value}', '{TournamentStatus.CANCELLED.value}')
                  AND CASE h.{cst.SEED_ROLE} {releasing_round_cases} ELSE TRUE END
            END
            """,
            [UUID(str(task_id)) for task_id in task_ids],
        )
    return {str(row[cst.TASK_ID]) for row in rows}


async def is_unattached_boss_round_seed(task_id: UUID | str, psql_db: PSQLDB) -> bool:
    """Whether a task is a seed (a boss-round compression or a text seed) that no round has attached yet.

    Its only player is the boss, so an evaluation of it now is the seed-time evaluation of the boss's entry.
    """
    seed = await get_seed_task_by_task_id(task_id, psql_db)
    if seed is None or seed.role not in ATTACHED_SEED_ROLES:
        return False
    return not await is_task_attached_to_round(task_id, psql_db)


async def tournament_is_decided(tournament_id: str, psql_db: PSQLDB) -> bool:
    """Whether the tournament will play no more rounds: it has a winner, or it is neither pending nor active."""
    tournament = await tournament_sql.get_tournament(tournament_id, psql_db)
    if tournament is None:
        return True
    return bool(tournament.winner_hotkey) or tournament.status not in (TournamentStatus.PENDING, TournamentStatus.ACTIVE)


async def boss_round_seed_ready_to_finalise(task_id: UUID | str, psql_db: PSQLDB) -> bool:
    """Whether a task may be finalised (leave TRAINING for PREEVALUATION, or be ranked and closed from TRAINING).

    Always true for a non-seed task, an input seed and the boss round's input. A seed attached to a round later (a text
    seed, a quant round seed or a boss-round compression seed) is held until a round has attached it, at least one non-boss
    player is assigned, and every assigned player has a terminal training row or a persisted quant_failure. Once its
    tournament is decided (no longer pending or active, or a winner is set) a seed no round attached is released, and so is
    any text or quant round seed: it closes with the boss's stored result instead of waiting in TRAINING for good. The hold is
    on finalisation only: each player's entry is evaluated as soon as its own training row succeeds, so the boss's entry is
    scored at seed time and stored, and the task is ranked once, from the stored rows of both players. Every path that
    finalises a TRAINING tournament task, or moves it to PREEVALUATION, applies this.
    """
    seed = await get_seed_task_by_task_id(task_id, psql_db)
    if seed is None or seed.role not in ATTACHED_SEED_ROLES:
        return True
    attached = await is_task_attached_to_round(task_id, psql_db)
    if (seed.role in SWAPPED_SEED_ROLES or not attached) and await tournament_is_decided(seed.tournament_id, psql_db):
        return True
    if not attached:
        return False
    hotkeys = [node.hotkey for node in await task_sql.get_nodes_assigned_to_task(str(task_id), psql_db)]
    if not any(hotkey != EMISSION_BURN_HOTKEY for hotkey in hotkeys):
        return False
    statuses = await tournament_sql.get_training_status_for_task(str(task_id), psql_db)
    failed = {
        row[cst.HOTKEY]
        for row in await submissions_and_scoring_sql.get_task_node_losses(UUID(str(task_id)), psql_db)
        if row[cst.QUANT_FAILURE]
    }
    return all(statuses.get(hotkey) in _TERMINAL_TRAINING_STATUSES or hotkey in failed for hotkey in hotkeys)
