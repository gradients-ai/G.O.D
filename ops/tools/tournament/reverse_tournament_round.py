#!/usr/bin/env python3
"""
Script to reverse a tournament round by deleting it and setting the previous round back to active.

Usage:
    python reverse_tournament_round.py <tournament_id> <round_id_to_delete>

This script will:
1. Delete the specified round and all its associated data
2. Set the previous round to 'completed' status
3. Un-eliminate participants who were eliminated in the deleted round
4. Set tournament status back to 'active' if it was completed
5. Allow the tournament cycle to re-run the advancement logic

Seed tasks. In a seeded tournament the task of a round is a seed task the boss trained upfront (a row in
tournament_seed_tasks: text rounds 1-3, the continuous task of the text boss round and the quant boss-round
compressions). Its result is the boss's and is not recomputed, so a seed task is never deleted:
  * the round's tournament_tasks row is removed, and so are the round miners' training rows, task_nodes
    assignments (with their evaluations) and submissions on it, plus the round's failure-review gate;
  * the task row, the boss's task_nodes row, the boss's training row and every score of the boss are kept;
  * the seed's attached marker is cleared so the re-created round attaches it again, for a text seed only, and only
    while it is safe: a quant boss-round compression keeps its marker (it guards against a late re-draw of boss
    training), and so does a seed whose task or boss run failed (an unattached seed would be replaced);
  * the task status is NOT reset. A seed whose round finished is SUCCESS (or failed) and a seed waiting for its round
    is TRAINING; the tool logs a warning when the status is past TRAINING so the operator resets it deliberately.
Every other task of the round is deleted with all its rows.

The tool REFUSES, changing nothing and exiting non-zero, when a live (not superseded) quant_task_sources row names a task
of the round as its source: a bound quant tournament depends on that text seed task. Delete the dependent quant
tasks first (cancelling the quant tournament does not lift the refusal, and the rows outlive a completed one: they go
only when the quant tasks are deleted, ON DELETE CASCADE on quant_task_id, or superseded), or leave the round as it is.

Example:
    python reverse_tournament_round.py tourn_f4eb788ad3d66f60_20250723 tourn_f4eb788ad3d66f60_20250723_round_002
"""

import asyncio
import os
import sys
from pathlib import Path
from uuid import UUID

from asyncpg.connection import Connection
from pydantic import BaseModel

import validator.db.constants as cst
from core.constants.network import NETUID
from core.logging import get_logger
from core.models.task_models import TaskStatus
from validator.db.database import PSQLDB
from validator.db.sql import quant_sources as quant_sources_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql import tournament_seeds as seeds_sql
from validator.db.sql import tournaments as tournament_sql
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import TEXT_SEED_ROLES
from validator.tournament.models import RoundStatus
from validator.tournament.models import TournamentSeedTask
from validator.tournament.models import TournamentStatus
from validator.tournament.models import TrainingStatus


logger = get_logger(__name__)


def load_database_url_from_env_file() -> str:
    env_file = Path(".vali.env")
    if not env_file.exists():
        raise FileNotFoundError(".vali.env file not found")

    database_url = None
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                database_url = line.split("=", 1)[1].strip("\"'")
                break

    if not database_url:
        raise ValueError("DATABASE_URL not found in .vali.env file")

    return database_url


# Task statuses a seed task has moved to only because its round (or the boss's finished run) went past the hold in TRAINING.
_PAST_TRAINING_STATUSES = frozenset(
    {
        TaskStatus.PREEVALUATION,
        TaskStatus.EVALUATING,
        TaskStatus.SUCCESS,
        TaskStatus.FAILURE,
        TaskStatus.PREP_TASK_FAILURE,
    }
)
_FAILED_TASK_STATUSES = frozenset({TaskStatus.FAILURE, TaskStatus.PREP_TASK_FAILURE})


class SeedTaskReversal(BaseModel):
    """What reversing a round does to one seed task of it, decided from reads before the first write."""

    seed: TournamentSeedTask
    task_status: TaskStatus | None
    boss_training_status: str | None
    clear_marker: bool
    marker_note: str


def _marker_decision(
    seed: TournamentSeedTask, round_id: str, task_status: TaskStatus | None, boss_status: str | None
) -> tuple[bool, str]:
    """Whether the seed's attached marker can be cleared, and why or why not.

    The marker blocks replacing the seed; a seed whose marker is cleared is replaced on any failure by the seed loop. That is
    the right state for a text seed that has simply not been attached yet, and wrong in three cases that keep the marker."""
    if seed.attached_round_id is None:
        return False, "was not set"
    if seed.attached_round_id != round_id:
        return False, f"kept: it names another round ({seed.attached_round_id})"
    if seed.role not in TEXT_SEED_ROLES:
        return False, "kept: a boss-round compression is never replaced once attached (a re-draw would train the boss late)"
    if task_status in _FAILED_TASK_STATUSES or boss_status == TrainingStatus.FAILURE.value:
        return False, "kept: the task or the boss's run failed, and an unattached seed would be replaced, erasing that entry"
    return True, "cleared: the re-created round attaches the seed again"


async def _plan_seed_reversal(seed: TournamentSeedTask, round_id: str, psql_db: PSQLDB) -> SeedTaskReversal:
    task = await task_sql.get_task(seed.task_id, psql_db)
    task_status = task.status if task is not None else None
    statuses = await tournament_sql.get_training_status_for_task_and_hotkeys(str(seed.task_id), [EMISSION_BURN_HOTKEY], psql_db)
    boss_status = statuses.get(EMISSION_BURN_HOTKEY)
    clear_marker, marker_note = _marker_decision(seed, round_id, task_status, boss_status)
    return SeedTaskReversal(
        seed=seed, task_status=task_status, boss_training_status=boss_status, clear_marker=clear_marker, marker_note=marker_note
    )


async def _find_dependent_quant_sources(round_id: str, task_ids: list[UUID], psql_db: PSQLDB) -> list[str]:
    """One message line per live quant task that names a task of the round as its source (empty: nothing depends on them)."""
    lines = []
    for task_id in task_ids:
        for source in await quant_sources_sql.list_live_quant_sources_naming_task(task_id, psql_db):
            lines.append(
                f"  task {task_id}: quant tournament {source.quant_tournament_id}, quant task {source.quant_task_id} "
                f"(slot {source.slot.value})"
            )
    return lines


async def _detach_seed_task(connection: Connection, round_id: str, plan: SeedTaskReversal) -> None:
    """Take a seed task out of the round, keeping the task and everything of the boss's, and log what was kept and removed."""
    task_id = plan.seed.task_id
    removed = {}
    removed["tournament_tasks row of this round"] = await connection.fetch(
        f"DELETE FROM {cst.TOURNAMENT_TASKS_TABLE} WHERE {cst.TASK_ID} = $1 AND {cst.ROUND_ID} = $2 RETURNING {cst.TASK_ID}",
        task_id,
        round_id,
    )
    removed["miner training rows"] = await connection.fetch(
        f"""
        DELETE FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
        WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} <> $2 RETURNING {cst.HOTKEY}
        """,
        task_id,
        EMISSION_BURN_HOTKEY,
    )
    removed["miner submissions"] = await connection.fetch(
        f"""
        DELETE FROM {cst.SUBMISSIONS_TABLE}
        WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2 AND {cst.HOTKEY} <> $3 RETURNING {cst.HOTKEY}
        """,
        task_id,
        NETUID,
        EMISSION_BURN_HOTKEY,
    )
    # The evaluation rows of a removed assignment go with it (ON DELETE CASCADE).
    removed["miner task_nodes assignments (and their evaluations)"] = await connection.fetch(
        f"""
        DELETE FROM {cst.TASK_NODES_TABLE}
        WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2 AND {cst.HOTKEY} <> $3 RETURNING {cst.HOTKEY}
        """,
        task_id,
        NETUID,
        EMISSION_BURN_HOTKEY,
    )
    removed["failure-review gate"] = await connection.fetch(
        f"DELETE FROM {cst.TOURNAMENT_TASK_FAILURE_REVIEWS_TABLE} WHERE {cst.TASK_ID} = $1 RETURNING {cst.TASK_ID}", task_id
    )
    if plan.clear_marker:
        removed["attached marker"] = await connection.fetch(
            f"""
            UPDATE {cst.TOURNAMENT_SEED_TASKS_TABLE}
            SET {cst.SEED_ATTACHED_ROUND_ID} = NULL, {cst.UPDATED_AT} = NOW()
            WHERE {cst.TASK_ID} = $1 AND {cst.SEED_ATTACHED_ROUND_ID} = $2 RETURNING {cst.TASK_ID}
            """,
            task_id,
            round_id,
        )

    removed_text = "; ".join(
        f"{what}: {', '.join(str(row[0]) for row in rows) if rows else 'none'}" for what, rows in removed.items()
    )
    logger.info(
        f"Seed task {task_id} ({plan.seed.role.value}) kept, not deleted: the task row, the boss's task_nodes row and scores, "
        f"the boss's training row (status {plan.boss_training_status}). Removed from round {round_id}: {removed_text}. "
        f"Attached marker {plan.marker_note}."
    )
    if plan.task_status in _PAST_TRAINING_STATUSES:
        logger.warning(
            f"WARNING seed task {task_id} ({plan.seed.role.value}) has status {plan.task_status.value}, which it only reached "
            f"through the round (a seed waits in TRAINING for its round). The tool did NOT reset it: ranking and closing a "
            f"seed is not safely repeatable. A re-created round will find the task already past training, so reset the "
            f"status by hand if the round's miners must train it again."
        )


async def reverse_tournament_round(tournament_id: str, round_id_to_delete: str, psql_db: PSQLDB) -> bool:
    """
    Reverse a tournament round by deleting it and resetting the previous round.

    The round's seed tasks are detached, not deleted (see the module docstring); its other tasks are deleted. Refuses, with no
    write at all, while a live quant_task_sources row names a task of the round as its source.

    Args:
        tournament_id: The ID of the tournament
        round_id_to_delete: The round ID to delete
        psql_db: Database connection

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get tournament
        tournament = await tournament_sql.get_tournament(tournament_id, psql_db)
        if not tournament:
            logger.error(f"Tournament {tournament_id} not found")
            return False

        logger.info(f"Found tournament: {tournament_id} (status: {tournament.status})")

        # Get all rounds
        rounds = await tournament_sql.get_tournament_rounds(tournament_id, psql_db)
        logger.info(f"Found {len(rounds)} rounds for tournament {tournament_id}")

        # Find the round to delete
        round_to_delete = None
        previous_round = None
        for i, round_data in enumerate(rounds):
            if round_data.round_id == round_id_to_delete:
                round_to_delete = round_data
                if i > 0:
                    previous_round = rounds[i - 1]
                break

        if not round_to_delete:
            logger.error(f"Round {round_id_to_delete} not found in tournament {tournament_id}")
            return False

        logger.info(f"Deleting round: {round_id_to_delete} (round number: {round_to_delete.round_number})")

        # Everything is read, and the refusal decided, before the transaction's first write.
        round_tasks = await tournament_sql.get_tournament_tasks(round_id_to_delete, psql_db)
        logger.info(f"Found {len(round_tasks)} tasks in the round")
        dependents = await _find_dependent_quant_sources(round_id_to_delete, [task.task_id for task in round_tasks], psql_db)
        if dependents:
            logger.error(
                f"REFUSING to reverse round {round_id_to_delete} of tournament {tournament_id}; nothing was changed. A bound "
                f"quant tournament depends on the round's text seed task(s):\n" + "\n".join(dependents) + "\n"
                "Reversing the round would drop the boss's trained seed that these quant tasks compress. Delete the "
                "dependent quant tasks first (then run this tool again): cancelling the quant tournament does not lift this, "
                "and the rows outlive a completed one; they go only when those tasks are deleted or superseded. "
                "Or leave the round as it is."
            )
            return False
        seed_plans: list[SeedTaskReversal] = []
        plain_task_ids: list[UUID] = []
        for task in round_tasks:
            seed = await seeds_sql.get_seed_task_by_task_id(task.task_id, psql_db)
            if seed is None:
                plain_task_ids.append(task.task_id)
            else:
                seed_plans.append(await _plan_seed_reversal(seed, round_id_to_delete, psql_db))

        # Start transaction: every write goes through this connection.
        async with await psql_db.connection() as connection:
            connection: Connection
            async with connection.transaction():
                # 1. Seed tasks are detached from the round; the other tasks are deleted.
                for plan in seed_plans:
                    await _detach_seed_task(connection, round_id_to_delete, plan)

                # Deleting the task cascades to tournament_task_hotkey_trainings and tournament_tasks (ON DELETE CASCADE).
                for task_id in plain_task_ids:
                    query = f"""
                        DELETE FROM {cst.TASKS_TABLE}
                        WHERE {cst.TASK_ID} = $1
                    """
                    await connection.execute(query, task_id)
                    logger.info(f"Deleted task {task_id} from tasks table (cascaded to related tables)")

                # 2. Delete pairs if it's a knockout round
                query = """
                    DELETE FROM tournament_pairs
                    WHERE round_id = $1
                """
                await connection.execute(query, round_id_to_delete)

                # 3. Delete group members and groups if it's a group round
                query = """
                    DELETE FROM tournament_group_members
                    WHERE group_id IN (
                        SELECT group_id FROM tournament_groups WHERE round_id = $1
                    )
                """
                await connection.execute(query, round_id_to_delete)

                query = """
                    DELETE FROM tournament_groups
                    WHERE round_id = $1
                """
                await connection.execute(query, round_id_to_delete)

                # 4. Un-eliminate participants who were eliminated in this round
                query = """
                    UPDATE tournament_participants
                    SET eliminated_in_round_id = NULL
                    WHERE tournament_id = $1 AND eliminated_in_round_id = $2
                """
                await connection.execute(query, tournament_id, round_id_to_delete)

                # 5. Delete the round
                query = """
                    DELETE FROM tournament_rounds
                    WHERE round_id = $1
                """
                await connection.execute(query, round_id_to_delete)

                # 6. Set previous round to active (so cycle can check completion and advance)
                if previous_round:
                    await connection.execute(
                        f"UPDATE {cst.TOURNAMENT_ROUNDS_TABLE} SET {cst.ROUND_STATUS} = $2 WHERE {cst.ROUND_ID} = $1",
                        previous_round.round_id,
                        RoundStatus.ACTIVE.value,
                    )
                    logger.info(f"Set previous round {previous_round.round_id} to active status")

                # 7. Set tournament back to active if it was completed, and clear its winner
                if tournament.status == TournamentStatus.COMPLETED:
                    await connection.execute(
                        f"""
                        UPDATE {cst.TOURNAMENTS_TABLE}
                        SET {cst.TOURNAMENT_STATUS} = $2, {cst.WINNER_HOTKEY} = NULL, {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                        WHERE {cst.TOURNAMENT_ID} = $1
                        """,
                        tournament_id,
                        TournamentStatus.ACTIVE.value,
                    )

        logger.info(f"Successfully reversed round {round_id_to_delete}")
        return True

    except Exception as e:
        logger.error(f"Error reversing tournament round: {str(e)}")
        return False


async def main():
    """Main function to handle command line arguments and execute reversal."""
    if len(sys.argv) != 3:
        print("Usage: python reverse_tournament_round.py <tournament_id> <round_id_to_delete>")
        print(
            "Example: python reverse_tournament_round.py tourn_f4eb788ad3d66f60_20250723 "
            "tourn_f4eb788ad3d66f60_20250723_round_002"
        )
        sys.exit(1)

    tournament_id = sys.argv[1]
    round_id_to_delete = sys.argv[2]

    try:
        # Load database URL from .vali.env
        database_url = load_database_url_from_env_file()
        print("Using database URL from .vali.env")

        # Set environment variable for PSQLDB
        os.environ["DATABASE_URL"] = database_url

        # Initialize database connection
        psql_db = PSQLDB()

        # Establish database connection
        await psql_db.connect()

        success = await reverse_tournament_round(tournament_id, round_id_to_delete, psql_db)
        if success:
            print(f"✅ Successfully reversed round {round_id_to_delete} in tournament {tournament_id}")
            print("🔄 The tournament cycle will now re-run the advancement logic")
            sys.exit(0)
        else:
            print("❌ Failed to reverse tournament round")
            sys.exit(1)
    except FileNotFoundError:
        print("❌ .vali.env file not found")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        if "psql_db" in locals():
            await psql_db.close()


if __name__ == "__main__":
    asyncio.run(main())
