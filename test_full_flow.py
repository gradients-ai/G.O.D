"""
Full e2e test: Create a pending environment tournament, populate participants
(which calls the miner and captures requested_datasets), then let the
orchestrator schedule training on the trainer.
"""

import asyncio
import json
import sys

from validator.core.config import load_config
from validator.db.sql import tournaments as tournament_sql
from validator.tournament.tournament_manager import (
    create_basic_tournament,
    populate_tournament_participants,
    create_first_round_for_active_tournament,
)
from validator.tournament.task_creator import create_environment_tournament_tasks
from core.models.tournament_models import TournamentType, TournamentStatus


async def main():
    config = load_config()
    await config.psql_db.connect()

    # 1. Create a pending environment tournament
    print("=== Step 1: Creating pending environment tournament ===")
    tournament_id = await create_basic_tournament(
        TournamentType.ENVIRONMENT, config.psql_db, config
    )
    print(f"Created tournament: {tournament_id}")

    # 2. Populate participants (this calls each miner's /training_repo/environment endpoint)
    print("\n=== Step 2: Populating participants (calling miners) ===")
    num_miners = await populate_tournament_participants(
        tournament_id, config, config.psql_db
    )
    print(f"Populated {num_miners} participant(s)")

    if num_miners == 0:
        print("ERROR: No miners responded. Check miner is running and registered.")
        sys.exit(1)

    # 3. Verify requested_datasets were stored in DB
    print("\n=== Step 3: Checking requested_datasets in DB ===")
    participants = await tournament_sql.get_tournament_participants(
        tournament_id, config.psql_db
    )
    for p in participants:
        ds = p.requested_datasets
        print(f"  Hotkey: {p.hotkey[:16]}... | Datasets: {ds}")
        if ds:
            print(f"  >>> requested_datasets stored correctly!")

    # 4. Activate tournament and create first round
    print("\n=== Step 4: Activating tournament ===")
    tournament = await tournament_sql.get_tournament(tournament_id, config.psql_db)
    if tournament and tournament.status == TournamentStatus.PENDING:
        await tournament_sql.update_tournament_status(
            tournament_id, TournamentStatus.ACTIVE, config.psql_db
        )
        print(f"Tournament {tournament_id} activated")

    print("\n=== Step 5: Creating first round ===")
    created = await create_first_round_for_active_tournament(
        tournament_id, config, config.psql_db
    )
    print(f"Round created: {created}")

    # 5. Check that training assignments have requested_datasets
    print("\n=== Step 6: Checking training assignments ===")
    from core.models.utility_models import TrainingStatus
    pending = await tournament_sql.get_tournament_training_tasks(
        config.psql_db, TrainingStatus.PENDING
    )
    for task in pending:
        print(f"  Task: {task.task.task_id} | Hotkey: {task.hotkey[:16]}... | "
              f"Repo: {task.training_repo} | Datasets: {task.requested_datasets}")

    print("\n=== Done! Orchestrator can now schedule these tasks. ===")
    print(f"Tournament ID: {tournament_id}")
    print("Start orchestrator: python -u -m validator.tournament.orchestrator")


if __name__ == "__main__":
    asyncio.run(main())
