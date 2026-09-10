#!/usr/bin/env python3
"""Backfill 2nd/3rd placements for environment tournaments that ended via boss retention.

Why this exists
---------------
When an environment round yields no winners (the boss beat every co-group challenger),
``advance_tournament`` used to set ``winner_hotkey = EMISSION_BURN_HOTKEY`` and return
without writing ``final_position`` for 2nd/3rd. Weight setting then paid the champion
100% of the environment pool.

The runtime fix now ranks non-boss challengers of the retention round and persists
placements via ``update_tournament_placements``. This tool re-runs that ranking for
already-completed tournaments that still have all ``final_position`` values NULL.

Usage
-----
    # Dry run (default) for the known affected tournaments.
    python ops/tools/tournament/backfill_boss_retention_placements.py

    # Dry run for specific tournament ids.
    python ops/tools/tournament/backfill_boss_retention_placements.py tourn_4f605297e6d85428_20260907

    # Actually write the recomputed placements to the DB.
    python ops/tools/tournament/backfill_boss_retention_placements.py --apply

Reads DATABASE_URL from .vali.env. Does not touch the chain or upload anything.
"""

import argparse
import asyncio

from dotenv import load_dotenv

from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_participants
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import update_tournament_placements
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import TournamentType
from validator.tournament.round_results import get_boss_retention_runners_up


logger = get_logger(__name__)

# Confirmed affected: boss retained with empty winners, no final_positions written.
DEFAULT_TOURNAMENT_IDS = (
    "tourn_4f605297e6d85428_20260907",
    "tourn_1e08b0b313ccc59f_20260817",
)


def _short(hotkey: str | None) -> str:
    if not hotkey:
        return "None"
    return f"{hotkey[:8]}..."


async def backfill_tournament(tournament_id: str, apply: bool, psql_db: PSQLDB) -> None:
    tournament = await get_tournament(tournament_id, psql_db)
    if not tournament:
        logger.error(f"[{tournament_id}] not found; skipping")
        return

    print(f"\n=== {tournament.tournament_type.value.upper()}  {tournament_id} ===")
    print(f"  winner_hotkey:      {_short(tournament.winner_hotkey)}")
    print(f"  base_winner_hotkey: {_short(tournament.base_winner_hotkey)}")

    if tournament.tournament_type != TournamentType.ENVIRONMENT:
        logger.warning(f"[{tournament_id}] not an environment tournament; skipping")
        return

    if tournament.winner_hotkey != EMISSION_BURN_HOTKEY:
        logger.warning(
            f"[{tournament_id}] winner_hotkey is not EMISSION_BURN_HOTKEY "
            f"({_short(tournament.winner_hotkey)}); skipping"
        )
        return

    participants = await get_tournament_participants(tournament_id, psql_db)
    placed = [p for p in participants if p.final_position is not None]
    if placed:
        by_position = {p.final_position: p.hotkey for p in placed}
        print(
            f"  -> placements already present "
            f"(1={_short(by_position.get(1))}, 2={_short(by_position.get(2))}, "
            f"3={_short(by_position.get(3))}); nothing to backfill"
        )
        return

    rounds = await get_tournament_rounds(tournament_id, psql_db)
    if not rounds:
        logger.warning(f"[{tournament_id}] no rounds found; skipping")
        return

    retention_round = max(rounds, key=lambda r: r.round_number)
    print(
        f"  retention round: {retention_round.round_id} "
        f"(round_number={retention_round.round_number}, is_final={retention_round.is_final_round})"
    )

    second_place, third_place = await get_boss_retention_runners_up(retention_round, psql_db)
    print(f"  -> resolved position 2: {_short(second_place)}")
    print(f"  -> resolved position 3: {_short(third_place)}")

    if second_place is None:
        print("  -> no clean 2nd place resolvable; leaving placements unchanged")
        return

    if not apply:
        print("  (dry run: pass --apply to write this to the DB)")
        return

    await update_tournament_placements(
        tournament_id,
        EMISSION_BURN_HOTKEY,
        second_place,
        third_place,
        psql_db,
    )
    print(
        f"  APPLIED: final_position written "
        f"(1={_short(EMISSION_BURN_HOTKEY)}, 2={_short(second_place)}, 3={_short(third_place)})"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill 2nd/3rd placements for environment boss-retention completions."
    )
    parser.add_argument(
        "tournament_ids",
        nargs="*",
        help=(
            "Specific tournament ids. Defaults to the known affected boss-retention "
            "tournaments (20260907 and 20260817)."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write recomputed placements to the DB (default is a dry run).",
    )
    args = parser.parse_args()

    load_dotenv(".vali.env")

    psql_db = PSQLDB()
    await psql_db.connect()
    try:
        tournament_ids = args.tournament_ids or list(DEFAULT_TOURNAMENT_IDS)

        if not args.apply:
            print("DRY RUN - no database writes. Re-run with --apply to persist.")

        for tournament_id in tournament_ids:
            await backfill_tournament(tournament_id, args.apply, psql_db)
    finally:
        await psql_db.close()


if __name__ == "__main__":
    asyncio.run(main())
