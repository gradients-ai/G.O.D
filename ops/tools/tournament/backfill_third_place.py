#!/usr/bin/env python3
"""Backfill 3rd-place (final_position = 3) for already-completed tournaments.

Why this exists
---------------
Until the fix in #1364, ``get_pre_boss_knockout_loser`` read the always-NULL
``tournament_pairs.winner_hotkey`` column, so its first guard tripped on every
real tournament and finalization silently fell back to a top-2-only payout. Any
tournament that finalized before that fix has its ``final_position`` frozen in the
DB with no 3rd place, and the weight/audit path only *reads* those stored
positions - so re-deploying the fix does not change emissions for those cohorts.

This tool re-runs the (now fixed) 3rd-place resolution against the boss-round data
and rewrites placements via ``update_tournament_placements``, preserving the
existing winner (position 1) and runner-up (position 2). It is a no-op for any
tournament whose 3rd place cannot be cleanly resolved.

Usage
-----
    # Dry run (default): show what would change for the latest completed
    # text/image/environment tournaments, write nothing.
    python ops/tools/tournament/backfill_third_place.py

    # Dry run for specific tournament ids.
    python ops/tools/tournament/backfill_third_place.py tourn_add1dc83b8fd58b0_20260831

    # Actually write the recomputed 3rd place to the DB.
    python ops/tools/tournament/backfill_third_place.py --apply

Reads DATABASE_URL from .vali.env. Does not touch the chain or upload anything.
"""

import argparse
import asyncio

from dotenv import load_dotenv

from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.db.sql.tournaments import get_latest_completed_tournament
from validator.db.sql.tournaments import get_tournament
from validator.db.sql.tournaments import get_tournament_participants
from validator.db.sql.tournaments import get_tournament_rounds
from validator.db.sql.tournaments import update_tournament_placements
from validator.tournament.models import TournamentData
from validator.tournament.models import TournamentRoundData
from validator.tournament.models import TournamentType
from validator.tournament.participants import get_pre_boss_knockout_loser
from validator.tournament.round_results import get_pre_boss_group_runner_up


logger = get_logger(__name__)


def _short(hotkey: str | None) -> str:
    if not hotkey:
        return "None"
    return f"{hotkey[:8]}..."


def _final_round(rounds: list[TournamentRoundData]) -> TournamentRoundData | None:
    final = next((r for r in rounds if r.is_final_round), None)
    if final is not None:
        return final
    return max(rounds, key=lambda r: r.round_number) if rounds else None


async def _resolve_third_place(
    tournament: TournamentData,
    final_round: TournamentRoundData,
    challenger_hotkey: str,
    psql_db: PSQLDB,
) -> str | None:
    if tournament.tournament_type in (TournamentType.TEXT, TournamentType.IMAGE):
        return await get_pre_boss_knockout_loser(tournament, final_round, challenger_hotkey, psql_db)
    if tournament.tournament_type == TournamentType.ENVIRONMENT:
        return await get_pre_boss_group_runner_up(
            tournament.tournament_id, final_round, challenger_hotkey, psql_db
        )
    return None


async def backfill_tournament(tournament_id: str, apply: bool, psql_db: PSQLDB) -> None:
    tournament = await get_tournament(tournament_id, psql_db)
    if not tournament:
        logger.error(f"[{tournament_id}] not found; skipping")
        return

    print(f"\n=== {tournament.tournament_type.value.upper()}  {tournament_id} ===")

    participants = await get_tournament_participants(tournament_id, psql_db)
    by_position = {p.final_position: p.hotkey for p in participants if p.final_position is not None}
    winner = by_position.get(1)
    second_place = by_position.get(2)
    existing_third = by_position.get(3)

    print(f"  current position 1 (winner):     {_short(winner)}")
    print(f"  current position 2 (runner-up):  {_short(second_place)}")
    print(f"  current position 3 (3rd place):  {_short(existing_third)}")

    if winner is None or second_place is None:
        logger.warning(
            f"[{tournament_id}] missing position 1 and/or 2; refusing to rewrite placements"
        )
        return

    if existing_third is not None:
        print("  -> 3rd place already persisted; nothing to backfill")
        return

    rounds = await get_tournament_rounds(tournament_id, psql_db)
    final_round = _final_round(rounds)
    if final_round is None:
        logger.warning(f"[{tournament_id}] no rounds found; skipping")
        return

    # The boss-round challenger (the miner that fought through to the final) is the
    # persisted runner-up. The 3rd-place resolvers exclude this hotkey by design.
    third_place = await _resolve_third_place(tournament, final_round, second_place, psql_db)

    if third_place is not None and third_place in {winner, second_place}:
        logger.warning(
            f"[{tournament_id}] resolved 3rd place {_short(third_place)} collides with "
            f"winner/runner-up; omitting"
        )
        third_place = None

    if third_place is None:
        print("  -> no clean 3rd place resolvable; leaving placements unchanged")
        return

    print(f"  -> resolved NEW position 3:      {_short(third_place)}  ({third_place})")

    if not apply:
        print("  (dry run: pass --apply to write this to the DB)")
        return

    await update_tournament_placements(
        tournament_id,
        winner,
        second_place,
        third_place,
        psql_db,
    )
    print("  APPLIED: final_position=3 written")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill tournament 3rd place placements.")
    parser.add_argument(
        "tournament_ids",
        nargs="*",
        help="Specific tournament ids. Defaults to the latest completed text/image/environment tournaments.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write recomputed 3rd place to the DB (default is a dry run).",
    )
    args = parser.parse_args()

    load_dotenv(".vali.env")

    psql_db = PSQLDB()
    await psql_db.connect()
    try:
        tournament_ids = args.tournament_ids
        if not tournament_ids:
            tournament_ids = []
            for tournament_type in (TournamentType.TEXT, TournamentType.IMAGE, TournamentType.ENVIRONMENT):
                latest = await get_latest_completed_tournament(psql_db, tournament_type)
                if latest:
                    tournament_ids.append(latest.tournament_id)
                else:
                    logger.warning(f"No completed {tournament_type.value} tournament found")

        if not args.apply:
            print("DRY RUN - no database writes. Re-run with --apply to persist.")

        for tournament_id in tournament_ids:
            await backfill_tournament(tournament_id, args.apply, psql_db)
    finally:
        await psql_db.close()


if __name__ == "__main__":
    asyncio.run(main())
