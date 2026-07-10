"""Resolve a boss-round challenger code review.

Usage:
    python -m ops.tools.tournament.code_review agree <tournament_id> <hotkey>
    python -m ops.tools.tournament.code_review skip  <tournament_id> <hotkey>
    python -m ops.tools.tournament.code_review show  <tournament_id>
"""

import argparse
import asyncio
import os
from pathlib import Path

import asyncpg
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[3]


def _database_url() -> str:
    load_dotenv(REPO_ROOT / ".vali.env", override=False)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit(f"DATABASE_URL was not found in {REPO_ROOT / '.vali.env'}")
    return database_url


async def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a boss-round challenger code review.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("agree", "skip"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("tournament_id")
        subparser.add_argument("hotkey")
    show = subparsers.add_parser("show")
    show.add_argument("tournament_id")
    args = parser.parse_args()

    connection = await asyncpg.connect(_database_url())
    try:
        if args.command == "show":
            row = await connection.fetchrow(
                "SELECT code_review FROM tournaments WHERE tournament_id = $1",
                args.tournament_id,
            )
            print(row["code_review"] if row else "tournament not found")
            return

        status = "accepted" if args.command == "agree" else "rejected"
        reviewable_statuses = ["pending"] if args.command == "agree" else ["pending", "error"]
        result = await connection.execute(
            """
            UPDATE tournaments
            SET code_review = $3, updated_at = now()
            WHERE tournament_id = $1 AND code_review = ANY($4::text[])
              AND EXISTS (
                  SELECT 1 FROM tournament_participants
                  WHERE tournament_id = $1 AND hotkey = $2
              )
            """,
            args.tournament_id,
            args.hotkey,
            status,
            reviewable_statuses,
        )
        if result != "UPDATE 1":
            raise SystemExit("No pending code review matched that tournament and hotkey.")
        print(f"Code review marked {status}. Tournament completion will resume on the next validator cycle.")
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
