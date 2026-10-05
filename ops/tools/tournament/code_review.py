"""Resolve a round-2 challenger code review.

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
    parser = argparse.ArgumentParser(description="Resolve a round-2 challenger code review.")
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
            rows = await connection.fetch(
                """
                SELECT hotkey, code_review
                FROM tournament_participants
                WHERE tournament_id = $1 AND code_review IS NOT NULL
                ORDER BY hotkey
                """,
                args.tournament_id,
            )
            if not rows:
                print("no round-2 code reviews")
                return
            for row in rows:
                print(f"{row['hotkey']}\t{row['code_review']}")
            return

        status = "accepted" if args.command == "agree" else "rejected"
        reviewable_statuses = ["pending"] if args.command == "agree" else ["pending", "error"]
        result = await connection.execute(
            """
            UPDATE tournament_participants
            SET code_review = $3
            WHERE tournament_id = $1 AND hotkey = $2 AND code_review = ANY($4::text[])
            """,
            args.tournament_id,
            args.hotkey,
            status,
            reviewable_statuses,
        )
        if result != "UPDATE 1":
            raise SystemExit("No pending code review matched that tournament and hotkey.")
        print(f"Code review marked {status}. Round 2 will resume on the next validator cycle.")
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())
