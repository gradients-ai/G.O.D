"""Read-only dry run of tournament submission de-duplication against real DB data.

Loads a tournament's participants (repo + commit + token) and runs the dedup pipeline
WITHOUT writing anything or eliminating anyone. Prints the markdown report.

Usage (on the validator, with .vali.env sourced for DATABASE_URL + ANTHROPIC_API_KEY):
    python -m ops.tools.tournament.dedup_dryrun <tournament_id>            # T0+T1 only (no Claude)
    python -m ops.tools.tournament.dedup_dryrun <tournament_id> --full     # + T2 Claude pairwise
"""

import argparse
import asyncio
import os

import asyncpg

from validator.infrastructure.repo_dedup import find_hash_duplicates
from validator.infrastructure.repo_dedup import render_report
from validator.infrastructure.repo_dedup import run_pairwise_dedup
from validator.scoring.constants import EMISSION_BURN_HOTKEY
from validator.tournament.models import RepoRef


async def _load_repos(tournament_id: str, hotkeys: list[str] | None = None) -> list[RepoRef]:
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        if hotkeys:
            rows = await conn.fetch(
                "SELECT hotkey, training_repo, training_commit_hash, github_token "
                "FROM tournament_participants WHERE tournament_id = $1 AND hotkey = ANY($2::text[]) ORDER BY hotkey",
                tournament_id,
                hotkeys,
            )
        else:
            rows = await conn.fetch(
                "SELECT hotkey, training_repo, training_commit_hash, github_token "
                "FROM tournament_participants WHERE tournament_id = $1 ORDER BY hotkey",
                tournament_id,
            )
    finally:
        await conn.close()
    return [
        RepoRef(
            hotkey=r["hotkey"],
            repo_url=r["training_repo"],
            commit_hash=r["training_commit_hash"],
            github_token=r["github_token"] or None,
        )
        for r in rows
        if r["training_repo"]
    ]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tournament_id")
    parser.add_argument("--full", action="store_true", help="run T2 Claude pairwise (otherwise T0+T1 only)")
    parser.add_argument("--hotkeys", help="comma-separated hotkeys to restrict to (e.g. the R2 cohort)")
    parser.add_argument("--out", help="also write the report to this path")
    args = parser.parse_args()

    hotkeys = [h.strip() for h in args.hotkeys.split(",") if h.strip()] if args.hotkeys else None
    repos = await _load_repos(args.tournament_id, hotkeys)
    print(f"Loaded {len(repos)} participant repos for {args.tournament_id}")
    print(f"Boss (protected) hotkey: {EMISSION_BURN_HOTKEY}")
    print(f"Mode: {'T0+T1+T2 (Claude)' if args.full else 'T0+T1 (deterministic only)'}\n")

    if args.full:
        result = await run_pairwise_dedup(repos, boss_hotkey=EMISSION_BURN_HOTKEY)
    else:
        result = await find_hash_duplicates(repos, boss_hotkey=EMISSION_BURN_HOTKEY)

    report = render_report(result, args.tournament_id, "DRYRUN", EMISSION_BURN_HOTKEY, include_distinct_verdicts=True)
    print(report)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(report)
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
