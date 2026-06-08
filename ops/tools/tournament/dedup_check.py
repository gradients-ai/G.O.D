"""Standalone de-dup self-check for a single pair of repos.

Runs the SAME pipeline the tournament gate uses (T0 identical-commit, T1 identical
normalized source, T2 Claude pairwise judgement) on exactly two repositories and prints
the verdict. Nothing is written to a DB and nothing is uploaded — pure local self-service
so a miner can check whether their submission would be flagged as a duplicate of another.

The Claude step needs ANTHROPIC_API_KEY in the environment. Run it from a checkout of the
G.O.D repo: the T2 agent is given that source so it can verify whether any claimed
differentiators are real training-time inputs rather than dead code.

Usage:
    export ANTHROPIC_API_KEY=...
    python -m ops.tools.tournament.dedup_check <repo_a_url> <repo_b_url> \
        --hash-a <commit_a> --hash-b <commit_b> \
        [--token-a <gh_token>] [--token-b <gh_token>] [--out report.md]
"""

import argparse
import asyncio

from validator.infrastructure.repo_dedup import run_pairwise_dedup
from validator.tournament.models import RepoRef


async def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone T0/T1/T2 de-dup check for two repos.")
    parser.add_argument("repo_a", help="Repository A clone URL")
    parser.add_argument("repo_b", help="Repository B clone URL")
    parser.add_argument("--hash-a", help="commit hash to check out for repo A (default: HEAD)")
    parser.add_argument("--hash-b", help="commit hash to check out for repo B (default: HEAD)")
    parser.add_argument("--token-a", help="GitHub token for repo A (if private)")
    parser.add_argument("--token-b", help="GitHub token for repo B (if private)")
    parser.add_argument("--out", help="also write the verdict to this path")
    args = parser.parse_args()

    repos = [
        RepoRef(hotkey="A", repo_url=args.repo_a, commit_hash=args.hash_a, github_token=args.token_a),
        RepoRef(hotkey="B", repo_url=args.repo_b, commit_hash=args.hash_b, github_token=args.token_b),
    ]

    print(f"Repository A: {args.repo_a} @ {args.hash_a or 'HEAD'}")
    print(f"Repository B: {args.repo_b} @ {args.hash_b or 'HEAD'}")
    print("Running T0/T1/T2 de-dup check (this calls Claude and may take a minute)...\n")

    # No boss in a standalone check — we just want the raw pairwise verdict.
    result = await run_pairwise_dedup(repos, boss_hotkey=None)

    unclonable = set(result.unclonable_hotkeys)
    if unclonable:
        which = ", ".join("Repository A" if h == "A" else "Repository B" for h in sorted(unclonable))
        print(f"ERROR: could not clone {which}. Check the URL / hash / token.")
        return

    if not result.pair_verdicts:
        print("ERROR: no verdict produced (both repos resolved to the same checkout?).")
        return

    verdict = result.pair_verdicts[0]
    lines = [
        f"VERDICT: {verdict.relationship.value.upper()}",
        f"Tier:        {verdict.tier.value}  (T0=identical commit, T1=identical normalized source, T2=Claude)",
        f"Confidence:  {verdict.confidence:.2f}",
        "",
        "Reason:",
        verdict.reason,
    ]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(report)
        print(f"Verdict written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
