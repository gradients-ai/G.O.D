"""Ops CLI for the freeform exposure publications (decoder releases and boss-round decoded models).

Read-only by default. Publishing is outward-facing (public HF repos), so every write is behind an explicit flag:

  (no flags)                                   print what the reconcile would publish; changes nothing
  --apply                                      run the reconcile for real and wait for it to finish
  --release --tournament-id T --hotkey H       one hand release of a decoder snapshot (the winner guard applies)
        [--allow-boss]                         with the burn hotkey: the reigning champion's pinned decoder
  --publish-decoded --tournament-id T         one boss-round decoded-model publication
  --json                                       machine-readable output

Needs the validator environment (.vali.env: DATABASE_URL, HF and MinIO credentials, DISCORD_URL).
"""

import argparse
import asyncio
import json
import sys

from validator.app.config import Config
from validator.app.config import load_config
from validator.tournament.freeform_release import publish_boss_round_decoded_models
from validator.tournament.freeform_release import reconcile_freeform_publications
from validator.tournament.freeform_release import release_freeform_decoder
from validator.tournament.freeform_release import wait_for_background_publications
from validator.tournament.models import FreeformReconcileReport
from validator.tournament.models import PendingDecoderRelease


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="run the reconcile for real (default: report only)")
    action.add_argument("--release", action="store_true", help="release one decoder snapshot by hand")
    action.add_argument("--publish-decoded", action="store_true", help="publish one tournament's boss-round decoded models")
    parser.add_argument("--tournament-id")
    parser.add_argument("--hotkey")
    parser.add_argument("--allow-boss", action="store_true", help="with --release: allow the burn hotkey (reigning champion)")
    parser.add_argument("--json", action="store_true", help="print JSON")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.release and not (args.tournament_id and args.hotkey):
        parser.error("--release needs --tournament-id and --hotkey")
    if args.publish_decoded and not args.tournament_id:
        parser.error("--publish-decoded needs --tournament-id")
    if args.allow_boss and not args.release:
        parser.error("--allow-boss only applies to --release")
    return args


def _items(title: str, items: list[PendingDecoderRelease]) -> list[str]:
    return [f"{title}: {len(items)}", *(f"  {item.tournament_id}  {item.hotkey}" for item in items)]


def _format_report(report: FreeformReconcileReport) -> str:
    lines = [f"Freeform publications ({'applied' if report.applied else 'read-only, nothing published'})"]
    lines += _items("Decoder releases pending", report.pending_releases)
    lines += _items("Champions blocked (no backup_repo, code not pushed)", report.blocked_releases)
    lines += [f"Tournaments with unpublished decoded models: {len(report.pending_decoded_tournaments)}"]
    lines += [f"  {tournament_id}" for tournament_id in report.pending_decoded_tournaments]
    if report.applied:
        lines += _items("Decoders released", report.released)
        lines += _items("Decoder releases failed or refused", report.failed_releases)
        lines += [f"Decoded models published: {len(report.published_models)}", *(f"  {r}" for r in report.published_models)]
    return "\n".join(lines)


def _emit(as_json: bool, payload: dict | FreeformReconcileReport, text: str) -> None:
    if not as_json:
        print(text)
    elif isinstance(payload, FreeformReconcileReport):
        print(payload.model_dump_json(indent=2))
    else:
        print(json.dumps(payload, indent=2))


async def run(args: argparse.Namespace, config: Config) -> int:
    psql_db = config.psql_db
    if args.release:
        url = await release_freeform_decoder(args.tournament_id, args.hotkey, config, psql_db, allow_boss=args.allow_boss)
        _emit(args.json, {"released_url": url}, f"released: {url}" if url else "refused or failed (see the validator log)")
        return 0 if url else 1
    if args.publish_decoded:
        repos = await publish_boss_round_decoded_models(args.tournament_id, config, psql_db)
        _emit(args.json, {"published_models": repos}, "\n".join([f"Decoded models published: {len(repos)}", *repos]))
        return 0
    report = await reconcile_freeform_publications(config, psql_db, apply=args.apply)
    if args.apply:
        await wait_for_background_publications()
    _emit(args.json, report, _format_report(report))
    return 0


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config()
    await config.psql_db.connect()
    try:
        return await run(args, config)
    finally:
        await config.psql_db.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
