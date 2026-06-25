"""Argparse entrypoint. With no subcommand it drops into the interactive menu."""

import argparse
import asyncio

from ops.tools.god_monitor import menu
from ops.tools.god_monitor import views as v
from ops.tools.god_monitor.db import MissingDatabaseConfig
from ops.tools.god_monitor.db import connect
from ops.tools.god_monitor.queries import Queries


def _selected_types(args) -> list[str] | None:
    selected = [name for name in ("text", "image", "env") if getattr(args, name, False)]
    if getattr(args, "all", False) or not selected:
        return None  # None == all types
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="god",
        description="G.O.D Tournament Monitor — read-only live monitoring CLI.",
    )
    sub = parser.add_subparsers(dest="command")

    summary = sub.add_parser("summary", help="Full report for active tournaments.")
    summary.add_argument("--text", action="store_true")
    summary.add_argument("--image", action="store_true")
    summary.add_argument("--env", action="store_true")
    summary.add_argument("--all", action="store_true")
    summary.add_argument("--pvp", action="store_true", help="Include PvP tables for env tournaments.")

    task = sub.add_parser("task", help="Jump straight to a task's detail menu.")
    task.add_argument("task_id")

    tourn = sub.add_parser("tournament", help="Open a tournament (id | current | last).")
    tourn.add_argument("selector", help="tournament_id, 'current', or 'last'")
    tourn.add_argument("--text", action="store_true")
    tourn.add_argument("--image", action="store_true")
    tourn.add_argument("--env", action="store_true")

    tlist = sub.add_parser("tournaments", help="List tournaments.")
    tlist.add_argument("--active", action="store_true")
    tlist.add_argument("--completed", action="store_true")

    sub.add_parser("deployments", help="Show active eval + PvP deployments.")
    sub.add_parser("trainers", help="Show trainers / GPU capacity.")
    return parser


def _type_alias_from(args) -> str:
    if getattr(args, "text", False):
        return "text"
    if getattr(args, "image", False):
        return "image"
    if getattr(args, "env", False):
        return "env"
    return "all"


async def _run(args) -> None:
    psql_db = await connect()
    q = Queries(psql_db)
    try:
        if args.command == "summary":
            await menu.render_full_summary(q, _selected_types(args), include_pvp=args.pvp)
        elif args.command == "task":
            await menu._task_detail(q, args.task_id)
        elif args.command == "tournament":
            await _open_tournament(q, args)
        elif args.command == "tournaments":
            await _list_tournaments(q, args)
        elif args.command == "deployments":
            v.deployments_table(await q.active_deployments(), await q.pvp_deployments())
        elif args.command == "trainers":
            v.trainers_table(await q.trainers())
        else:
            await menu.run(q)
    finally:
        await psql_db.close()


async def _open_tournament(q: Queries, args) -> None:
    selector = args.selector
    type_alias = _type_alias_from(args)
    tournament_id: str | None = selector
    if selector == "current":
        rows = await q.list_tournaments(["active"], None if type_alias == "all" else [type_alias])
        tournament_id = rows[0]["tournament_id"] if rows else None
    elif selector == "last":
        rows = await q.list_tournaments(["completed"], None if type_alias == "all" else [type_alias])
        tournament_id = rows[0]["tournament_id"] if rows else None
    if not tournament_id:
        v.print("No matching tournament.", style="yellow")
        return
    await menu._tournament_detail(q, tournament_id)


async def _list_tournaments(q: Queries, args) -> None:
    statuses = []
    if args.active:
        statuses.append("active")
    if args.completed:
        statuses.append("completed")
    rows = await q.list_tournaments(statuses or None, None)
    v.tournaments_table(rows, "Tournaments")


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(_run(args))
    except MissingDatabaseConfig as exc:
        v.print(f"[red]{exc}[/red]")
        raise SystemExit(1)
    except ConnectionError as exc:
        v.print(f"[red]Could not connect to the database:[/red] {exc}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        v.print("\nInterrupted.")
