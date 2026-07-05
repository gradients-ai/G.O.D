"""Interactive drill-down menu for the monitor.

Navigation convention across every screen:
  - type a number to pick an option
  - `b` to go back
  - `q` to quit
"""

from uuid import UUID

from ops.tools.god_monitor import views as v
from ops.tools.god_monitor.queries import Queries


class _Back(Exception):
    pass


class _Quit(Exception):
    pass


def ask(prompt: str) -> str:
    try:
        raw = input(f"{prompt} ").strip()
    except (EOFError, KeyboardInterrupt):
        raise _Quit()
    if raw.lower() in ("q", "quit", "exit"):
        raise _Quit()
    if raw.lower() in ("b", "back"):
        raise _Back()
    return raw


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def _pause() -> None:
    try:
        ask("\n[enter to continue]")
    except _Back:
        pass


# --- tournament selection --------------------------------------------------


async def _select_tournament(q: Queries, type_alias: str | None) -> str | None:
    types = None if type_alias in (None, "all") else [type_alias]
    v.print("\n[bold]Which tournament?[/bold]  1) current  2) last  3) other (browse/paste id)")
    choice = ask("select:")

    if choice == "1":  # current = active
        rows = await q.list_tournaments(["active"], types)
        if not rows:
            v.print("No active tournament found.", style="yellow")
            return None
        if len(rows) == 1:
            return rows[0]["tournament_id"]
        v.tournaments_table(rows, "Active Tournaments")
        return _pick_from(rows, ask("pick #:"))

    if choice == "2":  # last completed
        rows = await q.list_tournaments(["completed"], types)
        if not rows:
            v.print("No completed tournament found.", style="yellow")
            return None
        return rows[0]["tournament_id"]

    if choice == "3":  # other
        ident = ask("paste tournament id (or 'l' to list):")
        if ident.lower() == "l":
            rows = await q.list_tournaments(None, types)
            v.tournaments_table(rows, "All Tournaments")
            return _pick_from(rows, ask("pick #:"))
        return ident
    return None


def _pick_from(rows: list[dict], raw: str) -> str | None:
    if raw.isdigit() and 1 <= int(raw) <= len(rows):
        return rows[int(raw) - 1]["tournament_id"]
    if _is_uuid(raw) or raw.startswith("tourn_"):
        return raw
    v.print("Invalid selection.", style="red")
    return None


# --- tournament detail -----------------------------------------------------


async def _tournament_detail(q: Queries, tournament_id: str) -> None:
    tournament = await q.get_tournament(tournament_id)
    if not tournament:
        v.print(f"Tournament {tournament_id} not found.", style="red")
        return
    while True:
        v.print()
        v.tournament_header(tournament)
        v.print(
            "1) participant repos   2) tasks   3) rounds   "
            "4) deployments   5) training   6) full summary"
        )
        try:
            choice = ask("select (b=back):")
        except _Back:
            return

        if choice == "1":
            v.participants_table(await q.participants(tournament_id))
            _pause()
        elif choice == "2":
            rows = await q.tournament_tasks(tournament_id)
            v.tasks_table(rows)
            if rows:
                try:
                    pick = ask("task # for detail (b=skip):")
                except _Back:
                    continue
                if pick.isdigit() and 1 <= int(pick) <= len(rows):
                    await _task_detail(q, str(rows[int(pick) - 1]["task_id"]))
        elif choice == "3":
            v.rounds_table(await q.rounds(tournament_id))
            _pause()
        elif choice == "4":
            v.deployments_table(
                await q.active_deployments(),
                await q.pvp_deployments(),
                await q.individual_deployments(),
            )
            _pause()
        elif choice == "5":
            rows = await q.tournament_training(tournament_id)
            v.training_summary_table(rows)
            v.training_details_table(rows)
            v.hf_links(rows)
            _pause()
        elif choice == "6":
            await _render_tournament_summary(q, tournament, include_pvp=True)
            _pause()
        else:
            v.print("Unknown option.", style="red")


# --- task detail (sticky sub-menu, arbitrary id) ---------------------------


async def _task_detail(q: Queries, task_id: str) -> None:
    if not _is_uuid(task_id):
        v.print(f"'{task_id}' is not a valid task id (UUID).", style="red")
        return
    try:
        task = await q.task(task_id)
    except Exception as exc:  # noqa: BLE001 - surface lookup errors to the user
        v.print(f"Could not load task {task_id}: {exc}", style="red")
        return
    if task is None:
        v.print(f"Task {task_id} not found.", style="red")
        return

    tournament_id = await q.tournament_id_for_task(task_id)
    while True:
        v.print()
        v.header(
            f"Task: [bold]{task_id}[/bold]"
            + (f"\nTournament: {tournament_id}" if tournament_id else "")
        )
        v.print(
            "1) details & status   2) participants   3) evaluations   "
            "4) scores   5) pvp results"
        )
        try:
            choice = ask("select (b=back):")
        except _Back:
            return

        if choice == "1":
            v.task_detail_panel(task)
            _pause()
        elif choice == "2":
            v.task_participants_table(await q.task_scores(task_id), await q.task_training(task_id))
            _pause()
        elif choice == "3":
            v.evaluations_table(await q.task_evaluations(task_id))
            _pause()
        elif choice == "4":
            v.scores_table(await q.task_scores(task_id), await q.task_winner(task_id))
            _pause()
        elif choice == "5":
            v.pvp_pairs_table(await q.pvp_pairs(task_id), task_id)
            v.pvp_individual_table(await q.pvp_individual_scores(task_id))
            _pause()
        else:
            v.print("Unknown option.", style="red")


# --- summary ---------------------------------------------------------------


async def _render_tournament_summary(q: Queries, tournament, include_pvp: bool = False) -> None:
    v.tournament_header(tournament)
    rounds = await q.rounds(tournament.tournament_id)
    if rounds:
        v.rounds_table(rounds)
    tasks = await q.tournament_tasks(tournament.tournament_id)
    if tasks:
        v.tasks_table(tasks)
        ttype = getattr(tournament.tournament_type, "value", tournament.tournament_type)
        if include_pvp and ttype == "environment":
            for t in tasks:
                pairs = await q.pvp_pairs(str(t["task_id"]))
                if pairs:
                    v.pvp_pairs_table(pairs, str(t["task_id"]))
    training = await q.tournament_training(tournament.tournament_id)
    if training:
        v.training_summary_table(training)
        v.training_details_table(training)
        v.hf_links(training)
    synced = await q.synced_tasks(tournament.tournament_id)
    if synced:
        v.synced_tasks_table(synced)
    v.rule()


async def render_full_summary(q: Queries, types: list[str] | None, include_pvp: bool) -> None:
    rows = await q.list_tournaments(["active"], types)
    if not rows:
        v.print("No active tournaments found.", style="yellow")
        return
    v.tournaments_table(rows, "Active Tournaments")
    for row in rows:
        tournament = await q.get_tournament(row["tournament_id"])
        if tournament:
            v.print()
            await _render_tournament_summary(q, tournament, include_pvp)


# --- main menu -------------------------------------------------------------


async def run(q: Queries) -> None:
    v.header("[bold cyan]G.O.D Tournament Monitor[/bold cyan]  (read-only)")
    while True:
        v.print(
            "\n[bold]Main menu[/bold]\n"
            "  1) Tournaments        5) PvP (by task id)\n"
            "  2) Tasks (by id)      6) Trainers / GPUs\n"
            "  3) Evaluations        7) Full summary\n"
            "  4) Deployments        q) Quit"
        )
        try:
            choice = ask("select:")
        except _Back:
            continue
        except _Quit:
            v.print("Bye.")
            return

        try:
            if choice == "1":
                await _tournaments_menu(q)
            elif choice == "2":
                ident = ask("enter task id:")
                await _task_detail(q, ident)
            elif choice == "3":
                ident = ask("enter task id:")
                v.evaluations_table(await q.task_evaluations(ident)) if _is_uuid(ident) else v.print(
                    "Not a valid task id.", style="red"
                )
                _pause()
            elif choice == "4":
                v.deployments_table(
                    await q.active_deployments(),
                    await q.pvp_deployments(),
                    await q.individual_deployments(),
                )
                _pause()
            elif choice == "5":
                ident = ask("enter task id:")
                if _is_uuid(ident):
                    v.pvp_pairs_table(await q.pvp_pairs(ident), ident)
                    v.pvp_individual_table(await q.pvp_individual_scores(ident))
                else:
                    v.print("Not a valid task id.", style="red")
                _pause()
            elif choice == "6":
                v.trainers_table(await q.trainers())
                _pause()
            elif choice == "7":
                await render_full_summary(q, None, include_pvp=True)
                _pause()
            else:
                v.print("Unknown option.", style="red")
        except _Back:
            continue
        except _Quit:
            v.print("Bye.")
            return


async def _tournaments_menu(q: Queries) -> None:
    v.print("\n[bold]Tournament type[/bold]  1) text  2) image  3) env  4) all")
    try:
        choice = ask("select:")
    except _Back:
        return
    type_map = {"1": "text", "2": "image", "3": "env", "4": "all"}
    type_alias = type_map.get(choice)
    if not type_alias:
        v.print("Unknown option.", style="red")
        return
    try:
        tournament_id = await _select_tournament(q, type_alias)
    except _Back:
        return
    if tournament_id:
        await _tournament_detail(q, tournament_id)
