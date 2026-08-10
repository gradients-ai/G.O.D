#!/usr/bin/env python3
"""List and approve majority-training-failure review gates.

A task whose trainings are >50% failed is held back from completing until a human confirms
the failures belong to the miners rather than the infrastructure. This tool is that human
step: inspect what failed, then approve so the round can advance.

If the failures were an infrastructure fault, do NOT approve — reset those trainings to
pending instead and let them run again. The gate clears on its own once the ratio drops.

    python -m ops.tools.tournament.approve_task_failure_review --list
    python -m ops.tools.tournament.approve_task_failure_review --task-id <uuid> [--notes "..."]
"""

import argparse
import asyncio
import os

import asyncpg
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


console = Console()

load_dotenv(".vali.env")
DATABASE_URL = os.getenv("DATABASE_URL")

TABLE = "tournament_task_failure_reviews"


async def list_pending(conn: asyncpg.Connection) -> None:
    rows = await conn.fetch(
        f"""
        SELECT r.task_id, r.tournament_id, r.round_id, r.status, r.failed_hotkeys,
               r.total_trainings, r.created_at, r.reviewed_at
        FROM {TABLE} r
        ORDER BY r.status, r.created_at
        """
    )
    if not rows:
        console.print("No majority-failure reviews on record.", style="green")
        return

    table = Table(title="Majority-training-failure reviews")
    table.add_column("task_id", overflow="fold")
    table.add_column("round")
    table.add_column("status")
    table.add_column("failed")
    table.add_column("opened")
    for row in rows:
        failed = row["failed_hotkeys"]
        failed_count = len(failed) if isinstance(failed, list) else 0
        style = "yellow" if row["status"] == "pending_review" else "green"
        table.add_row(
            str(row["task_id"]),
            row["round_id"],
            f"[{style}]{row['status']}[/{style}]",
            f"{failed_count}/{row['total_trainings']}",
            row["created_at"].strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


async def show_detail(conn: asyncpg.Connection, task_id: str) -> asyncpg.Record | None:
    review = await conn.fetchrow(f"SELECT * FROM {TABLE} WHERE task_id = $1", task_id)
    if not review:
        console.print(f"No review row for task {task_id}", style="red")
        return None

    console.print(f"\n[bold]{task_id}[/bold]  round={review['round_id']}  status={review['status']}")
    trainings = await conn.fetch(
        """
        SELECT hotkey, training_status, n_training_attempts, trainer_ip
        FROM tournament_task_hotkey_trainings WHERE task_id = $1 ORDER BY training_status, hotkey
        """,
        task_id,
    )
    table = Table(title="Trainings")
    table.add_column("hotkey")
    table.add_column("status")
    table.add_column("attempts")
    table.add_column("trainer")
    for row in trainings:
        style = "red" if row["training_status"] == "failure" else "green"
        table.add_row(
            row["hotkey"][:12],
            f"[{style}]{row['training_status']}[/{style}]",
            str(row["n_training_attempts"]),
            row["trainer_ip"] or "-",
        )
    console.print(table)
    return review


async def approve(conn: asyncpg.Connection, task_id: str, notes: str | None) -> None:
    review = await show_detail(conn, task_id)
    if not review:
        return
    if review["status"] == "approved":
        console.print("Already approved; nothing to do.", style="yellow")
        return

    console.print(
        "\n[bold yellow]Approving accepts these failures as legitimate.[/bold yellow] "
        "The task will complete and the round will advance."
    )
    if input("Type the task_id to confirm: ").strip() != task_id:
        console.print("Aborted.", style="red")
        return

    await conn.execute(
        f"UPDATE {TABLE} SET status='approved', reviewed_at=now(), notes=COALESCE($2, notes) WHERE task_id = $1",
        task_id,
        notes,
    )
    console.print(f"Approved {task_id}. The next tournament cycle will let the round through.", style="green")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list all review rows")
    parser.add_argument("--task-id", help="task to inspect and approve")
    parser.add_argument("--notes", help="why this was approved")
    args = parser.parse_args()

    if not DATABASE_URL:
        console.print("DATABASE_URL not set (source .vali.env)", style="red")
        return

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        if args.task_id:
            await approve(conn, args.task_id, args.notes)
        else:
            await list_pending(conn)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
