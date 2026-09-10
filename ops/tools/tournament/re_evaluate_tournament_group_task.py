#!/usr/bin/env python3
"""Reset a completed tournament group task for partial retrain + full re-eval.

Use when a group task finished with unfair training failures (e.g. orchestrator/trainer
desync) but the round already advanced. This tool:

  reset               — Step 1: reset selected hotkeys to pending, wipe stale eval artifacts,
                        set task back to training so orchestrator + eval loops re-run.
  show-winner         — Step 3: print group winner from task_nodes.quality_score after re-eval.
  swap-round3-contender — Step 4: replace one round-3 finalist if the group winner changed.

Round-2 advancement is NOT re-triggered automatically once round 3 exists; use swap-round3-
contender when the corrected group winner differs from who advanced.

Examples:

    python -m ops.tools.tournament.re_evaluate_tournament_group_task reset \\
        --task-id 2f4d34af-f494-4128-8117-fa7fce4a5c99 \\
        --retrain-hotkeys 5EqKHiZG...,5HKEAZ...,5EqnMmpf... \\
        --dry-run

    python -m ops.tools.tournament.re_evaluate_tournament_group_task show-winner \\
        --task-id 2f4d34af-f494-4128-8117-fa7fce4a5c99

    python -m ops.tools.tournament.re_evaluate_tournament_group_task swap-round3-contender \\
        --tournament-id tourn_31f2e0fe36783f71_20260831 \\
        --round3-group-id tourn_31f2e0fe36783f71_20260831_round_003_group_001 \\
        --old-hotkey 5GU4Xkd3... --new-hotkey 5HKEAZxF... \\
        --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import os
from uuid import UUID

import asyncpg
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from core.constants.network import NETUID
from validator.db import constants as cst


console = Console()

load_dotenv(".vali.env")
DATABASE_URL = os.getenv("DATABASE_URL")


async def _connect() -> asyncpg.Connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set (source .vali.env)")
    return await asyncpg.connect(DATABASE_URL)


async def _get_task_hotkeys(conn: asyncpg.Connection, task_id: str) -> list[str]:
    rows = await conn.fetch(
        f"""
        SELECT {cst.HOTKEY} FROM {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
        WHERE {cst.TASK_ID} = $1 ORDER BY {cst.HOTKEY}
        """,
        task_id,
    )
    return [row[cst.HOTKEY] for row in rows]


async def cmd_reset(args: argparse.Namespace) -> None:
    task_id = args.task_id
    UUID(task_id)
    retrain_hotkeys = [hk.strip() for hk in args.retrain_hotkeys.split(",") if hk.strip()]

    conn = await _connect()
    try:
        all_hotkeys = await _get_task_hotkeys(conn, task_id)
        if not all_hotkeys:
            console.print(f"No training rows for task {task_id}", style="red")
            return

        unknown = set(retrain_hotkeys) - set(all_hotkeys)
        if unknown:
            console.print(f"Hotkeys not on task: {unknown}", style="red")
            return

        keep_hotkeys = [hk for hk in all_hotkeys if hk not in retrain_hotkeys]
        table = Table(title=f"Reset plan for task {task_id}")
        table.add_column("hotkey", overflow="fold")
        table.add_column("action")
        for hk in retrain_hotkeys:
            table.add_row(hk, "[yellow]retrain[/yellow] (pending, attempts=0, clear submission)")
        for hk in keep_hotkeys:
            table.add_row(hk, "[green]keep training status[/green]")
        table.add_row("—", "delete pvp_pair_results + pvp_individual_scores")
        table.add_row("—", "clear quality_score/test_loss/synth_loss for ALL hotkeys on task")
        table.add_row("—", "reset evaluations to pending")
        table.add_row("—", "set tasks.status = training")
        console.print(table)

        if args.dry_run:
            console.print("Dry run — no changes applied.", style="cyan")
            return

        async with conn.transaction():
            for hk in retrain_hotkeys:
                await conn.execute(
                    f"""
                    UPDATE {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
                    SET {cst.TRAINING_STATUS} = 'pending',
                        {cst.N_TRAINING_ATTEMPTS} = 0,
                        {cst.TRAINER_IP} = NULL,
                        {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                    WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
                    """,
                    task_id,
                    hk,
                )
                await conn.execute(
                    f"DELETE FROM {cst.SUBMISSIONS_TABLE} WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2",
                    task_id,
                    hk,
                )

            await conn.execute(
                f"DELETE FROM {cst.PVP_PAIR_RESULTS_TABLE} WHERE {cst.TASK_ID} = $1", task_id
            )
            await conn.execute(
                f"DELETE FROM {cst.PVP_INDIVIDUAL_SCORES_TABLE} WHERE {cst.TASK_ID} = $1", task_id
            )
            await conn.execute(
                f"""
                UPDATE {cst.TASK_NODES_TABLE}
                SET {cst.TASK_NODE_QUALITY_SCORE} = NULL,
                    {cst.TEST_LOSS} = NULL,
                    {cst.SYNTH_LOSS} = NULL
                WHERE {cst.TASK_ID} = $1
                """,
                task_id,
            )
            await conn.execute(
                f"""
                UPDATE {cst.EVALUATIONS_TABLE}
                SET {cst.EVALUATION_STATUS} = 'pending',
                    {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2
                """,
                task_id,
                NETUID,
            )
            await conn.execute(
                f"""
                UPDATE {cst.TASKS_TABLE}
                SET {cst.STATUS} = 'training', {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                WHERE {cst.TASK_ID} = $1
                """,
                task_id,
            )

        console.print(f"Reset complete for task {task_id}. Orchestrator will retrain pending hotkeys.", style="green")
    finally:
        await conn.close()


async def cmd_show_winner(args: argparse.Namespace) -> None:
    task_id = args.task_id
    conn = await _connect()
    try:
        rows = await conn.fetch(
            f"""
            SELECT {cst.HOTKEY}, {cst.TASK_NODE_QUALITY_SCORE}, {cst.TEST_LOSS}, {cst.SYNTH_LOSS}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            ORDER BY {cst.TASK_NODE_QUALITY_SCORE} DESC NULLS LAST, {cst.HOTKEY}
            """,
            task_id,
        )
        table = Table(title=f"Scores for task {task_id}")
        table.add_column("hotkey", overflow="fold")
        table.add_column("quality_score")
        table.add_column("test_loss")
        table.add_column("synth_loss")
        winner = None
        best_score = None
        for row in rows:
            qs = row[cst.TASK_NODE_QUALITY_SCORE]
            table.add_row(row[cst.HOTKEY], str(qs), str(row[cst.TEST_LOSS]), str(row[cst.SYNTH_LOSS]))
            if qs is not None and (best_score is None or qs > best_score):
                best_score = qs
                winner = row[cst.HOTKEY]
        console.print(table)
        if winner:
            console.print(f"Group winner (highest quality_score): {winner}", style="bold green")
        else:
            console.print("No scored winner yet — wait for re-eval to finish.", style="yellow")
    finally:
        await conn.close()


async def cmd_swap_round3_contender(args: argparse.Namespace) -> None:
    tournament_id = args.tournament_id
    round3_group_id = args.round3_group_id
    old_hotkey = args.old_hotkey
    new_hotkey = args.new_hotkey
    round2_id = args.round2_id or f"{tournament_id}_round_002"

    conn = await _connect()
    try:
        r3_tasks = await conn.fetch(
            f"""
            SELECT t.{cst.TASK_ID}::text
            FROM {cst.TOURNAMENT_TASKS_TABLE} tt
            JOIN {cst.TASKS_TABLE} t ON t.{cst.TASK_ID} = tt.{cst.TASK_ID}
            WHERE tt.{cst.GROUP_ID} = $1
            ORDER BY t.{cst.CREATED_AT}
            """,
            round3_group_id,
        )
        if not r3_tasks:
            console.print(f"No round-3 tasks for group {round3_group_id}", style="red")
            return

        task_ids = [row[cst.TASK_ID] for row in r3_tasks]
        table = Table(title="Round 3 contender swap plan")
        table.add_column("step")
        table.add_column("detail", overflow="fold")
        table.add_row("participants", f"clear eliminated_in_round_id for {new_hotkey}")
        table.add_row("participants", f"set eliminated_in_round_id={round2_id} for {old_hotkey}")
        table.add_row("group members", f"replace {old_hotkey} -> {new_hotkey} in {round3_group_id}")
        for tid in task_ids:
            table.add_row("task_nodes", f"reassign {tid} node from {old_hotkey} to {new_hotkey}")
            table.add_row("trainings", f"reset {new_hotkey} on {tid} to pending; leave {old_hotkey} orphaned")
        console.print(table)

        if args.dry_run:
            console.print("Dry run — no changes applied.", style="cyan")
            return

        async with conn.transaction():
            await conn.execute(
                f"""
                UPDATE {cst.TOURNAMENT_PARTICIPANTS_TABLE}
                SET {cst.ELIMINATED_IN_ROUND_ID} = NULL
                WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.HOTKEY} = $2
                """,
                tournament_id,
                new_hotkey,
            )
            await conn.execute(
                f"""
                UPDATE {cst.TOURNAMENT_PARTICIPANTS_TABLE}
                SET {cst.ELIMINATED_IN_ROUND_ID} = $2
                WHERE {cst.TOURNAMENT_ID} = $1 AND {cst.HOTKEY} = $3
                """,
                tournament_id,
                round2_id,
                old_hotkey,
            )
            await conn.execute(
                f"""
                UPDATE {cst.TOURNAMENT_GROUP_MEMBERS_TABLE}
                SET {cst.HOTKEY} = $3
                WHERE {cst.GROUP_ID} = $1 AND {cst.HOTKEY} = $2
                """,
                round3_group_id,
                old_hotkey,
                new_hotkey,
            )

            for tid in task_ids:
                old_node = await conn.fetchrow(
                    f"""
                    SELECT {cst.EXPECTED_REPO_NAME}, {cst.NODE_ID}
                    FROM {cst.TASK_NODES_TABLE}
                    WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
                    """,
                    tid,
                    old_hotkey,
                )
                if not old_node:
                    console.print(f"No task_node for {old_hotkey} on {tid}", style="red")
                    raise RuntimeError("missing old task_node")

                new_repo = await conn.fetchval(
                    f"""
                    SELECT {cst.EXPECTED_REPO_NAME}
                    FROM {cst.TASK_NODES_TABLE}
                    WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
                    """,
                    args.r2_task_id,
                    new_hotkey,
                )
                if not new_repo:
                    console.print(
                        f"No R2 expected_repo_name for {new_hotkey} on {args.r2_task_id}", style="red"
                    )
                    raise RuntimeError("missing new winner repo from R2 task")

                existing_new = await conn.fetchval(
                    f"""
                    SELECT 1 FROM {cst.TASK_NODES_TABLE}
                    WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
                    """,
                    tid,
                    new_hotkey,
                )
                if existing_new:
                    await conn.execute(
                        f"""
                        UPDATE {cst.TASK_NODES_TABLE}
                        SET {cst.EXPECTED_REPO_NAME} = $3,
                            {cst.TASK_NODE_QUALITY_SCORE} = NULL,
                            {cst.TEST_LOSS} = NULL,
                            {cst.SYNTH_LOSS} = NULL,
                            {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                        WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
                        """,
                        tid,
                        new_hotkey,
                        new_repo,
                    )
                else:
                    await conn.execute(
                        f"""
                        UPDATE {cst.TASK_NODES_TABLE}
                        SET {cst.HOTKEY} = $3,
                            {cst.EXPECTED_REPO_NAME} = $4,
                            {cst.TASK_NODE_QUALITY_SCORE} = NULL,
                            {cst.TEST_LOSS} = NULL,
                            {cst.SYNTH_LOSS} = NULL,
                            {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                        WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
                        """,
                        tid,
                        old_hotkey,
                        new_hotkey,
                        new_repo,
                    )

                await conn.execute(
                    f"""
                    UPDATE {cst.TOURNAMENT_TASK_HOTKEY_TRAININGS_TABLE}
                    SET {cst.TRAINING_STATUS} = 'pending',
                        {cst.N_TRAINING_ATTEMPTS} = 0,
                        {cst.TRAINER_IP} = NULL,
                        {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                    WHERE {cst.TASK_ID} = $1 AND {cst.HOTKEY} = $2
                    """,
                    tid,
                    new_hotkey,
                )

                task_status = await conn.fetchval(
                    f"SELECT {cst.STATUS} FROM {cst.TASKS_TABLE} WHERE {cst.TASK_ID} = $1", tid
                )
                if task_status == "success":
                    await conn.execute(
                        f"""
                        UPDATE {cst.TASKS_TABLE}
                        SET {cst.STATUS} = 'training', {cst.UPDATED_AT} = CURRENT_TIMESTAMP
                        WHERE {cst.TASK_ID} = $1
                        """,
                        tid,
                    )

        console.print("Round 3 contender swap complete.", style="green")
    finally:
        await conn.close()


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    reset_p = sub.add_parser("reset", help="Reset task for partial retrain + full re-eval")
    reset_p.add_argument("--task-id", required=True)
    reset_p.add_argument(
        "--retrain-hotkeys",
        required=True,
        help="Comma-separated hotkeys to reset to pending (others keep current training status)",
    )
    _add_dry_run(reset_p)

    show_p = sub.add_parser("show-winner", help="Show group winner from quality_score")
    show_p.add_argument("--task-id", required=True)

    swap_p = sub.add_parser("swap-round3-contender", help="Replace round-3 finalist after corrected R2 winner")
    swap_p.add_argument("--tournament-id", required=True)
    swap_p.add_argument("--round3-group-id", required=True)
    swap_p.add_argument("--old-hotkey", required=True, help="Hotkey currently in round 3 (incorrect advancer)")
    swap_p.add_argument("--new-hotkey", required=True, help="Corrected group winner from R2 re-eval")
    swap_p.add_argument(
        "--r2-task-id",
        required=True,
        help="Round-2 group task id (source of new winner expected_repo_name)",
    )
    swap_p.add_argument(
        "--round2-id",
        help="Round id for eliminating old contender (default: {tournament_id}_round_002)",
    )
    _add_dry_run(swap_p)

    args = parser.parse_args()
    args.dry_run = not getattr(args, "apply", False)

    if args.command == "reset":
        asyncio.run(cmd_reset(args))
    elif args.command == "show-winner":
        asyncio.run(cmd_show_winner(args))
    elif args.command == "swap-round3-contender":
        asyncio.run(cmd_swap_round3_contender(args))


if __name__ == "__main__":
    main()
