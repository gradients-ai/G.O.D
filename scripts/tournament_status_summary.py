#!/usr/bin/env python3

import argparse
import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional

import asyncpg
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

console = Console()

TOURNAMENT_TYPES = {
    "image": "image",
    "text": "text",
    "env": "environment",
}


#source .vali.env
load_dotenv( ".vali.env")
DATABASE_URL = os.getenv("DATABASE_URL")


class TournamentStatusSummary:
    def __init__(self):
        self.db_pool = None

    async def connect_db(self):
        """Connect to the database"""
        try:
            self.db_pool = await asyncpg.create_pool(DATABASE_URL)
            console.print("✅ Connected to database", style="green")
        except Exception as e:
            console.print(f"❌ Failed to connect to database: {e}", style="red")
            raise

    async def close_db(self):
        """Close database connection"""
        if self.db_pool:
            await self.db_pool.close()

    async def get_active_tournaments(self, tournament_types: Optional[List[str]] = None) -> List[Dict]:
        """Get all active tournaments"""
        async with self.db_pool.acquire() as conn:
            type_filter = ""
            params = []
            if tournament_types:
                type_filter = "AND tournament_type = ANY($1::text[])"
                params.append(tournament_types)

            query = """
                SELECT tournament_id, tournament_type, status, created_at, updated_at
                FROM tournaments 
                WHERE status = 'active'
                {type_filter}
                ORDER BY created_at DESC
            """.format(type_filter=type_filter)
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

    async def get_tournament_rounds(self, tournament_id: str) -> List[Dict]:
        """Get all rounds for a tournament"""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT round_id, round_number, status, created_at, started_at, completed_at
                FROM tournament_rounds 
                WHERE tournament_id = $1
                ORDER BY round_number
            """
            rows = await conn.fetch(query, tournament_id)
            return [dict(row) for row in rows]

    async def get_tournament_tasks(self, tournament_id: str) -> List[Dict]:
        """Get all tasks for a tournament with their status"""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT 
                    t.task_id,
                    t.task_type,
                    t.status,
                    t.created_at,
                    t.updated_at,
                    t.started_at,
                    t.completed_at,
                    t.hours_to_complete,
                    tt.round_id,
                    tt.group_id,
                    tt.pair_id
                FROM tasks t
                JOIN tournament_tasks tt ON t.task_id = tt.task_id
                WHERE tt.tournament_id = $1
                ORDER BY t.created_at
            """
            rows = await conn.fetch(query, tournament_id)
            return [dict(row) for row in rows]

    async def get_training_status(self, tournament_id: str) -> List[Dict]:
        """Get training status for all tasks in a tournament"""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT 
                    ttht.task_id,
                    ttht.hotkey,
                    ttht.training_status,
                    ttht.n_training_attempts,
                    ttht.created_at,
                    ttht.updated_at,
                    tn.expected_repo_name,
                    s.repo as submission_repo
                FROM tournament_task_hotkey_trainings ttht
                JOIN tournament_tasks tt ON ttht.task_id = tt.task_id
                LEFT JOIN task_nodes tn ON ttht.task_id = tn.task_id AND ttht.hotkey = tn.hotkey
                LEFT JOIN submissions s ON ttht.task_id = s.task_id AND ttht.hotkey = s.hotkey
                WHERE tt.tournament_id = $1
                ORDER BY ttht.task_id, ttht.hotkey
            """
            rows = await conn.fetch(query, tournament_id)
            return [dict(row) for row in rows]

    async def get_synced_tasks(self, tournament_id: str) -> List[Dict]:
        """Get synced tasks for a tournament"""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT 
                    brst.tournament_task_id,
                    brst.general_task_id,
                    t1.status as tournament_task_status,
                    t2.status as general_task_status
                FROM boss_round_synced_tasks brst
                JOIN tasks t1 ON brst.tournament_task_id = t1.task_id
                JOIN tasks t2 ON brst.general_task_id = t2.task_id
                JOIN tournament_tasks tt ON t1.task_id = tt.task_id
                WHERE tt.tournament_id = $1
            """
            rows = await conn.fetch(query, tournament_id)
            return [dict(row) for row in rows]

    async def get_pvp_pair_results(self, task_id: str) -> List[Dict]:
        """Get PvP pair results for a tournament task"""
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT
                    hotkey_a,
                    hotkey_b,
                    environment_name,
                    model_a_wins,
                    model_b_wins,
                    draws,
                    total_games,
                    status
                FROM pvp_pair_results
                WHERE task_id = $1
                ORDER BY environment_name, hotkey_a, hotkey_b
            """
            rows = await conn.fetch(query, task_id)
            return [dict(row) for row in rows]

    def format_duration(self, start_time, end_time=None) -> str:
        """Format duration between two timestamps"""
        if not start_time:
            return "N/A"

        if not end_time:
            end_time = datetime.utcnow()

        # Handle timezone-aware vs timezone-naive datetimes
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

        # Make both timezone-aware or both timezone-naive
        if start_time.tzinfo is None and end_time.tzinfo is not None:
            # Make start_time timezone-aware
            start_time = start_time.replace(tzinfo=end_time.tzinfo)
        elif start_time.tzinfo is not None and end_time.tzinfo is None:
            # Make end_time timezone-aware
            end_time = end_time.replace(tzinfo=start_time.tzinfo)

        duration = end_time - start_time
        hours = duration.total_seconds() / 3600

        if hours < 1:
            return f"{int(duration.total_seconds() / 60)}m"
        elif hours < 24:
            return f"{hours:.1f}h"
        else:
            days = hours / 24
            return f"{days:.1f}d"

    def create_tournament_summary_table(self, tournaments: List[Dict]) -> Table:
        """Create a table showing tournament summaries"""
        table = Table(title="🏆 Active Tournaments Summary")
        table.add_column("Tournament ID", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Created", style="yellow")
        table.add_column("Last Updated", style="yellow")
        table.add_column("Age", style="blue")

        for tournament in tournaments:
            age = self.format_duration(tournament["created_at"])
            created = tournament["created_at"].strftime("%Y-%m-%d %H:%M")
            updated = tournament["updated_at"].strftime("%Y-%m-%d %H:%M") if tournament["updated_at"] else "N/A"

            table.add_row(tournament["tournament_id"], tournament["tournament_type"], tournament["status"], created, updated, age)

        return table

    def create_rounds_table(self, rounds: List[Dict]) -> Table:
        """Create a table showing tournament rounds"""
        table = Table(title="🔄 Tournament Rounds")
        table.add_column("Round ID", style="cyan", no_wrap=True)
        table.add_column("Round #", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Created", style="yellow")
        table.add_column("Started", style="yellow")
        table.add_column("Completed", style="yellow")

        for round_data in rounds:
            created = round_data["created_at"].strftime("%Y-%m-%d %H:%M") if round_data["created_at"] else "N/A"
            started = round_data["started_at"].strftime("%Y-%m-%d %H:%M") if round_data["started_at"] else "N/A"
            completed = round_data["completed_at"].strftime("%Y-%m-%d %H:%M") if round_data["completed_at"] else "N/A"

            table.add_row(
                round_data["round_id"], str(round_data["round_number"]), round_data["status"], created, started, completed
            )

        return table

    def create_tasks_table(self, tasks: List[Dict]) -> Table:
        """Create a table showing tournament tasks"""
        table = Table(title="📋 Tournament Tasks")
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Pair/Group", style="blue")
        table.add_column("Created", style="yellow")
        table.add_column("Duration", style="blue")
        table.add_column("Training Hours", style="blue")

        for task in tasks:
            task_id = str(task["task_id"])
            pair_group = task["pair_id"] if task["pair_id"] else (task["group_id"] if task["group_id"] else "N/A")

            created = task["created_at"].strftime("%Y-%m-%d %H:%M")
            duration = self.format_duration(task["created_at"], task["completed_at"])
            training_hours = f"{task['hours_to_complete']}h" if task.get("hours_to_complete") else "N/A"

            # Color code status
            status_style = (
                "green"
                if task["status"] == "success"
                else "red"
                if task["status"] == "failure"
                else "yellow"
                if task["status"] in ["training", "looking_for_nodes"]
                else "blue"
            )

            table.add_row(
                task_id, task["task_type"], f"[{status_style}]{task['status']}[/{status_style}]", pair_group, created, duration, training_hours
            )

        return table

    def create_training_summary_table(self, training_data: List[Dict]) -> Table:
        """Create a table showing training status summary"""
        # Group by training status
        status_counts = {}
        for item in training_data:
            status = item["training_status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        table = Table(title="🏋️ Training Status Summary")
        table.add_column("Status", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Percentage", style="green")

        total = len(training_data)
        for status, count in status_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            status_style = (
                "green" if status == "success" else "red" if status == "failure" else "yellow" if status == "training" else "blue"
            )

            table.add_row(f"[{status_style}]{status}[/{status_style}]", str(count), f"{percentage:.1f}%")

        return table

    def create_training_details_table(self, training_data: List[Dict]) -> Table:
        """Create a table showing detailed training information"""
        table = Table(title="🏋️ Training Details")
        table.add_column("Task ID", style="cyan", no_wrap=True)
        table.add_column("Hotkey", style="magenta", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Attempts", style="blue")
        table.add_column("Last Updated", style="yellow")

        # Separate successful training entries for better link display
        successful_trainings = []

        for item in training_data:
            task_id = str(item["task_id"])
            hotkey = str(item["hotkey"])
            updated = item["updated_at"].strftime("%Y-%m-%d %H:%M") if item["updated_at"] else "N/A"

            # Color code status
            status_style = (
                "green"
                if item["training_status"] == "success"
                else "red"
                if item["training_status"] == "failure"
                else "yellow"
                if item["training_status"] == "training"
                else "blue"
            )

            # Store successful trainings for separate display
            if item["training_status"] == "success" and item["expected_repo_name"]:
                successful_trainings.append(
                    {"task_id": task_id, "hotkey": hotkey, "repo": item["expected_repo_name"], "updated": updated}
                )

            table.add_row(
                task_id,
                hotkey,
                f"[{status_style}]{item['training_status']}[/{status_style}]",
                str(item["n_training_attempts"]),
                updated,
            )

        return table, successful_trainings

    def create_synced_tasks_table(self, synced_tasks: List[Dict]) -> Table:
        """Create a table showing synced tasks"""
        table = Table(title="🔄 Synced Tasks")
        table.add_column("Tournament Task ID", style="cyan", no_wrap=True)
        table.add_column("General Task ID", style="magenta", no_wrap=True)
        table.add_column("Tournament Status", style="green")
        table.add_column("General Status", style="blue")

        for task in synced_tasks:
            tournament_id = str(task["tournament_task_id"])
            general_id = str(task["general_task_id"])

            table.add_row(tournament_id, general_id, task["tournament_task_status"], task["general_task_status"])

        return table

    def create_pvp_results_table(self, task: Dict, pvp_results: List[Dict]) -> Table:
        """Create a table showing PvP pair results for one environment task"""
        table = Table(title=f"⚔️ PvP Results - Task {task['task_id']}")
        table.add_column("Environment", style="cyan")
        table.add_column("Hotkey A", style="magenta", no_wrap=True)
        table.add_column("Hotkey B", style="magenta", no_wrap=True)
        table.add_column("A Wins", style="green", justify="right")
        table.add_column("B Wins", style="green", justify="right")
        table.add_column("Draws", style="blue", justify="right")
        table.add_column("Games", style="blue", justify="right")
        table.add_column("Status", style="green")

        for result in pvp_results:
            status = result["status"]
            status_style = (
                "green"
                if status == "complete"
                else "yellow"
                if status == "pending"
                else "blue"
            )

            table.add_row(
                result["environment_name"],
                result["hotkey_a"],
                result["hotkey_b"],
                str(result["model_a_wins"]),
                str(result["model_b_wins"]),
                str(result["draws"]),
                str(result["total_games"]),
                f"[{status_style}]{status}[/{status_style}]",
            )

        return table

    async def generate_summary(self, tournament_types: Optional[List[str]] = None, include_pvp: bool = False):
        """Generate comprehensive tournament summary"""
        try:
            await self.connect_db()

            # Get active tournaments
            tournaments = await self.get_active_tournaments(tournament_types)

            if not tournaments:
                console.print("No active tournaments found.", style="yellow")
                return

            # Display tournament summary
            console.print(self.create_tournament_summary_table(tournaments))
            console.print()

            # Process each tournament
            for tournament in tournaments:
                tournament_id = tournament["tournament_id"]

                # Create tournament header
                header = Panel(
                    f"Tournament: {tournament_id}\n"
                    f"Type: {tournament['tournament_type']} | "
                    f"Status: {tournament['status']} | "
                    f"Created: {tournament['created_at'].strftime('%Y-%m-%d %H:%M')}",
                    title="🏆 Tournament Details",
                    border_style="cyan",
                )
                console.print(header)

                # Get and display rounds
                rounds = await self.get_tournament_rounds(tournament_id)
                if rounds:
                    console.print(self.create_rounds_table(rounds))
                    console.print()

                # Get and display tasks
                tasks = await self.get_tournament_tasks(tournament_id)
                if tasks:
                    console.print(self.create_tasks_table(tasks))
                    console.print()

                    if include_pvp and tournament["tournament_type"] == "environment":
                        for task in tasks:
                            pvp_results = await self.get_pvp_pair_results(str(task["task_id"]))
                            if pvp_results:
                                console.print(self.create_pvp_results_table(task, pvp_results))
                                console.print()
                            else:
                                console.print(f"No PvP results found for task {task['task_id']}.", style="yellow")
                                console.print()

                # Get and display training summary
                training_data = await self.get_training_status(tournament_id)
                if training_data:
                    console.print(self.create_training_summary_table(training_data))
                    console.print()
                    # Get and display detailed training information
                    training_table, successful_trainings = self.create_training_details_table(training_data)
                    console.print(training_table)
                    console.print()

                    # Display successful training hotkeys with links
                    if successful_trainings:
                        console.print("🔗 Hugging Face Links for Successful Trainings:")
                        for item in successful_trainings:
                            hugging_face_link = f"https://huggingface.co/gradients-io-tournaments/{item['repo']}"
                            console.print(f"  - Task ID: {item['task_id']}, Hotkey: {item['hotkey']}, Link: {hugging_face_link}")
                        console.print()

                # Get and display synced tasks
                synced_tasks = await self.get_synced_tasks(tournament_id)
                if synced_tasks:
                    console.print(self.create_synced_tasks_table(synced_tasks))
                    console.print()
                else:
                    console.print("No synced tasks found for this tournament.", style="yellow")
                    console.print()

                # Add separator between tournaments
                console.print("=" * 80, style="dim")
                console.print()

        except Exception as e:
            console.print(f"❌ Error generating summary: {e}", style="red")
        finally:
            await self.close_db()


def parse_args():
    """Parse CLI arguments"""
    parser = argparse.ArgumentParser(description="Show active tournament status summaries.")
    parser.add_argument("--image", action="store_true", help="Show image tournaments.")
    parser.add_argument("--text", action="store_true", help="Show text tournaments.")
    parser.add_argument("--env", action="store_true", help="Show environment tournaments.")
    parser.add_argument("--all", action="store_true", help="Show all tournament types.")
    parser.add_argument("--pvp", action="store_true", help="Show PvP result tables for environment tournament tasks.")
    return parser.parse_args()


def selected_tournament_types(args) -> List[str]:
    """Resolve tournament type flags, defaulting to all."""
    selected = [
        tournament_type
        for flag, tournament_type in TOURNAMENT_TYPES.items()
        if getattr(args, flag)
    ]

    if args.all or not selected:
        return list(TOURNAMENT_TYPES.values())

    return selected


async def main():
    """Main function"""
    args = parse_args()
    tournament_types = selected_tournament_types(args)

    console.print("🏆 Tournament Status Summary Tool", style="bold cyan")
    console.print("=" * 50, style="dim")
    console.print(f"Types: {', '.join(tournament_types)}", style="dim")
    if args.pvp and "environment" not in tournament_types:
        console.print("--pvp is only applicable to environment tournaments.", style="yellow")
    console.print()

    summary_tool = TournamentStatusSummary()
    await summary_tool.generate_summary(tournament_types, args.pvp)


if __name__ == "__main__":
    asyncio.run(main())
