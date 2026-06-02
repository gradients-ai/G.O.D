from validator.tournament.models import BossRoundTaskPair
from validator.tournament.models import TaskScore
from validator.db import constants as cst
from validator.db.database import PSQLDB


async def get_boss_round_winner_task_pairs(tournament_id: str, psql_db: PSQLDB) -> list[BossRoundTaskPair]:
    async with await psql_db.connection() as connection:
        query = """
            SELECT t.task_id, t.task_type, tourn.winner_hotkey, brst.general_task_id
            FROM tasks t
            JOIN tournament_tasks tt ON tt.task_id = t.task_id
            JOIN tournament_rounds tr ON tr.round_id = tt.round_id
            JOIN tournaments tourn ON tourn.tournament_id = tr.tournament_id
            JOIN boss_round_synced_tasks brst ON brst.tournament_task_id = t.task_id
            WHERE tr.tournament_id = $1 AND tr.is_final_round = true
        """
        results = await connection.fetch(query, tournament_id)
        return [
            BossRoundTaskPair(
                tournament_task_id=str(row["task_id"]),
                synthetic_task_id=str(row["general_task_id"]),
                winner_hotkey=row["winner_hotkey"],
                task_type=row["task_type"],
            )
            for row in results
        ]


async def update_tournament_winning_performance(tournament_id: str, winning_performance_difference: float, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        query = f"""
            UPDATE {cst.TOURNAMENTS_TABLE}
            SET {cst.WINNING_PERFORMANCE_DIFFERENCE} = $2
            WHERE {cst.TOURNAMENT_ID} = $1
        """
        await connection.execute(query, tournament_id, winning_performance_difference)


async def get_task_scores_batch(task_ids: list[str], psql_db: PSQLDB) -> dict[str, list[TaskScore]]:
    """Fetch task scores for multiple tasks in a single query to avoid N+1 problem."""
    if not task_ids:
        return {}

    async with await psql_db.connection() as connection:
        # Convert task_ids to UUIDs for query
        query = f"""
            SELECT task_id, hotkey, test_loss, synth_loss, quality_score
            FROM {cst.TASK_NODES_TABLE}
            WHERE task_id = ANY($1::uuid[])
        """
        results = await connection.fetch(query, task_ids)

        # Group results by task_id
        task_scores = {}
        for row in results:
            task_id = str(row["task_id"])
            if task_id not in task_scores:
                task_scores[task_id] = []

            # Filter out NaN values
            test_loss = row["test_loss"]

            if test_loss is not None and not (isinstance(test_loss, float) and test_loss != test_loss):
                task_scores[task_id].append(
                    TaskScore(hotkey=row["hotkey"], test_loss=test_loss, synth_loss=test_loss, quality_score=row["quality_score"])
                )

        return task_scores
