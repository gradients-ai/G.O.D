"""Read-only data access for the monitor.

Reuses the existing `validator.db.sql` helpers wherever possible and falls back
to a handful of raw aggregate queries (run through `PSQLDB`) for the few views
that span tables the helpers don't expose directly.
"""

from uuid import UUID

from core.models.task_models import TaskType
from core.models.tournament_models import TournamentType
from validator.db.sql import submissions_and_scoring as scoring_sql
from validator.db.sql import tasks as tasks_sql
from validator.db.sql import tournaments as tournaments_sql


# text/img/env -> DB tournament_type value
TYPE_ALIASES = {
    "text": TournamentType.TEXT,
    "image": TournamentType.IMAGE,
    "img": TournamentType.IMAGE,
    "env": TournamentType.ENVIRONMENT,
    "environment": TournamentType.ENVIRONMENT,
}

ACTIVE_EVAL_STATUSES = ["pending", "evaluating"]


def _as_uuid(task_id) -> UUID:
    return task_id if isinstance(task_id, UUID) else UUID(str(task_id))


class Queries:
    """Thin async facade over the DB used by every screen."""

    def __init__(self, psql_db):
        self.db = psql_db

    # --- tournaments -------------------------------------------------------

    async def list_tournaments(
        self, statuses: list[str] | None = None, types: list[str] | None = None
    ) -> list[dict]:
        clauses = []
        params: list = []
        if statuses:
            params.append(statuses)
            clauses.append(f"status = ANY(${len(params)})")
        if types:
            params.append([TYPE_ALIASES[t].value if t in TYPE_ALIASES else t for t in types])
            clauses.append(f"tournament_type = ANY(${len(params)})")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT tournament_id, tournament_type, status, created_at, updated_at,
                   winner_hotkey, base_winner_hotkey
            FROM tournaments
            {where}
            ORDER BY created_at DESC
        """
        return await self.db.fetchall(query, *params)

    async def get_tournament(self, tournament_id: str):
        return await tournaments_sql.get_tournament(tournament_id, self.db)

    async def active_tournament(self, type_alias: str):
        return await tournaments_sql.get_active_tournament(self.db, TYPE_ALIASES[type_alias])

    async def latest_completed_tournament(self, type_alias: str):
        return await tournaments_sql.get_latest_completed_tournament(self.db, TYPE_ALIASES[type_alias])

    async def rounds(self, tournament_id: str):
        return await tournaments_sql.get_tournament_rounds(tournament_id, self.db)

    async def participants(self, tournament_id: str):
        return await tournaments_sql.get_tournament_participants(tournament_id, self.db)

    async def full_results(self, tournament_id: str):
        return await tournaments_sql.get_tournament_full_results(tournament_id, self.db)

    async def tournament_tasks(self, tournament_id: str) -> list[dict]:
        query = """
            SELECT t.task_id, t.task_type, t.status, t.created_at, t.started_at,
                   t.completed_at, t.hours_to_complete,
                   tt.round_id, tt.group_id, tt.pair_id
            FROM tasks t
            JOIN tournament_tasks tt ON t.task_id = tt.task_id
            WHERE tt.tournament_id = $1
            ORDER BY t.created_at
        """
        return await self.db.fetchall(query, tournament_id)

    async def tournament_training(self, tournament_id: str) -> list[dict]:
        query = """
            SELECT ttht.task_id, ttht.hotkey, ttht.training_status,
                   ttht.n_training_attempts, ttht.updated_at,
                   tn.expected_repo_name, s.repo AS submission_repo
            FROM tournament_task_hotkey_trainings ttht
            JOIN tournament_tasks tt ON ttht.task_id = tt.task_id
            LEFT JOIN task_nodes tn ON ttht.task_id = tn.task_id AND ttht.hotkey = tn.hotkey
            LEFT JOIN submissions s ON ttht.task_id = s.task_id AND ttht.hotkey = s.hotkey
            WHERE tt.tournament_id = $1
            ORDER BY ttht.task_id, ttht.hotkey
        """
        return await self.db.fetchall(query, tournament_id)

    async def synced_tasks(self, tournament_id: str) -> list[dict]:
        query = """
            SELECT brst.tournament_task_id, brst.general_task_id,
                   t1.status AS tournament_task_status,
                   t2.status AS general_task_status
            FROM boss_round_synced_tasks brst
            JOIN tasks t1 ON brst.tournament_task_id = t1.task_id
            JOIN tasks t2 ON brst.general_task_id = t2.task_id
            JOIN tournament_tasks tt ON t1.task_id = tt.task_id
            WHERE tt.tournament_id = $1
        """
        return await self.db.fetchall(query, tournament_id)

    # --- tasks -------------------------------------------------------------

    async def tournament_id_for_task(self, task_id: str) -> str | None:
        return await tournaments_sql.get_tournament_id_by_task_id(task_id, self.db)

    async def task(self, task_id: str):
        return await tasks_sql.get_task(_as_uuid(task_id), self.db)

    async def task_scores(self, task_id: str) -> list[dict]:
        return await scoring_sql.get_all_scores_and_losses_for_task(_as_uuid(task_id), self.db)

    async def task_winner(self, task_id: str) -> str | None:
        return await scoring_sql.get_task_winner(_as_uuid(task_id), self.db)

    async def task_evaluations(self, task_id: str) -> list[dict]:
        return await tasks_sql.get_task_evaluation_rows(_as_uuid(task_id), self.db)

    async def task_training(self, task_id: str) -> dict[str, str]:
        return await tournaments_sql.get_training_status_for_task(task_id, self.db)

    # --- pvp ---------------------------------------------------------------

    async def pvp_pairs(self, task_id: str):
        return await tournaments_sql.get_pvp_pair_results(task_id, self.db)

    async def pvp_individual_scores(self, task_id: str):
        return await tournaments_sql.get_individual_scores(task_id, self.db)

    # --- deployments / infra ----------------------------------------------

    async def active_deployments(self) -> list[dict]:
        query = """
            SELECT e.task_id, e.hotkey, e.evaluation_status, e.deployment_id,
                   e.gpu_count, t.task_type
            FROM evaluations e
            JOIN tasks t ON e.task_id = t.task_id
            WHERE e.deployment_id IS NOT NULL
              AND e.evaluation_status = ANY($1)
              AND t.task_type != $2
            ORDER BY e.evaluation_status, t.task_type, e.task_id
        """
        return await self.db.fetchall(query, ACTIVE_EVAL_STATUSES, TaskType.ENVIRONMENTTASK.value)

    async def pvp_deployments(self) -> list[dict]:
        query = """
            SELECT task_id, hotkey_a, hotkey_b, environment_name, status, deployment_id
            FROM pvp_pair_results
            WHERE deployment_id IS NOT NULL AND status != 'complete'
            ORDER BY task_id, environment_name
        """
        return await self.db.fetchall(query)

    async def individual_deployments(self) -> list[dict]:
        query = """
            SELECT task_id, hotkey, environment_name, status, deployment_id
            FROM pvp_individual_scores
            WHERE deployment_id IS NOT NULL AND status != 'complete'
            ORDER BY task_id, environment_name, hotkey
        """
        return await self.db.fetchall(query)

    async def trainers(self):
        return await tournaments_sql.get_trainers(self.db)

    async def training_stats(self) -> dict:
        return await tournaments_sql.get_tournament_training_stats(self.db)
