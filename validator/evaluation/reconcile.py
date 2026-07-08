"""Eval-deployment reconciler.

The `EVAL_MAX_GPUS` cap is enforced by summing `gpu_count` reservations on the `evaluations`
table (see `try_reserve_evaluation_gpus`). That ledger only stays honest if every reservation
tracks a live Basilica deployment 1:1 — but reservations get released (gpu_count -> NULL) on
retries/resets/crashes without the deployment being torn down, so the ledger drifts below the
real GPU usage and the cap silently stops holding.

This module is the backstop: it treats the live Basilica deployments as the source of truth and
reconciles them against the reservation ledger each eval cycle.

  - orphan deployment: live on Basilica, but no active (pending/evaluating) eval row references
    it -> delete it (reclaim the GPUs it was silently holding).
  - ghost reservation: an active eval row whose deployment_id is no longer live -> reset the row
    to pending so its reservation is released and it can redeploy.

Both sides are age-gated by `EVAL_ORPHAN_GRACE_SECONDS` so a deployment that is mid-startup (in
the reserve -> deploy -> persist window) is never reaped, and a just-created deployment missing
from a slightly stale `list()` is never treated as a ghost.
"""

import asyncio
import datetime
from dataclasses import dataclass

import basilica

import validator.db.sql.tasks as tasks_sql
import validator.db.sql.tournaments as tournament_sql
import validator.evaluation.constants as vcst
from core.logging import get_logger
from validator.db.database import PSQLDB
from validator.evaluation.basilica_deployments import cleanup_basilica_deployments_by_name


logger = get_logger(__name__)


@dataclass(frozen=True)
class LiveDeployment:
    name: str
    created_at: datetime.datetime


@dataclass(frozen=True)
class ActiveEvalRow:
    deployment_id: str
    updated_at: datetime.datetime


@dataclass(frozen=True)
class ReconcilePlan:
    orphan_deployments: set[str]  # live deployments to delete (no backing active eval row)
    ghost_deployment_ids: set[str]  # active eval rows whose deployment is gone -> reset to pending


def plan_eval_reconcile(
    live: list[LiveDeployment],
    active: list[ActiveEvalRow],
    backed_ids: set[str],
    now: datetime.datetime,
    grace: datetime.timedelta,
) -> ReconcilePlan:
    """Pure decision function (no I/O) so the reconcile logic is unit-testable.

    `backed_ids` is every deployment id that backs live eval work, from BOTH the `evaluations`
    table (per-repo evals) AND `pvp_pair_results` (PvP evals track their live deployment id there,
    not in `evaluations`) — a live deployment is only an orphan if it is in neither. `active` are
    the `evaluations` rows carrying a deployment id, used for ghost detection (that table is the
    only one whose rows this reconciler resets).

    A deployment/row is only acted on once it is older/staler than `grace`, which protects the
    reserve -> deploy -> persist window and any Basilica `list()` staleness.
    """
    live_names = {dep.name for dep in live}
    orphans = {dep.name for dep in live if dep.name not in backed_ids and (now - dep.created_at) >= grace}
    ghosts = {
        row.deployment_id
        for row in active
        if row.deployment_id not in live_names and (now - row.updated_at) >= grace
    }
    return ReconcilePlan(orphan_deployments=orphans, ghost_deployment_ids=ghosts)


def _parse_created_at(value) -> datetime.datetime | None:
    if isinstance(value, datetime.datetime):
        return value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.timezone.utc)
    return None


async def reconcile_eval_deployments(psql_db: PSQLDB) -> ReconcilePlan | None:
    """Reconcile live Basilica deployments against the eval reservation ledger.

    Kills orphaned deployments and releases ghost reservations so the `EVAL_MAX_GPUS` accounting
    reflects reality. Best-effort: never raises to the caller (the eval loop must keep running).
    Returns the plan that was applied, or None if the live deployments could not be listed.
    """
    try:
        client = basilica.BasilicaClient()
        raw_deployments = await asyncio.to_thread(client.list)
    except Exception as e:
        logger.warning(f"eval reconcile: could not list Basilica deployments, skipping: {e}")
        return None

    live: list[LiveDeployment] = []
    for dep in raw_deployments:
        name = getattr(dep, "name", None)
        created_at = _parse_created_at(getattr(dep, "created_at", None))
        if name and created_at:
            live.append(LiveDeployment(name=name, created_at=created_at))

    active_rows = await tasks_sql.get_active_evaluation_deployments(psql_db)
    active = [
        ActiveEvalRow(deployment_id=row["deployment_id"], updated_at=row["updated_at"])
        for row in active_rows
        if row.get("deployment_id") and row.get("updated_at")
    ]

    # PvP evals record their live deployment id in pvp_pair_results, not evaluations, so a live
    # deployment is only orphaned if it backs neither an evaluations row nor an active PvP pair.
    pvp_backed_ids = await tournament_sql.get_active_pvp_deployment_ids(psql_db)
    backed_ids = {row.deployment_id for row in active} | pvp_backed_ids

    now = datetime.datetime.now(datetime.timezone.utc)
    grace = datetime.timedelta(seconds=vcst.EVAL_ORPHAN_GRACE_SECONDS)
    plan = plan_eval_reconcile(live, active, backed_ids, now, grace)

    if plan.orphan_deployments:
        logger.warning(
            f"eval reconcile: reaping {len(plan.orphan_deployments)} orphaned Basilica deployment(s) "
            f"with no active eval row: {sorted(plan.orphan_deployments)}"
        )
        await cleanup_basilica_deployments_by_name(plan.orphan_deployments)

    for deployment_id in plan.ghost_deployment_ids:
        logger.warning(
            f"eval reconcile: releasing ghost reservation for deployment {deployment_id} "
            f"(no longer live); resetting its eval rows to pending"
        )
        await tasks_sql.reset_evaluation_rows_for_deployment(deployment_id, psql_db)

    if not plan.orphan_deployments and not plan.ghost_deployment_ids:
        logger.debug(
            f"eval reconcile: healthy — {len(live)} live deployment(s), "
            f"{len(backed_ids)} backed id(s) (evals + pvp), nothing to reconcile"
        )
    return plan
