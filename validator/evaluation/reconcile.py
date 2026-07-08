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
class PvpPairReservation:
    task_id: str
    hotkey_a: str
    hotkey_b: str
    deployment_id: str | None
    updated_at: datetime.datetime


@dataclass(frozen=True)
class ReconcilePlan:
    orphan_deployments: set[str]  # live deployments to delete (no backing active eval row)
    ghost_deployment_ids: set[str]  # active eval rows whose deployment is gone -> reset to pending
    ghost_pvp_pairs: tuple[PvpPairReservation, ...]  # pvp reservations whose deployment is gone -> release


def plan_eval_reconcile(
    live: list[LiveDeployment],
    active: list[ActiveEvalRow],
    backed_ids: set[str],
    pvp_reservations: list[PvpPairReservation],
    now: datetime.datetime,
    orphan_grace: datetime.timedelta,
    ghost_grace: datetime.timedelta,
) -> ReconcilePlan:
    """Pure decision function (no I/O) so the reconcile logic is unit-testable.

    `backed_ids` is every deployment id that backs live eval work, from BOTH the `evaluations`
    table (per-repo evals) AND `pvp_pair_results` (PvP evals track their live deployment id there,
    not in `evaluations`) — a live deployment is only an ORPHAN if it is in neither.

    Two directions, two graces:
    - ORPHAN reaping (live deployment with no backing) and boot-window stale-release use the LONG
      `orphan_grace` — a deployment may legitimately be mid-startup (reserved but not yet backing).
    - GHOST release (a reservation whose deployment is provably absent from a fresh `list()`) uses
      the SHORT `ghost_grace`: a reservation only carries a deployment id once it was stamped
      post-readiness, so a still-booting eval is never a ghost, and a dead deployment's GPUs should
      be freed fast rather than pinning the cap for the full orphan grace.

    `active` are the `evaluations` rows carrying a deployment id (individual evals); `pvp_reservations`
    are the per-pair reservations on `pvp_pair_results` (PvP evals).
    """
    live_names = {dep.name for dep in live}
    orphans = {dep.name for dep in live if dep.name not in backed_ids and (now - dep.created_at) >= orphan_grace}
    ghosts = {
        row.deployment_id
        for row in active
        if row.deployment_id not in live_names and (now - row.updated_at) >= ghost_grace
    }
    ghost_pvp = tuple(
        r
        for r in pvp_reservations
        if r.deployment_id and r.deployment_id not in live_names and (now - r.updated_at) >= ghost_grace
    )
    return ReconcilePlan(orphan_deployments=orphans, ghost_deployment_ids=ghosts, ghost_pvp_pairs=ghost_pvp)


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

    # PvP evals track their live deployment id + GPU reservation on pvp_pair_results (per-pair), not
    # evaluations. A live deployment is only orphaned if it backs neither an evaluations row nor an
    # active PvP pair; and PvP reservations get their own ghost/stale handling below.
    pvp_backed_ids = await tournament_sql.get_active_pvp_deployment_ids(psql_db)
    backed_ids = {row.deployment_id for row in active} | pvp_backed_ids

    pvp_reservation_rows = await tournament_sql.get_active_pvp_pair_reservations(psql_db)
    pvp_reservations = [
        PvpPairReservation(
            task_id=str(row["task_id"]),
            hotkey_a=row["hotkey_a"],
            hotkey_b=row["hotkey_b"],
            deployment_id=row.get("deployment_id"),
            updated_at=row["updated_at"],
        )
        for row in pvp_reservation_rows
        if row.get("updated_at")
    ]

    now = datetime.datetime.now(datetime.timezone.utc)
    orphan_grace = datetime.timedelta(seconds=vcst.EVAL_ORPHAN_GRACE_SECONDS)
    ghost_grace = datetime.timedelta(seconds=vcst.EVAL_GHOST_GRACE_SECONDS)
    plan = plan_eval_reconcile(live, active, backed_ids, pvp_reservations, now, orphan_grace, ghost_grace)

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

    for pair in plan.ghost_pvp_pairs:
        logger.warning(
            f"eval reconcile: releasing ghost PvP reservation for pair "
            f"{pair.task_id[:8]} {pair.hotkey_a[:8]}:{pair.hotkey_b[:8]} "
            f"(deployment {pair.deployment_id} no longer live)"
        )
        await tasks_sql.release_pvp_pair_gpus(pair.task_id, pair.hotkey_a, pair.hotkey_b, psql_db)

    # Release reservations that hold GPUs but never got a deployment_id stamped (deploy crashed
    # before persist) and have aged past the long orphan grace — invisible to the deployment-based
    # reconcile above, so both tables get an explicit stale sweep.
    released = await tasks_sql.release_stale_unreconcilable_reservations(vcst.EVAL_ORPHAN_GRACE_SECONDS, psql_db)
    released_pvp = await tournament_sql.release_stale_pvp_pair_reservations(vcst.EVAL_ORPHAN_GRACE_SECONDS, psql_db)
    if released or released_pvp:
        logger.warning(
            f"eval reconcile: released {released} evaluations + {released_pvp} PvP stale GPU "
            f"reservation(s) with no deployment_id (older than {vcst.EVAL_ORPHAN_GRACE_SECONDS}s)"
        )

    if not plan.orphan_deployments and not plan.ghost_deployment_ids and not plan.ghost_pvp_pairs:
        logger.debug(
            f"eval reconcile: healthy — {len(live)} live deployment(s), {len(backed_ids)} backed id(s), "
            f"{len(pvp_reservations)} pvp reservation(s), nothing to reconcile"
        )
    return plan
