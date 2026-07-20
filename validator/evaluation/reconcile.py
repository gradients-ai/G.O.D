"""Eval-deployment reconciler.

The `EVAL_MAX_GPUS` cap is enforced by summing `gpu_count` reservations on the `evaluations`
table (see `try_reserve_evaluation_gpus`). That ledger only stays honest if every reservation
tracks a live Basilica deployment 1:1 — but reservations get released (gpu_count -> NULL) on
retries/resets/crashes without the deployment being torn down, so the ledger drifts below the
real GPU usage and the cap silently stops holding.

This module is the backstop: it treats the live Basilica deployments as the source of truth and
reconciles them against the reservation ledger each eval cycle.

  - orphan deployment: validator-owned and live on Basilica, but no active evaluation,
    environment, individual, or PvP owner references it -> delete it.
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
from validator.db.sql import gpu_costs
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
    verified: bool
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
    managed_deployment_names: set[str] | None = None,
) -> ReconcilePlan:
    """Pure decision function (no I/O) so the reconcile logic is unit-testable.

    `backed_ids` is every deployment id that backs live eval work across
    `evaluations`, `pvp_individual_scores`, and `pvp_pair_results`. A deployment
    is only an orphan if no owner references it.

    Two directions, two graces:
    - ORPHAN reaping (live deployment with no backing) and boot-window stale-release use the LONG
      `orphan_grace` — a deployment may legitimately be mid-startup (reserved but not yet backing).
    - GHOST release (a reservation whose deployment is provably absent from a fresh `list()`) uses
      grace by verification state:
      - verified deployment ids (post-readiness stamp) use SHORT `ghost_grace`, so dead deployments
        free GPU reservations quickly.
      - unverified deployment ids (pre-deploy boot marker) use LONG `orphan_grace`, so a booting
        deployment is never falsely ghosted before the ready window expires.

    `active` are the `evaluations` rows carrying a deployment id (individual evals); `pvp_reservations`
    are the per-pair reservations on `pvp_pair_results` (PvP evals).
    """
    live_names = {dep.name for dep in live}
    managed_deployment_names = live_names if managed_deployment_names is None else managed_deployment_names
    orphans = {
        dep.name
        for dep in live
        if dep.name in managed_deployment_names
        and dep.name not in backed_ids
        and (now - dep.created_at) >= orphan_grace
    }
    ghosts = {
        row.deployment_id
        for row in active
        if row.deployment_id not in live_names and (now - row.updated_at) >= ghost_grace
    }
    ghost_pvp = tuple(
        r
        for r in pvp_reservations
        if r.deployment_id
        and r.deployment_id not in live_names
        and (now - r.updated_at) >= (ghost_grace if r.verified else orphan_grace)
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

    observed_at = datetime.datetime.now(datetime.timezone.utc)
    live: list[LiveDeployment] = []
    managed_deployment_names: set[str] = set()
    for dep in raw_deployments:
        name = getattr(dep, "name", None)
        created_at = _parse_created_at(getattr(dep, "created_at", None))
        if name:
            # Missing API metadata must not make a visibly live deployment look
            # like a DB ghost. Treat its age as unknown/fresh for orphan reaping.
            live.append(LiveDeployment(name=name, created_at=created_at or observed_at))
            friendly_name = getattr(dep, "friendly_name", None)
            if name.startswith(vcst.EVAL_BASILICA_DEPLOYMENT_NAME_PREFIX) or (
                isinstance(friendly_name, str)
                and friendly_name.startswith(vcst.EVAL_BASILICA_DEPLOYMENT_NAME_PREFIX)
            ):
                managed_deployment_names.add(name)

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
    individual_backed_ids = await tournament_sql.get_active_individual_deployment_ids(psql_db)
    backed_ids = (
        {row.deployment_id for row in active}
        | pvp_backed_ids
        | individual_backed_ids
    )

    pvp_reservation_rows = await tournament_sql.get_active_pvp_pair_reservations(psql_db)
    pvp_reservations = [
        PvpPairReservation(
            task_id=str(row["task_id"]),
            hotkey_a=row["hotkey_a"],
            hotkey_b=row["hotkey_b"],
            deployment_id=row.get("deployment_id"),
            verified=bool(row.get("deployment_verified")),
            updated_at=row["updated_at"],
        )
        for row in pvp_reservation_rows
        if row.get("updated_at")
    ]

    now = datetime.datetime.now(datetime.timezone.utc)
    orphan_grace = datetime.timedelta(seconds=vcst.EVAL_ORPHAN_GRACE_SECONDS)
    ghost_grace = datetime.timedelta(seconds=vcst.EVAL_GHOST_GRACE_SECONDS)
    plan = plan_eval_reconcile(
        live,
        active,
        backed_ids,
        pvp_reservations,
        now,
        orphan_grace,
        ghost_grace,
        managed_deployment_names,
    )
    try:
        closed_cost_runs = await gpu_costs.close_stale_evaluation_runs(
            live_deployment_names={deployment.name for deployment in live},
            older_than=now - orphan_grace,
            psql_db=psql_db,
        )
        if closed_cost_runs:
            logger.warning(f"eval reconcile: closed {closed_cost_runs} stale evaluation cost run(s)")
    except Exception as e:
        logger.warning(f"eval reconcile: could not close stale evaluation cost runs: {e}")

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
