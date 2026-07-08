import datetime

from validator.evaluation.reconcile import ActiveEvalRow
from validator.evaluation.reconcile import LiveDeployment
from validator.evaluation.reconcile import _parse_created_at
from validator.evaluation.reconcile import plan_eval_reconcile


NOW = datetime.datetime(2026, 7, 8, 1, 0, 0, tzinfo=datetime.timezone.utc)
GRACE = datetime.timedelta(seconds=1800)  # 30 min


def _ago(minutes: int) -> datetime.datetime:
    return NOW - datetime.timedelta(minutes=minutes)


def test_orphan_older_than_grace_is_reaped():
    # live deployment, no active eval row references it, old enough -> reap
    plan = plan_eval_reconcile([LiveDeployment("dep-orphan", _ago(60))], [], NOW, GRACE)
    assert plan.orphan_deployments == {"dep-orphan"}
    assert plan.ghost_deployment_ids == set()


def test_fresh_orphan_within_grace_is_protected():
    # a just-created deployment (mid reserve->deploy->persist) must never be reaped
    plan = plan_eval_reconcile([LiveDeployment("dep-fresh", _ago(5))], [], NOW, GRACE)
    assert plan.orphan_deployments == set()


def test_backed_deployment_is_never_touched_regardless_of_age():
    live = [LiveDeployment("dep-1", _ago(600))]
    active = [ActiveEvalRow("dep-1", _ago(500))]
    plan = plan_eval_reconcile(live, active, NOW, GRACE)
    assert plan.orphan_deployments == set()
    assert plan.ghost_deployment_ids == set()


def test_ghost_reservation_older_than_grace_is_reset():
    # active eval row whose deployment is no longer live -> release/reset
    plan = plan_eval_reconcile([], [ActiveEvalRow("dep-dead", _ago(60))], NOW, GRACE)
    assert plan.ghost_deployment_ids == {"dep-dead"}
    assert plan.orphan_deployments == set()


def test_fresh_ghost_within_grace_is_protected():
    # deployment may simply be missing from a slightly stale list(); don't reset yet
    plan = plan_eval_reconcile([], [ActiveEvalRow("dep-dead", _ago(5))], NOW, GRACE)
    assert plan.ghost_deployment_ids == set()


def test_mixed_scenario():
    live = [
        LiveDeployment("keep", _ago(60)),          # backed -> keep
        LiveDeployment("orphan-old", _ago(60)),    # unbacked + old -> reap
        LiveDeployment("orphan-young", _ago(2)),   # unbacked but fresh -> protected
    ]
    active = [
        ActiveEvalRow("keep", _ago(10)),           # backed by live -> fine
        ActiveEvalRow("ghost-old", _ago(90)),      # deployment gone + stale -> reset
        ActiveEvalRow("ghost-young", _ago(1)),     # deployment gone but fresh -> protected
    ]
    plan = plan_eval_reconcile(live, active, NOW, GRACE)
    assert plan.orphan_deployments == {"orphan-old"}
    assert plan.ghost_deployment_ids == {"ghost-old"}


def test_boundary_exactly_at_grace_is_acted_on():
    # exactly grace old counts as eligible (>=)
    plan = plan_eval_reconcile([LiveDeployment("dep", _ago(30))], [], NOW, GRACE)
    assert plan.orphan_deployments == {"dep"}


def test_parse_created_at_iso_string_and_naive():
    aware = _parse_created_at("2026-07-08T01:07:00.252574+00:00")
    assert aware is not None and aware.tzinfo is not None
    naive = _parse_created_at("2026-07-08T01:07:00")
    assert naive is not None and naive.tzinfo == datetime.timezone.utc
    assert _parse_created_at("not-a-date") is None
    assert _parse_created_at(None) is None
