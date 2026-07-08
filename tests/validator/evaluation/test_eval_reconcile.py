import datetime

from validator.evaluation.reconcile import ActiveEvalRow
from validator.evaluation.reconcile import LiveDeployment
from validator.evaluation.reconcile import PvpPairReservation
from validator.evaluation.reconcile import _parse_created_at
from validator.evaluation.reconcile import plan_eval_reconcile


NOW = datetime.datetime(2026, 7, 8, 1, 0, 0, tzinfo=datetime.timezone.utc)
ORPHAN_GRACE = datetime.timedelta(seconds=1800)  # 30 min — long, protects the boot window
GHOST_GRACE = datetime.timedelta(seconds=240)  # 4 min — short, frees provably-dead deployments fast


def _ago(minutes: float) -> datetime.datetime:
    return NOW - datetime.timedelta(minutes=minutes)


def _plan(live=(), active=(), backed=frozenset(), pvp=()):
    return plan_eval_reconcile(
        list(live), list(active), set(backed), list(pvp), NOW, ORPHAN_GRACE, GHOST_GRACE
    )


# ---- orphan reaping (long grace) ----

def test_orphan_older_than_orphan_grace_is_reaped():
    plan = _plan(live=[LiveDeployment("dep-orphan", _ago(60))])
    assert plan.orphan_deployments == {"dep-orphan"}
    assert plan.ghost_deployment_ids == set()
    assert plan.ghost_pvp_pairs == ()


def test_fresh_orphan_within_orphan_grace_is_protected():
    # 5 min < 30 min orphan grace: a deployment mid-startup must not be reaped
    plan = _plan(live=[LiveDeployment("dep-fresh", _ago(5))])
    assert plan.orphan_deployments == set()


def test_orphan_uses_long_grace_not_short():
    # 10 min is past the 4-min ghost grace but within the 30-min orphan grace -> still protected
    plan = _plan(live=[LiveDeployment("dep", _ago(10))])
    assert plan.orphan_deployments == set()


def test_backed_by_evaluations_or_pvp_is_never_reaped():
    live = [LiveDeployment("dep-e", _ago(600)), LiveDeployment("dep-p", _ago(600))]
    active = [ActiveEvalRow("dep-e", _ago(500))]
    plan = _plan(live=live, active=active, backed={"dep-e", "dep-p"})
    assert plan.orphan_deployments == set()


# ---- evaluations ghost release (short grace) ----

def test_evaluations_ghost_released_after_ghost_grace():
    # deployment gone, reservation older than 4-min ghost grace -> released fast
    plan = _plan(active=[ActiveEvalRow("dep-dead", _ago(5))], backed={"dep-dead"})
    assert plan.ghost_deployment_ids == {"dep-dead"}


def test_evaluations_ghost_within_ghost_grace_protected():
    plan = _plan(active=[ActiveEvalRow("dep-dead", _ago(2))], backed={"dep-dead"})
    assert plan.ghost_deployment_ids == set()


# ---- PvP ghost release (short grace, pair-scoped) ----

def _pvp(dep_id, age, task="t1", a="hkA", b="hkB"):
    return PvpPairReservation(task_id=task, hotkey_a=a, hotkey_b=b, deployment_id=dep_id, updated_at=_ago(age))


def test_pvp_ghost_released_after_ghost_grace():
    # pair's deployment gone from live, reservation older than ghost grace -> released
    plan = _plan(pvp=[_pvp("pvp-dead", 5)])
    assert len(plan.ghost_pvp_pairs) == 1
    assert plan.ghost_pvp_pairs[0].deployment_id == "pvp-dead"


def test_pvp_reservation_with_live_deployment_not_ghost():
    plan = _plan(live=[LiveDeployment("pvp-live", _ago(60))], backed={"pvp-live"}, pvp=[_pvp("pvp-live", 60)])
    assert plan.ghost_pvp_pairs == ()
    assert plan.orphan_deployments == set()


def test_pvp_ghost_within_ghost_grace_protected():
    plan = _plan(pvp=[_pvp("pvp-dead", 2)])
    assert plan.ghost_pvp_pairs == ()


def test_pvp_reservation_without_deployment_id_is_not_a_ghost():
    # booting pair (deployment_id NULL) is handled by the long stale sweep, never the ghost path
    plan = _plan(pvp=[PvpPairReservation("t1", "hkA", "hkB", None, _ago(60))])
    assert plan.ghost_pvp_pairs == ()


# ---- combined ----

def test_mixed_scenario():
    live = [
        LiveDeployment("keep-eval", _ago(60)),   # backed -> keep
        LiveDeployment("keep-pvp", _ago(60)),    # backed pvp -> keep
        LiveDeployment("orphan-old", _ago(60)),  # unbacked + old -> reap
        LiveDeployment("orphan-young", _ago(2)),  # unbacked but fresh -> protected
    ]
    active = [
        ActiveEvalRow("keep-eval", _ago(10)),
        ActiveEvalRow("eval-ghost", _ago(10)),   # gone + past ghost grace -> reset
        ActiveEvalRow("eval-ghost-fresh", _ago(1)),  # gone but fresh -> protected
    ]
    pvp = [
        _pvp("keep-pvp", 60, task="tk"),         # live -> keep
        _pvp("pvp-ghost", 10, task="tg"),        # gone + past ghost grace -> release
    ]
    plan = _plan(live=live, active=active, backed={"keep-eval", "keep-pvp"}, pvp=pvp)
    assert plan.orphan_deployments == {"orphan-old"}
    assert plan.ghost_deployment_ids == {"eval-ghost"}
    assert {p.deployment_id for p in plan.ghost_pvp_pairs} == {"pvp-ghost"}


def test_boundary_exactly_at_grace_is_acted_on():
    assert _plan(live=[LiveDeployment("d", _ago(30))]).orphan_deployments == {"d"}
    assert _plan(pvp=[_pvp("x", 4)]).ghost_pvp_pairs != ()  # exactly ghost grace


def test_parse_created_at_iso_string_and_naive():
    aware = _parse_created_at("2026-07-08T01:07:00.252574+00:00")
    assert aware is not None and aware.tzinfo is not None
    naive = _parse_created_at("2026-07-08T01:07:00")
    assert naive is not None and naive.tzinfo == datetime.timezone.utc
    assert _parse_created_at("not-a-date") is None
    assert _parse_created_at(None) is None
