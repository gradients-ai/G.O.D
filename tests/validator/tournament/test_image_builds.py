import types
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import httpx
import pytest

from core.models.payload_models import ImageBuilderStatus
from core.models.payload_models import ImageBuildStatusResponse
from core.models.task_models import TaskType
from validator.tournament.image_builds import builders
from validator.tournament.image_builds import constants as cst
from validator.tournament.image_builds import gate
from validator.tournament.image_builds import shadeform
from validator.tournament.image_builds import worker
from validator.tournament.image_builds.models import ImageBuild
from validator.tournament.image_builds.models import ImageBuildStatus
from validator.tournament.image_builds.models import ImageBuildVariant
from validator.tournament.image_builds.policy import choose_builder
from validator.tournament.image_builds.policy import in_prune_window
from validator.tournament.image_builds.policy import next_attempt_at
from validator.tournament.image_builds.registry_prune import orphan_blob_keys
from validator.tournament.image_builds.registry_prune import referenced_digests_from_keys


NOW = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)  # a Monday
DIGEST = "sha256:" + "c" * 64


def _build(**overrides) -> ImageBuild:
    values = dict(
        build_id="00000000-0000-0000-0000-000000000001",
        tournament_id="tourn_x",
        hotkey="5Hotkey",
        variant=ImageBuildVariant.TEXT,
        repo="https://github.com/a/b",
        commit_ref="abc",
        repository="god/tourn-x/5hotkey-123-text",
        status=ImageBuildStatus.BUILDING,
        attempts=1,
        next_attempt_at=NOW,
        builder="1.2.3.4",
        lease_expires_at=NOW + timedelta(minutes=5),
        created_at=NOW,
        updated_at=NOW,
    )
    values.update(overrides)
    return ImageBuild(**values)


# --- policy -------------------------------------------------------------------------------------------------------


def test_retry_policy_by_failure_class():
    assert next_attempt_at("dockerfile", 1, NOW) is None
    assert next_attempt_at("clone", 1, NOW) is None
    assert next_attempt_at("infrastructure", 1, NOW) == NOW + timedelta(seconds=60)
    assert next_attempt_at("infrastructure", 3, NOW) == NOW + timedelta(seconds=600)
    assert next_attempt_at("infrastructure", 4, NOW) is None
    assert next_attempt_at("build", 1, NOW) is not None
    assert next_attempt_at("build", 2, NOW) is None
    assert next_attempt_at("stalled", 2, NOW) is None


def test_choose_builder_groups_tournament_and_avoids_previous_builder():
    free = {"a": 4, "b": 10, "c": 0}
    assert choose_builder(free, {}, avoid=None) == "b"
    assert choose_builder(free, {"a": 3}, avoid=None) == "a"  # keep a tournament on one BuildKit for cache sharing
    assert choose_builder(free, {"a": 3}, avoid="a") == "b"
    assert choose_builder({"a": 2}, {}, avoid="a") == "a"  # only option: retry on the same builder
    assert choose_builder({"a": 0}, {}, avoid=None) is None


def test_prune_window_is_friday_afternoon_through_sunday():
    friday = datetime(2026, 11, 20, tzinfo=timezone.utc)
    assert not in_prune_window(friday.replace(hour=14))
    assert in_prune_window(friday.replace(hour=15))
    assert in_prune_window(friday + timedelta(days=2))
    assert not in_prune_window(friday + timedelta(days=3))


def test_busy_trainers_get_no_build_slots_unless_allowed():
    status = ImageBuilderStatus(enabled=True, slots_total=16, free_slots=5, gpu_jobs_running=1)
    assert worker.builder_free_slots(status, is_trainer=True, allow_busy=False) == 0
    assert worker.builder_free_slots(status, is_trainer=True, allow_busy=True) == 5
    assert worker.builder_free_slots(status, is_trainer=False, allow_busy=False) == 5
    assert worker.builder_free_slots(ImageBuilderStatus(enabled=False, free_slots=5), True, True) == 0
    assert worker.builder_free_slots(None, True, True) == 0


# --- gate ---------------------------------------------------------------------------------------------------------


def _training(priority=2, task_type=TaskType.INSTRUCTTEXTTASK, model_type=None):
    task = types.SimpleNamespace(task_id="task-1", task_type=task_type, model_type=model_type)
    return types.SimpleNamespace(
        task=task, hotkey="5Hotkey", priority=priority, training_repo="https://github.com/a/b",
        training_commit_hash="abc", prepared_image=None,
    )


@pytest.fixture
def gate_env(monkeypatch):
    monkeypatch.setenv(cst.ENABLED_ENV, "true")
    state = {"build": None}

    async def tournament_id(task_id, db):
        return "tourn_x"

    async def ensure_build(**kwargs):
        return state["build"]

    monkeypatch.setattr(gate.tournament_sql, "get_tournament_id_by_task_id", tournament_id)
    monkeypatch.setattr(gate.image_build_sql, "ensure_build", ensure_build)
    return state


async def test_gate_attaches_ready_image(gate_env):
    gate_env["build"] = _build(status=ImageBuildStatus.READY, digest=DIGEST)
    [training] = await gate.attach_prepared_images([_training()], types.SimpleNamespace(psql_db=None))
    assert training.prepared_image.digest == DIGEST


async def test_gate_holds_back_until_ready_then_falls_back(gate_env, monkeypatch):
    gate_env["build"] = _build(status=ImageBuildStatus.QUEUED, created_at=datetime.now(timezone.utc))
    assert await gate.attach_prepared_images([_training()], types.SimpleNamespace(psql_db=None)) == []

    gate_env["build"] = _build(
        status=ImageBuildStatus.BUILDING,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=cst.MAX_WAIT_FOR_IMAGE_SECONDS + 1),
    )
    [training] = await gate.attach_prepared_images([_training()], types.SimpleNamespace(psql_db=None))
    assert training.prepared_image is None  # waited too long: trainer builds it

    gate_env["build"] = _build(status=ImageBuildStatus.FAILED)
    [training] = await gate.attach_prepared_images([_training()], types.SimpleNamespace(psql_db=None))
    assert training.prepared_image is None  # failed build never fails the miner


async def test_gate_ignores_non_tournament_and_disabled(gate_env, monkeypatch):
    organic = _training(priority=1)
    assert await gate.attach_prepared_images([organic], types.SimpleNamespace(psql_db=None)) == [organic]
    monkeypatch.setenv(cst.ENABLED_ENV, "false")
    held = _training()
    gate_env["build"] = _build(status=ImageBuildStatus.QUEUED, created_at=datetime.now(timezone.utc))
    assert await gate.attach_prepared_images([held], types.SimpleNamespace(psql_db=None)) == [held]


async def test_gate_lookup_errors_fall_back_to_trainer_build(gate_env, monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(gate.image_build_sql, "ensure_build", boom)
    training = _training()
    assert await gate.attach_prepared_images([training], types.SimpleNamespace(psql_db=None)) == [training]


def test_variant_for_task():
    from core.models.image_models import ImageModelType

    assert gate.variant_for_task(_training().task) == ImageBuildVariant.TEXT
    image = _training(task_type=TaskType.IMAGETASK, model_type=ImageModelType.Z_IMAGE).task
    flux = _training(task_type=TaskType.IMAGETASK, model_type=ImageModelType.FLUX).task
    assert gate.variant_for_task(image) == ImageBuildVariant.IMAGE_TOOLKIT
    assert gate.variant_for_task(flux) == ImageBuildVariant.IMAGE_FLUX


# --- worker: polling --------------------------------------------------------------------------------------------


class FakeSql:
    def __init__(self, building=(), queued=()):
        self.building = list(building)
        self.queued = list(queued)
        self.calls = []

    async def list_building(self, db):
        return self.building

    async def list_due_queued(self, limit, db):
        return self.queued[:limit]

    async def mark_ready(self, *args):
        self.calls.append(("ready", args[:3]))
        return True

    async def record_failure(self, build_id, attempt, failure_class, error, retry_at, db):
        self.calls.append(("failure", failure_class, retry_at is not None))
        return True

    async def renew_lease(self, *args):
        self.calls.append(("renew",))

    async def start_attempt(self, build_id, expected, address, lease, db):
        self.calls.append(("start", address))
        return True

    async def undo_unsent_attempt(self, *args):
        self.calls.append(("undo",))


@pytest.fixture
def config():
    return types.SimpleNamespace(psql_db=None)


def _status(state, **kwargs):
    return ImageBuildStatusResponse(build_id="b", attempt=1, state=state, started_at=datetime.now(timezone.utc), **kwargs)


@pytest.mark.parametrize(
    "poll_result,expected",
    [
        (_status("ready", digest=DIGEST, size_bytes=10), ("ready",)),
        (_status("failed", failure_class="infrastructure", error="dns"), ("failure", "infrastructure", True)),
        (_status("failed", failure_class="dockerfile", error="missing"), ("failure", "dockerfile", False)),
        (None, ("failure", "cancelled", True)),
        (_status("building"), ("renew",)),
        (_status("ready", digest=None), ("failure", "infrastructure", True)),
    ],
)
async def test_poll_in_flight_transitions(monkeypatch, config, poll_result, expected):
    sql = FakeSql(building=[_build()])
    monkeypatch.setattr(worker, "image_build_sql", sql)

    async def poll(address, build_id, attempt):
        return poll_result

    monkeypatch.setattr(worker.builders, "poll", poll)
    await worker.poll_in_flight(config)
    assert sql.calls[0][: len(expected)] == expected


async def test_unreachable_builder_keeps_lease_until_it_expires(monkeypatch, config):
    async def poll(address, build_id, attempt):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(worker.builders, "poll", poll)
    sql = FakeSql(building=[_build(lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5))])
    monkeypatch.setattr(worker, "image_build_sql", sql)
    load = await worker.poll_in_flight(config)
    assert sql.calls == [] and load == {"1.2.3.4": {"tourn_x": 1}}

    sql = FakeSql(building=[_build(lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))])
    monkeypatch.setattr(worker, "image_build_sql", sql)
    await worker.poll_in_flight(config)
    assert sql.calls == [("failure", "lease_expired", True)]


# --- worker: dispatch -------------------------------------------------------------------------------------------


@pytest.fixture
def dispatch_env(monkeypatch):
    statuses = {
        "10.0.0.1": ImageBuilderStatus(enabled=True, slots_total=16, free_slots=1, gpu_jobs_running=0),
        "10.0.0.2": ImageBuilderStatus(enabled=True, slots_total=16, free_slots=8, gpu_jobs_running=2),
    }

    async def get_trainers(db):
        return [types.SimpleNamespace(trainer_ip=ip) for ip in statuses]

    async def fetch_status(address):
        return statuses.get(address)

    async def repo_info(hotkey, tournament_id, db):
        return types.SimpleNamespace(training_repo="https://github.com/a/b", github_token="tok")

    monkeypatch.setattr(worker.tournament_sql, "get_trainers", get_trainers)
    monkeypatch.setattr(worker.tournament_sql, "get_tournament_training_repo_and_commit", repo_info)
    monkeypatch.setattr(worker.builders, "fetch_status", fetch_status)
    submitted = []

    async def submit(address, request):
        submitted.append((address, request))
        return _status("queued")

    monkeypatch.setattr(worker.builders, "submit", submit)
    return submitted


async def test_dispatch_uses_only_idle_trainers_and_passes_token(monkeypatch, config, dispatch_env):
    queued = [
        _build(status=ImageBuildStatus.QUEUED, attempts=0, builder=None, build_id=f"0000000{i}-0000-0000-0000-000000000000")
        for i in range(3)
    ]
    sql = FakeSql(queued=queued)
    monkeypatch.setattr(worker, "image_build_sql", sql)
    fallback = types.SimpleNamespace(addresses=[])
    free_after = await worker.dispatch(config, {}, fallback)
    # 10.0.0.2 runs GPU jobs, 10.0.0.1 has one slot: exactly one dispatch.
    assert [address for address, _ in dispatch_env] == ["10.0.0.1"]
    request = dispatch_env[0][1]
    assert request.attempt == 1 and request.github_token == "tok" and request.tag.endswith("-a1")
    assert free_after == 0


async def test_dispatch_undoes_attempt_when_builder_rejects(monkeypatch, config, dispatch_env):
    async def submit(address, request):
        raise builders.BuilderRejected("HTTP 503")

    monkeypatch.setattr(worker.builders, "submit", submit)
    sql = FakeSql(queued=[_build(status=ImageBuildStatus.QUEUED, attempts=0, builder=None)])
    monkeypatch.setattr(worker, "image_build_sql", sql)
    await worker.dispatch(config, {}, types.SimpleNamespace(addresses=[]))
    assert ("undo",) in sql.calls


async def test_dispatch_does_not_send_token_for_backup_repo(monkeypatch, config, dispatch_env):
    backup = _build(
        status=ImageBuildStatus.QUEUED, attempts=0, builder=None, repo="https://github.com/mirror/x", commit_ref="HEAD"
    )
    sql = FakeSql(queued=[backup])
    monkeypatch.setattr(worker, "image_build_sql", sql)
    await worker.dispatch(config, {}, types.SimpleNamespace(addresses=[]))
    request = dispatch_env[0][1]
    assert request.github_token is None and request.commit is None


# --- Shadeform fallback -----------------------------------------------------------------------------------------


def _type(cloud, name, price, gpus=0, vcpus=64, storage=2000, available=True):
    return {
        "cloud": cloud,
        "shade_instance_type": name,
        "hourly_price": price,
        "configuration": {"num_gpus": gpus, "vcpus": vcpus, "storage_in_gb": storage},
        "availability": [{"region": "us-east", "available": available, "rental_type": "on_demand"}],
    }


def test_pick_instance_type_prefers_cheap_cpu_only():
    types_ = [
        _type("a", "gpu-small", 50, gpus=1),
        _type("b", "cpu-big", 150),
        _type("c", "cpu-tiny", 20, vcpus=4),
        _type("d", "cpu-unavailable", 10, available=False),
    ]
    item, region = shadeform.pick_instance_type(types_, allow_gpu=False)
    assert item["shade_instance_type"] == "cpu-big" and region == "us-east"
    item, _ = shadeform.pick_instance_type([_type("a", "gpu-small", 50, gpus=1)], allow_gpu=True)
    assert item["shade_instance_type"] == "gpu-small"
    assert shadeform.pick_instance_type([_type("a", "gpu-small", 50, gpus=1)], allow_gpu=False) is None


class FakeShadeform:
    def __init__(self, instances=()):
        self.instances = list(instances)
        self.created = []
        self.deleted = []

    async def list_builders(self):
        return [i for i in self.instances if i.instance_id not in self.deleted]

    async def instance_types(self):
        return [_type("b", "cpu-big", 150)]

    async def create(self, body):
        self.created.append(body)
        return "new-1"

    async def delete(self, instance_id):
        self.deleted.append(instance_id)


async def test_fallback_created_only_when_starving_and_once(monkeypatch):
    monkeypatch.setenv(shadeform.ORCHESTRATOR_IPS_ENV, "1.1.1.1")
    client = FakeShadeform()
    manager = shadeform.FallbackBuilderManager(client=client)
    now = datetime.now(timezone.utc)
    await manager.tick(queued_due=5, building=0, oldest_queued=now, trainer_free_slots=0)
    assert client.created == []  # not waiting long enough yet
    await manager.tick(queued_due=5, building=0, oldest_queued=now - timedelta(minutes=11), trainer_free_slots=3)
    assert client.created == []  # trainers still have capacity
    old = now - timedelta(seconds=cst.SHADEFORM_FALLBACK_AFTER_SECONDS + 1)
    await manager.tick(queued_due=5, building=0, oldest_queued=old, trainer_free_slots=0)
    assert len(client.created) == 1
    body = client.created[0]
    assert body["tags"] == [cst.SHADEFORM_TAG] and body["auto_delete"]["spend_threshold"]
    assert all(env["name"] != "SHADEFORM_API_KEY" for env in body["envs"])
    client.instances.append(shadeform.FallbackInstance("new-1", now, "pending", None))
    await manager.tick(queued_due=5, building=0, oldest_queued=old, trainer_free_slots=0)
    assert len(client.created) == 1


async def test_fallback_deleted_when_idle_too_old_or_stuck():
    now = datetime.now(timezone.utc)
    client = FakeShadeform(
        [
            shadeform.FallbackInstance("old", now - timedelta(hours=cst.SHADEFORM_MAX_LIFETIME_HOURS + 1), "active", "1.1.1.1"),
            shadeform.FallbackInstance("stuck", now - timedelta(seconds=cst.SHADEFORM_BOOT_TIMEOUT_SECONDS + 1), "pending", None),
            shadeform.FallbackInstance("fresh", now, "active", "2.2.2.2"),
        ]
    )
    manager = shadeform.FallbackBuilderManager(client=client)
    await manager.tick(queued_due=1, building=1, oldest_queued=now, trainer_free_slots=0)
    assert set(client.deleted) == {"old", "stuck"}
    assert manager.addresses == ["2.2.2.2:8001"]
    manager._idle_since = now - timedelta(seconds=cst.SHADEFORM_IDLE_SECONDS + 1)
    await manager.tick(queued_due=0, building=0, oldest_queued=None, trainer_free_slots=0)
    assert "fresh" in client.deleted


# --- registry GC ------------------------------------------------------------------------------------------------


def test_gc_keeps_referenced_and_young_blobs():
    root = "registry/docker/registry/v2"
    a, b, c = "a" * 64, "b" * 64, "c" * 64
    links = [
        f"{root}/repositories/god/t1/m-text/_layers/sha256/{a}/link",
        f"{root}/repositories/god/t1/m-text/_manifests/revisions/sha256/{b}/link",
        f"{root}/repositories/god/t1/m-text/_manifests/tags/x/current/link",
    ]
    referenced = referenced_digests_from_keys(links)
    assert referenced == {f"sha256:{a}", f"sha256:{b}"}
    now = datetime.now(timezone.utc)
    blobs = [
        (f"{root}/blobs/sha256/aa/{a}/data", now - timedelta(days=3)),
        (f"{root}/blobs/sha256/cc/{c}/data", now - timedelta(days=3)),
        (f"{root}/blobs/sha256/cc/{c}x/data", now - timedelta(days=3)),
        (f"{root}/blobs/sha256/dd/{'d' * 64}/data", now - timedelta(hours=1)),
    ]
    assert orphan_blob_keys(blobs, referenced, now) == [f"{root}/blobs/sha256/cc/{c}/data"]
