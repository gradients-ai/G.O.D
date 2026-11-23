import os
import subprocess
import sys
import threading
from collections import namedtuple
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from trainer import constants as cst
from trainer import image_builds
from trainer import runtime
from trainer.image_builds import BuildCacheState


REPO_ROOT = Path(__file__).resolve().parents[2]
DiskUsage = namedtuple("DiskUsage", "total used free")


@pytest.fixture
def docker_client(monkeypatch) -> MagicMock:
    """A fake Docker client on a half-full disk, with the cache on."""
    client = MagicMock()
    client.api.build.return_value = iter([])
    client.info.return_value = {"DockerRootDir": "/var/lib/docker"}
    monkeypatch.setattr(runtime.docker, "from_env", MagicMock(return_value=client))
    monkeypatch.setattr(cst, "IMAGE_BUILD_NO_CACHE", False)
    monkeypatch.setattr(cst, "IMAGE_BUILD_PULL_BASE", False)
    monkeypatch.setattr(cst, "PRUNE_HOST_BUILD_CACHE", True)
    monkeypatch.setattr(cst, "IMAGE_BUILD_CACHE_MAX_AGE_HOURS", 24.0)
    monkeypatch.setattr(cst, "IMAGE_BUILD_CACHE_MAX_DISK_USED_PERCENT", 80.0)
    _set_disk_used(monkeypatch, 50)
    return client


def _set_disk_used(monkeypatch, percent: int) -> None:
    monkeypatch.setattr(image_builds.shutil, "disk_usage", lambda path: DiskUsage(100, percent, 100 - percent))


def _set_emptied(hours_ago: float) -> None:
    emptied_at = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    Path(cst.IMAGE_BUILD_CACHE_STATE_PATH).write_text(BuildCacheState(emptied_at=emptied_at).model_dump_json())


def _emptied_hours_ago() -> float:
    state = BuildCacheState.model_validate_json(Path(cst.IMAGE_BUILD_CACHE_STATE_PATH).read_text())
    return (datetime.now(timezone.utc) - state.emptied_at).total_seconds() / 3600


def test_build_reuses_the_layer_cache_by_default(docker_client):
    tag, error = runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    assert (tag, error) == ("t:1", None)
    kwargs = docker_client.api.build.call_args.kwargs
    assert kwargs["nocache"] is False
    assert kwargs["pull"] is False
    assert kwargs["rm"] is True and kwargs["forcerm"] is True


def test_build_uses_no_cache_when_the_cache_is_off(monkeypatch, docker_client):
    monkeypatch.setattr(cst, "IMAGE_BUILD_NO_CACHE", True)

    runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    assert docker_client.api.build.call_args.kwargs["nocache"] is True


def test_build_argument_overrides_the_configured_cache_use(docker_client):
    runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1", no_cache=True)

    assert docker_client.api.build.call_args.kwargs["nocache"] is True


def test_build_pulls_the_base_image_when_configured(monkeypatch, docker_client):
    monkeypatch.setattr(cst, "IMAGE_BUILD_PULL_BASE", True)

    runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    assert docker_client.api.build.call_args.kwargs["pull"] is True


def test_job_cleanup_keeps_the_layers_of_a_recent_cache(docker_client):
    _set_emptied(hours_ago=1)

    runtime.delete_image_and_cleanup("standalone-text-trainer:abc")

    docker_client.images.remove.assert_called_once_with(image="standalone-text-trainer:abc", force=True, noprune=True)
    docker_client.images.prune.assert_not_called()
    docker_client.api.prune_builds.assert_not_called()
    assert _emptied_hours_ago() == pytest.approx(1, abs=0.01)


def test_cache_with_no_record_counts_as_new(docker_client):
    image_builds.prune_build_leftovers()

    docker_client.images.prune.assert_not_called()
    assert _emptied_hours_ago() == pytest.approx(0, abs=0.01)


def test_cache_past_its_age_is_emptied_and_its_age_restarts(docker_client):
    _set_emptied(hours_ago=25)

    image_builds.prune_build_leftovers()

    docker_client.images.prune.assert_called_once_with(filters={"dangling": True})
    docker_client.api.prune_builds.assert_called_once_with()
    assert _emptied_hours_ago() == pytest.approx(0, abs=0.01)

    image_builds.prune_build_leftovers()

    docker_client.images.prune.assert_called_once()


@pytest.mark.parametrize("used_percent, emptied", [(79, False), (80, True), (95, True)])
def test_cache_is_emptied_when_the_docker_disk_is_full(monkeypatch, docker_client, used_percent, emptied):
    _set_emptied(hours_ago=1)
    _set_disk_used(monkeypatch, used_percent)

    image_builds.prune_build_leftovers()

    assert docker_client.images.prune.called is emptied
    assert docker_client.api.prune_builds.called is emptied


def test_unreadable_docker_disk_falls_back_to_the_age_rule(monkeypatch, docker_client):
    def no_such_dir(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(image_builds.shutil, "disk_usage", no_such_dir)
    _set_emptied(hours_ago=1)
    image_builds.prune_build_leftovers()
    docker_client.images.prune.assert_not_called()

    _set_emptied(hours_ago=25)
    image_builds.prune_build_leftovers()
    docker_client.images.prune.assert_called_once()


def test_emptying_the_cache_leaves_the_host_build_cache_when_configured(monkeypatch, docker_client):
    monkeypatch.setattr(cst, "PRUNE_HOST_BUILD_CACHE", False)
    _set_emptied(hours_ago=25)

    image_builds.prune_build_leftovers()

    docker_client.images.prune.assert_called_once_with(filters={"dangling": True})
    docker_client.api.prune_builds.assert_not_called()


def test_cache_off_removes_the_image_layers_and_prunes_after_every_job(monkeypatch, docker_client):
    monkeypatch.setattr(cst, "IMAGE_BUILD_NO_CACHE", True)
    _set_emptied(hours_ago=1)

    runtime.delete_image_and_cleanup("standalone-text-trainer:abc")

    docker_client.images.remove.assert_called_once_with(image="standalone-text-trainer:abc", force=True, noprune=False)
    docker_client.images.prune.assert_called_once_with(filters={"dangling": True})


def _cached_build_stream(layer_id: str = "0123456789ab") -> list[dict]:
    return [
        {"stream": "Step 1/3 : FROM base\n"},
        {"stream": " ---> aaaaaaaaaaaa\n"},
        {"stream": "Step 2/3 : RUN pip install things\n"},
        {"stream": " ---> Using cache\n"},
        {"stream": f" ---> {layer_id}\n"},
        {"stream": "Step 3/3 : RUN pip install more\n"},
        {"stream": " ---> Using cache\n"},
        {"stream": " ---> bbbbbbbbbbbb\n"},
        {"stream": "Successfully built bbbbbbbbbbbb\n"},
    ]


def _set_layer_age(docker_client: MagicMock, hours: float) -> None:
    created = datetime.now(timezone.utc) - timedelta(hours=hours)
    docker_client.api.inspect_image.return_value = {"Created": created.strftime("%Y-%m-%dT%H:%M:%S.%f000Z")}


def _layer_ages(docker_client: MagicMock, hours_by_layer: dict[str, float]) -> None:
    def inspect_image(layer_id: str) -> dict:
        created = datetime.now(timezone.utc) - timedelta(hours=hours_by_layer[layer_id])
        return {"Created": created.strftime("%Y-%m-%dT%H:%M:%S.%f000Z")}

    docker_client.api.inspect_image.side_effect = inspect_image


def _record_builds(docker_client: MagicMock, streams: list) -> list:
    """Serve the given build streams in order, and return the list that the build and tag-removal calls are added to."""
    events = []

    def build(**kwargs):
        events.append(("build", kwargs["nocache"]))
        return streams.pop(0)

    docker_client.api.build.side_effect = build
    docker_client.images.remove.side_effect = lambda **kwargs: events.append(("remove", kwargs))
    return events


REMOVED_STALE_TAG = ("remove", {"image": "t:1", "force": True, "noprune": True})


def test_build_that_reuses_recent_layers_runs_once(docker_client):
    docker_client.api.build.return_value = iter(_cached_build_stream())
    _set_layer_age(docker_client, hours=23)

    tag, error = runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    assert (tag, error) == ("t:1", None)
    docker_client.api.build.assert_called_once()
    assert [call.args for call in docker_client.api.inspect_image.call_args_list] == [("0123456789ab",), ("bbbbbbbbbbbb",)]
    docker_client.images.remove.assert_not_called()


def test_build_that_reuses_a_too_old_layer_runs_again_without_cache(docker_client):
    """The stale build's tag is removed before the second build, which differs only in using no cache."""
    events = _record_builds(docker_client, [iter(_cached_build_stream()), iter([{"stream": "Step 1/3 : FROM base\n"}])])
    _set_layer_age(docker_client, hours=25)

    tag, error = runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    assert (tag, error) == ("t:1", None)
    assert events == [("build", False), REMOVED_STALE_TAG, ("build", True)]
    docker_client.api.inspect_image.assert_called_once_with("0123456789ab")
    first, second = docker_client.api.build.call_args_list
    assert {k: v for k, v in first.kwargs.items() if k != "nocache"} == {k: v for k, v in second.kwargs.items() if k != "nocache"}


def test_too_old_layer_in_a_later_stage_is_found(docker_client):
    """Each FROM starts a new stage, whose reused layers can be older than the ones before."""
    two_stages = [
        {"stream": "Step 1/4 : FROM base AS builder\n"},
        {"stream": " ---> aaaaaaaaaaaa\n"},
        {"stream": "Step 2/4 : RUN make\n"},
        {"stream": " ---> Using cache\n"},
        {"stream": " ---> 111111111111\n"},
        {"stream": "Step 3/4 : FROM base\n"},
        {"stream": " ---> aaaaaaaaaaaa\n"},
        {"stream": "Step 4/4 : RUN pip install things\n"},
        {"stream": " ---> Using cache\n"},
        {"stream": " ---> 222222222222\n"},
    ]
    events = _record_builds(docker_client, [iter(two_stages), iter([])])
    _layer_ages(docker_client, {"111111111111": 1, "222222222222": 25})

    tag, error = runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    assert (tag, error) == ("t:1", None)
    assert events == [("build", False), REMOVED_STALE_TAG, ("build", True)]


def _wait_for_no_build_in_flight() -> None:
    for _ in range(200):
        if image_builds._builds_in_flight == 0:
            return
        threading.Event().wait(0.05)
    raise AssertionError("the build stream thread is still running")


@pytest.mark.parametrize("gave_up_by", ["cancel", "timeout"])
def test_build_given_up_while_the_stale_build_ends_is_not_run_again(docker_client, gave_up_by):
    draining, release = threading.Event(), threading.Event()

    def stale_stream():
        yield from _cached_build_stream()[:5]
        draining.set()
        release.wait(timeout=10)

    events = _record_builds(docker_client, [stale_stream(), iter([])])
    _set_layer_age(docker_client, hours=25)
    cancelled = threading.Event()
    result = []

    def run():
        timeout_seconds = 30 if gave_up_by == "cancel" else 0.2
        result.append(runtime.build_docker_image("Dockerfile", tag="t:1", timeout_seconds=timeout_seconds, cancelled=cancelled))

    build = threading.Thread(target=run)
    build.start()
    assert draining.wait(timeout=10)
    if gave_up_by == "cancel":
        cancelled.set()
    build.join(timeout=10)
    assert result[0][0] is None
    assert ("cancelled" if gave_up_by == "cancel" else "exceeded") in result[0][1]

    release.set()
    _wait_for_no_build_in_flight()

    assert events == [("build", False), REMOVED_STALE_TAG]


def test_stale_image_that_cannot_be_removed_fails_the_build(docker_client):
    docker_client.api.build.return_value = iter(_cached_build_stream())
    docker_client.images.remove.side_effect = runtime.APIError("image is being used")
    _set_layer_age(docker_client, hours=25)

    tag, error = runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    assert tag is None
    assert "could not remove" in error and "image is being used" in error
    docker_client.api.build.assert_called_once()
    docker_client.images.get.assert_not_called()


def test_build_that_cannot_be_started_reports_the_reason(docker_client):
    docker_client.api.build.side_effect = runtime.APIError("Cannot locate specified Dockerfile")

    tag, error = runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    assert tag is None
    assert "Cannot locate specified Dockerfile" in error


def test_second_build_that_cannot_be_started_reports_the_reason(docker_client):
    streams = [iter(_cached_build_stream())]

    def build(**kwargs):
        if not streams:
            raise runtime.APIError("daemon went away")
        return streams.pop(0)

    docker_client.api.build.side_effect = build
    _set_layer_age(docker_client, hours=25)

    tag, error = runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    assert tag is None
    assert "daemon went away" in error


def test_build_with_the_cache_off_never_looks_at_layer_ages(monkeypatch, docker_client):
    monkeypatch.setattr(cst, "IMAGE_BUILD_NO_CACHE", True)
    docker_client.api.build.return_value = iter(_cached_build_stream())

    runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    docker_client.api.build.assert_called_once()
    docker_client.api.inspect_image.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        runtime.APIError("no such image"),
        requests.exceptions.ReadTimeout("read timed out"),
        requests.exceptions.ConnectionError("connection aborted"),
        RuntimeError("anything else"),
    ],
)
def test_layer_whose_age_cannot_be_read_is_kept(docker_client, failure):
    docker_client.api.build.return_value = iter(_cached_build_stream())
    docker_client.api.inspect_image.side_effect = failure

    tag, error = runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    assert (tag, error) == ("t:1", None)
    docker_client.api.build.assert_called_once()
    assert docker_client.api.inspect_image.call_count == 2


@pytest.mark.parametrize(
    "texts, expected",
    [
        ([" ---> Using cache\n", " ---> 0123456789ab\n"], ["0123456789ab"]),
        ([" ---> Using", " cache\n ---> sha256:" + "a" * 64 + "\nStep 3/3"], ["sha256:" + "a" * 64]),
        ([" ---> Using cache\n ---> 111111111111\n ---> Using cache\n ---> 222222222222\n"], ["111111111111", "222222222222"]),
        ([" ---> Running in 0123456789ab\n", " ---> 0123456789ab\n"], []),
        ([" ---> Using cache\n", "output\n", " ---> 0123456789ab\n"], []),
    ],
)
def test_reused_layers_are_read_from_the_stream(texts, expected):
    reused = image_builds.ReusedLayers()
    assert [layer for text in texts for layer in reused.feed({"stream": text})] == expected


def test_every_reused_layer_is_reported():
    reused = image_builds.ReusedLayers()
    assert [layer for chunk in _cached_build_stream() for layer in reused.feed(chunk)] == ["0123456789ab", "bbbbbbbbbbbb"]


def test_disk_still_full_after_emptying_is_logged_as_a_warning(monkeypatch, docker_client):
    warnings = []
    monkeypatch.setattr(image_builds.logger, "warning", warnings.append)
    _set_emptied(hours_ago=1)
    _set_disk_used(monkeypatch, 90)

    image_builds.prune_build_leftovers()

    docker_client.images.prune.assert_called_once()
    assert len(warnings) == 1 and "still 90% full" in warnings[0]


def test_disk_under_the_limit_after_emptying_is_not_a_warning(monkeypatch, docker_client):
    warnings = []
    monkeypatch.setattr(image_builds.logger, "warning", warnings.append)
    _set_emptied(hours_ago=1)
    readings = iter([DiskUsage(100, 90, 10), DiskUsage(100, 40, 60)])
    monkeypatch.setattr(image_builds.shutil, "disk_usage", lambda path: next(readings))

    image_builds.prune_build_leftovers()

    docker_client.images.prune.assert_called_once()
    assert warnings == []


def test_unwritable_state_file_keeps_the_cache_age_in_memory(monkeypatch, docker_client, tmp_path):
    monkeypatch.setattr(cst, "IMAGE_BUILD_CACHE_STATE_PATH", str(tmp_path / "missing_dir" / "state.json"))

    image_builds.prune_build_leftovers()
    docker_client.images.prune.assert_not_called()

    remembered = image_builds._emptied_at_in_memory
    monkeypatch.setattr(image_builds, "_emptied_at_in_memory", remembered - timedelta(hours=25))
    image_builds.prune_build_leftovers()

    docker_client.images.prune.assert_called_once()
    assert image_builds._emptied_at_in_memory > remembered - timedelta(hours=1)


def test_state_file_is_replaced_whole(docker_client):
    _set_emptied(hours_ago=25)

    image_builds.prune_build_leftovers()

    state_path = Path(cst.IMAGE_BUILD_CACHE_STATE_PATH)
    assert [path.name for path in state_path.parent.iterdir()] == [state_path.name]
    assert _emptied_hours_ago() == pytest.approx(0, abs=0.01)


def test_state_time_without_a_zone_is_read_as_utc(docker_client):
    naive = (datetime.now(timezone.utc) - timedelta(hours=25)).replace(tzinfo=None)
    Path(cst.IMAGE_BUILD_CACHE_STATE_PATH).write_text(f'{{"emptied_at": "{naive.isoformat()}"}}')

    image_builds.prune_build_leftovers()

    docker_client.images.prune.assert_called_once()


def test_state_file_is_never_the_real_one():
    assert Path(cst.IMAGE_BUILD_CACHE_STATE_PATH).resolve().parent != (REPO_ROOT / "trainer").resolve()


def _constants_from_env(overrides: dict[str, str], names: list[str]) -> list[str]:
    dropped = ("TRAINER_IMAGE_BUILD_", "TRAINER_DOCKER_BUILD_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(dropped)} | overrides
    code = f"from trainer import constants as c; print(*[getattr(c, n) for n in {names!r}], sep=',')"
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1].split(",")


@pytest.mark.parametrize("value, expected", [(None, "False"), ("false", "False"), ("true", "True"), ("TRUE", "True")])
def test_no_cache_flag_from_env(value, expected):
    overrides = {} if value is None else {"TRAINER_IMAGE_BUILD_NO_CACHE": value}
    assert _constants_from_env(overrides, ["IMAGE_BUILD_NO_CACHE"]) == [expected]


def test_build_timeout_default_and_override():
    names = ["DOCKER_BUILD_TIMEOUT_MINUTES", "STALE_TASK_GRACE_MINUTES"]
    assert _constants_from_env({}, names) == ["45", "55"]
    assert _constants_from_env({"TRAINER_DOCKER_BUILD_TIMEOUT_MINUTES": "60"}, names) == ["60", "70"]
