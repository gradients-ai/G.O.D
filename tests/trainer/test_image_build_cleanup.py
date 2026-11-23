import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trainer import constants as cst
from trainer import image_builds
from trainer import runtime


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def docker_client(monkeypatch) -> MagicMock:
    """A fake Docker client, with the layer cache off so every cleanup prunes (the cache is in test_image_build_cache.py)."""
    monkeypatch.setattr(cst, "IMAGE_BUILD_NO_CACHE", True)
    client = MagicMock()
    client.api.build.return_value = iter([])
    monkeypatch.setattr(runtime.docker, "from_env", MagicMock(return_value=client))
    return client


@pytest.mark.parametrize("prune, prunes_builds", [(True, True), (False, False)])
def test_cleanup_prunes_host_build_cache_only_when_configured(monkeypatch, docker_client, prune, prunes_builds):
    monkeypatch.setattr(cst, "PRUNE_HOST_BUILD_CACHE", prune)

    runtime.delete_image_and_cleanup("standalone-text-trainer:abc")

    docker_client.images.remove.assert_called_once_with(image="standalone-text-trainer:abc", force=True, noprune=False)
    docker_client.images.prune.assert_called_once_with(filters={"dangling": True})
    assert docker_client.api.prune_builds.called is prunes_builds


def _start_blocked_build(docker_client: MagicMock) -> tuple[threading.Thread, threading.Event]:
    """Start a build on another thread that stays inside the docker build call until the returned event is set."""
    building, release = threading.Event(), threading.Event()

    def build(**kwargs):
        building.set()
        release.wait(timeout=10)
        return iter([])

    docker_client.api.build.side_effect = build
    thread = threading.Thread(target=runtime.build_docker_image, args=("Dockerfile",), kwargs={"tag": "t:other"})
    thread.start()
    assert building.wait(timeout=10)
    return thread, release


def test_failed_job_cleanup_leaves_a_build_in_progress_alone(monkeypatch, docker_client):
    monkeypatch.setattr(cst, "PRUNE_HOST_BUILD_CACHE", True)
    thread, release = _start_blocked_build(docker_client)
    try:
        runtime.delete_image_and_cleanup("standalone-text-trainer:failed")

        docker_client.images.remove.assert_called_once_with(image="standalone-text-trainer:failed", force=True, noprune=False)
        docker_client.images.prune.assert_not_called()
        docker_client.api.prune_builds.assert_not_called()
    finally:
        release.set()
        thread.join(timeout=10)

    runtime.delete_image_and_cleanup("standalone-text-trainer:failed")

    docker_client.images.prune.assert_called_once_with(filters={"dangling": True})
    docker_client.api.prune_builds.assert_called_once()


def test_build_abandoned_on_timeout_counts_until_its_stream_ends(monkeypatch, docker_client):
    monkeypatch.setattr(cst, "PRUNE_HOST_BUILD_CACHE", True)
    release, stream_done = threading.Event(), threading.Event()

    def stream():
        release.wait(timeout=10)
        stream_done.set()
        yield from ()

    docker_client.api.build.return_value = stream()
    tag, error = runtime.build_docker_image("Dockerfile", tag="t:slow", timeout_seconds=0.1)
    assert tag is None and "exceeded" in error

    runtime.delete_image_and_cleanup("t:other")
    docker_client.api.prune_builds.assert_not_called()

    release.set()
    assert stream_done.wait(timeout=10)
    for _ in range(100):
        runtime.delete_image_and_cleanup("t:other")
        if docker_client.api.prune_builds.called:
            break
        threading.Event().wait(0.05)
    docker_client.api.prune_builds.assert_called_once()


def test_build_waits_for_a_prune_in_progress(monkeypatch, docker_client):
    monkeypatch.setattr(cst, "PRUNE_HOST_BUILD_CACHE", True)
    pruning, release = threading.Event(), threading.Event()
    build_started_during_prune = []

    def prune_builds():
        pruning.set()
        release.wait(timeout=10)
        build_started_during_prune.append(docker_client.api.build.called)

    docker_client.api.prune_builds.side_effect = prune_builds
    cleanup = threading.Thread(target=runtime.delete_image_and_cleanup, args=("t:done",))
    cleanup.start()
    assert pruning.wait(timeout=10)
    build = threading.Thread(target=runtime.build_docker_image, args=("Dockerfile",), kwargs={"tag": "t:next"})
    build.start()
    build.join(timeout=0.5)
    release.set()
    cleanup.join(timeout=10)
    build.join(timeout=10)

    assert build_started_during_prune == [False]
    docker_client.api.build.assert_called_once()


def test_build_with_the_cache_off_uses_no_cache_and_removes_its_intermediate_containers(docker_client):
    tag, error = runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    assert (tag, error) == ("t:1", None)
    kwargs = docker_client.api.build.call_args.kwargs
    assert kwargs["nocache"] is True
    assert kwargs["rm"] is True and kwargs["forcerm"] is True


def test_build_whose_step_failed_returns_an_error(docker_client):
    """A failed RUN step arrives as an {"error": ...} stream chunk with no exception; the missing image fails the build."""
    docker_client.api.build.return_value = iter(
        [{"stream": "Step 2/2 : RUN false\n"}, {"error": "The command '/bin/sh -c false' returned a non-zero code: 1"}]
    )
    docker_client.images.get.side_effect = runtime.ImageNotFound("No such image: t:1")

    tag, error = runtime.build_docker_image("Dockerfile", context_path="/tmp/ctx", tag="t:1")

    assert tag is None
    assert "t:1" in error and "build step failed" in error
    docker_client.images.get.assert_called_once_with("t:1")


@pytest.mark.parametrize("value, expected", [(None, True), ("true", True), ("false", False), ("FALSE", False)])
def test_prune_flag_from_env(value, expected):
    env = {k: v for k, v in os.environ.items() if k != "TRAINER_PRUNE_BUILD_CACHE"}
    if value is not None:
        env["TRAINER_PRUNE_BUILD_CACHE"] = value
    code = "from trainer import constants; print(constants.PRUNE_HOST_BUILD_CACHE)"
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == str(expected)


STEP_A = "0123456789ab" + "0" * 52
STEP_B = "ba9876543210" + "1" * 52
TRAINING_CONTAINER = "feedfacecafe" + "2" * 52


def _api_error(status_code: int, message: str) -> runtime.APIError:
    return runtime.APIError(message, response=MagicMock(status_code=status_code))


class FakeDaemon:
    """The daemon's containers behind a build: a step container is created, named in the stream, and only then started."""

    def __init__(self, docker_client: MagicMock):
        self._lock = threading.Lock()
        self.status: dict[str, str] = {}
        self.names: dict[str, str] = {}
        self.kill_attempts: list[str] = []
        self.step_ended = threading.Event()
        docker_client.api.containers.side_effect = self._list
        docker_client.api.kill.side_effect = self._kill
        docker_client.api.inspect_container.side_effect = self._inspect

    def create(self, container_id: str, name: str = "quirky_step") -> None:
        with self._lock:
            self.status[container_id] = "created"
            self.names[container_id] = name

    def start(self, container_id: str) -> None:
        with self._lock:
            self.status[container_id] = "running"

    def remove(self, container_id: str) -> None:
        with self._lock:
            del self.status[container_id]

    def _find(self, container_id: str) -> str:
        matches = [full for full in self.status if full.startswith(container_id)]
        if not matches:
            raise runtime.NotFound(f"No such container: {container_id}")
        return matches[0]

    def _list(self, all: bool = False) -> list[dict]:
        with self._lock:
            return [{"Id": full, "Names": [f"/{self.names[full]}"], "State": state} for full, state in self.status.items()]

    def _kill(self, container_id: str) -> None:
        with self._lock:
            self.kill_attempts.append(container_id)
            full = self._find(container_id)
            if self.status[full] != "running":
                raise _api_error(409, f"Container {full} is not running")
            self.status[full] = "exited"
            self.step_ended.set()

    def _inspect(self, container_id: str) -> dict:
        with self._lock:
            return {"State": {"Status": self.status[self._find(container_id)]}}


@pytest.fixture
def daemon(docker_client, monkeypatch) -> FakeDaemon:
    monkeypatch.setattr(cst, "BUILD_STOP_KILL_RETRY_SECONDS", 0.02)
    return FakeDaemon(docker_client)


class BlockingBuildStream:
    """A build stream that yields `before`, then hangs like a step that never ends until released, then runs `after`:
    dicts are yielded, callables are called (the daemon doing something between two lines)."""

    def __init__(self, before: list, after: list = ()):
        self.reached_hang, self.release, self.done = threading.Event(), threading.Event(), threading.Event()
        self._before, self._after = list(before), list(after)

    def __iter__(self):
        yield from self._play(self._before)
        self.reached_hang.set()
        self.release.wait(timeout=10)
        yield from self._play(self._after)
        self.done.set()

    @staticmethod
    def _play(items: list):
        for item in items:
            if callable(item):
                item()
            else:
                yield item


def _step(n: int) -> dict:
    return {"stream": f"Step {n}/9 : RUN pip install x\n"}


def _running_in(container_id: str) -> dict:
    return {"stream": f" ---> Running in {container_id[:12]}\n"}


def _run_step(daemon: FakeDaemon, n: int, container_id: str) -> list:
    """What the daemon does for one RUN step: create the container, name it in the stream, start it."""
    return [_step(n), lambda: daemon.create(container_id), _running_in(container_id), lambda: daemon.start(container_id)]


def test_timed_out_build_is_stopped_by_killing_its_step_container(docker_client, daemon):
    stream = BlockingBuildStream(_run_step(daemon, 2, STEP_A))
    docker_client.api.build.return_value = iter(stream)

    tag, error = runtime.build_docker_image("Dockerfile", tag="t:slow", timeout_seconds=0.2)
    stream.release.set()

    assert tag is None and "exceeded" in error
    assert daemon.status[STEP_A] == "exited"
    # Stopping a build removes nothing.
    docker_client.images.prune.assert_not_called()
    docker_client.images.remove.assert_not_called()
    docker_client.api.prune_builds.assert_not_called()


@pytest.mark.parametrize(
    "removal_line", ["Removing intermediate container {id}\n", " ---> Removed intermediate container {id}\n"]
)
def test_step_started_after_the_build_was_stopped_is_killed_once_it_starts(docker_client, daemon, removal_line):
    """The stream names a step container before the daemon starts it; a kill that early is refused and must be retried."""
    started_late = threading.Timer(0.3, daemon.start, args=(STEP_B,))
    stream = BlockingBuildStream(
        [
            *_run_step(daemon, 2, STEP_A),
            lambda: daemon.remove(STEP_A),
            {"stream": removal_line.format(id=STEP_A[:12])},
        ],
        after=[_step(3), lambda: daemon.create(STEP_B), started_late.start, _running_in(STEP_B)],
    )
    docker_client.api.build.return_value = iter(stream)

    tag, error = runtime.build_docker_image("Dockerfile", tag="t:slow", timeout_seconds=0.2)

    assert tag is None and "exceeded" in error
    assert daemon.kill_attempts == []  # the finished step's container is gone already

    stream.release.set()
    assert daemon.step_ended.wait(timeout=10), "the step started after the stop ran on"
    assert stream.done.wait(timeout=10)
    assert daemon.status[STEP_B] == "exited"
    assert len(daemon.kill_attempts) > 1  # refused while the container was only created


def test_step_container_that_never_starts_is_given_up_on(docker_client, daemon, monkeypatch):
    monkeypatch.setattr(cst, "BUILD_STOP_KILL_TIMEOUT_SECONDS", 0.1)
    stream = BlockingBuildStream([_step(2), lambda: daemon.create(STEP_A), _running_in(STEP_A)])
    docker_client.api.build.return_value = iter(stream)

    tag, error = runtime.build_docker_image("Dockerfile", tag="t:slow", timeout_seconds=0.2)
    stream.release.set()

    assert tag is None and "exceeded" in error
    assert daemon.status[STEP_A] == "created"
    assert len(daemon.kill_attempts) > 1


def test_finished_build_kills_nothing(docker_client, daemon):
    docker_client.api.build.return_value = iter(
        [*BlockingBuildStream._play(_run_step(daemon, 2, STEP_A)), {"stream": "Successfully built\n"}]
    )

    assert runtime.build_docker_image("Dockerfile", tag="t:1") == ("t:1", None)
    assert daemon.kill_attempts == []


def test_a_step_cannot_hide_its_container_by_printing_that_it_was_removed(docker_client, daemon):
    stream = BlockingBuildStream([*_run_step(daemon, 2, STEP_A), {"stream": f"Removing intermediate container {STEP_A[:12]}\n"}])
    docker_client.api.build.return_value = iter(stream)

    tag, error = runtime.build_docker_image("Dockerfile", tag="t:slow", timeout_seconds=0.2)
    stream.release.set()

    assert tag is None and "exceeded" in error
    assert daemon.status[STEP_A] == "exited"


def test_container_lines_in_step_output_are_not_followed(docker_client, daemon):
    """Only the first container line after a step line is the daemon's; a step echoing one does not get it killed."""
    daemon.create(STEP_B)
    daemon.start(STEP_B)
    stream = BlockingBuildStream([*_run_step(daemon, 2, STEP_A), _running_in(STEP_B), {"stream": "x ---> Running in y\n"}])
    docker_client.api.build.return_value = iter(stream)

    runtime.build_docker_image("Dockerfile", tag="t:slow", timeout_seconds=0.2)
    stream.release.set()

    assert daemon.status[STEP_A] == "exited"
    assert daemon.status[STEP_B] == "running"


def test_a_trainer_container_named_in_the_stream_is_never_killed(docker_client, daemon):
    daemon.create(TRAINING_CONTAINER, name="text-trainer-abc")
    daemon.start(TRAINING_CONTAINER)
    stream = BlockingBuildStream([_step(2), _running_in(TRAINING_CONTAINER)])
    docker_client.api.build.return_value = iter(stream)

    runtime.build_docker_image("Dockerfile", tag="t:slow", timeout_seconds=0.2)
    stream.release.set()

    assert daemon.status[TRAINING_CONTAINER] == "running"
    assert daemon.kill_attempts == []


async def test_cancelled_job_stops_its_build(docker_client, daemon):
    stream = BlockingBuildStream(_run_step(daemon, 2, STEP_A))
    docker_client.api.build.return_value = iter(stream)

    build = asyncio.create_task(runtime.build_job_image("task-1", "hk", dockerfile_path="Dockerfile", tag="t:1"))
    assert await asyncio.to_thread(stream.reached_hang.wait, 10)
    build.cancel()
    with pytest.raises(asyncio.CancelledError):
        await build

    assert await asyncio.to_thread(daemon.step_ended.wait, 10)
    stream.release.set()
    assert daemon.status[STEP_A] == "exited"


def test_build_gives_up_when_a_build_cleanup_never_finishes(monkeypatch, docker_client):
    """The wait for a cleanup in progress is before the build timer, so it has a limit of its own."""
    monkeypatch.setattr(cst, "BUILD_CLEANUP_WAIT_TIMEOUT_SECONDS", 0.1)

    with image_builds._lock:  # a cleanup that is still pruning
        tag, error = runtime.build_docker_image("Dockerfile", tag="t:1")

    assert tag is None and "not started" in error
    docker_client.api.build.assert_not_called()
    assert runtime.build_docker_image("Dockerfile", tag="t:1") == ("t:1", None)
