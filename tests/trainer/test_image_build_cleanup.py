import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trainer import constants as cst
from trainer import runtime


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def docker_client(monkeypatch) -> MagicMock:
    client = MagicMock()
    client.api.build.return_value = iter([])
    monkeypatch.setattr(runtime.docker, "from_env", MagicMock(return_value=client))
    return client


@pytest.mark.parametrize("prune, prunes_builds", [(True, True), (False, False)])
def test_cleanup_prunes_host_build_cache_only_when_configured(monkeypatch, docker_client, prune, prunes_builds):
    monkeypatch.setattr(cst, "PRUNE_HOST_BUILD_CACHE", prune)

    runtime.delete_image_and_cleanup("standalone-text-trainer:abc")

    docker_client.images.remove.assert_called_once_with(image="standalone-text-trainer:abc", force=True)
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

        docker_client.images.remove.assert_called_once_with(image="standalone-text-trainer:failed", force=True)
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


def test_build_keeps_no_cache_and_removes_its_intermediate_containers(docker_client):
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
