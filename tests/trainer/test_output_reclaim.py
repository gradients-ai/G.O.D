"""Output reclaim container spec and runner, against a fake docker client."""

import asyncio
import os
import threading

import pytest
from docker.errors import ImageNotFound

import trainer.constants as cst
from trainer import output_reclaim
from trainer.output_reclaim import SYMLINKED_OUTPUT_FAILURE
from trainer.output_reclaim import OutputReclaimError
from trainer.output_reclaim import reclaim_output_dir


HANG = "hang"


class FakeContainer:
    def __init__(self, name: str, exit_code: int | str, logs: bytes):
        self.name = name
        self.exit_code = exit_code
        self.log_bytes = logs
        self.killed = threading.Event()
        self.removed = False

    def wait(self) -> dict:
        if self.exit_code == HANG:
            self.killed.wait(timeout=5)
            return {"StatusCode": 137}
        return {"StatusCode": self.exit_code}

    def logs(self, stdout: bool = True, stderr: bool = True, tail: int | str = "all") -> bytes:
        return self.log_bytes

    def kill(self) -> None:
        self.killed.set()

    def remove(self, force: bool = False, v: bool = False) -> None:
        self.removed = True


class FakeClient:
    def __init__(self, exit_code: int | str | Exception = 0, logs: bytes = b""):
        self.exit_code = exit_code
        self.logs = logs
        self.kwargs: dict | None = None
        self.container: FakeContainer | None = None
        self.containers = self

    def run(self, **kwargs) -> FakeContainer:
        self.kwargs = kwargs
        if isinstance(self.exit_code, Exception):
            raise self.exit_code
        self.container = FakeContainer(kwargs["name"], self.exit_code, self.logs)
        return self.container


async def _reclaim(client: FakeClient, **kwargs) -> str | None:
    return await reclaim_output_dir(client, task_id="task-1", repo_name="repo-1", **kwargs)


async def test_spec_is_a_fixed_chown_with_only_the_needed_capabilities():
    client = FakeClient()

    assert await _reclaim(client) is None

    kwargs = client.kwargs
    assert kwargs["image"] == cst.HF_UPLOAD_DOCKER_IMAGE
    assert kwargs["user"] == "0"
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["cap_add"] == ["CHOWN", "DAC_READ_SEARCH", "FOWNER"]
    assert kwargs["security_opt"] == ["no-new-privileges"]
    assert kwargs["network_disabled"] is True
    assert kwargs["read_only"] is True
    assert kwargs["volumes"] == {cst.CHECKPOINTS_VOLUME_NAME: {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"}}
    assert kwargs["entrypoint"][:2] == ["/bin/sh", "-c"]
    assert kwargs["command"] == ["/app/checkpoints/task-1/repo-1", f"{os.geteuid()}:{os.getegid()}"]
    assert kwargs["name"].startswith("output-reclaim-")
    assert kwargs["name"].startswith(cst.TRAINER_CONTAINER_NAME_PREFIXES)
    assert client.container.removed


async def test_owner_and_source_can_be_overridden():
    client = FakeClient()

    await _reclaim(client, owner="1001:1001", checkpoints_source="/tmp/ck")

    assert client.kwargs["command"][1] == "1001:1001"
    assert client.kwargs["volumes"] == {"/tmp/ck": {"bind": cst.OUTPUT_CHECKPOINTS_PATH, "mode": "rw"}}


async def test_symlinked_output_is_a_rejection():
    client = FakeClient(exit_code=output_reclaim.SYMLINK_EXIT_CODE)

    assert await _reclaim(client) == SYMLINKED_OUTPUT_FAILURE
    assert client.container.removed


async def test_nonzero_exit_raises_with_the_log_tail():
    client = FakeClient(exit_code=1, logs=b"chown: changing ownership of 'x': Operation not permitted\n")

    with pytest.raises(OutputReclaimError, match="exited 1: chown: changing ownership"):
        await _reclaim(client)
    assert client.container.removed


async def test_container_error_raises():
    client = FakeClient(exit_code=ImageNotFound("no such image"))

    with pytest.raises(OutputReclaimError, match="output reclaim container error: no such image"):
        await _reclaim(client)


async def test_timeout_kills_and_raises(monkeypatch):
    monkeypatch.setattr(cst, "OUTPUT_RECLAIM_TIMEOUT_SECONDS", 0.05)
    client = FakeClient(exit_code=HANG)

    with pytest.raises(OutputReclaimError, match="timed out"):
        await _reclaim(client)
    assert client.container.killed.is_set() and client.container.removed
    await asyncio.sleep(0)
