"""Decoder sandbox container specs and the phase runner, against a fake docker client."""

import math
import os
import threading
from pathlib import Path

import pytest
from docker.errors import ImageNotFound
from pydantic import ValidationError

import trainer.constants as cst
from core.constants.quant import DECODE_CPUS
from core.constants.quant import DECODE_PIDS_LIMIT
from trainer.quant_decode import sandbox
from trainer.quant_decode.sandbox import DecodeContainerSpec
from trainer.quant_decode.sandbox import DecodePhase
from trainer.quant_decode.sandbox import build_phase_spec
from trainer.quant_decode.sandbox import run_phase
from trainer.quant_decode.sandbox import run_phase_spec


GB = 1024**3


class FakeContainer:
    def __init__(self, exit_code: int = 0, stderr: bytes = b"", hang: bool = False):
        self.name = "fake"
        self.exit_code = exit_code
        self.stderr = stderr
        self.killed = threading.Event()
        self.hang = hang
        self.removed_with: dict | None = None
        self.logs_tail: int | None = None

    def wait(self) -> dict:
        if self.hang:
            self.killed.wait(timeout=5)
            return {"StatusCode": 137}
        return {"StatusCode": self.exit_code}

    def kill(self) -> None:
        self.killed.set()

    def logs(self, stdout: bool = True, stderr: bool = True, tail: int | str = "all") -> bytes:
        assert stdout is False and stderr is True
        self.logs_tail = tail
        lines = self.stderr.splitlines(keepends=True)
        return b"".join(lines if tail == "all" else lines[-tail:])

    def remove(self, force: bool = False, v: bool = False) -> None:
        self.removed_with = {"force": force, "v": v}


class FakeContainers:
    def __init__(self, container: FakeContainer):
        self.container = container
        self.run_kwargs: dict | None = None

    def run(self, **kwargs) -> FakeContainer:
        self.run_kwargs = kwargs
        return self.container


class FakeClient:
    def __init__(self, container: FakeContainer):
        self.containers = FakeContainers(container)


def _run_spec(tmp_path: Path, bf16_model_bytes: int = 16 * GB) -> DecodeContainerSpec:
    return run_phase_spec(tmp_path / "artifact", tmp_path / "decoder", tmp_path / "output", bf16_model_bytes)


def _assert_sandboxed(kwargs: dict) -> None:
    assert "device_requests" not in kwargs
    assert "network" not in kwargs
    assert kwargs["network_disabled"] is True
    assert kwargs["read_only"] is True
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges"]
    assert kwargs["environment"] == {"HOME": "/tmp"}
    assert kwargs["user"] == f"{os.geteuid()}:{os.getegid()}"
    assert kwargs["user"] != "0:0" or os.geteuid() == 0
    assert kwargs["pids_limit"] == DECODE_PIDS_LIMIT
    assert kwargs["nano_cpus"] == DECODE_CPUS * 1e9
    assert kwargs["tmpfs"] == {"/tmp": "size=1g"}


def test_build_spec_mounts_and_limits(tmp_path: Path):
    spec = build_phase_spec(tmp_path / "src", tmp_path / "build")
    kwargs = spec.docker_run_kwargs()

    assert kwargs["image"] == cst.QUANT_DECODER_BUILD_IMAGE
    assert kwargs["volumes"] == {
        str(tmp_path / "src"): {"bind": "/decoder", "mode": "ro"},
        str(tmp_path / "build"): {"bind": "/build", "mode": "rw"},
    }
    assert kwargs["mem_limit"] == f"{cst.QUANT_DECODE_MIN_MEMORY_GB}g"
    assert kwargs["command"] == ["true"]
    _assert_sandboxed(kwargs)


def test_build_spec_runs_build_script_when_present(tmp_path: Path):
    (tmp_path / "build.sh").write_text("gcc -O2 -o /build/dec dec.c\n")
    spec = build_phase_spec(tmp_path, tmp_path / "build")
    assert spec.command == ["/bin/sh", "/decoder/build.sh"]


def test_run_spec_mounts_and_limits(tmp_path: Path):
    kwargs = _run_spec(tmp_path).docker_run_kwargs()

    assert kwargs["image"] == cst.QUANT_DECODER_RUN_IMAGE
    assert kwargs["volumes"] == {
        str(tmp_path / "artifact"): {"bind": "/artifact", "mode": "ro"},
        str(tmp_path / "decoder"): {"bind": "/decoder", "mode": "ro"},
        str(tmp_path / "output"): {"bind": "/output", "mode": "rw"},
    }
    assert kwargs["command"] == ["python", "/decoder/decode.py", "/artifact", "/output"]
    _assert_sandboxed(kwargs)


def test_run_spec_mounts_build_dir_read_only(tmp_path: Path):
    spec = run_phase_spec(tmp_path / "artifact", tmp_path / "decoder", tmp_path / "output", GB, build_dir=tmp_path / "build")
    assert spec.volumes[str(tmp_path / "build")] == {"bind": "/build", "mode": "ro"}
    assert len(spec.volumes) == 4


def test_run_spec_omits_build_mount_without_build_dir(tmp_path: Path):
    binds = {mount["bind"] for mount in _run_spec(tmp_path).volumes.values()}
    assert "/build" not in binds


def test_run_spec_memory_scales_above_floor(tmp_path: Path):
    bf16_bytes = 16 * GB + 1
    assert _run_spec(tmp_path, bf16_bytes).mem_limit == str(math.ceil(2.5 * bf16_bytes))


def test_run_spec_memory_floor(tmp_path: Path):
    assert _run_spec(tmp_path, GB).mem_limit == str(cst.QUANT_DECODE_MIN_MEMORY_GB * GB)


def test_run_spec_rejects_non_positive_size(tmp_path: Path):
    with pytest.raises(ValueError):
        _run_spec(tmp_path, 0)


@pytest.mark.parametrize("key", ["HF_TOKEN", "WANDB_API_KEY", "aws_secret", "MY_KEY"])
def test_spec_rejects_credential_env(tmp_path: Path, key: str):
    spec = _run_spec(tmp_path).model_dump()
    with pytest.raises(ValidationError):
        DecodeContainerSpec(**{**spec, "environment": {key: "x"}})


async def test_run_phase_exit_zero_ok(tmp_path: Path):
    container = FakeContainer(exit_code=0)
    client = FakeClient(container)

    outcome = await run_phase(client, _run_spec(tmp_path))

    assert outcome.ok
    assert outcome.phase == DecodePhase.RUN
    assert outcome.exit_code == 0
    assert outcome.failure is None
    assert client.containers.run_kwargs["detach"] is True
    assert client.containers.run_kwargs["name"].startswith("quant-decode-run-")
    assert container.removed_with == {"force": True, "v": True}


async def test_run_phase_nonzero_exit_reports_last_stderr_line(tmp_path: Path):
    container = FakeContainer(exit_code=3, stderr=b"loading\nValueError: bad \x1b[31mheader\n\n")

    outcome = await run_phase(FakeClient(container), _run_spec(tmp_path))

    assert not outcome.ok
    assert outcome.exit_code == 3
    assert outcome.failure.startswith("exit 3")
    assert outcome.failure == "exit 3: ValueError: bad [31mheader"
    assert container.removed_with == {"force": True, "v": True}


async def test_run_phase_timeout_kills(tmp_path: Path):
    container = FakeContainer(hang=True)
    spec = DecodeContainerSpec(**{**_run_spec(tmp_path).model_dump(), "timeout_seconds": 0.05})

    outcome = await run_phase(FakeClient(container), spec)

    assert not outcome.ok
    assert outcome.failure == "timeout"
    assert container.killed.is_set()
    assert container.removed_with == {"force": True, "v": True}


async def test_run_phase_docker_error_is_reported(tmp_path: Path):
    class FailingContainers:
        def run(self, **kwargs):
            raise ImageNotFound("no such image")

    client = FakeClient(FakeContainer())
    client.containers = FailingContainers()

    outcome = await run_phase(client, _run_spec(tmp_path))

    assert not outcome.ok
    assert outcome.failure == "container error: no such image"


async def test_run_phase_reads_stderr_tail_and_caps_failure(tmp_path: Path):
    stderr = b"".join(f"line {i}\n".encode() for i in range(10_000)) + b"E" * 5000 + b"\n"
    container = FakeContainer(exit_code=1, stderr=stderr)

    outcome = await run_phase(FakeClient(container), _run_spec(tmp_path))

    assert container.logs_tail == cst.QUANT_DECODE_LOG_TAIL_LINES
    assert outcome.failure.startswith("exit 1: EEE")
    assert len(outcome.failure) <= sandbox.FAILURE_DETAIL_MAX_CHARS == 350
