"""Decode work dirs against real sandbox containers (`sandbox.run_phase` with the real specs).

The specs run as the trainer's uid:gid with every capability dropped. With alpine standing in for the decoder images,
a decoder that leaves nested and 0o000 dirs behind must not leak them or break a re-run; with the real pinned decoder
images (when present), build.sh compiles into /build and decode.py imports torch, loads the build and writes /output.
Skipped without a docker daemon or the images.
"""

import os
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException

import trainer.constants as cst
from trainer.quant_decode.job import _WorkDirs
from trainer.quant_decode.job import remove_work_tree
from trainer.quant_decode.sandbox import DecodeContainerSpec
from trainer.quant_decode.sandbox import build_phase_spec
from trainer.quant_decode.sandbox import run_phase
from trainer.quant_decode.sandbox import run_phase_spec


ALPINE = "alpine:3.20"
GB = 1024**3


@pytest.fixture(scope="module")
def docker_client() -> docker.DockerClient:
    try:
        client = docker.from_env()
        client.ping()
    except DockerException as e:
        pytest.skip(f"no docker daemon: {e}")
    return client


def _require_image(client: docker.DockerClient, image: str, pull: bool) -> None:
    try:
        client.images.get(image)
        return
    except DockerException:
        if not pull:
            pytest.skip(f"{image} not present")
    try:
        client.images.pull(image)
    except DockerException as e:
        pytest.skip(f"{image} unavailable: {e}")


@pytest.fixture(scope="module")
def alpine(docker_client) -> docker.DockerClient:
    _require_image(docker_client, ALPINE, pull=True)
    return docker_client


@pytest.fixture(scope="module")
def decoder_images(docker_client) -> docker.DockerClient:
    for image in (cst.QUANT_DECODER_BUILD_IMAGE, cst.QUANT_DECODER_RUN_IMAGE):
        _require_image(docker_client, image, pull=False)
    return docker_client


@pytest.fixture
def work(tmp_path: Path) -> _WorkDirs:
    """Created under a strict umask: the sandbox must not need any group/other bits."""
    work = _WorkDirs(root=tmp_path / "quant_decode" / "task_hotkey")
    old_umask = os.umask(0o077)
    try:
        work.create()
    finally:
        os.umask(old_umask)
    yield work
    remove_work_tree(work.root)


def _with(spec: DecodeContainerSpec, image: str, script: str) -> DecodeContainerSpec:
    return spec.model_copy(update={"image": image, "command": ["sh", "-c", script]})


def _owned_by_me(root: Path) -> None:
    me = (os.geteuid(), os.getegid())
    for path in [root, *root.rglob("*")]:
        st = path.lstat()
        assert (st.st_uid, st.st_gid) == me, path


async def test_sandbox_writes_as_the_trainer_and_leaves_nothing_behind(alpine, work: _WorkDirs):
    build = _with(
        build_phase_spec(work.decoder, work.build),
        ALPINE,
        "id -u > /build/uid && echo $HOME > /build/home && mkdir -p /build/x/deep && touch /build/x/deep/y"
        " && chmod 0 /build/x && ! touch /decoder/z 2>/dev/null",
    )
    run = _with(
        run_phase_spec(work.artifact, work.decoder, work.output, GB, build_dir=work.build),
        ALPINE,
        "mkdir /output/sub && touch /output/sub/f /output/model.safetensors && chmod 0 /output/sub"
        " && ! touch /build/w 2>/dev/null && ! touch /artifact/w 2>/dev/null",
    )

    build_outcome = await run_phase(alpine, build)
    run_outcome = await run_phase(alpine, run)

    assert build_outcome.ok, build_outcome
    assert run_outcome.ok, run_outcome
    assert (work.build / "uid").read_text().strip() == str(os.geteuid())
    assert (work.build / "home").read_text().strip() == "/tmp"
    assert not (work.decoder / "z").exists()
    _owned_by_me(work.build / "uid")
    _owned_by_me(work.output / "model.safetensors")

    # The re-run for the same task and hotkey: the stale dir from the attempt above is emptied, never a FileExistsError.
    work.create()

    assert all(list(path.iterdir()) == [] for path in (work.decoder, work.build, work.artifact, work.output))
    remove_work_tree(work.root)
    assert list(work.root.parent.iterdir()) == []


BUILD_SH = "cc -O2 -shared -fPIC -o /build/libhelper.so /decoder/helper.c\n"
HELPER_C = "int helper_add(int x) { return x + 1; }\n"
DECODE_PY = """import ctypes, os, sys
import torch
from safetensors.torch import save_file
assert ctypes.CDLL("/build/libhelper.so").helper_add(41) == 42
os.makedirs(os.path.join(sys.argv[2], "scratch"))
save_file({"w": torch.ones(2, 2, dtype=torch.bfloat16)}, os.path.join(sys.argv[2], "model.safetensors"))
"""


async def test_real_decoder_images_build_and_run_as_the_trainer(decoder_images, work: _WorkDirs):
    (work.decoder / cst.DECODE_BUILD_SCRIPT).write_text(BUILD_SH)
    (work.decoder / "helper.c").write_text(HELPER_C)
    (work.decoder / "decode.py").write_text(DECODE_PY)

    build_outcome = await run_phase(decoder_images, build_phase_spec(work.decoder, work.build))
    run_outcome = await run_phase(
        decoder_images, run_phase_spec(work.artifact, work.decoder, work.output, GB, build_dir=work.build)
    )

    assert build_outcome.ok, build_outcome
    assert run_outcome.ok, run_outcome
    assert (work.build / "libhelper.so").is_file()
    assert (work.output / "model.safetensors").is_file()
    _owned_by_me(work.build)
    _owned_by_me(work.output)
    remove_work_tree(work.root)
    assert not work.root.exists()
