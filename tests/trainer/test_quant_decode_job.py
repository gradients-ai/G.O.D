"""Freeform decode job end to end against a local HTTP snapshot server, a fake HF download, fake docker and fake MinIO."""

import asyncio
import hashlib
import io
import json
import os
import stat
import struct
import tarfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from docker.errors import ImageNotFound
from pydantic import ValidationError

import trainer.constants as cst
from core.models.quant_artifact_models import QuantDecodedIndex
from core.models.quant_artifact_models import QuantDecodeRequest
from core.models.quant_artifact_models import QuantDecodeResult
from trainer.quant_decode import job
from trainer.quant_decode import sandbox
from trainer.quant_decode.job import _WorkDirs
from trainer.quant_decode.job import check_artifact_files
from trainer.quant_decode.job import run_freeform_decode
from validator.evaluation.quant_artifacts import DecodedModelIndex


TASK_ID = "task-1"
HOTKEY = "5HotkeyABCDEFGH"
PREFIX = f"quant-decoded/{TASK_ID}/{HOTKEY}"
SHARD = "model-00001-of-00001.safetensors"
BASE_TENSORS = {"embed.weight": [4, 3], "norm.weight": [3]}
BF16_ONE = 0x3F80
ARTIFACT_FILES = {".gitattributes": b"*.bin filter=lfs\n", "payload.bin": b"\x01" * 100, "meta/config.json": b"{}"}
DECODE_PY = b"print('decode')\n"
HANG = "hang"


def write_safetensors(path: Path, tensors: dict[str, list[int]], dtype: str = "BF16") -> None:
    header: dict = {}
    blobs: list[bytes] = []
    offset = 0
    for name, shape in tensors.items():
        blob = np.full(shape, BF16_ONE, dtype="<u2").tobytes()
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    header_bytes = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(blobs))


def _file_member(name: str, content: bytes) -> tuple[tarfile.TarInfo, io.BytesIO]:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mode = 0o644
    return info, io.BytesIO(content)


def write_decoder_xz(path: Path, files: dict[str, bytes], extra: list[tarfile.TarInfo] = ()) -> str:
    """A git-archive-shaped tar.xz: a `decoder/` dir entry, then members under `decoder/`. Returns its sha256."""
    with tarfile.open(path, mode="w:xz") as tar:
        root = tarfile.TarInfo("decoder/")
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        tar.addfile(root)
        for name, content in files.items():
            tar.addfile(*_file_member(name, content))
        for info in extra:
            tar.addfile(info)
    return hashlib.sha256(path.read_bytes()).hexdigest()


FAILED_RUN_LOGS = b"Traceback\nValueError: bad header\n"


class FakeContainer:
    def __init__(self, name: str, exit_code: int | str, on_run=None, logs: bytes = FAILED_RUN_LOGS):
        self.name = name
        self.exit_code = exit_code
        self.on_run = on_run
        self.log_bytes = logs
        self.killed = threading.Event()

    def wait(self) -> dict:
        if self.exit_code == HANG:
            self.killed.wait(timeout=5)
            return {"StatusCode": 137}
        if self.on_run is not None:
            self.on_run()
        return {"StatusCode": self.exit_code}

    def logs(self, stdout: bool = True, stderr: bool = True, tail: int | str = "all") -> bytes:
        return self.log_bytes

    def kill(self) -> None:
        self.killed.set()

    def remove(self, force: bool = False, v: bool = False) -> None:
        pass


class FakeContainers:
    """Runs each phase per its behaviour (an exit code, HANG, or an exception); the run phase calls write_output."""

    def __init__(self, build=0, run=0, write_output=None, write_build=None, build_logs: bytes = b"built\n"):
        self.behaviour = {"build": build, "run": run}
        self.write_output = write_output
        self.write_build = write_build
        self.build_logs = build_logs
        self.phases: list[str] = []
        self.run_volumes: dict | None = None
        self.users: list[str] = []

    def run(self, **kwargs) -> FakeContainer:
        phase = "build" if kwargs["image"] == cst.QUANT_DECODER_BUILD_IMAGE else "run"
        self.phases.append(phase)
        self.users.append(kwargs["user"])
        behaviour = self.behaviour[phase]
        if isinstance(behaviour, Exception):
            raise behaviour
        on_run = None
        if phase == "build":
            build_dir = next(Path(host) for host, mount in kwargs["volumes"].items() if mount["bind"] == "/build")
            on_run = partial(self.write_build, build_dir) if self.write_build else None
            if behaviour == 0:
                return FakeContainer(kwargs["name"], behaviour, on_run, logs=self.build_logs)
        if phase == "run":
            self.run_volumes = kwargs["volumes"]
            output_dir = next(Path(host) for host, mount in kwargs["volumes"].items() if mount["bind"] == "/output")
            on_run = partial(self.write_output, output_dir) if self.write_output else None
        return FakeContainer(kwargs["name"], behaviour, on_run)


class FakeDocker:
    def __init__(self, containers: FakeContainers):
        self.containers = containers


def write_good_shard(output_dir: Path) -> None:
    write_safetensors(output_dir / SHARD, BASE_TENSORS)


@pytest.fixture
def served_dir(tmp_path: Path):
    served = tmp_path / "served"
    served.mkdir()
    handler = partial(SimpleHTTPRequestHandler, directory=str(served))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield served, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


class FakeArtifactRepo:
    """Stands in for the miner's HF artifact repo; snapshot_download writes these files plus HF's local .cache."""

    def __init__(self):
        self.files: dict[str, bytes] = dict(ARTIFACT_FILES)
        self.modes: dict[str, int] = {}
        self.error: Exception | None = None

    def snapshot_download(self, repo_id: str, local_dir: Path, token: str | None) -> None:
        if self.error is not None:
            raise self.error
        for name, content in self.files.items():
            path = Path(local_dir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(self.modes.get(name, 0o644))
        cache = Path(local_dir) / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        (cache / "download.lock").write_bytes(b"x" * 1000)

    def total_bytes(self) -> int:
        return sum(len(content) for content in self.files.values())


@pytest.fixture
def artifact_repo(monkeypatch) -> FakeArtifactRepo:
    repo = FakeArtifactRepo()
    monkeypatch.setattr(job, "snapshot_download", repo.snapshot_download)
    return repo


@pytest.fixture
def uploads(monkeypatch) -> list[tuple[str, str, bytes]]:
    calls: list[tuple[str, str, bytes]] = []

    async def fake_upload(bucket, object_name, file_path, part_size=0, num_parallel_uploads=3):
        assert (part_size, num_parallel_uploads) == (cst.QUANT_DECODE_UPLOAD_PART_BYTES, cst.QUANT_DECODE_UPLOAD_PARALLEL)
        calls.append((bucket, object_name, Path(file_path).read_bytes()))
        return True

    monkeypatch.setattr(job.async_minio_client, "upload_file", fake_upload)
    return calls


def expected_artifact_files(repo: FakeArtifactRepo) -> list[dict]:
    return [
        {"name": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in sorted(repo.files.items())
    ]


class Setup:
    def __init__(self, tmp_path: Path, served: Path, base_url: str):
        self.tmp_path = tmp_path
        self.served = served
        self.base_url = base_url
        self.base_model_dir = tmp_path / "base_model"
        self.base_model_dir.mkdir()
        write_safetensors(self.base_model_dir / "model.safetensors", BASE_TENSORS)
        self.work_root = tmp_path / "work_root"
        self.decoder_sha256 = ""

    def write_decoder(self, files: dict[str, bytes] | None = None, extra: list[tarfile.TarInfo] = ()) -> None:
        files = {"decoder/decode.py": DECODE_PY} if files is None else files
        self.decoder_sha256 = write_decoder_xz(self.served / "decoder.tar.xz", files, extra)

    @property
    def xz_bytes(self) -> int:
        return (self.served / "decoder.tar.xz").stat().st_size

    def request(self, decoder_sha256: str | None = None) -> QuantDecodeRequest:
        return QuantDecodeRequest(
            decoder_snapshot_url=f"{self.base_url}/decoder.tar.xz",
            decoder_sha256=decoder_sha256 or self.decoder_sha256,
            bucket="decoded",
            object_prefix=PREFIX,
        )

    async def decode(self, containers: FakeContainers, decoder_sha256: str | None = None, report=None) -> QuantDecodeResult:
        kwargs = {} if report is None else {"report": report}
        return await run_freeform_decode(
            self.request(decoder_sha256),
            task_id=TASK_ID,
            hotkey=HOTKEY,
            artifact_repo="user/artifact",
            base_model_dir=self.base_model_dir,
            hf_token=None,
            work_root=self.work_root,
            docker_client=FakeDocker(containers),
            **kwargs,
        )

    def assert_work_removed(self) -> None:
        assert [p.name for p in self.work_root.iterdir()] == [job.WORK_ROOT_MARKER]


@pytest.fixture
def setup(tmp_path: Path, served_dir, artifact_repo, uploads) -> Setup:
    served, base_url = served_dir
    setup = Setup(tmp_path, served, base_url)
    setup.write_decoder()
    return setup


async def test_happy_path_uploads_checked_shard_then_index(setup: Setup, artifact_repo, uploads):
    containers = FakeContainers(write_output=write_good_shard)

    result = await setup.decode(containers)

    assert result.ok, result
    assert result.uploaded_files == [SHARD, "index.json"]
    assert [object_name for _, object_name, _ in uploads] == [f"{PREFIX}/{SHARD}", f"{PREFIX}/index.json"]
    assert result.index_object == f"{PREFIX}/index.json"
    assert result.n_tensors == len(BASE_TENSORS)
    assert result.artifact_bytes == artifact_repo.total_bytes()
    assert result.decoder_bytes == setup.xz_bytes
    assert result.decoder_sha256 == setup.decoder_sha256
    assert containers.phases == ["build", "run"]

    shard_bytes = uploads[0][2]
    index = json.loads(uploads[1][2])
    assert index["files"] == [{"name": SHARD, "bytes": len(shard_bytes), "sha256": hashlib.sha256(shard_bytes).hexdigest()}]
    assert index["n_tensors"] == len(BASE_TENSORS)
    assert index["decoder_sha256"] == setup.decoder_sha256
    assert index["artifact_bytes"] == artifact_repo.total_bytes()
    assert (index["task_id"], index["hotkey"]) == (TASK_ID, HOTKEY)
    expected_artifact = expected_artifact_files(artifact_repo)
    assert index["artifact_files"] == expected_artifact
    assert [f.model_dump() for f in result.artifact_files] == expected_artifact
    assert [f.model_dump() for f in result.decoded_files] == index["files"]
    setup.assert_work_removed()


def write_libhelper(build_dir: Path) -> None:
    (build_dir / "libhelper.so").write_bytes(b"\x7fELF")


BUILD_SH = b"cc -O2 -shared helper.c -o /build/libhelper.so || echo 'helper.c build failed'\n"


async def test_no_build_script_never_warns(setup: Setup):
    result = await setup.decode(FakeContainers(write_output=write_good_shard))

    assert result.ok and result.build_warning is None


async def test_clean_build_never_warns(setup: Setup):
    setup.write_decoder({"decoder/decode.py": DECODE_PY, "decoder/build.sh": BUILD_SH})

    result = await setup.decode(FakeContainers(write_output=write_good_shard, write_build=write_libhelper))

    assert result.ok and result.build_warning is None


@pytest.mark.parametrize(
    ("write_build", "build_logs", "expected"),
    [
        (None, b"helper.c build failed\n", ["left /build empty", "logged 'helper.c build failed'"]),
        (None, b"", ["left /build empty"]),
        (write_libhelper, b"helper.c:3:1: error: expected ';'\n", ["logged", "helper.c:3:1: error: expected ';'"]),
    ],
)
async def test_silent_build_failure_warns_but_still_decodes(setup: Setup, caplog, write_build, build_logs, expected):
    """A decoder under cap-drop ALL whose build.sh prints its failure and exits 0 still runs and is scored."""
    setup.write_decoder({"decoder/decode.py": DECODE_PY, "decoder/build.sh": BUILD_SH})
    containers = FakeContainers(write_output=write_good_shard, write_build=write_build, build_logs=build_logs)

    with caplog.at_level("WARNING"):
        result = await setup.decode(containers)

    assert result.ok and result.quant_failure is None
    assert containers.phases == ["build", "run"]
    for reason in expected:
        assert reason in result.build_warning
    assert ("left /build empty" in result.build_warning) == (write_build is None)
    assert "QUANT DECODER BUILD WARNING" in caplog.text
    assert QuantDecodeResult.model_validate_json(result.model_dump_json()).build_warning == result.build_warning


async def test_run_phase_mounts_build_output_read_only(setup: Setup):
    containers = FakeContainers(write_output=write_good_shard)

    await setup.decode(containers)

    build_mounts = [mount for mount in containers.run_volumes.values() if mount["bind"] == "/build"]
    assert build_mounts == [{"bind": "/build", "mode": "ro"}]


def _mode_and_owner(path: Path) -> tuple[int, int, int]:
    st = path.stat()
    return stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid


async def test_sandbox_runs_as_the_trainer_and_work_dirs_stay_private(setup: Setup):
    """Both phases run as the trainer's own uid:gid, so the work dirs need no group/other write bits: under a strict
    umask they stay owner-only, and the trainer owns whatever the decoder writes."""
    modes: dict[str, tuple[int, int, int]] = {}

    def record_build(build_dir: Path) -> None:
        work = build_dir.parent
        modes.update({name: _mode_and_owner(work / name) for name in ("build", "output", "decoder", "artifact")})

    containers = FakeContainers(write_output=write_good_shard, write_build=record_build)
    old_umask = os.umask(0o077)
    try:
        result = await setup.decode(containers)
    finally:
        os.umask(old_umask)

    assert result.ok, result
    me = (os.geteuid(), os.getegid())
    assert containers.users == [f"{me[0]}:{me[1]}"] * 2
    for name, (mode, *owner) in modes.items():
        assert tuple(owner) == me and mode & 0o077 == 0, (name, oct(mode))


async def test_work_root_dirs_and_files_are_owner_only(setup: Setup, artifact_repo, monkeypatch):
    """Root, every work subdir and every file we write are 0o700/0o600 whatever the umask, so a co-tenant root
    container (no DAC override caps) only ever gets the 'other' bits, which are empty."""
    artifact_repo.files["payload.bin"] = b"\x01" * 100
    artifact_repo.modes["payload.bin"] = 0o664
    seen: dict[str, int] = {}

    def record_run(output_dir: Path) -> None:
        write_good_shard(output_dir)
        work = output_dir.parent
        for path in (setup.work_root, work, *work.iterdir(), *(work / "artifact").rglob("*")):
            seen[str(path.relative_to(setup.work_root))] = stat.S_IMODE(path.stat().st_mode)

    old_umask = os.umask(0o000)
    try:
        result = await setup.decode(FakeContainers(write_output=record_run))
    finally:
        os.umask(old_umask)

    assert result.ok, result
    assert stat.S_IMODE(setup.work_root.stat().st_mode) == 0o700
    expected_dirs = {
        ".",
        f"{TASK_ID}_{HOTKEY[:8]}",
        *(f"{TASK_ID}_{HOTKEY[:8]}/{n}" for n in ("decoder", "build", "artifact", "output", "artifact/meta")),
    }
    for name, mode in seen.items():
        if name in expected_dirs:
            assert mode == 0o700, (name, oct(mode))
        else:
            assert mode == 0o600, (name, oct(mode))
    assert {"{0}/decoder.tar.xz".format(f"{TASK_ID}_{HOTKEY[:8]}"), f"{TASK_ID}_{HOTKEY[:8]}/artifact/payload.bin"} <= set(seen)


async def test_work_root_is_private_with_a_marker_and_its_new_ancestors_keep_default_modes(setup: Setup):
    setup.work_root = setup.tmp_path / "a" / "b" / "root"
    old_umask = os.umask(0o022)
    try:
        result = await setup.decode(FakeContainers(write_output=write_good_shard))
        assert result.ok, result
        setup.work_root.chmod(0o755)
        result = await setup.decode(FakeContainers(write_output=write_good_shard))
    finally:
        os.umask(old_umask)

    assert result.ok, result
    assert stat.S_IMODE(setup.work_root.stat().st_mode) == 0o700
    marker = setup.work_root / job.WORK_ROOT_MARKER
    assert marker.is_file() and stat.S_IMODE(marker.stat().st_mode) == 0o600
    for ancestor in (setup.work_root.parent, setup.work_root.parent.parent):
        assert stat.S_IMODE(ancestor.stat().st_mode) == 0o755, ancestor


async def test_empty_existing_work_root_is_adopted(setup: Setup):
    setup.work_root.mkdir(mode=0o755)
    setup.work_root.chmod(0o755)

    result = await setup.decode(FakeContainers(write_output=write_good_shard))

    assert result.ok, result
    assert stat.S_IMODE(setup.work_root.stat().st_mode) == 0o700
    assert (setup.work_root / job.WORK_ROOT_MARKER).is_file()


async def test_marked_work_root_with_leftovers_is_reused(setup: Setup):
    setup.work_root.mkdir()
    (setup.work_root / job.WORK_ROOT_MARKER).touch()
    (setup.work_root / "old_task.stale-abcd1234").mkdir()
    setup.work_root.chmod(0o755)

    result = await setup.decode(FakeContainers(write_output=write_good_shard))

    assert result.ok, result
    assert stat.S_IMODE(setup.work_root.stat().st_mode) == 0o700
    assert (setup.work_root / "old_task.stale-abcd1234").is_dir()


async def test_shared_non_empty_directory_is_never_chmodded_and_is_an_infra_error(setup: Setup):
    shared = setup.tmp_path / "data"
    shared.mkdir()
    (shared / "someone_elses.bin").write_bytes(b"x")
    shared.chmod(0o755)
    setup.work_root = shared

    result = await setup.decode(FakeContainers(write_output=write_good_shard))

    assert result.infra_error and "QUANT_DECODE_WORK_ROOT" in result.infra_error
    assert result.quant_failure is None and not result.ok
    assert stat.S_IMODE(shared.stat().st_mode) == 0o755
    assert sorted(p.name for p in shared.iterdir()) == ["someone_elses.bin"]


def test_work_root_that_is_a_file_is_an_oserror(tmp_path: Path):
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x")

    with pytest.raises(OSError):
        job.prepare_work_root(not_a_dir)


async def test_index_json_is_owner_only_and_parses_in_the_evaluator_model(setup: Setup, artifact_repo, uploads, monkeypatch):
    index_modes: list[int] = []
    real_upload = job._upload

    async def spy(bucket, object_name, path, report=job._no_report):
        if path.name == "index.json":
            index_modes.append(stat.S_IMODE(path.stat().st_mode))
        await real_upload(bucket, object_name, path, report)

    monkeypatch.setattr(job, "_upload", spy)
    result = await setup.decode(FakeContainers(write_output=write_good_shard))

    assert result.ok, result
    assert index_modes == [0o600]
    raw = uploads[-1][2]
    assert QuantDecodedIndex.model_validate_json(raw).artifact_files == result.artifact_files
    evaluator_view = DecodedModelIndex.model_validate_json(raw)
    assert evaluator_view.n_tensors == len(BASE_TENSORS)
    assert evaluator_view.artifact_bytes == artifact_repo.total_bytes()


async def test_failures_carry_the_known_artifact_and_decoded_files(setup: Setup, artifact_repo, uploads, monkeypatch):
    scored = await setup.decode(FakeContainers(run=1))
    assert scored.quant_failure is not None
    assert [f.model_dump() for f in scored.artifact_files] == expected_artifact_files(artifact_repo)
    assert scored.decoded_files == []

    async def fail_index(bucket, object_name, file_path, part_size=0, num_parallel_uploads=3):
        return not object_name.endswith("index.json")

    monkeypatch.setattr(job.async_minio_client, "upload_file", fail_index)
    infra = await setup.decode(FakeContainers(write_output=write_good_shard))
    assert infra.infra_error is not None
    assert [f.name for f in infra.decoded_files] == [SHARD]
    assert len(infra.artifact_files) == len(ARTIFACT_FILES)


def test_old_format_result_and_index_still_parse():
    old_result = '{"ok": true, "artifact_bytes": 5, "n_tensors": 2, "uploaded_files": ["a"]}'
    result = QuantDecodeResult.model_validate_json(old_result)
    assert (result.artifact_files, result.decoded_files) == ([], [])
    old_index = '{"files": [], "n_tensors": 1, "decoder_sha256": "x", "artifact_bytes": 1, "task_id": "t", "hotkey": "h"}'
    assert QuantDecodedIndex.model_validate_json(old_index).artifact_files == []


def write_locked_subdirs(output_dir: Path) -> None:
    """What `mkdir /output/x && touch /output/x/y && chmod 0 /output/x` leaves: a dir the trainer cannot list."""
    write_good_shard(output_dir)
    locked = output_dir / "x"
    locked.mkdir()
    (locked / "y").write_bytes(b"y")
    locked.chmod(0)


async def test_locked_subdir_from_the_decoder_is_removed_and_a_rerun_succeeds(setup: Setup):
    first = await setup.decode(FakeContainers(write_output=write_locked_subdirs))
    setup.assert_work_removed()

    second = await setup.decode(FakeContainers(write_output=write_good_shard))

    assert not first.ok and "unexpected" in first.quant_failure, first
    assert second.ok, second
    setup.assert_work_removed()


def _stale_work(root: Path) -> _WorkDirs:
    work = _WorkDirs(root=root)
    work.create()
    (work.build / "x").mkdir()
    (work.build / "x" / "y").write_bytes(b"old")
    (work.output / SHARD).write_bytes(b"old")
    (work.build / "x").chmod(0)
    return work


def _assert_fresh(work: _WorkDirs) -> None:
    for path in (work.decoder, work.build, work.artifact, work.output):
        assert path.is_dir() and list(path.iterdir()) == [], path


def test_create_empties_a_stale_work_dir(tmp_path: Path):
    work = _stale_work(tmp_path / "w" / "task_hotkey")

    work.create()

    _assert_fresh(work)
    assert sorted(path.name for path in work.root.parent.iterdir()) == [job.WORK_ROOT_MARKER, "task_hotkey"]


def test_create_moves_aside_a_work_dir_it_cannot_remove(tmp_path: Path, monkeypatch):
    """A leftover owned by root (an old sandbox) survives removal; the re-run must not fail on it."""
    work = _stale_work(tmp_path / "w" / "task_hotkey")
    remove_work_tree = job.remove_work_tree
    monkeypatch.setattr(job, "remove_work_tree", lambda root: None)

    work.create()

    _assert_fresh(work)
    [stale] = [path for path in work.root.parent.iterdir() if path not in (work.root, work.root.parent / job.WORK_ROOT_MARKER)]
    assert stale.name.startswith(f"task_hotkey{job.STALE_WORK_DIR_MARKER}")
    assert (stale / "output" / SHARD).read_bytes() == b"old"
    remove_work_tree(stale)
    assert not stale.exists()


async def test_unusable_work_root_is_infra_error(setup: Setup):
    setup.work_root.write_bytes(b"not a dir")
    containers = FakeContainers(write_output=write_good_shard)

    result = await setup.decode(containers)

    assert result.infra_error.startswith("decode work dir unusable:"), result
    assert result.quant_failure is None
    assert containers.phases == []


async def test_unwritable_work_root_is_infra_error(setup: Setup, monkeypatch):
    real_touch = Path.touch

    def deny_probe(self, *args, **kwargs):
        if self.name.startswith(".probe-"):
            raise PermissionError(13, "denied")
        return real_touch(self, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", deny_probe)
    containers = FakeContainers(write_output=write_good_shard)

    result = await setup.decode(containers)

    assert result.infra_error.startswith("decode work dir unusable:"), result
    assert result.quant_failure is None
    assert containers.phases == []


def test_remove_work_tree_never_follows_a_symlink(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep").write_bytes(b"k")
    link = tmp_path / "link"
    link.symlink_to(target)

    job.remove_work_tree(link)

    assert not link.exists() and (target / "keep").exists()


async def test_snapshot_sha_mismatch_is_infra_error(setup: Setup, uploads):
    containers = FakeContainers()

    result = await setup.decode(containers, decoder_sha256="0" * 64)

    assert result.infra_error == "decoder snapshot sha mismatch"
    assert result.quant_failure is None
    assert containers.phases == []
    assert uploads == []
    setup.assert_work_removed()


async def test_unreachable_snapshot_is_infra_error(setup: Setup):
    (setup.served / "decoder.tar.xz").unlink()

    result = await setup.decode(FakeContainers())

    assert result.infra_error == "decoder snapshot download failed (HTTPError)"
    assert setup.base_url not in result.infra_error


def _symlink(name: str, target: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info


@pytest.mark.parametrize(
    ("files", "extra"),
    [
        ({"decoder/decode.py": DECODE_PY, "decoder/../x": b"escape"}, []),
        ({"decoder/decode.py": DECODE_PY, "../x": b"escape"}, []),
        ({"decoder/decode.py": DECODE_PY, "/etc/x": b"absolute"}, []),
        ({"decoder/decode.py": DECODE_PY}, [_symlink("decoder/weights", "/artifact")]),
        ({"decoder/decode.py": DECODE_PY}, [_symlink("decoder/base", "../../base_model/model.safetensors")]),
    ],
)
async def test_unsafe_decoder_archive_is_scored_fail(setup: Setup, files, extra):
    setup.write_decoder(files, extra)
    containers = FakeContainers()

    result = await setup.decode(containers)

    assert result.quant_failure == "decoder archive contains unsafe members"
    assert containers.phases == []
    assert not (setup.tmp_path / "x").exists()
    setup.assert_work_removed()


async def test_decoder_expanding_past_cap_is_scored_fail(setup: Setup, monkeypatch):
    monkeypatch.setattr(job, "DECODER_MAX_EXTRACTED_BYTES", 1024)
    setup.write_decoder({"decoder/decode.py": DECODE_PY, "decoder/blob.bin": b"\0" * 4096})

    result = await setup.decode(FakeContainers())

    assert result.quant_failure == "decoder expands past cap"
    setup.assert_work_removed()


async def test_decoder_without_entrypoint_is_scored_fail(setup: Setup):
    setup.write_decoder({"decoder/main.py": DECODE_PY})

    result = await setup.decode(FakeContainers())

    assert result.quant_failure == "decoder has no decode.py"


@pytest.mark.parametrize(
    ("name", "mode", "failure"),
    [
        ("helper.py", 0o644, "artifact file helper.py has forbidden suffix .py"),
        ("lib/helper.SO", 0o644, "artifact file lib/helper.SO has forbidden suffix .SO"),
        ("payload2.bin", 0o755, "artifact file payload2.bin is executable"),
    ],
)
async def test_executable_artifact_is_scored_fail_before_any_phase(setup: Setup, artifact_repo, name, mode, failure):
    artifact_repo.files[name] = b"payload"
    artifact_repo.modes[name] = mode
    containers = FakeContainers()

    result = await setup.decode(containers)

    assert result.quant_failure == failure
    assert result.decoder_bytes == setup.xz_bytes
    assert containers.phases == []
    setup.assert_work_removed()


def test_symlinked_artifact_file_is_rejected(tmp_path: Path):
    (tmp_path / "payload.bin").write_bytes(b"x")
    (tmp_path / "weights.bin").symlink_to(tmp_path / "payload.bin")

    assert check_artifact_files(tmp_path) == "artifact file weights.bin is a symlink"


async def test_hf_download_failure_is_infra_error(setup: Setup, artifact_repo):
    artifact_repo.error = OSError("hub unreachable")

    result = await setup.decode(FakeContainers())

    assert result.infra_error == "artifact download failed: hub unreachable"


async def test_base_model_without_safetensors_is_infra_error(setup: Setup):
    (setup.base_model_dir / "model.safetensors").unlink()

    result = await setup.decode(FakeContainers())

    assert result.infra_error == "base model has no safetensors"


async def test_build_container_error_is_infra_error(setup: Setup):
    containers = FakeContainers(build=ImageNotFound("no such image"))

    result = await setup.decode(containers)

    assert result.infra_error == "container error: no such image"
    assert result.quant_failure is None
    setup.assert_work_removed()


async def test_build_nonzero_exit_is_scored_fail(setup: Setup):
    result = await setup.decode(FakeContainers(build=2))

    assert result.quant_failure == "decoder build: exit 2: ValueError: bad header"


async def test_run_nonzero_exit_is_scored_fail(setup: Setup):
    containers = FakeContainers(run=1)

    result = await setup.decode(containers)

    assert result.quant_failure == "decoder: exit 1: ValueError: bad header"
    assert containers.phases == ["build", "run"]
    setup.assert_work_removed()


async def test_run_timeout_is_scored_fail(setup: Setup, monkeypatch):
    monkeypatch.setattr(sandbox, "DECODE_TIMEOUT_SECONDS", 0.05)
    containers = FakeContainers(run=HANG)

    result = await setup.decode(containers)

    assert result.quant_failure == "decoder timed out after 900s"
    assert result.run_seconds >= 0.05
    setup.assert_work_removed()


async def test_bad_output_is_scored_fail(setup: Setup, uploads):
    def write_wrong_dtype(output_dir: Path) -> None:
        write_safetensors(output_dir / SHARD, BASE_TENSORS, dtype="F16")

    result = await setup.decode(FakeContainers(write_output=write_wrong_dtype))

    assert result.quant_failure == "decoded output: tensor embed.weight has dtype F16, expected BF16"
    assert uploads == []
    setup.assert_work_removed()


async def test_stray_index_is_deleted_and_not_uploaded(setup: Setup, uploads):
    def write_shard_and_lying_index(output_dir: Path) -> None:
        write_good_shard(output_dir)
        index = {"weight_map": {"embed.weight": "model-00002-of-00002.safetensors"}}
        (output_dir / "model.safetensors.index.json").write_text(json.dumps(index))

    result = await setup.decode(FakeContainers(write_output=write_shard_and_lying_index))

    assert result.ok
    assert result.uploaded_files == [SHARD, "index.json"]
    assert all("model.safetensors.index.json" not in object_name for _, object_name, _ in uploads)


async def test_failed_upload_is_infra_error(setup: Setup, monkeypatch):
    async def failing_upload(bucket, object_name, file_path, **kwargs):
        return False

    monkeypatch.setattr(job.async_minio_client, "upload_file", failing_upload)

    result = await setup.decode(FakeContainers(write_output=write_good_shard))

    assert result.infra_error == f"upload of {PREFIX}/{SHARD} failed"
    assert result.index_object is None
    setup.assert_work_removed()


def test_result_rejects_ok_with_quant_failure():
    with pytest.raises(ValidationError):
        QuantDecodeResult(ok=True, quant_failure="decoder: exit 1")


def test_result_rejects_no_outcome():
    with pytest.raises(ValidationError):
        QuantDecodeResult()


async def test_stalled_upload_is_infra_error_not_a_hang(setup: Setup, monkeypatch):
    """A stalled multipart upload ends as an infra_error rather than hanging."""

    async def stalled_upload(bucket, object_name, file_path, **kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(job.async_minio_client, "upload_file", stalled_upload)
    monkeypatch.setattr(job, "upload_timeout_seconds", lambda size_bytes: 0.05)

    result = await asyncio.wait_for(setup.decode(FakeContainers(write_output=write_good_shard)), timeout=10)

    assert result.infra_error == f"upload of {PREFIX}/{SHARD} timed out after 0s"
    assert result.index_object is None
    setup.assert_work_removed()


def test_upload_deadline_scales_with_file_size():
    floor = cst.QUANT_DECODE_UPLOAD_MIN_TIMEOUT_SECONDS
    two_gb = 2 * 1000 * 1000 * 1000
    assert job.upload_timeout_seconds(0) == floor
    assert job.upload_timeout_seconds(two_gb) == floor + two_gb / cst.QUANT_DECODE_UPLOAD_MIN_BYTES_PER_SECOND


async def test_every_stage_is_reported_to_the_job_log(setup: Setup, uploads):
    lines: list[str] = []

    async def report(message: str) -> None:
        lines.append(message)

    result = await setup.decode(FakeContainers(write_output=write_good_shard), report=report)

    assert result.ok, result
    stages = [line.split(" (")[0].split(" in ")[0] for line in lines]
    assert stages == [
        "decode: artifact fetched",
        "decode: decoder built",
        "decode: decoder ran",
        "decode: output checked",
        "decode: uploading 1 decoded shards",
        f"decode: uploaded {SHARD}",
        "decode: uploaded index.json",
    ]
