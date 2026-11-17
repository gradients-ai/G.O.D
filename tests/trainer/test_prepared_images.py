import io
import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import pytest

from core.image_registry import ImageRegistryConfig
from core.image_registry import sha256_digest
from core.models.payload_models import ImageBuildRequest
from trainer import constants as cst
from trainer.prepared_images import builder as builder_mod
from trainer.prepared_images import cache as cache_mod
from trainer.prepared_images.builder import BuilderUnavailable
from trainer.prepared_images.builder import ImageBuilder
from trainer.prepared_images.builder import classify_build_output
from trainer.prepared_images.builder import classify_clone_error
from trainer.prepared_images.builder import compute_slots
from trainer.prepared_images.builder import detect_build_compression
from trainer.prepared_images.builder import run_streaming
from trainer.prepared_images.local_registry import LocalRegistry
from trainer.prepared_images.r2_transfer import R2ImageStore
from trainer.prepared_images.r2_transfer import fetch_image
from trainer.prepared_images.r2_transfer import publish_image


# --- failure classification (strings taken from real failed builds) ---------------------------------------------


@pytest.mark.parametrize(
    "output,expected",
    [
        (
            "58.94 fatal: unable to access 'https://github.com/kohya-ss/musubi-tuner.git/': Could not resolve host: github.com",
            "infrastructure",
        ),
        ("error: unable to read sha1 file of fiber/__init__.py (9f1f3e41)", "infrastructure"),
        ("toomanyrequests: You have reached your pull rate limit", "infrastructure"),
        ("ERROR: failed to solve: process \"/bin/sh -c pip install foo\" did not complete successfully: exit code: 1", "build"),
        ("ERROR: failed to solve: failed to read dockerfile: open Dockerfile: no such file or directory", "dockerfile"),
        ("ERROR: failed to solve: failed to compute cache key: \"/scripts\" not found", "dockerfile"),
    ],
)
def test_classify_build_output(output, expected):
    assert classify_build_output(output) == expected


def test_classify_clone_error():
    assert classify_clone_error("fatal: could not read Username for 'https://github.com': terminal prompts disabled") == "clone"
    assert classify_clone_error("Invalid commit hash 'abc' - commit not found in repository") == "clone"
    assert classify_clone_error("fatal: unable to access: Could not resolve host: github.com") == "infrastructure"


def test_compression_follows_the_docker_image_store():
    # overlay2 silently ignores compression=zstd (verified on Docker 27), so it builds gzip.
    assert detect_build_compression([["driver-type", "io.containerd.snapshotter.v1"]]) == "zstd"
    assert detect_build_compression([["Backing Filesystem", "extfs"], ["Supports d_type", "true"]]) == "gzip"
    assert detect_build_compression(None) == "gzip"


# --- capacity ---------------------------------------------------------------------------------------------------


def test_compute_slots_on_measured_h100_host():
    # 104 vCPU / 885 GB RAM / ~11 TB free: capped at 16 (external fetch limits), not by the machine.
    assert compute_slots(free_disk_gb=10900, vcpus=104, ram_gb=885, active=0, max_slots=16) == (16, 16)
    assert compute_slots(free_disk_gb=10900, vcpus=104, ram_gb=885, active=10, max_slots=16) == (16, 6)


def test_compute_slots_is_disk_bound_and_never_negative():
    assert compute_slots(free_disk_gb=200, vcpus=104, ram_gb=885, active=0, max_slots=16) == (16, 2)
    assert compute_slots(free_disk_gb=50, vcpus=104, ram_gb=885, active=3, max_slots=16) == (16, 0)
    assert compute_slots(free_disk_gb=5000, vcpus=8, ram_gb=16, active=5, max_slots=16) == (2, 0)


# --- streaming subprocess with stall/timeout --------------------------------------------------------------------


async def test_run_streaming_kills_stalled_process(tmp_path):
    calls = []
    rc, output, reason = await run_streaming(
        ["bash", "-c", "echo started; sleep 30; echo never"],
        log_path=tmp_path / "log",
        timeout_seconds=20,
        stall_seconds=1,
        on_progress=lambda: calls.append(1),
    )
    assert reason == "stalled"
    assert "started" in output and "never" not in output
    assert calls


async def test_run_streaming_timeout_and_success(tmp_path):
    _, _, reason = await run_streaming(
        ["bash", "-c", "while true; do echo tick; sleep 0.2; done"],
        log_path=tmp_path / "log",
        timeout_seconds=1,
        stall_seconds=5,
        on_progress=lambda: None,
    )
    assert reason == "timeout"
    rc, output, reason = await run_streaming(
        ["bash", "-c", "echo ok; exit 3"],
        log_path=tmp_path / "log2",
        timeout_seconds=10,
        stall_seconds=5,
        on_progress=lambda: None,
    )
    assert (rc, reason) == (3, None) and "ok" in output


# --- R2 publish / fetch round trip against an in-memory S3 ------------------------------------------------------


class _Stat:
    def __init__(self, size):
        self.size = size


class _Response:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def stream(self, amt):
        for start in range(0, len(self._data), amt):
            yield self._data[start : start + amt]

    def close(self):
        pass

    def release_conn(self):
        pass


class FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploads = 0

    def stat_object(self, bucket, key):
        from minio.error import S3Error

        if key not in self.objects:
            raise S3Error("NoSuchKey", "missing", key, "r", "h", None)
        return _Stat(len(self.objects[key]))

    def fput_object(self, bucket, key, path, part_size=0, num_parallel_uploads=1):
        self.uploads += 1
        self.objects[key] = Path(path).read_bytes()

    def put_object(self, bucket, key, data: io.BytesIO, length):
        self.objects[key] = data.read()

    def get_object(self, bucket, key, offset=0, length=0):
        data = self.objects[key]
        return _Response(data[offset : offset + length] if length else data[offset:])


def _store(fake: FakeS3) -> R2ImageStore:
    store = R2ImageStore.__new__(R2ImageStore)
    store.config = ImageRegistryConfig(endpoint="https://x", bucket="b", access_key="k", secret_key="s")
    store.client = fake
    return store


def _write_blob(registry: LocalRegistry, data: bytes) -> str:
    digest = sha256_digest(data)
    path = registry.blob_path(digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return digest


def _make_image(registry: LocalRegistry) -> tuple[str, list[str]]:
    layers = [_write_blob(registry, bytes([i]) * (3 * 1024 * 1024 + i)) for i in range(3)]
    config = _write_blob(registry, b'{"architecture":"amd64"}')
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"digest": config, "size": 24},
        "layers": [{"digest": d, "size": 1} for d in layers],
    }
    return _write_blob(registry, json.dumps(manifest).encode()), [config, *layers]


def test_publish_then_fetch_round_trip_and_dedupe(tmp_path, monkeypatch):
    monkeypatch.setattr(cst, "R2_PART_SIZE_BYTES", 1024 * 1024)  # force multi-range downloads
    fake = FakeS3()
    source = LocalRegistry(storage_path=str(tmp_path / "src"))
    digest, blobs = _make_image(source)

    result = publish_image(_store(fake), source, "god/t/miner-text", "b1-a1", digest)
    assert result.uploaded_bytes == result.size_bytes > 0
    uploads = fake.uploads
    again = publish_image(_store(fake), source, "god/t/other-text", "b2-a1", digest)
    assert again.uploaded_bytes == 0 and fake.uploads == uploads  # identical layers are not re-uploaded
    assert any(key.endswith("/_manifests/tags/b1-a1/current/link") for key in fake.objects)

    target = LocalRegistry(storage_path=str(tmp_path / "dst"))
    fetched = fetch_image(_store(fake), target, "god/t/miner-text", digest)
    assert fetched == sum(source.blob_path(d).stat().st_size for d in blobs)
    for blob in blobs + [digest]:
        assert target.blob_path(blob).read_bytes() == source.blob_path(blob).read_bytes()
    assert (target.root / "repositories/god/t/miner-text/_layers/sha256" / blobs[0].split(":")[1] / "link").exists()


def test_fetch_rejects_corrupted_blob(tmp_path):
    fake = FakeS3()
    source = LocalRegistry(storage_path=str(tmp_path / "src"))
    digest, blobs = _make_image(source)
    store = _store(fake)
    publish_image(store, source, "god/t/m", "b1-a1", digest)
    corrupt_key = next(key for key in fake.objects if blobs[1].split(":")[1] in key and key.endswith("/data"))
    fake.objects[corrupt_key] = b"x" * len(fake.objects[corrupt_key])
    target = LocalRegistry(storage_path=str(tmp_path / "dst"))
    with pytest.raises(IOError, match="digest mismatch"):
        fetch_image(store, target, "god/t/m", digest)
    assert not target.blob_path(blobs[1]).exists()


def test_publish_refuses_when_local_blobs_are_missing(tmp_path):
    source = LocalRegistry(storage_path=str(tmp_path / "src"))
    digest, blobs = _make_image(source)
    source.blob_path(blobs[2]).unlink()
    with pytest.raises(FileNotFoundError):
        publish_image(_store(FakeS3()), source, "god/t/m", "b1-a1", digest)


# --- staging storage leases ---------------------------------------------------------------------------------------


def test_local_registry_storage_is_purged_only_after_last_lease(tmp_path):
    registry = LocalRegistry(storage_path=str(tmp_path))
    with registry.lease():
        with registry.lease():
            _write_blob(registry, b"data")
        assert (registry.data / "docker").exists()
    assert not (registry.data / "docker").exists()  # renamed away synchronously
    registry._delete_trash()
    assert not list(tmp_path.glob("trash-*"))


# --- cache eviction ---------------------------------------------------------------------------------------------


def test_select_evictions_ttl_in_use_and_disk_pressure(monkeypatch):
    now = datetime(2026, 11, 20, tzinfo=timezone.utc)
    recent = (now - timedelta(hours=1)).isoformat()
    old = (now - timedelta(hours=cst.PREPARED_IMAGE_TTL_HOURS + 1)).isoformat()
    images = [("old", ["r/old@d"]), ("recent", ["r/recent@d"]), ("busy-old", ["r/busy@d"]), ("unknown", ["r/u@d"])]
    used = {"r/old@d": old, "r/recent@d": recent, "r/busy@d": old}
    evict = cache_mod.select_evictions(images, used, {"busy-old"}, now, free_disk_gb=10_000, image_sizes_gb={})
    assert set(evict) == {"old", "unknown"}
    pressure = cache_mod.select_evictions(
        images, used, {"busy-old"}, now, free_disk_gb=0, image_sizes_gb={"recent": 1000, "old": 10, "unknown": 10}
    )
    assert set(pressure) == {"old", "unknown", "recent"}
    assert "busy-old" not in pressure


# --- builder admission ------------------------------------------------------------------------------------------


def _request(**overrides) -> ImageBuildRequest:
    values = dict(build_id="b1", attempt=1, repo="https://github.com/a/b", variant="text", repository="god/t/m", tag="b1-a1")
    values.update(overrides)
    return ImageBuildRequest(**values)


async def test_builder_rejects_when_disabled_or_full_and_is_idempotent(monkeypatch):
    builder = ImageBuilder(registry=LocalRegistry(storage_path="/tmp/unused"))
    monkeypatch.setenv(cst.IMAGE_BUILDER_ENABLED_ENV, "false")
    with pytest.raises(BuilderUnavailable):
        await builder.submit(_request())

    monkeypatch.setattr(builder, "_enabled_reason", lambda: None)
    monkeypatch.setattr(builder_mod.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(builder_mod, "_ram_gb", lambda: 256.0)
    monkeypatch.setattr(builder_mod, "_free_disk_gb", lambda: 10)  # no disk for a build slot
    with pytest.raises(BuilderUnavailable, match="no free build slots"):
        await builder.submit(_request())

    monkeypatch.setattr(builder_mod, "_free_disk_gb", lambda: 10_000)
    started = []

    async def fake_run(job):
        started.append(job)
        job.state = "ready"

    monkeypatch.setattr(builder, "_run", fake_run)
    first = await builder.submit(_request())
    second = await builder.submit(_request())
    assert first is second
    await first.task
    assert len(started) == 1
