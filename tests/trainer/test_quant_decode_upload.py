"""Whole-file retry of decoded uploads (trainer/quant_decode/upload.py): transient errors retry, others fail at once."""

import asyncio
import http.client
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import urllib3
from minio import Minio
from minio.error import InvalidResponseError
from minio.error import S3Error
from minio.error import ServerError
from urllib3.exceptions import LocationParseError
from urllib3.exceptions import MaxRetryError
from urllib3.exceptions import ProtocolError
from urllib3.exceptions import SSLError as Urllib3SSLError
from urllib3.util.retry import Retry

import trainer.constants as cst
from trainer.quant_decode import job
from trainer.quant_decode import upload
from trainer.quant_decode.upload import UploadFailed
from trainer.quant_decode.upload import UploadRetryPolicy
from trainer.quant_decode.upload import upload_with_retry


BUCKET = "bucket"
OBJECT = "prefix/model-00001.safetensors"
FAST = UploadRetryPolicy(attempts=4, base_delay_seconds=0, max_delay_seconds=0)


def s3_error(code: str) -> S3Error:
    return S3Error(code, f"{code} message", f"/{BUCKET}/{OBJECT}", "request-id", "host-id", None)


class FakeStorage:
    """put_file stand-in: raises the queued errors in order, then succeeds."""

    def __init__(self, errors: list[Exception]) -> None:
        self.errors = list(errors)
        self.calls: list[tuple[str, str, Path]] = []

    def put_file(self, bucket: str, object_name: str, path: Path, sha256: str | None = None) -> None:
        self.calls.append((bucket, object_name, path))
        if self.errors:
            raise self.errors.pop(0)


@pytest.fixture
def shard(tmp_path: Path) -> Path:
    path = tmp_path / "model-00001.safetensors"
    path.write_bytes(b"x" * 64)
    return path


@pytest.fixture
def reports() -> list[str]:
    return []


@pytest.fixture
def report(reports: list[str]):
    async def record(message: str) -> None:
        reports.append(message)

    return record


def install(monkeypatch, errors: list[Exception]) -> FakeStorage:
    storage = FakeStorage(errors)
    monkeypatch.setattr(upload, "put_file", storage.put_file)
    return storage


async def test_transient_failures_then_success_retry_the_whole_file(monkeypatch, shard, report, reports):
    storage = install(
        monkeypatch,
        [
            http.client.RemoteDisconnected("Remote end closed connection without response"),
            ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
        ],
    )

    await upload_with_retry(BUCKET, OBJECT, shard, report, FAST)

    assert storage.calls == [(BUCKET, OBJECT, shard)] * 3  # same key every time: a retry overwrites
    assert len(reports) == 2
    assert "attempt 1/4 failed (RemoteDisconnected)" in reports[0]
    assert "attempt 2/4 failed (SSLEOFError)" in reports[1]


async def test_persistent_transient_failure_surfaces_after_the_cap(monkeypatch, shard, report, reports):
    storage = install(monkeypatch, [ConnectionResetError("reset")] * 10)

    with pytest.raises(UploadFailed, match=rf"upload of {OBJECT} failed after 4 attempts \(ConnectionResetError\)"):
        await upload_with_retry(BUCKET, OBJECT, shard, report, FAST)

    assert len(storage.calls) == 4
    assert len(reports) == 3  # a retry line per failed attempt that is retried


@pytest.mark.parametrize(
    "error",
    [
        s3_error("AccessDenied"),
        s3_error("InvalidAccessKeyId"),
        s3_error("SignatureDoesNotMatch"),
        s3_error("NoSuchBucket"),
        FileNotFoundError("gone"),
        PermissionError("denied"),
        ValueError("bad part size"),
    ],
)
async def test_non_retryable_errors_fail_on_the_first_attempt(monkeypatch, shard, report, reports, error):
    storage = install(monkeypatch, [error, error])

    with pytest.raises(UploadFailed, match=rf"upload of {OBJECT} failed \({type(error).__name__}"):
        await upload_with_retry(BUCKET, OBJECT, shard, report, FAST)

    assert len(storage.calls) == 1
    assert reports == []


async def test_failure_message_carries_no_url_or_request_details(monkeypatch, shard):
    install(monkeypatch, [s3_error("AccessDenied")])

    with pytest.raises(UploadFailed) as excinfo:
        await upload_with_retry(BUCKET, OBJECT, shard, policy=FAST)

    assert str(excinfo.value) == f"upload of {OBJECT} failed (S3Error AccessDenied)"


async def test_backoff_is_taken_after_each_failed_attempt(monkeypatch, shard, report, reports):
    install(monkeypatch, [ConnectionResetError(), ConnectionResetError()])
    asked: list[int] = []
    real_delay = UploadRetryPolicy.delay_seconds

    def recording_delay(self: UploadRetryPolicy, failed_attempt: int) -> float:
        asked.append(failed_attempt)
        real_delay(self, failed_attempt)
        return 0.0

    monkeypatch.setattr(UploadRetryPolicy, "delay_seconds", recording_delay)

    await upload_with_retry(BUCKET, OBJECT, shard, report, UploadRetryPolicy(attempts=3))

    assert asked == [1, 2]
    assert all("retrying in" in line for line in reports)


@pytest.mark.parametrize(
    "error",
    [
        http.client.RemoteDisconnected("closed"),
        ConnectionResetError("reset"),
        TimeoutError("timed out"),
        ssl.SSLEOFError(8, "EOF"),
        ProtocolError("Connection aborted."),
        MaxRetryError(None, "/bucket/key", ProtocolError("Connection aborted.")),
        s3_error("SlowDown"),
        s3_error("InternalError"),
        s3_error("ServiceUnavailable"),
        s3_error("RequestTimeout"),
        s3_error("IncompleteBody"),
        s3_error("BadDigest"),
        ServerError("server error", 503),
        InvalidResponseError(502, "text/html", "<html>bad gateway</html>"),
    ],
)
def test_transient_errors(error):
    assert upload.is_transient(error)


@pytest.mark.parametrize(
    "error",
    [
        s3_error("AccessDenied"),
        s3_error("NoSuchBucket"),
        InvalidResponseError(403, "text/html", "forbidden"),
        FileNotFoundError("gone"),
        IsADirectoryError("dir"),
        ValueError("bad"),
        LocationParseError("bad:endpoint"),
        MaxRetryError(None, "/bucket/key", Urllib3SSLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))),
    ],
)
def test_permanent_errors(error):
    assert not upload.is_transient(error)


@pytest.mark.parametrize("failed_attempt", [1, 2, 3, 4, 5, 10])
def test_backoff_doubles_with_jitter_and_is_capped(failed_attempt):
    policy = UploadRetryPolicy(attempts=10, base_delay_seconds=10, max_delay_seconds=60)
    ceiling = min(60, 10 * 2 ** (failed_attempt - 1))

    delays = [policy.delay_seconds(failed_attempt) for _ in range(200)]

    assert all(ceiling / 2 <= delay <= ceiling for delay in delays)
    assert len(set(delays)) > 1  # jittered


def test_policy_rejects_zero_attempts():
    with pytest.raises(ValueError):
        UploadRetryPolicy(attempts=0)


def test_put_file_uses_the_decode_part_size_and_parallelism(monkeypatch, shard):
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(upload._storage, "fput_object", lambda *args, **kwargs: calls.append((args, kwargs)))

    upload.put_file(BUCKET, OBJECT, shard)

    assert calls == [
        (
            (BUCKET, OBJECT, str(shard)),
            {
                "metadata": None,
                "part_size": cst.QUANT_DECODE_UPLOAD_PART_BYTES,
                "num_parallel_uploads": cst.QUANT_DECODE_UPLOAD_PARALLEL,
            },
        )
    ]


def test_http_client_retries_parts_with_a_real_backoff():
    retries = upload._storage._http.connection_pool_kw["retries"]

    assert retries.total == cst.QUANT_DECODE_UPLOAD_HTTP_RETRIES
    assert retries.backoff_factor == cst.QUANT_DECODE_UPLOAD_HTTP_BACKOFF_FACTOR
    assert 503 in retries.status_forcelist


def chained(outer: Exception, inner: Exception) -> Exception:
    """outer raised while handling inner, as minio's abort-on-failure does."""
    try:
        try:
            raise inner
        except Exception:
            raise outer
    except Exception as e:
        return e


def test_abort_error_masking_a_dropped_connection_is_transient():
    """complete-multipart lands but its response is lost; minio's abort then gets NoSuchUpload and raises that."""
    error = chained(s3_error("NoSuchUpload"), http.client.RemoteDisconnected("closed"))

    assert upload.is_transient(error)


def test_abort_error_masking_a_permanent_error_is_permanent():
    assert not upload.is_transient(chained(s3_error("NoSuchUpload"), s3_error("AccessDenied")))


async def test_masked_transient_error_is_retried(monkeypatch, shard, report, reports):
    storage = install(monkeypatch, [chained(s3_error("NoSuchUpload"), ConnectionResetError("reset"))])

    await upload_with_retry(BUCKET, OBJECT, shard, report, FAST)

    assert len(storage.calls) == 2
    assert "S3Error NoSuchUpload" in reports[0]


def test_a_stalled_request_leaves_room_for_a_whole_file_retry():
    """Worst case for one attempt (every HTTP retry of one request waits the full read timeout, plus every backoff)
    fits at least twice inside the deadline of a typical 2.2 GB shard."""
    retries = cst.QUANT_DECODE_UPLOAD_HTTP_RETRIES
    backoff = sum(  # urllib3: no sleep before the first retry, then factor x 2^(n-1), capped, plus jitter
        min(cst.QUANT_DECODE_UPLOAD_HTTP_BACKOFF_MAX_SECONDS, cst.QUANT_DECODE_UPLOAD_HTTP_BACKOFF_FACTOR * 2 ** (n - 1))
        + cst.QUANT_DECODE_UPLOAD_HTTP_BACKOFF_JITTER_SECONDS
        for n in range(2, retries + 1)
    )
    one_attempt = (retries + 1) * cst.QUANT_DECODE_UPLOAD_HTTP_READ_TIMEOUT_SECONDS + backoff

    assert 2 * one_attempt + cst.QUANT_DECODE_UPLOAD_RETRY_BASE_SECONDS <= job.upload_timeout_seconds(2_200_000_000)


PART = 5 * 1024 * 1024  # minio's minimum part size


class FlakyS3:
    """A minimal multipart S3 endpoint whose part PUTs drop the connection without a response `drops` times."""

    def __init__(self, drops: int) -> None:
        self.drops = drops
        self.objects: dict[str, bytes] = {}
        self.parts: dict[int, bytes] = {}
        self.aborted = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _reply(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
                self.send_response(status)
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> bytes:
                return self.rfile.read(int(self.headers.get("Content-Length", 0)))

            def do_POST(self):  # noqa: N802
                self._body()
                if "uploads" in self.path:
                    xml = (
                        b"<InitiateMultipartUploadResult><Bucket>bucket</Bucket><Key>k</Key>"
                        b"<UploadId>u1</UploadId></InitiateMultipartUploadResult>"
                    )
                    return self._reply(200, xml)
                key = self.path.split("?", 1)[0].removeprefix(f"/{BUCKET}/")
                outer.objects[key] = b"".join(outer.parts[n] for n in sorted(outer.parts))
                bucket_key = f"<Bucket>{BUCKET}</Bucket><Key>{key}</Key>"
                xml = f'<CompleteMultipartUploadResult>{bucket_key}<ETag>"e"</ETag></CompleteMultipartUploadResult>'
                self._reply(200, xml.encode())

            def do_PUT(self):  # noqa: N802
                body = self._body()
                if outer.drops > 0:
                    outer.drops -= 1
                    self.close_connection = True
                    self.connection.close()  # RemoteDisconnected on the client
                    return
                part = int(self.path.split("partNumber=", 1)[1].split("&", 1)[0])
                outer.parts[part] = body
                self._reply(200, headers={"ETag": f'"p{part}"'})

            def do_DELETE(self):  # noqa: N802
                outer.aborted += 1
                self._reply(204)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def client(self, http_retries: int, pool_manager: type[urllib3.PoolManager] = urllib3.PoolManager) -> Minio:
        http_client = pool_manager(retries=Retry(total=http_retries, backoff_factor=0))
        endpoint = f"127.0.0.1:{self.server.server_port}"
        return Minio(endpoint, access_key="a", secret_key="b", secure=False, region="us-east-1", http_client=http_client)


@pytest.fixture
def big_shard(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(cst, "QUANT_DECODE_UPLOAD_PART_BYTES", PART)
    path = tmp_path / "model-00001.safetensors"
    path.write_bytes(bytes(range(256)) * (2 * PART // 256 + 1))
    return path


async def test_real_client_retries_a_dropped_part_at_the_http_layer(monkeypatch, big_shard, report, reports):
    s3 = FlakyS3(drops=1)
    monkeypatch.setattr(upload, "_storage", s3.client(http_retries=2))
    try:
        await upload_with_retry(BUCKET, OBJECT, big_shard, report, FAST)
    finally:
        s3.server.shutdown()

    assert s3.objects[OBJECT] == big_shard.read_bytes()
    assert reports == []  # the HTTP client's own retry absorbed it


async def test_real_client_exhausted_part_retries_fall_back_to_a_whole_file_retry(monkeypatch, big_shard, report, reports):
    s3 = FlakyS3(drops=3)  # more drops than one attempt's HTTP retries (1 + 1)
    monkeypatch.setattr(upload, "_storage", s3.client(http_retries=1))
    try:
        await upload_with_retry(BUCKET, OBJECT, big_shard, report, FAST)
    finally:
        s3.server.shutdown()

    assert s3.objects[OBJECT] == big_shard.read_bytes()
    assert s3.aborted >= 1  # minio aborted the failed multipart upload before the retry
    assert len(reports) >= 1 and "attempt 1/4 failed" in reports[0]


def test_failed_part_surfaces_instead_of_blocking_the_upload(monkeypatch, tmp_path):
    """With more parts than upload threads, a failed part must raise from put_file, not leave it waiting on the pool."""
    monkeypatch.setattr(cst, "QUANT_DECODE_UPLOAD_PART_BYTES", PART)
    path = tmp_path / "model-00001.safetensors"
    with path.open("wb") as f:
        f.truncate(4 * cst.QUANT_DECODE_UPLOAD_PARALLEL * PART)  # sparse
    s3 = FlakyS3(drops=10**6)  # every part fails
    monkeypatch.setattr(upload, "_storage", s3.client(http_retries=0))
    errors: list[Exception] = []

    def run() -> None:
        try:
            upload.put_file(BUCKET, OBJECT, path)
        except Exception as e:
            errors.append(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=30)
    s3.server.shutdown()

    assert not thread.is_alive(), "put_file is still blocked after a part failed"
    assert len(errors) == 1 and upload.is_transient(errors[0])


async def test_sliced_part_bodies_arrive_intact_and_are_resent_whole_on_a_retry(monkeypatch, big_shard, report, reports):
    monkeypatch.setattr(cst, "QUANT_DECODE_UPLOAD_SEND_SLICE_BYTES", 64 * 1024)
    s3 = FlakyS3(drops=1)
    monkeypatch.setattr(upload, "_storage", s3.client(http_retries=2, pool_manager=upload._SlicingPoolManager))
    try:
        await upload_with_retry(BUCKET, OBJECT, big_shard, report, FAST, "f" * 64)
    finally:
        s3.server.shutdown()

    assert s3.objects[OBJECT] == big_shard.read_bytes()
    assert reports == []


def test_storage_client_slices_part_bodies():
    assert isinstance(upload._storage._http, upload._SlicingPoolManager)


@pytest.mark.parametrize(
    "body, headers, sliced",
    [
        (b"x" * 100, {"Content-Length": "100"}, True),
        (b"x" * 100, {"content-length": "100"}, False),  # a plain dict is case-sensitive; minio sends an HTTPHeaderDict
        (b"x" * 100, urllib3.HTTPHeaderDict({"content-length": "100"}), True),
        (b"x" * 100, {}, False),  # no Content-Length: slicing would turn the request into chunked encoding
        (b"x" * 10, {"Content-Length": "10"}, False),  # one slice or less
        (None, {}, False),
    ],
)
def test_only_large_bodies_with_a_content_length_are_sliced(monkeypatch, body, headers, sliced):
    monkeypatch.setattr(cst, "QUANT_DECODE_UPLOAD_SEND_SLICE_BYTES", 16)
    sent: list[object] = []
    monkeypatch.setattr(urllib3.PoolManager, "urlopen", lambda self, method, url, redirect=True, **kw: sent.append(kw["body"]))

    upload._SlicingPoolManager().urlopen("PUT", "http://storage.example/key", body=body, headers=headers)

    assert isinstance(sent[0], upload._SlicedBody) is sliced
    if sliced:
        slices = [bytes(part) for part in sent[0]]
        assert b"".join(slices) == body and max(map(len, slices)) == 16
        assert [bytes(part) for part in sent[0]] == slices  # a retry iterates again from the start


def test_slicing_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(cst, "QUANT_DECODE_UPLOAD_SEND_SLICE_BYTES", 0)
    sent: list[object] = []
    monkeypatch.setattr(urllib3.PoolManager, "urlopen", lambda self, method, url, redirect=True, **kw: sent.append(kw["body"]))

    upload._SlicingPoolManager().urlopen("PUT", "http://storage.example/key", body=b"x" * 100, headers={"Content-Length": "100"})

    assert sent == [b"x" * 100]


class SlowReader:
    """An HTTP endpoint that reads a request body steadily but slowly, then answers 200."""

    def __init__(self, bytes_per_second: int) -> None:
        chunk = 16 * 1024

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_PUT(self):  # noqa: N802
                remaining = int(self.headers["Content-Length"])
                while remaining > 0:
                    data = self.rfile.read(min(chunk, remaining))
                    if not data:
                        return
                    remaining -= len(data)
                    time.sleep(chunk / bytes_per_second)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *_args):
                pass

        class Server(ThreadingHTTPServer):
            def server_bind(self) -> None:
                # Small buffers, so the sender is held to the reader's pace instead of filling the kernel's queue.
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024)
                super().server_bind()

        self.server = Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def put(self, pool_manager: type[urllib3.PoolManager], body: bytes, send_timeout: float) -> int:
        client = pool_manager(
            timeout=urllib3.Timeout(connect=send_timeout, read=30),
            retries=False,
            socket_options=[(socket.SOL_SOCKET, socket.SO_SNDBUF, 32 * 1024)],
        )
        url = f"http://127.0.0.1:{self.server.server_port}/part"
        return client.urlopen("PUT", url, body=body, headers={"Content-Length": str(len(body))}).status


def test_a_slow_but_moving_part_times_out_in_one_write_and_survives_when_sliced(monkeypatch):
    """The socket timeout while a body is sent is the connect timeout and covers one write. A whole part in one write
    must leave within it; sliced, only each slice must."""
    monkeypatch.setattr(cst, "QUANT_DECODE_UPLOAD_SEND_SLICE_BYTES", 64 * 1024)
    body = b"x" * (4 * 1024 * 1024)  # ~4 s at the reader's pace
    slow = SlowReader(bytes_per_second=1024 * 1024)
    try:
        with pytest.raises(urllib3.exceptions.ProtocolError, match="timed out"):
            slow.put(urllib3.PoolManager, body, send_timeout=1.0)
        assert slow.put(upload._SlicingPoolManager, body, send_timeout=1.0) == 200  # sent after the whole body is read
    finally:
        slow.server.shutdown()


class FakeStat:
    def __init__(self, size: int, metadata: dict[str, str]) -> None:
        self.size = size
        self.metadata = urllib3.HTTPHeaderDict(metadata)


@pytest.mark.parametrize(
    "stat, uploaded",
    [
        (FakeStat(64, {"X-Amz-Meta-Sha256": "a" * 64}), True),
        (FakeStat(63, {"x-amz-meta-sha256": "a" * 64}), False),  # another size
        (FakeStat(64, {"x-amz-meta-sha256": "b" * 64}), False),  # other content, e.g. an earlier decode of the job
        (FakeStat(64, {}), False),  # no metadata: cannot be confirmed
        (s3_error("NoSuchKey"), False),
        (ConnectionResetError("reset"), False),
    ],
)
def test_is_uploaded_needs_the_same_size_and_sha256(monkeypatch, stat, uploaded):
    def stat_object(bucket, object_name):
        if isinstance(stat, Exception):
            raise stat
        return stat

    monkeypatch.setattr(upload._storage, "stat_object", stat_object)

    assert upload.is_uploaded(BUCKET, OBJECT, 64, "a" * 64) is uploaded


def test_put_file_stores_the_sha256_as_object_metadata(monkeypatch, shard):
    calls: list[dict] = []
    monkeypatch.setattr(upload._storage, "fput_object", lambda *args, **kwargs: calls.append(kwargs))

    upload.put_file(BUCKET, OBJECT, shard, "a" * 64)

    assert calls[0]["metadata"] == {"sha256": "a" * 64}
    assert upload.SHA256_METADATA_HEADER == "x-amz-meta-sha256"  # how minio sends that key and how stat returns it


async def test_upload_failed_says_whether_a_later_attempt_can_succeed(monkeypatch, shard, report):
    install(monkeypatch, [ConnectionResetError("reset")] * 4)
    with pytest.raises(UploadFailed) as transient:
        await upload_with_retry(BUCKET, OBJECT, shard, report, FAST)
    install(monkeypatch, [s3_error("AccessDenied")])
    with pytest.raises(UploadFailed) as permanent:
        await upload_with_retry(BUCKET, OBJECT, shard, report, FAST)

    assert transient.value.transient is True
    assert permanent.value.transient is False


# --- stopping an upload the caller gave up on


class BlockingPut:
    """put_file stand-in that sends until its upload's cancel event is set, like a body sender between slices."""

    def __init__(self) -> None:
        self.running = 0
        self.peak = 0
        self.finished = 0
        self.lock = threading.Lock()

    def put_file(self, bucket: str, object_name: str, path: Path, sha256: str | None = None) -> None:
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            cancel = upload._cancel_events[object_name]
            while not cancel.is_set():
                time.sleep(0.01)
            time.sleep(0.1)  # winding down: the request in flight failing, the multipart upload being aborted
        finally:
            with self.lock:
                self.running -= 1
                self.finished += 1


async def wait_until(condition) -> None:
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


async def test_an_upload_that_runs_out_of_time_stops_its_thread(monkeypatch, shard, report):
    blocking = BlockingPut()
    monkeypatch.setattr(upload, "put_file", blocking.put_file)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(upload_with_retry(BUCKET, OBJECT, shard, report, FAST), timeout=0.1)

    await wait_until(lambda: blocking.finished == 1)
    assert blocking.running == 0
    assert OBJECT not in upload._cancel_events


async def test_a_new_upload_of_an_object_waits_for_the_abandoned_one_to_leave(monkeypatch, shard, report):
    blocking = BlockingPut()
    monkeypatch.setattr(upload, "put_file", blocking.put_file)
    monkeypatch.setattr(upload, "CANCEL_POLL_SECONDS", 0.01)

    for _ in range(3):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(upload_with_retry(BUCKET, OBJECT, shard, report, FAST), timeout=0.05)

    await wait_until(lambda: blocking.running == 0)
    assert blocking.peak == 1  # never two threads sending the same object


def test_a_cancelled_upload_stops_between_slices_and_refuses_new_parts(monkeypatch):
    monkeypatch.setattr(cst, "QUANT_DECODE_UPLOAD_SEND_SLICE_BYTES", 16)
    cancel = threading.Event()
    monkeypatch.setitem(upload._cancel_events, "prefix/model.safetensors", cancel)
    sent: list[object] = []
    monkeypatch.setattr(
        urllib3.PoolManager, "urlopen", lambda self, method, url, redirect=True, **kw: sent.append(kw.get("body"))
    )
    client = upload._SlicingPoolManager()
    part_url = "http://storage.example/bucket/prefix/model.safetensors?partNumber=1&uploadId=u1"
    headers = {"Content-Length": "64"}

    client.urlopen("PUT", part_url, body=b"x" * 64, headers=headers)
    slices = iter(sent[0])
    next(slices)
    cancel.set()

    with pytest.raises(upload.UploadCancelled):
        next(slices)
    with pytest.raises(upload.UploadCancelled):
        client.urlopen("PUT", part_url, body=b"x" * 64, headers=headers)
    client.urlopen("DELETE", part_url)  # minio can still abort the multipart upload
    client.urlopen("PUT", "http://storage.example/bucket/prefix/other.safetensors", body=b"x" * 64, headers=headers)
    assert len(sent) == 3


def test_a_cancelled_upload_is_never_retried():
    assert not upload.is_transient(upload.UploadCancelled("upload cancelled"))


# --- settings from the environment


@pytest.mark.parametrize(
    "raw, expected, warned",
    [
        (None, 16, False),
        ("", 16, False),
        ("32", 32, False),
        (" 32 ", 32, False),
        ("sixteen", 16, True),  # not a number: the default
        ("1", 5, True),  # under the minimum: the minimum
        ("-3", 5, True),
    ],
)
def test_integer_settings_fall_back_instead_of_stopping_the_trainer(monkeypatch, caplog, raw, expected, warned):
    if raw is None:
        monkeypatch.delenv("TEST_QUANT_SETTING", raising=False)
    else:
        monkeypatch.setenv("TEST_QUANT_SETTING", raw)

    with caplog.at_level("WARNING", logger=cst.__name__):
        assert cst._env_int("TEST_QUANT_SETTING", 16, 5) == expected

    assert bool(caplog.records) is warned


@pytest.mark.parametrize("raw, expected", [("0.5", 1.0), ("nan", 1.0), ("2.5", 2.5), ("fast", 30.0)])
def test_float_settings_fall_back_instead_of_stopping_the_trainer(monkeypatch, raw, expected):
    monkeypatch.setenv("TEST_QUANT_SETTING", raw)

    assert cst._env_float("TEST_QUANT_SETTING", 30.0, 1.0) == expected


def test_upload_settings_respect_what_storage_accepts():
    assert cst.QUANT_DECODE_UPLOAD_PART_BYTES >= cst.S3_MIN_PART_BYTES
    assert cst.QUANT_DECODE_UPLOAD_PARALLEL >= 1
    assert cst.QUANT_DECODE_UPLOAD_RETRY_ROUND_MAX_SECONDS >= cst.QUANT_DECODE_UPLOAD_RETRY_ROUND_BASE_SECONDS
