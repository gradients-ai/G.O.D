"""Whole-file retry of decoded uploads (trainer/quant_decode/upload.py): transient errors retry, others fail at once."""

import http.client
import ssl
import threading
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

    def put_file(self, bucket: str, object_name: str, path: Path) -> None:
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
            {"part_size": cst.QUANT_DECODE_UPLOAD_PART_BYTES, "num_parallel_uploads": cst.QUANT_DECODE_UPLOAD_PARALLEL},
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

    def client(self, http_retries: int) -> Minio:
        http_client = urllib3.PoolManager(retries=Retry(total=http_retries, backoff_factor=0))
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
