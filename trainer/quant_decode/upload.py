"""Upload of decoded files to S3-compatible storage, retried as a whole file on transient errors.

Two layers of retry. The HTTP client retries each request (one multipart part) on connection errors and 429/5xx.
When a part still fails, minio aborts the multipart upload and raises; if the error is transient (a dropped
connection, a TLS EOF, a throttle or a 5xx) the whole file is uploaded again after an exponential backoff with jitter.
Re-uploading writes the same key, so a retry overwrites and is safe. Anything else (bad credentials, a missing bucket,
an unreadable local file) fails at once.
"""

import asyncio
import http.client
import os
import random
import ssl
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path

import certifi
import urllib3
from minio import Minio
from minio.error import InvalidResponseError
from minio.error import S3Error
from minio.error import ServerError
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from urllib3.exceptions import HTTPError as Urllib3HTTPError
from urllib3.exceptions import LocationValueError
from urllib3.exceptions import MaxRetryError
from urllib3.exceptions import SSLError as Urllib3SSLError
from urllib3.util.retry import Retry

import trainer.constants as cst
from trainer.telemetry import logger
from validator.infrastructure.minio_client import S3_COMPATIBLE_ACCESS_KEY
from validator.infrastructure.minio_client import S3_COMPATIBLE_ENDPOINT
from validator.infrastructure.minio_client import S3_COMPATIBLE_SECRET_KEY
from validator.infrastructure.minio_client import S3_REGION
from validator.infrastructure.minio_client import S3_SECURE


TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
# IncompleteBody, BadDigest and XAmzContentSHA256Mismatch are a body cut or corrupted in transit.
TRANSIENT_S3_CODES = frozenset(
    {
        "BadDigest",
        "IncompleteBody",
        "InternalError",
        "OperationAborted",
        "RequestTimeout",
        "ServiceUnavailable",
        "SlowDown",
        "XAmzContentSHA256Mismatch",
    }
)
# Network-level failures: RemoteDisconnected, ConnectionResetError, ssl.SSLError (e.g. UNEXPECTED_EOF), socket timeouts
# and urllib3's wrappers of them (MaxRetryError, ProtocolError) once the HTTP client's own retries are spent.
TRANSIENT_NETWORK_ERRORS = (
    ConnectionError,
    TimeoutError,
    ssl.SSLError,
    http.client.HTTPException,
    Urllib3HTTPError,
)

RetryReporter = Callable[[str], Awaitable[None]]


class UploadFailed(Exception):
    """The file could not be uploaded: a non-transient error, or transient ones on every attempt."""


class UploadRetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempts: int = Field(default=cst.QUANT_DECODE_UPLOAD_ATTEMPTS, ge=1)
    base_delay_seconds: float = Field(default=cst.QUANT_DECODE_UPLOAD_RETRY_BASE_SECONDS, ge=0)
    max_delay_seconds: float = Field(default=cst.QUANT_DECODE_UPLOAD_RETRY_MAX_SECONDS, ge=0)

    def delay_seconds(self, failed_attempt: int) -> float:
        """Backoff after the n-th failed attempt (1-based): base x 2^(n-1), capped, with equal jitter (half fixed,
        half random) so parallel jobs that failed together do not retry together."""
        ceiling = min(self.max_delay_seconds, self.base_delay_seconds * 2 ** (failed_attempt - 1))
        return ceiling / 2 + random.uniform(0, ceiling / 2)


def _http_client() -> urllib3.PoolManager:
    """minio's default client with a longer backoff: its own (5 retries, 0.2 s factor) gives up within ~3 s, shorter
    than a typical blip on a long-haul link. Same CA handling as minio's default."""
    return urllib3.PoolManager(
        timeout=urllib3.Timeout(
            connect=cst.QUANT_DECODE_UPLOAD_HTTP_CONNECT_TIMEOUT_SECONDS,
            read=cst.QUANT_DECODE_UPLOAD_HTTP_READ_TIMEOUT_SECONDS,
        ),
        maxsize=max(10, cst.QUANT_DECODE_UPLOAD_PARALLEL),
        cert_reqs="CERT_REQUIRED",
        ca_certs=os.environ.get("SSL_CERT_FILE") or certifi.where(),
        retries=Retry(
            total=cst.QUANT_DECODE_UPLOAD_HTTP_RETRIES,
            backoff_factor=cst.QUANT_DECODE_UPLOAD_HTTP_BACKOFF_FACTOR,
            backoff_max=cst.QUANT_DECODE_UPLOAD_HTTP_BACKOFF_MAX_SECONDS,
            backoff_jitter=cst.QUANT_DECODE_UPLOAD_HTTP_BACKOFF_JITTER_SECONDS,
            status_forcelist=sorted(TRANSIENT_HTTP_STATUSES),
            retry_after_max=cst.QUANT_DECODE_UPLOAD_HTTP_BACKOFF_MAX_SECONDS,  # a long Retry-After must not eat the deadline
        ),
    )


_storage = Minio(
    S3_COMPATIBLE_ENDPOINT,
    access_key=S3_COMPATIBLE_ACCESS_KEY,
    secret_key=S3_COMPATIBLE_SECRET_KEY,
    secure=S3_SECURE,
    region=S3_REGION,
    http_client=_http_client(),
)


def put_file(bucket: str, object_name: str, path: Path) -> None:
    """One multipart upload of the whole file; raises on any failure (minio aborts the multipart upload first)."""
    _storage.fput_object(
        bucket,
        object_name,
        str(path),
        part_size=cst.QUANT_DECODE_UPLOAD_PART_BYTES,
        num_parallel_uploads=cst.QUANT_DECODE_UPLOAD_PARALLEL,
    )


def is_transient(error: Exception) -> bool:
    """True for errors a later attempt can fix: network drops, throttling, server-side 5xx.

    The whole implicit chain counts: when a part or the complete call fails, minio aborts the multipart upload inside
    its `except`, and an error from that abort (e.g. NoSuchUpload after a complete that landed but whose response was
    lost) replaces the original one, which is kept as its __context__.
    """
    if _is_config_error(error):
        return False
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, Exception) and _is_transient_one(current):
            return True
        current = current.__context__
    return False


def _is_config_error(error: Exception) -> bool:
    """A bad endpoint or a failed certificate check: no retry fixes these."""
    if isinstance(error, LocationValueError):
        return True
    reason = error.reason if isinstance(error, MaxRetryError) else error
    if isinstance(reason, Urllib3SSLError) and reason.args:
        reason = reason.args[0]
    return isinstance(reason, ssl.SSLCertVerificationError)


def _is_transient_one(error: Exception) -> bool:
    if _is_config_error(error):
        return False
    if isinstance(error, S3Error):
        return error.code in TRANSIENT_S3_CODES
    if isinstance(error, ServerError):
        return error.status_code >= 500 or error.status_code in TRANSIENT_HTTP_STATUSES
    if isinstance(error, InvalidResponseError):  # non-XML body, e.g. a proxy's 502 page
        return getattr(error, "_code", None) in TRANSIENT_HTTP_STATUSES
    return isinstance(error, TRANSIENT_NETWORK_ERRORS)


def describe_error(error: Exception) -> str:
    """Type (and S3 code) only: messages can carry URLs and request ids."""
    if isinstance(error, S3Error) and error.code:
        return f"{type(error).__name__} {error.code}"
    return type(error).__name__


async def _no_report(message: str) -> None:
    return None


async def upload_with_retry(
    bucket: str,
    object_name: str,
    path: Path,
    report: RetryReporter = _no_report,
    policy: UploadRetryPolicy = UploadRetryPolicy(),
) -> None:
    """Upload `path` to bucket/object_name, retrying the whole file on transient errors. Raises UploadFailed.

    Bounded by attempts x (one upload + its backoff); the caller puts a deadline around the whole call.
    """
    for attempt in range(1, policy.attempts + 1):
        try:
            await asyncio.to_thread(put_file, bucket, object_name, path)
            return
        except Exception as e:
            detail = describe_error(e)
            if not is_transient(e):
                raise UploadFailed(f"upload of {object_name} failed ({detail})") from None
            if attempt == policy.attempts:
                raise UploadFailed(f"upload of {object_name} failed after {attempt} attempts ({detail})") from None
            delay = policy.delay_seconds(attempt)
            message = (
                f"decode: upload of {path.name} attempt {attempt}/{policy.attempts} failed ({detail}); retrying in {delay:.0f}s"
            )
            logger.warning(message)
            await report(message)
            await asyncio.sleep(delay)
