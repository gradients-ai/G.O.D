"""Evaluation failure classes and the one predicate that decides whose fault a failure is.

An evaluation can end three ways short of a score:

- the evaluator judged the miner's repo and it failed (a quant scored fail, a model that does not load, a
  non-finite loss): the miner's fault, scored. These come back as results, never as exceptions, except for the
  per-repo error string a text/image evaluator reports, which arrives as RepoEvaluationFailed;
- infrastructure failed (network, timeouts with or without a message, HTTP 5xx/429, the hub, object storage, the
  dstack/Basilica control plane, the DB): EvaluationInfraError;
- our own code failed (a bug, an input we should have provided): also EvaluationInfraError, because it is
  equally not the miner's fault.

EvaluationInfraError is an EvaluationRetryableError: lifecycle resets the rows to pending, counts the attempt per
row and alerts, and after MAX_EVAL_INFRA_ATTEMPTS holds the row (never a scored fail, never a failed task).
EvaluationCapacityUnavailable is a wait for capacity, not an error, and is never counted. DeploymentNotReadyError is
uncounted for text and image; for quant tasks it is counted (a pod that never turns healthy would otherwise be
re-launched every cycle forever). Any other EvaluationRetryableError is counted like an EvaluationInfraError.

Standard library only: the evaluator images import this module too (the image evaluator has no core/validator
stack at all), and none of them carry the DB or the object-storage client.
"""

import asyncio
import http.client
import socket
import urllib.error


# Results-file key an evaluator writes, as {EVALUATION_ERROR_RESULT_KEY: "<Type>: <message>"} in place of a repo's
# result, when evaluating that repo failed for a reason that is not the miner's (the hub or object storage
# unreachable, the evaluator's own setup failing). The validator re-queues such a repo instead of scoring it.
EVALUATION_ERROR_RESULT_KEY = "evaluation_error"


class EvaluationRetryableError(RuntimeError):
    """Evaluation should be retried later without marking rows as failed."""


class EvaluationCapacityUnavailable(EvaluationRetryableError):
    pass


class DeploymentNotReadyError(EvaluationRetryableError):
    pass


class EvaluationInfraError(EvaluationRetryableError):
    """Infrastructure failure or an evaluation error on our side: never the miner's fault.

    Re-queued like any retryable error, but counted per evaluation row and alerted; a row that keeps hitting it is
    held for a human instead of being scored.
    """


class RepoEvaluationFailed(Exception):
    """The evaluator ran this repo and reported an error for it: the miner's failure, scored as such."""

    def __init__(self, message: str):
        super().__init__(message or "the evaluator reported an empty error")


# 408 Request Timeout and 429 Too Many Requests are the server telling us to come back; 5xx is the server failing.
_RETRYABLE_CLIENT_STATUS_CODES = frozenset({408, 429})

_STDLIB_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    TimeoutError,
    asyncio.TimeoutError,
    ConnectionError,  # includes http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError
    socket.gaierror,
    http.client.IncompleteRead,
)

# Third-party transport/server failures, matched by (top-level package, class name) anywhere in the exception's MRO
# rather than by isinstance, so this module needs none of these packages installed.
_THIRD_PARTY_TRANSPORT_ERRORS = frozenset(
    {
        ("httpx", "TransportError"),  # ReadTimeout (whose str() is ''), ConnectError, RemoteProtocolError, ...
        ("requests", "ConnectionError"),
        ("requests", "Timeout"),
        ("requests", "ChunkedEncodingError"),
        ("urllib3", "HTTPError"),  # every urllib3 error is a transport error
        ("asyncpg", "PostgresConnectionError"),
        ("asyncpg", "InterfaceError"),
        ("asyncpg", "TooManyConnectionsError"),
        ("asyncpg", "CannotConnectNowError"),
        ("asyncpg", "TransactionRollbackError"),  # deadlock / serialization failure: retry the transaction
        ("minio", "ServerError"),  # object storage answered with a non-S3 error page (a proxy 5xx)
        ("minio", "InvalidResponseError"),
    }
)

# HTTP error classes whose status decides: 5xx/408/429 are infra, any other status is a real answer.
_THIRD_PARTY_HTTP_STATUS_ERRORS = frozenset(
    {
        ("httpx", "HTTPStatusError"),
        ("requests", "HTTPError"),  # includes huggingface_hub's HfHubHTTPError (0.x)
        ("huggingface_hub", "HfHubHTTPError"),
        ("minio", "S3Error"),
    }
)


def _class_keys(exc: BaseException) -> set[tuple[str, str]]:
    return {(cls.__module__.split(".")[0], cls.__name__) for cls in type(exc).__mro__}


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status an HTTP error carries, None when it carries none."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code
    if not _class_keys(exc) & _THIRD_PARTY_HTTP_STATUS_ERRORS:
        return None
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) or getattr(response, "status", None)
    return status if isinstance(status, int) else None


def _is_infra_exception(exc: BaseException) -> bool:
    if isinstance(exc, (EvaluationRetryableError, *_STDLIB_TRANSPORT_ERRORS)):
        return True
    if _class_keys(exc) & _THIRD_PARTY_TRANSPORT_ERRORS:
        return True
    status = _http_status(exc)
    if status is not None:
        return status >= 500 or status in _RETRYABLE_CLIENT_STATUS_CODES
    # A URLError that is not an HTTPError never got a response (DNS, refused, reset).
    return isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError)


def is_infra_error(exc: BaseException) -> bool:
    """True when exc, or anything in its cause/context chain, is a transport, timeout, 5xx/429 or retryable failure.

    The chain matters: a client wraps the transport error it hit (an OSError from transformers around a hub
    timeout, a RuntimeError around httpx.ReadTimeout), and the wrapper alone says nothing about whose fault it was.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if _is_infra_exception(current):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def describe_error(exc: BaseException) -> str:
    """Type and message, so an exception whose str() is empty (httpx.ReadTimeout) is never reported blank."""
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else f"{type(exc).__name__} (no message)"


def as_infra_error(exc: BaseException, context: str) -> EvaluationInfraError:
    """exc as an EvaluationInfraError (unchanged when it already is one), for `raise as_infra_error(e, ...) from e`."""
    if isinstance(exc, EvaluationInfraError):
        return exc
    return EvaluationInfraError(f"{context}: {describe_error(exc)}")


def evaluation_error_result(exc: BaseException) -> dict[str, str]:
    """The results-file entry an evaluator writes for a repo it could not evaluate through no fault of the miner;
    the validator re-queues it as an EvaluationInfraError instead of scoring it."""
    return {EVALUATION_ERROR_RESULT_KEY: describe_error(exc)}
