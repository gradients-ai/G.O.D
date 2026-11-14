"""The data fetches of task prep, and DataFetchError: the one error that marks a prep failure as infrastructure.

A prep failure is either a property of the task (a near-duplicate dataset, too few rows, a model that cannot load: the
task is failed and the tournament replaces it) or a failure to reach the data (an HTTP error or timeout on a data URL,
a hub connection error, an object-store upload that did not go through). Only the second raises DataFetchError, and
the prep loop retries the same task with backoff instead of failing it (validator/lifecycle/tasks.py).
"""

import httpx
import requests
from huggingface_hub.errors import HfHubHTTPError

from core.downloads import download_s3_file
from core.logging import redact_url_queries


# Hub HTTP statuses that say nothing about the dataset itself: rate limiting and server-side errors.
_TRANSIENT_HUB_STATUS_FLOOR = 500
_HUB_RATE_LIMITED_STATUS = 429


class DataFetchError(RuntimeError):
    """Prep could not fetch or store the task's data; nothing is known to be wrong with the task. Messages are redacted
    (data URLs are presigned)."""


async def fetch_s3_file(url: str, save_path: str | None = None) -> str:
    """download_s3_file, with any HTTP-level failure (connection, timeout, non-2xx status) raised as DataFetchError."""
    try:
        return await download_s3_file(url, save_path=save_path)
    except httpx.HTTPError as e:
        raise DataFetchError(redact_url_queries(f"could not download {url}: {type(e).__name__}: {e}")) from e


def is_transient_hub_error(error: BaseException) -> bool:
    """Whether a Hugging Face hub / datasets error is a connection failure, a timeout, rate limiting or a server error."""
    if isinstance(error, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(error, HfHubHTTPError) and error.response is not None:
        status = error.response.status_code
        return status >= _TRANSIENT_HUB_STATUS_FLOOR or status == _HUB_RATE_LIMITED_STATUS
    return False
