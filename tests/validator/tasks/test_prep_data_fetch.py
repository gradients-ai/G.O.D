"""What prep counts as a data-fetch failure (DataFetchError, retried) and what it leaves as a failure of the task."""

from unittest.mock import AsyncMock

import httpx
import pytest
import requests as requests_lib
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.errors import RepositoryNotFoundError

from core.models.dataset_models import FileFormat
from validator.tasks.datasets import fetch
from validator.tasks.datasets import preparation
from validator.tasks.datasets.fetch import DataFetchError


URL = "https://bucket.s3.amazonaws.com/continuous-sft/blobs/abc.json?X-Amz-Signature=secret"


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", URL)
    return httpx.HTTPStatusError(f"{status} for url {URL}", request=request, response=httpx.Response(status, request=request))


def _hub_error(status: int) -> HfHubHTTPError:
    response = requests_lib.Response()
    response.status_code = status
    return HfHubHTTPError(f"{status} from the hub", response=response)


@pytest.mark.parametrize(
    "error", [_status_error(404), _status_error(503), httpx.ReadTimeout("timed out"), httpx.ConnectError("refused")]
)
async def test_an_http_failure_on_a_data_url_is_a_redacted_fetch_error(monkeypatch, error):
    monkeypatch.setattr(fetch, "download_s3_file", AsyncMock(side_effect=error))

    with pytest.raises(DataFetchError) as raised:
        await fetch.fetch_s3_file(URL)

    assert "secret" not in str(raised.value) and raised.value.__cause__ is error


async def test_a_non_http_download_error_is_not_a_fetch_error(monkeypatch):
    monkeypatch.setattr(fetch, "download_s3_file", AsyncMock(side_effect=ValueError("Download exceeds the size limit")))

    with pytest.raises(ValueError, match="size limit"):
        await fetch.fetch_s3_file(URL)


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (requests_lib.ConnectionError("offline"), True),
        (requests_lib.Timeout("slow"), True),
        (_hub_error(502), True),
        (_hub_error(429), True),
        (_hub_error(403), False),
        (RepositoryNotFoundError("gone", response=_hub_error(404).response), False),
        (ValueError("no train split"), False),
    ],
)
def test_hub_errors_are_transient_only_for_connection_timeout_rate_limit_and_server_errors(error, transient):
    assert fetch.is_transient_hub_error(error) is transient


@pytest.mark.parametrize(
    ("error", "raised"), [(requests_lib.ConnectionError("offline"), DataFetchError), (ValueError("bad"), ValueError)]
)
async def test_a_hub_dataset_load_raises_a_fetch_error_only_when_the_hub_was_unreachable(monkeypatch, error, raised):
    monkeypatch.setattr(preparation, "resolve_code_free_hub_dataset", lambda name: "rev")
    monkeypatch.setattr(preparation, "get_default_dataset_config", lambda name, revision: None)

    def load(*args, **kwargs):
        raise error

    monkeypatch.setattr(preparation, "load_dataset", load)

    with pytest.raises(raised):
        await preparation.download_and_load_dataset("org/data", FileFormat.HF)
