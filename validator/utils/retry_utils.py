from requests.exceptions import HTTPError
from tenacity import retry
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from validator.utils.logging import get_logger


logger = get_logger(__name__)


def should_retry_model_loading_on_exception(e):
    ephemeral_error_patterns = [
        "does not appear to have a file named",
        "does not appear to have files named",
        "Too Many Requests for url",
    ]

    while e is not None:
        if isinstance(e, HTTPError):
            if e.response is None:
                logger.error(f"HTTPError with no response: {e}, cause: {e.__cause__}")
                return True
            elif 500 <= e.response.status_code < 600:
                return True
        if any(pattern in str(e) for pattern in ephemeral_error_patterns):
            return True
        e = e.__cause__
    return False


def retry_on_5xx():
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2.5, min=30, max=600),
        retry=retry_if_exception(should_retry_model_loading_on_exception),
        reraise=True,
    )