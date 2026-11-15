"""huggingface_hub errors for tests: their constructors require the HTTP response that produced them."""

import httpx
from huggingface_hub.errors import RepositoryNotFoundError


def hub_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api/models/org/repo"))


def repo_not_found(message: str) -> RepositoryNotFoundError:
    return RepositoryNotFoundError(message, response=hub_response(404))
