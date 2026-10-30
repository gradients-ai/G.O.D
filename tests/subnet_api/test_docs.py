import httpx
import pytest

from subnet_api.app import create_app
from subnet_api.docs import OPERATIONS
from subnet_api.docs import PARAMETERS
from subnet_api.docs import SCALAR_INTEGRITY
from subnet_api.docs import TAGS
from subnet_api.settings import Settings


def app(public_url: str = "https://api.sn56.io"):
    settings = Settings(validator_url="http://validator.test", validator_api_key="key", public_url=public_url)
    return create_app(settings)


async def get(path: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app(**kwargs)), base_url="http://test") as api:
        return await api.get(path)


@pytest.mark.asyncio
async def test_docs_page_loads_the_pinned_reference_with_its_integrity_hash() -> None:
    response = await get("/docs")
    assert response.status_code == 200
    assert SCALAR_INTEGRITY in response.text
    assert (await get("/redoc")).status_code == 404


@pytest.mark.asyncio
async def test_every_public_route_is_documented_and_tagged() -> None:
    schema = (await get("/openapi.json")).json()
    tags = {tag["name"] for tag in TAGS}
    documented = {(method, path) for path, item in schema["paths"].items() for method in item}
    assert documented == set(OPERATIONS)
    for item in schema["paths"].values():
        for operation in item.values():
            assert operation["description"] and set(operation["tags"]) <= tags
            for parameter in operation.get("parameters", []):
                assert parameter["name"] in PARAMETERS


@pytest.mark.asyncio
async def test_code_samples_use_the_configured_public_url() -> None:
    schema = (await get("/openapi.json", public_url="https://testnet.example")).json()
    assert schema["servers"] == [{"url": "https://testnet.example"}]
