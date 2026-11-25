"""Stored presigned URLs are re-signed with the current key before an auditing endpoint returns them."""

from urllib.parse import parse_qs
from urllib.parse import urlparse

from pydantic import BaseModel

from validator.infrastructure.minio_client import AsyncMinioClient
from validator.infrastructure.presigned import resign_presigned_urls


ENDPOINT = "s3.eu-central-003.backblazeb2.com"
OLD = (
    f"https://{ENDPOINT}/gradients-validator/abc_train_data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=OLDKEY%2F20261123%2Feu-central-003%2Fs3%2Faws4_request&X-Amz-Date=20261123T234010Z"
    "&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=deadbeef"
)
FOREIGN = "https://other.example.com/bucket/x.zip?X-Amz-Signature=keep"


def _client() -> AsyncMinioClient:
    return AsyncMinioClient(ENDPOINT, "NEWKEY", "secret", "eu-central-003")


def _credential(url: str) -> str:
    return parse_qs(urlparse(url).query)["X-Amz-Credential"][0].split("/")[0]


class Pair(BaseModel):
    image_url: str
    text_url: str | None = None


class Task(BaseModel):
    task_id: str
    training_data: str | None = None
    image_text_pairs: list[Pair] = []
    extra: dict = {}


async def test_our_urls_get_the_current_key_and_others_are_untouched():
    task = Task(
        task_id="t",
        training_data=OLD,
        image_text_pairs=[Pair(image_url=OLD, text_url=FOREIGN)],
        extra={"nested": [OLD, "plain text"]},
    )

    result = await resign_presigned_urls([task], client=_client())

    fresh = result[0]
    for url in (fresh.training_data, fresh.image_text_pairs[0].image_url, fresh.extra["nested"][0]):
        assert urlparse(url).netloc == ENDPOINT
        assert urlparse(url).path == "/gradients-validator/abc_train_data.zip"
        assert _credential(url) == "NEWKEY"
    assert fresh.image_text_pairs[0].text_url == FOREIGN
    assert fresh.extra["nested"][1] == "plain text"


async def test_a_bare_url_string_is_re_signed():
    assert _credential(await resign_presigned_urls(OLD, client=_client())) == "NEWKEY"
