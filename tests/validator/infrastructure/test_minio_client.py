import pytest

from validator.infrastructure.minio_client import AsyncMinioClient


def _client(endpoint: str) -> AsyncMinioClient:
    return AsyncMinioClient(endpoint=endpoint, access_key="a", secret_key="s", region="us-east-1")


@pytest.mark.parametrize(
    "endpoint, url, expected",
    [
        # Virtual-host style: minio-py signs these for AWS endpoints only. The bucket is the first host label.
        (
            "s3.amazonaws.com",
            "https://my-bucket.s3.amazonaws.com/abc_test_data.json?X-Amz-Signature=x",
            ("my-bucket", "abc_test_data.json"),
        ),
        (
            "s3.us-east-1.amazonaws.com",
            "https://my-bucket.s3.us-east-1.amazonaws.com/abc_test_data.json?X-Amz-Signature=x",
            ("my-bucket", "abc_test_data.json"),
        ),
        # Path style on the client's own endpoint: the prod shape. Prod's S3_COMPATIBLE_ENDPOINT is a multi-label
        # non-AWS host, for which minio-py presigns host/bucket/key, so the first host label is never the bucket.
        (
            "s3.eu-central-1.objectstore.example.com",
            "https://s3.eu-central-1.objectstore.example.com/my-bucket/abc_test_data.json?X-Amz-Signature=x",
            ("my-bucket", "abc_test_data.json"),
        ),
        # Path style on the client's own endpoint (plain MinIO).
        (
            "localhost:9000",
            "http://localhost:9000/my-bucket/abc_test_data.json?X-Amz-Signature=x",
            ("my-bucket", "abc_test_data.json"),
        ),
        ("minio.internal:9000", "https://minio.internal:9000/b/dir/obj.json", ("b", "dir/obj.json")),
        # Path style on a host the client does not know: an IP literal or a dotless host never carries a bucket.
        ("s3.amazonaws.com", "http://10.0.0.5:9000/b/obj.json", ("b", "obj.json")),
        ("s3.amazonaws.com", "http://minio:9000/b/obj.json", ("b", "obj.json")),
        # Percent-encoded object names are decoded.
        ("localhost:9000", "http://localhost:9000/b/a%20b.json", ("b", "a b.json")),
    ],
)
def test_parse_s3_url(endpoint, url, expected):
    assert _client(endpoint).parse_s3_url(url) == expected


@pytest.mark.parametrize("endpoint", ["s3.eu-central-1.objectstore.example.com", "s3.us-east-1.amazonaws.com", "localhost:9000"])
def test_parse_round_trips_what_minio_py_presigns(endpoint):
    # get_new_presigned_url parses URLs this same client minted, whichever style minio-py picked for the endpoint.
    client = _client(endpoint)
    presigned = client.client.presigned_get_object("my-bucket", "dir/abc test_data.json")
    assert client.parse_s3_url(presigned) == ("my-bucket", "dir/abc test_data.json")
