import pytest
from pydantic import ValidationError

from core.image_registry import ImageRegistryConfig
from core.image_registry import blob_digest_from_key
from core.image_registry import blob_key
from core.image_registry import load_registry_config
from core.image_registry import manifest_child_digests
from core.image_registry import repository_key
from core.image_registry import repository_name
from core.image_registry import tag_link_keys
from core.image_registry import tournament_repository_prefix
from core.models.payload_models import PreparedTrainingImage
from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainRequestText


D1 = "sha256:" + "a" * 64
D2 = "sha256:" + "b" * 64
CONFIG = ImageRegistryConfig(endpoint="https://acct.r2.cloudflarestorage.com", bucket="b", access_key="k", secret_key="s")


def test_repository_name_is_lowercase_stable_and_case_sensitive_on_hotkey():
    a = repository_name("tourn_e63574fc205e9b06_20261116", "5GU4Xkd3dCGTU3s8VLcHGc5wsD5M8XyxDca5yDQhYm1mVXFu", "text")
    b = repository_name("tourn_e63574fc205e9b06_20261116", "5gu4xkd3dcgtu3s8vlchgc5wsd5m8xyxdca5ydqhym1mvxfu", "text")
    assert a == a.lower()
    assert a.startswith(tournament_repository_prefix("tourn_e63574fc205e9b06_20261116") + "/")
    assert a != b  # hotkeys differing only in case must not share a repository
    assert repository_name("t", "5Abc", "image-toolkit").endswith("-image-toolkit")


def test_tournament_prefixes_do_not_collide():
    key_1 = repository_key(CONFIG, tournament_repository_prefix("tourn_1"))
    key_10 = repository_key(CONFIG, tournament_repository_prefix("tourn_10"))
    assert not key_10.startswith(key_1)


def test_registry_layout_matches_distribution():
    assert blob_key(CONFIG, D1) == f"registry/docker/registry/v2/blobs/sha256/aa/{'a' * 64}/data"
    index, current = tag_link_keys(CONFIG, "god/t/x", "b1-a1", D1)
    assert current == "registry/docker/registry/v2/repositories/god/t/x/_manifests/tags/b1-a1/current/link"
    assert index.endswith(f"/_manifests/tags/b1-a1/index/sha256/{'a' * 64}/link")
    assert blob_digest_from_key(blob_key(CONFIG, D1)) == D1
    assert blob_digest_from_key("registry/docker/registry/v2/blobs/sha256/aa/") is None
    with pytest.raises(ValueError):
        blob_key(CONFIG, "sha256:nothex")


def test_manifest_child_digests_for_image_and_index():
    assert manifest_child_digests({"config": {"digest": D1}, "layers": [{"digest": D2}]}) == [D1, D2]
    assert manifest_child_digests({"manifests": [{"digest": D2}]}) == [D2]


def test_load_registry_config_requires_all_values(monkeypatch):
    for name in ("IMAGE_REGISTRY_ENDPOINT", "IMAGE_REGISTRY_BUCKET", "IMAGE_REGISTRY_ACCESS_KEY", "IMAGE_REGISTRY_SECRET_KEY"):
        monkeypatch.setenv(name, "x")
    assert load_registry_config() is not None
    monkeypatch.setenv("IMAGE_REGISTRY_SECRET_KEY", "")
    assert load_registry_config() is None


def test_trainer_proxy_request_is_backwards_compatible_and_validates_digest():
    data = TrainRequestText(
        model="m",
        task_id="t",
        hours_to_complete=1,
        expected_repo_name="r",
        dataset="s3://d",
        dataset_type={"field_instruction": "i", "field_output": "o"},
        file_format="s3",
    )
    request = TrainerProxyRequest(training_data=data, github_repo="https://github.com/a/b", gpu_ids=[0], hotkey="h")
    assert request.prepared_image is None
    with pytest.raises(ValidationError):
        PreparedTrainingImage(build_id="x", repository="god/t/x", digest="sha256:short", variant="text")
