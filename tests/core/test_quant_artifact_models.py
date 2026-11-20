"""quant_manifest.json schema (QuantManifest) and the task-type to track mapping."""

import pytest
from pydantic import ValidationError

from core.constants.quant import QUANT_MANIFEST_MAX_FILES
from core.constants.quant import QUANT_MANIFEST_NAME
from core.models.quant_artifact_models import QuantManifest
from core.models.quant_artifact_models import track_for
from core.models.task_models import TaskType


def _manifest(track: str = "fixed", files: list[dict] | None = None, **extra) -> dict:
    files = files if files is not None else [{"name": "model.safetensors", "bytes": 10}]
    return {"version": 1, "task_id": "task-1", "track": track, "files": files, **extra}


def test_accepts_fixed_manifest():
    manifest = QuantManifest.model_validate(
        _manifest("fixed", [{"name": "model.safetensors", "bytes": 10}, {"name": "config.json", "bytes": 0}])
    )

    assert manifest.track == "fixed"
    assert [f.name for f in manifest.files] == ["model.safetensors", "config.json"]


def test_accepts_freeform_manifest_with_nested_names():
    manifest = QuantManifest.model_validate(_manifest("freeform", [{"name": "codes/layer0.bin", "bytes": 5}]))

    assert manifest.track == "freeform"
    assert manifest.files[0].name == "codes/layer0.bin"


@pytest.mark.parametrize(
    "payload",
    [
        _manifest(extra_field=1),
        _manifest(version=2),
        _manifest(track="other"),
        _manifest(files=[{"name": "a.bin", "bytes": 1}, {"name": "a.bin", "bytes": 1}]),
        _manifest(files=[{"name": "a.bin", "bytes": 1}, {"name": "./a.bin", "bytes": 1}]),
        _manifest(files=[{"name": "/etc/passwd", "bytes": 1}]),
        _manifest(files=[{"name": "../other/model.safetensors", "bytes": 1}]),
        _manifest(files=[{"name": "codes/../../x", "bytes": 1}]),
        _manifest(files=[{"name": QUANT_MANIFEST_NAME, "bytes": 1}]),
        _manifest(files=[{"name": "", "bytes": 1}]),
        _manifest(files=[{"name": "a.bin", "bytes": -1}]),
        _manifest(files=[{"name": "a.bin", "bytes": 1, "sha256": "x"}]),
        _manifest(files=[]),
        _manifest(files=[{"name": f"shard-{i}.bin", "bytes": 1} for i in range(QUANT_MANIFEST_MAX_FILES + 1)]),
    ],
    ids=[
        "unknown-field",
        "version",
        "track",
        "duplicate",
        "duplicate-normalised",
        "absolute",
        "dotdot",
        "nested-dotdot",
        "manifest-name",
        "empty-name",
        "negative-bytes",
        "unknown-file-field",
        "zero-files",
        "too-many-files",
    ],
)
def test_rejects_invalid_manifest(payload: dict):
    with pytest.raises(ValidationError):
        QuantManifest.model_validate(payload)


def test_accepts_max_files():
    files = [{"name": f"shard-{i}.bin", "bytes": 1} for i in range(QUANT_MANIFEST_MAX_FILES)]

    assert len(QuantManifest.model_validate(_manifest(files=files)).files) == QUANT_MANIFEST_MAX_FILES


def test_track_for_maps_both_quant_task_types():
    assert track_for(TaskType.QUANTFIXEDTASK) == "fixed"
    assert track_for(TaskType.QUANTFREEFORMTASK) == "freeform"
