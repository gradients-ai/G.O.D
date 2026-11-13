import json
import zipfile

import pytest

from core.models.image_models import ImageTrainingMode
from ops.docker.scripts import image_toolkit_entrypoint


def test_edit_manifest_is_converted_to_ai_toolkit_control_layout(monkeypatch, tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    cache_root = tmp_path / "cache" / "datasets"
    image_root = tmp_path / "prepared"
    cache_root.mkdir(parents=True)
    monkeypatch.setattr(image_toolkit_entrypoint, "DATASET_ZIP_ROOT", cache_root)
    monkeypatch.setattr(image_toolkit_entrypoint, "IMAGE_DATASET_ROOT", image_root)
    monkeypatch.setattr(image_toolkit_entrypoint, "EXTRACT_ROOT", tmp_path / "extracted")

    source_root = tmp_path / "source"
    (source_root / "targets").mkdir(parents=True)
    (source_root / "sources" / "sample-a").mkdir(parents=True)
    image_module.new("RGB", (32, 32), "green").save(source_root / "targets" / "sample-a.png")
    image_module.new("RGB", (32, 32), "red").save(source_root / "sources" / "sample-a" / "0.png")
    manifest_row = {
        "sample_id": "sample-a",
        "target": "targets/sample-a.png",
        "instruction": "Change the red square to green.",
        "sources": ["sources/sample-a/0.png"],
    }
    (source_root / "manifest.jsonl").write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")

    archive_path = cache_root / "task_tourn.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in source_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source_root))

    target_path, control_paths = image_toolkit_entrypoint.prepare_dataset(
        "task",
        ImageTrainingMode.EDIT.value,
    )

    assert target_path == image_root / "targets"
    assert control_paths == [image_root / "controls" / "0"]
    assert (target_path / "sample-a.png").is_file()
    assert (target_path / "sample-a.txt").read_text() == "Change the red square to green.\n"
    assert (control_paths[0] / "sample-a.png").is_file()
