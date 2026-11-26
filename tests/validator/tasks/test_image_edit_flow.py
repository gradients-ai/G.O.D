import json
import zipfile
from pathlib import Path

from PIL import Image

from core.models.image_models import IMAGE_MODEL_CAPABILITIES
from core.models.image_models import ImageModelType
from core.models.image_models import ImageTrainingMode
from validator.evaluation.image_test_data import decoded_generation_images
from validator.evaluation.image_test_data import read_generation_dataset
from validator.tasks.datasets.preparation import train_test_split_image


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)


def test_qwen_edit_has_edit_only_capability():
    assert IMAGE_MODEL_CAPABILITIES[ImageModelType.QWEN_IMAGE_EDIT_2511] == frozenset(
        {ImageTrainingMode.EDIT}
    )


def test_edit_split_writes_manifest_and_preserves_sources(tmp_path):
    for index in range(10):
        _write_image(tmp_path / f"{index}.png", (index, 20, 30))
        (tmp_path / f"{index}.txt").write_text(f"Apply edit goal {index}")
        _write_image(tmp_path / "controls" / "0" / f"{index}.png", (30, 20, index))

    test_zip, train_zip = train_test_split_image(tmp_path, require_controls=True)

    for archive_path in (test_zip, train_zip):
        with zipfile.ZipFile(archive_path) as archive:
            manifests = [name for name in archive.namelist() if name.endswith("manifest.jsonl")]
            assert len(manifests) == 1
            rows = [
                json.loads(line)
                for line in archive.read(manifests[0]).decode("utf-8").splitlines()
                if line.strip()
            ]
            assert rows
            for row in rows:
                base = Path(manifests[0]).parent
                assert (base / row["target"]).as_posix() in archive.namelist()
                assert (base / row["sources"][0]).as_posix() in archive.namelist()


def test_generation_eval_reads_edit_manifest(tmp_path):
    root = tmp_path / "edit"
    _write_image(root / "targets" / "sample.png", (100, 110, 120))
    _write_image(root / "sources" / "sample.png", (10, 20, 30))
    (root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "instruction": "Apply the shared transformation.",
                "sources": ["sources/sample.png"],
                "target": "targets/sample.png",
            }
        )
        + "\n"
    )

    rows = read_generation_dataset(root, tmp_path / "unused.zip", "edit")
    decoded = decoded_generation_images(rows, "edit")

    assert decoded[0]["caption"] == "Apply the shared transformation."
    assert decoded[0]["source"].size == (1024, 1024)
    assert decoded[0]["image"].size == (1024, 1024)
