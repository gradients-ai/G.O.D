"""run_quant_task_prep and the calibration set: tokenised like the eval set, eval-equal rows dropped, never fails prep."""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.tasks.prep.constants as prep_cst
from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from validator.evaluation.quant_loss import load_examples
from validator.evaluation.quant_loss import quant_eval_set_fingerprint
from validator.tasks import requests
from validator.tasks.datasets.preparation import PreparedTextSplit
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tasks.quant_eval_set import build_quant_eval_set


TRAIN_URL = "https://minio/train.json"
TEST_URL = "https://minio/test.json"
CAL_URL = "https://minio/cal.json"
RAW_INSTRUCT = InstructTextDatasetType(field_instruction="question", field_output="answer", field_input="context")


class FakeTokenizer:
    vocab_size = 256
    context_len = None
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return ([1] if add_special_tokens else []) + [ord(c) % 250 for c in text]


def _rows(start: int, n: int) -> list[dict]:
    return [{"instruct": f"q{i}", "input": f"c{i}", "output": f"a{i}"} for i in range(start, start + n)]


def _fields() -> dict:
    return dict(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.PREPARING_DATA,
        model_id="org/base",
        ds="org/data",
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 6),
        dataset_type=RAW_INSTRUCT,
    )


def _freeform() -> QuantFreeformRawTask:
    return QuantFreeformRawTask(**_fields())


def _fixed() -> QuantFixedRawTask:
    return QuantFixedRawTask(**_fields(), quant_spec=QuantFormat.W4A16_SYM_G128.spec())


@pytest.fixture
def prep(monkeypatch, tmp_path):
    """Test rows 0..149; a calibration file of rows 150..399 unless a test changes it; uploads copied by object name."""
    seams = SimpleNamespace(
        test_rows=_rows(0, 150),
        calibration_rows=_rows(150, 250),
        calibration_url=CAL_URL,
        downloads=[],
        uploads={},
        warnings=[],
        download_error=None,
    )

    async def download(url, save_path=None):
        seams.downloads.append(url)
        if url == CAL_URL and seams.download_error:
            raise seams.download_error
        rows = seams.test_rows if url == TEST_URL else seams.calibration_rows
        path = os.path.join(save_path, f"{len(seams.downloads)}.json")
        with open(path, "w") as f:
            json.dump(rows, f)
        return path

    async def upload(path, bucket, object_name):
        suffix = object_name.split("_", 1)[1]
        shutil.copy(path, tmp_path / suffix)
        seams.uploads[suffix] = tmp_path / suffix
        return f"https://minio/{suffix}"

    seams.prepare_text_task = AsyncMock(
        side_effect=lambda *args, **kwargs: PreparedTextSplit(
            test_url=TEST_URL, train_url=TRAIN_URL, calibration_url=seams.calibration_url
        )
    )
    monkeypatch.setattr(requests, "prepare_text_task", seams.prepare_text_task)
    monkeypatch.setattr(requests, "download_s3_file", download)
    monkeypatch.setattr(requests, "upload_file_to_minio", upload)
    monkeypatch.setattr(requests.logger, "warning", lambda message: seams.warnings.append(message))
    return seams


async def _run(task):
    return await requests.run_quant_task_prep(task, keypair=None, psql_db="db", tokenizer_loader=lambda _: FakeTokenizer())


def _eval_examples(rows: list[dict]):
    dataset_type = requests._standard_instruct_dataset_type(RAW_INSTRUCT)
    return build_quant_eval_set(rows, dataset_type, FakeTokenizer(), prep_cst.QUANT_EVAL_MAX_SEQUENCE_LEN).examples


async def test_calibration_is_tokenised_with_the_eval_rules_and_published(prep):
    task = await _run(_freeform())

    published = load_examples(Path(prep.uploads["quant_calibration_set.jsonl"]))
    assert published == _eval_examples(prep.calibration_rows)
    assert task.calibration_data == "https://minio/quant_calibration_set.jsonl"
    assert task.calibration_fingerprint == quant_eval_set_fingerprint(published)
    assert task.calibration_n_examples == 250
    assert load_examples(Path(prep.uploads["quant_eval_set.jsonl"])) == _eval_examples(prep.test_rows)


async def test_calibration_examples_equal_to_an_eval_example_are_dropped(prep):
    prep.calibration_rows = _rows(150, 250) + _rows(0, 20)  # 20 rows identical to test rows

    task = await _run(_freeform())

    published = load_examples(Path(prep.uploads["quant_calibration_set.jsonl"]))
    eval_keys = {tuple(e.input_ids) for e in _eval_examples(prep.test_rows)}
    assert len(published) == 250 and task.calibration_n_examples == 250
    assert not any(tuple(e.input_ids) in eval_keys for e in published)


async def test_fewer_than_the_minimum_publishes_nothing_and_warns(prep):
    prep.calibration_rows = _rows(150, prep_cst.QUANT_CALIBRATION_MIN_EXAMPLES - 1)

    task = await _run(_freeform())

    assert "quant_calibration_set.jsonl" not in prep.uploads
    assert (task.calibration_data, task.calibration_fingerprint, task.calibration_n_examples) == (None, None, None)
    assert task.status == TaskStatus.LOOKING_FOR_NODES
    assert any("publishing no calibration set" in message for message in prep.warnings)


async def test_a_calibration_failure_never_fails_prep(prep):
    prep.download_error = RuntimeError("minio down")

    task = await _run(_freeform())

    assert task.calibration_data is None and task.status == TaskStatus.LOOKING_FOR_NODES
    assert task.test_data == "https://minio/quant_eval_set.jsonl"


async def test_fixed_task_publishes_the_calibration_set_like_a_freeform_one(prep):
    task = await _run(_fixed())

    published = load_examples(Path(prep.uploads["quant_calibration_set.jsonl"]))
    assert published == _eval_examples(prep.calibration_rows)
    assert task.calibration_data == "https://minio/quant_calibration_set.jsonl"
    assert task.calibration_fingerprint == quant_eval_set_fingerprint(published)
    assert task.calibration_n_examples == 250
    assert load_examples(Path(prep.uploads["quant_eval_set.jsonl"])) == _eval_examples(prep.test_rows)


async def test_fixed_task_without_a_calibration_url_publishes_nothing(prep):
    prep.calibration_url = None
    task = _fixed()
    task.calibration_data = "https://minio/stale.json"  # never published as is: only prep's own tokenised set counts

    task = await _run(task)

    assert (task.calibration_data, task.calibration_fingerprint, task.calibration_n_examples) == (None, None, None)
    assert prep.downloads == [TEST_URL]
    assert "quant_calibration_set.jsonl" not in prep.uploads


async def test_fixed_calibration_examples_equal_to_an_eval_example_are_dropped(prep):
    prep.calibration_rows = _rows(150, 250) + _rows(0, 20)

    task = await _run(_fixed())

    published = load_examples(Path(prep.uploads["quant_calibration_set.jsonl"]))
    eval_keys = {tuple(e.input_ids) for e in _eval_examples(prep.test_rows)}
    assert len(published) == 250 and task.calibration_n_examples == 250
    assert not any(tuple(e.input_ids) in eval_keys for e in published)


async def test_fixed_calibration_failure_never_fails_prep_and_redacts_the_url(prep):
    prep.download_error = RuntimeError(f"cannot fetch {CAL_URL}?X-Amz-Signature=secret")

    task = await _run(_fixed())

    assert task.calibration_data is None and task.status == TaskStatus.LOOKING_FOR_NODES
    assert not any("secret" in message for message in prep.warnings)


async def test_uncalibrated_task_prep_is_unchanged_and_has_no_calibration(prep):
    prep.calibration_url = None

    task = await _run(_freeform())

    assert prep.downloads == [TEST_URL]
    assert list(prep.uploads) == ["quant_eval_set.jsonl"]
    assert load_examples(Path(prep.uploads["quant_eval_set.jsonl"])) == _eval_examples(prep.test_rows)
    assert (task.calibration_data, task.calibration_fingerprint, task.calibration_n_examples) == (None, None, None)
