"""A continuous-SFT task holds a calibration set out of its chunk's train rows at prep, and the compressions of its model take
exactly those rows: none of them is a row the model trained on, and none is a test row."""

import json
import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest
from datasets import Dataset

import validator.tasks.datasets.constants as cst
from core.constants.environments import TrainingStartPoint
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import FileFormat
from core.models.task_models import TaskStatus
from validator.tasks.datasets import preparation
from validator.tasks.datasets.calibration import CalibrationUnavailableError
from validator.tasks.models import ChatRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tournament import constants as t_cst
from validator.tournament.models import ContinuousSftTaskChunk


N_TRAIN = 9_000
N_TEST = 300
CHUNK_INDEX = 7
TRAIN_URL = "https://minio/chunk-train.json"
TEST_URL = "https://minio/chunk-test.json"


def _conversation(i: int) -> dict:
    answer = f"answer {i} " + "x" * (i % 60)
    return {"conversations": [{"from": "human", "value": f"question {i}"}, {"from": "gpt", "value": answer}]}


def _rows(start: int, n: int) -> list[dict]:
    return [_conversation(i) for i in range(start, start + n)]


def _key(row: dict) -> str:
    return json.dumps(row, sort_keys=True)


def _text_task(carve: bool = True, label: str = "chunk-7") -> ChatRawTask:
    return ChatRawTask(
        task_id=uuid.uuid4(),
        model_id="org/carried-winner",
        ds=t_cst.continuous_sft_ds("qwen3-8b", label),
        status=TaskStatus.PREPARING_DATA,
        is_organic=False,
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 7),
        file_format=FileFormat.S3,
        training_data=TRAIN_URL,
        test_data=TEST_URL,
        training_start_point=TrainingStartPoint.CONTINUOUS_SFT,
        chat_template="tokenizer_default",
        carve_calibration=carve,
    )


def _compression_task(calibration_data: str | None) -> QuantFreeformRawTask:
    return QuantFreeformRawTask(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.PREPARING_DATA,
        model_id="hub/boss-continuous-output",
        ds=t_cst.continuous_quant_ds(CHUNK_INDEX, "quant-tourn", "another-label"),
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 7),
        dataset_type=ChatTemplateDatasetType(chat_template="tokenizer_default", chat_column="conversations"),
        file_format=FileFormat.S3,
        training_data=TRAIN_URL,
        test_data=TEST_URL,
        calibration_data=calibration_data,
        training_start_point=TrainingStartPoint.DEFAULT,
    )


@pytest.fixture
def seams(monkeypatch):
    state = SimpleNamespace(
        train=_rows(0, N_TRAIN), test=_rows(N_TRAIN, N_TEST), uploads={}, urls={}, record_calls=[], chunks={}
    )

    async def download(name, file_format, max_file_size_bytes=None):
        return Dataset.from_list(state.train if name == TRAIN_URL else state.test)

    async def upload(path, bucket, object_name):
        with open(path) as f:
            rows = json.load(f)
        url = f"https://minio/{object_name}?sig=secret"
        state.uploads[object_name.split("_", 1)[1]] = rows
        state.urls[url] = rows
        return url

    async def record(ds, hashes, task_id, psql_db):
        state.record_calls.append((ds, hashes, task_id))

    async def chunk_of(task_id, psql_db):
        index = state.chunks.get(task_id)
        if index is None:
            return None
        return ContinuousSftTaskChunk(
            task_id=task_id, lineage="qwen3-8b", train_index=index, chunk_scheme=t_cst.CONTINUOUS_SFT_CHUNK_SCHEME
        )

    monkeypatch.setattr(preparation, "download_and_load_dataset", download)
    monkeypatch.setattr(preparation, "upload_file_to_minio", upload)
    monkeypatch.setattr(preparation, "record_calibration_rows", record)
    monkeypatch.setattr(preparation, "get_continuous_sft_task_chunk", chunk_of)
    return state


async def _prep_text(seams, task: ChatRawTask, train_index: int = CHUNK_INDEX):
    seams.chunks[task.task_id] = train_index
    return await preparation.prepare_text_task(task, keypair=None, psql_db="db")


async def test_the_text_task_trains_without_its_calibration_rows_and_its_test_split_is_untouched(seams):
    split = await _prep_text(seams, _text_task())

    calibration = seams.uploads["calibration_data.json"]
    train = seams.uploads["train_data.json"]
    assert split.calibration_url is not None and split.calibration_row_hashes
    assert cst.QUANT_CALIBRATION_MIN_ROWS <= len(calibration) <= cst.QUANT_CALIBRATION_MAX_ROWS
    assert len(train) + len(calibration) == N_TRAIN
    assert {_key(r) for r in calibration}.isdisjoint({_key(r) for r in train})
    assert {_key(r) for r in calibration} <= {_key(r) for r in seams.train}
    assert seams.uploads["test_data.json"] == seams.test
    assert seams.record_calls == []  # a given split is never re-split, so nothing is recorded per dataset


async def test_calibration_rows_never_include_a_test_row(seams):
    seams.train[10] = dict(seams.test[0])  # duplicates of test rows sit in the train file
    seams.train[11] = dict(seams.test[1])

    await _prep_text(seams, _text_task())

    test_keys = {_key(r) for r in seams.test}
    assert not any(_key(r) in test_keys for r in seams.uploads["calibration_data.json"])


async def test_the_carve_is_reproducible_per_chunk_whatever_the_task_or_label(seams):
    await _prep_text(seams, _text_task(label="chunk-7"))
    first = seams.uploads["calibration_data.json"]
    seams.uploads.clear()

    await _prep_text(seams, _text_task(label="a-different-label"))
    assert seams.uploads["calibration_data.json"] == first
    seams.uploads.clear()

    await _prep_text(seams, _text_task(), train_index=CHUNK_INDEX + 1)
    assert seams.uploads["calibration_data.json"] != first


async def test_a_task_with_no_chunk_record_fails_prep(seams):
    with pytest.raises(CalibrationUnavailableError, match="no chunk record"):
        await preparation.prepare_text_task(_text_task(), keypair=None, psql_db="db")


async def test_an_unflagged_continuous_task_trains_on_every_row(seams):
    split = await _prep_text(seams, _text_task(carve=False))

    assert split.calibration_url is None and "calibration_data.json" not in seams.uploads
    assert len(seams.uploads["train_data.json"]) == N_TRAIN


async def test_the_compression_task_takes_exactly_the_rows_the_text_task_held_out(seams):
    text_split = await _prep_text(seams, _text_task())
    held_out = seams.urls[text_split.calibration_url]
    text_train = {_key(r) for r in seams.uploads["train_data.json"]}
    seams.uploads.clear()

    quant_split = await preparation.prepare_text_task(
        _compression_task(text_split.calibration_url), keypair=None, psql_db="db"
    )

    assert quant_split.calibration_url == text_split.calibration_url  # passed through, not carved again
    assert "calibration_data.json" not in seams.uploads
    assert not any(_key(r) in text_train for r in held_out)  # the input model never trained on them
    assert {_key(r) for r in held_out}.isdisjoint({_key(r) for r in seams.uploads["test_data.json"]})
