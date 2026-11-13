"""prepare_text_task on a quant task with a given split: the freeform / fixed compression tasks carve a calibration set
unless they carry calibration rows."""

import json
import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest
from datasets import Dataset

import validator.tasks.datasets.constants as cst
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import FileFormat
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from validator.tasks.datasets import preparation
from validator.tasks.datasets.calibration import CalibrationUnavailableError
from validator.tasks.datasets.calibration import shared_split_seed
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask


N_TRAIN = 9_000
N_TEST = 300
TRAIN_URL = "https://minio/chunk-train.json"
TEST_URL = "https://minio/chunk-test.json"
CHAT = ChatTemplateDatasetType(chat_template="tokenizer_default", chat_column="conversations")


def _conversation(i: int) -> dict:
    answer = f"answer {i} " + "x" * (i % 60)
    return {"conversations": [{"from": "human", "value": f"question {i}"}, {"from": "gpt", "value": answer}]}


def _rows(start: int, n: int) -> list[dict]:
    return [_conversation(i) for i in range(start, start + n)]


def _task(cls=QuantFreeformRawTask, carve=True, label="chunk-7", **overrides):
    fields = dict(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.PREPARING_DATA,
        model_id="org/base",
        ds=f"continuous-quant:7:{label}",  # no tournament id: the whole chunk is kept
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 7),
        dataset_type=CHAT,
        file_format=FileFormat.S3,
        training_data=TRAIN_URL,
        test_data=TEST_URL,
        carve_calibration=carve,
        **overrides,
    )
    if cls is QuantFixedRawTask:
        fields["quant_spec"] = QuantFormat.W4A16_SYM_G128.spec()
    return cls(**fields)


@pytest.fixture
def seams(monkeypatch):
    state = SimpleNamespace(train=_rows(0, N_TRAIN), test=_rows(N_TRAIN, N_TEST), uploads={}, record_calls=[])

    async def download(name, file_format, max_file_size_bytes=None):
        return Dataset.from_list(state.train if name == TRAIN_URL else state.test)

    async def upload(path, bucket, object_name):
        with open(path) as f:
            state.uploads[object_name.split("_", 1)[1]] = json.load(f)
        return f"https://minio/{object_name}?sig=secret"

    async def record(ds, hashes, task_id, psql_db):
        state.record_calls.append((ds, hashes, task_id))

    monkeypatch.setattr(preparation, "download_and_load_dataset", download)
    monkeypatch.setattr(preparation, "upload_file_to_minio", upload)
    monkeypatch.setattr(preparation, "record_calibration_rows", record)
    return state


def _key(row: dict) -> str:
    return json.dumps(row, sort_keys=True)


async def test_given_split_carves_from_the_train_file_and_reuploads_train_without_the_rows(seams):
    split = await preparation.prepare_text_task(_task(), keypair=None, psql_db="db")

    calibration = seams.uploads["calibration_data.json"]
    new_train = seams.uploads["train_data.json"]
    assert split.calibration_url is not None
    assert cst.QUANT_CALIBRATION_MIN_ROWS <= len(calibration) <= cst.QUANT_CALIBRATION_MAX_ROWS
    assert len(calibration) <= int(N_TRAIN * cst.QUANT_CALIBRATION_MAX_TRAIN_FRACTION)
    assert len(new_train) + len(calibration) == N_TRAIN
    assert {_key(r) for r in calibration}.isdisjoint({_key(r) for r in new_train})
    assert {_key(r) for r in calibration} <= {_key(r) for r in seams.train}
    assert seams.uploads["test_data.json"] == seams.test  # the test side is untouched


async def test_calibration_rows_never_include_a_test_row(seams):
    seams.train[10] = dict(seams.test[0])  # a duplicate of a test row sits in the train file
    seams.train[11] = dict(seams.test[1])

    await preparation.prepare_text_task(_task(), keypair=None, psql_db="db")

    test_keys = {_key(r) for r in seams.test}
    assert not any(_key(r) in test_keys for r in seams.uploads["calibration_data.json"])


async def test_sibling_tasks_on_one_chunk_carve_identical_rows_whatever_their_label_or_id(seams):
    await preparation.prepare_text_task(_task(QuantFreeformRawTask, label="chunk-7"), keypair=None, psql_db="db")
    freeform = seams.uploads["calibration_data.json"]
    seams.uploads.clear()

    await preparation.prepare_text_task(_task(QuantFixedRawTask, label="a-different-label"), keypair=None, psql_db="db")

    assert seams.uploads["calibration_data.json"] == freeform


async def test_a_different_chunk_carves_different_rows(seams):
    await preparation.prepare_text_task(_task(), keypair=None, psql_db="db")
    first = seams.uploads["calibration_data.json"]
    seams.uploads.clear()
    other = _task()
    other.ds = "continuous-quant:8:chunk-8"

    await preparation.prepare_text_task(other, keypair=None, psql_db="db")

    assert seams.uploads["calibration_data.json"] != first
    assert shared_split_seed(other.ds) != shared_split_seed(_task().ds)


async def test_the_flag_does_not_matter_a_given_split_is_carved(seams):
    split = await preparation.prepare_text_task(_task(carve=False), keypair=None, psql_db="db")

    assert "calibration_data.json" in seams.uploads and split.calibration_url is not None
    assert len(seams.uploads["train_data.json"]) < N_TRAIN


async def test_a_task_that_already_carries_calibration_rows_is_not_carved_again(seams):
    split = await preparation.prepare_text_task(_task(calibration_data="https://minio/cal.json"), keypair=None, psql_db="db")

    assert "calibration_data.json" not in seams.uploads
    assert split.calibration_url == "https://minio/cal.json"


async def test_a_chunk_too_small_to_give_the_minimum_fails_prep(seams):
    # Every train row duplicates a test row, so none is eligible and the minimum cannot be met.
    seams.train = [dict(seams.test[i % N_TEST]) for i in range(8_000)]

    with pytest.raises(CalibrationUnavailableError):
        await preparation.prepare_text_task(_task(), keypair=None, psql_db="db")


async def test_a_given_split_carve_is_not_recorded_per_dataset(seams):
    split = await preparation.prepare_text_task(_task(), keypair=None, psql_db="db")

    assert seams.record_calls == [] and split.calibration_row_hashes != []
