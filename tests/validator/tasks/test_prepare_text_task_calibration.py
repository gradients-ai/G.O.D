"""prepare_text_task on a quant instruct task: it always carves; a carve or a record leaves the test split unchanged."""

import json
import uuid
from collections import Counter
from datetime import datetime
from types import SimpleNamespace

import pytest
from datasets import Dataset

import validator.tasks.datasets.constants as cst
from core.models.dataset_models import FileFormat
from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from validator.tasks.datasets import preparation
from validator.tasks.datasets.calibration import CalibrationUnavailableError
from validator.tasks.datasets.calibration import raw_row_hash
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantFixedRawTask


N_ROWS = 9_000
DS = "org/fixture"

def _rows() -> list[dict]:
    rows = [
        {"question": f"question {i}", "context": f"ctx {i}", "answer": f"answer {i} " + "x" * (i % 40), "meta": i}
        for i in range(N_ROWS)
    ]
    # Exact duplicates of each other, so a duplicate can land on both sides of the split.
    for i in range(1, 400):
        rows[N_ROWS - i] = dict(rows[i])
    return rows

def _base_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The base split, reimplemented: one shuffled seed-42 train_test_split of min(10%, 1,000) rows."""
    split = Dataset.from_list(rows).train_test_split(
        test_size=min(int(len(rows) * cst.TRAIN_TEST_SPLIT_PERCENTAGE), cst.MAX_TEST_DATA_POINTS), shuffle=True, seed=42
    )
    return list(split["train"]), list(split["test"])

def _standard(row: dict) -> dict:
    return {
        cst.STANDARD_INSTRUCT_COLUMN: row["question"],
        cst.STANDARD_INPUT_COLUMN: row["context"],
        cst.STANDARD_OUTPUT_COLUMN: row["answer"],
    }

def _task(carve: bool = False, **overrides) -> QuantFixedRawTask:
    return QuantFixedRawTask(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.PREPARING_DATA,
        model_id="org/base",
        ds=DS,
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 6),
        dataset_type=InstructTextDatasetType(field_instruction="question", field_input="context", field_output="answer"),
        quant_spec=QuantFormat.W4A16_SYM_G128.spec(),
        file_format=FileFormat.S3,
        carve_calibration=carve,
        **overrides,
    )

@pytest.fixture
def seams(monkeypatch):
    """The fixture dataset in, uploaded JSON files captured by name, and a recording stand-in for the DB."""
    state = SimpleNamespace(rows=_rows(), uploads={}, recorded={}, record_calls=[])

    async def download(name, file_format, max_file_size_bytes=None):
        return Dataset.from_list(state.rows)

    async def upload(path, bucket, object_name):
        with open(path) as f:
            state.uploads[object_name.split("_", 1)[1]] = json.load(f)
        return f"https://minio/{object_name}?sig=secret"

    async def get_recorded(ds, psql_db, exclude_task_id=None):
        return set(state.recorded.get(ds, set())) if psql_db is not None else set()

    async def record(ds, hashes, task_id, psql_db):
        state.record_calls.append((ds, list(hashes), task_id))
        state.recorded.setdefault(ds, set()).update(hashes)

    monkeypatch.setattr(preparation, "download_and_load_dataset", download)
    monkeypatch.setattr(preparation, "upload_file_to_minio", upload)
    monkeypatch.setattr(preparation, "get_calibration_row_hashes", get_recorded)
    monkeypatch.setattr(preparation, "record_calibration_rows", record)
    return state

def _file(state, suffix: str) -> list[dict]:
    return state.uploads[suffix]

async def test_a_quant_task_carves_whatever_its_flag_says(seams):
    _, test = _base_split(seams.rows)

    split = await preparation.prepare_text_task(_task(carve=False), keypair=None, psql_db="db")

    assert _file(seams, "test_data.json") == [_standard(r) for r in test]
    assert split.calibration_url is not None and "calibration_data.json" in seams.uploads

async def test_carving_leaves_the_test_split_identical_and_train_is_the_rest(seams):
    train, test = _base_split(seams.rows)

    split = await preparation.prepare_text_task(_task(), keypair=None, psql_db="db")

    assert _file(seams, "test_data.json") == [_standard(r) for r in test]
    calibration = _file(seams, "calibration_data.json")
    new_train = _file(seams, "train_data.json")
    expected_train = [_standard(r) for r in train]
    # Train is the base train minus the calibration rows (as a multiset: duplicates exist), order preserved.
    assert len(calibration) == int(len(train) * cst.QUANT_CALIBRATION_MAX_TRAIN_FRACTION)
    key = lambda row: json.dumps(row, sort_keys=True)  # noqa: E731
    assert Counter(map(key, new_train)) + Counter(map(key, calibration)) == Counter(map(key, expected_train))
    positions = iter(map(key, expected_train))
    assert all(any(k == candidate for candidate in positions) for k in map(key, new_train))  # a subsequence: order kept
    assert split.calibration_url is not None and len(split.calibration_row_hashes) == len(calibration)

async def test_no_calibration_row_is_a_test_row_even_with_a_duplicate_in_train(seams):
    _, test = _base_split(seams.rows)
    test_keys = {json.dumps(_standard(r), sort_keys=True) for r in test}
    assert any(json.dumps(_standard(r), sort_keys=True) in test_keys for r in _base_split(seams.rows)[0])  # fixture has one

    await preparation.prepare_text_task(_task(), keypair=None, psql_db="db")

    assert not any(json.dumps(row, sort_keys=True) in test_keys for row in _file(seams, "calibration_data.json"))

async def test_carve_is_deterministic_per_task_id(seams):
    task = _task()
    await preparation.prepare_text_task(task, keypair=None, psql_db=None)
    first = _file(seams, "calibration_data.json")
    await preparation.prepare_text_task(task, keypair=None, psql_db=None)

    assert _file(seams, "calibration_data.json") == first

async def test_carved_hashes_are_recorded_for_the_dataset_after_upload(seams):
    task = _task()

    split = await preparation.prepare_text_task(task, keypair=None, psql_db="db")

    [(ds, hashes, task_id)] = seams.record_calls
    assert (ds, task_id) == (DS, task.task_id) and hashes == split.calibration_row_hashes
    assert set(hashes) <= {raw_row_hash(r) for r in seams.rows}

async def test_without_a_database_nothing_is_recorded_but_the_carve_still_happens(seams):
    await preparation.prepare_text_task(_task(), keypair=None, psql_db=None)

    assert "calibration_data.json" in seams.uploads

async def test_recorded_rows_never_land_in_test_and_are_appended_to_train(seams):
    train, test = _base_split(seams.rows)
    # Record rows the base split would have TESTED, plus some it would have trained on.
    tested = [raw_row_hash(r) for r in test[:30]]
    trained = [raw_row_hash(r) for r in train[:30]]
    seams.recorded[DS] = set(tested) | set(trained)

    await preparation.prepare_text_task(_task(), keypair=None, psql_db="db")

    new_test = _file(seams, "test_data.json")
    new_train = _file(seams, "train_data.json")
    calibration = _file(seams, "calibration_data.json")
    recorded_standard = {json.dumps(_standard(r), sort_keys=True) for r in seams.rows if raw_row_hash(r) in seams.recorded[DS]}
    assert len(new_test) == len(test)  # the size is still the full dataset's 10 % cap
    assert not any(json.dumps(row, sort_keys=True) in recorded_standard for row in new_test)
    assert len(new_train) + len(calibration) + len(new_test) == N_ROWS

async def test_a_record_that_matches_no_row_leaves_the_test_split_alone(seams):
    _, test = _base_split(seams.rows)
    seams.recorded[DS] = {"0" * 32}

    await preparation.prepare_text_task(_task(), keypair=None, psql_db="db")

    assert _file(seams, "test_data.json") == [_standard(r) for r in test]

async def test_carve_after_a_record_still_excludes_recorded_rows_from_test(seams):
    first = _task()
    await preparation.prepare_text_task(first, keypair=None, psql_db="db")
    first_calibration = {json.dumps(r, sort_keys=True) for r in _file(seams, "calibration_data.json")}
    seams.uploads.clear()

    await preparation.prepare_text_task(_task(), keypair=None, psql_db="db")

    new_test = {json.dumps(r, sort_keys=True) for r in _file(seams, "test_data.json")}
    assert new_test.isdisjoint(first_calibration)

async def test_a_given_split_is_never_carved_and_carries_the_calibration_url_through(seams):
    seams.uploads.clear()
    task = _task(
        carve=True,
        training_data="https://minio/train.json",
        test_data="https://minio/test.json",
        calibration_data="https://minio/cal.json",
    )
    seams.rows = [_standard(r) for r in _rows()]
    task.dataset_type = InstructTextDatasetType(
        field_instruction=cst.STANDARD_INSTRUCT_COLUMN,
        field_input=cst.STANDARD_INPUT_COLUMN,
        field_output=cst.STANDARD_OUTPUT_COLUMN,
    )

    split = await preparation.prepare_text_task(task, keypair=None, psql_db="db")

    assert "calibration_data.json" not in seams.uploads
    assert split.calibration_url == "https://minio/cal.json"
    assert seams.record_calls == []

async def test_env_task_still_returns_dummy_urls():
    split = await preparation.prepare_text_task(preparation.EnvRawTask.model_construct(), keypair=None)

    assert split.test_url == split.train_url == "env_task_dummy_dataset"


async def test_plain_instruct_task_ignores_recorded_calibration_rows(seams, monkeypatch):
    async def accept_dataset(task, train_dataset):
        return None

    monkeypatch.setattr(preparation, "reject_near_duplicate_dataset", accept_dataset)
    seams.recorded[DS] = {raw_row_hash(r) for r in seams.rows[:50]}
    task = InstructTextRawTask(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.PREPARING_DATA,
        model_id="org/base",
        ds=DS,
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 6),
        field_instruction="question",
        field_input="context",
        field_output="answer",
        file_format=FileFormat.S3,
    )

    split = await preparation.prepare_text_task(task, keypair=None, psql_db="db")

    assert split.calibration_url is None and seams.record_calls == []
    assert "calibration_data.json" not in seams.uploads


async def test_a_quant_task_whose_train_split_cannot_give_up_calibration_fails_prep(seams, monkeypatch):
    monkeypatch.setattr(preparation, "carve_calibration", lambda *args: None)  # the train split is under the minimum carve

    with pytest.raises(CalibrationUnavailableError, match="cannot give one up"):
        await preparation.prepare_text_task(_task(), keypair=None, psql_db="db")


async def test_a_flagged_instruct_task_too_small_to_carve_fails_prep(seams, monkeypatch):
    async def accept_dataset(task, train_dataset):
        return None

    monkeypatch.setattr(preparation, "reject_near_duplicate_dataset", accept_dataset)
    monkeypatch.setattr(preparation, "carve_calibration", lambda *args: None)
    task = InstructTextRawTask(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.PREPARING_DATA,
        model_id="org/base",
        ds=DS,
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 11, 6),
        field_instruction="question",
        field_input="context",
        field_output="answer",
        file_format=FileFormat.S3,
        carve_calibration=True,
    )

    with pytest.raises(CalibrationUnavailableError):
        await preparation.prepare_text_task(task, keypair=None, psql_db="db")
