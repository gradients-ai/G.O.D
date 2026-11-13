"""Quant task config dispatch, run_quant_task_prep (split, tokenise, upload) and the organic train request."""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import validator.tasks.datasets.constants as data_cst
import validator.tasks.prep.constants as prep_cst
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from core.models.task_models import TaskType
from validator.evaluation.quant_loss import load_examples
from validator.evaluation.quant_loss import quant_eval_set_fingerprint
from validator.tasks import requests
from validator.tasks.config import QuantTaskConfig
from validator.tasks.config import get_task_config
from validator.tasks.datasets.preparation import PreparedTextSplit
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask
from validator.tasks.models import RawTask
from validator.tasks.quant_eval_set import build_quant_eval_set


TRAIN_URL = "https://minio/train.json"
TEST_URL = "https://minio/test.json"
CAL_URL = "https://minio/cal.json"
EVAL_SET_URL = "https://minio/quant_eval_set.jsonl"
SPEC = QuantFormat.W4A16_SYM_G128.spec()
RAW_INSTRUCT = InstructTextDatasetType(
    field_instruction="question", field_output="answer", field_input="context", format="{instruction}\n{input}"
)
CHAT = ChatTemplateDatasetType(chat_template="tokenizer_default")


class FakeTokenizer:
    vocab_size = 256
    context_len = None
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return ([1] if add_special_tokens else []) + [ord(c) for c in text]

    def chat_labeller(self, chat_template: str | None) -> "FakeChatTokenizer":
        return FakeChatTokenizer()


class FakeChatTokenizer:
    def render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        return "".join(f"<{m['role']}>{m['content']}</>" for m in messages) + ("<assistant>" if add_generation_prompt else "")

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        return [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]


def _fields(**overrides) -> dict:
    return dict(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.PREPARING_DATA,
        model_id="org/base",
        ds="org/data",
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 10, 19),
        **overrides,
    )


def _fixed(dataset_type=RAW_INSTRUCT) -> QuantFixedRawTask:
    return QuantFixedRawTask(**_fields(), dataset_type=dataset_type, quant_spec=SPEC)


def _freeform(dataset_type=RAW_INSTRUCT, **overrides) -> QuantFreeformRawTask:
    return QuantFreeformRawTask(**_fields(**overrides), dataset_type=dataset_type)


def _instruct_rows(n: int) -> list[dict]:
    return [{"instruct": f"q{i}", "input": f"c{i}", "output": f"a{i}"} for i in range(n)]


def _chat_rows(n: int) -> list[dict]:
    return [{"conversations": [{"from": "user", "value": f"u{i}"}, {"from": "assistant", "value": f"a{i}"}]} for i in range(n)]


TEXT_KEYS = {"instruct", "input", "output", "value", "conversations"}


def _distinct(value):
    """The same row shape with every string changed, so no calibration row equals an eval row."""
    if isinstance(value, str):
        return value + "~"
    if isinstance(value, list):
        return [_distinct(item) for item in value]
    if isinstance(value, dict):
        return {key: _distinct(item) if key in TEXT_KEYS else item for key, item in value.items()}
    return value


@pytest.fixture
def prep(monkeypatch, tmp_path):
    """Patched split, download and upload; the last uploaded JSONL (the eval set) is copied to tmp_path/uploaded.jsonl."""
    seams = SimpleNamespace(rows=_instruct_rows(150), uploaded=tmp_path / "uploaded.jsonl", object_names=[])

    async def download(url, save_path=None):
        assert url in (TEST_URL, CAL_URL)
        path = os.path.join(save_path, "test.json")
        with open(path, "w") as f:
            json.dump(seams.rows if url == TEST_URL else _distinct(seams.rows), f)
        return path

    async def upload(path, bucket, object_name):
        shutil.copy(path, seams.uploaded)
        seams.object_names.append(object_name)
        return EVAL_SET_URL

    seams.prepare_text_task = AsyncMock(
        return_value=PreparedTextSplit(test_url=TEST_URL, train_url=TRAIN_URL, calibration_url=CAL_URL)
    )
    monkeypatch.setattr(requests, "prepare_text_task", seams.prepare_text_task)
    monkeypatch.setattr(requests, "download_s3_file", download)
    monkeypatch.setattr(requests, "upload_file_to_minio", upload)
    return seams


async def _run(task):
    loaded = []

    def loader(model_id: str) -> FakeTokenizer:
        loaded.append(model_id)
        return FakeTokenizer()

    task = await requests.run_quant_task_prep(task, keypair=None, psql_db="db", tokenizer_loader=loader)
    return task, loaded


@pytest.mark.parametrize("make_task", [_fixed, _freeform])
def test_get_task_config_returns_quant_config(make_task):
    task = make_task()

    config = get_task_config(task)

    assert isinstance(config, QuantTaskConfig)
    assert config.task_type == task.task_type
    assert config.task_prep_function is requests.run_quant_task_prep
    assert config.task_request_prepare_function is requests.prepare_quant_task_request


def test_get_task_config_still_raises_for_unknown_task():
    class UnknownRawTask(RawTask):
        pass

    with pytest.raises(ValueError, match="Unsupported task type"):
        get_task_config(UnknownRawTask(**_fields(), task_type=TaskType.INSTRUCTTEXTTASK))


async def test_prep_sets_urls_and_status(prep):
    task, loaded = await _run(_freeform(augmented_model_id="org/augmented"))

    prep.prepare_text_task.assert_awaited_once()
    assert task.training_data == TRAIN_URL
    assert task.test_data == EVAL_SET_URL
    assert task.synthetic_data is None
    assert task.status == TaskStatus.LOOKING_FOR_NODES
    assert loaded == ["org/base"]  # the base tokenizer, never the augmented id
    assert prep.object_names[-1].endswith("_quant_eval_set.jsonl")


async def test_prep_rewrites_instruct_dataset_type_to_standard_columns(prep):
    task, _ = await _run(_fixed())

    assert task.dataset_type == InstructTextDatasetType(
        field_instruction=data_cst.STANDARD_INSTRUCT_COLUMN,
        field_output=data_cst.STANDARD_OUTPUT_COLUMN,
        field_input=data_cst.STANDARD_INPUT_COLUMN,
        field_system=None,
        format="{instruction}\n{input}",
    )


async def test_prep_leaves_chat_dataset_type_alone(prep):
    prep.rows = _chat_rows(150)

    task, _ = await _run(_freeform(CHAT))

    assert task.dataset_type == CHAT
    assert len(load_examples(prep.uploaded)) == 150


async def test_prep_raises_below_minimum_examples(prep):
    prep.rows = _instruct_rows(prep_cst.QUANT_EVAL_MIN_EXAMPLES - 1)

    task = _freeform()

    with pytest.raises(ValueError, match="minimum required"):
        await _run(task)
    # The failed task is persisted and recreated from its columns, which must still name ds's raw ones.
    assert task.dataset_type == RAW_INSTRUCT


async def test_uploaded_jsonl_round_trips_with_identical_fingerprint(prep):
    task, _ = await _run(_freeform())
    expected = build_quant_eval_set(prep.rows, task.dataset_type, FakeTokenizer(), prep_cst.QUANT_EVAL_MAX_SEQUENCE_LEN)

    loaded = load_examples(Path(prep.uploaded))

    assert loaded == expected.examples
    assert quant_eval_set_fingerprint(loaded) == quant_eval_set_fingerprint(expected.examples)


@pytest.mark.parametrize("make_task", [_fixed, _freeform])
def test_prepare_quant_task_request_is_the_orchestrators_job(make_task):
    with pytest.raises(NotImplementedError, match="tournament orchestrator"):
        requests.prepare_quant_task_request(make_task())
