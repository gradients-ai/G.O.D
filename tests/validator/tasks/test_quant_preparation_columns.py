"""Dataset prep column handling for quant tasks (dataset_type instead of field_* attributes)."""

import uuid
from datetime import datetime

import pytest
from datasets import Dataset

import validator.tasks.datasets.constants as cst
from core.models.dataset_models import ChatTemplateDatasetType
from core.models.dataset_models import InstructTextDatasetType
from core.models.quant_models import QuantFormat
from core.models.task_models import TaskStatus
from validator.tasks.datasets.preparation import change_to_json_format
from validator.tasks.datasets.preparation import pick_columns_to_sample
from validator.tasks.datasets.preparation import standardize_column_names
from validator.tasks.datasets.preparation import standardize_instruct_columns
from validator.tasks.models import InstructTextRawTask
from validator.tasks.models import QuantFixedRawTask
from validator.tasks.models import QuantFreeformRawTask


def _fields() -> dict:
    return dict(
        task_id=uuid.uuid4(),
        is_organic=False,
        status=TaskStatus.PENDING,
        model_id="org/base",
        ds="org/data",
        account_id=uuid.uuid4(),
        hours_to_complete=2.0,
        created_at=datetime(2026, 10, 19),
    )


def _quant_instruct(field_input: str | None = None, field_system: str | None = None) -> QuantFreeformRawTask:
    dataset_type = InstructTextDatasetType(
        field_instruction="q", field_output="a", field_input=field_input, field_system=field_system
    )
    return QuantFreeformRawTask(**_fields(), dataset_type=dataset_type)


def _quant_chat() -> QuantFixedRawTask:
    return QuantFixedRawTask(
        **_fields(),
        dataset_type=ChatTemplateDatasetType(chat_column="messages", chat_role_field="role", chat_content_field="content"),
        quant_spec=QuantFormat.W4A16_SYM_G128.spec(),
    )


def _instruct_dataset() -> Dataset:
    return Dataset.from_dict({"q": ["x", "y"], "a": ["1", "2"], "ctx": ["c", ""], "sys": ["s", "s"]})


def test_standardize_instruct_columns_renames_to_standard_columns():
    dataset = standardize_instruct_columns(_instruct_dataset(), "q", "ctx", "a", "sys")

    assert set(dataset.column_names) == {
        cst.STANDARD_INSTRUCT_COLUMN,
        cst.STANDARD_OUTPUT_COLUMN,
        cst.STANDARD_INPUT_COLUMN,
        cst.STANDARD_SYSTEM_COLUMN,
    }


def test_standardize_instruct_columns_raises_on_missing_column():
    with pytest.raises(ValueError, match="Output column"):
        standardize_instruct_columns(_instruct_dataset(), "q", None, "missing", None)


def test_standardize_column_names_still_works_for_instruct_tasks():
    task = InstructTextRawTask(**_fields(), field_instruction="q", field_output="a", field_input="ctx")

    dataset = standardize_column_names(_instruct_dataset(), task)

    assert {cst.STANDARD_INSTRUCT_COLUMN, cst.STANDARD_OUTPUT_COLUMN, cst.STANDARD_INPUT_COLUMN} <= set(dataset.column_names)
    assert "sys" in dataset.column_names


def test_quant_instruct_columns_without_optional_fields():
    dataset = standardize_instruct_columns(_instruct_dataset(), "q", None, "a", None)

    assert pick_columns_to_sample(_quant_instruct(), dataset) == [cst.STANDARD_INSTRUCT_COLUMN, cst.STANDARD_OUTPUT_COLUMN]


def test_quant_instruct_columns_with_input_and_system():
    dataset = standardize_instruct_columns(_instruct_dataset(), "q", "ctx", "a", "sys")

    assert pick_columns_to_sample(_quant_instruct("ctx", "sys"), dataset) == [
        cst.STANDARD_INSTRUCT_COLUMN,
        cst.STANDARD_OUTPUT_COLUMN,
        cst.STANDARD_INPUT_COLUMN,
        cst.STANDARD_SYSTEM_COLUMN,
    ]


def test_quant_instruct_input_column_absent_from_dataset_is_not_sampled():
    dataset = standardize_instruct_columns(_instruct_dataset(), "q", None, "a", None)

    assert cst.STANDARD_INPUT_COLUMN not in pick_columns_to_sample(_quant_instruct("ctx"), dataset)


def test_quant_instruct_rows_become_strings():
    dataset = standardize_instruct_columns(_instruct_dataset(), "q", "ctx", "a", None)
    columns = pick_columns_to_sample(_quant_instruct("ctx"), dataset)

    rows = change_to_json_format(dataset, columns, _quant_instruct("ctx"))

    assert rows == [{"instruct": "x", "output": "1", "input": "c"}, {"instruct": "y", "output": "2", "input": ""}]


def test_quant_chat_samples_the_dataset_type_chat_column():
    assert pick_columns_to_sample(_quant_chat()) == ["messages"]


def test_quant_chat_default_column_when_unset():
    task = _quant_chat()
    task.dataset_type = task.dataset_type.model_copy(update={"chat_column": None})

    assert pick_columns_to_sample(task) == [cst.STANDARD_CHAT_MESSAGES_COLUMN]


def test_quant_chat_rows_use_dataset_type_role_and_content_fields():
    good = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    one_message = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": None}]
    dataset = Dataset.from_dict({"messages": [good, one_message, good]})

    rows = change_to_json_format(dataset, ["messages"], _quant_chat())

    # The ChatRawTask defaults ("from"/"value") would have cleaned every message away.
    assert rows == [{"messages": good}, {"messages": good}]
