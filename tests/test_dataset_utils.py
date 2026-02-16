"""Tests for core.dataset_utils module — dataset adaptation functions."""

import json
import os
import tempfile

import pytest

from core.dataset_utils import (
    _dpo_format_chosen,
    _dpo_format_prompt,
    _dpo_format_rejected,
    adapt_columns_for_dpo_dataset,
    adapt_columns_for_grpo_dataset,
    adapt_columns_for_environment_dataset,
)
from core.models.utility_models import DpoDatasetType, GrpoDatasetType, EnvironmentDatasetType
import core.constants as cst


class TestDpoFormatPrompt:
    """Tests for the _dpo_format_prompt helper."""

    def test_basic_prompt_substitution(self):
        row = {cst.DPO_DEFAULT_FIELD_PROMPT: "What is AI?"}
        result = _dpo_format_prompt(row, "Question: {prompt}")
        assert result == "Question: What is AI?"

    def test_system_substitution(self):
        row = {
            cst.DPO_DEFAULT_FIELD_PROMPT: "Hello",
            cst.DPO_DEFAULT_FIELD_SYSTEM: "You are helpful.",
        }
        result = _dpo_format_prompt(row, "{system}\n{prompt}")
        assert result == "You are helpful.\nHello"

    def test_no_matching_placeholders(self):
        row = {cst.DPO_DEFAULT_FIELD_PROMPT: "test"}
        result = _dpo_format_prompt(row, "static text")
        assert result == "static text"

    def test_missing_field_leaves_placeholder(self):
        row = {}
        result = _dpo_format_prompt(row, "Q: {prompt}")
        assert result == "Q: {prompt}"

    def test_nan_value_leaves_placeholder(self):
        import pandas as pd
        row = {cst.DPO_DEFAULT_FIELD_PROMPT: float("nan")}
        result = _dpo_format_prompt(row, "Q: {prompt}")
        assert result == "Q: {prompt}"


class TestDpoFormatChosen:
    """Tests for the _dpo_format_chosen helper."""

    def test_chosen_substitution(self):
        row = {cst.DPO_DEFAULT_FIELD_CHOSEN: "Good answer"}
        result = _dpo_format_chosen(row, "Answer: {chosen}")
        assert result == "Answer: Good answer"

    def test_chosen_with_prompt_and_system(self):
        row = {
            cst.DPO_DEFAULT_FIELD_CHOSEN: "answer",
            cst.DPO_DEFAULT_FIELD_PROMPT: "question",
            cst.DPO_DEFAULT_FIELD_SYSTEM: "system",
        }
        result = _dpo_format_chosen(row, "{system}: {prompt} -> {chosen}")
        assert result == "system: question -> answer"


class TestDpoFormatRejected:
    """Tests for the _dpo_format_rejected helper."""

    def test_rejected_substitution(self):
        row = {cst.DPO_DEFAULT_FIELD_REJECTED: "Bad answer"}
        result = _dpo_format_rejected(row, "Rejected: {rejected}")
        assert result == "Rejected: Bad answer"

    def test_rejected_with_all_fields(self):
        row = {
            cst.DPO_DEFAULT_FIELD_REJECTED: "wrong",
            cst.DPO_DEFAULT_FIELD_PROMPT: "q",
            cst.DPO_DEFAULT_FIELD_SYSTEM: "sys",
        }
        result = _dpo_format_rejected(row, "{system} | {prompt} | {rejected}")
        assert result == "sys | q | wrong"


class TestAdaptColumnsForDpoDataset:
    """Tests for adapt_columns_for_dpo_dataset."""

    def _create_dataset_file(self, tmp_path, data):
        filepath = str(tmp_path / "dataset.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        return filepath

    def test_column_renaming(self, tmp_path):
        data = [
            {"q": "What?", "sys": "Be helpful", "good": "Yes", "bad": "No"},
            {"q": "How?", "sys": "Be concise", "good": "This way", "bad": "That way"},
        ]
        filepath = self._create_dataset_file(tmp_path, data)

        dataset_type = DpoDatasetType(
            field_prompt="q",
            field_system="sys",
            field_chosen="good",
            field_rejected="bad",
        )

        adapt_columns_for_dpo_dataset(filepath, dataset_type)

        with open(filepath) as f:
            result = json.load(f)

        assert result[0][cst.DPO_DEFAULT_FIELD_PROMPT] == "What?"
        assert result[0][cst.DPO_DEFAULT_FIELD_SYSTEM] == "Be helpful"
        assert result[0][cst.DPO_DEFAULT_FIELD_CHOSEN] == "Yes"
        assert result[0][cst.DPO_DEFAULT_FIELD_REJECTED] == "No"

    def test_formatting_applied(self, tmp_path):
        data = [
            {
                cst.DPO_DEFAULT_FIELD_PROMPT: "Hello",
                cst.DPO_DEFAULT_FIELD_SYSTEM: "System",
                cst.DPO_DEFAULT_FIELD_CHOSEN: "Good",
                cst.DPO_DEFAULT_FIELD_REJECTED: "Bad",
            }
        ]
        filepath = self._create_dataset_file(tmp_path, data)

        dataset_type = DpoDatasetType(
            field_prompt=cst.DPO_DEFAULT_FIELD_PROMPT,
            field_system=cst.DPO_DEFAULT_FIELD_SYSTEM,
            field_chosen=cst.DPO_DEFAULT_FIELD_CHOSEN,
            field_rejected=cst.DPO_DEFAULT_FIELD_REJECTED,
            prompt_format="<prompt>{prompt}</prompt>",
            chosen_format="<chosen>{chosen}</chosen>",
            rejected_format="<rejected>{rejected}</rejected>",
        )

        adapt_columns_for_dpo_dataset(filepath, dataset_type, apply_formatting=True)

        with open(filepath) as f:
            result = json.load(f)

        assert result[0][cst.DPO_DEFAULT_FIELD_PROMPT] == "<prompt>Hello</prompt>"
        assert result[0][cst.DPO_DEFAULT_FIELD_CHOSEN] == "<chosen>Good</chosen>"
        assert result[0][cst.DPO_DEFAULT_FIELD_REJECTED] == "<rejected>Bad</rejected>"

    def test_no_formatting_when_default_templates(self, tmp_path):
        """Default format strings like '{prompt}' should not trigger formatting."""
        data = [
            {
                cst.DPO_DEFAULT_FIELD_PROMPT: "Hello",
                cst.DPO_DEFAULT_FIELD_CHOSEN: "Good",
                cst.DPO_DEFAULT_FIELD_REJECTED: "Bad",
            }
        ]
        filepath = self._create_dataset_file(tmp_path, data)

        dataset_type = DpoDatasetType(
            field_prompt=cst.DPO_DEFAULT_FIELD_PROMPT,
            field_chosen=cst.DPO_DEFAULT_FIELD_CHOSEN,
            field_rejected=cst.DPO_DEFAULT_FIELD_REJECTED,
        )

        adapt_columns_for_dpo_dataset(filepath, dataset_type, apply_formatting=True)

        with open(filepath) as f:
            result = json.load(f)

        assert result[0][cst.DPO_DEFAULT_FIELD_PROMPT] == "Hello"

    def test_empty_dataset(self, tmp_path):
        filepath = self._create_dataset_file(tmp_path, [])

        dataset_type = DpoDatasetType(
            field_prompt="q", field_chosen="c", field_rejected="r"
        )

        adapt_columns_for_dpo_dataset(filepath, dataset_type)

        with open(filepath) as f:
            result = json.load(f)

        assert result == []


class TestAdaptColumnsForGrpoDataset:
    """Tests for adapt_columns_for_grpo_dataset."""

    def test_column_renaming(self, tmp_path):
        data = [
            {"question": "What is 2+2?"},
            {"question": "Explain gravity."},
        ]
        filepath = str(tmp_path / "grpo.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        dataset_type = GrpoDatasetType(field_prompt="question")
        adapt_columns_for_grpo_dataset(filepath, dataset_type)

        with open(filepath) as f:
            result = json.load(f)

        assert all(cst.GRPO_DEFAULT_FIELD_PROMPT in row for row in result)
        assert result[0][cst.GRPO_DEFAULT_FIELD_PROMPT] == "What is 2+2?"

    def test_empty_prompts_filtered(self, tmp_path):
        data = [
            {"question": "Valid prompt"},
            {"question": ""},
            {"question": None},
            {"question": "Another valid"},
        ]
        filepath = str(tmp_path / "grpo.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        dataset_type = GrpoDatasetType(field_prompt="question")
        adapt_columns_for_grpo_dataset(filepath, dataset_type)

        with open(filepath) as f:
            result = json.load(f)

        assert len(result) == 2
        assert result[0][cst.GRPO_DEFAULT_FIELD_PROMPT] == "Valid prompt"
        assert result[1][cst.GRPO_DEFAULT_FIELD_PROMPT] == "Another valid"


class TestAdaptColumnsForEnvironmentDataset:
    """Tests for adapt_columns_for_environment_dataset."""

    def test_renames_prompt_column(self, tmp_path):
        data = [
            {"prompt": "Solve this puzzle"},
            {"prompt": "Navigate the maze"},
        ]
        filepath = str(tmp_path / "env.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        dataset_type = EnvironmentDatasetType(environment_name="test_env")
        adapt_columns_for_environment_dataset(filepath, dataset_type)

        with open(filepath) as f:
            result = json.load(f)

        assert all(cst.GRPO_DEFAULT_FIELD_PROMPT in row for row in result)

    def test_filters_empty_entries(self, tmp_path):
        data = [
            {"prompt": "Valid"},
            {"prompt": ""},
            {"prompt": "Also valid"},
        ]
        filepath = str(tmp_path / "env.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        dataset_type = EnvironmentDatasetType(environment_name="test_env")
        adapt_columns_for_environment_dataset(filepath, dataset_type)

        with open(filepath) as f:
            result = json.load(f)

        assert len(result) == 2
