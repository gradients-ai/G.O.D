"""Tests for core.config.config_handler — dataset entry creation and config utilities."""

import json
import os
import tempfile

import pytest

from core.config.config_handler import (
    create_dataset_entry,
    create_reward_funcs_file,
    save_config,
    update_flash_attention,
    _process_dpo_dataset_fields,
    _process_grpo_dataset_fields,
    _process_instruct_dataset_fields,
    _process_chat_template_dataset_fields,
    _process_environment_dataset_fields,
)
from core.models.utility_models import (
    ChatTemplateDatasetType,
    DpoDatasetType,
    EnvironmentDatasetType,
    FileFormat,
    GrpoDatasetType,
    InstructTextDatasetType,
)
import core.constants as cst


class TestProcessGrpoDatasetFields:
    """Tests for _process_grpo_dataset_fields."""

    def test_returns_train_split(self):
        result = _process_grpo_dataset_fields(GrpoDatasetType())
        assert result == {"split": "train"}


class TestProcessEnvironmentDatasetFields:
    """Tests for _process_environment_dataset_fields."""

    def test_returns_train_split(self):
        result = _process_environment_dataset_fields(EnvironmentDatasetType())
        assert result == {"split": "train"}


class TestProcessDpoDatasetFields:
    """Tests for _process_dpo_dataset_fields."""

    def test_returns_correct_type_and_split(self):
        result = _process_dpo_dataset_fields(DpoDatasetType())
        assert result["type"] == cst.DPO_DEFAULT_DATASET_TYPE
        assert result["split"] == "train"


class TestProcessInstructDatasetFields:
    """Tests for _process_instruct_dataset_fields."""

    def test_completion_type_when_no_output(self):
        fields = {"field_instruction": "instruction"}
        result = _process_instruct_dataset_fields(fields)
        assert result["type"] == "completion"
        assert result["field"] == "instruction"

    def test_custom_format_with_output(self):
        fields = {"field_instruction": "inst", "field_output": "out"}
        result = _process_instruct_dataset_fields(fields)
        assert result["format"] == "custom"
        assert isinstance(result["type"], dict)
        assert result["type"]["no_input_format"] == "{instruction}"

    def test_custom_format_with_input_and_output(self):
        fields = {"field_instruction": "inst", "field_input": "inp", "field_output": "out"}
        result = _process_instruct_dataset_fields(fields)
        assert result["type"]["format"] == "{instruction} {input}"


class TestProcessChatTemplateDatasetFields:
    """Tests for _process_chat_template_dataset_fields."""

    def test_default_chat_template(self):
        dt = ChatTemplateDatasetType()
        result = _process_chat_template_dataset_fields(dt)
        assert result["type"] == "chat_template"
        assert result["chat_template"] == "chatml"
        assert result["field_messages"] == "conversations"
        assert result["roles"]["user"] == ["user"]
        assert result["roles"]["assistant"] == ["assistant"]

    def test_custom_chat_template(self):
        dt = ChatTemplateDatasetType(
            chat_template="llama",
            chat_column="messages",
            chat_role_field="role",
            chat_content_field="content",
            chat_user_reference="human",
            chat_assistant_reference="gpt",
        )
        result = _process_chat_template_dataset_fields(dt)
        assert result["chat_template"] == "llama"
        assert result["field_messages"] == "messages"
        assert result["roles"]["user"] == ["human"]
        assert result["roles"]["assistant"] == ["gpt"]


class TestCreateDatasetEntry:
    """Tests for create_dataset_entry."""

    def test_hf_format_instruct(self):
        dt = InstructTextDatasetType(field_instruction="instruction", field_output="output")
        entry = create_dataset_entry("my-org/my-dataset", dt, FileFormat.HF)
        assert entry["path"] == "my-org/my-dataset"
        assert "ds_type" not in entry

    def test_json_format_uses_workspace_path(self):
        dt = GrpoDatasetType(field_prompt="prompt")
        entry = create_dataset_entry("data.json", dt, FileFormat.JSON)
        assert entry["path"] == "/workspace/input_data/"
        assert entry["ds_type"] == "json"
        assert entry["data_files"] == ["data.json"]

    def test_json_format_eval_uses_full_path(self):
        dt = GrpoDatasetType(field_prompt="prompt")
        entry = create_dataset_entry("/some/path/eval.json", dt, FileFormat.JSON, is_eval=True)
        assert entry["path"] == "/workspace/input_data/eval.json"

    def test_invalid_dataset_type_raises(self):
        with pytest.raises(ValueError, match="Invalid dataset_type"):
            create_dataset_entry("data", "not_a_dataset_type", FileFormat.HF)


class TestUpdateFlashAttention:
    """Tests for update_flash_attention."""

    def test_sets_flash_attention_false(self):
        config = {"some_key": "value"}
        result = update_flash_attention(config, "meta-llama/Llama-2-7b")
        assert result["flash_attention"] is False

    def test_overwrites_existing_flash_attention(self):
        config = {"flash_attention": True}
        result = update_flash_attention(config, "any-model")
        assert result["flash_attention"] is False


class TestSaveConfig:
    """Tests for save_config."""

    def test_saves_yaml_config(self, tmp_path):
        import yaml

        config = {"model": "test", "learning_rate": 0.001, "epochs": 3}
        config_path = str(tmp_path / "config.yml")
        save_config(config, config_path)

        with open(config_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded == config


class TestCreateRewardFuncsFile:
    """Tests for create_reward_funcs_file."""

    def test_creates_file_with_functions(self, tmp_path):
        reward_funcs = [
            'def reward_length(completions, **kwargs):\n    return [len(c) for c in completions]',
            'def reward_quality(completions, **kwargs):\n    return [1.0 for c in completions]',
        ]
        dest = str(tmp_path)
        filename, func_names = create_reward_funcs_file(reward_funcs, "task123", destination_dir=dest)

        assert filename == "rewards_task123"
        assert func_names == ["reward_length", "reward_quality"]
        assert os.path.exists(os.path.join(dest, "rewards_task123.py"))

        with open(os.path.join(dest, "rewards_task123.py")) as f:
            content = f.read()

        assert "def reward_length" in content
        assert "def reward_quality" in content
        assert "Auto-generated" in content

    def test_empty_reward_funcs(self, tmp_path):
        dest = str(tmp_path)
        filename, func_names = create_reward_funcs_file([], "empty_task", destination_dir=dest)
        assert func_names == []
        assert os.path.exists(os.path.join(dest, "rewards_empty_task.py"))
