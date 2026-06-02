"""Tests for dataset whitelist validation, field propagation through
tournament models, and model anonymizer determinism.
"""

import json
import os
import tempfile

from core.models.payload_models import TrainingRepoResponse
from validator.tournament.models import TaskTrainingAssignment
from validator.tournament.models import TournamentParticipant
from core.datasets.whitelist import MAX_REQUESTED_DATASETS
from core.datasets.whitelist import WHITELISTED_SFT_DATASETS
from core.datasets.whitelist import validate_requested_datasets
from trainer.model_artifacts import get_anonymous_model_dir
from trainer.model_artifacts import scrub_model_identity


# --- 7a: validate_requested_datasets ---


class TestValidateRequestedDatasets:
    def test_valid_datasets_pass_through(self):
        valid = list(WHITELISTED_SFT_DATASETS)[:2]
        result = validate_requested_datasets(valid)
        assert result == valid

    def test_invalid_filtered_out(self):
        valid_one = list(WHITELISTED_SFT_DATASETS)[0]
        result = validate_requested_datasets([valid_one, "totally/fake-dataset"])
        assert result == [valid_one]

    def test_truncated_to_max(self):
        all_valid = list(WHITELISTED_SFT_DATASETS)
        result = validate_requested_datasets(all_valid)
        assert len(result) <= MAX_REQUESTED_DATASETS

    def test_all_invalid_returns_empty(self):
        result = validate_requested_datasets(["fake/one", "fake/two"])
        assert result == []

    def test_none_returns_empty(self):
        assert validate_requested_datasets(None) == []

    def test_empty_list_returns_empty(self):
        assert validate_requested_datasets([]) == []


# --- 7b: requested_datasets field propagation through tournament models ---


class TestDatasetFieldPropagation:
    def test_training_repo_response_has_field(self):
        resp = TrainingRepoResponse(
            github_repo="https://github.com/org/repo",
            commit_hash="abc123",
            requested_datasets=["SoelMgd/Poker_Dataset"],
        )
        assert resp.requested_datasets == ["SoelMgd/Poker_Dataset"]

    def test_tournament_participant_has_field(self):
        p = TournamentParticipant(
            tournament_id="tourn_001",
            hotkey="5GAlice",
            requested_datasets=["SoelMgd/Poker_Dataset", "RZ412/PokerBench"],
        )
        assert p.requested_datasets == ["SoelMgd/Poker_Dataset", "RZ412/PokerBench"]

    def test_task_training_assignment_has_field(self):
        from datetime import datetime
        a = TaskTrainingAssignment(
            task_id="task_001",
            hotkey="5GAlice",
            created_at=datetime.now(),
            requested_datasets=["SoelMgd/Poker_Dataset"],
        )
        assert a.requested_datasets == ["SoelMgd/Poker_Dataset"]

    def test_roundtrip_through_json(self):
        """requested_datasets survives JSON serialization on all models."""
        datasets = ["SoelMgd/Poker_Dataset"]

        resp = TrainingRepoResponse(
            github_repo="https://github.com/org/repo",
            commit_hash="abc123",
            requested_datasets=datasets,
        )
        restored = TrainingRepoResponse.model_validate_json(resp.model_dump_json())
        assert restored.requested_datasets == datasets

        p = TournamentParticipant(
            tournament_id="t1", hotkey="hk1", requested_datasets=datasets,
        )
        restored_p = TournamentParticipant.model_validate_json(p.model_dump_json())
        assert restored_p.requested_datasets == datasets


# --- 7c: Model anonymizer ---


class TestModelAnonymizer:
    def test_deterministic(self):
        os.environ["MODEL_HASH_SALT"] = "test_salt_123"
        try:
            h1 = get_anonymous_model_dir("Qwen/Qwen2.5-7B-Instruct")
            h2 = get_anonymous_model_dir("Qwen/Qwen2.5-7B-Instruct")
            assert h1 == h2
        finally:
            del os.environ["MODEL_HASH_SALT"]

    def test_different_salt_different_hash(self):
        os.environ["MODEL_HASH_SALT"] = "salt_a"
        h_a = get_anonymous_model_dir("model_x")
        os.environ["MODEL_HASH_SALT"] = "salt_b"
        h_b = get_anonymous_model_dir("model_x")
        del os.environ["MODEL_HASH_SALT"]
        assert h_a != h_b

    def test_different_model_different_hash(self):
        os.environ["MODEL_HASH_SALT"] = "same_salt"
        try:
            h1 = get_anonymous_model_dir("model_a")
            h2 = get_anonymous_model_dir("model_b")
            assert h1 != h2
        finally:
            del os.environ["MODEL_HASH_SALT"]

    def test_downloader_and_model_prep_see_same_path(self):
        """Both run_downloader_container and run_model_prep_container use
        get_anonymous_model_dir to build cache paths. If the hash changes
        between calls (e.g. non-deterministic), the model won't be found."""
        os.environ["MODEL_HASH_SALT"] = "prod_salt_value"
        try:
            model_id = "Qwen/Qwen2.5-7B-Instruct"
            # Simulate: downloader computes path, then model prep computes path later
            downloader_hash = get_anonymous_model_dir(model_id)
            model_prep_hash = get_anonymous_model_dir(model_id)
            assert downloader_hash == model_prep_hash
            # Both would build: /cache/models/{hash}
            assert f"/cache/models/{downloader_hash}" == f"/cache/models/{model_prep_hash}"
        finally:
            del os.environ["MODEL_HASH_SALT"]

    def test_scrub_removes_name_or_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config = {"_name_or_path": "secret/model-name", "hidden_size": 768}
            with open(config_path, "w") as f:
                json.dump(config, f)

            scrub_model_identity(tmpdir)

            with open(config_path) as f:
                cleaned = json.load(f)

            assert "_name_or_path" not in cleaned
            assert cleaned["hidden_size"] == 768

    def test_scrub_no_config_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scrub_model_identity(tmpdir)  # should not raise
