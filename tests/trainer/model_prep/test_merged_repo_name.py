"""generate_merged_repo_name (LoRA-merge publish target) + trainer/eval chain-depth parity.

When a LoRA wins a continuous task, model-prep flattens it and re-publishes a full-weight repo at
this deterministic, opaque name so next week's base is a flat model (not a raw adapter). The name
must be (a) deterministic, (b) keyed on the RAW submitted repo so two miners never clobber each
other, and (c) in a different namespace from augmented-* repos. Uses importorskip because the
trainer entrypoint pulls peft/torch, absent on a validator-only box.
"""

import hashlib

import pytest


class TestGenerateMergedRepoName:
    def _fn(self):
        ep = pytest.importorskip("trainer.model_prep.entrypoint")
        return ep

    def test_deterministic_and_exact_hash(self, monkeypatch):
        monkeypatch.delenv("HUGGINGFACE_USERNAME", raising=False)
        ep = self._fn()
        model_id = "some-miner/adapter-xyz"
        expected_hash = hashlib.sha256(f"{model_id}:lora-merge".encode()).hexdigest()[:16]
        name = ep.generate_merged_repo_name(model_id)
        assert name == f"gradients-io/merged-{expected_hash}"
        assert ep.generate_merged_repo_name(model_id) == name  # stable across calls

    def test_distinct_per_model_id(self, monkeypatch):
        monkeypatch.delenv("HUGGINGFACE_USERNAME", raising=False)
        ep = self._fn()
        assert ep.generate_merged_repo_name("minerA/adapter") != ep.generate_merged_repo_name("minerB/adapter")

    def test_keyed_on_raw_repo_and_distinct_from_augmented_namespace(self, monkeypatch):
        monkeypatch.delenv("HUGGINGFACE_USERNAME", raising=False)
        ep = self._fn()
        model_id = "some-miner/adapter-xyz"
        merged = ep.generate_merged_repo_name(model_id)
        anonymized = ep.generate_anonymous_repo_name(model_id, 0)
        assert "/merged-" in merged
        assert merged != anonymized  # merge and augmentation repos never collide

    def test_honors_username_env(self, monkeypatch):
        ep = self._fn()
        monkeypatch.setenv("HUGGINGFACE_USERNAME", "my-org")
        assert ep.generate_merged_repo_name("m/a").startswith("my-org/merged-")


def test_eval_chain_depth_matches_trainer_literal():
    # The trainer's detect_and_merge_lora walks the adapter chain with a hardcoded range(10); the
    # eval side uses MAX_CHAIN_DEPTH. They MUST agree or the two sides flatten to different models.
    materialize = pytest.importorskip("validator.evaluation.pvp.materialize")
    assert materialize.MAX_CHAIN_DEPTH == 10
