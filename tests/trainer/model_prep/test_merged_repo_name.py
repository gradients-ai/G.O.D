"""generate_merged_repo_name / merge_publish_target + trainer/eval chain-depth parity.

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


class TestMergePublishTarget:
    """merge_publish_target decides whether / where to publish a flat merged base."""

    def _fn(self):
        ep = pytest.importorskip("trainer.model_prep.entrypoint")
        return ep

    def test_publishes_when_local_dir_already_merged_but_source_is_adapter(self, monkeypatch):
        # Regression: the downloader flattens the adapter in place before model-prep runs, so
        # was_lora is False even though the source HF repo is still an adapter. Publish must fire.
        monkeypatch.delenv("HUGGINGFACE_USERNAME", raising=False)
        ep = self._fn()
        cache_path = "/cache/models/anon-hash"
        source = "gradients-io-tournaments/tournament-winner-adapter"
        target = ep.merge_publish_target(cache_path, source, was_lora=False, source_is_adapter=True)
        assert target == ep.generate_merged_repo_name(source)
        assert target != ep.generate_merged_repo_name(cache_path)

    def test_keyed_on_source_repo_id_not_cache_path(self, monkeypatch):
        monkeypatch.delenv("HUGGINGFACE_USERNAME", raising=False)
        ep = self._fn()
        source_a = "org/adapter-a"
        source_b = "org/adapter-b"
        target_a = ep.merge_publish_target("/cache/models/x", source_a, was_lora=True, source_is_adapter=False)
        target_b = ep.merge_publish_target("/cache/models/x", source_b, was_lora=True, source_is_adapter=False)
        assert target_a != target_b
        # Retries with the same source reuse one repo.
        assert target_a == ep.merge_publish_target("/cache/models/y", source_a, was_lora=False, source_is_adapter=True)

    def test_no_publish_for_plain_full_weight_base(self, monkeypatch):
        monkeypatch.delenv("HUGGINGFACE_USERNAME", raising=False)
        ep = self._fn()
        assert (
            ep.merge_publish_target("Qwen/Qwen3-14B", "Qwen/Qwen3-14B", was_lora=False, source_is_adapter=False)
            is None
        )

    def test_falls_back_to_model_arg_when_source_missing(self, monkeypatch):
        monkeypatch.delenv("HUGGINGFACE_USERNAME", raising=False)
        ep = self._fn()
        model_arg = "org/still-an-adapter"
        target = ep.merge_publish_target(model_arg, None, was_lora=True, source_is_adapter=False)
        assert target == ep.generate_merged_repo_name(model_arg)


def test_eval_chain_depth_matches_trainer_literal():
    # The trainer's detect_and_merge_lora walks the adapter chain with a hardcoded range(10); the
    # eval side uses MAX_CHAIN_DEPTH. They MUST agree or the two sides flatten to different models.
    materialize = pytest.importorskip("validator.evaluation.pvp.materialize")
    assert materialize.MAX_CHAIN_DEPTH == 10
