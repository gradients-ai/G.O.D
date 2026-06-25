import validator.evaluation.pvp.__main__ as pvp_main
import validator.evaluation.pvp.materialize as materialize
from core.models.pvp_models import PvPModelSpec


def test_prepare_model_continuation_serves_reconstructed_base(monkeypatch):
    foundation = "org/foundation"
    calls = {}

    monkeypatch.setattr(pvp_main, "check_for_lora", lambda repo, local_files_only=False: True)
    monkeypatch.setattr(pvp_main, "tool_call_parser_for", lambda path, **kwargs: "qwen25")

    def fake_materialize(foundation_repo, base_chain, label="", device=None):
        calls["args"] = (foundation_repo, list(base_chain), label, device)
        return f"/tmp/base_chain_{label}_merged_0" if base_chain else foundation_repo

    monkeypatch.setattr(pvp_main, "materialize_base_model", fake_materialize)

    spec = PvPModelSpec(repo="org/miner-round2", original_model=foundation, base_chain=["org/miner-round1"])
    prepared = pvp_main._prepare_model(spec, "a", gpu_id=0)

    assert prepared.sglang_model_path == "/tmp/base_chain_a_merged_0"
    assert prepared.sglang_model_path != foundation
    assert calls["args"] == (foundation, ["org/miner-round1"], "a", "cuda:0")
    assert "org/miner-round2" in prepared.extra_sglang_args
    assert "--enable-lora" in prepared.extra_sglang_args
    assert prepared.tool_call_parser == "qwen25"


def test_prepare_model_round1_unchanged(monkeypatch):
    foundation = "org/foundation"

    monkeypatch.setattr(pvp_main, "check_for_lora", lambda repo, local_files_only=False: True)

    spec = PvPModelSpec(repo="org/miner-round1", original_model=foundation, base_chain=[])
    prepared = pvp_main._prepare_model(spec, "b")

    assert prepared.sglang_model_path == foundation
    assert "org/miner-round1" in prepared.extra_sglang_args
    assert prepared.tool_call_parser is None


def test_materialize_uses_distinct_dirs_per_label(monkeypatch):
    monkeypatch.setattr(materialize, "_declared_base", lambda repo: None)
    monkeypatch.setattr(materialize, "_download_lora_with_retry", lambda repo, directory, **kwargs: directory)
    monkeypatch.setattr(materialize, "_download_model_with_retry", lambda repo, **kwargs: f"/base/{repo}")
    monkeypatch.setattr(
        materialize,
        "_merge_base_and_lora",
        lambda base, lora, output_dir, device=None: output_dir,
    )

    path_a = materialize.materialize_base_model("org/foundation", ["org/minerA-round1"], label="a")
    path_b = materialize.materialize_base_model("org/foundation", ["org/minerB-round1"], label="b")

    assert path_a != path_b
    assert (path_a, path_b) == ("/tmp/base_chain_a_merged_0", "/tmp/base_chain_b_merged_0")


def test_materialize_empty_chain_merges_lora_foundation(monkeypatch):
    declared = {"org/prev-winner": "org/foundation", "org/foundation": None}
    monkeypatch.setattr(materialize, "_declared_base", lambda repo: declared[repo])
    monkeypatch.setattr(materialize, "_download_lora_with_retry", lambda repo, directory, **kwargs: directory)
    monkeypatch.setattr(materialize, "_download_model_with_retry", lambda repo, **kwargs: f"/base/{repo}")
    monkeypatch.setattr(
        materialize,
        "_merge_base_and_lora",
        lambda base, lora, output_dir, device=None: output_dir,
    )

    path = materialize.materialize_base_model("org/prev-winner", [], label="prev")

    assert path == "/tmp/base_chain_prev_merged_0"


def test_resolve_chain_walks_unflattened_lineage(monkeypatch):
    declared = {
        "org/R3": "org/R2",
        "org/R2": "org/R1",
        "org/R1": "org/foundation",
        "org/foundation": None,
    }
    monkeypatch.setattr(materialize, "_declared_base", lambda repo: declared[repo])

    foundation, adapters = materialize._resolve_chain("org/R3", "org/wrong-fallback")

    assert foundation == "org/foundation"
    assert adapters == ["org/R1", "org/R2", "org/R3"]


def test_resolve_chain_flattened_is_single_hop(monkeypatch):
    declared = {"org/R2": "org/foundation", "org/foundation": None}
    monkeypatch.setattr(materialize, "_declared_base", lambda repo: declared[repo])

    foundation, adapters = materialize._resolve_chain("org/R2", "org/foundation")

    assert foundation == "org/foundation"
    assert adapters == ["org/R2"]


def test_pvp_model_spec_defaults_to_empty_base_chain():
    spec = PvPModelSpec(repo="org/miner", original_model="org/foundation")

    assert spec.base_chain == []
