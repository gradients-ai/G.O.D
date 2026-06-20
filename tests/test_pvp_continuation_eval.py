"""Tests for serving PvP continuation miners on their trained base.

  * `_get_continuation_base_chains` derives each miner's previous-round lineage.
  * `_prepare_model` serves foundation + previous adapter (the trained base) for a
    continuation miner, foundation alone for round 1.
  * A real-transformer check that sequential LoRA merge reconstructs the trained
    model and that dropping the previous adapter changes the output.
"""

import asyncio
from types import SimpleNamespace

import pytest
import torch

import validator.evaluation.pvp.__main__ as pvp_main
from core.models.pvp_models import PvPModelSpec
from core.models.scoring_models import MinerRepos
from validator.evaluation import scoring


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# 1. Per-miner lineage derivation                                             #
# --------------------------------------------------------------------------- #


def test_get_continuation_base_chains_only_for_real_continuations(monkeypatch):
    raw_foundation = "org/foundation"          # task.model_id
    augmented_foundation = "org/foundation-aug"  # base_model (post-prep)
    starting = {
        "hk_cont": "org/hk_cont-round1",       # genuine LoRA continuation -> chain
        "hk_round1": None,                     # no starting repo -> no chain
        "hk_augmented": augmented_foundation,  # starting repo == augmented base -> no chain
        "hk_fallback": raw_foundation,         # missing-prev fallback to raw model_id -> no chain
        "hk_fullmodel": "org/hk-full-ft",      # starting repo is a full model (not LoRA) -> no chain
    }
    non_lora = {"org/hk-full-ft"}

    async def fake_get_starting_model_repo(task_id, hotkey, psql_db):
        return starting[hotkey]

    monkeypatch.setattr(scoring, "get_starting_model_repo", fake_get_starting_model_repo)
    monkeypatch.setattr(scoring, "check_for_lora", lambda repo, local_files_only=False: repo not in non_lora)

    task = SimpleNamespace(task_id="task-1", model_id=raw_foundation)
    miners = MinerRepos(by_hotkey={hk: f"org/{hk}-out" for hk in starting})
    config = SimpleNamespace(psql_db=None)

    chains = _run(scoring._get_continuation_base_chains(task, miners, augmented_foundation, config))

    assert chains == {"hk_cont": ["org/hk_cont-round1"]}, (
        "only a miner whose starting repo is a real LoRA adapter distinct from both the "
        "raw and augmented foundation should get a reconstruction chain"
    )


# --------------------------------------------------------------------------- #
# 2. _prepare_model serves the reconstructed base, not the bare foundation     #
# --------------------------------------------------------------------------- #


def test_prepare_model_continuation_serves_reconstructed_base(monkeypatch):
    foundation = "org/foundation"
    materialized = "/tmp/base_chain_a_merged_0"
    calls = {}

    monkeypatch.setattr(pvp_main, "check_for_lora", lambda repo, local_files_only=False: True)
    monkeypatch.setattr(pvp_main, "tool_call_parser_for", lambda path, **kw: "qwen25")

    def fake_materialize(foundation_repo, base_chain, label="", device=None):
        calls["args"] = (foundation_repo, list(base_chain), label)
        return f"/tmp/base_chain_{label}_merged_0" if base_chain else foundation_repo

    monkeypatch.setattr(pvp_main, "materialize_base_model", fake_materialize)

    spec = PvPModelSpec(repo="org/miner-round2", original_model=foundation, base_chain=["org/miner-round1"])
    prepared = pvp_main._prepare_model(spec, "a", gpu_id=0)

    # Served on the reconstructed base, not the bare foundation.
    assert prepared.sglang_model_path == materialized
    assert prepared.sglang_model_path != foundation
    assert calls["args"] == (foundation, ["org/miner-round1"], "a")
    # The miner's own adapter is still applied on top of the reconstructed base.
    assert "org/miner-round2" in prepared.extra_sglang_args
    assert "--enable-lora" in prepared.extra_sglang_args
    assert prepared.tool_call_parser == "qwen25"


def test_prepare_model_round1_unchanged(monkeypatch):
    """A round-1 miner (empty chain) is served exactly as before: foundation + lora."""
    foundation = "org/foundation"

    monkeypatch.setattr(pvp_main, "check_for_lora", lambda repo, local_files_only=False: True)

    spec = PvPModelSpec(repo="org/miner-round1", original_model=foundation, base_chain=[])
    prepared = pvp_main._prepare_model(spec, "b")

    assert prepared.sglang_model_path == foundation
    assert "org/miner-round1" in prepared.extra_sglang_args
    # No explicit parser override; the server resolves it from the foundation repo id.
    assert prepared.tool_call_parser is None


def test_materialize_uses_distinct_dirs_per_label(monkeypatch):
    """Two models are prepared before either server starts, so their merge scratch
    dirs must differ — otherwise the second clobbers the first's reconstructed base."""
    import validator.evaluation.pvp.materialize as mat

    monkeypatch.setattr(mat, "_declared_base", lambda repo: None)  # treat each starting repo as foundation-relative
    monkeypatch.setattr(mat, "_download_lora_with_retry", lambda repo, d, **kw: d)
    monkeypatch.setattr(mat, "_download_model_with_retry", lambda repo, **kw: f"/base/{repo}")
    monkeypatch.setattr(mat, "_merge_base_and_lora", lambda base, lora, output_dir, device=None: output_dir)

    path_a = mat.materialize_base_model("org/foundation", ["org/minerA-round1"], label="a")
    path_b = mat.materialize_base_model("org/foundation", ["org/minerB-round1"], label="b")

    assert path_a != path_b, "two models must materialize to distinct dirs (no /tmp clobber)"
    assert (path_a, path_b) == ("/tmp/base_chain_a_merged_0", "/tmp/base_chain_b_merged_0")


def test_resolve_chain_walks_unflattened_lineage(monkeypatch):
    """When an adapter's base pointer wasn't flattened, eval must still walk down to
    the real foundation and merge every adapter — matching the trainer, not crash
    trying to load an intermediate adapter as a full model."""
    import validator.evaluation.pvp.materialize as mat

    # R3 -> R2 -> R1 -> foundation (none flattened).
    declared = {
        "org/R3": "org/R2",
        "org/R2": "org/R1",
        "org/R1": "org/foundation",
        "org/foundation": None,
    }
    monkeypatch.setattr(mat, "_declared_base", lambda repo: declared[repo])

    foundation, adapters = mat._resolve_chain("org/R3", "org/wrong-fallback")

    assert foundation == "org/foundation"
    # bottom-to-top merge order, exactly what the trainer merges.
    assert adapters == ["org/R1", "org/R2", "org/R3"]


def test_resolve_chain_flattened_is_single_hop(monkeypatch):
    """The normal (flattened) case: the starting adapter points straight at the
    foundation, so only it is merged."""
    import validator.evaluation.pvp.materialize as mat

    declared = {"org/R2": "org/foundation", "org/foundation": None}
    monkeypatch.setattr(mat, "_declared_base", lambda repo: declared[repo])

    foundation, adapters = mat._resolve_chain("org/R2", "org/foundation")

    assert foundation == "org/foundation"
    assert adapters == ["org/R2"]


# --------------------------------------------------------------------------- #
# 3. Sequential merge reconstructs the trained model (real transformer)        #
# --------------------------------------------------------------------------- #


def _tiny_causal_lm():
    from transformers import LlamaConfig
    from transformers import LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    )
    return LlamaForCausalLM(cfg).eval()


def _attach_random_lora(model, seed, scale):
    """Merge a deterministic LoRA adapter into the model's attention projections.

    This is exactly what `peft`'s `merge_and_unload` does — add the low-rank delta
    dW = (alpha / r) * (B @ A) into each target weight — implemented directly so the
    adapter is byte-identical regardless of which base it is merged onto, and to
    avoid an unrelated peft<->torchao version incompatibility in this environment.
    A LoRA delta B @ A is base-independent, so "the same adapter" merged onto the
    foundation vs onto M1 isolates exactly the dropped round-1 delta.
    """
    r, alpha = 8, 16
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in model.model.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.v_proj):
                out_dim, in_dim = proj.weight.shape
                a = torch.randn(r, in_dim, generator=g) * scale
                b = torch.randn(out_dim, r, generator=g) * scale
                proj.weight.add_((alpha / r) * (b @ a))
    return model.eval()


def _logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids).logits[0, -1]


@pytest.mark.filterwarnings("ignore")
def test_sequential_merge_reconstructs_trained_base():
    base = _tiny_causal_lm()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    # Round 1: adapter R1 on the foundation -> M1.
    import copy

    m1 = _attach_random_lora(copy.deepcopy(base), seed=11, scale=0.5)

    # Round 2: adapter R2 trained ON TOP of M1 -> the model the miner produced.
    trained = _attach_random_lora(copy.deepcopy(m1), seed=22, scale=0.1)

    # The same R2 adapter on the bare foundation (same seed/scale => identical delta).
    foundation_only = _attach_random_lora(copy.deepcopy(base), seed=22, scale=0.1)

    trained_logits = _logits(trained, input_ids)
    foundation_logits = _logits(foundation_only, input_ids)

    # The base choice changes the output distribution and the argmax token.
    assert not torch.allclose(trained_logits, foundation_logits, atol=1e-3)
    assert int(trained_logits.argmax()) != int(foundation_logits.argmax())

    # Reconstructing the base (foundation -> merge R1 -> M1) reproduces M1 exactly.
    m1_reconstructed = _attach_random_lora(copy.deepcopy(base), seed=11, scale=0.5)
    assert torch.allclose(_logits(m1, input_ids), _logits(m1_reconstructed, input_ids), atol=1e-5)
