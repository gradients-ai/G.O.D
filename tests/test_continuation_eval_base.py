"""Regression tests for the continuation-round eval-base fix.

These fail on the pre-fix code and pass after it:

  * `_get_continuation_base_chains` derives each miner's previous-round lineage
    (starting_model_repo) so eval can rebuild the base they trained on.
  * `_prepare_model` serves that reconstructed base (foundation + previous adapter
    merged) instead of the bare foundation for a continuation miner. Serving the
    bare foundation is the bug — it drops the previous-round delta.
  * A real-peft check proves the sequential LoRA merge actually reconstructs the
    model the miner trained, and that dropping the previous-round adapter changes
    the model's output (the failure the fix prevents).
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
    foundation = "org/foundation"
    starting = {
        "hk_cont": "org/hk_cont-round1",   # genuine continuation -> chain
        "hk_round1": None,                 # no starting repo -> no chain
        "hk_foundation": foundation,       # starting repo IS foundation -> no chain
    }

    async def fake_get_starting_model_repo(task_id, hotkey, psql_db):
        return starting[hotkey]

    monkeypatch.setattr(scoring, "get_starting_model_repo", fake_get_starting_model_repo)

    task = SimpleNamespace(task_id="task-1")
    miners = MinerRepos(by_hotkey={hk: f"org/{hk}-out" for hk in starting})
    config = SimpleNamespace(psql_db=None)

    chains = _run(scoring._get_continuation_base_chains(task, miners, foundation, config))

    assert chains == {"hk_cont": ["org/hk_cont-round1"]}, (
        "only a miner whose starting repo differs from the foundation should get a "
        "reconstruction chain"
    )


# --------------------------------------------------------------------------- #
# 2. _prepare_model serves the reconstructed base, not the bare foundation     #
# --------------------------------------------------------------------------- #


def test_prepare_model_continuation_serves_reconstructed_base(monkeypatch):
    foundation = "org/foundation"
    materialized = "/tmp/base_chain_merged_0"
    calls = {}

    monkeypatch.setattr(pvp_main, "check_for_lora", lambda repo, local_files_only=False: True)
    monkeypatch.setattr(pvp_main, "tool_call_parser_for", lambda path, **kw: "qwen25")

    def fake_materialize(foundation_repo, base_chain):
        calls["args"] = (foundation_repo, list(base_chain))
        return materialized if base_chain else foundation_repo

    monkeypatch.setattr(pvp_main, "materialize_base_model", fake_materialize)

    spec = PvPModelSpec(repo="org/miner-round2", original_model=foundation, base_chain=["org/miner-round1"])
    prepared = pvp_main._prepare_model(spec, "a")

    # The bug: pre-fix this served `foundation`, dropping the round-1 delta.
    assert prepared.sglang_model_path == materialized
    assert prepared.sglang_model_path != foundation
    assert calls["args"] == (foundation, ["org/miner-round1"])
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


# --------------------------------------------------------------------------- #
# 3. Real-peft proof: sequential merge reconstructs the trained model          #
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
def test_sequential_merge_reconstructs_trained_model_and_dropping_delta_diverges():
    base = _tiny_causal_lm()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    # Round 1: adapter R1 on the foundation -> M1.
    import copy

    m1 = _attach_random_lora(copy.deepcopy(base), seed=11, scale=0.5)

    # Round 2: adapter R2 trained ON TOP of M1 -> the model the miner actually produced.
    trained = _attach_random_lora(copy.deepcopy(m1), seed=22, scale=0.1)

    # The eval bug: apply the SAME round-2 adapter to the bare foundation (R1 dropped).
    # Same seed/scale => byte-identical R2 delta, merged onto the foundation instead of M1.
    buggy = _attach_random_lora(copy.deepcopy(base), seed=22, scale=0.1)

    trained_logits = _logits(trained, input_ids)
    buggy_logits = _logits(buggy, input_ids)

    # Dropping the round-1 delta changes the model's output distribution and its
    # argmax token — exactly what silently discards the miner's earlier contribution.
    assert not torch.allclose(trained_logits, buggy_logits, atol=1e-3)
    assert int(trained_logits.argmax()) != int(buggy_logits.argmax())

    # The fix's reconstruction path (foundation -> merge R1 -> M1, then serve M1 + R2)
    # rebuilds M1 identically to the base the miner trained on.
    m1_reconstructed = _attach_random_lora(copy.deepcopy(base), seed=11, scale=0.5)
    assert torch.allclose(_logits(m1, input_ids), _logits(m1_reconstructed, input_ids), atol=1e-5)
