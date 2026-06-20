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

    # Served on the reconstructed base, not the bare foundation.
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
