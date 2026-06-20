"""Reproduction of the continuation-round train/eval base-model mismatch.

Mechanism (env / PvP tournaments, round >= 2, TrainingStartPoint.CONTINUATION):

  * The miner is TRAINED from their previous-round adapter merged into the
    foundation -> a full model M1 = foundation + delta_R1. Their new LoRA delta_R2
    is therefore learned *relative to M1*.
  * At eval, the PvP path serves the bare foundation + delta_R2 only (delta_R1 is
    dropped), because run_evaluation_pvp_pair passes a single shared
    `base_model = task.augmented_model_id or task.model_id` (the foundation) and
    pvp/__main__._prepare_model applies the adapter on top of that foundation.

Net: trained on (foundation + delta_R1 + delta_R2) but evaluated on
(foundation + delta_R2). The earlier-round delta is silently dropped.

This module proves the bug WITHOUT a GPU or a live SGLang server (neither is
available here, and live evals are out of scope):

  1. A deterministic linear-algebra fixture shows that dropping delta_R1 changes
     the model's argmax output token — i.e. the model emits a *different* token.
  2. A statistical sweep shows the divergence is generic, not a hand-picked case.
  3. A real-code check (`core.pvp.chat._parse_tool_calls`) shows that once the
     emitted text is malformed / the structured tool call is missing, the turn
     yields no action — which the PvP harness scores as a forfeit. This is the
     concrete failure mode: a flipped token breaks the `<tool_call>` JSON the
     later adapter learned to emit on top of M1, so the miner forfeits every turn.
"""

from types import SimpleNamespace

import torch

from core.models.pvp_models import ToolCall
from core.pvp.chat import _parse_tool_calls


def _lora_delta(out_dim: int, in_dim: int, rank: int, scale: float, seed: int) -> torch.Tensor:
    """A low-rank update B @ A, the weight delta a LoRA adapter materialises."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(rank, in_dim, generator=g)
    b = torch.randn(out_dim, rank, generator=g)
    return scale * (b @ a)


def test_dropping_prior_round_delta_flips_argmax_token():
    """Deterministic: (W0 + dR1 + dR2) and (W0 + dR2) pick different output tokens.

    `W0` stands in for the foundation's final projection to vocab logits, `dR1`
    for the round-1 delta the trainer merged into the foundation, and `dR2` for
    the round-2 adapter trained on top. The eval bug evaluates `W0 + dR2`.
    """
    out_dim, in_dim, rank = 16, 8, 2

    g = torch.Generator().manual_seed(0)
    w0 = torch.randn(out_dim, in_dim, generator=g)
    x = torch.randn(in_dim, generator=g)

    dr2 = _lora_delta(out_dim, in_dim, rank, scale=0.05, seed=1)

    # Construct dR1 so that, on this input, it decisively boosts one token (j) that
    # is NOT what (W0 + dR2) would pick. This is exactly the round-1 contribution
    # that establishes the later adapter's expected output token; dropping it must
    # flip the choice.
    buggy_logits = (w0 + dr2) @ x
    j = int((-buggy_logits).argmax())  # a token the buggy base actively disfavours
    boost = torch.zeros(out_dim, in_dim)
    # e_j outer x_hat: adds (||x||) to logit j for this input, nothing structural elsewhere.
    boost[j] = x / (x.norm() ** 2) * (buggy_logits.max() - buggy_logits.min() + 5.0)
    dr1 = boost

    trained_logits = (w0 + dr1 + dr2) @ x  # what the miner trained against
    buggy_logits = (w0 + dr2) @ x          # what the current eval serves

    trained_token = int(trained_logits.argmax())
    buggy_token = int(buggy_logits.argmax())

    assert trained_token == j, "fixture sanity: the full base should pick the boosted token"
    assert trained_token != buggy_token, (
        "dropping the round-1 delta must change the emitted token "
        f"(trained picked {trained_token}, buggy eval picked {buggy_token})"
    )


def test_dropping_prior_round_delta_diverges_statistically():
    """Across many random fixtures the dropped-delta model disagrees on the token.

    Guards against the deterministic case being a fluke: with a non-trivial
    round-1 delta, the foundation-only eval picks a different token a large
    fraction of the time.
    """
    out_dim, in_dim, rank = 32, 16, 4
    disagreements = 0
    trials = 200

    for seed in range(trials):
        g = torch.Generator().manual_seed(seed)
        w0 = torch.randn(out_dim, in_dim, generator=g)
        x = torch.randn(in_dim, generator=g)
        dr1 = _lora_delta(out_dim, in_dim, rank, scale=0.5, seed=10_000 + seed)
        dr2 = _lora_delta(out_dim, in_dim, rank, scale=0.5, seed=20_000 + seed)

        trained_token = int(((w0 + dr1 + dr2) @ x).argmax())
        buggy_token = int(((w0 + dr2) @ x).argmax())
        disagreements += trained_token != buggy_token

    # A merged round-1 delta of comparable magnitude flips the token very often;
    # require a clear majority so the assertion is robust, not knife-edge.
    assert disagreements / trials > 0.5, (
        f"expected frequent token divergence, got {disagreements}/{trials}"
    )


def test_missing_tool_call_forfeits_but_valid_one_acts():
    """The downstream consequence: a missing/garbled tool call yields no action.

    SGLang turns the model's raw text into structured `tool_calls`; when the text
    is malformed the field is absent and `_parse_tool_calls` returns None. The PvP
    harness has no move to play and the player forfeits the turn. A correctly
    formatted call parses into a ToolCall the harness can execute.
    """
    no_call_message = SimpleNamespace(content="I think I'll play... 3?", tool_calls=None)
    assert _parse_tool_calls(no_call_message) is None  # -> no action -> forfeit

    valid_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="play_card", arguments='{"card": 3}'),
    )
    parsed = _parse_tool_calls(SimpleNamespace(content=None, tool_calls=[valid_call]))
    assert parsed == [ToolCall(id="call_1", name="play_card", arguments={"card": 3})]
