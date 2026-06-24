"""Continuation miners are sensitive to which base they're served on.

A continuation miner's LoRA is trained on foundation + previous adapter. These
GPU-free tests confirm the base matters: changing it shifts the model's argmax
token, and a tool call that doesn't parse makes the turn forfeit (via the real
`_parse_tool_calls`).
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


def test_base_change_shifts_argmax_token():
    """Deterministic: (W0 + dR1 + dR2) and (W0 + dR2) pick different output tokens.

    `W0` is the foundation's projection to vocab logits, `dR1` the previous-round
    delta merged into the base, `dR2` the adapter trained on top.
    """
    out_dim, in_dim, rank = 16, 8, 2

    g = torch.Generator().manual_seed(0)
    w0 = torch.randn(out_dim, in_dim, generator=g)
    x = torch.randn(in_dim, generator=g)

    dr2 = _lora_delta(out_dim, in_dim, rank, scale=0.05, seed=1)

    # Pick dR1 so that, on this input, it decisively boosts a token (j) that
    # (W0 + dR2) would not pick — the previous-round contribution that sets the
    # later adapter's expected token.
    foundation_logits = (w0 + dr2) @ x
    j = int((-foundation_logits).argmax())
    boost = torch.zeros(out_dim, in_dim)
    # e_j outer x_hat: adds (||x||) to logit j for this input, nothing structural elsewhere.
    boost[j] = x / (x.norm() ** 2) * (foundation_logits.max() - foundation_logits.min() + 5.0)
    dr1 = boost

    trained_logits = (w0 + dr1 + dr2) @ x   # served on the trained base
    foundation_logits = (w0 + dr2) @ x      # served on the foundation alone

    trained_token = int(trained_logits.argmax())
    foundation_token = int(foundation_logits.argmax())

    assert trained_token == j, "fixture sanity: the full base should pick the boosted token"
    assert trained_token != foundation_token, (
        "the base choice must change the emitted token "
        f"(trained base picked {trained_token}, foundation picked {foundation_token})"
    )


def test_base_change_shifts_token_across_seeds():
    """Across many random fixtures the two bases disagree on the token.

    Guards against the deterministic case being a fluke: with a non-trivial
    previous-round delta, the two bases pick a different token a large fraction of
    the time.
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
        foundation_token = int(((w0 + dr2) @ x).argmax())
        disagreements += trained_token != foundation_token

    # Require a clear majority so the assertion is robust, not knife-edge.
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
