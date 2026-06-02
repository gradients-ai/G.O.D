#!/usr/bin/env python3
"""Play a PvP matchup with two Anthropic models, via the real harness.

A thin adapter bridges our ChatFn (ChatMessage list + ToolSchema list -> ChatResult)
to the Anthropic Messages API with tool use, so no GPU/SGLang is needed — Claude
models act as the players. Each side can be a different model.

  ANTHROPIC_API_KEY must be set.
  python scripts/pvp_anthropic_match.py
  python scripts/pvp_anthropic_match.py --model-a claude-haiku-4-5 --model-b claude-3-haiku-20240307 \
      --env leduc_poker --num-games 1 --verbose

Note: claude-3-5-haiku is retired (404). The cheapest active "old" model is
claude-3-haiku-20240307 (deprecated but active), used as model-b by default.
"""

import argparse
import functools

import anthropic

from core.constants import EnvironmentName
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatResult
from core.models.pvp_models import ChatRole
from core.models.pvp_models import PvPMatchupConfig
from core.models.pvp_models import ToolCall
from core.models.pvp_models import ToolSchema
from validator.evaluation.pvp.game_runner import Player
from validator.evaluation.pvp.game_runner import run_matchup


# --- Anthropic adapter: our ChatFn contract -> Messages API with tool use ---


def _to_anthropic(messages: list[ChatMessage]) -> tuple[str, list[dict]]:
    """Convert our messages into (system_text, anthropic_messages).

    System messages fold into the top-level system param. Consecutive tool
    results merge into one user message (Anthropic requires tool_result blocks
    in a user turn following the assistant's tool_use).
    """
    system_parts: list[str] = []
    out: list[dict] = []
    pending_tool_results: list[dict] = []

    def flush_results() -> None:
        nonlocal pending_tool_results
        if pending_tool_results:
            out.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    for msg in messages:
        if msg.role == ChatRole.SYSTEM:
            if msg.content:
                system_parts.append(msg.content)
            continue
        if msg.role == ChatRole.TOOL:
            pending_tool_results.append(
                {"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.content or ""}
            )
            continue
        flush_results()
        if msg.role == ChatRole.USER:
            out.append({"role": "user", "content": msg.content or ""})
        elif msg.role == ChatRole.ASSISTANT:
            blocks: list[dict] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls or []:
                blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
            out.append({"role": "assistant", "content": blocks})
    flush_results()
    return "\n\n".join(system_parts), out


def _to_anthropic_tools(tools: list[ToolSchema] | None) -> list[dict]:
    return [
        {"name": t.function.name, "description": t.function.description, "input_schema": t.function.parameters}
        for t in tools or []
    ]


def anthropic_chat(
    client: anthropic.Anthropic,
    config: ChatCompletionConfig,
    messages: list[ChatMessage],
    tools: list[ToolSchema] | None = None,
) -> ChatResult:
    """ChatFn backed by the Anthropic Messages API. Model id comes from config.inference_model."""
    system, anthropic_messages = _to_anthropic(messages)
    kwargs: dict = {
        "model": config.inference_model,
        "max_tokens": config.max_tokens,
        "system": system,
        "messages": anthropic_messages,
    }
    if tools:
        kwargs["tools"] = _to_anthropic_tools(tools)
        kwargs["tool_choice"] = {"type": "auto"}

    response = client.messages.create(**kwargs)

    content_text: str | None = None
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            content_text = (content_text or "") + block.text
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
    return ChatResult(content=content_text, tool_calls=tool_calls or None)


def _traced(chat_fn, label: str):
    """Wrap a ChatFn to print each call's legal actions + the model's tool calls."""

    def wrapped(config, messages, tools=None):
        result = chat_fn(config, messages, tools)
        calls = ", ".join(
            f"{c.name}({c.arguments})" if c.name != "game_action" else f"game_action -> {c.arguments.get('action_id')}"
            for c in (result.tool_calls or [])
        ) or "(no tool call)"
        print(f"  [{label} · {config.inference_model}] {calls}")
        return result

    return wrapped


def _player(client: anthropic.Anthropic, model: str, port: int, label: str | None) -> Player:
    config = ChatCompletionConfig(inference_model=model, base_url=f"http://anthropic:{port}/v1")
    chat = functools.partial(anthropic_chat, client)
    if label:
        chat = _traced(chat, label)
    return Player(client=client, config=config, chat_fn=chat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="claude-haiku-4-5")
    parser.add_argument("--model-b", default="claude-3-haiku-20240307")
    parser.add_argument("--env", default="leduc_poker")
    parser.add_argument("--num-games", type=int, default=1, help="seeds; each played twice (position swap)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    player_a = _player(client, args.model_a, 30000, "A" if args.verbose else None)
    player_b = _player(client, args.model_b, 30001, "B" if args.verbose else None)

    print(f"=== {args.model_a} (A) vs {args.model_b} (B) — {args.env}, {args.num_games} seed(s) ===")
    result = run_matchup(
        env_name=EnvironmentName(args.env),
        matchup_config=PvPMatchupConfig(num_games=args.num_games),
        player_a=player_a,
        player_b=player_b,
        base_seed=args.seed,
    )
    print("\n=== RESULT ===")
    print(f"  {args.model_a} (A) wins: {result.model_a_wins}")
    print(f"  {args.model_b} (B) wins: {result.model_b_wins}")
    print(f"  draws: {result.draws}   total games: {result.total_games}")


if __name__ == "__main__":
    main()
