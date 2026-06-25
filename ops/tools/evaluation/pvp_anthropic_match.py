#!/usr/bin/env python3
"""Play a PvP matchup with two Anthropic models through the real harness."""

import argparse
import functools

import anthropic

from core.constants.environments import EnvironmentName
from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatResult
from core.models.pvp_models import ChatRole
from core.models.pvp_models import PvPMatchupConfig
from core.models.pvp_models import ToolCall
from core.models.pvp_models import ToolSchema
from validator.evaluation.pvp.game_runner import Player
from validator.evaluation.pvp.game_runner import run_matchup


def _to_anthropic(messages: list[ChatMessage]) -> tuple[str, list[dict]]:
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
        {"name": tool.function.name, "description": tool.function.description, "input_schema": tool.function.parameters}
        for tool in tools or []
    ]


def anthropic_chat(
    client: anthropic.Anthropic,
    config: ChatCompletionConfig,
    messages: list[ChatMessage],
    tools: list[ToolSchema] | None = None,
) -> ChatResult:
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
    def wrapped(config, messages, tools=None):
        result = chat_fn(config, messages, tools)
        calls = ", ".join(
            f"{call.name}({call.arguments})"
            if call.name != "game_action"
            else f"game_action -> {call.arguments.get('action_id')}"
            for call in (result.tool_calls or [])
        ) or "(no tool call)"
        print(f"  [{label} - {config.inference_model}] {calls}")
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
    parser.add_argument("--time-budget-seconds", type=float, default=900.0, help="wall-clock budget for the matchup")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    player_a = _player(client, args.model_a, 30000, "A" if args.verbose else None)
    player_b = _player(client, args.model_b, 30001, "B" if args.verbose else None)

    print(f"=== {args.model_a} (A) vs {args.model_b} (B) - {args.env}, {args.time_budget_seconds:.0f}s budget ===")
    result = run_matchup(
        env_name=EnvironmentName(args.env),
        matchup_config=PvPMatchupConfig(time_budget_seconds=args.time_budget_seconds),
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
