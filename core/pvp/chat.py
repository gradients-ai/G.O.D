"""Synchronous OpenAI-compatible chat client for PvP bot inference.

Uses the sync openai.OpenAI client since evaluate_bots calls
bot.step() synchronously — no async machinery needed.
"""

import json
import logging
import re
import time

import openai

from core.models.pvp_models import ChatCompletionConfig
from core.models.pvp_models import ChatMessage
from core.models.pvp_models import ChatResult
from core.models.pvp_models import JsonScalar
from core.models.pvp_models import ToolCall
from core.models.pvp_models import ToolSchema
from core.pvp import constants as cst


logger = logging.getLogger(__name__)


class ChatUnavailableError(RuntimeError):
    """Inference endpoint failed after exhausting retries.

    Carries the underlying attempt history so callers can distinguish a slow
    model (any timeout in the retry chain) from an unreachable server (pure
    connection failures). A bare RuntimeError here used to crash the whole
    matchup; this typed error lets the bot layer choose forfeit vs infra replay.
    """

    def __init__(
        self,
        cause: BaseException | None,
        attempts: int,
        causes: list[BaseException] | None = None,
    ):
        self.cause = cause
        self.causes = tuple(causes or ([cause] if cause is not None else []))
        super().__init__(f"Chat failed after {attempts} attempts: {cause}")

    @property
    def timed_out(self) -> bool:
        return any(isinstance(cause, (TimeoutError, openai.APITimeoutError)) for cause in self.causes)

    def cause_types(self) -> str:
        return ", ".join(type(cause).__name__ for cause in self.causes) or "unknown"


_THINK_COMPLETE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think(?:ing)?>.*", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Remove <think>/<thinking> blocks from model output."""
    cleaned = _THINK_COMPLETE.sub("", text)
    for tag in ("</think>", "</thinking>"):
        if tag in cleaned:
            cleaned = cleaned.split(tag)[-1]
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    return cleaned.strip()


def create_client(config: ChatCompletionConfig) -> openai.OpenAI:
    """Create a reusable sync OpenAI client from config. Caller owns lifecycle."""
    return openai.OpenAI(
        base_url=config.base_url.rstrip("/"),
        api_key=config.api_key,
        timeout=config.read_timeout,
        max_retries=0,
    )


def chat_completion(
    client: openai.OpenAI,
    config: ChatCompletionConfig,
    messages: list[ChatMessage],
    tools: list[ToolSchema] | None = None,
) -> ChatResult:
    """Call chat endpoint with retries. Client should be created via create_client()."""
    return _with_retries(client, config, messages, tools)


def _with_retries(
    client: openai.OpenAI,
    config: ChatCompletionConfig,
    messages: list[ChatMessage],
    tools: list[ToolSchema] | None = None,
) -> ChatResult:
    """Execute chat with exponential backoff on transient failures."""
    last_exc: BaseException | None = None
    transient_causes: list[BaseException] = []
    attempts = config.max_retries + 1

    for attempt in range(attempts):
        try:
            return _call(client, config, messages, tools)

        except (TimeoutError, openai.APITimeoutError, openai.APIConnectionError) as exc:
            last_exc = exc
            transient_causes.append(exc)
            if attempt < attempts - 1:
                wait = min(2**attempt, cst.PVP_RETRY_BACKOFF_CAP_SECONDS)
                logger.warning(
                    "Chat attempt %d/%d failed (%s), retrying in %ds",
                    attempt + 1, attempts, type(exc).__name__, wait,
                )
                time.sleep(wait)

        except openai.APIStatusError as exc:
            if exc.status_code >= 500 and attempt < attempts - 1:
                last_exc = exc
                transient_causes.append(exc)
                time.sleep(min(2**attempt, cst.PVP_RETRY_BACKOFF_CAP_SECONDS))
                continue
            raise

    raise ChatUnavailableError(last_exc, attempts, transient_causes)


def _call(
    client: openai.OpenAI,
    config: ChatCompletionConfig,
    messages: list[ChatMessage],
    tools: list[ToolSchema] | None = None,
) -> ChatResult:
    """Execute a single chat completion request."""
    kwargs = {
        "model": config.inference_model,
        "messages": [msg.to_openai() for msg in messages],
        "temperature": config.temperature,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
    }
    if tools:
        kwargs["tools"] = [tool.to_openai() for tool in tools]
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message if response.choices else None

    content: str | None = None
    if message is not None and message.content:
        raw = message.content.strip()
        content = strip_think_tags(raw) if raw else None

    tool_calls = _parse_tool_calls(message)

    usage: dict[str, int] | None = None
    if response.usage:
        usage = response.usage.model_dump()

    return ChatResult(content=content, tool_calls=tool_calls, usage=usage)


def _parse_tool_calls(message: object | None) -> list[ToolCall] | None:
    """Normalise the SDK's tool_calls into our ToolCall model, decoding arguments JSON."""
    raw_calls = getattr(message, "tool_calls", None)
    if not isinstance(raw_calls, list) or not raw_calls:
        return None
    return [
        ToolCall(id=call.id, name=call.function.name, arguments=_decode_arguments(call.function.arguments))
        for call in raw_calls
    ]


def _decode_arguments(raw: str | None) -> dict[str, JsonScalar]:
    """Decode a tool call's JSON arguments string; tolerate malformed output.

    String values are scrubbed of think tags: reasoning models can leak
    <think> blocks into a memory write's content, which would otherwise be
    stored verbatim and waste slot budget.
    """
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {k: _coerce_argument_value(v) for k, v in decoded.items()}


def _coerce_argument_value(value) -> JsonScalar:
    if isinstance(value, str):
        return strip_think_tags(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True)
