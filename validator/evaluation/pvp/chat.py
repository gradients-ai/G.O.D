"""Synchronous OpenAI-compatible chat client for PvP bot inference.

Uses the sync openai.OpenAI client since evaluate_bots calls
bot.step() synchronously — no async machinery needed.
"""

import logging
import re
import time

import openai

from core.models.pvp_models import ChatCompletionConfig, ChatMessage, ChatResult
from validator.core import constants as vcst

logger = logging.getLogger(__name__)

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
) -> ChatResult:
    """Call chat endpoint with retries. Client should be created via create_client()."""
    return _with_retries(client, config, messages)


def _with_retries(
    client: openai.OpenAI,
    config: ChatCompletionConfig,
    messages: list[ChatMessage],
) -> ChatResult:
    """Execute chat with exponential backoff on transient failures."""
    last_exc: BaseException | None = None
    attempts = config.max_retries + 1

    for attempt in range(attempts):
        try:
            return _call(client, config, messages)

        except (TimeoutError, openai.APITimeoutError, openai.APIConnectionError) as exc:
            last_exc = exc
            if attempt < attempts - 1:
                wait = min(2**attempt, vcst.PVP_RETRY_BACKOFF_CAP_SECONDS)
                logger.warning(
                    "Chat attempt %d/%d failed (%s), retrying in %ds",
                    attempt + 1, attempts, type(exc).__name__, wait,
                )
                time.sleep(wait)

        except openai.APIStatusError as exc:
            if exc.status_code >= 500 and attempt < attempts - 1:
                last_exc = exc
                time.sleep(min(2**attempt, vcst.PVP_RETRY_BACKOFF_CAP_SECONDS))
                continue
            raise

    raise RuntimeError(f"Chat failed after {attempts} attempts: {last_exc}")


def _call(
    client: openai.OpenAI,
    config: ChatCompletionConfig,
    messages: list[ChatMessage],
) -> ChatResult:
    """Execute a single chat completion request."""
    messages_dicts = [msg.model_dump() for msg in messages]

    response = client.chat.completions.create(
        model=config.inference_model,
        messages=messages_dicts,
        temperature=config.temperature,
        seed=config.seed,
        max_tokens=config.max_tokens,
    )

    content: str | None = None
    if response.choices and response.choices[0].message.content:
        raw = response.choices[0].message.content.strip()
        content = strip_think_tags(raw) if raw else None

    usage: dict[str, int] | None = None
    if response.usage:
        usage = response.usage.model_dump()

    return ChatResult(content=content, usage=usage)
