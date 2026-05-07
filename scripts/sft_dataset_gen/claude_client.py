import asyncio
import logging
import random
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)

MODEL_PRICING = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
}

MAX_RETRIES = 5


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, model: str) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        pricing = MODEL_PRICING.get(model, {"input": 3.0, "output": 15.0})
        self.cost_usd += (
            input_tokens * pricing["input"] + output_tokens * pricing["output"]
        ) / 1_000_000


class ClaudePlayer:
    def __init__(
        self,
        sonnet_model: str = "claude-sonnet-4-6",
        haiku_model: str = "claude-haiku-4-5-20251001",
        haiku_ratio: float = 0.8,
        temperature: float = 0.7,
    ):
        self.client = anthropic.AsyncAnthropic()
        self.sonnet_model = sonnet_model
        self.haiku_model = haiku_model
        self.haiku_ratio = haiku_ratio
        self.temperature = temperature
        self.usage = TokenUsage()

    def pick_model(self) -> str:
        if random.random() < self.haiku_ratio:
            return self.haiku_model
        return self.sonnet_model

    async def get_action(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_override: str | None = None,
    ) -> tuple[str, str]:
        model = model_override or self.pick_model()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self.client.messages.create(
                    model=model,
                    max_tokens=512,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=messages,
                )
                self.usage.add(
                    response.usage.input_tokens, response.usage.output_tokens, model,
                )
                return response.content[0].text, model
            except anthropic.RateLimitError:
                if attempt < MAX_RETRIES:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.warning(
                        "Rate limited (attempt %d/%d), waiting %.1fs",
                        attempt, MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise
        raise RuntimeError("Unreachable")
