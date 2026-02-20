"""Shared LLM client with retry logic for all Claude API calls."""

from __future__ import annotations

import random
import time

import anthropic
from dotenv import load_dotenv

from src.logger import setup_logger

load_dotenv()

log = setup_logger("llm")

MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 5
BASE_DELAY = 5
MAX_DELAY = 60


def call_claude(
    *,
    system: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> anthropic.types.Message:
    """Call Claude with automatic retry on transient API errors (429/529).

    Uses exponential backoff with jitter to avoid thundering herd.
    """
    client = anthropic.Anthropic()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
            log.info(
                f"API call succeeded: {response.usage.input_tokens:,} input, "
                f"{response.usage.output_tokens:,} output tokens"
            )
            return response

        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < MAX_RETRIES - 1:
                delay = min(BASE_DELAY * 2**attempt, MAX_DELAY)
                jitter = random.uniform(0, delay * 0.5)
                total_wait = delay + jitter
                log.warning(
                    f"API error {e.status_code}, retrying in {total_wait:.1f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})..."
                )
                time.sleep(total_wait)
            else:
                raise

    raise RuntimeError("Unreachable — loop should either return or raise")
