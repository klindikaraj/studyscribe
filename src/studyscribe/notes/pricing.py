"""List prices per million tokens and the cost of a request from its usage.

Checked against the providers' pricing pages on PRICES_VERIFIED; prices change, so the CLI
prints that date next to every estimate. Unknown models have no price (cost None).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from studyscribe.notes.providers.base import Usage

PRICES_VERIFIED = "2026-11-03"
MTOK = 1_000_000


@dataclass(frozen=True)
class Price:
    input: float  # USD per million input tokens
    output: float  # USD per million output tokens (thinking included)
    cache_read: float | None = None  # None: cached tokens charged as input
    cache_write: float | None = None
    # Above this many input tokens per request (cache reads and writes included) the whole
    # request is charged at the `long_*` prices (Claude Haiku 5.5: 100,000).
    long_threshold: int | None = None
    long_input: float | None = None
    long_output: float | None = None
    long_cache_read: float | None = None


# (provider, model) -> list of (valid from, price), oldest first.
PRICES: dict[tuple[str, str], list[tuple[date, Price]]] = {
    ("anthropic", "claude-haiku-5-5"): [(date(2026, 1, 1), Price(
        input=0.10, output=0.50, cache_read=0.01, cache_write=0.125,  # 5 min cache write
        long_threshold=100_000, long_input=0.50, long_output=2.50, long_cache_read=0.05,
    ))],
    ("anthropic", "claude-sonnet-5-5"): [(date(2026, 1, 1), Price(
        input=2.0, output=10.0, cache_read=0.10, cache_write=2.50,
    ))],
    # Gemini cache prices were not checked: cached tokens are charged as input (an upper bound).
    ("gemini", "gemini-3.8-flash"): [
        (date(2026, 1, 1), Price(input=0.75, output=3.75)),
        (date(2027, 1, 1), Price(input=1.50, output=7.50)),
    ],
    ("gemini", "gemini-3.5-flash-lite"): [(date(2026, 1, 1), Price(input=0.30, output=2.50))],
}


def price_for(provider: str, model: str, on: date | None = None) -> Price | None:
    periods = PRICES.get((provider, model))
    if not periods:
        return None
    on = on or date.today()
    current = None
    for valid_from, price in periods:
        if valid_from <= on:
            current = price
    return current or periods[0][1]


def list_cost(provider: str, model: str, usage: Usage, on: date | None = None) -> float | None:
    """Cost in USD at list price, or None for a model without a known price. Local models
    (Ollama) cost nothing."""
    if provider == "ollama":
        return 0.0
    price = price_for(provider, model, on)
    if price is None:
        return None
    long = price.long_threshold is not None and usage.input_tokens > price.long_threshold
    input_price = price.long_input if long else price.input
    output_price = price.long_output if long else price.output
    cache_read = (price.long_cache_read if long else price.cache_read)
    if cache_read is None:
        cache_read = input_price
    cache_write = price.cache_write if price.cache_write is not None else input_price
    plain_input = usage.input_tokens - usage.cached_tokens - usage.cache_write_tokens
    return (plain_input * input_price
            + usage.cached_tokens * cache_read
            + usage.cache_write_tokens * cache_write
            + usage.output_tokens * output_price) / MTOK


def charged_cost(provider: str, model: str, usage: Usage, *, free_tier: bool,
                 on: date | None = None) -> float | None:
    """What the request actually costs: 0 on the Gemini free tier, else the list price."""
    if free_tier and provider == "gemini":
        return 0.0
    return list_cost(provider, model, usage, on)
