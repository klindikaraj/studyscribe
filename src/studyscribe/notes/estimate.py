"""Estimate of a notes generation before anything is sent: requests, tokens, time and cost.

Pure: the caller passes the size of each map prompt (rendered, so the template and the
transcript are counted exactly in characters) and the reduce budget; output sizes come from
assumptions measured in the probes. The hierarchical reduce is simulated with the same grouping
the pipeline uses. Estimates are about ±30%; the real figures come from the stored usage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from studyscribe.notes import pricing
from studyscribe.notes.providers.base import Usage
from studyscribe.notes.reduce_plan import fits, plan_groups

MAX_LEVELS = 6


@dataclass(frozen=True)
class Assumptions:
    chars_per_token: float = 3.4  # Gemini's counter: 3.38 (en), 3.51 (it); not measured for qwen
    map_output_tokens: int = 2500  # probe: about 1.7-2.6k per 15-minute block, thinking off
    partial_output_tokens: int = 4000
    reduce_output_tokens: int = 6000
    thinking_tokens: int = 7000  # per request with thinking on (probe: 7,183 on a map block)
    retry_margin: float = 0.15  # validation retries
    prefill_tps: float = 1500.0  # qwen3.5:9b on the RTX 3060 (probe: 1,424-1,691)
    generate_tps: float = 45.0  # (probe: 44-53)


@dataclass(frozen=True)
class RequestEstimate:
    stage: str  # "map", "partial", "reduce"
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Estimate:
    requests: tuple[RequestEstimate, ...]
    retry_margin: float

    def count(self, stage: str | None = None) -> int:
        return sum(1 for r in self.requests if stage is None or r.stage == stage)

    @property
    def input_tokens(self) -> int:
        return math.ceil(sum(r.input_tokens for r in self.requests) * (1 + self.retry_margin))

    @property
    def output_tokens(self) -> int:
        return math.ceil(sum(r.output_tokens for r in self.requests) * (1 + self.retry_margin))

    def seconds(self, prefill_tps: float, generate_tps: float) -> float:
        """Local model: reading the prompts plus writing the answers (load time not included)."""
        return self.input_tokens / prefill_tps + self.output_tokens / generate_tps

    def cost(self, provider: str, model: str, *, free_tier: bool) -> tuple[float, float] | None:
        """(charged, list price) in USD, request by request (prices can depend on a request's
        size), with the retry margin; None for a model without a known price."""
        charged = listed = 0.0
        for request in self.requests:
            usage = Usage(input_tokens=request.input_tokens, output_tokens=request.output_tokens)
            one = pricing.list_cost(provider, model, usage)
            if one is None:
                return None
            listed += one
            charged += pricing.charged_cost(provider, model, usage, free_tier=free_tier) or 0.0
        factor = 1 + self.retry_margin
        return charged * factor, listed * factor


def estimate_notes(map_prompt_chars: Sequence[int], reduce_system_chars: int,
                   reduce_budget_tokens: int, *, thinking: bool,
                   assumptions: Assumptions = Assumptions()) -> Estimate:
    """`map_prompt_chars`: system + user characters of each block's map prompt;
    `reduce_system_chars`: characters of a reduce prompt without its input notes;
    `reduce_budget_tokens`: input-note tokens one reduce request may hold."""
    a = assumptions
    extra = a.thinking_tokens if thinking else 0
    requests = [RequestEstimate("map", _tokens(chars, a), a.map_output_tokens + extra)
                for chars in map_prompt_chars]
    system = _tokens(reduce_system_chars, a)
    sizes = [a.map_output_tokens] * len(map_prompt_chars)
    for _ in range(MAX_LEVELS):
        if not sizes or fits(sizes, reduce_budget_tokens):
            break
        groups = plan_groups(sizes, reduce_budget_tokens)
        if all(len(g) == 1 for g in groups):
            break  # the pipeline stops here with an error; the estimate just ends
        next_sizes = []
        for group in groups:
            if len(group) == 1:
                next_sizes.append(sizes[group[0]])
                continue
            requests.append(RequestEstimate("partial", system + sum(sizes[i] for i in group),
                                            a.partial_output_tokens + extra))
            next_sizes.append(a.partial_output_tokens)
        sizes = next_sizes
    if sizes:
        requests.append(RequestEstimate("reduce", system + sum(sizes),
                                        a.reduce_output_tokens + extra))
    return Estimate(tuple(requests), a.retry_margin)


def _tokens(chars: int, assumptions: Assumptions) -> int:
    return math.ceil(chars / assumptions.chars_per_token)
