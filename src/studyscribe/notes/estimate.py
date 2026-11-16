"""Estimate of a notes generation before anything is sent: requests, tokens, time and cost.

Pure: the caller passes the size of each map prompt (rendered, so the template and the
transcript are counted exactly in characters), the speech of each block and the reduce budget.
The output of every request is sized from the template's targets for that request (items times
the tokens an item takes), because that is what the prompt asks for; the hierarchical reduce
is simulated with the grouping the pipeline uses. Estimates are about ±30%; the real figures
come from the stored usage.

Measured on 2026-11-16 (qwen3.5:9b, thinking off, map prompts v4 and v5 on a 15-minute block of
each test lecture): 860-1,518 output tokens per block, 55-69 tokens per item, 50.9-52.3
tokens/s generated, 1,882-1,903 tokens/s of prefill, 3.63-3.87 characters per token. Reduce
requests have not been measured yet.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from studyscribe.notes import pricing
from studyscribe.notes.providers.base import Usage
from studyscribe.notes.reduce_plan import fits, plan_groups
from studyscribe.notes.templates import Template

MAX_LEVELS = 6


@dataclass(frozen=True)
class Assumptions:
    chars_per_token: float = 3.7  # qwen3.5:9b on the map prompts: 3.63-3.87
    grouping_chars_per_token: float = 2.5  # what the pipeline sizes reduce groups with
    tokens_per_item: int = 65  # JSON included (measured 55-69)
    tokens_per_sentence: int = 35  # summary sentences (not measured separately)
    answer_overhead_tokens: int = 30  # braces, keys of empty sections
    thinking_tokens: int = 7000  # per request with thinking on (probe: 7,183 on a map block)
    retry_margin: float = 0.15  # validation retries
    prefill_tps: float = 1850.0  # measured 1,882-1,903
    generate_tps: float = 50.0  # measured 50.9-52.3


@dataclass(frozen=True)
class RequestEstimate:
    stage: str  # "map", "partial", "reduce"
    input_tokens: int
    output_tokens: int  # the answer, thinking included


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

    def largest_output(self, stage: str) -> int:
        return max((r.output_tokens for r in self.requests if r.stage == stage), default=0)

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


def answer_tokens(template: Template, speech_ms: int, *, whole_lecture: bool,
                  assumptions: Assumptions = Assumptions()) -> int:
    """Tokens of an answer that meets the template's targets for this much speech."""
    a = assumptions
    targets = template.targets(speech_ms, whole_lecture=whole_lecture)
    tokens = a.answer_overhead_tokens
    for section in template.sections:
        per = a.tokens_per_sentence if section.kind == "summary" else a.tokens_per_item
        tokens += targets[section.key] * per
    return tokens


def estimate_notes(template: Template, block_duration_ms: Sequence[int],
                   block_speech_ms: Sequence[int], map_prompt_chars: Sequence[int],
                   reduce_system_chars: int, reduce_budget_tokens: int, *, thinking: bool,
                   assumptions: Assumptions = Assumptions()) -> Estimate:
    """Per block: its duration (the map's targets), its speech (the reduces' targets) and the
    characters of its map prompt (system + user), as the pipeline uses them;
    `reduce_system_chars`: characters of a reduce prompt without its input notes;
    `reduce_budget_tokens`: what one reduce may hold of input notes, in the pipeline's
    conservative tokens (characters / `grouping_chars_per_token`)."""
    a = assumptions
    if not len(block_duration_ms) == len(block_speech_ms) == len(map_prompt_chars):
        raise ValueError("one duration, one speech and one prompt size per block")
    extra = a.thinking_tokens if thinking else 0
    parts = []  # (answer tokens, speech ms) of the parts the next reduce gets
    requests = []
    for duration, speech, chars in zip(block_duration_ms, block_speech_ms, map_prompt_chars,
                                       strict=True):
        answer = answer_tokens(template, duration, whole_lecture=False, assumptions=a)
        requests.append(RequestEstimate("map", _tokens(chars, a), answer + extra))
        parts.append((answer, speech))
    system = _tokens(reduce_system_chars, a)
    grouping = a.chars_per_token / a.grouping_chars_per_token
    for _ in range(MAX_LEVELS):
        sizes = [math.ceil(tokens * grouping) for tokens, _ in parts]
        if not sizes or fits(sizes, reduce_budget_tokens):
            break
        groups = plan_groups(sizes, reduce_budget_tokens)
        if all(len(g) == 1 for g in groups):
            break  # the pipeline stops here with an error; the estimate just ends
        merged = []
        for group in groups:
            if len(group) == 1:
                merged.append(parts[group[0]])
                continue
            speech = sum(parts[i][1] for i in group)
            answer = answer_tokens(template, speech, whole_lecture=False, assumptions=a)
            requests.append(RequestEstimate("partial", system + sum(parts[i][0] for i in group),
                                            answer + extra))
            merged.append((answer, speech))
        parts = merged
    if parts:
        speech = sum(s for _, s in parts)
        answer = answer_tokens(template, speech, whole_lecture=True, assumptions=a)
        requests.append(RequestEstimate("reduce", system + sum(t for t, _ in parts),
                                        answer + extra))
    return Estimate(tuple(requests), a.retry_margin)


def _tokens(chars: int, assumptions: Assumptions) -> int:
    return math.ceil(chars / assumptions.chars_per_token)
