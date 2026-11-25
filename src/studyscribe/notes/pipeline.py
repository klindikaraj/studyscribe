"""The notes pipeline: map every block, reduce (hierarchically when needed), check, resolve.

No database and no files: the runner (piece 3) gives it the transcript lines and a provider,
stores every call it reports through `on_call`, and passes back the answers of the map blocks
already accepted to resume a generation.

For every request:
- the answer is validated against the template's output model; on failure the request is sent
  again with the validation error appended (at most `validation_retries` more times);
- references are checked (refs.py): a map may cite only its block's lines, a reduce only the
  lines cited in its input. Invalid ones are dropped and counted; above a quarter invalid, the
  answer is not trusted and the request is sent again like an invalid one;
- `Truncated` (the output cap was reached): sent once more with twice the cap when the context
  allows it, otherwise the generation stops. The input of every reduce leaves that room free
  (thinking off); the final reduce, which may write far more than it is asked, gets it at once;
- transient provider errors are retried by retry.py; any other provider error stops the
  generation (the work done so far is in the calls already reported).

Nothing else is dropped: code items get a graded check (code_check.py) and a map answer that
stops citing before 85% of its block gets a coverage warning; both only warn.

The reduce works on counted tokens, never on the estimate (estimate.py is only for the quote
before a run): every part's notes are counted by the provider as the reduce input holds them,
the reduce system prompt is counted, and every reduce request is counted before it is sent
(and refused if it does not fit). The expected answer of a reduce is its targets times the
tokens per item counted on the parts' answers as the model wrote them (indented JSON), times
TARGET_OVERSHOOT: a partial group never covers more speech than an answer under the output cap
can hold, and the final targets are scaled down (with a warning) when the template asks for more
than one answer can hold. Parts too large to pair are shortened one by one rather than stopping
the run.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from studyscribe.notes.blocks import Block
from studyscribe.notes.code_check import annotate_code
from studyscribe.notes.prompts import PartNotes, Prompt, map_prompt, reduce_prompt
from studyscribe.notes.providers.base import (
    Effort,
    LlmError,
    LlmProvider,
    LlmRequest,
    LlmResponse,
    Timings,
    Truncated,
    Usage,
)
from studyscribe.notes.providers.retry import Pacer, RetryPolicy, call_with_retry
from studyscribe.notes.reduce_plan import ReduceTooLarge, fits, plan_groups
from studyscribe.notes.refs import RefReport, check_refs, refs_in, resolve
from studyscribe.notes.schema import output_model, output_schema
from studyscribe.notes.templates import PARTIAL_SLACK, Template
from studyscribe.notes.transcript import Line
from studyscribe.notes.uncertain import UncertainWord, uncertain_words
from studyscribe.transcribe.export import clock

Stage = Literal["map", "partial", "reduce"]
DEFAULT_NUM_CTX = 32768
MARGIN_TOKENS = 512
PART_OVERHEAD_TOKENS = 20  # {"part":1,"from":"0:00:09","to":"0:12:41","notes":...}
MIN_COVERAGE = 0.85  # share of a block's lines up to the last one a map answer cites
# How far past its targets a reduce may write. Measured in piece 5 (qwen3.5:9b, 2026-11-25): up to
# 1.92 times the items asked for (159 list items for 83) and 1.65 times the expected tokens.
TARGET_OVERSHOOT = 2.0
ANSWER_OVERHEAD_TOKENS = 50  # braces and keys of an answer
# The counted reduce system prompt versus the one sent, plus the feedback of a retry.
REQUEST_MARGIN_TOKENS = 512
_SENTENCE = re.compile(r"[^.!?]+[.!?]")


class NotesError(Exception):
    """The generation cannot go on; `message` is for the user."""

    def __init__(self, message: str, *, remedy: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy


class InvalidAnswer(NotesError):
    """The model's answer was still invalid after the retries."""


@dataclass(frozen=True)
class Lecture:
    title: str
    course: str | None
    language: str  # the language the notes are written in


def _default_max_input(max_output_tokens: int) -> int:
    return DEFAULT_NUM_CTX - max_output_tokens - MARGIN_TOKENS


@dataclass(frozen=True)
class PipelineOptions:
    effort: Effort = "none"
    map_output_tokens: int = 6144
    reduce_output_tokens: int = 8192
    thinking_output_tokens: int = 16384  # any request with thinking on (a block went past 6144)
    # Input tokens one request may have for a given output cap (Ollama: provider.budget).
    max_input_tokens: Callable[[int], int] = _default_max_input
    # Upper bound (small models lose quality on long inputs); the context usually allows less.
    reduce_max_input: int = 16_000
    chars_per_token: float = 2.5  # conservative, to size reduce groups
    validation_retries: int = 2
    max_levels: int = 4  # partial-reduce levels
    retry_policy: RetryPolicy = RetryPolicy()
    pacer: Pacer | None = None
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random | None = None

    def output_cap(self, stage: Stage) -> int:
        if self.effort != "none":
            return self.thinking_output_tokens
        return self.map_output_tokens if stage == "map" else self.reduce_output_tokens

    def reduce_reserve(self) -> int:
        """Output tokens the input of a reduce must leave free: the doubled cap of a retry. With
        thinking on the cap alone, already that large: doubled it would fill a 32k context."""
        cap = self.output_cap("reduce")
        return 2 * cap if self.effort == "none" else cap


@dataclass(frozen=True)
class CallRecord:
    """One request sent to the provider, for the `llm_calls` table."""

    stage: Stage
    block: int  # map: block index; partial: running group number; reduce: 1
    attempt: int  # from 1 for each stage and block, every request sent counted
    status: Literal["ok", "invalid", "error"]
    usage: Usage | None
    duration_s: float | None
    model_version: str | None = None
    request_id: str | None = None
    timings: Timings | None = None
    output_json: str | None = None  # the raw answer; with 'error', the text of a cut one
    error: str | None = None


@dataclass(frozen=True)
class CoverageWarning:
    """A map answer whose last cited line comes before MIN_COVERAGE of its block: the end of
    the block is probably missing from the notes. A warning, nothing is dropped."""

    block: int
    last_cited: int | None  # None: the answer cites nothing
    uncovered_from: int  # first S-number after the last cited one
    uncovered_to: int  # the block's last S-number
    start_ms: int  # start of the first uncovered line
    end_ms: int  # end of the block

    def as_dict(self) -> dict[str, Any]:
        return {"block": self.block, "last_cited": self.last_cited,
                "uncovered": [self.uncovered_from, self.uncovered_to],
                "start_ms": self.start_ms, "end_ms": self.end_ms}

    def describe(self) -> str:
        cited = f"S{self.last_cited}" if self.last_cited is not None else "nothing"
        return (f"block {self.block}: the notes cite up to {cited}; lines "
                f"S{self.uncovered_from}-S{self.uncovered_to} ({clock(self.start_ms)}-"
                f"{clock(self.end_ms)}) are not covered")


def coverage_gap(notes: Mapping[str, Any], block: Block,
                 min_share: float = MIN_COVERAGE) -> CoverageWarning | None:
    """The warning for a map answer (plain notes, references already checked) whose last cited
    line is before `min_share` of the block's lines, or None."""
    cited = [n for n in refs_in([notes]) if block.first_n <= n <= block.last_n]
    last = max(cited) if cited else None
    covered = (last - block.first_n + 1) if last is not None else 0
    if covered >= min_share * len(block.lines):
        return None
    first_uncovered = block.lines[covered]
    return CoverageWarning(block.index, last, first_uncovered.n, block.last_n,
                           first_uncovered.start_ms, block.end_ms)


@dataclass(frozen=True)
class _Part:
    """The notes of a part as the reduce gets them, with their counted size."""

    notes: PartNotes
    speech_ms: int
    tokens: int  # counted by the provider on the JSON the reduce input holds, plus overhead
    written: int  # counted on the answer as the model wrote it
    items: int  # list items plus summary sentences


def count_items(notes: Mapping[str, Any]) -> int:
    """Items of plain notes: every list item, and every sentence of a text section."""
    total = 0
    for value in notes.values():
        if isinstance(value, list):
            total += len(value)
        elif isinstance(value, str) and value.strip():
            total += max(1, len(_SENTENCE.findall(value)))
    return total


@dataclass
class NotesResult:
    notes: dict[str, Any]  # final notes: S-numbers, code checks added
    resolved: dict[str, Any]  # the same with every reference as segment id + start time
    map_answers: dict[int, str]  # block index -> accepted raw answer (for resume)
    uncertain: list[UncertainWord]
    stats: dict[str, Any] = field(default_factory=dict)
    coverage: list[CoverageWarning] = field(default_factory=list)


def generate_notes(provider: LlmProvider, template: Template, lines: Sequence[Line],
                   blocks: Sequence[Block], lecture: Lecture, *,
                   options: PipelineOptions = PipelineOptions(),
                   done_maps: Mapping[int, str] | None = None,
                   on_call: Callable[[CallRecord], None] | None = None,
                   on_progress: Callable[[str], None] | None = None) -> NotesResult:
    """The whole generation. Raises NotesError or LlmError when it cannot finish; every request
    sent until then has been reported through `on_call`."""
    if not blocks:
        raise NotesError("The transcript has no lines to take notes from.")
    run = _Run(provider, template, lecture, options, on_call or (lambda record: None),
               on_progress or (lambda text: None))
    return run.generate(lines, blocks, dict(done_maps or {}))


class _Run:
    def __init__(self, provider: LlmProvider, template: Template, lecture: Lecture,
                 options: PipelineOptions, on_call: Callable[[CallRecord], None],
                 on_progress: Callable[[str], None]) -> None:
        self.provider = provider
        self.template = template
        self.lecture = lecture
        self.options = options
        self.on_call = on_call
        self.progress = on_progress
        self.model = output_model(template)
        self.schema = output_schema(template)
        self.reports: dict[str, RefReport] = {s: RefReport() for s in ("map", "partial",
                                                                         "reduce")}
        self.requests = 0
        self.total_speech_ms = 0
        self.reduce_stats: dict[str, Any] = {"partial": 0, "shortened": 0,
                                             "tokens_per_item": None}
        self._system_tokens: int | None = None

    # --- stages ----------------------------------------------------------------------------------

    def generate(self, lines: Sequence[Line], blocks: Sequence[Block],
                 done_maps: dict[int, str]) -> NotesResult:
        parts: list[_Part] = []
        map_answers: dict[int, str] = {}
        coverage: list[CoverageWarning] = []
        resumed = 0
        for block in blocks:
            allowed = range(block.first_n, block.last_n + 1)
            raw = done_maps.get(block.index)
            notes = self._resume(raw, allowed)
            if notes is not None:
                resumed += 1
            else:
                self.progress(f"map {block.index}/{len(blocks)}")
                # The block's duration, not its speech: v5 with speech-based targets asked for
                # one item less per section than v4 and lost content (2026-11-16).
                targets = self.template.targets(block.duration_ms, whole_lecture=False)
                prompt = map_prompt(self.template, block.lines, part=block.index,
                                    parts=len(blocks), targets=targets, title=self.lecture.title,
                                    course=self.lecture.course, language=self.lecture.language)
                notes, raw = self._ask("map", block.index, prompt, allowed)
            map_answers[block.index] = raw
            gap = coverage_gap(notes.model_dump(), block)
            if gap is not None:
                coverage.append(gap)
                self.progress(f"warning: {gap.describe()}")
            parts.append(self._part(PartNotes(block.start_ms, block.end_ms,
                                              notes.model_dump()), raw, block.speech_ms))
        final = self._reduce(parts)
        by_n = {line.n: line for line in lines}
        code_keys = [s.key for s in self.template.sections if s.kind == "examples"]
        annotated, code_levels = annotate_code(final.model_dump(), code_keys, by_n)
        uncertain = uncertain_words(lines)
        stats = {
            "requests": self.requests,
            "resumed_blocks": resumed,
            "refs": {stage: report.as_dict() for stage, report in self.reports.items()},
            "code_checks": code_levels,
            "uncertain_words": len(uncertain),
            "coverage_warnings": [gap.as_dict() for gap in coverage],
            "reduce": dict(self.reduce_stats),
        }
        return NotesResult(notes=annotated, resolved=resolve(annotated, by_n),
                           map_answers=map_answers, uncertain=uncertain, stats=stats,
                           coverage=coverage)

    def _reduce(self, parts: list[_Part]) -> BaseModel:
        """Partial reduces until the parts fit one request, then the final reduce. Groups are
        planned on counted tokens, and a group never covers more speech than one answer can
        hold; parts too large to pair are shortened one by one instead of stopping."""
        self.total_speech_ms = sum(part.speech_ms for part in parts)
        cap = self.options.output_cap("reduce")
        final_targets = self.template.targets(self.total_speech_ms, whole_lecture=True)
        final_targets = self._fit_final_targets(final_targets, parts,
                                                self.options.reduce_reserve())
        number = 0
        for level in range(self.options.max_levels + 1):
            budget = self._reduce_budget()
            sizes = [part.tokens for part in parts]
            if fits(sizes, budget):
                break
            if level == self.options.max_levels:
                raise NotesError(f"The notes still do not fit one request after "
                                 f"{self.options.max_levels} levels of partial reduces.",
                                 remedy="Lower the template's maxima, or use a larger context.")
            # Room to choose only once: with it again, partial notes of partial notes would be as
            # long as their input, and never fit.
            slack = PARTIAL_SLACK if level == 0 else 1.0
            # Answers that fit the cap even past their targets; when that keeps every part apart,
            # answers that fit the retry's cap (with thinking on the two are the same).
            for room in (cap, self.options.reduce_reserve()):
                limit = self._max_group_speech(parts, room, slack)
                try:
                    groups = plan_groups(sizes, budget, weights=[p.speech_ms for p in parts],
                                         weight_limit=limit)
                except ReduceTooLarge:
                    groups = [[i] for i in range(len(parts))]
                if any(len(group) > 1 for group in groups):
                    break
            else:
                parts, number = self._shorten(parts, budget, number)
                continue
            merged: list[_Part] = []
            for group in groups:
                if len(group) == 1:
                    merged.append(parts[group[0]])
                    continue
                number += 1
                inputs = [parts[i] for i in group]
                speech = sum(part.speech_ms for part in inputs)
                self.progress(f"partial reduce {number}")
                targets = self.template.partial_targets(speech, self.total_speech_ms, slack)
                merged.append(self._partial(number, inputs, targets, speech))
                self.reduce_stats["partial"] += 1
            parts = merged
        self.progress("final reduce")
        return self._reduce_request("reduce", 1, parts, final_targets)[0]

    def _fit_final_targets(self, targets: dict[str, int], parts: Sequence[_Part],
                           cap: int) -> dict[str, int]:
        """The final targets, scaled down when their expected answer (at the tokens per item
        measured on the parts) would exceed the output cap; recorded and reported."""
        expected = self._expected_output(targets, parts)
        if expected <= cap:
            return targets
        items = sum(targets.values())
        room_items = (cap - ANSWER_OVERHEAD_TOKENS) / (self._tokens_per_item(parts)
                                                      * TARGET_OVERSHOOT)
        factor = room_items / items
        scaled = {s.key: max(min(1, s.min), math.floor(targets[s.key] * factor))
                  for s in self.template.sections}
        if self._expected_output(scaled, parts) > cap:
            raise NotesError(
                f"The final notes would need about {expected:,} output tokens, more than the "
                f"cap of {cap:,} for one answer, even with one item per section.",
                remedy="Use a template with fewer sections, or a larger output cap.")
        self.reduce_stats["final_targets_scaled"] = round(factor, 2)
        self.progress(f"warning: the final notes are asked for {round(factor * 100)}% of the "
                      f"template's items, so the answer fits {cap:,} output tokens")
        return scaled

    def _shorten(self, parts: list[_Part], budget: int, number: int) -> tuple[list[_Part], int]:
        """No two neighbours fit one request: every part larger than half the budget is
        reduced on its own, to its plain share of the final targets (no slack)."""
        shortened: list[_Part] = []
        for part in parts:
            if part.tokens <= budget // 2:
                shortened.append(part)
                continue
            number += 1
            self.progress(f"partial reduce {number} (shortening one part)")
            targets = self.template.partial_targets(part.speech_ms, self.total_speech_ms,
                                                    slack=1.0)
            short = self._partial(number, [part], targets, part.speech_ms)
            if short.tokens >= part.tokens:
                raise NotesError("The partial notes do not get shorter when merged or "
                                 "shortened: they cannot be reduced into one request.",
                                 remedy="Lower the template's maxima, or use a larger context.")
            self.reduce_stats["shortened"] += 1
            shortened.append(short)
        return shortened, number

    def _partial(self, number: int, inputs: Sequence[_Part], targets: Mapping[str, int],
                 speech_ms: int) -> _Part:
        notes, raw = self._reduce_request("partial", number, inputs, targets)
        return self._part(PartNotes(inputs[0].notes.start_ms, inputs[-1].notes.end_ms,
                                    notes.model_dump()), raw, speech_ms)

    def _reduce_request(self, stage: Stage, number: int, inputs: Sequence[_Part],
                        targets: Mapping[str, int]) -> tuple[BaseModel, str]:
        final = stage == "reduce"
        prompt = reduce_prompt(self.template, [part.notes for part in inputs], final=final,
                               targets=targets, title=self.lecture.title,
                               course=self.lecture.course, language=self.lecture.language)
        allowed = refs_in(part.notes.notes for part in inputs)
        return self._ask(stage, number, prompt, allowed, count_input=True)

    # --- one request -----------------------------------------------------------------------------

    def _ask(self, stage: Stage, number: int, prompt: Prompt, allowed, *,
             count_input: bool = False) -> tuple[BaseModel, str]:
        """A validated, reference-checked answer and its raw text. `count_input`: count the
        request's tokens with the provider and refuse it if it does not fit (reduces, whose
        size is planned on counted tokens)."""
        cap = self.options.output_cap(stage)
        enlarged = False
        if stage == "reduce" and self.options.reduce_reserve() > cap:
            # Its input leaves this room anyway: a long answer is not cut and asked again.
            cap, enlarged = self.options.reduce_reserve(), True
        attempt = 0
        problems = 0
        feedback = ""
        while True:
            counted = None
            if count_input:
                counted = self._count(prompt.system, prompt.user + feedback)
                room = self.options.max_input_tokens(cap)
                if counted > room:
                    raise NotesError(
                        f"The request for {_where(stage, number)} has {counted:,} input tokens, "
                        f"more than the {room:,} the context leaves. Nothing was sent.",
                        remedy="This is a bug in how the notes are grouped; please report it.")
            request = LlmRequest(system=prompt.system, user=prompt.user + feedback,
                                 schema=self.schema, max_output_tokens=cap,
                                 effort=self.options.effort, input_tokens=counted)

            def send(request: LlmRequest = request) -> LlmResponse:
                nonlocal attempt
                attempt += 1
                self.requests += 1
                return self.provider.generate(request)

            def failed(error: LlmError, _: int) -> None:
                self.on_call(CallRecord(stage, number, attempt, "error", error.usage, None,
                                        output_json=getattr(error, "text", None),
                                        error=error.user_message()))

            try:
                response = call_with_retry(
                    send, policy=self.options.retry_policy, pacer=self.options.pacer,
                    sleep=self.options.sleep, rng=self.options.rng, on_error=failed,
                    on_wait=lambda reason, s: self.progress(f"waiting {s:.0f} s ({reason})"))
            except Truncated as error:
                larger = cap * 2
                needed = (request.input_tokens if request.input_tokens is not None
                          else self._estimate(request))
                if enlarged or needed > self.options.max_input_tokens(larger):
                    if not error.remedy:
                        why = (f"It already had room for twice the normal answer ({cap:,} "
                               "tokens). Use" if enlarged else
                               f"Asking again needs {larger:,} tokens for the answer, which the "
                               "context does not leave. Use a larger context,")
                        error.remedy = (f"{why} another model or a template with fewer items. "
                                        "The cut text is stored with the call.")
                    raise
                self.progress(f"the answer was cut at {cap:,} tokens: asking again with room "
                              f"for {larger:,}")
                cap, enlarged = larger, True
                continue
            problem, notes, report = self._check(response.text, allowed)
            status = "invalid" if problem else "ok"
            self.on_call(CallRecord(stage, number, attempt, status, response.usage,
                                    response.duration_s, response.model_version,
                                    response.request_id, response.timings, response.text,
                                    problem))
            if not problem:
                self.reports[stage].add(report)
                return notes, response.text
            problems += 1
            if problems > self.options.validation_retries:
                raise InvalidAnswer(
                    f"The model's answer for {_where(stage, number)} was still invalid after "
                    f"{problems} attempts: {problem}",
                    remedy="Try again, or try another model.")
            feedback = ("\n\nYour previous answer could not be used: "
                        f"{problem}\nAnswer again, with JSON that follows the schema.")

    def _check(self, text: str, allowed) -> tuple[str | None, BaseModel | None, RefReport]:
        try:
            notes = self.model.model_validate_json(text)
        except ValidationError as error:
            return _describe(error), None, RefReport()
        checked, report = check_refs(notes, allowed)
        if report.too_many_invalid:
            return (f"{report.invalid} of {report.refs} references are not lines of the input. "
                    "Cite only the S-numbers given."), None, report
        return None, checked, report

    def _resume(self, raw: str | None, allowed) -> BaseModel | None:
        """A stored map answer, checked again (deterministic); None when there is none or it
        no longer passes (then the block is asked again)."""
        if raw is None:
            return None
        problem, notes, report = self._check(raw, allowed)
        if problem:
            return None
        self.reports["map"].add(report)
        return notes

    # --- sizes -----------------------------------------------------------------------------------

    def _tokens(self, chars: int) -> int:
        return math.ceil(chars / self.options.chars_per_token)

    def _estimate(self, request: LlmRequest) -> int:
        return self._tokens(len(request.system) + len(request.user))

    def _count(self, system: str, user: str) -> int:
        """Exact input tokens, counted by the provider (retried like a request)."""
        return call_with_retry(lambda: self.provider.count_tokens(system, user),
                               policy=self.options.retry_policy, sleep=self.options.sleep,
                               rng=self.options.rng)

    def _part(self, notes: PartNotes, raw: str, speech_ms: int) -> _Part:
        text = json.dumps(notes.notes, ensure_ascii=False, separators=(",", ":"))
        return _Part(notes, speech_ms, self._count("", text) + PART_OVERHEAD_TOKENS,
                     self._count("", raw), count_items(notes.notes))

    def _tokens_per_item(self, parts: Sequence[_Part]) -> float:
        tokens = sum(part.written for part in parts)
        items = sum(part.items for part in parts)
        value = tokens / items if items else 100.0
        self.reduce_stats["tokens_per_item"] = round(value, 1)
        return value

    def _expected_output(self, targets: Mapping[str, int], parts: Sequence[_Part]) -> int:
        """Answer tokens for these targets at the tokens per item measured on the parts, if the
        model writes TARGET_OVERSHOOT times as many items."""
        items = sum(targets.values())
        return math.ceil(items * self._tokens_per_item(parts) * TARGET_OVERSHOOT
                         + ANSWER_OVERHEAD_TOKENS)

    def _max_group_speech(self, parts: Sequence[_Part], cap: int, slack: float) -> int:
        """The most speech a partial reduce may cover with its expected answer under `cap`
        (its targets grow with the speech)."""
        low, high = 0, self.total_speech_ms
        while low < high:
            middle = (low + high + 1) // 2
            targets = self.template.partial_targets(middle, self.total_speech_ms, slack)
            if self._expected_output(targets, parts) <= cap:
                low = middle
            else:
                high = middle - 1
        return low

    def _reduce_budget(self) -> int:
        """Input-note tokens one reduce request may hold: what the context leaves after the
        counted system prompt and the longest answer a retry may write (`reduce_reserve`),
        never more than `reduce_max_input`."""
        if self._system_tokens is None:
            targets = self.template.targets(self.total_speech_ms, whole_lecture=True)
            system = reduce_prompt(self.template, [PartNotes(0, 0, {})], final=True,
                                   targets=targets, title=self.lecture.title,
                                   course=self.lecture.course,
                                   language=self.lecture.language).system
            self._system_tokens = self._count(system, "") + REQUEST_MARGIN_TOKENS
        room = (self.options.max_input_tokens(self.options.reduce_reserve())
                - self._system_tokens)
        budget = min(self.options.reduce_max_input, room)
        if budget <= 0:
            raise NotesError("The context is too small for the reduce prompt.",
                             remedy="Use a larger context (ollama.num_ctx).")
        return budget


def _where(stage: Stage, number: int) -> str:
    return {"map": f"block {number}", "partial": f"partial reduce {number}",
            "reduce": "the final reduce"}[stage]


def _describe(error: ValidationError, limit: int = 5) -> str:
    parts = []
    for item in error.errors()[:limit]:
        where = ".".join(str(p) for p in item["loc"]) or "answer"
        parts.append(f"{where}: {item['msg']}")
    more = error.error_count() - limit
    return "; ".join(parts) + (f"; and {more} more" if more > 0 else "")
