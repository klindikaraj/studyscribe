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
  allows it, otherwise the generation stops;
- transient provider errors are retried by retry.py; any other provider error stops the
  generation (the work done so far is in the calls already reported).

Nothing else is dropped: code items get a graded check (code_check.py) that only warns.
"""

from __future__ import annotations

import json
import math
import random
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
from studyscribe.notes.templates import Template
from studyscribe.notes.transcript import Line
from studyscribe.notes.uncertain import UncertainWord, uncertain_words

Stage = Literal["map", "partial", "reduce"]
DEFAULT_NUM_CTX = 32768
MARGIN_TOKENS = 512
PART_OVERHEAD_TOKENS = 20  # {"part":1,"from":"0:00:09","to":"0:12:41","notes":...}


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
    reduce_max_input: int = 16_000  # small models lose quality on long inputs
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
    output_json: str | None = None  # the raw answer
    error: str | None = None


@dataclass
class NotesResult:
    notes: dict[str, Any]  # final notes: S-numbers, code checks added
    resolved: dict[str, Any]  # the same with every reference as segment id + start time
    map_answers: dict[int, str]  # block index -> accepted raw answer (for resume)
    uncertain: list[UncertainWord]
    stats: dict[str, Any] = field(default_factory=dict)


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

    # --- stages ----------------------------------------------------------------------------------

    def generate(self, lines: Sequence[Line], blocks: Sequence[Block],
                 done_maps: dict[int, str]) -> NotesResult:
        parts: list[tuple[PartNotes, int]] = []  # (notes of a part, its speech in ms)
        map_answers: dict[int, str] = {}
        resumed = 0
        for block in blocks:
            allowed = range(block.first_n, block.last_n + 1)
            notes = self._resume(done_maps.get(block.index), allowed)
            if notes is not None:
                resumed += 1
                map_answers[block.index] = done_maps[block.index]
            else:
                self.progress(f"map {block.index}/{len(blocks)}")
                targets = self.template.targets(block.speech_ms, whole_lecture=False)
                prompt = map_prompt(self.template, block.lines, part=block.index,
                                    parts=len(blocks), targets=targets, title=self.lecture.title,
                                    course=self.lecture.course, language=self.lecture.language)
                notes, raw = self._ask("map", block.index, prompt, allowed)
                map_answers[block.index] = raw
            parts.append((PartNotes(block.start_ms, block.end_ms, notes.model_dump()),
                          block.speech_ms))
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
        }
        return NotesResult(notes=annotated, resolved=resolve(annotated, by_n),
                           map_answers=map_answers, uncertain=uncertain, stats=stats)

    def _reduce(self, parts: list[tuple[PartNotes, int]]) -> BaseModel:
        group_number = 0
        for _level in range(self.options.max_levels + 1):
            budget = self._reduce_budget()
            sizes = [self._size(part) for part, _ in parts]
            if fits(sizes, budget):
                break
            if _level == self.options.max_levels:
                raise NotesError(f"The notes still do not fit one request after "
                                 f"{self.options.max_levels} levels of partial reduces.")
            try:
                groups = plan_groups(sizes, budget)
            except ReduceTooLarge as error:
                raise NotesError(f"The notes of one part are too long to merge ({error}).",
                                 remedy="Use a model with a larger context.") from None
            if all(len(group) == 1 for group in groups):
                raise NotesError("The partial notes do not get shorter when merged: they "
                                 "cannot be reduced into one request.",
                                 remedy="Use a model with a larger context.")
            merged: list[tuple[PartNotes, int]] = []
            for group in groups:
                if len(group) == 1:
                    merged.append(parts[group[0]])
                    continue
                group_number += 1
                inputs = [parts[i][0] for i in group]
                speech = sum(parts[i][1] for i in group)
                self.progress(f"partial reduce {group_number}")
                notes = self._reduce_request("partial", group_number, inputs, speech)
                merged.append((PartNotes(inputs[0].start_ms, inputs[-1].end_ms,
                                         notes.model_dump()), speech))
            parts = merged
        self.progress("final reduce")
        return self._reduce_request("reduce", 1, [part for part, _ in parts],
                                    sum(speech for _, speech in parts))

    def _reduce_request(self, stage: Stage, number: int, inputs: list[PartNotes],
                        speech_ms: int) -> BaseModel:
        final = stage == "reduce"
        targets = self.template.targets(speech_ms, whole_lecture=final)
        prompt = reduce_prompt(self.template, inputs, final=final, targets=targets,
                               title=self.lecture.title, course=self.lecture.course,
                               language=self.lecture.language)
        notes, _ = self._ask(stage, number, prompt, refs_in(p.notes for p in inputs))
        return notes

    # --- one request -----------------------------------------------------------------------------

    def _ask(self, stage: Stage, number: int, prompt: Prompt, allowed) -> tuple[BaseModel, str]:
        """A validated, reference-checked answer and its raw text."""
        cap = self.options.output_cap(stage)
        attempt = 0
        problems = 0
        enlarged = False
        feedback = ""
        while True:
            request = LlmRequest(system=prompt.system, user=prompt.user + feedback,
                                 schema=self.schema, max_output_tokens=cap,
                                 effort=self.options.effort)

            def send(request: LlmRequest = request) -> LlmResponse:
                nonlocal attempt
                attempt += 1
                self.requests += 1
                return self.provider.generate(request)

            def failed(error: LlmError, _: int) -> None:
                self.on_call(CallRecord(stage, number, attempt, "error", error.usage, None,
                                        error=error.user_message()))

            try:
                response = call_with_retry(
                    send, policy=self.options.retry_policy, pacer=self.options.pacer,
                    sleep=self.options.sleep, rng=self.options.rng, on_error=failed,
                    on_wait=lambda reason, s: self.progress(f"waiting {s:.0f} s ({reason})"))
            except Truncated:
                larger = cap * 2
                if enlarged or self._estimate(request) > self.options.max_input_tokens(larger):
                    raise
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

    def _size(self, part: PartNotes) -> int:
        text = json.dumps(part.notes, ensure_ascii=False, separators=(",", ":"))
        return self._tokens(len(text)) + PART_OVERHEAD_TOKENS

    def _reduce_budget(self) -> int:
        """Input-note tokens one reduce request may hold: what the context leaves after the
        system prompt and the answer, and never more than `reduce_max_input`."""
        sample = PartNotes(0, 0, {})
        targets = self.template.targets(0, whole_lecture=True)
        system = reduce_prompt(self.template, [sample], final=True, targets=targets,
                               title=self.lecture.title, course=self.lecture.course,
                               language=self.lecture.language).system
        cap = self.options.output_cap("reduce")
        room = self.options.max_input_tokens(cap) - self._tokens(len(system))
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
