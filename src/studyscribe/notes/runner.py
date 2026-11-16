"""Generate a session's notes into the database: checks, lock, lifecycle, resume, call records.

Two steps, so the CLI can show the estimate and ask before anything is sent:

- `prepare` (no network): the session has a done transcript with at least 5 minutes of speech
  and a language, the template exists; the blocks and the estimate.
- `generate_session_notes`: the notes lock (byte 3 of session.lock), the open generation
  (resumed when its options match, else discarded), the pipeline, one `llm_calls` row per
  request (committed as it comes, failed ones too), and the end state. The provider is closed
  (Ollama: the model unloaded) in a `finally`.

An interruption (Ctrl+C, a provider error, a killed process) leaves the generation
'interrupted' with every call stored; the same command resumes it from the accepted map
blocks. `--restart` or different options discard it ('failed', with the reason) and start a new
one: every generation is kept.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from studyscribe.db import repo
from studyscribe.db.connection import transaction
from studyscribe.notes import pricing
from studyscribe.notes.blocks import Block, cut_blocks
from studyscribe.notes.estimate import Assumptions, Estimate, estimate_notes
from studyscribe.notes.pipeline import (
    CallRecord,
    Lecture,
    NotesError,
    NotesResult,
    PipelineOptions,
    generate_notes,
)
from studyscribe.notes.prompts import (
    PROMPT_VERSION,
    PartNotes,
    map_prompt,
    prompt_hash,
    reduce_prompt,
)
from studyscribe.notes.providers.base import Effort, LlmError, LlmProvider
from studyscribe.notes.providers.retry import Pacer
from studyscribe.notes.templates import (
    DEFAULT_TEMPLATE,
    Template,
    TemplateError,
    builtin_templates,
    parse_template,
)
from studyscribe.notes.transcript import Line, make_lines, speech_ms
from studyscribe.session import NOTES_LOCK_BYTE, session_lock

Emit = Callable[..., None]  # emit(kind, **info)
MIN_SPEECH_MS = 5 * 60_000
MIN_NUM_CTX = 32_768  # below it the reduce of a 3-hour lecture does not converge
CLOUD_MAX_INPUT = 90_000  # tokens per request for the cloud providers (plan: the 90k guard)
CLOUD_REDUCE_MAX_INPUT = 60_000
OLLAMA_REDUCE_MAX_INPUT = 16_000
GEMINI_FREE_RPM = 5
DEFAULT_EFFORT: dict[str, Effort] = {"ollama": "none", "gemini": "medium",
                                     "anthropic": "medium"}


class CannotGenerate(RuntimeError):
    """The notes cannot be generated now; the message says why and what to do. Nothing was
    sent and nothing was changed."""


@dataclass(frozen=True)
class NotesRequest:
    provider: str
    model: str
    effort: Effort
    template_id: str = DEFAULT_TEMPLATE
    language: str | None = None  # --notes-language; None: the transcript's
    restart: bool = False
    free_tier: bool = False  # Gemini free tier: cost 0, paced
    num_ctx: int | None = None  # Ollama's context window


@dataclass
class Prepared:
    session_id: str
    title: str
    course: str | None
    transcript_run_id: int
    lines: list[Line]
    blocks: list[Block]
    template: Template
    language: str
    language_source: str  # 'transcript' or 'forced'
    speech_ms: int
    estimate: Estimate
    request: NotesRequest

    @property
    def lecture(self) -> Lecture:
        return Lecture(self.title, self.course, self.language)


@dataclass
class Result:
    notes_id: int
    resumed_blocks: int
    result: NotesResult
    stats: dict[str, Any] = field(default_factory=dict)


# --- templates -----------------------------------------------------------------------------------


def sync_templates(conn: sqlite3.Connection) -> None:
    """Write the built-in templates into the `templates` table (they may change with the
    program); user templates are left alone."""
    with transaction(conn):
        for template in builtin_templates().values():
            repo.upsert_template(conn, template.id, json.dumps(template.title),
                                 template.description, template.to_json(), builtin=True)


def import_template(conn: sqlite3.Connection, template: Template) -> None:
    row = repo.get_template(conn, template.id)
    if row is not None and row["builtin"]:
        raise TemplateError(f"{template.id!r} is a built-in template: give yours another id.")
    with transaction(conn):
        repo.upsert_template(conn, template.id, json.dumps(template.title),
                             template.description, template.to_json(), builtin=False)


def load_template(conn: sqlite3.Connection, template_id: str) -> Template:
    sync_templates(conn)
    row = repo.get_template(conn, template_id)
    if row is None:
        known = ", ".join(r["id"] for r in repo.list_templates(conn))
        raise CannotGenerate(f"unknown template {template_id!r} (available: {known}).")
    return parse_template(row["template_json"], f"template {template_id}")


# --- checks --------------------------------------------------------------------------------------


def prepare(conn: sqlite3.Connection, session_id: str, request: NotesRequest,
            assumptions: Assumptions | None = None) -> Prepared:
    """Every check that needs no network, the blocks and the estimate. Raises CannotGenerate."""
    session = repo.get_session(conn, session_id)
    if session is None:
        raise CannotGenerate(f"no session {session_id} in this data directory.")
    run = repo.get_done_run(conn, session_id)
    if run is None:
        open_run = repo.get_open_run(conn, session_id)
        hint = (" (its transcription was interrupted: run the same `transcribe` command to "
                "finish it)" if open_run is not None else "")
        raise CannotGenerate(f"the session has no transcript yet{hint}. Run: studyscribe "
                             f"transcribe {session_id}")
    lines = make_lines(repo.list_transcript(conn, session_id))
    speech = speech_ms(lines)
    if speech < MIN_SPEECH_MS:
        raise CannotGenerate(f"the transcript has only {speech / 60_000:.1f} minutes of speech; "
                             f"notes need at least {MIN_SPEECH_MS // 60_000}.")
    if request.language:
        language, source = request.language, "forced"
    elif run["language"]:
        language, source = run["language"], "transcript"
    else:
        raise CannotGenerate("the transcript has no language (no speech was detected). Give "
                             "one with --notes-language, e.g. --notes-language it.")
    template = load_template(conn, request.template_id)
    blocks = cut_blocks(lines)
    course = None
    if session["course_id"] is not None:
        row = conn.execute("SELECT title FROM courses WHERE id = ?",
                           (session["course_id"],)).fetchone()
        course = row["title"] if row else None
    title = session["title"] or session_id
    lecture = Lecture(title, course, language)
    estimate = _estimate(template, blocks, lecture, request,
                         assumptions or _assumptions(conn, request))
    return Prepared(session_id, title, course, run["id"], lines, blocks, template, language,
                    source, speech, estimate, request)


def check_provider(provider: LlmProvider, request: NotesRequest) -> str:
    """The checks that need the provider: for Ollama the context size, then (every provider)
    the model check (Ollama running, model downloaded; cloud: key and model valid). Returns
    the model's description. Raises CannotGenerate or LlmError, both with a remedy."""
    if request.provider == "ollama":
        num_ctx = request.num_ctx or MIN_NUM_CTX
        if num_ctx < MIN_NUM_CTX:
            raise CannotGenerate(
                f"ollama.num_ctx is {num_ctx:,}: with less than {MIN_NUM_CTX:,} tokens of "
                "context the reduce of a long lecture does not converge (its input budget must "
                "stay well above the final answer of up to 8,192 tokens). Set it back with: "
                f"studyscribe config set ollama.num_ctx {MIN_NUM_CTX}")
    return provider.check()


def pipeline_options(provider: LlmProvider, request: NotesRequest) -> PipelineOptions:
    if request.provider == "ollama" and hasattr(provider, "budget"):
        return PipelineOptions(effort=request.effort, max_input_tokens=provider.budget,
                               reduce_max_input=OLLAMA_REDUCE_MAX_INPUT)
    pacer = Pacer(rpm=GEMINI_FREE_RPM) if request.free_tier else None
    return PipelineOptions(effort=request.effort, max_input_tokens=lambda cap: CLOUD_MAX_INPUT,
                           reduce_max_input=CLOUD_REDUCE_MAX_INPUT, pacer=pacer)


def run_options(prepared: Prepared) -> dict[str, Any]:
    """Everything that shapes a generation's output: it is resumed only if these match."""
    request = prepared.request
    return {
        "provider": request.provider, "model": request.model, "effort": request.effort,
        "num_ctx": request.num_ctx if request.provider == "ollama" else None,
        "template": prepared.template.to_json(), "language": prepared.language,
        "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(),
        "transcript_run_id": prepared.transcript_run_id,
    }


# --- generation ----------------------------------------------------------------------------------


def generate_session_notes(conn: sqlite3.Connection, data_dir: Path, prepared: Prepared,
                           provider: LlmProvider, *, emit: Emit = lambda kind, **info: None,
                           options: PipelineOptions | None = None) -> Result:
    """Generate (or resume) the notes; closes the provider at the end. Raises CannotGenerate
    (nothing done), LlmError or NotesError (the generation stays interrupted and resumable),
    KeyboardInterrupt (likewise)."""
    lock = session_lock(data_dir, prepared.session_id, NOTES_LOCK_BYTE)
    try:
        locked = lock.acquire()
    except OSError as exc:
        provider.close()
        raise CannotGenerate(f"cannot lock the session folder: {exc}") from exc
    if not locked:
        provider.close()
        raise CannotGenerate("another process is generating this session's notes.")
    notes_id = None
    counting = _CountingProvider(provider)
    try:
        notes_id, done_maps = _open_generation(conn, prepared, emit)
        started = time.perf_counter()

        def on_call(record: CallRecord) -> None:
            _store_call(conn, notes_id, record, prepared.request)
            emit("call", record=record)

        result = generate_notes(
            counting, prepared.template, prepared.lines, prepared.blocks, prepared.lecture,
            options=options or pipeline_options(provider, prepared.request),
            done_maps=done_maps, on_call=on_call,
            on_progress=lambda text: emit("progress", text=text))
        stats = dict(result.stats)
        stats["count_tokens"] = {"calls": counting.count_calls,
                                 "seconds": round(counting.count_seconds, 2)}
        stats["wall_s"] = round(time.perf_counter() - started, 1)
        stats.update(_call_totals(conn, notes_id))
        with transaction(conn):
            repo.finish_notes(
                conn, notes_id,
                content_json=json.dumps(result.resolved, ensure_ascii=False),
                uncertain_json=json.dumps([w.__dict__ for w in result.uncertain],
                                          ensure_ascii=False),
                stats_json=json.dumps(stats, ensure_ascii=False))
        return Result(notes_id, result.stats.get("resumed_blocks", 0), result, stats)
    except BaseException as exc:
        if notes_id is not None:
            if isinstance(exc, KeyboardInterrupt):
                error = "interrupted by the user"
            elif isinstance(exc, LlmError):
                error = exc.user_message()
            else:
                error = str(exc) or type(exc).__name__
            with transaction(conn):
                repo.set_notes_status(conn, notes_id, "interrupted", error)
        raise
    finally:
        provider.close()
        lock.release()


def _open_generation(conn: sqlite3.Connection, prepared: Prepared,
                     emit: Emit) -> tuple[int, dict[int, str]]:
    """(generation id, accepted map answers to reuse): the open generation when its options
    match and there is no --restart, else a new one (the old one is kept as 'failed')."""
    options_json = json.dumps(run_options(prepared), sort_keys=True)
    blocks_json = json.dumps([[b.first_n, b.last_n] for b in prepared.blocks])
    with transaction(conn):
        old = repo.get_open_notes(conn, prepared.session_id)
        if old is not None:
            same = (old["options_json"], old["blocks_json"]) == (options_json, blocks_json)
            if same and not prepared.request.restart:
                repo.set_notes_status(conn, old["id"], "running")
                done = repo.accepted_map_answers(conn, old["id"])
                emit("resumed", notes_id=old["id"], maps=len(done),
                     blocks=len(prepared.blocks))
                return old["id"], done
            reason = "--restart" if prepared.request.restart else "different options"
            repo.set_notes_status(conn, old["id"], "failed", f"discarded: {reason}")
            emit("discarded", notes_id=old["id"], reason=reason)
        notes_id = repo.create_notes(
            conn, session_id=prepared.session_id, transcript_run_id=prepared.transcript_run_id,
            provider=prepared.request.provider, model=prepared.request.model,
            template_id=prepared.template.id, template_json=prepared.template.to_json(),
            language=prepared.language, language_source=prepared.language_source,
            prompt_version=PROMPT_VERSION, prompt_hash=prompt_hash(),
            options_json=options_json, blocks_json=blocks_json)
    emit("started", notes_id=notes_id, blocks=len(prepared.blocks))
    return notes_id, {}


def _store_call(conn: sqlite3.Connection, notes_id: int, record: CallRecord,
                request: NotesRequest) -> None:
    usage = record.usage
    cost = list_cost = None
    if usage is not None:
        list_cost = pricing.list_cost(request.provider, request.model, usage)
        cost = pricing.charged_cost(request.provider, request.model, usage,
                                    free_tier=request.free_tier)
    timings = record.timings
    with transaction(conn):
        repo.add_llm_call(
            conn, notes_id, stage=record.stage, block=record.block, attempt=record.attempt,
            status=record.status, model_version=record.model_version,
            request_id=record.request_id,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            thinking_tokens=usage.thinking_tokens if usage else None,
            cached_tokens=usage.cached_tokens if usage else None,
            cost_usd=cost, list_cost_usd=list_cost, duration_s=record.duration_s,
            load_s=timings.load_s if timings else None,
            prefill_s=timings.prefill_s if timings else None,
            generate_s=timings.generate_s if timings else None,
            output_json=record.output_json, error=record.error)


def _call_totals(conn: sqlite3.Connection, notes_id: int) -> dict[str, Any]:
    """Time, tokens and cost of every call of the generation (this attempt and earlier ones),
    per stage."""
    totals: dict[str, Any] = {}
    for stage in ("map", "partial", "reduce"):
        rows = [r for r in repo.list_llm_calls(conn, notes_id) if r["stage"] == stage]
        totals[f"{stage}_calls"] = len(rows)
        totals[f"{stage}_s"] = round(sum(r["duration_s"] or 0 for r in rows), 1)
    rows = repo.list_llm_calls(conn, notes_id)
    totals["input_tokens"] = sum(r["input_tokens"] or 0 for r in rows)
    totals["output_tokens"] = sum(r["output_tokens"] or 0 for r in rows)
    totals["cost_usd"] = round(sum(r["cost_usd"] or 0 for r in rows), 6)
    totals["list_cost_usd"] = round(sum(r["list_cost_usd"] or 0 for r in rows), 6)
    return totals


class _CountingProvider:
    """The provider, counting how many token counts the pipeline asks for and their time."""

    def __init__(self, provider: LlmProvider) -> None:
        self._provider = provider
        self.name = provider.name
        self.model = provider.model
        self.count_calls = 0
        self.count_seconds = 0.0

    def generate(self, request):
        return self._provider.generate(request)

    def count_tokens(self, system: str, user: str) -> int:
        started = time.perf_counter()
        try:
            return self._provider.count_tokens(system, user)
        finally:
            self.count_calls += 1
            self.count_seconds += time.perf_counter() - started

    def check(self) -> str:
        return self._provider.check()

    def close(self) -> None:
        self._provider.close()


# --- estimate ------------------------------------------------------------------------------------


def _assumptions(conn: sqlite3.Connection, request: NotesRequest) -> Assumptions:
    """Default assumptions, with the speeds measured on this model's past calls when there
    are some (median)."""
    rows = repo.model_timings(conn, request.provider, request.model)
    base = Assumptions()
    generate = [r["output_tokens"] / r["generate_s"] for r in rows
                if r["output_tokens"] and r["generate_s"]]
    prefill = [(r["input_tokens"] - (r["cached_tokens"] or 0)) / r["prefill_s"] for r in rows
               if r["prefill_s"] and r["input_tokens"]]
    return Assumptions(
        generate_tps=statistics.median(generate) if len(generate) >= 3 else base.generate_tps,
        prefill_tps=statistics.median(prefill) if len(prefill) >= 3 else base.prefill_tps,
        thinking_tokens=base.thinking_tokens)


def _estimate(template: Template, blocks: list[Block], lecture: Lecture, request: NotesRequest,
              assumptions: Assumptions) -> Estimate:
    chars = []
    for block in blocks:
        prompt = map_prompt(template, block.lines, part=block.index, parts=len(blocks),
                            targets=template.targets(block.duration_ms, whole_lecture=False),
                            title=lecture.title, course=lecture.course,
                            language=lecture.language)
        chars.append(len(prompt.system) + len(prompt.user))
    system = reduce_prompt(template, [PartNotes(0, 0, {})], final=True,
                           targets=template.targets(0, whole_lecture=True), title=lecture.title,
                           course=lecture.course, language=lecture.language).system
    if request.provider == "ollama":
        cap = 16_384 if request.effort != "none" else 8_192
        room = (request.num_ctx or MIN_NUM_CTX) - cap - 512
        budget = min(OLLAMA_REDUCE_MAX_INPUT, room - int(len(system) / assumptions.chars_per_token))
    else:
        budget = CLOUD_REDUCE_MAX_INPUT
    return estimate_notes(template, [b.duration_ms for b in blocks],
                          [b.speech_ms for b in blocks], chars, len(system), max(budget, 1),
                          thinking=request.effort != "none", assumptions=assumptions)
