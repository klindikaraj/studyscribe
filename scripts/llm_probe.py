"""LLM probe for step 3: one tiny structured request, exact token counts of transcripts and, for
the local provider, one real map-like request on a block of a lecture.

Usage:
    .venv\\Scripts\\python scripts\\llm_probe.py                      (Ollama, qwen3.5:9b)
    .venv\\Scripts\\python scripts\\llm_probe.py --provider ollama --data-dir PATH
        --session ID [--session ID ...] [--block-from 0:30:00] [--block-minutes 15]
        [--model qwen3.5:9b] [--num-ctx 32768] [--effort none|low|medium|high] [--allow-cpu]
    .venv\\Scripts\\python scripts\\llm_probe.py --corrections --data-dir PATH --session ID
        [--effort none|high]                                 (Ollama: the correction pass)
    .venv\\Scripts\\python scripts\\llm_probe.py --provider gemini [--model gemini-3.8-flash]
        [--effort low|medium|high] [--paid]
    .venv\\Scripts\\python scripts\\llm_probe.py --provider gemini --count-only
        --data-dir PATH --session ID [--session ID ...]

Every provider: model check, then one request of a few hundred tokens asking for JSON with
three fields, validated with pydantic; tokens, time, cost. With --session: the input tokens of
each session's whole transcript in the pipeline's line format ("S<n> text", a [H:MM:SS] marker
every minute), counted by the provider (free). --count-only skips the generation requests.

Ollama, in addition: Ollama's version and the model's details; GPU memory (nvidia-smi) before
loading, after loading and after unloading; what Ollama allocated (`/api/ps`: size, size in
GPU memory, context); load time, prefill and generation speed; the tiny request sent twice to
see whether the fixed seed gives the same answer; with --session, a map request on one block
of the first session (--prompt v1-v4: the drafts in this file; v5: the pipeline's own prompt,
schema and targets for the Studio template), with the same measures for every prompt (items
per section, references outside the block, code items per level of the pipeline's code
check), and the answer saved to %TEMP%\\studyscribe-probe\\ in a file named after the
session, prompt, model and effort; at the end the model is unloaded and Ollama's log is
searched for ignored options or truncated prompts.

The database is opened read-only. Only text is sent (to this PC, for Ollama), never audio.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from studyscribe.config import db_path
from studyscribe.notes import gpu, keys, pricing
from studyscribe.notes.blocks import Block, cut_blocks
from studyscribe.notes.code_check import check_code, cited_texts, vocabulary
from studyscribe.notes.pipeline import coverage_gap
from studyscribe.notes.prompts import PROMPT_VERSION, map_prompt
from studyscribe.notes.providers import DEFAULT_MODELS, DEFAULT_PROVIDER, make_provider
from studyscribe.notes.providers.base import EFFORTS, PROVIDERS, LlmError, LlmRequest
from studyscribe.notes.providers.ollama import (
    DEFAULT_NUM_CTX,
    OllamaOptions,
    OllamaProvider,
    is_local,
)
from studyscribe.notes.providers.retry import call_with_retry
from studyscribe.notes.schema import output_model
from studyscribe.notes.templates import builtin_template
from studyscribe.notes.transcript import format_lines, make_lines
from studyscribe.transcribe.export import clock

PROBE_SYSTEM = (
    "You read a short passage from a lecture transcript and describe it. Use only the passage."
)
PROBE_TEXT = (
    "S1 So in PyTorch, the loss dot backward call computes the gradients for every tensor "
    "that has requires grad set to true.\n"
    "S2 Then the optimizer dot step updates the weights, and we call zero grad so the "
    "gradients don't pile up between iterations."
)
BLOCK_OUTPUT_TOKENS = 6144
OLLAMA_LOG = Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "server.log"
S_NUMBER = re.compile(r"\bS\d+\b")
STRING_LITERAL = re.compile(r"\"[^\"]*\"|'[^']*'|`[^`]*`")
PROSE = re.compile(r"[A-Za-z']+(?:\s+[A-Za-z']+){4,}")  # five plain words in a row
PROBE_DIR = Path(tempfile.gettempdir()) / "studyscribe-probe"
_SENTENCE = re.compile(r"[^.!?]+[.!?]")
LOG_MARKERS = ("invalid option", "truncating input", "out of memory")


class ProbeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(description="ISO 639-1 code of the passage's language")
    topic: str = Field(description="the topic in at most eight words")
    identifiers: list[str] = Field(description="code identifiers the speaker mentions, in code "
                                   "form; never the line labels S1, S2")


# --- draft map schema and prompt (piece 2 owns the real ones) -----------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Topic(_Strict):
    title: str
    ref: int = Field(description="S-number of the line where the topic starts")


class Concept(_Strict):
    name: str
    explanation: str
    refs: list[int]


class Definition(_Strict):
    term: str
    definition: str = Field(description="the lecturer's wording, fillers removed")
    refs: list[int]


class CodeExample(_Strict):
    code: str = Field(description="as it would appear in a source file; never an S-number")
    explanation: str
    refs: list[int]


class Correction(_Strict):
    heard: str
    written: str
    refs: list[int]


class BlockNotes(_Strict):
    topics: list[Topic]
    concepts: list[Concept]
    definitions: list[Definition]
    code: list[CodeExample]
    open_points: list[Concept]
    corrections: list[Correction]


BLOCK_SYSTEM = """You turn one part of a recorded lecture into study notes. The input is an
automatic transcript: lines "S<n> text" (speech recognition, it can contain errors), with
[H:MM:SS] markers. Lecture: "{title}". Part from {start} to {end}.

Rules
1. Use only what the lecturer says in this part. Never add facts, definitions, examples or code
   that are not in the transcript, even if you know them. If something is unclear, put it in
   open_points instead of guessing.
2. Every item cites in "refs" the S-numbers of the lines it comes from, the line where it
   starts first. Cite only S-numbers that appear in this part ({first} to {last}).
3. Write in {language}. Keep technical terms as the lecturer uses them.
4. Write code the lecturer reads aloud in code form ("torch dot nn" -> `torch.nn`). Fix a
   speech-recognition error only when the context makes the intended word certain, and list
   every fix in "corrections" (heard, written, refs).
5. Definitions: keep the lecturer's wording; remove only spoken fillers.

Answer with JSON that follows this schema:
{schema}"""

# v2 (2026-11-10), for small local models: explicit code and correction rules and a made-up
# example. The example deliberately avoids the acceptance test's cases ("TEC" -> detach,
# torch.nn, y_pred), so the test still measures the model and not the prompt.
BLOCK_SYSTEM_V2 = """You turn one part of a recorded lecture into study notes. The input is an
automatic transcript: lines "S<n> text" (speech recognition, it can contain errors), with
[H:MM:SS] markers. Lecture: "{title}". Part from {start} to {end}.

Rules
1. Use only what the lecturer says in this part. Never add facts, definitions, examples or code
   that are not in the transcript, even if you know them. If something is unclear, put it in
   open_points instead of guessing.
2. "ref" and "refs" hold the S-numbers of the lines an item comes from, the line where it
   starts first. Cite only S-numbers of this part ({first} to {last}). S-numbers go only in
   "ref" and "refs", never inside names, explanations or code.
3. Write in {language}. Keep technical terms as the lecturer uses them.
4. Code. Lecturers read code aloud and name code elements: modules, classes, functions,
   methods, variables, operators, expressions, statements. Every one of them goes into "code",
   even a single method name, written as it would appear in a source file:
   - spoken punctuation becomes symbols: "dot" or "punto" -> ".", "equals" or "uguale" -> "=",
     "plus equals" -> "+=";
   - a name spoken as several words is joined the way the programming language being taught
     does it (camelCase in JavaScript, snake_case in Python), unless the lecturer spells it;
   - text values get quotes, calls get parentheses with the arguments the lecturer gives;
   - "explanation" says in one sentence what the code does in the lecture.
5. Corrections. Speech recognition often turns a technical word into a similar-sounding wrong
   word, a short nonsense word or a wrong expansion. When the context makes the intended word
   certain (for example a method being called on an object), write the right word in the notes
   and add an entry to "corrections": what was heard, what you wrote, the refs. When you are
   not certain, keep the transcript's word and add an open point.
6. Definitions: keep the lecturer's wording; remove only spoken fillers.

Example of a made-up lecture, only to show the format. Never copy its content.
Transcript:
S901 Now we call scores dot bush with the new score, to add it to the array.
S902 And then total score plus equals new score.
Fragments of the answer:
"code": [{{"code": "scores.push(newScore)", "explanation": "adds the new score to the array",
"refs": [901]}}, {{"code": "totalScore += newScore", "explanation": "adds the new score to the
running total", "refs": [902]}}],
"corrections": [{{"heard": "bush", "written": "push", "refs": [901]}}]

Answer with JSON that follows this schema:
{schema}"""

# v3 (2026-11-10): v2 plus targets per section, so code items do not crowd out concepts and
# definitions, and a rule against invented names in code (v2 wrote `y_prediatti` for "y
# previsti", `misure_notte` for "the two tensors", pseudo-code for a formula said in words).
BLOCK_SYSTEM_V3 = """You turn one part of a recorded lecture into study notes. The input is an
automatic transcript: lines "S<n> text" (speech recognition, it can contain errors), with
[H:MM:SS] markers. Lecture: "{title}". Part from {start} to {end}.

Rules
1. Use only what the lecturer says in this part. Never add facts, definitions, examples or code
   that are not in the transcript, even if you know them. If something is unclear, put it in
   open_points instead of guessing.
2. "ref" and "refs" hold the S-numbers of the lines an item comes from, the line where it
   starts first. Cite only S-numbers of this part ({first} to {last}). S-numbers go only in
   "ref" and "refs", never inside names, explanations or code.
3. Write in {language}. Keep technical terms as the lecturer uses them.
4. Every section is filled on its own: code items never replace concepts or definitions. For
   this part aim for about {topics} topics, {concepts} concepts and {definitions} definitions;
   fewer only when the lecturer really says less. Concepts explain ideas in a few sentences;
   definitions are the terms the lecturer defines.
5. Code. Lecturers read code aloud and name code elements: modules, classes, functions,
   methods, variables, operators, expressions, statements. Every one of them goes into "code",
   even a single method name, written as it would appear in a source file:
   - spoken punctuation becomes symbols: "dot" or "punto" -> ".", "equals" or "uguale" -> "=",
     "plus equals" -> "+=";
   - a name spoken as several words is joined the way the programming language being taught
     does it (camelCase in JavaScript, snake_case in Python), unless the lecturer spells it;
   - text values get quotes, calls get parentheses with the arguments the lecturer gives;
   - use only names that are in the transcript, or their evident code form (words joined, or
     a misheard word fixed and listed in "corrections"). Never invent a variable name, an
     alias, an argument or a value. Code shown on screen but not said is not in the
     transcript: do not rebuild it. If a name is not clear, leave the item out or add an
     open point;
   - a sentence that describes an idea or a formula in words is not code: it belongs in
     concepts;
   - "explanation" says in one sentence what the code does in the lecture.
6. Corrections. Speech recognition often turns a technical word into a similar-sounding wrong
   word, a short nonsense word or a wrong expansion. When the context makes the intended word
   certain (for example a method being called on an object), write the right word in the notes
   and add an entry to "corrections": what was heard, what you wrote, the refs. When you are
   not certain, keep the transcript's word and add an open point.
7. Definitions: keep the lecturer's wording; remove only spoken fillers.

Example of a made-up lecture, only to show the format. Never copy its content.
Transcript:
S901 Now we call scores dot bush with the new score, to add it to the array.
S902 And then total score plus equals new score.
Fragments of the answer:
"code": [{{"code": "scores.push(newScore)", "explanation": "adds the new score to the array",
"refs": [901]}}, {{"code": "totalScore += newScore", "explanation": "adds the new score to the
running total", "refs": [902]}}],
"corrections": [{{"heard": "bush", "written": "push", "refs": [901]}}]

Answer with JSON that follows this schema:
{schema}"""

# v4 (2026-11-10): v3 without the made-up example. With v3 the 9B model copied the example's
# correction ("bush" -> "push", cited on an unrelated line) and invented another one.
_EXAMPLE_START = "Example of a made-up lecture"
BLOCK_SYSTEM_V4 = (BLOCK_SYSTEM_V3[:BLOCK_SYSTEM_V3.index(_EXAMPLE_START)]
                   + BLOCK_SYSTEM_V3[BLOCK_SYSTEM_V3.index("Answer with JSON"):])

PROMPTS = {"v1": BLOCK_SYSTEM, "v2": BLOCK_SYSTEM_V2, "v3": BLOCK_SYSTEM_V3,
           "v4": BLOCK_SYSTEM_V4}
V5 = PROMPT_VERSION  # no string here: the pipeline's own map prompt (v5, v5.1, ...)


# --- draft correction pass (2026-11-16) ----------------------------------------------------------
# One request per lecture, before the map: only the words Whisper was unsure about, each with a
# few lines of context, and the model decides "keep" or "fix" for every one of them. Accepted
# fixes would become the map's glossary. The candidates are chosen by code, not by the model.

LOW_PROBABILITY = 0.4  # Whisper's word probability below which a word is a candidate
MIN_LETTERS = 3  # shorter words are mostly function words ("the", "di")
RARE_MAX_COUNT = 2  # a word said more often in the lecture is taken as known
CONTEXT_LINES = 2  # transcript lines shown on each side of a candidate line
CORRECTION_OUTPUT_TOKENS = 4096
CORRECTION_MAX_WORDS = 3  # "written" replaces one word: a longer text is a rewrite
WORD = re.compile(r"\w+")


class WordVerdict(_Strict):
    ref: int = Field(description="S-number of the line with the marked word")
    heard: str = Field(description="the marked word, exactly as between [[ and ]]")
    reason: str = Field(description="one short sentence: why it fits, or what was said instead")
    decision: Literal["keep", "fix"]
    written: str = Field(description="the word for the notes: the marked word itself for "
                         "\"keep\"; for \"fix\" the intended word, code in code form")


class CorrectionPass(_Strict):
    words: list[WordVerdict]


CORRECTION_SYSTEM = """You check the words a speech recogniser was unsure about in the automatic
transcript of a lecture. Lecture: "{title}". Language of the lecture: {language}.

The input holds excerpts of the transcript, lines "S<n> text". In some lines one or more words
are marked like [[this]]: the recogniser gave them a low confidence. Decide for every marked
word:
- "keep" when the word fits the sentence and the topic. Low confidence alone is not an error:
  most marked words are right. Also "keep" when the word looks wrong but you are not certain
  what was said.
- "fix" only when the word does not fit and the context makes the intended word certain: a term
  the lecturer uses in the nearby lines, a code element (keyword, function, method, variable)
  that the sentence needs, or a word that sounds alike and is the only one that makes sense.
  "written" is then the intended word as it should appear in study notes, code in code form.
Never translate, never rewrite the sentence, never change a word that is not marked. "written"
replaces only the marked word: at most three words.
Answer with one item per marked word, in the order of the lines, as JSON that follows this
schema:
{schema}"""
# Items per hour of speech for the v3 targets (the Studio template's rates, scaled to a block).
TARGETS_PER_HOUR = {"topics": 16, "concepts": 24, "definitions": 16}
EXAMPLE_TEXTS = ("scores.push", "totalScore", "newScore", "bush")  # copied from the example?
THINKING_OUTPUT_TOKENS = 16384
LANGUAGE_NAMES = {"en": "English", "it": "Italian"}


# --- main ----------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=PROVIDERS, default=DEFAULT_PROVIDER)
    parser.add_argument("--model", help="default: the provider's default model")
    parser.add_argument("--effort", choices=EFFORTS,
                        help="default: none for ollama (thinking off), low for the others")
    parser.add_argument("--paid", action="store_true",
                        help="Gemini paid tier (only changes the cost shown)")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--session", action="append", default=[], metavar="ID")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="ollama only")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="ollama only: go on when the model is partly on the CPU")
    parser.add_argument("--block-from", default="0:30:00", metavar="H:MM:SS",
                        help="ollama only: start of the block of the first session")
    parser.add_argument("--block-minutes", type=float, default=15.0)
    parser.add_argument("--prompt", choices=[*sorted(PROMPTS), V5], default=V5,
                        help="ollama only: draft map prompt of the block test")
    parser.add_argument("--expect", action="append", default=[], metavar="WORD",
                        help="ollama only: report where WORD appears in the block's notes")
    parser.add_argument("--seed", type=int, default=None,
                        help="ollama only: sampling seed (default: the provider's); another seed "
                        "shows how much the answers vary")
    parser.add_argument("--corrections", action="store_true",
                        help="ollama only: the correction pass on every --session (one request "
                        "each) instead of the tiny request and the block test")
    args = parser.parse_args()
    if args.session and args.data_dir is None:
        parser.error("--session needs --data-dir")
    args.model = args.model or DEFAULT_MODELS[args.provider]
    args.effort = args.effort or ("none" if args.provider == "ollama" else "low")
    try:
        if args.provider == "ollama":
            return run_ollama(args)
        return run_cloud(args)
    except LlmError as exc:
        print(f"\nError: {exc.user_message()}")
        return 1


def run_cloud(args: argparse.Namespace) -> int:
    free_tier = args.provider == "gemini" and not args.paid
    found = keys.require_key(args.provider)
    print(f"Provider {args.provider}, model {args.model}, key {keys.mask(found.key)} "
          f"from {found.source}")
    if free_tier:
        print("Gemini free tier: Google may use this text to improve its products.")
    provider = make_provider(args.provider, args.model, found.key)
    print(f"Model check (free): {provider.check()}")
    if not args.count_only:
        probe_request(provider, args.effort, free_tier)
    for session_id in args.session:
        count_session(provider, args.data_dir, session_id)
    return 0


def run_ollama(args: argparse.Namespace) -> int:
    options = OllamaOptions(num_ctx=args.num_ctx, allow_cpu=args.allow_cpu)
    if args.seed is not None:
        options = dataclasses.replace(options, seed=args.seed)
    provider = OllamaProvider(args.model, options, on_progress=_progress)
    where = "this PC" if is_local(options.host) else "ANOTHER MACHINE: the text leaves this PC"
    print(f"Provider ollama at {options.host} ({where}), model {args.model}, "
          f"context {args.num_ctx:,} tokens, effort {args.effort}")
    log_start = OLLAMA_LOG.stat().st_size if OLLAMA_LOG.exists() else None
    print(f"GPU before loading: {_gpu_text(gpu.query())}")
    print(f"Model check: {provider.check()}")
    info = provider.model_info()
    kv = info.kv
    print(f"  family {info.family}, capabilities {', '.join(info.capabilities)}, thinking "
          f"values {info.thinking_values}, max context {info.context_length}")
    if kv:
        window = (f" + {kv.window_bytes / 2**20:.0f} MiB of sliding windows"
                  if kv.window_bytes else "")
        print(f"  KV cache (f16): {kv.per_token / 1024:.0f} KiB/token{window} = "
              f"{kv.bytes_for(args.num_ctx) / 2**30:.2f} GiB at {args.num_ctx:,} tokens")
    try:
        load = provider.ensure_loaded()
        print(f"Loaded in {load.load_s:.1f} s: Ollama allocated {load.size_bytes / 1e9:.2f} GB, "
              f"{load.vram_bytes / 1e9:.2f} GB in GPU memory "
              f"({'entirely on the GPU' if load.on_gpu_entirely else 'PARTLY ON THE CPU'}), "
              f"context {load.context_length:,}")
        print(f"GPU after loading: {_gpu_text(load.gpu_after)}")
        if args.corrections:
            for session_id in args.session:
                correction_probe(provider, args, session_id)
            return 0
        if not args.count_only:
            first = probe_request(provider, args.effort, free_tier=False)
            second = probe_request(provider, args.effort, free_tier=False, quiet=True)
            same = first is not None and first == second
            print(f"Same request again with seed {options.seed}: "
                  f"{'identical answer' if same else 'DIFFERENT answer'}"
                  + ("" if same else f"\n  1st: {first!r}\n  2nd: {second!r}"))
        for session_id in args.session:
            count_session(provider, args.data_dir, session_id)
        if args.session and not args.count_only:
            block_probe(provider, args, args.session[0])
    finally:
        provider.close()
        try:
            still = [m.get("name") for m in provider.running_models()]
            print(f"\nUnloaded: {'yes' if not still else 'NO, still loaded: ' + ', '.join(still)}")
        except LlmError as exc:
            print(f"\nCould not check the unload: {exc.message}")
        print(f"GPU after unloading: {_gpu_text(gpu.query())}")
        _print_log_warnings(log_start)
    return 0


# --- requests ------------------------------------------------------------------------------------


def probe_request(provider, effort: str, free_tier: bool, quiet: bool = False) -> str | None:
    request = LlmRequest(system=PROBE_SYSTEM, user=PROBE_TEXT,
                         schema=ProbeAnswer.model_json_schema(),
                         max_output_tokens=_output_cap(provider, effort, 4000), effort=effort)
    response = call_with_retry(
        lambda: provider.generate(request),
        on_wait=lambda reason, s: print(f"  waiting {s:.0f} s ({reason})"),
    )
    _end_progress()
    if quiet:
        return response.text
    usage = response.usage
    print(f"Answer in {response.duration_s:.1f} s, model version {response.model_version}, "
          f"request {response.request_id}")
    print(f"Tokens: input {usage.input_tokens} (cached {usage.cached_tokens}), output "
          f"{usage.output_tokens} (thinking {usage.thinking_tokens})")
    _print_timings(response)
    list_cost = pricing.list_cost(provider.name, provider.model, usage)
    charged = pricing.charged_cost(provider.name, provider.model, usage, free_tier=free_tier)
    print(f"Cost: {_usd(charged)} charged, {_usd(list_cost)} at list price "
          f"(prices checked {pricing.PRICES_VERIFIED})")
    try:
        answer = ProbeAnswer.model_validate_json(response.text)
        print(f"Valid JSON: {answer.model_dump()}")
    except ValidationError as exc:
        print(f"INVALID JSON ({exc.error_count()} errors): {response.text[:500]!r}")
    return response.text


def block_probe(provider: OllamaProvider, args: argparse.Namespace, session_id: str) -> None:
    rows, language, title = load_transcript(args.data_dir, session_id)
    if rows is None:
        return
    lines = make_lines(rows)
    by_n = {line.n: line for line in lines}
    start_ms = _parse_clock(args.block_from)
    end_ms = start_ms + int(args.block_minutes * 60_000)
    block = [line for line in lines if start_ms <= line.start_ms < end_ms]
    if not block:
        print(f"\nBlock {clock(start_ms)}-{clock(end_ms)}: no segments there")
        return
    first, last = block[0].n, block[-1].n
    cap = _output_cap(provider, args.effort, BLOCK_OUTPUT_TOKENS)
    if args.prompt == V5:
        # The pipeline's own prompt, schema and targets (piece 2), on the same lines as v1-v4.
        template = builtin_template()
        blocks = cut_blocks(lines)
        part = next((b.index for b in blocks if b.start_ms <= block[0].start_ms <= b.end_ms), 1)
        prompt = map_prompt(template, block, part=part, parts=len(blocks),
                            targets=template.targets(block[-1].end_ms - block[0].start_ms,
                                                     whole_lecture=False),
                            title=title, course=None, language=language or "en")
        system, user = prompt.system, prompt.user
        model, code_key = output_model(template), "examples"
    else:
        user = format_lines(block)
        system = PROMPTS[args.prompt].format(
            title=title, start=clock(start_ms), end=clock(end_ms), first=f"S{first}",
            last=f"S{last}",
            language=LANGUAGE_NAMES.get(language or "", language or "the transcript's language"),
            schema=json.dumps(BlockNotes.model_json_schema(), separators=(",", ":")),
            **{key: max(1, round(rate * args.block_minutes / 60))
               for key, rate in TARGETS_PER_HOUR.items()})
        model, code_key = BlockNotes, "code"
    print(f"\nBlock {clock(start_ms)}-{clock(end_ms)} of {session_id}: segments S{first}-S{last}, "
          f"{len(system) + len(user):,} characters of prompt (prompt {args.prompt}), "
          f"output cap {cap:,} tokens")
    request = LlmRequest(system=system, user=user, schema=model.model_json_schema(),
                         max_output_tokens=cap, effort=args.effort)
    response = call_with_retry(
        lambda: provider.generate(request),
        on_wait=lambda reason, s: print(f"  waiting {s:.0f} s ({reason})"),
    )
    _end_progress()
    usage = response.usage
    chars = len(system) + len(user)
    rate = response.timings.generate_rate(usage.output_tokens) if response.timings else None
    print(f"Answer in {response.duration_s:.1f} s: input {usage.input_tokens:,} tokens "
          f"({chars / usage.input_tokens:.2f} chars/token), output {usage.output_tokens:,} tokens"
          f" (thinking {usage.thinking_tokens})" + (f", {rate:.1f} tokens/s" if rate else ""))
    _print_timings(response)
    out = _probe_file("block", session_id, args)
    out.write_text(response.text, encoding="utf-8")
    try:
        notes = model.model_validate_json(response.text)
    except ValidationError as exc:
        print(f"Valid JSON at the first attempt: NO ({exc.error_count()} errors), saved to {out}")
        for error in exc.errors()[:5]:
            print(f"  {'.'.join(str(p) for p in error['loc'])}: {error['msg']}")
        return
    print(f"Valid JSON at the first attempt: yes, saved to {out}")
    data = notes.model_dump()
    report_block(data, first, last, by_n, code_key, args, response.text)
    gap = coverage_gap(data, Block(1, tuple(block)))
    print("  coverage: " + (f"WARNING, {gap.describe()}" if gap
                            else "ok (cites past 85% of the block)"))


def report_block(data: dict, first: int, last: int, by_n: dict, code_key: str,
                 args: argparse.Namespace, text: str) -> None:
    """The same measures for every prompt: items per section, references outside the block and
    the items they would drop, code items per level of the pipeline's code check."""
    counts = {key: len(value) if isinstance(value, list) else len(_SENTENCE.findall(value))
              for key, value in data.items()}
    print("  items: " + ", ".join(f"{key} {n}" for key, n in counts.items())
          + "  (summary: sentences)")
    refs = outside = 0
    dropped: dict[str, int] = {}
    for key, value in data.items():
        if not isinstance(value, list):
            continue
        for item in value:
            item_refs = [item["ref"]] if "ref" in item else list(item.get("refs", []))
            bad = [r for r in item_refs if not first <= r <= last]
            refs += len(item_refs)
            outside += len(bad)
            if len(bad) == len(item_refs):
                dropped[key] = dropped.get(key, 0) + 1
    print(f"  references: {refs}, outside the block: {outside}; items the pipeline would drop: "
          + (", ".join(f"{k} {n}" for k, n in dropped.items()) if dropped else "none"))
    for topic in data.get("topics", data.get("outline", [])):
        at = clock(by_n[topic["ref"]].start_ms) if topic["ref"] in by_n else "?"
        print(f"  [{at}] S{topic['ref']} {topic['title']}")
    vocab = vocabulary(line.text for line in by_n.values())
    levels: Counter[str] = Counter()
    prose_items = 0
    for item in data.get(code_key, []):
        if not item["code"].strip():
            print(f"  example without code: {item['explanation'][:70]!r} {item['refs']}")
            continue
        result = check_code(item["code"], cited_texts(item["refs"], by_n), vocab)
        levels[result.level] += 1
        prose = bool(PROSE.search(STRING_LITERAL.sub("", item["code"])))
        prose_items += prose
        flags = [f"missing {', '.join(result.missing)}" if result.missing else "",
                 f"partial {', '.join(result.partial)}" if result.partial else "",
                 "PROSE" if prose else ""]
        print(f"  code {result.level:11} {item['code']!r} {item['refs']}  "
              + "  ".join(f for f in flags if f))
    print(f"  code items: ok {levels['ok']}, check {levels['check']}, "
          f"unsupported {levels['unsupported']}; prose written as code: {prose_items}")
    with_s = [c["code"] for c in data.get(code_key, []) if S_NUMBER.search(c["code"])]
    print(f"  code with an S-number inside: {len(with_s)}" + (f" {with_s[:5]}" if with_s else ""))
    for fix in data.get("corrections", []):
        print(f"  correction: {fix['heard']!r} -> {fix['written']!r} {fix['refs']}")
    if args.prompt != "v1":
        copied = [t for t in EXAMPLE_TEXTS if t in text]
        print(f"  copied from the prompt's example: {copied if copied else 'nothing'}")
    for word in args.expect:
        where = sorted({section for section, items in data.items()
                        if word.lower() in json.dumps(items, ensure_ascii=False).lower()})
        print(f"  {word!r}: " + (f"in {', '.join(where)}" if where else "NOT FOUND"))


def _probe_file(kind: str, session_id: str, args: argparse.Namespace) -> Path:
    """One file per session, prompt, model and effort, so runs never overwrite each other."""
    PROBE_DIR.mkdir(exist_ok=True)
    model = re.sub(r"[^\w.-]", "-", args.model)
    seed = f"-seed{args.seed}" if args.seed is not None else ""
    return PROBE_DIR / f"{kind}-{session_id}-{args.prompt}-{model}-{args.effort}{seed}.json"


def correction_probe(provider: OllamaProvider, args: argparse.Namespace, session_id: str) -> None:
    rows, language, title = load_transcript(args.data_dir, session_id)
    if rows is None:
        return
    suspects = suspect_words(rows)
    if not suspects:
        print(f"\n{session_id}: no candidate words, nothing to send")
        return
    user = correction_excerpts(rows, suspects)
    cap = _output_cap(provider, args.effort, CORRECTION_OUTPUT_TOKENS)
    system = CORRECTION_SYSTEM.format(
        title=title,
        language=LANGUAGE_NAMES.get(language or "", language or "unknown"),
        schema=json.dumps(CorrectionPass.model_json_schema(), separators=(",", ":")))
    print(f"\nCorrection pass on {session_id} ({language}): {len(suspects)} lines with "
          f"{sum(map(len, suspects.values()))} candidate words (probability < {LOW_PROBABILITY}, "
          f"{MIN_LETTERS}+ letters, at most {RARE_MAX_COUNT} times in the lecture), "
          f"{len(system) + len(user):,} characters of prompt, output cap {cap:,} tokens")
    for n, words in suspects.items():
        print(f"  S{n} [{clock(rows[n - 1]['start_ms'])}] {', '.join(words)}")
    request = LlmRequest(system=system, user=user, schema=CorrectionPass.model_json_schema(),
                         max_output_tokens=cap, effort=args.effort)
    response = call_with_retry(
        lambda: provider.generate(request),
        on_wait=lambda reason, s: print(f"  waiting {s:.0f} s ({reason})"),
    )
    _end_progress()
    usage = response.usage
    print(f"Answer in {response.duration_s:.1f} s: input {usage.input_tokens:,} tokens, output "
          f"{usage.output_tokens:,} tokens (thinking {usage.thinking_tokens})")
    _print_timings(response)
    out = _probe_file("corrections", session_id, args)
    out.write_text(response.text, encoding="utf-8")
    try:
        answer = CorrectionPass.model_validate_json(response.text)
    except ValidationError as exc:
        print(f"INVALID JSON ({exc.error_count()} errors), saved to {out}")
        return
    print(f"Valid JSON, saved to {out}")
    answered = set()
    accepted = []
    for item in answer.words:
        problem = check_verdict(item, suspects)
        key = (item.ref, _norm(item.heard))
        if not problem and key in answered:
            problem = "answered twice"
        answered.add(key)
        if item.decision == "fix" and not problem:
            accepted.append(item)
        status = "" if not problem else f"  <- REJECTED: {problem}"
        print(f"  S{item.ref} {item.heard!r}: {item.decision.upper()}"
              + (f" -> {item.written!r}" if item.decision == "fix" else "")
              + f"{status}\n      reason: {item.reason}")
    missing = [f"S{n} {w}" for n, words in suspects.items() for w in words
               if (n, _norm(w)) not in answered]
    fixes = sum(item.decision == "fix" for item in answer.words)
    print(f"  {len(answer.words)} answers, {fixes} fixes proposed, {len(accepted)} pass the "
          f"checks: " + (", ".join(f"{i.heard!r} -> {i.written!r} (S{i.ref})" for i in accepted)
                         or "none"))
    print(f"  candidates without an answer (kept): {', '.join(missing) if missing else 'none'}")


def suspect_words(rows) -> dict[int, list[str]]:
    """S-number -> the words of that line Whisper was unsure about: probability below
    LOW_PROBABILITY, at least MIN_LETTERS letters, said at most RARE_MAX_COUNT times in the
    lecture (counted on Whisper's words, punctuation removed, case ignored)."""
    words = [json.loads(row["words_json"] or "[]") for row in rows]
    counts = Counter(_norm(w[2]) for line in words for w in line)
    found: dict[int, list[str]] = {}
    for n, line in enumerate(words, start=1):
        for word in line:
            core = _norm(word[2])
            if (word[3] < LOW_PROBABILITY and sum(c.isalpha() for c in core) >= MIN_LETTERS
                    and counts[core] <= RARE_MAX_COUNT):
                found.setdefault(n, []).append(word[2].strip(" .,;:!?\"'"))
    return found


def correction_excerpts(rows, suspects: dict[int, list[str]]) -> str:
    """Excerpts of CONTEXT_LINES lines around every candidate line (overlapping excerpts are
    joined), with the candidate words marked as [[word]]. Lines are rebuilt from Whisper's words,
    which join to exactly the segment's text."""
    spans: list[list[int]] = []
    for n in sorted(suspects):
        lo, hi = max(1, n - CONTEXT_LINES), min(len(rows), n + CONTEXT_LINES)
        if spans and lo <= spans[-1][1] + 1:
            spans[-1][1] = hi
        else:
            spans.append([lo, hi])
    parts = []
    for i, (lo, hi) in enumerate(spans, start=1):
        lines = [f"--- excerpt {i}"]
        for n in range(lo, hi + 1):
            marked = set(suspects.get(n, []))
            text = "".join(
                w[2].replace(core, f"[[{core}]]", 1) if (core := w[2].strip(" .,;:!?\"'"))
                in marked and w[3] < LOW_PROBABILITY else w[2]
                for w in json.loads(rows[n - 1]["words_json"] or "[]"))
            lines.append(f"S{n} {text.strip()}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def check_verdict(item: WordVerdict, suspects: dict[int, list[str]]) -> str | None:
    """Why a verdict cannot be used, or None. The checks need no knowledge of the lecture."""
    if item.ref not in suspects:
        return f"S{item.ref} has no marked word"
    if _norm(item.heard) not in {_norm(w) for w in suspects[item.ref]}:
        return "the heard word is not the marked word of that line"
    if item.decision == "keep":
        return None
    if _norm(item.written) == _norm(item.heard):
        return "the written form is the heard word"
    if len(WORD.findall(item.written)) > CORRECTION_MAX_WORDS or len(item.written) > 40:
        return "a rewrite, not a word"
    return None


def _norm(text: str) -> str:
    return "".join(WORD.findall(text)).lower()


# --- transcripts ---------------------------------------------------------------------------------


def load_transcript(data_dir: Path, session_id: str):
    uri = db_path(data_dir).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        run = conn.execute(
            "SELECT id, language FROM transcription_runs WHERE session_id = ? AND pass = 'final'"
            " AND status = 'done'", (session_id,)).fetchone()
        session = conn.execute("SELECT title FROM sessions WHERE id = ?",
                               (session_id,)).fetchone()
        if run is None:
            print(f"{session_id}: no transcript")
            return None, None, None
        rows = conn.execute(
            "SELECT id, start_ms, end_ms, text, words_json FROM transcript_segments"
            " WHERE run_id = ?"
            " AND dropped IS NULL ORDER BY start_ms, id", (run["id"],)).fetchall()
    finally:
        conn.close()
    title = (session["title"] if session is not None else None) or session_id
    return rows, run["language"], title


def count_session(provider, data_dir: Path, session_id: str) -> None:
    rows, language, _ = load_transcript(data_dir, session_id)
    if rows is None:
        return
    text = format_lines(make_lines(rows))
    tokens = provider.count_tokens(PROBE_SYSTEM, text)
    _end_progress()
    print(f"{session_id} ({language}): {len(rows)} segments, {len(text):,} characters, "
          f"{tokens:,} input tokens with {provider.model} ({len(text) / tokens:.2f} chars/token)")


# --- output helpers ------------------------------------------------------------------------------


def _progress(chunks: int, seconds: float) -> None:
    if chunks % 25 == 0:
        rate = f", {chunks / seconds:.0f}/s overall" if seconds > 0 else ""
        print(f"\r  receiving: {chunks} chunks in {seconds:.0f} s{rate}   ", end="", flush=True)


def _end_progress() -> None:
    print("\r" + " " * 60 + "\r", end="", flush=True)


def _print_timings(response) -> None:
    t = response.timings
    if t is None:
        return
    usage = response.usage
    prefill = t.prefill_rate(usage.input_tokens - usage.cached_tokens)
    generate = t.generate_rate(usage.output_tokens)
    print(f"Timings: load {t.load_s:.1f} s, prefill {t.prefill_s:.1f} s"
          + (f" ({prefill:,.0f} tokens/s)" if prefill else "")
          + f", generation {t.generate_s:.1f} s"
          + (f" ({generate:.1f} tokens/s)" if generate else ""))


def _gpu_text(memory: gpu.GpuMemory | None) -> str:
    if memory is None:
        return "unknown (nvidia-smi not available)"
    return f"{memory.used_mib:,} of {memory.total_mib:,} MiB in use ({memory.name})"


def _print_log_warnings(start: int | None) -> None:
    if start is None or not OLLAMA_LOG.exists():
        print("Ollama log: not found")
        return
    with OLLAMA_LOG.open("rb") as log:
        log.seek(start if OLLAMA_LOG.stat().st_size >= start else 0)
        lines = log.read().decode("utf-8", "replace").splitlines()
    hits = sorted({line.strip() for line in lines
                   if any(marker in line.lower() for marker in LOG_MARKERS)})
    layers = sorted({line.split("load_tensors:")[-1].strip() for line in lines
                     if "load_tensors: offloaded" in line})
    for line in layers:
        print(f"Ollama log: {line}")
    print(f"Ollama log during the probe: {len(lines)} lines, "
          + ("no ignored options or truncations" if not hits else f"{len(hits)} warnings:"))
    for line in hits[:20]:
        print(f"  {line[:200]}")


def _output_cap(provider, effort: str, cap: int) -> int:
    """Thinking with Ollama needs room: it went past 6144 tokens on a 15-minute block."""
    return THINKING_OUTPUT_TOKENS if provider.name == "ollama" and effort != "none" else cap


def _parse_clock(text: str) -> int:
    parts = [float(p) for p in text.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return int(seconds * 1000)


def _usd(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.5f}"


if __name__ == "__main__":
    sys.exit(main())
