"""LLM probe for step 3: one tiny structured request, exact token counts of transcripts and, for
the local provider, one real map-like request on a block of a lecture.

Usage:
    .venv\\Scripts\\python scripts\\llm_probe.py                      (Ollama, qwen3.5:9b)
    .venv\\Scripts\\python scripts\\llm_probe.py --provider ollama --data-dir PATH
        --session ID [--session ID ...] [--block-from 0:30:00] [--block-minutes 15]
        [--model qwen3.5:9b] [--num-ctx 32768] [--effort none|low|medium|high] [--allow-cpu]
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
see whether the fixed seed gives the same answer; with --session, a map-like request on one
block of the first session (draft prompt and schema of the notes pipeline), its references
checked against the block, and the answer saved to a JSON file; at the end the model is
unloaded and Ollama's log is searched for ignored options or truncated prompts.

The database is opened read-only. Only text is sent (to this PC, for Ollama), never audio.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from studyscribe.config import db_path
from studyscribe.notes import gpu, keys, pricing
from studyscribe.notes.providers import DEFAULT_MODELS, DEFAULT_PROVIDER, make_provider
from studyscribe.notes.providers.base import EFFORTS, PROVIDERS, LlmError, LlmRequest
from studyscribe.notes.providers.ollama import (
    DEFAULT_NUM_CTX,
    OllamaOptions,
    OllamaProvider,
    is_local,
)
from studyscribe.notes.providers.retry import call_with_retry
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
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STRING_LITERAL = re.compile(r"\"[^\"]*\"|'[^']*'|`[^`]*`")
PROSE = re.compile(r"[A-Za-z']+(?:\s+[A-Za-z']+){4,}")  # five plain words in a row
CODE_KEYWORDS = {
    "var", "let", "const", "function", "return", "if", "else", "for", "while", "new", "import",
    "from", "as", "def", "class", "self", "this", "true", "false", "null", "undefined", "none",
    "in", "of", "and", "or", "not", "print", "console", "log", "len", "range",
}
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
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v2",
                        help="ollama only: draft map prompt of the block test")
    parser.add_argument("--expect", action="append", default=[], metavar="WORD",
                        help="ollama only: report where WORD appears in the block's notes")
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
    provider = OllamaProvider(args.model, options, on_progress=_progress)
    where = "this PC" if is_local(options.host) else "ANOTHER MACHINE: the text leaves this PC"
    print(f"Provider ollama at {options.host} ({where}), model {args.model}, "
          f"context {args.num_ctx:,} tokens, effort {args.effort}")
    log_start = OLLAMA_LOG.stat().st_size if OLLAMA_LOG.exists() else None
    print(f"GPU before loading: {_gpu_text(gpu.query())}")
    print(f"Model check: {provider.check()}")
    info = provider.model_info()
    kv = info.kv_bytes_per_token
    print(f"  family {info.family}, capabilities {', '.join(info.capabilities)}, thinking "
          f"values {info.thinking_values}, max context {info.context_length}")
    if kv:
        print(f"  KV cache (f16): {kv / 1024:.0f} KiB/token = {kv * args.num_ctx / 2**30:.2f} GiB "
              f"at {args.num_ctx:,} tokens")
    try:
        load = provider.ensure_loaded()
        print(f"Loaded in {load.load_s:.1f} s: Ollama allocated {load.size_bytes / 1e9:.2f} GB, "
              f"{load.vram_bytes / 1e9:.2f} GB in GPU memory "
              f"({'entirely on the GPU' if load.on_gpu_entirely else 'PARTLY ON THE CPU'}), "
              f"context {load.context_length:,}")
        print(f"GPU after loading: {_gpu_text(load.gpu_after)}")
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
    start_ms = _parse_clock(args.block_from)
    end_ms = start_ms + int(args.block_minutes * 60_000)
    numbered = [(n, row) for n, row in enumerate(rows, start=1)
                if start_ms <= row["start_ms"] < end_ms]
    if not numbered:
        print(f"\nBlock {clock(start_ms)}-{clock(end_ms)}: no segments there")
        return
    first, last = numbered[0][0], numbered[-1][0]
    cap = _output_cap(provider, args.effort, BLOCK_OUTPUT_TOKENS)
    user = transcript_lines([row for _, row in numbered], first_n=first)
    system = PROMPTS[args.prompt].format(
        title=title, start=clock(start_ms), end=clock(end_ms), first=f"S{first}", last=f"S{last}",
        language=LANGUAGE_NAMES.get(language or "", language or "the transcript's language"),
        schema=json.dumps(BlockNotes.model_json_schema(), separators=(",", ":")),
        **{key: max(1, round(rate * args.block_minutes / 60))
           for key, rate in TARGETS_PER_HOUR.items()})
    print(f"\nBlock {clock(start_ms)}-{clock(end_ms)} of {session_id}: segments S{first}-S{last}, "
          f"{len(system) + len(user):,} characters of prompt (prompt {args.prompt}), "
          f"output cap {cap:,} tokens")
    request = LlmRequest(system=system, user=user, schema=BlockNotes.model_json_schema(),
                         max_output_tokens=cap, effort=args.effort)
    response = call_with_retry(
        lambda: provider.generate(request),
        on_wait=lambda reason, s: print(f"  waiting {s:.0f} s ({reason})"),
    )
    _end_progress()
    usage = response.usage
    chars = len(system) + len(user)
    print(f"Answer in {response.duration_s:.1f} s: input {usage.input_tokens:,} tokens "
          f"({chars / usage.input_tokens:.2f} chars/token), output {usage.output_tokens:,} tokens"
          f" (thinking {usage.thinking_tokens})")
    _print_timings(response)
    out = Path(tempfile.gettempdir()) / (
        f"studyscribe-probe-block-{session_id}-{args.prompt}-{args.effort}.json")
    out.write_text(response.text, encoding="utf-8")
    try:
        notes = BlockNotes.model_validate_json(response.text)
    except ValidationError as exc:
        print(f"INVALID JSON ({exc.error_count()} errors), saved to {out}")
        return
    refs = [r for t in notes.topics for r in [t.ref]]
    for items in (notes.concepts, notes.definitions, notes.code, notes.open_points,
                  notes.corrections):
        refs += [r for item in items for r in item.refs]
    outside = [r for r in refs if not first <= r <= last]
    print(f"Valid JSON, saved to {out}")
    print(f"  {len(notes.topics)} topics, {len(notes.concepts)} concepts, "
          f"{len(notes.definitions)} definitions, {len(notes.code)} code examples, "
          f"{len(notes.open_points)} open points, {len(notes.corrections)} corrections")
    print(f"  references: {len(refs)}, outside the block: {len(outside)}"
          + (f" {sorted(set(outside))[:10]}" if outside else ""))
    for topic in notes.topics:
        at = clock(rows[topic.ref - 1]["start_ms"]) if 1 <= topic.ref <= len(rows) else "?"
        print(f"  [{at}] S{topic.ref} {topic.title}")
    for fix in notes.corrections:
        cited = " ".join(rows[n - 1]["text"] for r in fix.refs for n in (r - 1, r, r + 1)
                         if 1 <= n <= len(rows)).lower()
        heard = fix.heard.lower().strip()
        verdict = ("" if heard and heard in cited
                   else "  <- INVALID: the heard text is not in the cited lines (+-1)")
        if not verdict and fix.written.lower().strip() in cited:
            verdict = "  <- INVALID: the written form is already in the transcript"
        print(f"  correction: {fix.heard!r} -> {fix.written!r} {fix.refs}{verdict}")
    fixed = {w.lower() for fix in notes.corrections for w in IDENTIFIER.findall(fix.written)}
    flagged = 0
    for example in notes.code:
        cited = [rows[n - 1]["text"] for r in example.refs for n in (r - 1, r, r + 1)
                 if 1 <= n <= len(rows)]
        missing = unsupported_names(example.code, cited, fixed)
        prose = PROSE.search(STRING_LITERAL.sub("", example.code))
        flagged += bool(missing or prose)
        print(f"  code: {example.code!r} {example.refs}"
              + (f"  <- not in the cited lines: {', '.join(missing)}" if missing else "")
              + ("  <- prose, not code" if prose else ""))
    print(f"  code items flagged (names not in the cited lines +-1, or prose): {flagged} of "
          f"{len(notes.code)}")
    with_s = [c.code for c in notes.code if S_NUMBER.search(c.code)]
    print(f"  code with an S-number inside: {len(with_s)}" + (f" {with_s[:5]}" if with_s else ""))
    if args.prompt != "v1":
        copied = [t for t in EXAMPLE_TEXTS if t in response.text]
        print(f"  copied from the prompt's example: {copied if copied else 'nothing'}")
    data = notes.model_dump()
    for word in args.expect:
        where = sorted({section for section, items in data.items()
                        if word.lower() in json.dumps(items, ensure_ascii=False).lower()})
        print(f"  {word!r}: " + (f"in {', '.join(where)}" if where else "NOT FOUND"))


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
            "SELECT start_ms, end_ms, text FROM transcript_segments WHERE run_id = ?"
            " AND dropped IS NULL ORDER BY start_ms, id", (run["id"],)).fetchall()
    finally:
        conn.close()
    title = (session["title"] if session is not None else None) or session_id
    return rows, run["language"], title


def count_session(provider, data_dir: Path, session_id: str) -> None:
    rows, language, _ = load_transcript(data_dir, session_id)
    if rows is None:
        return
    text = transcript_lines(rows)
    tokens = provider.count_tokens(PROBE_SYSTEM, text)
    _end_progress()
    print(f"{session_id} ({language}): {len(rows)} segments, {len(text):,} characters, "
          f"{tokens:,} input tokens with {provider.model} ({len(text) / tokens:.2f} chars/token)")


def transcript_lines(rows, first_n: int = 1) -> str:
    """The pipeline's input format (piece 2 will own it): S<n> text, a marker every minute."""
    lines = []
    next_marker = 0
    for n, row in enumerate(rows, start=first_n):
        if row["start_ms"] >= next_marker:
            lines.append(f"[{clock(row['start_ms'])}]")
            next_marker = (row["start_ms"] // 60_000 + 1) * 60_000
        lines.append(f"S{n} {row['text'].strip()}")
    return "\n".join(lines)


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


def unsupported_names(code: str, cited: list[str], fixed: set[str]) -> list[str]:
    """Names in a code item that no word of the cited transcript lines resembles: each name is
    split into its words (snake_case, camelCase); a word of 3+ letters is supported when a
    transcript word contains it or is at least 80% similar to it, or when it is the written
    form of a listed correction. Keywords and string literals are not checked."""
    words = {w.lower() for line in cited for w in re.findall(r"\w+", line)}
    missing = []
    for name in IDENTIFIER.findall(STRING_LITERAL.sub("", code)):
        if name.lower() in CODE_KEYWORDS or name.lower() in fixed:
            continue
        parts = [p.lower() for p in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+",
                                                name.replace("_", " "))]
        for part in parts:
            if len(part) < 3 or part.isdigit() or part in fixed:
                continue
            if not any(part in w or SequenceMatcher(None, part, w).ratio() >= 0.8
                       for w in words):
                missing.append(name)
                break
    return list(dict.fromkeys(missing))


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
