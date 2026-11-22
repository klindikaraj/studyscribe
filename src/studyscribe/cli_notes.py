"""CLI of the notes: `notes`, `transcribe --notes` (its options), `template`, `config set`.

`notes` recovers abandoned work first, prepares the generation (checks with no network, blocks,
estimate), shows the estimate and, for a cloud provider above the cost threshold, asks before
sending anything; then it checks the provider (Ollama running, model downloaded, context size)
and generates, printing one line per request. Ctrl+C leaves the generation resumable.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from studyscribe.db import repo
from studyscribe.db.connection import transaction
from studyscribe.notes import gpu, keys, pricing
from studyscribe.notes.export import NotesUnavailable, export_notes, warnings
from studyscribe.notes.pipeline import NotesError
from studyscribe.notes.providers import DEFAULT_MODELS, DEFAULT_PROVIDER, make_provider
from studyscribe.notes.providers.base import EFFORTS, PROVIDERS, LlmError, LlmProvider
from studyscribe.notes.runner import (
    DEFAULT_EFFORT,
    MIN_NUM_CTX,
    CannotGenerate,
    NotesRequest,
    Prepared,
    check_provider,
    generate_session_notes,
    import_template,
    prepare,
    sync_templates,
)
from studyscribe.notes.templates import (
    DEFAULT_TEMPLATE,
    TemplateError,
    load_template_file,
    parse_template,
)
from studyscribe.recovery import RecoveredNotes

# name -> (type, default, minimum, help)
SETTINGS: dict[str, tuple[type, str, float, str]] = {
    "ollama.num_ctx": (int, str(MIN_NUM_CTX), 2048,
                       "Ollama's context window in tokens (the notes need at least 32768)"),
    "notes.confirm_usd": (float, "0.50", 0.0,
                          "cloud providers: ask before a generation estimated above this cost"),
}

Make = Callable[..., LlmProvider]
Confirm = Callable[[str], bool]


# --- arguments -----------------------------------------------------------------------------------


def add_notes_options(parser: argparse.ArgumentParser, *, in_transcribe: bool = False) -> None:
    """The options of a notes generation; in `transcribe --notes` the model option is
    --notes-model (--model is Whisper's) and there is no --restart (it is the transcription's)."""
    parser.add_argument("--provider", choices=PROVIDERS, default=None,
                        help=f"LLM provider (default {DEFAULT_PROVIDER}: local, on this PC)")
    parser.add_argument("--notes-model" if in_transcribe else "--model", dest="notes_model",
                        help="LLM model (default: the provider's, "
                        f"{DEFAULT_MODELS[DEFAULT_PROVIDER]} for {DEFAULT_PROVIDER})")
    parser.add_argument("--effort", choices=EFFORTS,
                        help="thinking: none (default for ollama), low, medium, high")
    parser.add_argument("--notes-language", metavar="CODE",
                        help="write the notes in this language (default: the transcript's)")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE,
                        help=f"note template (default {DEFAULT_TEMPLATE}; see `template list`)")
    parser.add_argument("--yes", action="store_true",
                        help="cloud providers: do not ask before an expensive generation")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="ollama: go on when the model fits only partly in GPU memory (slow)")
    parser.add_argument("--paid", action="store_true",
                        help="gemini: the paid tier (default: the free tier, cost $0)")
    if not in_transcribe:
        parser.add_argument("--restart", dest="notes_restart", action="store_true",
                            help="start a new generation instead of resuming an interrupted one")


def add_commands(commands: argparse._SubParsersAction) -> None:
    notes = commands.add_parser(
        "notes", help="write study notes from a session's transcript with an LLM",
        description="Generate study notes from the transcript (local Ollama by default: "
        "nothing leaves this PC). An interrupted generation is resumed by running the same "
        "command again.")
    notes.add_argument("session_id", metavar="SESSION_ID")
    add_notes_options(notes)
    template = commands.add_parser("template", help="list or import note templates")
    actions = template.add_subparsers(dest="template_action", metavar="ACTION", required=True)
    actions.add_parser("list", help="the templates and their sections")
    load = actions.add_parser("import", help="add or replace a template from a JSON file")
    load.add_argument("file", type=Path)


def add_config_set(config_actions: argparse._SubParsersAction) -> None:
    config_set = config_actions.add_parser(
        "set", help="change a setting: " + ", ".join(SETTINGS),
        description="\n".join(f"{name}: {info[3]} (default {info[1]})"
                              for name, info in SETTINGS.items()))
    config_set.add_argument("name", choices=sorted(SETTINGS))
    config_set.add_argument("value")


# --- commands ------------------------------------------------------------------------------------


def cmd_notes(conn: sqlite3.Connection, data_dir: Path, args: argparse.Namespace, out: TextIO,
              *, make: Make = make_provider, confirm: Confirm | None = None,
              find_key: Callable[[str], Any] | None = None) -> int:
    """Generate a session's notes; 0 when they are done. `make`, `confirm` and `find_key` let
    tests run without a provider, a console or the credential store."""
    try:
        request = notes_request(conn, args)
        prepared = prepare(conn, args.session_id, request)
    except CannotGenerate as exc:
        print(f"Cannot write notes for {args.session_id}: {exc}", file=out)
        return 2 if repo.get_session(conn, args.session_id) is None else 1
    print_plan(prepared, out)
    if not _confirmed(conn, prepared, args, out, confirm):
        print("Nothing sent.", file=out)
        return 1
    try:
        provider = _make(request, args, make, find_key)
        if hasattr(provider, "on_notice"):  # Ollama: what loading does with models it holds
            provider.on_notice = lambda text: print(f"  {text}", file=out, flush=True)
        print(f"Model check: {check_provider(provider, request)}", file=out)
    except CannotGenerate as exc:
        print(f"Cannot write notes for {args.session_id}: {exc}", file=out)
        return 1
    except LlmError as exc:
        print(f"Error: {exc.user_message()}", file=out)
        return 1
    if request.provider == "ollama":
        memory = gpu.query()
        if memory is not None:
            print(f"GPU memory before loading: {memory.used_mib:,} of {memory.total_mib:,} MiB "
                  "in use", file=out)
    progress = NotesProgress(out, prepared)
    try:
        result = generate_session_notes(conn, data_dir, prepared, provider, emit=progress.show)
    except CannotGenerate as exc:
        print(f"Cannot write notes for {args.session_id}: {exc}", file=out)
        return 1
    except KeyboardInterrupt:
        print(f"\nInterrupted {progress.where()}. Run the same command to resume.", file=out)
        return 1
    except LlmError as exc:
        print(f"Error: {exc.user_message()}\nThe work done is kept: run the same command to "
              "resume.", file=out)
        return 1
    except NotesError as exc:
        remedy = f"\n{exc.remedy}" if exc.remedy else ""
        print(f"Error: {exc.message}{remedy}", file=out)
        return 1
    print_summary(result.notes_id, result.stats, result.result, out)
    # The notes are in the database whatever happens here: a failed export is only reported.
    write_notes_exports(conn, data_dir, args.session_id, out)
    return 0


def write_notes_exports(conn: sqlite3.Connection, data_dir: Path, session_id: str, out: TextIO,
                        *, version: int | None = None) -> int:
    """Write notes.md and notes.html (or notes-vN.*) and say what was written and what the user
    should know; 0 when written, 1 when not (the message says why)."""
    try:
        result = export_notes(conn, data_dir, session_id, version=version)
    except NotesUnavailable as exc:
        print(f"Cannot export the notes: {exc}", file=out)
        return 1
    except OSError as exc:
        print(f"Cannot write the notes exports: {exc}", file=out)
        return 1
    doc = result.doc
    print(f"Notes version {doc.version} of {doc.total_versions} ({doc.model}, {doc.finished}) "
          f"in {result.folder}: {result.markdown.name}, {result.html.name}", file=out)
    for line in warnings(result):
        print(f"  warning: {line}", file=out)
    return 0


def notes_request(conn: sqlite3.Connection, args: argparse.Namespace) -> NotesRequest:
    provider = args.provider or DEFAULT_PROVIDER
    language = getattr(args, "notes_language", None)
    return NotesRequest(
        provider=provider,
        model=args.notes_model or DEFAULT_MODELS[provider],
        effort=args.effort or DEFAULT_EFFORT[provider],
        template_id=args.template,
        language=language.lower() if language else None,
        restart=getattr(args, "notes_restart", False),
        free_tier=provider == "gemini" and not args.paid,
        num_ctx=int(setting(conn, "ollama.num_ctx")) if provider == "ollama" else None,
    )


def cmd_template(conn: sqlite3.Connection, args: argparse.Namespace, out: TextIO) -> int:
    sync_templates(conn)
    if args.template_action == "import":
        try:
            template = load_template_file(args.file)
            import_template(conn, template)
        except TemplateError as exc:
            print(f"Not imported: {exc}", file=out)
            return 1
        print(f"Template {template.id!r} imported ({len(template.sections)} sections).",
              file=out)
        return 0
    for row in repo.list_templates(conn):
        template = parse_template(row["template_json"], row["id"])
        kind = "built-in" if row["builtin"] else "imported"
        print(f"{template.id} ({kind}): {template.title_for('en')} - {template.description}",
              file=out)
        for section in template.sections:
            print(f"  {section.key} ({section.kind}): {section.per_hour:g} per hour, "
                  f"{section.min}-{section.max}", file=out)
    return 0


def config_set(conn: sqlite3.Connection, args: argparse.Namespace, out: TextIO) -> int:
    kind, _, minimum, _ = SETTINGS[args.name]
    try:
        value = kind(args.value)
    except ValueError:
        print(f"{args.name} needs a number, not {args.value!r}.", file=out)
        return 2
    if value < minimum:
        print(f"{args.name} must be at least {minimum:g}.", file=out)
        return 2
    with transaction(conn):
        repo.set_setting(conn, args.name, str(value))
    note = ""
    if args.name == "ollama.num_ctx" and value < MIN_NUM_CTX:
        note = (f" Note: below {MIN_NUM_CTX:,} the notes command refuses to run (the reduce of "
                "a long lecture does not converge).")
    print(f"{args.name} = {value}.{note}", file=out)
    return 0


def setting(conn: sqlite3.Connection, name: str) -> str:
    return repo.get_setting(conn, name, SETTINGS[name][1]) or SETTINGS[name][1]


def settings_lines(conn: sqlite3.Connection) -> list[str]:
    return [f"{name} = {setting(conn, name)}" for name in SETTINGS]


# --- output --------------------------------------------------------------------------------------


def print_plan(prepared: Prepared, out: TextIO) -> None:
    request, estimate = prepared.request, prepared.estimate
    print(f"Notes for {prepared.session_id} \"{prepared.title}\": language "
          f"{prepared.language} ({prepared.language_source}), template {prepared.template.id}, "
          f"{format_minutes(prepared.speech_ms)} of speech", file=out)
    requests = (f"{estimate.count('map')} map + {estimate.count('partial')} partial + "
                f"{estimate.count('reduce')} final")
    print(f"  {len(prepared.blocks)} blocks; requests (estimate): {requests}; about "
          f"{estimate.input_tokens:,} tokens in, {estimate.output_tokens:,} out", file=out)
    if estimate.problem:
        print(f"  warning: {estimate.problem}", file=out)
    if request.provider == "ollama":
        from studyscribe.notes.providers.ollama import host_from_env, is_local

        host = host_from_env()
        where = ("this PC: nothing leaves it" if is_local(host)
                 else "ANOTHER MACHINE: the transcript leaves this PC")
        print(f"  ollama {request.model} at {host} ({where}), context {request.num_ctx:,} "
              f"tokens, effort {request.effort}", file=out)
        seconds = estimate.seconds(*prepared.speeds)
        print(f"  estimated time: about {format_minutes(seconds * 1000)} (plus loading the "
              "model)", file=out)
        return
    print(f"  {request.provider} {request.model}, effort {request.effort}", file=out)
    cost = estimate.cost(request.provider, request.model, free_tier=request.free_tier)
    if cost is None:
        print("  cost: unknown (no price for this model)", file=out)
    else:
        charged, listed = cost
        print(f"  estimated cost: ${charged:.4f} (${listed:.4f} at list price; prices checked "
              f"{pricing.PRICES_VERIFIED})", file=out)
    if request.free_tier:
        print("  Gemini free tier: Google may use this transcript to improve its products.",
              file=out)


def _confirmed(conn: sqlite3.Connection, prepared: Prepared, args: argparse.Namespace,
               out: TextIO, confirm: Confirm | None) -> bool:
    request = prepared.request
    if request.provider == "ollama" or args.yes:
        return True
    cost = prepared.estimate.cost(request.provider, request.model, free_tier=request.free_tier)
    threshold = float(setting(conn, "notes.confirm_usd"))
    if cost is not None and cost[0] <= threshold:
        return True
    what = "an unknown cost" if cost is None else f"about ${cost[0]:.2f}"
    ask = confirm or _ask
    return ask(f"This generation costs {what} (threshold ${threshold:.2f}). Go on? [y/N] ")


def _ask(prompt: str) -> bool:
    if sys.stdin is None or not sys.stdin.isatty():
        print(prompt + "no terminal to answer: use --yes")
        return False
    return input(prompt).strip().lower() in ("y", "yes")


def _make(request: NotesRequest, args: argparse.Namespace, make: Make,
          find_key: Callable[[str], Any] | None) -> LlmProvider:
    if request.provider == "ollama":
        from studyscribe.notes.providers.ollama import OllamaOptions

        return make("ollama", request.model, None,
                    ollama=OllamaOptions(num_ctx=request.num_ctx or MIN_NUM_CTX,
                                         allow_cpu=args.allow_cpu))
    found = (find_key or keys.require_key)(request.provider)
    return make(request.provider, request.model, found.key)


class NotesProgress:
    """One line per step and per request (no redraw: a run takes minutes, the log stays)."""

    def __init__(self, out: TextIO, prepared: Prepared) -> None:
        self.out = out
        self.blocks = len(prepared.blocks)
        self.last = "before the first request"

    def show(self, kind: str, **info: Any) -> None:
        out = self.out
        if kind == "started":
            print(f"Generation #{info['notes_id']} started.", file=out)
        elif kind == "resumed":
            print(f"Resuming generation #{info['notes_id']}: {info['maps']} of "
                  f"{info['blocks']} map blocks already done.", file=out)
        elif kind == "discarded":
            print(f"Previous generation #{info['notes_id']} discarded ({info['reason']}); its "
                  "calls stay stored.", file=out)
        elif kind == "progress":
            text = info["text"]
            if not text.startswith("warning:"):
                self.last = f"at {text}"
            print(f"  {text}", file=out, flush=True)
        elif kind == "call":
            record = info["record"]
            usage = record.usage
            tokens = (f", {usage.input_tokens:,} in / {usage.output_tokens:,} out tokens"
                      if usage else "")
            rate = ""
            if record.timings and usage and record.timings.generate_s > 0:
                rate = f", {usage.output_tokens / record.timings.generate_s:.1f} tokens/s"
            took = f" in {record.duration_s:.1f} s" if record.duration_s else ""
            error = f": {record.error.splitlines()[0]}" if record.error else ""
            print(f"    {record.status}{took}{tokens}{rate}{error}", file=out, flush=True)

    def where(self) -> str:
        return self.last


def print_summary(notes_id: int, stats: dict[str, Any], result: Any, out: TextIO) -> None:
    reduce = stats.get("reduce", {})
    print(f"Done (generation #{notes_id}) in {format_minutes(stats['wall_s'] * 1000)}: maps "
          f"{stats['map_calls']} calls {format_minutes(stats['map_s'] * 1000)}, partial "
          f"reduces {stats['partial_calls']} calls {format_minutes(stats['partial_s'] * 1000)}, "
          f"final reduce {format_minutes(stats['reduce_s'] * 1000)}", file=out)
    counts = stats["count_tokens"]
    print(f"  tokens {stats['input_tokens']:,} in / {stats['output_tokens']:,} out; "
          f"{reduce.get('tokens_per_item')} tokens per item; {counts['calls']} token counts "
          f"in {counts['seconds']:.1f} s", file=out)
    if stats.get("cost_usd") or stats.get("list_cost_usd"):
        print(f"  cost ${stats['cost_usd']:.4f} (${stats['list_cost_usd']:.4f} at list price)",
              file=out)
    code = stats.get("code_checks", {})
    print(f"  code items: {code.get('ok', 0)} ok, {code.get('check', 0)} to verify, "
          f"{code.get('unsupported', 0)} with names not in the transcript", file=out)
    coverage = stats.get("coverage_warnings", [])
    print(f"  coverage warnings: {len(coverage) or 'none'}", file=out)
    for gap in coverage:
        print(f"    block {gap['block']}: lines S{gap['uncovered'][0]}-S{gap['uncovered'][1]} "
              "not covered by the notes", file=out)
    if reduce.get("final_targets_scaled"):
        print(f"  final notes asked for {reduce['final_targets_scaled']:.0%} of the template's "
              "items (to fit one answer)", file=out)
    refs = stats.get("refs", {})
    dropped = sum(r.get("invalid_refs", 0) for r in refs.values())
    print(f"  references dropped: {dropped}; words to check by listening: "
          f"{stats.get('uncertain_words', 0)}", file=out)
    print("  The notes are stored in the database; the notes.md / notes.html export is not "
          "written yet.", file=out)


def print_notes_recovery(results: list[RecoveredNotes], out: TextIO) -> None:
    for r in results:
        print(f"Notes of {r.session_id} interrupted ({r.maps_done} of {r.blocks} map blocks "
              "done): run `notes` again to resume them.", file=out)


def format_minutes(ms: float) -> str:
    seconds = round(ms / 1000)
    if seconds < 60:
        return f"{seconds} s"
    return f"{seconds // 60} min {seconds % 60:02d} s"
