"""LLM probe for step 3: one tiny structured request, and exact token counts of transcripts.

Usage:
    .venv\\Scripts\\python scripts\\llm_probe.py --provider gemini [--model gemini-3.8-flash]
        [--effort low|medium|high] [--paid]
    .venv\\Scripts\\python scripts\\llm_probe.py --provider anthropic --model claude-haiku-5-5
    .venv\\Scripts\\python scripts\\llm_probe.py --provider gemini --count-only
        --data-dir PATH --session ID [--session ID ...]

The first form checks the key and the model name (free), then sends one request of a few
hundred tokens asking for JSON with three fields, validates it with pydantic and prints model
version, request ID, latency, tokens (thinking included) and cost at list price.

With --session it also counts, with the provider's own (free) token counter, the input tokens
of each session's whole transcript in the line format the notes pipeline will send
("S<n> text", a [H:MM:SS] marker every minute). The database is opened read-only.
--count-only skips the generation request.

The key comes from the Windows Credential Manager or the environment (studyscribe config
set-key). Only text is sent, never audio.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from studyscribe.config import db_path
from studyscribe.notes import keys, pricing
from studyscribe.notes.providers import DEFAULT_MODELS, make_provider
from studyscribe.notes.providers.base import EFFORTS, PROVIDERS, LlmError, LlmRequest
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


class ProbeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(description="ISO 639-1 code of the passage's language")
    topic: str = Field(description="the topic in at most eight words")
    identifiers: list[str] = Field(description="code identifiers mentioned, in code form")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=PROVIDERS, required=True)
    parser.add_argument("--model", help="default: the provider's default model")
    parser.add_argument("--effort", choices=EFFORTS, default="low")
    parser.add_argument("--paid", action="store_true",
                        help="Gemini paid tier (only changes the cost shown)")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--session", action="append", default=[], metavar="ID")
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    model = args.model or DEFAULT_MODELS[args.provider]
    free_tier = args.provider == "gemini" and not args.paid

    try:
        found = keys.require_key(args.provider)
        print(f"Provider {args.provider}, model {model}, key {keys.mask(found.key)} "
              f"from {found.source}")
        if free_tier:
            print("Gemini free tier: Google may use this text to improve its products.")
        provider = make_provider(args.provider, model, found.key)
        print(f"Model check (free): {provider.check()}")
        if not args.count_only:
            probe_request(provider, args.effort, free_tier)
        for session_id in args.session:
            if args.data_dir is None:
                parser.error("--session needs --data-dir")
            count_session(provider, args.data_dir, session_id)
    except LlmError as exc:
        print(f"Error: {exc.user_message()}")
        return 1
    return 0


def probe_request(provider, effort: str, free_tier: bool) -> None:
    request = LlmRequest(system=PROBE_SYSTEM, user=PROBE_TEXT,
                         schema=ProbeAnswer.model_json_schema(), max_output_tokens=4000,
                         effort=effort)
    response = call_with_retry(
        lambda: provider.generate(request),
        on_wait=lambda reason, s: print(f"  waiting {s:.0f} s ({reason})"),
    )
    usage = response.usage
    list_cost = pricing.list_cost(provider.name, provider.model, usage)
    charged = pricing.charged_cost(provider.name, provider.model, usage, free_tier=free_tier)
    print(f"Answer in {response.duration_s:.1f} s, model version {response.model_version}, "
          f"request {response.request_id}")
    print(f"Tokens: input {usage.input_tokens} (cached {usage.cached_tokens}), output "
          f"{usage.output_tokens} (thinking {usage.thinking_tokens})")
    print(f"Cost: {_usd(charged)} charged, {_usd(list_cost)} at list price "
          f"(prices checked {pricing.PRICES_VERIFIED})")
    try:
        answer = ProbeAnswer.model_validate_json(response.text)
        print(f"Valid JSON: {answer.model_dump()}")
    except ValidationError as exc:
        print(f"INVALID JSON ({exc.error_count()} errors): {response.text[:500]!r}")


def count_session(provider, data_dir: Path, session_id: str) -> None:
    uri = db_path(data_dir).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        run = conn.execute(
            "SELECT id, language FROM transcription_runs WHERE session_id = ? AND pass = 'final'"
            " AND status = 'done'", (session_id,)).fetchone()
        if run is None:
            print(f"{session_id}: no transcript")
            return
        rows = conn.execute(
            "SELECT start_ms, end_ms, text FROM transcript_segments WHERE run_id = ?"
            " AND dropped IS NULL ORDER BY start_ms, id", (run["id"],)).fetchall()
    finally:
        conn.close()
    text = transcript_lines(rows)
    tokens = provider.count_tokens(PROBE_SYSTEM, text)
    print(f"{session_id} ({run['language']}): {len(rows)} segments, {len(text):,} characters, "
          f"{tokens:,} input tokens with {provider.model} ({len(text) / tokens:.2f} chars/token)")


def transcript_lines(rows) -> str:
    """The pipeline's input format (piece 2 will own it): S<n> text, a marker every minute."""
    lines = []
    next_marker = 0
    for n, row in enumerate(rows, start=1):
        if row["start_ms"] >= next_marker:
            lines.append(f"[{clock(row['start_ms'])}]")
            next_marker = (row["start_ms"] // 60_000 + 1) * 60_000
        lines.append(f"S{n} {row['text'].strip()}")
    return "\n".join(lines)


def _usd(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.5f}"


if __name__ == "__main__":
    sys.exit(main())
