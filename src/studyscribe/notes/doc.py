"""One generation of notes, read from the database for the exports (notes.md, notes.html).

The renderers take a `NotesDoc` and return text: they never touch the database or the disk. What
is decided here, once:

- **Version.** A session's notes are numbered from 1 over its completed generations, oldest
  first. No version means the latest.
- **Same transcript or not.** The notes cite transcript lines by segment id and keep the line's
  start time. When the session's current transcript is the one the notes were written from,
  every reference is checked against it (the stored start time must be the segment's) and the
  text of the lines is available for the appendix. After a re-transcription the times still point
  to the right place in the audio, but the line numbers and ids may not mean anything any more:
  the check is skipped and the exports say so.
- **Audio.** Whether `audio.flac` is in the exports folder. Nothing here creates it.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studyscribe.db import repo
from studyscribe.notes.templates import Template, TemplateError, parse_template
from studyscribe.notes.transcript import make_lines
from studyscribe.notes.uncertain import UncertainWord
from studyscribe.transcribe.export import AUDIO_FILE, exports_dir, local_time


class NotesUnavailable(RuntimeError):
    """There are no notes to export (or not that version); the message says what exists."""


@dataclass(frozen=True)
class NotesDoc:
    session_id: str
    title: str
    course: str | None
    recorded: str  # local date and time
    duration_ms: int
    language: str  # the language the notes are written in
    notes_id: int
    version: int  # position among the session's completed generations, from 1
    total_versions: int
    provider: str
    model: str
    effort: str
    prompt_version: str
    finished: str  # local date and time
    template: Template
    content: dict[str, Any]  # the notes, references as {"n", "segment_id", "start_ms"}
    uncertain: tuple[UncertainWord, ...]
    stats: dict[str, Any]
    line_texts: Mapping[int, str] | None  # S-number -> text; None when the transcript changed
    time_mismatches: int  # references whose time is not their segment's (same transcript only)
    audio_present: bool

    @property
    def same_transcript(self) -> bool:
        return self.line_texts is not None

    @property
    def coverage(self) -> list[dict[str, Any]]:
        return list(self.stats.get("coverage_warnings", []))

    @property
    def dropped(self) -> tuple[int, int]:
        """(references, items) the generation left out because the references were invalid."""
        refs = items = 0
        for report in self.stats.get("refs", {}).values():
            refs += int(report.get("invalid_refs", 0))
            items += sum(int(n) for n in report.get("dropped_items", {}).values())
        return refs, items


def load_notes(conn: sqlite3.Connection, data_dir: Path, session_id: str,
               version: int | None = None) -> NotesDoc:
    """The notes of a session: its latest completed generation, or `version`. Raises
    NotesUnavailable."""
    session = repo.get_session_summary(conn, session_id)
    if session is None:
        raise NotesUnavailable(f"no session {session_id} in this data directory.")
    done = repo.list_done_notes(conn, session_id)
    if not done:
        raise NotesUnavailable(f"session {session_id} has no finished notes yet. Run: "
                               f"studyscribe notes {session_id}")
    if version is None:
        version = len(done)
    elif not 1 <= version <= len(done):
        raise NotesUnavailable(f"session {session_id} has no notes version {version}. "
                               f"Available: {describe_versions(done)}")
    row = done[version - 1]
    try:
        template = parse_template(row["template_json"], "the notes' template")
    except TemplateError as exc:
        raise NotesUnavailable(str(exc)) from exc
    content = json.loads(row["content_json"])
    options = json.loads(row["options_json"] or "{}")

    line_texts: dict[int, str] | None = None
    mismatches = 0
    run = repo.get_done_run(conn, session_id)
    if run is not None and row["transcript_run_id"] == run["id"]:
        lines = make_lines(repo.list_run_segments(conn, run["id"], include_dropped=False))
        line_texts = {line.n: line.text for line in lines}
        mismatches = count_time_mismatches(content, lines)

    return NotesDoc(
        session_id=session_id,
        title=session["title"] or session_id,
        course=session["course"],
        recorded=local_time(session["started_at"]),
        duration_ms=int(session["duration_ms"]),
        language=row["language"],
        notes_id=row["id"],
        version=version,
        total_versions=len(done),
        provider=row["provider"],
        model=row["model"],
        effort=str(options.get("effort", "?")),
        prompt_version=row["prompt_version"],
        finished=local_time(row["finished_at"]) if row["finished_at"] else "",
        template=template,
        content=content,
        uncertain=tuple(UncertainWord(**word) for word in json.loads(row["uncertain_json"]
                                                                      or "[]")),
        stats=json.loads(row["stats_json"] or "{}"),
        line_texts=line_texts,
        time_mismatches=mismatches,
        audio_present=(exports_dir(data_dir, session_id) / AUDIO_FILE).is_file(),
    )


def describe_versions(done: Iterable[sqlite3.Row]) -> str:
    """"1 (2026-11-16 10:05, ollama qwen3.5:9b, effort none), 2 (...)"."""
    parts = []
    for number, row in enumerate(done, start=1):
        effort = json.loads(row["options_json"] or "{}").get("effort", "?")
        when = local_time(row["finished_at"]) if row["finished_at"] else "?"
        parts.append(f"{number} ({when}, {row['provider']} {row['model']}, effort {effort})")
    return ", ".join(parts)


# --- references ----------------------------------------------------------------------------------


def iter_refs(content: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Every reference of the notes: a topic's `ref`, the `refs` of the other items."""
    for value in content.values():
        if isinstance(value, list):
            for item in value:
                if "ref" in item:
                    yield item["ref"]
                yield from item.get("refs", ())


def seek_seconds(ref: Mapping[str, Any]) -> int:
    """Where a timestamp link seeks: the start second of the line, up to 1 s before its first
    word and never after it."""
    return int(ref["start_ms"]) // 1000


def timestamps(refs: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The references to show next to an item: in time order, one per second."""
    seen: set[int] = set()
    out = []
    for ref in sorted(refs, key=lambda r: int(r["start_ms"])):
        second = seek_seconds(ref)
        if second not in seen:
            seen.add(second)
            out.append(ref)
    return out


def count_time_mismatches(content: Mapping[str, Any], lines: Iterable[Any]) -> int:
    """References whose line is not the segment they name, or whose time is not that segment's
    start. `lines`: the transcript the notes were written from (see transcript.make_lines)."""
    by_n = {line.n: line for line in lines}
    starts = {line.segment_id: line.start_ms for line in by_n.values()}
    bad = 0
    for ref in iter_refs(content):
        line = by_n.get(int(ref["n"]))
        if (line is None or line.segment_id != ref["segment_id"]
                or starts.get(ref["segment_id"]) != int(ref["start_ms"])):
            bad += 1
    return bad
