"""What notes.md and notes.html share: the texts around the notes, the warnings, the appendix rows.

Everything here returns plain, *unescaped* text (and numbers): each renderer escapes for its own
format. Keeping the choices (what is a warning, what the appendix lists, how a time is written)
in one place means the two files cannot disagree.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from studyscribe.notes import labels
from studyscribe.notes.doc import NotesDoc
from studyscribe.transcribe.export import clock


def header_rows(doc: NotesDoc) -> list[tuple[str, str]]:
    """(label, value) lines under the title."""
    def label(key: str) -> str:
        return labels.text(doc.language, key)

    rows = []
    if doc.course:
        rows.append((label("course"), doc.course))
    rows += [
        (label("recorded"), doc.recorded),
        (label("duration"), clock(doc.duration_ms)),
        (label("language"), doc.language),
        (label("notes_info"), labels.text(doc.language, "notes_info_value", version=doc.version,
                                          total=doc.total_versions, model=doc.model,
                                          date=doc.finished)),
    ]
    return rows


def difficulty_label(doc: NotesDoc, difficulty: str) -> str:
    try:
        return labels.text(doc.language, f"difficulty_{difficulty}")
    except KeyError:  # not one of easy, medium, hard (the schema allows only those)
        return difficulty


@dataclass(frozen=True)
class CodeFlag:
    kind: str  # "check" or "unsupported"
    text: str  # "to verify: y_pred"


def code_flags(doc: NotesDoc, item: Mapping[str, Any]) -> list[CodeFlag]:
    """The warnings of a code item: nothing when every name was found in the cited lines."""
    check = item.get("check") or {}
    flags = []
    missing, partial = list(check.get("missing", [])), list(check.get("partial", []))
    if check.get("level") == "unsupported" or missing:
        flags.append(CodeFlag("unsupported", _with_names(
            labels.text(doc.language, "code_unsupported"), missing)))
    if partial or check.get("level") == "check":
        flags.append(CodeFlag("check", _with_names(labels.text(doc.language, "code_check"),
                                                   partial)))
    return flags


def _with_names(label: str, names: list[str]) -> str:
    return f"{label}: {', '.join(names)}" if names else label


@dataclass(frozen=True)
class Banner:
    kind: str  # "coverage", "earlier", "mismatch"
    text: str
    links: tuple[tuple[int, str], ...] = ()  # (seek second, text) for a coverage warning


def banners(doc: NotesDoc) -> list[Banner]:
    """The warnings shown above the notes (the missing-audio one belongs to the HTML only)."""
    found = []
    if not doc.same_transcript:
        found.append(Banner("earlier", labels.text(doc.language, "earlier_transcript")))
    if doc.time_mismatches:
        found.append(Banner("mismatch", labels.text(doc.language, "time_mismatch",
                                                    count=doc.time_mismatches)))
    if doc.coverage:
        links = []
        for gap in doc.coverage:
            first, last = gap["uncovered"]
            links.append((int(gap["start_ms"]) // 1000, labels.text(
                doc.language, "coverage_item", start=clock(int(gap["start_ms"])),
                end=clock(int(gap["end_ms"])), first=f"S{first}", last=f"S{last}")))
        found.append(Banner("coverage", labels.text(doc.language, "coverage_title"),
                            tuple(links)))
    return found


@dataclass(frozen=True)
class AppendixRow:
    word: str
    line: str  # "S186"
    seconds: int  # where the time link seeks
    time: str  # "0:14:11"
    probability: str  # "0.35"
    before: str | None  # the line's text around the word; None when it is not available
    match: str = ""
    after: str = ""


def appendix_rows(doc: NotesDoc) -> list[AppendixRow]:
    rows = []
    for word in doc.uncertain:
        parts = None
        if doc.line_texts is not None and word.n in doc.line_texts:
            parts = locate_word(doc.line_texts[word.n], word.word)
        rows.append(AppendixRow(
            word=word.word, line=f"S{word.n}", seconds=word.start_ms // 1000,
            time=clock(word.start_ms), probability=f"{word.probability:.2f}",
            before=parts[0] if parts else (doc.line_texts or {}).get(word.n),
            match=parts[1] if parts else "", after=parts[2] if parts else ""))
    return rows


def locate_word(text: str, word: str) -> tuple[str, str, str] | None:
    """(before, the word as written, after) for the first whole-word occurrence of `word` in
    `text` (case ignored), or any occurrence; None when it is not there."""
    if not word:
        return None
    found = (re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE)
             or re.search(re.escape(word), text, re.IGNORECASE))
    if found is None:
        return None
    return text[:found.start()], found.group(0), text[found.end():]


def footer_lines(doc: NotesDoc) -> list[str]:
    """How the notes were made and what was left out, one line each; none for what is unknown."""
    lines = []
    stats = doc.stats
    if "input_tokens" in stats and "output_tokens" in stats:
        lines.append(labels.text(
            doc.language, "footer_generated", model=doc.model, effort=doc.effort,
            prompt=doc.prompt_version, input_tokens=f"{int(stats['input_tokens']):,}",
            output_tokens=f"{int(stats['output_tokens']):,}",
            seconds=round(float(stats.get("wall_s", 0)))))
    refs, items = doc.dropped
    if refs or items:
        lines.append(labels.text(doc.language, "footer_dropped", refs=refs, items=items))
    return lines
