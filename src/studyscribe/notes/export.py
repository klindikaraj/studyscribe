"""Write a session's notes to <session>/exports/: notes.md and notes.html.

The latest completed generation is `notes.md` / `notes.html`; an explicit version N is
`notes-vN.md` / `notes-vN.html` (its Markdown links point to its own HTML page). Both are
rendered from the database every time (nothing is cached), and each is written to a temporary
name first and renamed, so a reader never sees half a file. `audio.flac` is never created here:
`studyscribe export --audio` does that, and the page says so when it is missing.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from studyscribe.notes.doc import NotesDoc, NotesUnavailable, load_notes
from studyscribe.notes.render_html import render_html
from studyscribe.notes.render_md import render_markdown
from studyscribe.transcribe.export import AUDIO_FILE, exports_dir, write_text

__all__ = ["NotesExport", "NotesUnavailable", "export_notes", "notes_file_names", "warnings"]


@dataclass(frozen=True)
class NotesExport:
    markdown: Path
    html: Path
    doc: NotesDoc

    @property
    def folder(self) -> Path:
        return self.html.parent


def notes_file_names(version: int | None) -> tuple[str, str]:
    """(markdown, html) file names: `notes.*` for the latest, `notes-vN.*` for a version."""
    stem = "notes" if version is None else f"notes-v{version}"
    return f"{stem}.md", f"{stem}.html"


def export_notes(conn: sqlite3.Connection, data_dir: Path, session_id: str, *,
                 version: int | None = None) -> NotesExport:
    """Write the notes files. `version`: None for the latest completed generation. Raises
    NotesUnavailable (nothing to export) or OSError (the files could not be written)."""
    doc = load_notes(conn, data_dir, session_id, version)
    md_name, html_name = notes_file_names(version)
    folder = exports_dir(data_dir, session_id)
    folder.mkdir(parents=True, exist_ok=True)
    write_text(folder / html_name, render_html(doc))
    write_text(folder / md_name, render_markdown(doc, html_name))
    return NotesExport(folder / md_name, folder / html_name, doc)


def warnings(result: NotesExport) -> list[str]:
    """What the user should know about the files just written, one line each."""
    doc = result.doc
    found = []
    if not doc.audio_present:
        found.append(f"{AUDIO_FILE} is not in {result.folder}: the timestamps in "
                     f"{result.html.name} are plain text until you run: "
                     f"studyscribe export {doc.session_id} --audio")
    if doc.time_mismatches:
        found.append(f"{doc.time_mismatches} timestamps in the notes do not match the "
                     "transcript's segments: generate the notes again.")
    if not doc.same_transcript:
        found.append("these notes were written from an earlier transcript: their times still "
                     "point to the right place in the audio, line numbers (S...) may not match.")
    return found
