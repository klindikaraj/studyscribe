"""notes.html: one self-contained page with the notes, an audio player and timestamps that seek.

The page works offline, opened from disk: one inline style, one inline script, no external
resource. `<audio>` points to `audio.flac` in the same folder (a relative path, so the folder must
stay together); this module never creates that file. When it is not there the page says so
without needing JavaScript, and the timestamps are still readable text.

Safety, in layers:

1. Every string that is not ours (the notes, the transcript's words, the title, the course, the
   template's headings, the model's name) goes through `esc()` (`html.escape`, quotes included)
   before it is put in the page. Numbers go through `int()`. Of the notes' own markup only
   backticks are honoured, and only after escaping, so they cannot open a tag.
2. The script reads nothing from the notes: timestamps carry their second in `data-t`.
3. A Content-Security-Policy meta tag allows nothing but this page's own script and style (by
   hash), media from the page's folder, and no network, forms or base URL: even a hole in 1 could
   not run a script or call out.
"""

from __future__ import annotations

import base64
import hashlib
import html
import re
from collections.abc import Iterable, Mapping
from functools import lru_cache
from importlib import resources
from typing import Any

from studyscribe.notes import labels
from studyscribe.notes.doc import NotesDoc, seek_seconds, timestamps
from studyscribe.notes.render_common import (
    appendix_rows,
    banners,
    code_flags,
    difficulty_label,
    footer_lines,
    header_rows,
)
from studyscribe.transcribe.export import AUDIO_FILE, clock

_BACKTICKS = re.compile(r"`([^`\n]+)`")


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def inline(text: str) -> str:
    """Escaped text with `backtick` spans as <code>."""
    return _BACKTICKS.sub(r"<code>\1</code>", esc(" ".join(text.split())))


def paragraphs(text: str) -> str:
    return "".join(f"<p>{inline(p)}</p>" for p in re.split(r"\n\s*\n", text.strip()) if p.strip())


@lru_cache(maxsize=1)
def _asset(name: str) -> str:
    return (resources.files("studyscribe.notes") / "resources" / "export" / name).read_text(
        encoding="utf-8")


def _hash(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode(
        "ascii") + "'"


def content_security_policy() -> str:
    # media-src: a page opened from disk has an opaque origin, where 'self' may not match.
    return (f"default-src 'none'; style-src {_hash(_asset('notes.css'))}; "
            f"script-src {_hash(_asset('notes.js'))}; media-src file: 'self'; "
            "base-uri 'none'; form-action 'none'")


def ts_link(ref: Mapping[str, Any]) -> str:
    seconds = seek_seconds(ref)
    return (f'<a class="ts" href="#t={seconds}" data-t="{seconds}" '
            f'data-ms="{int(ref["start_ms"])}" data-n="{int(ref["n"])}">'
            f"{esc(clock(int(ref['start_ms'])))}</a>")


def _refs(refs: Iterable[Mapping[str, Any]]) -> str:
    links = "".join(ts_link(ref) for ref in timestamps(refs))
    return f'<p class="refs">{links}</p>' if links else ""


def render_html(doc: NotesDoc) -> str:
    def say(key: str, **values: object) -> str:
        return labels.text(doc.language, key, **values)

    body: list[str] = []
    command = f"studyscribe export {doc.session_id} --audio"
    body.append(
        '<header class="player">'
        f'<audio id="player" controls preload="metadata" src="{esc(AUDIO_FILE)}" '
        f'aria-label="{esc(say("player"))}"></audio>'
        f'<p id="no-audio" class="warn"{"" if not doc.audio_present else " hidden"}>'
        f"{esc(say('no_audio', command=command))}</p>"
        f'<p id="audio-error" class="warn" hidden>{esc(say("audio_error"))}</p>'
        "</header>")
    body.append("<main>")
    body.append(f"<h1>{esc(doc.title)}</h1>")
    body.append('<ul class="meta">' + "".join(
        f"<li>{esc(label)}: {esc(value)}</li>" for label, value in header_rows(doc)) + "</ul>")
    for banner in banners(doc):
        links = "".join(
            f'<li><a class="ts" href="#t={seconds}" data-t="{seconds}">{esc(text)}</a></li>'
            for seconds, text in banner.links)
        body.append(f'<aside class="warn" data-kind="{esc(banner.kind)}"><p><strong>'
                    f'{esc(banner.text)}</strong></p>'
                    f'{f"<ul>{links}</ul>" if links else ""}</aside>')

    for section in doc.template.sections:
        value = doc.content.get(section.key)
        if not value:
            continue
        body.append(f'<section data-section="{esc(section.key)}">'
                    f"<h2>{esc(section.heading_for(doc.language))}</h2>")
        body.append(_section(doc, section.kind, value))
        body.append("</section>")

    body.append(_appendix(doc))
    footer = footer_lines(doc)
    if footer:
        body.append("<footer>" + "".join(f"<p>{esc(line)}</p>" for line in footer) + "</footer>")
    body.append("</main>")

    css, js = _asset("notes.css"), _asset("notes.js")
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="{esc(labels.table_language(doc.language))}">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta http-equiv="Content-Security-Policy" content="{esc(content_security_policy())}">\n'
        f"<title>{esc(doc.title)}</title>\n<style>{css}</style>\n</head>\n<body>\n"
        + "\n".join(body) + f"\n<script>{js}</script>\n</body>\n</html>\n")


def _section(doc: NotesDoc, kind: str, items: Any) -> str:
    if kind == "summary":
        return paragraphs(str(items))
    parts: list[str] = []
    if kind == "outline":
        parts.append("<ul>" + "".join(
            f"<li>{ts_link(i['ref'])} {inline(i['title'])}</li>" for i in items) + "</ul>")
    elif kind == "concepts":
        parts += [f'<article class="item"><h3>{inline(i["name"])}</h3>'
                  f"<p>{inline(i['explanation'])}</p>{_refs(i['refs'])}</article>"
                  for i in items]
    elif kind == "definitions":
        parts.append("<dl>" + "".join(
            f"<dt>{inline(i['term'])}</dt><dd>{inline(i['definition'])}{_refs(i['refs'])}</dd>"
            for i in items) + "</dl>")
    elif kind == "examples":
        parts += [_example(doc, i) for i in items]
    elif kind == "questions":
        parts += [
            f'<article class="item"><p><strong>{inline(i["question"])}</strong> '
            f'<span class="difficulty">{esc(difficulty_label(doc, i["difficulty"]))}</span></p>'
            f"{_refs(i['refs'])}<details><summary>{esc(labels.text(doc.language, 'show_answer'))}"
            f"</summary><p>{inline(i['answer'])}</p></details></article>" for i in items]
    else:  # open_points, bullets
        field = "point" if kind == "open_points" else "text"
        parts.append("<ul>" + "".join(
            f"<li>{inline(i[field])}{_refs(i['refs'])}</li>" for i in items) + "</ul>")
    return "".join(parts)


def _example(doc: NotesDoc, item: Mapping[str, Any]) -> str:
    flags = "".join(f' <span class="flag" data-kind="{esc(f.kind)}">{esc(f.text)}</span>'
                    for f in code_flags(doc, item))
    code = str(item.get("code", "")).strip("\n")
    block = f"<pre><code>{esc(code)}</code></pre>" if code.strip() else ""
    return (f'<article class="item"><p>{inline(item["explanation"])}{flags}</p>{block}'
            f"{_refs(item['refs'])}</article>")


def _appendix(doc: NotesDoc) -> str:
    def say(key: str) -> str:
        return esc(labels.text(doc.language, key))

    out = [f'<section data-section="appendix"><h2>{say("appendix_title")}</h2>',
           f"<p>{say('appendix_intro')}</p>"]
    rows = appendix_rows(doc)
    if not rows:
        return "".join(out) + f"<p><em>{say('appendix_empty')}</em></p></section>"
    out.append(f'<table><thead><tr><th scope="col">{say("col_word")}</th>'
               f'<th scope="col">{say("col_line")}</th><th scope="col">{say("col_time")}</th>'
               f'<th scope="col">{say("col_probability")}</th></tr></thead><tbody>')
    for row in rows:
        line = esc(row.line)
        if row.before is not None:
            text = (f"{esc(row.before)}<mark>{esc(row.match)}</mark>{esc(row.after)}"
                    if row.match else esc(row.before))
            line += f": {text}"
        out.append(
            f"<tr><td>{esc(row.word)}</td><td>{line}</td>"
            f'<td><a class="ts" href="#t={row.seconds}" data-t="{row.seconds}">'
            f"{esc(row.time)}</a></td><td>{esc(row.probability)}</td></tr>")
    return "".join(out) + "</tbody></table></section>"
