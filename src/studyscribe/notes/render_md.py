"""notes.md: the notes as Markdown, with timestamps that open the HTML page at that second.

The text of the notes comes from an LLM and the transcript: it is escaped (`&`, `<`, `>`, `[`,
`]`) so it cannot become HTML or a link in a viewer that allows raw HTML. Backticks are left
alone (models use them for code). Code goes in fenced blocks whose fence is longer than any
backtick run inside. A review question's answer is hidden in a `<details>` element, which GitHub,
VS Code and Obsidian render; in a plain-text editor the answer is simply readable in the source.
Sections with nothing in them are left out.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
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
from studyscribe.transcribe.export import clock


def md_text(text: str) -> str:
    """`text` made safe to put in Markdown as plain text."""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("[", "\\[").replace("]", "\\]"))


def one_line(text: str) -> str:
    """Escaped and on one line (list items and table cells must not span lines)."""
    return md_text(" ".join(text.split()))


def spaced(text: str) -> str:
    """Escaped with its whitespace collapsed but not trimmed (pieces of one line)."""
    return md_text(re.sub(r"\s+", " ", text))


def fence_for(code: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", code)), default=0)
    return "`" * max(3, longest + 1)


def render_markdown(doc: NotesDoc, html_name: str = "notes.html") -> str:
    def link(seconds: int, label: str) -> str:
        return f"[{label}]({html_name}#t={seconds})"

    def links(refs: Iterable[Mapping[str, Any]]) -> str:
        return " ".join(link(seek_seconds(r), clock(int(r["start_ms"])))
                        for r in timestamps(refs))

    def tail(refs: Iterable[Mapping[str, Any]]) -> str:
        shown = links(refs)
        return f" {shown}" if shown else ""

    out: list[str] = [f"# {one_line(doc.title)}", ""]
    out += [f"- {one_line(label)}: {one_line(value)}" for label, value in header_rows(doc)]
    out.append("")
    for banner in banners(doc):
        out.append(f"> **{one_line(banner.text)}**")
        for seconds, text in banner.links:
            out.append(f"> - {link(seconds, one_line(text))}")
        out.append("")

    for section in doc.template.sections:
        value = doc.content.get(section.key)
        if not value:
            continue
        out += [f"## {one_line(section.heading_for(doc.language))}", ""]
        if section.kind == "summary":
            out += [md_text(str(value).strip()), ""]
            continue
        for number, item in enumerate(value, start=1):
            refs = item.get("refs", [item["ref"]] if "ref" in item else [])
            if section.kind == "outline":
                out += [f"- {link(seek_seconds(item['ref']), clock(int(item['ref']['start_ms'])))}"
                        f" {one_line(item['title'])}"]
            elif section.kind == "concepts":
                out += [f"- **{one_line(item['name'])}**: {one_line(item['explanation'])}"
                        f"{tail(refs)}"]
            elif section.kind == "definitions":
                out += [f"- **{one_line(item['term'])}**: {one_line(item['definition'])}"
                        f"{tail(refs)}"]
            elif section.kind == "examples":
                out += _example(doc, item, tail(refs))
            elif section.kind == "questions":
                out += [
                    f"{number}. **{one_line(item['question'])}** "
                    f"*({one_line(difficulty_label(doc, item['difficulty']))})*{tail(refs)}",
                    "",
                    f"   <details><summary>{md_text(labels.text(doc.language, 'show_answer'))}"
                    "</summary>", "",
                    f"   {one_line(item['answer'])}", "",
                    "   </details>", ""]
            elif section.kind == "open_points":
                out += [f"- {one_line(item['point'])}{tail(refs)}"]
            else:  # bullets
                out += [f"- {one_line(item['text'])}{tail(refs)}"]
        if section.kind in ("outline", "concepts", "definitions", "open_points", "bullets"):
            out.append("")

    out += _appendix(doc, html_name)
    footer = footer_lines(doc)
    if footer:
        out += ["---", ""] + [f"*{one_line(line)}*  " for line in footer] + [""]
    return "\n".join(out).rstrip() + "\n"


def _example(doc: NotesDoc, item: Mapping[str, Any], tail: str) -> list[str]:
    flags = "".join(f" **({one_line(flag.text)})**" for flag in code_flags(doc, item))
    lines = [f"{one_line(item['explanation'])}{tail}{flags}", ""]
    code = str(item.get("code", "")).strip("\n")
    if code.strip():
        fence = fence_for(code)
        lines += [fence, code, fence, ""]
    return lines


def _appendix(doc: NotesDoc, html_name: str) -> list[str]:
    def say(key: str) -> str:
        return md_text(labels.text(doc.language, key))

    out = [f"## {say('appendix_title')}", "", say("appendix_intro"), ""]
    rows = appendix_rows(doc)
    if not rows:
        return out + [f"*{say('appendix_empty')}*", ""]
    out += [f"| {say('col_word')} | {say('col_line')} | {say('col_time')} | "
            f"{say('col_probability')} |", "|---|---|---|---|"]
    for row in rows:
        line = row.line
        if row.before is not None:
            text = (f"{spaced(row.before)}**{spaced(row.match)}**{spaced(row.after)}"
                    if row.match else one_line(row.before))
            line = f"{row.line}: {text}"
        out.append(f"| {_cell(one_line(row.word))} | {_cell(line)} | "
                   f"[{row.time}]({html_name}#t={row.seconds}) | {row.probability} |")
    return out + [""]


def _cell(text: str) -> str:
    return text.replace("|", "\\|")
