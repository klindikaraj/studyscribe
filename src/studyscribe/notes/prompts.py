"""The prompts of the notes pipeline: map (one block of transcript) and reduce (partial or final).

The wording lives in `resources/prompts/*.md`; this module fills it in from the template (one
line per section with its target), the lecture and the part. Every generation stores
`PROMPT_VERSION` and `prompt_hash()`, so notes can be traced to the exact wording.

The prompts have no example: in the probe the 9B model copied the made-up example into the
notes. The acceptance test's cases ("TEC", torch.nn, y_pred) never appear here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

from studyscribe.notes.schema import schema_text
from studyscribe.notes.templates import Section, Template
from studyscribe.notes.transcript import Line, format_lines
from studyscribe.transcribe.export import clock

# v1-v4: the probe's drafts (scripts/llm_probe.py). v5: v4 without corrections (2026-11-16).
# v5.1 (2026-11-16, after the v4/v5 probe): "about N" for every section but the open points,
# v4's code rule in bullets, a rule against prose as code, an explicit no-fix rule, coverage of
# the whole part.
PROMPT_VERSION = "v5.1"
PROMPT_FILES = ("map.md", "code-rule.md", "reduce.md")
LANGUAGE_NAMES = {"en": "English", "it": "Italian", "fr": "French", "de": "German",
                  "es": "Spanish", "pt": "Portuguese"}


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str


@dataclass(frozen=True)
class PartNotes:
    """Validated notes of one part (a block, or a group already reduced), as reduce input."""

    start_ms: int
    end_ms: int
    notes: Mapping[str, Any]  # the output model dumped to plain data


def map_prompt(template: Template, lines: Sequence[Line], *, part: int, parts: int,
               targets: Mapping[str, int], title: str, course: str | None,
               language: str) -> Prompt:
    examples = _first_of_kind(template, "examples")
    unclear = _unclear(template)
    code_rule = (_file("code-rule.md").format(key=examples.key, unclear=unclear,
                                              concept_keys=_concept_keys(template))
                 if examples else "")
    outline = _first_of_kind(template, "outline")
    coverage = (f'every subject the lecturer talks about gets a topic in "{outline.key}"'
                if outline else "the notes take in every subject the lecturer talks about")
    open_points = _first_of_kind(template, "open_points")
    unclear_quote = (f'add an item to "{open_points.key}" that quotes it' if open_points
                     else "leave it as it is")
    system = _file("map.md").format(
        lecture=lecture_line(title, course), part=part, parts=parts,
        start=clock(lines[0].start_ms), end=clock(lines[-1].end_ms),
        first=lines[0].label, last=lines[-1].label, unclear=unclear, coverage=coverage,
        unclear_quote=unclear_quote, language=language_name(language), code_rule=code_rule,
        sections=section_lines(template, targets, scope="this part", all_code=True),
        schema=schema_text(template))
    return Prompt(system=system, user=format_lines(lines))


def reduce_prompt(template: Template, inputs: Sequence[PartNotes], *, final: bool,
                  targets: Mapping[str, int], title: str, course: str | None,
                  language: str) -> Prompt:
    """`final`: the notes of the whole lecture; otherwise a partial reduce, whose output is
    again the notes of one (longer) part."""
    scope = ("the final study notes of the whole lecture" if final
             else "the notes of one longer part of the lecture")
    summary = _first_of_kind(template, "summary")
    summary_rule = (f'"{summary.key}" is written anew from the summaries in the input: the thread '
                    f"of {'the whole lecture' if final else 'these parts'}, without refs."
                    if summary else "Keep the items in the order the lecturer says them.")
    system = _file("reduce.md").format(
        scope=scope, lecture=lecture_line(title, course), count=len(inputs),
        start=clock(inputs[0].start_ms), end=clock(inputs[-1].end_ms),
        language=language_name(language), summary_rule=summary_rule,
        sections=section_lines(template, targets,
                               scope="the whole lecture" if final else "these parts together"),
        schema=schema_text(template))
    user = json.dumps(
        [{"part": i, "from": clock(p.start_ms), "to": clock(p.end_ms), "notes": p.notes}
         for i, p in enumerate(inputs, start=1)],
        ensure_ascii=False, separators=(",", ":"))
    return Prompt(system=system, user=user)


def section_lines(template: Template, targets: Mapping[str, int], *, scope: str,
                  all_code: bool = False) -> str:
    """One line per section with its target. `all_code` (map): the code section has no target,
    since the map is asked for every code element the lecturer says; the reduce sizes it."""
    return "\n".join(_section_line(s, targets[s.key], scope, all_code)
                     for s in template.sections)


def _section_line(section: Section, target: int, scope: str, all_code: bool) -> str:
    text = f'- "{section.key}": {section.instructions}'
    if section.kind == "summary":
        return f"{text} About {_count(max(1, target), 'sentence')} on {scope}, without refs."
    if section.kind == "examples" and all_code:
        return f"{text} Every one the lecturer reads aloud or names; empty if there are none."
    if section.kind == "open_points" or target == 0:
        return f"{text} Only if there are any; up to about {_count(max(1, target), 'item')}."
    return f"{text} About {_count(target, 'item')}."


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def lecture_line(title: str, course: str | None) -> str:
    return f'Lecture: "{title}"' + (f', course: "{course}"' if course else "")


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def prompt_hash() -> str:
    """First 12 hex digits of the SHA-256 of the prompt files, in a fixed order."""
    digest = hashlib.sha256()
    for name in PROMPT_FILES:
        digest.update(name.encode() + b"\0" + _file(name).encode("utf-8") + b"\0")
    return digest.hexdigest()[:12]


def _unclear(template: Template) -> str:
    open_points = _first_of_kind(template, "open_points")
    return (f'add an item to "{open_points.key}"' if open_points
            else "leave it out")


def _concept_keys(template: Template) -> str:
    keys = [f'"{s.key}"' for s in template.sections if s.kind in ("concepts", "definitions")]
    return " or ".join(keys) if keys else "the other sections"


def _first_of_kind(template: Template, kind: str) -> Section | None:
    return next((s for s in template.sections if s.kind == kind), None)


@cache
def _file(name: str) -> str:
    path = resources.files("studyscribe.notes") / "resources" / "prompts" / name
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")
