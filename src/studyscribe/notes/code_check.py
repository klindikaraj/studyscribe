"""Graded check of the names in a code item against the transcript lines it cites.

A warning, never a drop: code read aloud is written in code form ("ipsilon pred" -> `y_pred`,
"PyTorch.neuralNetwork" -> `torch.nn`), so a right name often does not appear literally in the
transcript. The check tells a code form of words that were said from a name that has nothing
to do with them (`y_prediatti` for "y previsti", `misure_notte` for "i due tensori", code
rebuilt from a description).

Every name (string literals and language keywords left out) is split into its words: snake_case,
camelCase, dots, digits. A word of 3+ letters is found when a word of the cited lines (±1 line)
contains it, is at least 80% similar to it, or has it as initials of consecutive words ("mean
squared error" -> `MSE`). Shorter words (`y`, `nn`, `lr`) are not checked. A name is found when
all its checked words are found there; partly found when some are, or when its words are found
elsewhere in the lecture (the same tests on the whole transcript, initials left out: often the
name was said, but the item cites the wrong lines); missing when none is found anywhere. Levels:

- `ok`: every name found (or nothing checkable);
- `check`: some name only partly found ("to verify");
- `unsupported`: some name with no word found ("name not in the transcript, to verify").

Known blind spots: an invented name close to a spoken word passes (`inputi` for "input"); a
right name never said (an alias only shown on screen, `plt`) gets a warning, and so does an
English name for something the lecturer says in another language ("ottimizzatore" for
`optimizer`), which is the usual case in an Italian lecture.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from studyscribe.notes.transcript import Line

Level = Literal["ok", "check", "unsupported"]
LEVELS: tuple[Level, ...] = ("ok", "check", "unsupported")
MIN_PART = 3
SIMILAR = 0.8
CONTEXT_LINES = 1
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STRING_LITERAL = re.compile(r"\"[^\"]*\"|'[^']*'|`[^`]*`")
NAME_PARTS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
TRANSCRIPT_WORD = re.compile(r"[^\W\d_]+|\d+")
# Keywords and builtins of the languages most courses teach; not names a lecturer must say.
KEYWORDS = frozenset("""
var let const function return if else elif for while do break continue switch case default
new delete typeof instanceof void this super class extends import export from as def self
lambda yield async await try catch finally except raise throw with pass global nonlocal in of
is and or not true false null undefined none nan print console log len range int float str
bool list dict set tuple
""".split())


@dataclass(frozen=True)
class CodeCheck:
    level: Level
    partial: tuple[str, ...] = ()  # names partly found
    missing: tuple[str, ...] = ()  # names with no word found

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "partial": list(self.partial),
                "missing": list(self.missing)}


def vocabulary(texts: Iterable[str]) -> set[str]:
    """The distinct lowercase words of some transcript text."""
    return {w.lower() for text in texts for w in TRANSCRIPT_WORD.findall(text)}


def check_code(code: str, cited: Sequence[str],
               lecture: Collection[str] = frozenset()) -> CodeCheck:
    """`cited`: the text of the cited lines ±1; `lecture`: the vocabulary of the whole
    transcript (see `vocabulary`), for names said elsewhere."""
    lines = [[w.lower() for w in TRANSCRIPT_WORD.findall(text)] for text in cited]
    near = {w for line in lines for w in line}
    partial: list[str] = []
    missing: list[str] = []
    for name in dict.fromkeys(IDENTIFIER.findall(STRING_LITERAL.sub(" ", code))):
        if name.lower() in KEYWORDS:
            continue
        parts = [p.lower() for p in NAME_PARTS.findall(name.replace("_", " "))
                 if len(p) >= MIN_PART and not p.isdigit()]
        if not parts:
            continue
        found = sum(_found(p, near, lines) for p in parts)
        if found == len(parts):
            continue
        if found or any(_found(p, lecture, ()) for p in parts):
            partial.append(name)
        else:
            missing.append(name)
    level: Level = "unsupported" if missing else "check" if partial else "ok"
    return CodeCheck(level, tuple(partial), tuple(missing))


def cited_texts(refs: Sequence[int], lines: Mapping[int, Line],
                context: int = CONTEXT_LINES) -> list[str]:
    """The text of the cited lines and `context` lines on each side, in order, once each."""
    wanted = sorted({r + d for r in refs for d in range(-context, context + 1)})
    return [lines[n].text for n in wanted if n in lines]


def annotate_code(notes: Mapping[str, Any], code_keys: Sequence[str],
                  lines: Mapping[int, Line]) -> tuple[dict[str, Any], dict[str, int]]:
    """Plain notes (S-numbers not resolved yet) with a "check" added to every code item of the
    `code_keys` sections, and the count of items per level. Items with empty code get none."""
    out = dict(notes)
    counts = dict.fromkeys(LEVELS, 0)
    lecture = vocabulary(line.text for line in lines.values())
    for key in code_keys:
        items = []
        for item in notes.get(key, []):
            item = dict(item)
            if item.get("code", "").strip():
                result = check_code(item["code"], cited_texts(item.get("refs", []), lines),
                                    lecture)
                item["check"] = result.as_dict()
                counts[result.level] += 1
            items.append(item)
        out[key] = items
    return out, counts


def _found(part: str, words: Collection[str], lines: Sequence[Sequence[str]]) -> bool:
    if any(part in word or SequenceMatcher(None, part, word).ratio() >= SIMILAR
           for word in words):
        return True
    size = len(part)
    return any("".join(w[0] for w in line[i:i + size]) == part
               for line in lines for i in range(len(line) - size + 1))
