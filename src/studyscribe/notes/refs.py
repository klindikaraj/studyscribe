"""Reference checks and resolution.

Every item cites transcript lines by S-number. A reference is valid when it is in the allowed
set: the block's lines for a map response, the references present in the input for a reduce
(so a reduce can never cite a line the map did not). Invalid references are dropped, and an
item left without a valid reference is dropped (a topic has a single `ref`, so an invalid one
drops it). Every drop is counted. When more than a quarter of a response's references are
invalid, the response is not trusted at all: the pipeline asks again.

Citations written inside text ("as said in (S12)") are removed when bracketed and counted
either way: S-numbers belong in refs only. Code with an S-number already fails validation in
schema.py.

Resolution turns the S-numbers of the final notes into the segment id and its start time, which
are what the stored notes keep (they stay readable after a re-transcription).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from studyscribe.notes.schema import S_NUMBER, Topic
from studyscribe.notes.transcript import Line

MAX_INVALID_RATIO = 0.25
_BRACKETED = re.compile(r"\s*[(\[]\s*S\d+(?:\s*[,;\-–]\s*S?\d+)*\s*[)\]]")


@dataclass
class RefReport:
    refs: int = 0  # references given by the model
    invalid: int = 0  # of which not allowed (do not exist, or outside the block / input)
    dropped_items: dict[str, int] = field(default_factory=dict)  # section key -> items dropped
    s_numbers_in_text: int = 0  # S-numbers written inside text fields

    @property
    def invalid_ratio(self) -> float:
        return self.invalid / self.refs if self.refs else 0.0

    @property
    def too_many_invalid(self) -> bool:
        return self.invalid_ratio > MAX_INVALID_RATIO

    def add(self, other: RefReport) -> None:
        self.refs += other.refs
        self.invalid += other.invalid
        self.s_numbers_in_text += other.s_numbers_in_text
        for key, count in other.dropped_items.items():
            self.dropped_items[key] = self.dropped_items.get(key, 0) + count

    def as_dict(self) -> dict[str, Any]:
        return {"refs": self.refs, "invalid_refs": self.invalid,
                "dropped_items": dict(self.dropped_items),
                "s_numbers_in_text": self.s_numbers_in_text}


def check_refs(notes: BaseModel, allowed: Collection[int] | Callable[[int], bool]
               ) -> tuple[BaseModel, RefReport]:
    """A copy of `notes` with invalid references and the items left without one removed, and
    bracketed S-numbers removed from text; the report of what was dropped."""
    is_allowed = allowed if callable(allowed) else allowed.__contains__
    report = RefReport()
    update: dict[str, Any] = {}
    for key, value in notes:
        if isinstance(value, str):
            update[key] = _clean_text(value, report)
            continue
        kept = []
        for item in value:
            item = _clean_item(item, report)
            if isinstance(item, Topic):
                report.refs += 1
                if is_allowed(item.ref):
                    kept.append(item)
                else:
                    report.invalid += 1
                continue
            refs = list(item.refs)
            report.refs += len(refs)
            valid = list(dict.fromkeys(r for r in refs if is_allowed(r)))
            report.invalid += sum(1 for r in refs if not is_allowed(r))
            if valid:
                kept.append(item.model_copy(update={"refs": valid}))
        if len(kept) < len(value):
            report.dropped_items[key] = len(value) - len(kept)
        update[key] = kept
    return notes.model_copy(update=update), report


def refs_in(notes: Iterable[Mapping[str, Any]]) -> set[int]:
    """Every S-number cited in plain (dumped) notes: what a reduce may cite."""
    found: set[int] = set()
    for one in notes:
        for value in one.values():
            if isinstance(value, list):
                for item in value:
                    if "ref" in item:
                        found.add(int(item["ref"]))
                    found.update(int(r) for r in item.get("refs", ()))
    return found


def resolve(notes: Mapping[str, Any], lines: Mapping[int, Line]) -> dict[str, Any]:
    """Plain notes with every S-number replaced by {"n", "segment_id", "start_ms"}. Every
    reference must exist (they were checked); a missing one is a bug and raises KeyError."""
    def one(n: int) -> dict[str, int]:
        line = lines[int(n)]
        return {"n": line.n, "segment_id": line.segment_id, "start_ms": line.start_ms}

    out: dict[str, Any] = {}
    for key, value in notes.items():
        if not isinstance(value, list):
            out[key] = value
            continue
        items = []
        for item in value:
            item = dict(item)
            if "ref" in item:
                item["ref"] = one(item["ref"])
            if "refs" in item:
                item["refs"] = [one(r) for r in item["refs"]]
            items.append(item)
        out[key] = items
    return out


def _clean_item(item: BaseModel, report: RefReport) -> BaseModel:
    update = {}
    for name, value in item:
        if isinstance(value, str) and name != "code":
            cleaned = _clean_text(value, report)
            if cleaned != value:
                update[name] = cleaned
    return item.model_copy(update=update) if update else item


def _clean_text(text: str, report: RefReport) -> str:
    report.s_numbers_in_text += len(S_NUMBER.findall(text))
    return _BRACKETED.sub("", text)
