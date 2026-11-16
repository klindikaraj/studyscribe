"""Note templates: which sections the notes have, their headings and how long each one is.

A template is data (JSON), not code: `{id, title, description, sections}`. Each section has a
`key` (its field in the model's output), a `kind` (which item shape and rendering it uses), a
heading per language, instructions for the model, and a size: items per hour of speech (for a
summary: sentences), clamped to `min`..`max` for the whole lecture. A new template needs no code;
a new kind needs a schema and a rendering.

Built-in templates live in `resources/templates/*.json` inside the package. Storing them in the
database and importing user templates (`template import`) belong to the runner and the CLI
(piece 3).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SectionKind = Literal["summary", "outline", "concepts", "definitions", "examples", "questions",
                      "open_points", "bullets"]
SECTION_KINDS: tuple[str, ...] = SectionKind.__args__  # type: ignore[attr-defined]
DEFAULT_TEMPLATE = "studio"
FALLBACK_LANGUAGE = "en"
HOUR_MS = 3_600_000
PARTIAL_SLACK = 1.5


class TemplateError(ValueError):
    """A template file that cannot be used, with a message for the user."""


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    kind: SectionKind
    heading: dict[str, str]
    instructions: str = Field(min_length=1)
    per_hour: float = Field(gt=0)
    min: int = Field(ge=0)
    max: int = Field(ge=1)

    @model_validator(mode="after")
    def _check(self) -> Section:
        if self.min > self.max:
            raise ValueError(f"section {self.key!r}: min {self.min} is above max {self.max}")
        if FALLBACK_LANGUAGE not in self.heading:
            raise ValueError(f"section {self.key!r}: needs an {FALLBACK_LANGUAGE!r} heading")
        return self

    def heading_for(self, language: str | None) -> str:
        return self.heading.get(language or "", self.heading[FALLBACK_LANGUAGE])

    def target(self, speech_ms: int, *, whole_lecture: bool) -> int:
        """Items (sentences for a summary) for this much speech. The whole lecture is clamped
        to min..max; a part of it (a map block, a partial reduce) only to max, and to at least
        one item when the section must not be empty (min >= 1)."""
        wanted = round(self.per_hour * speech_ms / HOUR_MS)
        low = self.min if whole_lecture else min(1, self.min)
        return max(low, min(self.max, wanted))


class Template(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    title: dict[str, str]
    description: str = ""
    sections: tuple[Section, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Template:
        keys = [s.key for s in self.sections]
        if len(set(keys)) != len(keys):
            raise ValueError("section keys must be unique")
        if sum(s.kind == "summary" for s in self.sections) > 1:
            raise ValueError("at most one summary section")
        return self

    def title_for(self, language: str | None) -> str:
        return self.title.get(language or "", self.title.get(FALLBACK_LANGUAGE, self.id))

    def targets(self, speech_ms: int, *, whole_lecture: bool) -> dict[str, int]:
        return {s.key: s.target(speech_ms, whole_lecture=whole_lecture) for s in self.sections}

    def partial_targets(self, part_speech_ms: int, total_speech_ms: int,
                        slack: float = PARTIAL_SLACK) -> dict[str, int]:
        """Targets of a partial reduce: its share of the final notes' targets, times `slack`
        (room for the final reduce to choose and merge), never above its own part targets.
        Without the share, partial answers are as long as the final one and two of them no
        longer fit the final reduce's input (found by the estimate, 2026-11-16)."""
        final = self.targets(total_speech_ms, whole_lecture=True)
        share = part_speech_ms / total_speech_ms if total_speech_ms > 0 else 1.0
        out = {}
        for section in self.sections:
            low = min(1, section.min)
            wanted = max(low, round(final[section.key] * share * slack))
            out[section.key] = min(section.target(part_speech_ms, whole_lecture=False), wanted)
        return out

    def to_json(self) -> str:
        """The copy stored with every generation (a template can change afterwards)."""
        return self.model_dump_json()


def parse_template(text: str, source: str = "template") -> Template:
    try:
        return Template.model_validate_json(text)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'template'}: {e['msg']}"
            for e in error.errors())
        raise TemplateError(f"{source} is not a valid template: {problems}") from None


def load_template_file(path: Path) -> Template:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise TemplateError(f"Cannot read {path}: {error.strerror or error}") from None
    return parse_template(text, str(path))


def builtin_templates() -> dict[str, Template]:
    folder = resources.files("studyscribe.notes") / "resources" / "templates"
    found = {}
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".json"):
            template = parse_template(entry.read_text(encoding="utf-8"), entry.name)
            found[template.id] = template
    return found


def builtin_template(template_id: str = DEFAULT_TEMPLATE) -> Template:
    templates = builtin_templates()
    if template_id not in templates:
        raise TemplateError(f"Unknown template {template_id!r}. Built-in: "
                            f"{', '.join(sorted(templates))}.")
    return templates[template_id]
