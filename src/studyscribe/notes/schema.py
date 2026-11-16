"""The shape of the model's output: one pydantic model per section kind, and the output model of
a template built from its sections.

The map (one block), the partial reduces and the final reduce all answer with the same model:
only the targets in the prompt change. References are S-numbers (`ref` for a topic, `refs` for
the other items); whether they exist and fall in the right range is checked by refs.py, not here.
Here a response fails only when it is not the right JSON, or when code contains an S-number
(S-numbers belong in refs: the model is asked again).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, create_model, field_validator

from studyscribe.notes.templates import Template

S_NUMBER = re.compile(r"\bS\d+\b")


def _ref(value: Any) -> Any:
    """Accept "S12" and "12" as well as 12: small models sometimes quote the label."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in ("S", "s"):
            text = text[1:]
        if text.isdigit():
            return int(text)
    return value


Ref = Annotated[int, BeforeValidator(_ref)]
REFS_DESCRIPTION = ("S-numbers of the transcript lines this item comes from, the line where it "
                    "starts first")


class _Item(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Topic(_Item):
    title: str
    ref: Ref = Field(description="S-number of the line where the topic starts")


class Concept(_Item):
    name: str
    explanation: str
    refs: list[Ref] = Field(description=REFS_DESCRIPTION)


class Definition(_Item):
    term: str
    definition: str = Field(description="the lecturer's wording, spoken fillers removed")
    refs: list[Ref] = Field(description=REFS_DESCRIPTION)


class Example(_Item):
    code: str = Field(description="code as it would appear in a source file; empty for an "
                      "example without code; never an S-number")
    explanation: str = Field(description="what the code or the example shows, in one sentence")
    refs: list[Ref] = Field(description=REFS_DESCRIPTION)

    @field_validator("code")
    @classmethod
    def _no_s_numbers(cls, value: str) -> str:
        if S_NUMBER.search(value):
            raise ValueError("code must not contain S-numbers: they go only in refs")
        return value


class Question(_Item):
    question: str
    answer: str = Field(description="the answer, from the lecture")
    difficulty: Literal["easy", "medium", "hard"]
    refs: list[Ref] = Field(description=REFS_DESCRIPTION)


class OpenPoint(_Item):
    point: str
    refs: list[Ref] = Field(description=REFS_DESCRIPTION)


class Bullet(_Item):
    text: str
    refs: list[Ref] = Field(description=REFS_DESCRIPTION)


ITEM_MODELS: dict[str, type[_Item]] = {
    "outline": Topic,
    "concepts": Concept,
    "definitions": Definition,
    "examples": Example,
    "questions": Question,
    "open_points": OpenPoint,
    "bullets": Bullet,
}  # "summary" is a plain string


def output_model(template: Template) -> type[BaseModel]:
    """The output model of a template: one field per section, in the template's order."""
    return _output_model(template.to_json())


@lru_cache(maxsize=16)
def _output_model(template_json: str) -> type[BaseModel]:
    template = Template.model_validate_json(template_json)
    fields: dict[str, Any] = {}
    for section in template.sections:
        if section.kind == "summary":
            fields[section.key] = (str, Field(description=section.instructions))
        else:
            fields[section.key] = (list[ITEM_MODELS[section.kind]],
                                   Field(description=section.instructions))
    name = "Notes_" + re.sub(r"\W", "_", template.id)
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


def output_schema(template: Template) -> dict[str, Any]:
    return output_model(template).model_json_schema()


def schema_text(template: Template) -> str:
    """The schema as compact JSON, for the prompt (the docs recommend repeating it there)."""
    return json.dumps(output_schema(template), separators=(",", ":"), ensure_ascii=False)


def refs_of(item: BaseModel) -> list[int]:
    """An item's references: a topic's `ref`, or the `refs` of the other kinds."""
    if isinstance(item, Topic):
        return [item.ref]
    return list(getattr(item, "refs", []))
