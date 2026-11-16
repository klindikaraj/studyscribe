import json

import pytest
from pydantic import ValidationError

from studyscribe.notes.schema import (
    Concept,
    Topic,
    output_model,
    output_schema,
    refs_of,
    schema_text,
)
from studyscribe.notes.templates import builtin_template, parse_template


def studio_answer(**changes):
    data = {
        "summary": "The lecture explains variables.",
        "outline": [{"title": "Variables", "ref": 3}],
        "concepts": [{"name": "Variable", "explanation": "A label for data.", "refs": [3, 4]}],
        "definitions": [{"term": "var", "definition": "a keyword", "refs": [5]}],
        "examples": [{"code": "var myName = \"Beau\"", "explanation": "declares", "refs": [6]},
                     {"code": "", "explanation": "a box you fill", "refs": [4]}],
        "questions": [{"question": "What is var?", "answer": "A keyword.",
                       "difficulty": "easy", "refs": [5]}],
        "open_points": [],
    }
    data.update(changes)
    return json.dumps(data)


def test_studio_output_model_parses_an_answer():
    model = output_model(builtin_template())
    notes = model.model_validate_json(studio_answer())
    assert list(type(notes).model_fields) == [
        "summary", "outline", "concepts", "definitions", "examples", "questions", "open_points"]
    assert notes.outline[0].ref == 3
    assert notes.examples[1].code == ""


def test_the_model_is_built_once_per_template():
    assert output_model(builtin_template()) is output_model(builtin_template())


def test_refs_written_as_labels_are_accepted():
    model = output_model(builtin_template())
    notes = model.model_validate_json(studio_answer(
        outline=[{"title": "Variables", "ref": "S3"}],
        concepts=[{"name": "x", "explanation": "y", "refs": ["S4", "5", 6]}]))
    assert notes.outline[0].ref == 3 and notes.concepts[0].refs == [4, 5, 6]


@pytest.mark.parametrize("changes", [
    {"outline": [{"title": "Variables", "ref": "start"}]},
    {"questions": [{"question": "q", "answer": "a", "difficulty": "trivial", "refs": [1]}]},
    {"concepts": [{"name": "x", "explanation": "y", "refs": [1], "extra": True}]},
    {"examples": [{"code": "print(S12)", "explanation": "e", "refs": [12]}]},
    {"summary": ["a", "list"]},
])
def test_invalid_answers_fail(changes):
    with pytest.raises(ValidationError):
        output_model(builtin_template()).model_validate_json(studio_answer(**changes))


def test_missing_section_fails():
    data = json.loads(studio_answer())
    del data["open_points"]
    with pytest.raises(ValidationError):
        output_model(builtin_template()).model_validate(data)


def test_schema_has_every_section_and_forbids_extra_fields():
    schema = output_schema(builtin_template())
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["summary"]["type"] == "string"
    text = schema_text(builtin_template())
    assert json.loads(text) == schema and '{"$defs":' in text  # compact JSON


def test_a_template_with_bullets():
    template = parse_template(json.dumps({
        "id": "short", "title": {"en": "Short"},
        "sections": [{"key": "points", "kind": "bullets", "heading": {"en": "Points"},
                      "instructions": "Main points.", "per_hour": 10, "min": 1, "max": 5}]}))
    notes = output_model(template).model_validate_json(
        '{"points": [{"text": "one", "refs": [1]}]}')
    assert notes.points[0].text == "one"


def test_refs_of():
    assert refs_of(Topic(title="t", ref=7)) == [7]
    assert refs_of(Concept(name="n", explanation="e", refs=[1, 2])) == [1, 2]
