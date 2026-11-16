import json

import pytest

from studyscribe.notes.templates import (
    SECTION_KINDS,
    Template,
    TemplateError,
    builtin_template,
    builtin_templates,
    load_template_file,
    parse_template,
)

HOUR = 3_600_000


def minimal(**changes):
    data = {"id": "mine", "title": {"en": "Mine"},
            "sections": [{"key": "points", "kind": "bullets", "heading": {"en": "Points"},
                          "instructions": "The main points.", "per_hour": 10, "min": 2,
                          "max": 12}]}
    data.update(changes)
    return data


def test_studio_is_built_in_with_seven_sections_and_both_languages():
    templates = builtin_templates()
    assert "studio" in templates
    studio = builtin_template()
    assert [s.kind for s in studio.sections] == [
        "summary", "outline", "concepts", "definitions", "examples", "questions", "open_points"]
    assert all(set(s.heading) == {"en", "it"} for s in studio.sections)
    assert studio.title_for("it") == "Studio"
    assert studio.sections[6].heading_for("it") == "Punti poco chiari"
    assert studio.sections[6].heading_for("de") == "Unclear points"


def test_targets_for_the_whole_lecture_are_clamped():
    studio = builtin_template()
    three_hours = studio.targets(int(2.77 * HOUR), whole_lecture=True)
    assert three_hours["summary"] == 8  # 12 per hour, max 8
    assert three_hours["concepts"] == 20  # 66, max 20 (lowered so the final reduce fits 8k)
    assert three_hours["questions"] == 10
    twenty_minutes = studio.targets(int(0.29 * HOUR), whole_lecture=True)
    assert twenty_minutes["summary"] == 5  # 3.5, min 5
    assert twenty_minutes["questions"] == 5
    assert twenty_minutes["open_points"] == 1


def test_targets_for_a_part_have_no_minimum_beyond_one():
    studio = builtin_template()
    block = studio.targets(13 * 60_000, whole_lecture=False)
    assert block["summary"] == 3  # 2.6
    assert block["concepts"] == 5
    assert block["questions"] == 2  # not the lecture's minimum of 5
    tiny = studio.targets(60_000, whole_lecture=False)
    assert tiny["outline"] == 1  # min >= 1: at least one
    assert tiny["open_points"] == 0  # min 0: may be empty


def test_parse_and_load_a_user_template(tmp_path):
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(minimal()), encoding="utf-8")
    template = load_template_file(path)
    assert isinstance(template, Template) and template.sections[0].kind == "bullets"
    assert parse_template(template.to_json()) == template


@pytest.mark.parametrize("data, message", [
    (minimal(id="Bad Id"), "id"),
    (minimal(extra=1), "extra"),
    (minimal(sections=[]), "sections"),
    (minimal(sections=[{**minimal()["sections"][0], "kind": "poem"}]), "kind"),
    (minimal(sections=[{**minimal()["sections"][0], "min": 5, "max": 3}]), "min 5 is above"),
    (minimal(sections=[{**minimal()["sections"][0], "heading": {"it": "Punti"}}]), "'en'"),
    (minimal(sections=[minimal()["sections"][0]] * 2), "unique"),
])
def test_invalid_templates_are_refused_with_a_readable_message(data, message):
    with pytest.raises(TemplateError, match="not a valid template") as error:
        parse_template(json.dumps(data), "mine.json")
    assert message in str(error.value)


def test_unknown_builtin_and_unreadable_file(tmp_path):
    with pytest.raises(TemplateError, match="Unknown template 'nope'"):
        builtin_template("nope")
    with pytest.raises(TemplateError, match="Cannot read"):
        load_template_file(tmp_path / "missing.json")


def test_kinds():
    assert "open_points" in SECTION_KINDS and len(SECTION_KINDS) == 8
