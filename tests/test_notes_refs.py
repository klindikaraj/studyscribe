import json

import pytest

from studyscribe.notes.refs import check_refs, refs_in, resolve
from studyscribe.notes.schema import output_model
from studyscribe.notes.templates import builtin_template
from studyscribe.notes.transcript import Line

MODEL = output_model(builtin_template())


def notes(**sections):
    data = {"summary": "", "outline": [], "concepts": [], "definitions": [], "examples": [],
            "questions": [], "open_points": []}
    data.update(sections)
    return MODEL.model_validate_json(json.dumps(data))


def concept(*refs, name="c"):
    return {"name": name, "explanation": "e", "refs": list(refs)}


def test_valid_refs_are_kept_and_deduplicated():
    checked, report = check_refs(notes(concepts=[concept(12, 10, 12)],
                                       outline=[{"title": "t", "ref": 10}]), range(10, 20))
    assert checked.concepts[0].refs == [12, 10]
    assert checked.outline[0].ref == 10
    assert report.refs == 4 and report.invalid == 0 and report.dropped_items == {}
    assert not report.too_many_invalid


def test_invalid_refs_are_dropped_and_items_without_one_too():
    original = notes(concepts=[concept(5, 12, name="partly outside"), concept(25, name="out"),
                               concept(name="no refs")],
                     outline=[{"title": "inside", "ref": 11}, {"title": "outside", "ref": 99}])
    checked, report = check_refs(original, range(10, 20))
    assert [c.name for c in checked.concepts] == ["partly outside"]
    assert checked.concepts[0].refs == [12]
    assert [t.title for t in checked.outline] == ["inside"]
    assert report.refs == 5 and report.invalid == 3
    assert report.dropped_items == {"concepts": 2, "outline": 1}
    assert report.too_many_invalid  # 3 of 5
    assert original.concepts[0].refs == [5, 12]  # the input is not changed


def test_the_quarter_rule_is_strict():
    _, report = check_refs(notes(concepts=[concept(1, 2, 3, 99)]), range(1, 10))
    assert report.invalid_ratio == 0.25 and not report.too_many_invalid


def test_allowed_can_be_a_predicate():
    checked, report = check_refs(notes(concepts=[concept(2, 3)]), lambda n: n % 2 == 0)
    assert checked.concepts[0].refs == [2] and report.invalid == 1


def test_bracketed_s_numbers_are_removed_from_text_and_all_are_counted():
    checked, report = check_refs(notes(
        summary="Variables come first (S12, S14).",
        concepts=[{"name": "var", "explanation": "A keyword [S3-S5] as S4 says.",
                   "refs": [3]}],
        examples=[{"code": "var a = 1", "explanation": "declares a (S3)", "refs": [3]}]),
        range(1, 20))
    assert checked.summary == "Variables come first."
    assert checked.concepts[0].explanation == "A keyword as S4 says."
    assert checked.examples[0].explanation == "declares a"
    assert report.s_numbers_in_text == 6  # 2 + 3 + 1


def test_report_add_and_as_dict():
    _, first = check_refs(notes(concepts=[concept(1, 99)]), range(1, 10))
    _, second = check_refs(notes(concepts=[concept(99)]), range(1, 10))
    first.add(second)
    assert first.as_dict() == {"refs": 3, "invalid_refs": 2,
                               "dropped_items": {"concepts": 1}, "s_numbers_in_text": 0}


def test_refs_in_collects_refs_of_plain_notes():
    plain = [notes(outline=[{"title": "t", "ref": 4}], concepts=[concept(5, 6)]).model_dump(),
             {"summary": "x", "questions": [{"question": "q", "answer": "a",
                                             "difficulty": "easy", "refs": [9]}]}]
    assert refs_in(plain) == {4, 5, 6, 9}


def test_resolve_maps_s_numbers_to_segments_and_times():
    lines = {n: Line(n, 100 + n, n * 1000, n * 1000 + 900, f"line {n}") for n in range(1, 6)}
    plain = notes(summary="s", outline=[{"title": "t", "ref": 2}],
                  concepts=[concept(3, 4)]).model_dump()
    resolved = resolve(plain, lines)
    assert resolved["summary"] == "s"
    assert resolved["outline"][0]["ref"] == {"n": 2, "segment_id": 102, "start_ms": 2000}
    assert [r["start_ms"] for r in resolved["concepts"][0]["refs"]] == [3000, 4000]
    assert plain["outline"][0]["ref"] == 2  # the input is not changed
    with pytest.raises(KeyError):
        resolve({"concepts": [concept(42)]}, lines)
