import json
import re

from studyscribe.notes import prompts
from studyscribe.notes.prompts import PartNotes, map_prompt, prompt_hash, reduce_prompt
from studyscribe.notes.schema import schema_text
from studyscribe.notes.templates import builtin_template, parse_template
from studyscribe.notes.transcript import Line

STUDIO = builtin_template()
LINES = [Line(41, 41, 754_000, 760_000, "So var is one of them."),
         Line(42, 42, 761_000, 770_000, "Then let our name equals free code camp.")]
TARGETS = STUDIO.targets(13 * 60_000, whole_lecture=False)


def studio_map(**kwargs):
    args = dict(part=2, parts=15, targets=TARGETS, title="Learn JavaScript",
                course="Web basics", language="en")
    args.update(kwargs)
    return map_prompt(STUDIO, LINES, **args)


def test_map_prompt_has_lecture_part_range_language_sections_and_schema():
    prompt = studio_map()
    system = prompt.system
    assert 'Lecture: "Learn JavaScript", course: "Web basics". Part 2 of 15, from 0:12:34 to ' \
           '0:12:50.' in system
    assert "Cite only S-numbers of this part (S41 to S42)" in system
    assert "Write in English." in system
    assert 'goes into\n  "examples"' in system
    assert '- "summary": ' in system and "About 3 sentences on this part, without refs." in system
    assert '- "concepts": ' in system and "About 5 items." in system
    assert '- "open_points": ' in system and "Only if there are any; up to about 1 item." \
        in system
    assert "Every one the lecturer reads aloud or names; empty if there are none." in system
    assert system.split('- "definitions": ')[1].splitlines()[0].endswith("About 3 items.")
    assert 'add an item to "open_points"' in system
    assert system.endswith(schema_text(STUDIO) + "\n")
    assert prompt.user == "[0:12:34]\nS41 So var is one of them.\nS42 Then let our name " \
                          "equals free code camp."


def test_map_prompt_has_no_corrections_no_example_and_no_acceptance_cases():
    system = studio_map().system.lower()
    for word in ("corrections?", "glossary", "example of", "detach", "tec", "torch", "y_pred",
                 "bush", "push", "scores"):
        assert not re.search(rf"\b{word}\b", system), word
    assert "keep the transcript's words" in system


def test_map_prompt_v5_1_rules():
    system = studio_map().system
    assert "Cover the whole part, from S41 to S42: every subject the lecturer talks about " \
           'gets a topic in "outline", the last minutes included.' in system
    assert "Never fix a word you think was misheard" in system
    assert 'you add an item to "open_points" that quotes it' in system
    assert 'goes into "concepts" or "definitions", even when a library has something' in system
    assert 'leave the\n    item out or add an item to "open_points";' in system
    assert "worked examples" not in system


def test_map_prompt_without_course_in_italian_and_without_code_or_open_points():
    template = parse_template(json.dumps({
        "id": "short", "title": {"en": "Short"},
        "sections": [{"key": "points", "kind": "bullets", "heading": {"en": "Points"},
                      "instructions": "Main points.", "per_hour": 10, "min": 1, "max": 5}]}))
    prompt = map_prompt(template, LINES, part=1, parts=1, targets={"points": 2},
                        title="Lezione", course=None, language="it")
    assert 'Lecture: "Lezione". Part 1 of 1' in prompt.system
    assert "Write in Italian." in prompt.system
    assert "- Code." not in prompt.system
    assert "leave it out" in prompt.system
    assert "{" not in prompt.system.split("Answer with JSON")[0]  # every placeholder filled


def test_reduce_prompt_final_and_partial():
    parts = [PartNotes(9_000, 750_000, {"summary": "Part one.", "outline": []}),
             PartNotes(754_000, 1_500_000, {"summary": "Part two, è così.", "outline": []})]
    targets = STUDIO.targets(25 * 60_000, whole_lecture=True)
    final = reduce_prompt(STUDIO, parts, final=True, targets=targets, title="T", course=None,
                          language="it")
    assert "into the final study notes of the whole lecture" in final.system
    assert "notes of 2 consecutive parts, in order, from 0:00:09 to 0:25:00" in final.system
    assert '"summary" is written anew from the summaries in the input: the thread of the ' \
           'whole lecture' in final.system
    assert "About 5 sentences on the whole lecture" in final.system
    examples_line = final.system.split('- "examples": ')[1].splitlines()[0]
    assert examples_line.endswith("About 10 items.")
    data = json.loads(final.user)
    assert data[1] == {"part": 2, "from": "0:12:34", "to": "0:25:00",
                       "notes": {"summary": "Part two, è così.", "outline": []}}
    assert "è così" in final.user  # not escaped: fewer tokens
    partial = reduce_prompt(STUDIO, parts, final=False, targets=targets, title="T",
                            course=None, language="en")
    assert "into the notes of one longer part of the lecture" in partial.system
    assert "the thread of these parts" in partial.system
    assert not re.search(r"\{[a-z_]+\}", partial.system)


def test_prompt_hash_is_stable_and_follows_the_files(monkeypatch):
    first = prompt_hash()
    assert re.fullmatch(r"[0-9a-f]{12}", first) and prompt_hash() == first
    original = prompts._file

    def changed(name):
        return original(name) + ("x" if name == "reduce.md" else "")

    monkeypatch.setattr(prompts, "_file", changed)
    assert prompt_hash() != first
