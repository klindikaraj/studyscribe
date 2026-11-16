import json

import pytest

from fakes import FakeLlm, echo_notes, input_refs, studio_notes

from studyscribe.notes.blocks import cut_blocks
from studyscribe.notes.pipeline import (
    InvalidAnswer,
    Lecture,
    NotesError,
    PipelineOptions,
    generate_notes,
)
from studyscribe.notes.providers.base import BadRequest, ServerError, Truncated, Usage
from studyscribe.notes.providers.retry import RetryPolicy
from studyscribe.notes.templates import builtin_template
from studyscribe.notes.transcript import Line, Word

STUDIO = builtin_template()
LECTURE = Lecture(title="Lezione", course="PyTorch", language="it")


def lecture_lines(minutes, step_s=10):
    lines = []
    for n in range(1, minutes * 60 // step_s + 1):
        start = (n - 1) * step_s * 1000
        text = "stacchiamoli col metodo di TEC" if n == 5 else f"frase numero {n}"
        words = tuple(Word(start + i * 100, start + (i + 1) * 100, " " + w,
                           0.35 if w == "TEC" else 0.95) for i, w in enumerate(text.split()))
        lines.append(Line(n, 1000 + n, start, start + step_s * 1000 - 500, text, words))
    return lines


def options(**changes):
    base = dict(sleep=lambda s: None, retry_policy=RetryPolicy(max_attempts=3))
    base.update(changes)
    return PipelineOptions(**base)


def run(responder, minutes=24, **kwargs):
    lines = lecture_lines(minutes)
    blocks = cut_blocks(lines)
    llm = FakeLlm(responder)
    calls = []
    kwargs.setdefault("options", options())
    result = generate_notes(llm, STUDIO, lines, blocks, LECTURE, on_call=calls.append, **kwargs)
    return result, llm, calls, blocks


def test_two_blocks_map_then_final_reduce():
    result, llm, calls, blocks = run(echo_notes)
    assert len(blocks) == 2
    assert [(c.stage, c.block, c.attempt, c.status) for c in calls] == [
        ("map", 1, 1, "ok"), ("map", 2, 1, "ok"), ("reduce", 1, 1, "ok")]
    assert all(isinstance(c.usage, Usage) and c.output_json for c in calls)
    first_map, second_map, reduce = llm.requests
    assert first_map.user.startswith("[0:00:00]\nS1 frase numero 1")
    assert "Part 2 of 2" in second_map.system
    assert reduce.user.startswith('[{"part":1,')
    assert first_map.max_output_tokens == 6144 and reduce.max_output_tokens == 8192
    assert first_map.effort == "none"
    assert set(result.map_answers) == {1, 2}
    assert result.stats["requests"] == 3 and result.stats["resumed_blocks"] == 0


def test_result_has_resolved_refs_code_checks_and_uncertain_words():
    result, *_ = run(echo_notes)
    topic = result.resolved["outline"][0]
    assert topic["ref"] == {"n": 1, "segment_id": 1001, "start_ms": 0}
    assert result.notes["outline"][0]["ref"] == 1
    assert result.notes["examples"][0]["check"]["level"] == "ok"  # "x" is not checked
    assert result.stats["code_checks"] == {"ok": 1, "check": 0, "unsupported": 0}
    assert [(w.word, w.n) for w in result.uncertain] == [("TEC", 5)]
    assert result.stats["uncertain_words"] == 1
    assert result.stats["refs"]["map"]["invalid_refs"] == 0


def test_invalid_json_is_asked_again_with_the_error():
    def responder(request, index):
        return "{not json" if index == 0 else echo_notes(request, index)

    result, llm, calls, _ = run(responder)
    assert [(c.stage, c.attempt, c.status) for c in calls[:2]] == [("map", 1, "invalid"),
                                                                    ("map", 2, "ok")]
    assert calls[0].error and "Invalid JSON" in calls[0].error
    assert "Your previous answer could not be used" in llm.requests[1].user
    assert llm.requests[1].system == llm.requests[0].system


def test_schema_errors_three_times_stop_the_generation():
    def responder(request, index):
        return json.dumps({"summary": "only"})

    with pytest.raises(InvalidAnswer, match="block 1 was still invalid after 3 attempts"):
        run(responder)


def test_code_with_an_s_number_is_asked_again():
    def responder(request, index):
        refs = input_refs(request)
        return studio_notes(refs, code="print(S3)" if index == 0 else "print(x)")

    _, _, calls, _ = run(responder)
    assert calls[0].status == "invalid" and "S-numbers" in calls[0].error


def test_a_few_refs_outside_the_block_are_dropped_and_counted():
    def responder(request, index):
        refs = input_refs(request)
        if index == 0:  # 1 invalid of 7 refs
            return studio_notes(refs, open_points=[{"point": "p", "refs": [9999]}])
        return echo_notes(request, index)

    result, _, calls, _ = run(responder)
    assert calls[0].status == "ok"
    assert result.stats["refs"]["map"]["invalid_refs"] == 1
    assert result.stats["refs"]["map"]["dropped_items"] == {"open_points": 1}


def test_too_many_refs_outside_the_block_ask_again():
    def responder(request, index):
        if index == 0:
            return studio_notes([9001, 9002, 9003])
        return echo_notes(request, index)

    _, llm, calls, _ = run(responder)
    assert calls[0].status == "invalid" and "references are not lines" in calls[0].error
    assert "Cite only the S-numbers given" in llm.requests[1].user


def test_the_reduce_may_cite_only_refs_of_its_input():
    def responder(request, index):
        if index == 2:  # the final reduce cites a line no map cited
            refs = input_refs(request)
            return studio_notes(refs, open_points=[{"point": "p", "refs": [refs[0] + 1]}])
        return echo_notes(request, index)

    result, *_ = run(responder)
    assert result.notes["open_points"] == []
    assert result.stats["refs"]["reduce"]["dropped_items"] == {"open_points": 1}


def test_truncated_once_is_retried_with_twice_the_cap():
    def responder(request, index):
        if index == 0:
            raise Truncated("cut", usage=Usage(100, 6144))
        return echo_notes(request, index)

    _, llm, calls, _ = run(responder)
    assert calls[0].status == "error" and calls[0].usage.output_tokens == 6144
    assert [r.max_output_tokens for r in llm.requests[:2]] == [6144, 12288]
    assert llm.requests[2].max_output_tokens == 6144  # the next block starts again from the cap


def test_truncated_twice_stops():
    def responder(request, index):
        raise Truncated("cut")

    with pytest.raises(Truncated):
        run(responder)


def test_truncated_without_room_for_a_larger_cap_stops():
    def responder(request, index):
        raise Truncated("cut")

    small = options(max_input_tokens=lambda cap: 32768 - cap - 512 if cap < 12000 else 10)
    with pytest.raises(Truncated):
        run(responder, options=small)


def test_transient_errors_are_retried_and_recorded():
    waits = []

    def responder(request, index):
        if index == 0:
            raise ServerError("overloaded")
        return echo_notes(request, index)

    _, _, calls, _ = run(responder, options=options(sleep=waits.append))
    assert [(c.attempt, c.status) for c in calls[:2]] == [(1, "error"), (2, "ok")]
    assert len(waits) == 1


def test_other_provider_errors_stop_and_are_recorded():
    calls = []

    def responder(request, index):
        raise BadRequest("unknown model")

    lines = lecture_lines(24)
    with pytest.raises(BadRequest):
        generate_notes(FakeLlm(responder), STUDIO, lines, cut_blocks(lines), LECTURE,
                       options=options(), on_call=calls.append)
    assert [(c.stage, c.status) for c in calls] == [("map", "error")]
    assert "unknown model" in calls[0].error


def test_long_lecture_uses_partial_reduces():
    progress = []
    result, llm, calls, blocks = run(echo_notes, minutes=75,
                                     options=options(reduce_max_input=400),
                                     on_progress=progress.append)
    stages = [c.stage for c in calls]
    assert stages.count("map") == len(blocks) == 5
    assert stages.count("partial") >= 2 and stages[-1] == "reduce"
    partial_numbers = [c.block for c in calls if c.stage == "partial"]
    assert partial_numbers == list(range(1, len(partial_numbers) + 1))
    first_partial = llm.requests[len(blocks)]
    assert "into the notes of one longer part of the lecture" in first_partial.system
    assert "final reduce" in progress and "partial reduce 1" in progress
    assert result.notes["outline"]


def test_partial_reduces_that_do_not_shrink_stop():
    big = "x" * 3000

    def responder(request, index):
        return studio_notes(input_refs(request), summary=big)

    with pytest.raises(NotesError, match="do not get shorter|too long"):
        run(responder, minutes=75, options=options(reduce_max_input=2500))


def test_resume_skips_blocks_already_accepted():
    first, _, _, _ = run(echo_notes)
    result, llm, calls, _ = run(echo_notes, done_maps={1: first.map_answers[1]})
    assert [(c.stage, c.block) for c in calls] == [("map", 2), ("reduce", 1)]
    assert result.stats["resumed_blocks"] == 1
    assert result.map_answers[1] == first.map_answers[1]


def test_resume_asks_again_for_a_stored_answer_that_no_longer_passes():
    _, _, calls, _ = run(echo_notes, done_maps={1: "{broken"})
    assert [(c.stage, c.block) for c in calls] == [("map", 1), ("map", 2), ("reduce", 1)]


def test_thinking_uses_the_larger_cap_and_passes_the_effort():
    _, llm, _, _ = run(echo_notes, options=options(effort="high"))
    assert {r.max_output_tokens for r in llm.requests} == {16384}
    assert {r.effort for r in llm.requests} == {"high"}


def test_no_blocks():
    with pytest.raises(NotesError, match="no lines"):
        generate_notes(FakeLlm(echo_notes), STUDIO, [], [], LECTURE)
