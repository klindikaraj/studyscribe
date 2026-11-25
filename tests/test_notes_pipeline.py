import json
import re

import pytest

from fakes import FakeLlm, echo_notes, input_refs, studio_notes, targeted_notes

from studyscribe.notes.blocks import Block, cut_blocks
from studyscribe.notes.pipeline import (
    InvalidAnswer,
    Lecture,
    NotesError,
    PipelineOptions,
    count_items,
    coverage_gap,
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
    # the final reduce gets at once the whole room reserved for a long answer (piece 5b)
    assert first_map.max_output_tokens == 6144 and reduce.max_output_tokens == 16384
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


def test_a_truncated_answer_is_stored_and_the_retry_announced():
    progress = []

    def responder(request, index):
        if index == 0:
            raise Truncated("cut", usage=Usage(100, 6144), text='{"summary": "hal')
        return echo_notes(request, index)

    _, _, calls, _ = run(responder, on_progress=progress.append)
    assert (calls[0].status, calls[0].output_json) == ("error", '{"summary": "hal')
    assert "the answer was cut at 6,144 tokens: asking again with room for 12,288" in progress


@pytest.mark.parametrize("changes, why", [
    ({"max_input_tokens": lambda cap: 32768 - cap - 512 if cap < 12000 else 10},
     "context does not leave"),
    ({}, "twice the normal answer"),  # the retry was possible, and it was cut too
])
def test_a_run_that_stops_never_promises_another_try(changes, why):
    progress, calls = [], []

    def responder(request, index):
        raise Truncated("cut", text="{")

    lines = lecture_lines(24)
    with pytest.raises(Truncated) as info:
        generate_notes(FakeLlm(responder), STUDIO, lines, cut_blocks(lines), LECTURE,
                       options=options(**changes), on_call=calls.append,
                       on_progress=progress.append)
    message = info.value.user_message()
    assert why in message and "asks again" not in message
    # announced only when it was really sent
    assert sum("asking again" in p for p in progress) == len(calls) - 1
    assert all(c.output_json == "{" for c in calls)


def test_a_truncation_while_thinking_keeps_the_providers_remedy():
    def responder(request, index):
        raise Truncated("cut while still thinking", remedy="Use --effort none.")

    with pytest.raises(Truncated) as info:
        run(responder)
    assert info.value.remedy == "Use --effort none."


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


def test_final_targets_are_scaled_down_when_the_answer_would_not_fit():
    progress = []
    result, llm, calls, _ = run(targeted_notes(padding=600), minutes=75,
                                options=options(
                                    reduce_max_input=40_000,
                                    max_input_tokens=lambda cap: 65_536 - cap - 512),
                                on_progress=progress.append)
    scaled = result.stats["reduce"]["final_targets_scaled"]
    assert 0 < scaled < 1
    assert any("of the template's items" in p for p in progress)
    assert calls[-1].stage == "reduce" and calls[-1].usage.output_tokens <= 8192


def test_final_notes_that_cannot_fit_even_scaled_stop_before_any_reduce():
    calls = []
    lines = lecture_lines(75)
    with pytest.raises(NotesError, match="even with one item per section"):
        generate_notes(FakeLlm(targeted_notes(padding=600)), STUDIO, lines, cut_blocks(lines),
                       LECTURE, options=options(reduce_output_tokens=600), on_call=calls.append)
    assert {c.stage for c in calls} == {"map"}  # no reduce was sent


def test_parts_that_never_get_shorter_stop():
    def responder(request, index):
        refs = input_refs(request)
        if "longer part" in request.system:  # partial reduces copy every item back
            return json.dumps({"summary": "S.", "outline": [], "concepts": [
                {"name": f"c{i}", "explanation": "x" * 400, "refs": [refs[0]]}
                for i in range(12)], "definitions": [], "examples": [], "questions": [],
                "open_points": []})
        return targeted_notes(padding=300)(request, index)

    with pytest.raises(NotesError, match="do not get shorter"):
        run(responder, minutes=75, options=options(reduce_max_input=1500))


def test_reduce_groups_follow_counted_tokens_when_outputs_are_longer_than_expected():
    # 3 hours of blocks; every item is long (about 100 tokens instead of 55-69), so the map
    # answers are much larger than the estimate: the pipeline must make more partial reduces
    calls = []
    lines = lecture_lines(180)
    blocks = cut_blocks(lines)
    llm = FakeLlm(targeted_notes(padding=400))
    result = generate_notes(llm, STUDIO, lines, blocks, LECTURE, options=options(),
                            on_call=calls.append)
    stages = [c.stage for c in calls]
    assert stages.count("map") == len(blocks) == 12
    assert stages.count("partial") >= 3 and stages[-1] == "reduce"
    reduces = [r for r in llm.requests if r.user.startswith('[{"part"')]
    room = options().max_input_tokens(8192)
    assert all(r.input_tokens is not None and r.input_tokens <= room for r in reduces)
    assert all(r.input_tokens == llm.count_tokens(r.system, r.user) for r in reduces)
    assert result.stats["reduce"]["partial"] >= 3
    assert result.stats["reduce"]["tokens_per_item"] > 90


def test_parts_too_large_to_pair_are_shortened_not_stopped():
    # the first partial reduces write three times what they are asked: two of them no longer
    # fit one request, so they are shortened one by one and the run finishes
    progress = []
    result, llm, calls, _ = run(targeted_notes(padding=300, ignore_targets_for=4),
                                minutes=120, options=options(reduce_max_input=12_000),
                                on_progress=progress.append)
    assert result.stats["reduce"]["shortened"] >= 1
    assert any("(shortening one part)" in p for p in progress)
    assert [c.stage for c in calls][-1] == "reduce"


def test_partial_groups_stay_under_the_output_cap():
    # a reduce cap so small that only short groups may be merged at once
    calls = []
    lines = lecture_lines(120)
    llm = FakeLlm(targeted_notes(padding=100))
    generate_notes(llm, STUDIO, lines, cut_blocks(lines), LECTURE,
                   options=options(reduce_output_tokens=3000, reduce_max_input=6000),
                   on_call=calls.append)
    for call in calls:
        if call.stage in ("partial", "reduce"):
            assert call.usage.output_tokens <= 3000


# --- the real numbers of piece 5 (3h-copy, 2026-11-25) --------------------------------------------
#
# One reduce of all the maps: about 16,100 tokens in; the model writes 1.65 times the expected
# tokens (9,406), over the 8,192 cap; a retry at 16,384 leaves 15,872 tokens for the input.


def one_shot(system):
    """A final reduce that reads many maps at once, not the notes of a few partial reduces."""
    return ("into the final study notes" in system
            and int(re.search(r"the notes of (\d+) consecutive parts", system)[1]) > 6)


def piece5_notes(final_overshoot=1.02):
    def overshoot(request):
        if "longer part" in request.system:
            return 1.56
        if "into the final study notes" in request.system:
            return 1.7 if one_shot(request.system) else final_overshoot
        return 1.0

    return targeted_notes(padding=133, overshoot=overshoot, indent=3)


def run_piece5(responder, **changes):
    lines = lecture_lines(180)
    llm = FakeLlm(responder)
    calls, progress = [], []
    result = generate_notes(llm, STUDIO, lines, cut_blocks(lines), LECTURE,
                            options=options(**changes), on_call=calls.append,
                            on_progress=progress.append)
    return result, llm, calls, progress


def reduce_requests(llm):
    return [r for r in llm.requests if r.user.startswith('[{"part"')]


def test_piece5_fixture_has_the_real_sizes():
    # no limit on the input: one final reduce of all the maps, as in piece 5
    _, llm, calls, _ = run_piece5(piece5_notes(), max_input_tokens=lambda cap: 200_000,
                                  reduce_max_input=200_000, reduce_output_tokens=50_000)
    reduce = llm.requests[-1]
    maps = [json.loads(c.output_json) for c in calls if c.stage == "map"]
    items = sum(count_items(notes) for notes in maps)
    assert one_shot(reduce.system)
    assert llm.count_tokens(reduce.system, reduce.user) == pytest.approx(16_092, abs=300)
    assert calls[-1].usage.output_tokens == pytest.approx(9_406, abs=300)
    written = sum(c.usage.output_tokens for c in calls if c.stage == "map")
    assert written / items == pytest.approx(62.6, abs=3)  # as written, indentation included
    compact = sum(len(json.dumps(n, separators=(",", ":"))) // 4 for n in maps)
    assert compact / items == pytest.approx(46.2, abs=3)  # what was counted until piece 5


def test_piece5_three_hours_finish_with_the_default_settings():
    result, llm, calls, progress = run_piece5(piece5_notes())
    assert {c.status for c in calls} == {"ok"}
    assert result.stats["reduce"]["partial"] >= 2
    assert all(r.input_tokens <= 32768 - 16384 - 512 for r in reduce_requests(llm))
    assert "final_targets_scaled" not in result.stats["reduce"]  # asked in full
    assert not any("asking again" in p for p in progress)


def test_piece5_a_long_final_answer_is_not_cut():
    # the final reduce writes 9,400 tokens from the partial notes too
    _, llm, calls, progress = run_piece5(piece5_notes(final_overshoot=1.7))
    assert [c.status for c in calls if c.stage == "reduce"] == ["ok"]
    assert calls[-1].usage.output_tokens == pytest.approx(9_406, abs=300)
    assert llm.requests[-1].max_output_tokens == 16_384
    assert not any("asking again" in p for p in progress)


@pytest.mark.parametrize("num_ctx", [32_768, 65_536])  # with 65,536 a retry would even fit
def test_a_final_reduce_that_never_ends_is_asked_once(num_ctx):
    notes = piece5_notes()

    def responder(request, index):
        if "into the final study notes" in request.system:
            return '{"summary": "' + "x" * 80_000
        return notes(request, index)

    lines = lecture_lines(180)
    calls, progress = [], []
    with pytest.raises(Truncated) as info:
        generate_notes(FakeLlm(responder), STUDIO, lines, cut_blocks(lines), LECTURE,
                       options=options(max_input_tokens=lambda cap: num_ctx - cap - 512),
                       on_call=calls.append, on_progress=progress.append)
    final = [c for c in calls if c.stage == "reduce"]
    assert len(final) == 1 and len(final[0].output_json) == 16_384 * 4
    assert not any("asking again" in p for p in progress)
    assert info.value.remedy and "asks again" not in info.value.user_message()


def test_a_partial_reduce_cut_at_the_cap_is_asked_again_and_fits():
    notes = piece5_notes()
    cut = []

    def responder(request, index):
        if "longer part" in request.system and not cut:
            cut.append(index)
            raise Truncated("cut", text="{")
        return notes(request, index)

    _, llm, calls, progress = run_piece5(responder)
    partial = [r for r in llm.requests if "longer part" in r.system]
    assert [r.max_output_tokens for r in partial[:2]] == [8192, 16384]
    assert partial[0].user == partial[1].user
    assert any("asking again with room for 16,384" in p for p in progress)
    assert (calls[-1].stage, calls[-1].status) == ("reduce", "ok")


def test_partial_groups_leave_room_for_answers_twice_their_targets():
    # every partial reduce writes twice what it is asked: never cut
    notes = targeted_notes(padding=133, indent=3,
                           overshoot=lambda r: 2.0 if "longer part" in r.system else 1.0)
    _, _, calls, progress = run_piece5(notes)
    partial = [c for c in calls if c.stage == "partial"]
    assert partial and {c.status for c in partial} == {"ok"}
    assert max(c.usage.output_tokens for c in partial) <= 8192
    assert not any("asking again" in p for p in progress)


def test_a_larger_context_keeps_the_input_limit():
    _, llm, _, _ = run_piece5(piece5_notes(), max_input_tokens=lambda cap: 65_536 - cap - 512)
    assert all(llm.count_tokens("", r.user) <= 16_000 for r in reduce_requests(llm))


def test_thinking_still_runs_in_a_32k_context():
    # with thinking on the cap (16,384) is the reserve: doubling it would leave no input
    result, llm, _, _ = run_piece5(piece5_notes(), effort="high")
    assert {r.max_output_tokens for r in llm.requests} == {16_384}
    assert result.notes["outline"]


def test_the_tokens_per_item_are_those_of_the_answers_as_written():
    result, _, _, _ = run_piece5(piece5_notes())
    assert result.stats["reduce"]["tokens_per_item"] == pytest.approx(62.6, abs=3)


def test_resumed_maps_are_sized_like_fresh_ones():
    first, _, _, _ = run_piece5(piece5_notes())
    lines = lecture_lines(180)
    calls = []
    again = generate_notes(FakeLlm(piece5_notes()), STUDIO, lines, cut_blocks(lines), LECTURE,
                           options=options(), done_maps=first.map_answers, on_call=calls.append)
    assert "map" not in {c.stage for c in calls}
    assert again.stats["reduce"] == first.stats["reduce"]


@pytest.mark.parametrize("minutes", [24, 180])  # without and with partial reduces
def test_the_final_answer_is_not_counted(minutes):
    # nothing uses its size, and with Ollama each count takes seconds
    llm = FakeLlm(piece5_notes())
    count, sent = llm.count_tokens, []
    llm.count_tokens = lambda system, user: (sent.append(len(llm.requests)), count(system, user))[1]
    lines = lecture_lines(minutes)
    generate_notes(llm, STUDIO, lines, cut_blocks(lines), LECTURE, options=options())
    assert "into the final study notes" in llm.requests[-1].system
    assert max(sent) == len(llm.requests) - 1  # the last count is the final request's input


def test_count_items():
    assert count_items({"summary": "One. Two! Three?", "outline": [1, 2], "x": []}) == 5
    assert count_items({"summary": "no full stop"}) == 1
    assert count_items({"summary": ""}) == 0


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


def block_of(first, count, index=1):
    lines = tuple(Line(n, n, (n - first) * 10_000, (n - first) * 10_000 + 9_000, f"t{n}")
                  for n in range(first, first + count))
    return Block(index, lines)


def test_coverage_gap_none_when_the_last_cited_line_is_late_enough():
    block = block_of(101, 20)  # S101-S120; 85% = 17 lines, up to S117
    assert coverage_gap({"concepts": [{"refs": [101, 117]}]}, block) is None
    assert coverage_gap({"outline": [{"ref": 120}]}, block) is None


def test_coverage_gap_reports_the_uncovered_lines():
    block = block_of(101, 20, index=3)
    gap = coverage_gap({"summary": "s", "outline": [{"ref": 101}],
                        "concepts": [{"refs": [104, 116]}]}, block)
    assert (gap.block, gap.last_cited, gap.uncovered_from, gap.uncovered_to) == (3, 116, 117, 120)
    assert gap.start_ms == 160_000 and gap.end_ms == 199_000
    assert gap.describe() == ("block 3: the notes cite up to S116; lines S117-S120 "
                              "(0:02:40-0:03:19) are not covered")
    assert gap.as_dict()["uncovered"] == [117, 120]


def test_coverage_gap_when_nothing_is_cited():
    gap = coverage_gap({"summary": "s", "concepts": []}, block_of(1, 10))
    assert gap.last_cited is None and (gap.uncovered_from, gap.uncovered_to) == (1, 10)
    assert "cite up to nothing" in gap.describe()


def test_a_map_answer_that_stops_early_gets_a_warning_not_a_drop():
    progress = []

    def responder(request, index):
        refs = input_refs(request)
        return studio_notes(refs[: len(refs) // 2]) if index == 0 else echo_notes(request, index)

    result, _, calls, blocks = run(responder, on_progress=progress.append)
    assert [c.status for c in calls] == ["ok", "ok", "ok"]
    assert [w.block for w in result.coverage] == [1]
    assert result.stats["coverage_warnings"][0]["block"] == 1
    assert result.stats["coverage_warnings"][0]["uncovered"][1] == blocks[0].last_n
    assert any(p.startswith("warning: block 1: the notes cite up to S") for p in progress)


def test_map_targets_follow_the_block_duration_not_its_speech():
    # lines of 5 s every 10 s: a block of about 12 minutes with 6 of speech; concepts are 24
    # per hour, so 5 by duration (2 by speech)
    lines = [Line(n, n, (n - 1) * 10_000, (n - 1) * 10_000 + 5_000, f"t{n}")
             for n in range(1, 73)]
    llm = FakeLlm(echo_notes)
    generate_notes(llm, STUDIO, lines, cut_blocks(lines), LECTURE, options=options())
    assert "About 5 items." in llm.requests[0].system.split('- "concepts": ')[1].splitlines()[0]
