import json
import math

import pytest

from studyscribe.notes.estimate import Assumptions, answer_tokens, estimate_notes
from studyscribe.notes.templates import builtin_template, parse_template

MIN = 60_000
# one section of plain items: 10 per hour, so 6 minutes of speech = 1 item
ONE = parse_template(json.dumps({
    "id": "one", "title": {"en": "One"},
    "sections": [{"key": "points", "kind": "bullets", "heading": {"en": "Points"},
                  "instructions": "Points.", "per_hour": 10, "min": 1, "max": 100}]}))
PLAIN = Assumptions(chars_per_token=4.0, grouping_chars_per_token=4.0, tokens_per_item=100,
                    answer_overhead_tokens=0, thinking_tokens=7000, retry_margin=0.0)


def test_answer_tokens_follow_the_template_targets():
    assert answer_tokens(ONE, 30 * MIN, whole_lecture=False, assumptions=PLAIN) == 500
    studio = builtin_template()
    block = answer_tokens(studio, 13 * MIN, whole_lecture=False)
    assert 900 < block < 1600  # measured 860-1,518 on 15-minute blocks


def test_short_lecture_two_maps_and_one_reduce():
    estimate = estimate_notes(ONE, [12 * MIN, 6 * MIN], [20_000, 12_000], 4_000, 16_000,
                              thinking=False, assumptions=PLAIN)
    assert [r.stage for r in estimate.requests] == ["map", "map", "reduce"]
    assert [r.input_tokens for r in estimate.requests] == [5_000, 3_000, 1_000 + 300]
    assert [r.output_tokens for r in estimate.requests] == [200, 100, 300]
    assert estimate.count("map") == 2 and estimate.count() == 3


def test_long_lecture_gets_partial_reduces_sized_from_their_speech():
    # 14 blocks of 30 minutes: 500 tokens each, 7,000 in all, budget 2,000 -> 4 partials
    # (4, 4, 4 and 2 blocks), then the final reduce on their answers
    estimate = estimate_notes(ONE, [30 * MIN] * 14, [24_000] * 14, 4_000, 2_000,
                              thinking=False, assumptions=PLAIN)
    assert estimate.count("map") == 14 and estimate.count("partial") == 4
    partial = [r for r in estimate.requests if r.stage == "partial"]
    assert [r.output_tokens for r in partial] == [2000, 2000, 2000, 1000]
    final = estimate.requests[-1]
    assert final.stage == "reduce" and final.output_tokens == 7_000
    assert estimate.largest_output("reduce") == 7_000


def test_grouping_uses_the_pipelines_conservative_size():
    loose = estimate_notes(ONE, [30 * MIN] * 4, [1_000] * 4, 400, 2_000, thinking=False,
                           assumptions=PLAIN)
    tight = estimate_notes(ONE, [30 * MIN] * 4, [1_000] * 4, 400, 2_000, thinking=False,
                           assumptions=Assumptions(chars_per_token=4.0,
                                                   grouping_chars_per_token=2.0,
                                                   tokens_per_item=100,
                                                   answer_overhead_tokens=0))
    assert loose.count("partial") == 0 and tight.count("partial") > 0


def test_thinking_adds_tokens_to_every_request():
    off = estimate_notes(ONE, [12 * MIN] * 2, [20_000] * 2, 4_000, 16_000, thinking=False,
                         assumptions=PLAIN)
    on = estimate_notes(ONE, [12 * MIN] * 2, [20_000] * 2, 4_000, 16_000, thinking=True,
                        assumptions=PLAIN)
    assert on.output_tokens - off.output_tokens == 3 * 7_000


def test_retry_margin_time_and_cost():
    a = Assumptions(chars_per_token=4.0, tokens_per_item=100, answer_overhead_tokens=0,
                    retry_margin=0.15)
    estimate = estimate_notes(ONE, [12 * MIN, 6 * MIN], [20_000, 12_000], 4_000, 16_000,
                              thinking=False, assumptions=a)
    assert estimate.input_tokens == math.ceil(9_300 * 1.15)
    assert estimate.output_tokens == math.ceil(600 * 1.15)
    assert estimate.seconds(1000, 50) == pytest.approx(
        estimate.input_tokens / 1000 + estimate.output_tokens / 50)
    charged, listed = estimate.cost("gemini", "gemini-3.5-flash-lite", free_tier=True)
    assert charged == 0.0
    assert listed == pytest.approx((9_300 * 0.30 + 600 * 2.50) / 1e6 * 1.15)
    assert estimate.cost("ollama", "qwen3.5:9b", free_tier=False) == (0.0, 0.0)
    assert estimate.cost("anthropic", "unknown-model", free_tier=False) is None


def test_no_blocks_no_requests_and_mismatched_inputs():
    assert estimate_notes(ONE, [], [], 4_000, 16_000, thinking=False).requests == ()
    with pytest.raises(ValueError):
        estimate_notes(ONE, [MIN], [], 4_000, 16_000, thinking=False)

