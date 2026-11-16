import pytest

from studyscribe.notes.estimate import Assumptions, estimate_notes

PLAIN = Assumptions(chars_per_token=4.0, map_output_tokens=2000, partial_output_tokens=3000,
                    reduce_output_tokens=5000, thinking_tokens=7000, retry_margin=0.0)


def test_short_lecture_two_maps_and_one_reduce():
    estimate = estimate_notes([20_000, 12_000], 4_000, 16_000, thinking=False, assumptions=PLAIN)
    assert [r.stage for r in estimate.requests] == ["map", "map", "reduce"]
    assert [r.input_tokens for r in estimate.requests] == [5_000, 3_000, 1_000 + 4_000]
    assert estimate.input_tokens == 13_000 and estimate.output_tokens == 9_000
    assert estimate.count("map") == 2 and estimate.count() == 3


def test_long_lecture_gets_partial_reduces():
    estimate = estimate_notes([24_000] * 14, 4_000, 16_000, thinking=False, assumptions=PLAIN)
    # 14 x 2000 = 28k > 16k: two partials of 7 (14k each), then the final on 2 x 3000
    assert estimate.count("map") == 14 and estimate.count("partial") == 2
    assert estimate.count("reduce") == 1
    assert estimate.requests[-1].input_tokens == 1_000 + 6_000


def test_thinking_adds_tokens_to_every_request():
    off = estimate_notes([20_000, 12_000], 4_000, 16_000, thinking=False, assumptions=PLAIN)
    on = estimate_notes([20_000, 12_000], 4_000, 16_000, thinking=True, assumptions=PLAIN)
    assert on.output_tokens - off.output_tokens == 3 * 7_000


def test_retry_margin_time_and_cost():
    estimate = estimate_notes([20_000, 12_000], 4_000, 16_000, thinking=False,
                              assumptions=Assumptions(chars_per_token=4.0, retry_margin=0.15,
                                                      map_output_tokens=2000,
                                                      reduce_output_tokens=5000))
    assert estimate.input_tokens == round(13_000 * 1.15)
    assert estimate.seconds(1000, 50) == pytest.approx(13_000 * 1.15 / 1000 + 9_000 * 1.15 / 50)
    charged, listed = estimate.cost("gemini", "gemini-3.5-flash-lite", free_tier=True)
    assert charged == 0.0
    assert listed == pytest.approx((13_000 * 0.30 + 9_000 * 2.50) / 1e6 * 1.15)
    assert estimate.cost("ollama", "qwen3.5:9b", free_tier=False) == (0.0, 0.0)
    assert estimate.cost("anthropic", "unknown-model", free_tier=False) is None


def test_no_blocks_no_requests():
    assert estimate_notes([], 4_000, 16_000, thinking=False).requests == ()
