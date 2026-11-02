import numpy as np

from studyscribe.transcribe.windows import choose_cut

RATE = 16_000


def test_cut_in_the_middle_of_the_longest_pause() -> None:
    region = np.zeros(30 * RATE, dtype=np.float32)
    speech = [(0, 5 * RATE), (int(5.4 * RATE), 20 * RATE), (21 * RATE, 30 * RATE)]
    cut = choose_cut(region, 1_000_000, speech)
    assert cut.in_pause
    assert cut.frame == 1_000_000 + int(20.5 * RATE)


def test_edges_of_the_region_count_as_pauses() -> None:
    region = np.zeros(30 * RATE, dtype=np.float32)
    cut = choose_cut(region, 0, [(0, 27 * RATE)])
    assert cut.in_pause and cut.frame == int(28.5 * RATE)


def test_no_pause_falls_back_to_the_quietest_stretch() -> None:
    rng = np.random.default_rng(1)
    region = (rng.standard_normal(30 * RATE) * 0.1).astype(np.float32)
    region[12 * RATE : int(12.6 * RATE)] *= 0.01
    speech = [(0, 15 * RATE), (int(15.1 * RATE), 30 * RATE)]  # 100 ms gap: too short
    cut = choose_cut(region, 0, speech)
    assert not cut.in_pause
    assert 12 * RATE <= cut.frame <= int(12.6 * RATE)
