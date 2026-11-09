import random

import pytest

from studyscribe.notes.providers.base import (
    DailyQuotaExhausted,
    InvalidKey,
    NetworkError,
    RateLimited,
    ServerError,
)
from studyscribe.notes.providers.retry import Pacer, RetryPolicy, backoff_delay, call_with_retry


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def failing(errors, result="ok"):
    """A call that raises the given errors in turn, then returns `result`."""
    errors = list(errors)
    calls = []

    def fn():
        calls.append(1)
        if errors:
            raise errors.pop(0)
        return result

    fn.calls = calls
    return fn


NO_JITTER = RetryPolicy(jitter=0.0)


def test_backoff_doubles_up_to_the_cap():
    rng = random.Random(1)
    delays = [backoff_delay(n, NO_JITTER, rng) for n in range(1, 9)]
    assert delays == [2, 4, 8, 16, 32, 64, 120, 120]


def test_jitter_stays_within_bounds():
    rng = random.Random(7)
    policy = RetryPolicy(jitter=0.25)
    for _ in range(100):
        assert 3.0 <= backoff_delay(2, policy, rng) <= 5.0


def test_retryable_errors_are_retried_with_backoff():
    clock = Clock()
    fn = failing([NetworkError("down"), ServerError("529")])
    waits = []
    assert call_with_retry(fn, policy=NO_JITTER, sleep=clock.sleep,
                           on_wait=lambda reason, s: waits.append((reason, s))) == "ok"
    assert len(fn.calls) == 3
    assert clock.slept == [2.0, 4.0]
    assert waits == [("NetworkError", 2.0), ("ServerError", 4.0)]


def test_server_suggested_delay_wins_when_longer():
    clock = Clock()
    fn = failing([RateLimited("slow down", retry_after_s=37.0)])
    call_with_retry(fn, policy=NO_JITTER, sleep=clock.sleep)
    assert clock.slept == [37.0]


def test_suggested_delay_too_long_is_not_waited_for():
    clock = Clock()
    fn = failing([RateLimited("slow down", retry_after_s=3600.0)])
    with pytest.raises(RateLimited):
        call_with_retry(fn, policy=NO_JITTER, sleep=clock.sleep)
    assert clock.slept == []


@pytest.mark.parametrize("error", [DailyQuotaExhausted("daily"), InvalidKey("bad key")])
def test_non_retryable_errors_are_raised_at_once(error):
    clock = Clock()
    fn = failing([error])
    seen = []
    with pytest.raises(type(error)):
        call_with_retry(fn, sleep=clock.sleep, on_error=lambda e, n: seen.append((e, n)))
    assert len(fn.calls) == 1 and clock.slept == []
    assert seen == [(error, 1)]


def test_attempts_run_out():
    clock = Clock()
    fn = failing([NetworkError("down")] * 10)
    with pytest.raises(NetworkError):
        call_with_retry(fn, policy=RetryPolicy(max_attempts=3, jitter=0.0), sleep=clock.sleep)
    assert len(fn.calls) == 3
    assert clock.slept == [2.0, 4.0]


def test_pacer_keeps_five_requests_per_minute():
    clock = Clock()
    pacer = Pacer(rpm=5, clock=clock, sleep=clock.sleep)
    start = clock.now
    for _ in range(5):
        assert pacer.wait() == 0
        clock.now += 1  # each request takes 1 s
    # The 6th must wait until the 1st is 60 s old.
    assert pacer.wait() == pytest.approx(55.0)
    assert clock.now == pytest.approx(start + 60.0)
    # The 7th waits for the 2nd (sent at start + 1).
    assert pacer.wait() == pytest.approx(1.0)


def test_pacer_keeps_tokens_per_minute():
    clock = Clock()
    pacer = Pacer(tpm=250_000, clock=clock, sleep=clock.sleep)
    assert pacer.wait(200_000) == 0
    clock.now += 10
    # 200k + 100k > 250k: wait until the first request leaves the window.
    assert pacer.wait(100_000) == pytest.approx(50.0)
    clock.now += 1
    assert pacer.wait(100_000) == 0  # the first one has left the window: 100k + 100k
    assert pacer.delay_for(100_000) == pytest.approx(59.0)  # 300k: wait for the 2nd to leave


def test_call_with_retry_paces_every_attempt():
    clock = Clock()
    pacer = Pacer(rpm=1, clock=clock, sleep=clock.sleep)
    fn = failing([ServerError("503")])
    waits = []
    call_with_retry(fn, policy=NO_JITTER, pacer=pacer, sleep=clock.sleep,
                    on_wait=lambda reason, s: waits.append((reason, round(s, 3))))
    # First attempt immediately; backoff 2 s; then the pacer waits the rest of the minute.
    assert waits == [("ServerError", 2.0), ("pacing", 58.0)]
