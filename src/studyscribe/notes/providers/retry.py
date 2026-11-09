"""Client-side pacing and retries with backoff (pure: clock, sleep and randomness injected).

SDK retries are turned off in both providers, so every retry goes through here: one place that
honours the server's suggested delay, reports each wait, and never retries an error that a wait
cannot fix (daily quota, invalid key, spend limit, refusal, truncated output).
"""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from studyscribe.notes.providers.base import LlmError

T = TypeVar("T")
WINDOW_S = 60.0


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6  # including the first one
    base_delay_s: float = 2.0  # backoff: base * 2^(attempt-1), with jitter
    max_delay_s: float = 120.0
    jitter: float = 0.25  # +- fraction of the delay
    max_server_delay_s: float = 600.0  # a suggested wait longer than this is not waited for


class Pacer:
    """Keeps requests under a requests-per-minute and a tokens-per-minute limit (sliding
    60 s window). Limits of None mean no pacing."""

    def __init__(self, rpm: float | None = None, tpm: float | None = None, *,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[tuple[float, int]] = deque()  # (time, tokens)

    def delay_for(self, tokens: int = 0) -> float:
        """Seconds to wait before a request of `tokens` input tokens may be sent."""
        now = self._clock()
        while self._sent and now - self._sent[0][0] >= WINDOW_S:
            self._sent.popleft()
        delay = 0.0
        if self.rpm is not None and len(self._sent) >= self.rpm:
            oldest = self._sent[len(self._sent) - int(self.rpm)][0]
            delay = max(delay, oldest + WINDOW_S - now)
        if self.tpm is not None and tokens:
            used = sum(t for _, t in self._sent)
            if used + tokens > self.tpm:
                # Wait until enough of the window has expired.
                excess = used + tokens - self.tpm
                for sent_at, sent_tokens in self._sent:
                    excess -= sent_tokens
                    if excess <= 0:
                        delay = max(delay, sent_at + WINDOW_S - now)
                        break
        return max(0.0, delay)

    def wait(self, tokens: int = 0, on_wait: Callable[[float], None] | None = None) -> float:
        delay = self.delay_for(tokens)
        if delay > 0:
            if on_wait:
                on_wait(delay)
            self._sleep(delay)
        self._sent.append((self._clock(), tokens))
        return delay


def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    """Delay after failed attempt number `attempt` (1-based)."""
    delay = min(policy.max_delay_s, policy.base_delay_s * 2 ** (attempt - 1))
    return max(0.0, delay * (1 + rng.uniform(-policy.jitter, policy.jitter)))


def call_with_retry(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy = RetryPolicy(),
    pacer: Pacer | None = None,
    tokens: int = 0,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    on_wait: Callable[[str, float], None] | None = None,
    on_error: Callable[[LlmError, int], None] | None = None,
) -> T:
    """Call `fn` (one provider request), pacing it and retrying retryable LlmErrors.

    `on_wait(reason, seconds)` is told about every wait; `on_error(error, attempt)` about every
    failed attempt (so a caller can record failed requests that were billed). The last error is
    raised when the attempts run out or the error is not retryable."""
    rng = rng or random.Random()
    attempt = 0
    while True:
        attempt += 1
        if pacer is not None:
            pacer.wait(tokens, (lambda s: on_wait("pacing", s)) if on_wait else None)
        try:
            return fn()
        except LlmError as error:
            if on_error:
                on_error(error, attempt)
            if not error.retryable or attempt >= policy.max_attempts:
                raise
            delay = backoff_delay(attempt, policy, rng)
            if error.retry_after_s is not None:
                if error.retry_after_s > policy.max_server_delay_s:
                    raise
                delay = max(delay, error.retry_after_s)
            if on_wait:
                on_wait(type(error).__name__, delay)
            sleep(delay)
