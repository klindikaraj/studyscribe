"""Provider-neutral interface for one structured LLM request, its usage and its errors.

A provider sends text only (system instructions plus a user message) and asks for JSON that
follows a JSON schema. It returns the raw JSON text: validation belongs to the caller, so a
response that fails validation is still recorded with its usage (it is paid for).

Every error raised by a provider is an `LlmError` with a message and a remedy meant for the
user, never an SDK exception, and never containing an API key.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

# "none" turns thinking off; only the local provider (Ollama) accepts it.
Effort = Literal["none", "low", "medium", "high"]
EFFORTS: tuple[Effort, ...] = ("none", "low", "medium", "high")
PROVIDERS = ("ollama", "gemini", "anthropic")
KEY_PROVIDERS = ("gemini", "anthropic")  # the providers that need an API key


@dataclass(frozen=True)
class LlmRequest:
    system: str
    user: str
    schema: dict  # JSON schema of the expected output (e.g. Model.model_json_schema())
    max_output_tokens: int
    effort: Effort = "medium"
    # Exact input tokens when the caller counted them (count_tokens); a provider that guards
    # its context uses this instead of its own estimate from the characters.
    input_tokens: int | None = None


@dataclass(frozen=True)
class Usage:
    """Tokens of one request, normalised across providers.

    `input_tokens` counts every input token, cached ones included (Haiku's price threshold is
    on that total). `output_tokens` includes thinking tokens, which are billed as output.
    """

    input_tokens: int
    output_tokens: int
    thinking_tokens: int | None = None
    cached_tokens: int = 0  # input tokens read from a cache (part of input_tokens)
    cache_write_tokens: int = 0  # input tokens written to a cache (part of input_tokens)


@dataclass(frozen=True)
class Timings:
    """Where the time of a local request went (Ollama reports it; the cloud providers do not)."""

    load_s: float  # loading the model, 0 when it was already loaded
    prefill_s: float  # reading the prompt
    generate_s: float  # writing the answer

    def prefill_rate(self, tokens: int) -> float | None:
        return tokens / self.prefill_s if self.prefill_s > 0 and tokens else None

    def generate_rate(self, tokens: int) -> float | None:
        return tokens / self.generate_s if self.generate_s > 0 and tokens else None


@dataclass(frozen=True)
class LlmResponse:
    text: str
    usage: Usage
    model_version: str | None
    request_id: str | None
    duration_s: float
    timings: Timings | None = None


class LlmProvider(Protocol):
    name: str  # one of PROVIDERS
    model: str

    def generate(self, request: LlmRequest) -> LlmResponse:
        """One request, no retries (see retry.py). Raises LlmError."""

    def count_tokens(self, system: str, user: str) -> int:
        """Exact input tokens of a request with this model (free, needs the key)."""

    def check(self) -> str:
        """Check the key and the model name without generating anything; returns the model's
        display name or ID."""

    def close(self) -> None:
        """Release what the provider holds (Ollama: unload the model from GPU memory). Never
        raises; called in a `finally`."""


# --- errors --------------------------------------------------------------------------------------


class LlmError(Exception):
    """An error with a message and a remedy for the user. `retryable` errors are retried by
    retry.py; `usage` is set when the provider billed the failed request."""

    retryable = False

    def __init__(self, message: str, *, remedy: str = "", retry_after_s: float | None = None,
                 usage: Usage | None = None, status: int | None = None) -> None:
        self.message = redact(message)
        self.remedy = remedy
        self.retry_after_s = retry_after_s
        self.usage = usage
        self.status = status
        super().__init__(self.message)

    def user_message(self) -> str:
        return f"{self.message}\n{self.remedy}" if self.remedy else self.message


class MissingKey(LlmError):
    pass


class InvalidKey(LlmError):
    pass


class BadRequest(LlmError):
    """The request itself is wrong (unknown model, unsupported parameter, schema too complex)."""


class RateLimited(LlmError):
    """Per-minute limit: wait and retry."""

    retryable = True


class DailyQuotaExhausted(LlmError):
    """Per-day limit of the Gemini free tier: retrying today is pointless."""


class SpendLimitReached(LlmError):
    """No credit left, or a spend limit set on the account."""


class ServerError(LlmError):
    """Overloaded (529, 503) or another server-side error: wait and retry."""

    retryable = True


class NetworkError(LlmError):
    retryable = True


class Refused(LlmError):
    """The model refused, or the provider blocked the response (Gemini SAFETY, RECITATION...)."""

    def __init__(self, message: str, *, reason: str, **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.reason = reason


class Truncated(LlmError):
    """The output hit max_tokens: the JSON is incomplete. `text`: the answer so far."""

    def __init__(self, message: str, *, text: str | None = None, **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.text = text


class ContextTooLong(LlmError):
    """The request does not fit the model's context window. Raised before sending when the
    estimate says so, or when the server refuses it, or when the response shows that the input
    may have been cut. Never retried: the caller has to send less."""

    def __init__(self, message: str, *, tokens: int | None = None, **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.tokens = tokens  # input tokens counted by the server, when known


# --- key redaction -------------------------------------------------------------------------------

# Shapes of the keys of both providers, plus every key the app has loaded in this process.
_KEY_PATTERNS = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|sk-ant-[0-9A-Za-z_\-]{10,}")
_known_secrets: set[str] = set()
REDACTED = "[key redacted]"


def register_secret(secret: str) -> None:
    if len(secret) >= 8:
        _known_secrets.add(secret)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    for secret in sorted({*_known_secrets, *secrets}, key=len, reverse=True):
        if secret:
            text = text.replace(secret, REDACTED)
    return _KEY_PATTERNS.sub(REDACTED, text)
