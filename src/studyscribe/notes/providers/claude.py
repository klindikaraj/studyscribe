"""Anthropic Claude through the anthropic SDK (the only module that imports it).

Streaming request (`messages.stream` + `get_final_message`, so long outputs never hit an HTTP
timeout) with structured output in `output_config.format` and adaptive thinking at the chosen
effort. No sampling parameters: Claude 5.5 models reject them. The SDK's own retries are off
(`max_retries=0`); the key is passed explicitly, so the SDK never falls back to environment
variables or profiles on disk.
"""

from __future__ import annotations

import time
from typing import Any

from studyscribe.notes.providers.base import (
    BadRequest,
    InvalidKey,
    LlmError,
    LlmRequest,
    LlmResponse,
    NetworkError,
    RateLimited,
    Refused,
    ServerError,
    SpendLimitReached,
    Truncated,
    Usage,
    register_secret,
)

TIMEOUT_S = 600.0
KEY_REMEDY = ("Check the key with `studyscribe config set-key anthropic` "
              "(https://platform.claude.com/docs/en/get-api-key).")
BILLING_REMEDY = "Check credit and spend limits in the Claude Console (Billing, Limits)."


class ClaudeProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str) -> None:
        import anthropic

        register_secret(api_key)
        self.model = model
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=TIMEOUT_S)

    def generate(self, request: LlmRequest) -> LlmResponse:
        schema = self._anthropic.transform_schema(request.schema)
        started = time.perf_counter()

        def send():
            with self._client.messages.stream(
                model=self.model,
                max_tokens=request.max_output_tokens,
                system=request.system,
                messages=[{"role": "user", "content": request.user}],
                thinking={"type": "adaptive"},
                output_config={"effort": request.effort,
                               "format": {"type": "json_schema", "schema": schema}},
            ) as stream:
                return stream.get_final_message(), stream.request_id

        message, request_id = self._call(send)
        duration = time.perf_counter() - started
        usage = usage_from(message.usage)
        if message.stop_reason == "refusal":
            details = message.stop_details
            category = getattr(details, "category", None) or "unspecified"
            raise Refused(f"Claude declined to answer (refusal, {category}).", reason=category,
                          usage=usage, remedy="Try the other model or provider.")
        if message.stop_reason in ("max_tokens", "model_context_window_exceeded"):
            raise Truncated(f"The answer was cut at {request.max_output_tokens} output tokens.",
                            usage=usage, remedy="The notes pipeline asks again with more room.")
        text = "".join(block.text for block in message.content if block.type == "text")
        return LlmResponse(text=text, usage=usage, model_version=message.model,
                           request_id=request_id, duration_s=duration)

    def count_tokens(self, system: str, user: str) -> int:
        result = self._call(lambda: self._client.messages.count_tokens(
            model=self.model, system=system, messages=[{"role": "user", "content": user}]))
        return int(result.input_tokens)

    def check(self) -> str:
        info = self._call(lambda: self._client.models.retrieve(self.model))
        return info.display_name or info.id

    def _call(self, fn):
        try:
            return fn()
        except LlmError:
            raise
        except Exception as error:
            raise map_error(error, self._anthropic) from None


# --- usage and errors (pure, tested without network) ---------------------------------------------


def usage_from(usage: Any) -> Usage:
    """Claude's `input_tokens` excludes cache reads and writes; `output_tokens` includes
    thinking."""
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    details = getattr(usage, "output_tokens_details", None)
    return Usage(
        input_tokens=usage.input_tokens + cache_read + cache_write,
        output_tokens=usage.output_tokens,
        thinking_tokens=getattr(details, "thinking_tokens", None) if details else None,
        cached_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def map_error(error: Exception, anthropic: Any = None) -> LlmError:
    if anthropic is None:
        import anthropic
    if isinstance(error, anthropic.APIStatusError):
        return _map_status(error.status_code, _error_body(error), _retry_after(error),
                           getattr(error, "request_id", None))
    if isinstance(error, anthropic.APIConnectionError):  # APITimeoutError included
        kind = "timed out" if isinstance(error, anthropic.APITimeoutError) else "unreachable"
        return NetworkError(f"Claude API {kind}.", remedy="Check the internet connection.")
    if isinstance(error, anthropic.AnthropicError):
        return LlmError(f"Claude SDK error: {type(error).__name__}: {error}")
    return LlmError(f"Unexpected error calling Claude: {type(error).__name__}: {error}")


def _map_status(status: int, body: dict, retry_after: float | None,
                request_id: str | None) -> LlmError:
    error = body.get("error") if isinstance(body.get("error"), dict) else {}
    message = str(error.get("message") or "")
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    suffix = f" (request {request_id})" if request_id else ""
    text = f"Claude API error {status} {error.get('type', '')}: {message}{suffix}"
    lowered = message.lower()
    if details.get("error_code") == "enforced_spend_limit_reached" or (
            status == 400 and ("usage limits" in lowered or "credit balance" in lowered)):
        return SpendLimitReached(f"Claude API: no credit or spend limit reached{suffix}.",
                                 remedy=BILLING_REMEDY, status=status)
    if status == 401:
        return InvalidKey("Claude API rejected the key.", remedy=KEY_REMEDY, status=status)
    if status == 403:
        return InvalidKey(f"Claude API refused access: {message}", remedy=KEY_REMEDY,
                          status=status)
    if status == 404:
        return BadRequest(f"Claude API: not found ({message}).",
                          remedy="Check the model name (--model).", status=status)
    if status == 429:
        return RateLimited("Claude API rate limit reached.", retry_after_s=retry_after,
                           status=status)
    if status >= 500:  # 500, 503, 529 overloaded
        return ServerError(text, retry_after_s=retry_after, status=status)
    return BadRequest(text, status=status)


def _error_body(error: Any) -> dict:
    body = getattr(error, "body", None)
    return body if isinstance(body, dict) else {}


def _retry_after(error: Any) -> float | None:
    response = getattr(error, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
