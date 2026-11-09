"""Google Gemini through google-genai (the only module that imports it).

`client.models.generate_content` with `response_mime_type="application/json"` and
`response_json_schema`. The key is passed explicitly (the SDK would otherwise read
GOOGLE_API_KEY / GEMINI_API_KEY by itself), SDK retries are off (one attempt) and so is
automatic continuation, so every request the provider bills is one the app has paced and seen.
"""

from __future__ import annotations

import re
import time
from typing import Any

from studyscribe.notes.providers.base import (
    BadRequest,
    DailyQuotaExhausted,
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

TIMEOUT_S = 600
KEY_REMEDY = "Check the key with `studyscribe config set-key gemini` (https://aistudio.google.com/apikey)."
DAILY_REMEDY = ("The free tier's daily count resets at midnight Pacific time (09:00 in Italy). "
                "Run the same command again then: finished blocks are not sent again.")
_EFFORT_LEVELS = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
_BLOCK_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE",
                  "OTHER"}


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, api_key: str) -> None:
        from google import genai
        from google.genai import types

        register_secret(api_key)
        self.model = model
        self._types = types
        self._client = genai.Client(
            enterprise=False,
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=TIMEOUT_S * 1000,  # milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def generate(self, request: LlmRequest) -> LlmResponse:
        _require_thinking_effort(request)
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=request.system,
            response_mime_type="application/json",
            response_json_schema=request.schema,
            max_output_tokens=request.max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_level=_EFFORT_LEVELS[request.effort]),
            automatic_continuation=False,
        )
        started = time.perf_counter()
        response = self._call(lambda: self._client.models.generate_content(
            model=self.model, contents=request.user, config=config))
        duration = time.perf_counter() - started
        usage = usage_from(response.usage_metadata)

        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason is not None:
            reason = _enum_name(feedback.block_reason)
            raise Refused(f"Gemini blocked the request ({reason}).", reason=reason, usage=usage,
                          remedy=_refusal_remedy(reason))
        candidates = response.candidates or []
        if not candidates:
            raise Refused("Gemini returned no answer.", reason="EMPTY", usage=usage,
                          remedy="Try again; if it persists, try another model.")
        finish = _enum_name(candidates[0].finish_reason) if candidates[0].finish_reason else "STOP"
        if finish in ("MAX_TOKENS", "CONTINUATION"):
            raise Truncated(f"The answer was cut at {request.max_output_tokens} output tokens.",
                            usage=usage, remedy="The notes pipeline asks again with more room.")
        if finish in _BLOCK_REASONS:
            raise Refused(f"Gemini stopped the answer ({finish}).", reason=finish, usage=usage,
                          remedy=_refusal_remedy(finish))
        text = response.text or ""
        return LlmResponse(text=text, usage=usage, model_version=response.model_version,
                           request_id=response.response_id, duration_s=duration)

    def count_tokens(self, system: str, user: str) -> int:
        # The Gemini Developer API does not take a system instruction in countTokens: count the
        # two texts as contents (the system instruction is billed as input all the same).
        response = self._call(lambda: self._client.models.count_tokens(
            model=self.model, contents=[system, user]))
        return int(response.total_tokens or 0)

    def check(self) -> str:
        model = self._call(lambda: self._client.models.get(model=self.model))
        return model.display_name or model.name or self.model

    def close(self) -> None:
        pass

    def _call(self, fn):
        try:
            return fn()
        except LlmError:
            raise
        except Exception as error:
            raise map_error(error) from None


def _require_thinking_effort(request: LlmRequest) -> None:
    if request.effort == "none":
        raise BadRequest("Gemini cannot turn thinking off (effort none).",
                         remedy="Use --effort low, medium or high with this provider.")


# --- usage and errors (pure, tested without network) ---------------------------------------------


def usage_from(meta: Any) -> Usage:
    """Gemini counts thinking apart from the answer (`thoughts_token_count`); both are billed as
    output. `prompt_token_count` includes cached tokens."""
    if meta is None:
        return Usage(0, 0)
    thoughts = meta.thoughts_token_count or 0
    return Usage(
        input_tokens=meta.prompt_token_count or 0,
        output_tokens=(meta.candidates_token_count or 0) + thoughts,
        thinking_tokens=thoughts,
        cached_tokens=meta.cached_content_token_count or 0,
    )


def map_error(error: Exception) -> LlmError:
    from google.genai import errors

    if isinstance(error, errors.APIError):
        return _map_api_error(error.code, error.status, error.message or "", error.details)
    if _is_network_error(error):
        return NetworkError(f"Cannot reach Gemini: {type(error).__name__}.",
                            remedy="Check the internet connection.")
    if isinstance(error, ValueError):  # the SDK validates some arguments locally
        return BadRequest(f"Gemini request rejected by the SDK: {error}")
    return LlmError(f"Unexpected error from the Gemini SDK: {type(error).__name__}: {error}")


def _map_api_error(code: int | None, status: str | None, message: str, details: Any) -> LlmError:
    text = f"Gemini error {code} {status or ''}: {message}".strip()
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        quota_ids = _quota_ids(details)
        retry_after = _retry_delay(details)
        if any("perday" in q.lower() for q in quota_ids):
            return DailyQuotaExhausted("Gemini daily request quota used up.", remedy=DAILY_REMEDY,
                                       status=code)
        if "billing" in message.lower() or "credit" in message.lower():
            return SpendLimitReached(text, remedy="Check billing in Google AI Studio.", status=code)
        return RateLimited("Gemini per-minute limit reached.", retry_after_s=retry_after,
                           status=code)
    if code == 400 and ("api key" in message.lower() or "API_KEY" in str(details)):
        return InvalidKey("Gemini rejected the API key.", remedy=KEY_REMEDY, status=code)
    if code in (401, 403):
        return InvalidKey(f"Gemini refused access ({code}): {message}", remedy=KEY_REMEDY,
                          status=code)
    if code == 404:
        return BadRequest(f"Gemini: model not found ({message}).",
                          remedy="Check the model name (--model).", status=code)
    if code is not None and code >= 500:
        return ServerError(text, status=code)
    return BadRequest(text, status=code)


def _quota_ids(details: Any) -> list[str]:
    """quotaId values of QuotaFailure details, e.g.
    'GenerateRequestsPerDayPerProjectPerModel-FreeTier'."""
    ids = []
    for item in _detail_items(details):
        for violation in item.get("violations", []) or []:
            if isinstance(violation, dict):
                ids.extend(str(violation.get(k, "")) for k in ("quotaId", "quotaMetric"))
    return [i for i in ids if i]


def _retry_delay(details: Any) -> float | None:
    """RetryInfo.retryDelay, a duration like '37s' or '1.5s'."""
    for item in _detail_items(details):
        delay = item.get("retryDelay")
        if isinstance(delay, str):
            match = re.fullmatch(r"\s*([\d.]+)s\s*", delay)
            if match:
                return float(match.group(1))
    return None


def _detail_items(details: Any) -> list[dict]:
    if not isinstance(details, dict):
        return []
    error = details.get("error", details)
    items = error.get("details", []) if isinstance(error, dict) else []
    return [i for i in items if isinstance(i, dict)]


def _is_network_error(error: Exception) -> bool:
    classes: list[type] = [ConnectionError, TimeoutError]
    for module_name in ("httpx", "httpx2"):
        try:
            module = __import__(module_name)
            classes.append(module.TransportError)
        except (ImportError, AttributeError):
            pass
    try:
        import requests

        classes.append(requests.exceptions.ConnectionError)
        classes.append(requests.exceptions.Timeout)
    except ImportError:
        pass
    return isinstance(error, tuple(classes))


def _refusal_remedy(reason: str) -> str:
    if reason == "RECITATION":
        return ("Gemini blocks answers that quote long passages of known text; the notes "
                "pipeline asks again for a paraphrase.")
    return "Try another model (--provider anthropic) if it happens again."


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value))
