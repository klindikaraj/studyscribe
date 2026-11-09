"""Provider modules without network: error mapping from the SDKs' own exception classes, usage
normalisation, and response handling with the SDK clients replaced by stubs."""

from contextlib import contextmanager
from types import SimpleNamespace

import anthropic
import httpx
import httpx2
import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from studyscribe.notes.providers import claude, gemini, make_provider
from studyscribe.notes.providers.base import (
    BadRequest,
    DailyQuotaExhausted,
    InvalidKey,
    LlmRequest,
    NetworkError,
    RateLimited,
    Refused,
    ServerError,
    SpendLimitReached,
    Truncated,
)

REQUEST = LlmRequest(system="sys", user="S1 hello", schema={"type": "object"},
                     max_output_tokens=1000)
FAKE_GEMINI_KEY = "AIza-test-key-not-real-0000000000000"
FAKE_CLAUDE_KEY = "sk-ant-test-key-not-real-000000"

# --- Gemini --------------------------------------------------------------------------------------


def gemini_error(code, status, message, details=()):
    body = {"error": {"code": code, "status": status, "message": message,
                      "details": list(details)}}
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, body)


def quota(quota_id):
    return {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaMetric": "generativelanguage.googleapis.com/"
                            "generate_content_free_tier_requests", "quotaId": quota_id}]}


RETRY_37 = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}


def test_gemini_per_minute_quota_is_retryable_with_the_suggested_delay():
    error = gemini.map_error(gemini_error(
        429, "RESOURCE_EXHAUSTED", "Quota exceeded",
        [quota("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"), RETRY_37]))
    assert isinstance(error, RateLimited) and error.retryable
    assert error.retry_after_s == 37.0


def test_gemini_daily_quota_is_recognised_from_the_429():
    error = gemini.map_error(gemini_error(
        429, "RESOURCE_EXHAUSTED", "Quota exceeded",
        [quota("GenerateRequestsPerDayPerProjectPerModel-FreeTier"), RETRY_37]))
    assert isinstance(error, DailyQuotaExhausted) and not error.retryable
    assert "09:00" in error.user_message()


def test_gemini_429_without_details_is_a_rate_limit():
    error = gemini.map_error(gemini_error(429, "RESOURCE_EXHAUSTED", "Too many requests"))
    assert isinstance(error, RateLimited) and error.retry_after_s is None


def test_gemini_invalid_key():
    error = gemini.map_error(gemini_error(
        400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.",
        [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]))
    assert isinstance(error, InvalidKey)
    assert "config set-key gemini" in error.user_message()


def test_gemini_other_errors():
    assert isinstance(gemini.map_error(gemini_error(404, "NOT_FOUND", "models/x not found")),
                      BadRequest)
    assert isinstance(gemini.map_error(gemini_error(503, "UNAVAILABLE", "overloaded")),
                      ServerError)
    assert isinstance(gemini.map_error(gemini_error(400, "INVALID_ARGUMENT", "bad schema")),
                      BadRequest)


@pytest.mark.parametrize("exc", [
    httpx.ConnectError("no route"),
    httpx2.ReadTimeout("slow"),
    ConnectionResetError(),
])
def test_gemini_network_errors(exc):
    assert isinstance(gemini.map_error(exc), NetworkError)


def test_gemini_error_text_never_contains_the_key():
    error = gemini.map_error(gemini_error(
        400, "INVALID_ARGUMENT", f"bad request for key={FAKE_GEMINI_KEY}"))
    assert FAKE_GEMINI_KEY not in error.user_message()


def test_gemini_usage_counts_thinking_as_output():
    meta = genai_types.GenerateContentResponseUsageMetadata(
        prompt_token_count=5000, candidates_token_count=800, thoughts_token_count=1200,
        cached_content_token_count=1000)
    usage = gemini.usage_from(meta)
    assert (usage.input_tokens, usage.output_tokens, usage.thinking_tokens,
            usage.cached_tokens) == (5000, 2000, 1200, 1000)


def gemini_response(text='{"a": 1}', finish="STOP", block=None):
    return genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
            finish_reason=finish)] if finish else [],
        prompt_feedback=(genai_types.GenerateContentResponsePromptFeedback(block_reason=block)
                         if block else None),
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=10, thoughts_token_count=5),
        model_version="gemini-3.8-flash-001", response_id="resp-1")


def stub_gemini(response=None, error=None):
    provider = make_provider("gemini", None, FAKE_GEMINI_KEY)
    sent = {}

    def generate_content(*, model, contents, config):
        sent.update(model=model, contents=contents, config=config)
        if error is not None:
            raise error
        return response

    provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return provider, sent


def test_gemini_generate_sends_text_schema_and_effort():
    provider, sent = stub_gemini(gemini_response())
    result = provider.generate(REQUEST)
    assert result.text == '{"a": 1}'
    assert result.usage.output_tokens == 15
    assert (result.model_version, result.request_id) == ("gemini-3.8-flash-001", "resp-1")
    config = sent["config"]
    assert sent["model"] == "gemini-3.8-flash" and sent["contents"] == "S1 hello"
    assert config.system_instruction == "sys"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == {"type": "object"}
    assert config.thinking_config.thinking_level == genai_types.ThinkingLevel.MEDIUM
    assert config.automatic_continuation is False
    assert config.temperature is None


def test_gemini_truncated_answer_keeps_its_usage():
    provider, _ = stub_gemini(gemini_response(text='{"a": ', finish="MAX_TOKENS"))
    with pytest.raises(Truncated) as info:
        provider.generate(REQUEST)
    assert info.value.usage.input_tokens == 100


@pytest.mark.parametrize("finish", ["RECITATION", "SAFETY"])
def test_gemini_blocked_answer(finish):
    provider, _ = stub_gemini(gemini_response(text="", finish=finish))
    with pytest.raises(Refused) as info:
        provider.generate(REQUEST)
    assert info.value.reason == finish and info.value.usage is not None


def test_gemini_blocked_prompt():
    provider, _ = stub_gemini(gemini_response(finish=None, block="PROHIBITED_CONTENT"))
    with pytest.raises(Refused) as info:
        provider.generate(REQUEST)
    assert info.value.reason == "PROHIBITED_CONTENT"


def test_gemini_sdk_errors_become_llm_errors():
    provider, _ = stub_gemini(error=gemini_error(503, "UNAVAILABLE", "overloaded"))
    with pytest.raises(ServerError):
        provider.generate(REQUEST)


def test_gemini_sdk_retries_are_off():
    provider = make_provider("gemini", None, FAKE_GEMINI_KEY)
    options = provider._client._api_client._http_options
    assert options.retry_options.attempts == 1


# --- Claude --------------------------------------------------------------------------------------

API = "https://api.anthropic.com/v1/messages"


def claude_status_error(cls, status, body, headers=None):
    response = httpx2.Response(status, headers=headers or {"request-id": "req_1"},
                               request=httpx2.Request("POST", API))
    return cls("error", response=response, body=body)


def body(kind, message, **details):
    error = {"type": kind, "message": message}
    if details:
        error["details"] = details
    return {"type": "error", "error": error}


def test_claude_rate_limit_honours_retry_after():
    error = claude.map_error(claude_status_error(
        anthropic.RateLimitError, 429, body("rate_limit_error", "slow down"),
        headers={"retry-after": "12", "request-id": "req_1"}))
    assert isinstance(error, RateLimited) and error.retry_after_s == 12.0


def test_claude_spend_cap_429_is_not_retried():
    error = claude.map_error(claude_status_error(
        anthropic.RateLimitError, 429,
        body("rate_limit_error", "spend limit", error_code="enforced_spend_limit_reached")))
    assert isinstance(error, SpendLimitReached) and not error.retryable


@pytest.mark.parametrize("message", [
    "You have reached your specified API usage limits. You will regain access on 2026-12-01.",
    "Your credit balance is too low to access the Anthropic API.",
])
def test_claude_billing_400s(message):
    error = claude.map_error(claude_status_error(
        anthropic.BadRequestError, 400, body("invalid_request_error", message)))
    assert isinstance(error, SpendLimitReached)


def test_claude_other_statuses():
    assert isinstance(claude.map_error(claude_status_error(
        anthropic.AuthenticationError, 401, body("authentication_error", "invalid x-api-key"))),
        InvalidKey)
    overloaded = claude.map_error(claude_status_error(
        anthropic.OverloadedError, 529, body("overloaded_error", "Overloaded")))
    assert isinstance(overloaded, ServerError) and overloaded.retryable
    assert "req_1" in overloaded.message
    assert isinstance(claude.map_error(claude_status_error(
        anthropic.NotFoundError, 404, body("not_found_error", "model: claude-x"))), BadRequest)
    assert isinstance(claude.map_error(claude_status_error(
        anthropic.BadRequestError, 400, body("invalid_request_error", "temperature"))),
        BadRequest)


def test_claude_network_errors():
    request = httpx2.Request("POST", API)
    assert isinstance(claude.map_error(anthropic.APIConnectionError(request=request)),
                      NetworkError)
    assert isinstance(claude.map_error(anthropic.APITimeoutError(request=request)), NetworkError)


def claude_message(stop_reason="end_turn", text='{"a": 1}', stop_details=None):
    return anthropic.types.Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "thinking", "thinking": "...", "signature": "sig"},
                    {"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details,
        "usage": {"input_tokens": 900, "output_tokens": 400, "cache_read_input_tokens": 100,
                  "cache_creation_input_tokens": 0,
                  "output_tokens_details": {"thinking_tokens": 250}},
    })


def stub_claude(message=None, error=None):
    provider = make_provider("anthropic", None, FAKE_CLAUDE_KEY)
    sent = {}

    @contextmanager
    def stream(**kwargs):
        sent.update(kwargs)
        if error is not None:
            raise error
        yield SimpleNamespace(get_final_message=lambda: message, request_id="req_9")

    provider._client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    return provider, sent


def test_claude_generate_sends_schema_effort_and_no_sampling():
    provider, sent = stub_claude(claude_message())
    result = provider.generate(REQUEST)
    assert result.text == '{"a": 1}'
    assert result.request_id == "req_9"
    assert (result.usage.input_tokens, result.usage.cached_tokens, result.usage.output_tokens,
            result.usage.thinking_tokens) == (1000, 100, 400, 250)
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["system"] == "sys"
    assert sent["messages"] == [{"role": "user", "content": "S1 hello"}]
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"]["effort"] == "medium"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert not {"temperature", "top_p", "top_k"} & sent.keys()


def test_claude_schema_is_made_strict():
    provider, sent = stub_claude(claude_message())
    provider.generate(LlmRequest(system="s", user="u", max_output_tokens=10, schema={
        "type": "object", "properties": {"x": {"type": "string", "maxLength": 5}},
        "required": ["x"]}))
    schema = sent["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert "maxLength" not in schema["properties"]["x"]


def test_claude_refusal():
    provider, _ = stub_claude(claude_message(stop_reason="refusal", text="",
                                             stop_details={"type": "refusal",
                                                           "category": "general_harms"}))
    with pytest.raises(Refused) as info:
        provider.generate(REQUEST)
    assert info.value.reason == "general_harms" and info.value.usage.output_tokens == 400


def test_claude_truncated():
    provider, _ = stub_claude(claude_message(stop_reason="max_tokens", text='{"a"'))
    with pytest.raises(Truncated) as info:
        provider.generate(REQUEST)
    assert info.value.usage is not None


def test_claude_sdk_errors_become_llm_errors():
    provider, _ = stub_claude(error=claude_status_error(
        anthropic.OverloadedError, 529, body("overloaded_error", "Overloaded")))
    with pytest.raises(ServerError):
        provider.generate(REQUEST)


def test_claude_sdk_retries_are_off_and_key_is_explicit(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-environment-0000000")
    provider = make_provider("anthropic", "claude-sonnet-5-5", FAKE_CLAUDE_KEY)
    assert provider._client.max_retries == 0
    assert provider._client.api_key == FAKE_CLAUDE_KEY
    assert provider.model == "claude-sonnet-5-5"


def test_unknown_provider():
    with pytest.raises(BadRequest):
        make_provider("openai", None, "x")
