"""LLM providers behind one interface (base.py). SDKs are imported only by their own module."""

from __future__ import annotations

from studyscribe.notes.providers.base import PROVIDERS, BadRequest, LlmProvider

DEFAULT_MODELS = {
    "gemini": "gemini-3.8-flash",
    "anthropic": "claude-haiku-5-5",
}


def make_provider(provider: str, model: str | None, api_key: str) -> LlmProvider:
    """Build a provider with an explicit key (never one the SDK finds by itself)."""
    if provider not in PROVIDERS:
        raise BadRequest(f"Unknown provider {provider!r}.",
                         remedy=f"Use one of: {', '.join(PROVIDERS)}.")
    model = model or DEFAULT_MODELS[provider]
    if provider == "gemini":
        from studyscribe.notes.providers.gemini import GeminiProvider

        return GeminiProvider(model, api_key)
    from studyscribe.notes.providers.claude import ClaudeProvider

    return ClaudeProvider(model, api_key)
