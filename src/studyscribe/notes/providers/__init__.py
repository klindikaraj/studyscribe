"""LLM providers behind one interface (base.py). SDKs are imported only by their own module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from studyscribe.notes.providers.base import (
    KEY_PROVIDERS,
    PROVIDERS,
    BadRequest,
    LlmProvider,
    MissingKey,
)

if TYPE_CHECKING:
    from studyscribe.notes.providers.ollama import OllamaOptions

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODELS = {
    "ollama": "qwen3.5:9b",
    "gemini": "gemini-3.8-flash",
    "anthropic": "claude-haiku-5-5",
}


def make_provider(provider: str, model: str | None, api_key: str | None = None, *,
                  ollama: OllamaOptions | None = None) -> LlmProvider:
    """Build a provider. Cloud providers get an explicit key (never one the SDK finds by
    itself); Ollama takes no key."""
    if provider not in PROVIDERS:
        raise BadRequest(f"Unknown provider {provider!r}.",
                         remedy=f"Use one of: {', '.join(PROVIDERS)}.")
    model = model or DEFAULT_MODELS[provider]
    if provider == "ollama":
        from studyscribe.notes.providers.ollama import OllamaProvider

        return OllamaProvider(model, ollama)
    if provider in KEY_PROVIDERS and not api_key:
        raise MissingKey(f"No API key for {provider}.",
                         remedy=f"Run `studyscribe config set-key {provider}`.")
    if provider == "gemini":
        from studyscribe.notes.providers.gemini import GeminiProvider

        return GeminiProvider(model, api_key)
    from studyscribe.notes.providers.claude import ClaudeProvider

    return ClaudeProvider(model, api_key)
