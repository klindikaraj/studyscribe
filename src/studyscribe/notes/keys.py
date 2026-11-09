"""API keys: Windows Credential Manager through keyring, environment variables as fallback.

Lookup order: the credential store (service "studyscribe", user = provider), then
STUDYSCRIBE_<PROVIDER>_API_KEY, then the provider's usual variable (GEMINI_API_KEY,
ANTHROPIC_API_KEY). Keys are never written to files or printed: `mask` shows only their ends.
This is the only module that imports keyring.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from studyscribe.notes.providers.base import PROVIDERS, LlmError, MissingKey, register_secret

SERVICE = "studyscribe"
ENV_VARS = {
    "gemini": ("STUDYSCRIBE_GEMINI_API_KEY", "GEMINI_API_KEY"),
    "anthropic": ("STUDYSCRIBE_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
}
KEY_PAGES = {
    "gemini": "https://aistudio.google.com/apikey",
    "anthropic": "https://platform.claude.com/docs/en/get-api-key",
}
KEYRING_SOURCE = "Windows Credential Manager"


class KeyStore(Protocol):
    """The part of the keyring API used here (tests pass a dict-backed fake)."""

    def get_password(self, service: str, username: str) -> str | None: ...
    def set_password(self, service: str, username: str, password: str) -> None: ...
    def delete_password(self, service: str, username: str) -> None: ...


class KeyStoreError(LlmError):
    pass


@dataclass(frozen=True)
class FoundKey:
    provider: str
    key: str
    source: str  # KEYRING_SOURCE or the name of the environment variable

    def __repr__(self) -> str:  # never show the key in a traceback or a debug print
        return f"FoundKey(provider={self.provider!r}, key={mask(self.key)!r}, source={self.source!r})"


def default_store() -> KeyStore:
    import keyring

    return keyring.get_keyring()


def _check_provider(provider: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}")


def _store_error(action: str, error: Exception) -> KeyStoreError:
    return KeyStoreError(
        f"Could not {action} the Windows Credential Manager: {type(error).__name__}.",
        remedy="Use an environment variable instead (see `studyscribe config show`).",
    )


def find_key(provider: str, *, store: KeyStore | None = None,
             environ: Mapping[str, str] = os.environ) -> FoundKey | None:
    _check_provider(provider)
    try:
        stored = (store or default_store()).get_password(SERVICE, provider)
    except Exception as error:  # keyring backends raise their own types
        raise _store_error("read from", error) from None
    found = None
    if stored and stored.strip():
        found = FoundKey(provider, stored.strip(), KEYRING_SOURCE)
    else:
        for name in ENV_VARS[provider]:
            value = environ.get(name, "").strip()
            if value:
                found = FoundKey(provider, value, name)
                break
    if found:
        register_secret(found.key)
    return found


def require_key(provider: str, *, store: KeyStore | None = None,
                environ: Mapping[str, str] = os.environ) -> FoundKey:
    found = find_key(provider, store=store, environ=environ)
    if found is None:
        raise MissingKey(
            f"No API key for {provider}.",
            remedy=f"Run `studyscribe config set-key {provider}` (key from {KEY_PAGES[provider]}),"
            f" or set {ENV_VARS[provider][0]}.",
        )
    return found


def clean_key(raw: str) -> str:
    key = raw.strip().strip('"').strip("'").strip()
    if not key:
        raise ValueError("the key is empty")
    if any(ch.isspace() for ch in key) or not key.isprintable():
        raise ValueError("the key contains spaces or invisible characters")
    return key


def store_key(provider: str, key: str, *, store: KeyStore | None = None) -> None:
    _check_provider(provider)
    try:
        (store or default_store()).set_password(SERVICE, provider, key)
    except Exception as error:
        raise _store_error("write to", error) from None
    register_secret(key)


def delete_key(provider: str, *, store: KeyStore | None = None) -> bool:
    """Remove the stored key; False if there was none."""
    _check_provider(provider)
    store = store or default_store()
    try:
        if store.get_password(SERVICE, provider) is None:
            return False
        store.delete_password(SERVICE, provider)
    except Exception as error:
        raise _store_error("delete from", error) from None
    return True


def mask(key: str) -> str:
    if len(key) < 12:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"
