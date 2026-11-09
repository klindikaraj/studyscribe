from datetime import date

import pytest
from fakes import DictKeyStore

from studyscribe.notes import keys, pricing
from studyscribe.notes.providers.base import LlmError, MissingKey, Usage, redact, register_secret


def test_haiku_below_the_threshold():
    usage = Usage(input_tokens=40_000, output_tokens=10_000)
    assert pricing.list_cost("anthropic", "claude-haiku-5-5", usage) == pytest.approx(
        40_000 * 0.10 / 1e6 + 10_000 * 0.50 / 1e6)


def test_haiku_above_the_threshold_charges_the_whole_request_more():
    usage = Usage(input_tokens=100_001, output_tokens=1_000)
    assert pricing.list_cost("anthropic", "claude-haiku-5-5", usage) == pytest.approx(
        100_001 * 0.50 / 1e6 + 1_000 * 2.50 / 1e6)


def test_haiku_threshold_counts_cached_tokens():
    usage = Usage(input_tokens=120_000, output_tokens=0, cached_tokens=90_000)
    cost = pricing.list_cost("anthropic", "claude-haiku-5-5", usage)
    assert cost == pytest.approx((30_000 * 0.50 + 90_000 * 0.05) / 1e6)


def test_sonnet():
    usage = Usage(input_tokens=130_000, output_tokens=90_000)
    assert pricing.list_cost("anthropic", "claude-sonnet-5-5", usage) == pytest.approx(
        0.26 + 0.90)


def test_gemini_flash_price_doubles_in_2027():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert pricing.list_cost("gemini", "gemini-3.8-flash", usage,
                             on=date(2026, 12, 31)) == pytest.approx(4.50)
    assert pricing.list_cost("gemini", "gemini-3.8-flash", usage,
                             on=date(2027, 1, 1)) == pytest.approx(9.00)


def test_free_tier_is_charged_nothing_but_has_a_list_price():
    usage = Usage(input_tokens=10_000, output_tokens=1_000)
    assert pricing.charged_cost("gemini", "gemini-3.8-flash", usage, free_tier=True) == 0.0
    assert pricing.charged_cost("gemini", "gemini-3.8-flash", usage, free_tier=False) > 0
    # The free tier flag means nothing for Claude.
    assert pricing.charged_cost("anthropic", "claude-haiku-5-5", usage, free_tier=True) > 0


def test_unknown_model_has_no_price():
    assert pricing.list_cost("gemini", "gemini-9-ultra", Usage(1, 1)) is None


# --- keys ----------------------------------------------------------------------------------------


def test_key_lookup_order():
    store = DictKeyStore()
    env = {"GEMINI_API_KEY": "AIza-generic-key-000000", "STUDYSCRIBE_GEMINI_API_KEY": ""}
    found = keys.find_key("gemini", store=store, environ=env)
    assert (found.key, found.source) == ("AIza-generic-key-000000", "GEMINI_API_KEY")

    env["STUDYSCRIBE_GEMINI_API_KEY"] = "AIza-app-key-1111111111"
    assert keys.find_key("gemini", store=store, environ=env).source == "STUDYSCRIBE_GEMINI_API_KEY"

    keys.store_key("gemini", "AIza-stored-key-22222222", store=store)
    found = keys.find_key("gemini", store=store, environ=env)
    assert (found.key, found.source) == ("AIza-stored-key-22222222", keys.KEYRING_SOURCE)


def test_providers_do_not_share_keys():
    store = DictKeyStore()
    keys.store_key("gemini", "AIza-stored-key-22222222", store=store)
    assert keys.find_key("anthropic", store=store, environ={}) is None


def test_missing_key_message_has_a_remedy():
    with pytest.raises(MissingKey) as info:
        keys.require_key("anthropic", store=DictKeyStore(), environ={})
    assert "studyscribe config set-key anthropic" in info.value.user_message()
    assert "STUDYSCRIBE_ANTHROPIC_API_KEY" in info.value.user_message()


def test_delete_key():
    store = DictKeyStore()
    assert keys.delete_key("gemini", store=store) is False
    keys.store_key("gemini", "AIza-stored-key-22222222", store=store)
    assert keys.delete_key("gemini", store=store) is True
    assert store.data == {}


def test_store_failure_is_a_clean_error():
    with pytest.raises(keys.KeyStoreError) as info:
        keys.find_key("gemini", store=DictKeyStore(fail=True), environ={})
    assert "environment variable" in info.value.user_message()


@pytest.mark.parametrize("raw, key", [
    ("  AIzaSyABC123  \n", "AIzaSyABC123"),
    ('"sk-ant-api03-xyz"', "sk-ant-api03-xyz"),
])
def test_clean_key(raw, key):
    assert keys.clean_key(raw) == key


@pytest.mark.parametrize("raw", ["", "   ", "AIza with space", "AIza\x00x"])
def test_clean_key_rejects(raw):
    with pytest.raises(ValueError):
        keys.clean_key(raw)


def test_mask_and_repr_never_show_the_key():
    key = "sk-ant-api03-SECRETSECRETSECRET-abcd"
    assert keys.mask(key) == "sk-a...abcd"
    found = keys.FoundKey("anthropic", key, "ANTHROPIC_API_KEY")
    assert "SECRET" not in repr(found)


# --- redaction -----------------------------------------------------------------------------------


def test_errors_never_carry_a_loaded_key():
    secret = "my-unusual-secret-value-123"
    register_secret(secret)
    error = LlmError(f"request failed for key {secret}")
    assert secret not in str(error)
    assert secret not in error.user_message()


def test_key_shapes_are_redacted_even_when_unknown():
    text = ("url ?key=AIzaSyA1b2C3d4E5f6G7h8I9j0KlMnOpQrStUvW and "
            "x-api-key: sk-ant-api03-AbCdEf123456_-xyz")
    cleaned = redact(text)
    assert "AIzaSy" not in cleaned and "sk-ant-api03" not in cleaned
    assert cleaned.count("[key redacted]") == 2
