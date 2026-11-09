import io

import pytest

from fakes import DictKeyStore

from studyscribe import cli
from studyscribe.notes import keys
from studyscribe.notes.providers.base import InvalidKey, NetworkError

KEY = "AIzaSyTestKeyNotReal_0123456789abcdefg"


class StubProvider:
    def __init__(self, error=None):
        self.error = error

    def check(self):
        if self.error:
            raise self.error
        return "Gemini 3.8 Flash"


def run(argv, store, *, secret=KEY, error=None, environ=None):
    args = cli.build_parser().parse_args(["config", *argv])
    out = io.StringIO()
    made = []

    def make(provider, model, key):
        made.append((provider, model, key))
        return StubProvider(error)

    code = cli.cmd_config(args, out, store=store, environ=environ or {},
                          read_secret=lambda prompt: secret, make=make,
                          ollama_status=lambda: ["ollama: version 9.9 (fake)"])
    return code, out.getvalue(), made


def test_set_key_checks_and_stores_without_printing_it():
    store = DictKeyStore()
    code, text, made = run(["set-key", "gemini"], store, secret=f"  {KEY}\n")
    assert code == 0
    assert store.data == {(keys.SERVICE, "gemini"): KEY}
    assert made == [("gemini", None, KEY)]
    assert "checked: Gemini 3.8 Flash" in text
    assert KEY not in text and "AIza...defg" in text


def test_set_key_refuses_an_invalid_key():
    store = DictKeyStore()
    code, text, _ = run(["set-key", "gemini"], store, error=InvalidKey("Gemini rejected it."))
    assert code == 1 and store.data == {}
    assert "Not stored" in text


def test_set_key_stores_when_the_check_cannot_run():
    store = DictKeyStore()
    code, text, _ = run(["set-key", "gemini"], store, error=NetworkError("offline"))
    assert code == 0 and store.data
    assert "could not check it" in text


def test_set_key_no_check():
    store = DictKeyStore()
    code, text, made = run(["set-key", "anthropic", "--no-check"], store,
                           secret="sk-ant-api03-abcdefghijkl")
    assert code == 0 and made == [] and "not checked" in text


def test_set_key_empty():
    store = DictKeyStore()
    code, text, _ = run(["set-key", "gemini"], store, secret="  ")
    assert code == 1 and store.data == {}


def test_show_masks_keys_and_names_their_source():
    store = DictKeyStore()
    keys.store_key("gemini", KEY, store=store)
    code, text, _ = run(["show"], store,
                        environ={"ANTHROPIC_API_KEY": "sk-ant-api03-zzzzzzzzzzzz9999"})
    assert code == 0
    assert "gemini: AIza...defg from Windows Credential Manager" in text
    assert "anthropic: sk-a...9999 from ANTHROPIC_API_KEY" in text
    assert KEY not in text


def test_delete_key():
    store = DictKeyStore()
    keys.store_key("gemini", KEY, store=store)
    code, text, _ = run(["delete-key", "gemini"], store)
    assert code == 0 and store.data == {} and "removed" in text
    code, text, _ = run(["delete-key", "gemini"], store)
    assert code == 0 and "No stored key" in text


def test_credential_store_failure_is_a_clean_message():
    code, text, _ = run(["show"], DictKeyStore(fail=True))
    assert code == 1
    assert "\nError: Could not read from the Windows Credential Manager" in text


def test_show_starts_with_the_local_provider():
    code, text, _ = run(["show"], DictKeyStore())
    assert code == 0
    assert text.splitlines()[:2] == ["Default provider: ollama (qwen3.5:9b)",
                                     "ollama: version 9.9 (fake)"]


@pytest.mark.parametrize("action", ["set-key", "delete-key"])
def test_ollama_takes_no_key(action):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["config", action, "ollama"])
