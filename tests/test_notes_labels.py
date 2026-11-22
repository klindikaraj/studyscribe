"""The fixed texts of the notes exports: both tables are complete, formatted and fall back."""

import string

from studyscribe.notes import labels


def test_tables_have_the_same_keys():
    assert set(labels._TEXTS["en"]) == set(labels._TEXTS["it"])


def test_tables_use_the_same_placeholders():
    def fields(template):
        return {name for _, name, _, _ in string.Formatter().parse(template) if name}

    for key, english in labels._TEXTS["en"].items():
        assert fields(english) == fields(labels._TEXTS["it"][key]), key


def test_the_wording_the_user_asked_for():
    assert labels.text("it", "code_check") == "da verificare"
    assert labels.text("it", "code_unsupported") == "nome assente dalla trascrizione"
    assert labels.text("it", "appendix_title") == "Parole da verificare ascoltando"
    assert labels.text("en", "appendix_title") == "Words to check by listening"


def test_unknown_and_missing_languages_fall_back_to_english():
    assert labels.text("fr", "show_answer") == "Show answer"
    assert labels.text(None, "show_answer") == "Show answer"
    assert labels.text("IT-it", "show_answer") == "Mostra la risposta"


def test_values_are_formatted():
    assert "version 2 of 3" in labels.text("en", "notes_info_value", version=2, total=3,
                                           model="m", date="d")
    command = "studyscribe export X --audio"
    assert command in labels.text("it", "no_audio", command=command)

