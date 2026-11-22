"""notes.md: sections, timestamps, hidden answers, code states, warnings, appendix, escaping."""

import re
from dataclasses import replace

from notes_fixtures import hostile_doc, make_doc

from studyscribe.notes.render_md import fence_for, render_markdown


def headings(text):
    return re.findall(r"^## (.+)$", text, flags=re.MULTILINE)


def test_the_seven_sections_and_the_appendix_in_template_order():
    text = render_markdown(make_doc())
    assert text.startswith("# Lezione PyTorch\n")
    assert headings(text) == [
        "Riepilogo", "Scaletta", "Concetti chiave", "Definizioni", "Esempi e codice",
        "Domande di ripasso", "Punti poco chiari", "Parole da verificare ascoltando"]


def test_headings_follow_the_notes_language():
    text = render_markdown(make_doc(language="en"))
    assert headings(text)[:3] == ["Summary", "Outline", "Key concepts"]
    assert headings(text)[-1] == "Words to check by listening"
    assert "- Course: Reti neurali" in text
    assert headings(render_markdown(make_doc(language="fr")))[0] == "Summary"


def test_empty_sections_are_left_out():
    doc = make_doc()
    content = {**doc.content, "examples": [], "open_points": []}
    text = render_markdown(replace(doc, content=content))
    assert "Esempi e codice" not in text and "Punti poco chiari" not in text


def test_timestamps_open_the_html_at_the_second_of_the_line():
    text = render_markdown(make_doc())
    assert "- [0:01:06](notes.html#t=66) Introduzione" in text  # 66.98 s: never later
    assert "[0:01:16](notes.html#t=76) [0:01:24](notes.html#t=84)" in text
    assert "(notes-v2.html#t=" in render_markdown(make_doc(), html_name="notes-v2.html")
    assert "(notes.html#t=" not in render_markdown(make_doc(), html_name="notes-v2.html")


def test_the_answer_is_hidden_in_details():
    text = render_markdown(make_doc())
    question = text.index("**Quanti sono gli step?**")
    details = text.index("<details><summary>Mostra la risposta</summary>")
    assert question < details < text.index("Quattro step.") < text.index("</details>")
    assert "Quattro step." not in text[:details]
    assert "<summary>Show answer</summary>" in render_markdown(make_doc(language="en"))


def test_code_states_are_shown_next_to_the_item():
    text = render_markdown(make_doc())
    assert "Calcola la previsione. [0:08:20](notes.html#t=500)\n" in text  # ok: no flag
    assert "**(da verificare: y_pred)**" in text
    assert "**(nome assente dalla trascrizione: misure_notte)**" in text
    english = render_markdown(make_doc(language="en"))
    assert "**(to verify: y_pred)**" in english
    assert "**(name not in the transcript: misure_notte)**" in english


def test_an_example_without_code_has_no_block():
    text = render_markdown(make_doc())
    assert text.count("\n```\n") == 6  # three code blocks, open and close
    assert "Solo a parole. [0:15:50](notes.html#t=950)\n\n## " in text


def test_coverage_warning_is_visible_with_a_link():
    text = render_markdown(make_doc())
    assert "> **Parti della lezione che le note potrebbero non coprire**" in text
    assert "> - [da 0:17:12 a 0:18:07 (righe da S226 a S241)](notes.html#t=1032)" in text
    assert "potrebbero non coprire" not in render_markdown(make_doc(stats={}))


def test_appendix_has_word_line_time_and_probability():
    text = render_markdown(make_doc())
    assert "| Parola | Riga | Tempo | Probabilità |" in text
    assert ("| TEC | S186: stacchiamoli col metodo di **TEC** | [0:14:11](notes.html#t=851)"
            " | 0.35 |") in text


def test_appendix_after_a_new_transcription_has_no_line_text_and_a_warning():
    text = render_markdown(make_doc(line_texts=None))
    assert "| TEC | S186 | [0:14:11](notes.html#t=851) | 0.35 |" in text
    assert "trascrizione precedente" in text
    assert "trascrizione precedente" not in render_markdown(make_doc())


def test_appendix_without_words():
    text = render_markdown(make_doc(uncertain=()))
    assert "Nessuna parola sotto la soglia di confidenza." in text and "|---|" not in text


def test_time_mismatch_warning():
    assert "3 timestamp non corrispondono" in render_markdown(make_doc(time_mismatches=3))


def test_footer_says_how_the_notes_were_made_and_what_was_left_out():
    text = render_markdown(make_doc())
    assert "qwen3.5:9b (effort none, prompt v5.1), 12,167 token" in text
    assert "2 riferimenti, 1 elementi" in text
    assert "\n---\n" not in render_markdown(make_doc(stats={}))
    assert "\n---\n" in text


def test_items_stay_on_one_line_and_pipes_do_not_break_the_table():
    doc = make_doc()
    content = {**doc.content, "concepts": [{"name": "A", "explanation": "uno\n\ndue",
                                            "refs": doc.content["concepts"][0]["refs"]}]}
    text = render_markdown(replace(doc, content=content,
                                   uncertain=replace_word(doc, "a|b")))
    assert "- **A**: uno due [" in text
    assert "| a\\|b |" in text


def replace_word(doc, word):
    from studyscribe.notes.uncertain import UncertainWord
    return (UncertainWord(word, 186, 851_792, 0.3),)


def test_code_fence_is_longer_than_any_backtick_run_inside():
    assert fence_for("x = 1") == "```"
    assert fence_for("a ``` b") == "````"
    assert fence_for("`````x") == "``````"
    doc = make_doc()
    content = {**doc.content, "examples": [{"code": "print('```')", "explanation": "e",
                                             "refs": doc.content["examples"][0]["refs"]}]}
    assert "\n````\nprint('```')\n````\n" in render_markdown(replace(doc, content=content))


def test_hostile_text_cannot_become_html_or_a_link():
    text = render_markdown(hostile_doc())
    outside = re.sub(r"(`{3,})\n.*?\n\1\n", "", text, flags=re.DOTALL)  # fenced code is inert
    allowed = re.sub(r"</?(?:details|summary)>?", "", outside)
    allowed = re.sub(r"<details><summary>.*?</summary>", "", allowed)
    assert "<" not in allowed.replace("&lt;", "")
    assert ">" not in re.sub(r"^\s*>", "", allowed.replace("&gt;", ""), flags=re.MULTILINE)
    assert not re.search(r"(?<!\\)\]\(javascript:", outside)  # only escaped brackets
    assert text.count("<details>") == 1 and text.count("</details>") == 1
