"""notes.html: sections, timestamps that seek, hidden answers, code states, the missing-audio
page, and the safety audit (hostile text everywhere, nothing but our own script and style)."""

import base64
import hashlib
import re
from dataclasses import replace
from html.parser import HTMLParser

from notes_fixtures import HOSTILE, hostile_doc, make_doc

from studyscribe.notes.render_html import _asset, content_security_policy, render_html

ALLOWED_TAGS = {
    "html", "head", "meta", "title", "style", "body", "header", "main", "audio", "p", "h1", "h2",
    "h3", "dl", "dt", "dd", "section", "aside", "article", "ul", "li", "a", "span", "strong",
    "em", "code", "pre", "details", "summary", "table", "thead", "tbody", "tr", "th", "td",
    "mark", "footer", "script"}
ALLOWED_ATTRS = {
    "html": {"lang"}, "meta": {"charset", "name", "content", "http-equiv"},
    "audio": {"id", "controls", "preload", "src", "aria-label"},
    "p": {"id", "class", "hidden"}, "section": {"data-section"}, "aside": {"class", "data-kind"},
    "header": {"class"}, "ul": {"class"}, "article": {"class"}, "span": {"class", "data-kind"},
    "th": {"scope"}, "a": {"class", "href", "data-t", "data-ms", "data-n"}}


class Page(HTMLParser):
    """The page as a list of elements, to check what it contains rather than how it is spelt."""

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tags, self.elements = [], []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.elements.append((tag, dict(attrs)))

    def find(self, tag, **attrs):
        return [a for t, a in self.elements
                if t == tag and all(a.get(k) == v for k, v in attrs.items())]


def headings(text):
    return re.findall(r"<h2>(.*?)</h2>", text)


def test_the_seven_sections_and_the_appendix_in_template_order():
    assert headings(render_html(make_doc())) == [
        "Riepilogo", "Scaletta", "Concetti chiave", "Definizioni", "Esempi e codice",
        "Domande di ripasso", "Punti poco chiari", "Parole da verificare ascoltando"]


def test_headings_follow_the_notes_language():
    english = render_html(make_doc(language="en"))
    assert headings(english)[:3] == ["Summary", "Outline", "Key concepts"]
    assert '<html lang="en">' in english and '<html lang="it">' in render_html(make_doc())
    assert '<html lang="en">' in render_html(make_doc(language="fr"))


def test_empty_sections_are_left_out():
    doc = make_doc()
    text = render_html(replace(doc, content={**doc.content, "examples": [], "open_points": []}))
    assert "Esempi e codice" not in text and "Punti poco chiari" not in text


def test_timestamps_seek_the_second_of_the_line_and_carry_their_exact_time():
    page = Page(render_html(make_doc()))
    first = page.find("a", **{"data-n": "12"})[0]
    assert first["href"] == "#t=66" and first["data-t"] == "66" and first["data-ms"] == "66980"
    assert ">0:01:06</a>" in render_html(make_doc())
    for link in page.find("a", **{"class": "ts"}):
        assert link["href"] == f"#t={link['data-t']}"


def test_a_concept_shows_its_timestamps_in_time_order_one_per_second():
    doc = make_doc()
    refs = doc.content["concepts"][0]["refs"]
    content = {**doc.content, "concepts": [{**doc.content["concepts"][0],
                                            "refs": [refs[1], refs[0], dict(refs[0], n=99)]}]}
    page = Page(render_html(replace(doc, content=content)))
    shown = [(a["data-t"], a["data-n"]) for a in page.find("a", **{"class": "ts"})
             if a.get("data-n") in {"14", "15", "99"}]
    assert shown == [("76", "14"), ("84", "15")]


def test_the_answer_is_hidden_in_details():
    text = render_html(make_doc())
    assert ("<details><summary>Mostra la risposta</summary><p>Quattro step.</p></details>"
            in text)
    assert "Quanti sono gli step?" in text.split("<details>")[0]
    assert "<summary>Show answer</summary>" in render_html(make_doc(language="en"))


def test_code_states_are_shown_next_to_the_item():
    page = Page(render_html(make_doc()))
    assert [f["data-kind"] for f in page.find("span", **{"class": "flag"})] == [
        "check", "unsupported"]
    text = render_html(make_doc())
    assert "da verificare: y_pred" in text
    assert "nome assente dalla trascrizione: misure_notte" in text
    english = render_html(make_doc(language="en"))
    assert "to verify: y_pred" in english
    assert "name not in the transcript: misure_notte" in english


def test_an_example_without_code_has_no_block():
    text = render_html(make_doc())
    assert text.count("<pre>") == 3 and "Solo a parole." in text


def test_backtick_spans_become_code_and_nothing_else_does():
    doc = make_doc()
    content = {**doc.content, "concepts": [{**doc.content["concepts"][0],
                                            "explanation": "usa `x < 1` e **non** <b>questo</b>"}]}
    text = render_html(replace(doc, content=content))
    assert "usa <code>x &lt; 1</code> e **non** &lt;b&gt;questo&lt;/b&gt;" in text


def test_warnings_are_visible():
    page = Page(render_html(make_doc()))
    coverage = page.find("aside", **{"data-kind": "coverage"})
    assert coverage and "hidden" not in coverage[0]
    text = render_html(make_doc())
    assert "potrebbero non coprire" in text and 'data-t="1032"' in text
    assert "trascrizione precedente" not in text
    assert "trascrizione precedente" in render_html(make_doc(line_texts=None))
    assert "5 timestamp non corrispondono" in render_html(make_doc(time_mismatches=5))
    assert "potrebbero non coprire" not in render_html(make_doc(stats={}))


def test_appendix_rows_and_the_word_highlighted_in_its_line():
    text = render_html(make_doc())
    assert '<th scope="col">Probabilità</th>' in text
    assert ("<tr><td>TEC</td><td>S186: stacchiamoli col metodo di <mark>TEC</mark></td>"
            '<td><a class="ts" href="#t=851" data-t="851">0:14:11</a></td><td>0.35</td></tr>'
            ) in text
    without = render_html(make_doc(line_texts=None))
    assert "<tr><td>TEC</td><td>S186</td>" in without and "<mark>" not in without
    assert "Nessuna parola sotto la soglia" in render_html(make_doc(uncertain=()))


def test_the_player_always_points_to_audio_flac_next_to_the_page():
    for present in (True, False):
        page = Page(render_html(make_doc(audio_present=present)))
        assert page.find("audio")[0]["src"] == "audio.flac"
        assert page.find("audio")[0]["preload"] == "metadata"


def test_without_audio_the_page_says_so_with_no_javascript_needed():
    text = render_html(make_doc(audio_present=False))
    banner = Page(text).find("p", id="no-audio")[0]
    assert "hidden" not in banner
    assert "audio.flac non è stato trovato" in text
    assert "studyscribe export 20261102-173400-e04c56 --audio" in text
    present = Page(render_html(make_doc(audio_present=True)))
    assert "hidden" in present.find("p", id="no-audio")[0]
    assert "hidden" in present.find("p", id="audio-error")[0]


def test_content_security_policy_matches_the_inline_style_and_script():
    text = render_html(make_doc())
    policy = Page(text).find("meta", **{"http-equiv": "Content-Security-Policy"})[0]["content"]
    assert policy == content_security_policy()

    def digest(source):
        raw = hashlib.sha256(source.encode()).digest()
        return "'sha256-" + base64.b64encode(raw).decode() + "'"

    style = re.search(r"<style>(.*?)</style>", text, re.DOTALL).group(1)
    script = re.search(r"<script>(.*?)</script>", text, re.DOTALL).group(1)
    assert f"style-src {digest(style)};" in policy
    assert f"script-src {digest(script)};" in policy
    assert "default-src 'none'" in policy and "unsafe" not in policy and "http" not in policy
    assert "base-uri 'none'" in policy and "form-action 'none'" in policy


def test_the_inline_assets_use_nothing_unsafe_or_external():
    css, js = _asset("notes.css"), _asset("notes.js")
    for word in ("http", "innerHTML", "outerHTML", "eval(", "document.write", "XMLHttpRequest",
                 "fetch(", "import(", "localStorage", "WebSocket"):
        assert word not in js, word
    for word in ("http", "@import", "url("):
        assert word not in css, word


def test_hostile_text_is_inert_everywhere():
    text = render_html(hostile_doc())
    page = Page(text)
    assert set(page.tags) <= ALLOWED_TAGS, set(page.tags) - ALLOWED_TAGS
    assert page.tags.count("script") == 1 and page.tags.count("style") == 1
    for tag, attrs in page.elements:
        for name in attrs:
            assert name in ALLOWED_ATTRS.get(tag, set()), (tag, name)
    for link in page.find("a"):
        assert re.fullmatch(r"#t=\d+", link["href"])
    assert [a["src"] for _, a in page.elements if "src" in a] == ["audio.flac"]
    # the hostile strings are in the page as text, not as markup
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "&lt;img src=x onerror=alert(1)&gt;" in text
    assert HOSTILE[0] not in text


def test_hostile_values_never_reach_an_attribute():
    page = Page(render_html(hostile_doc()))
    for tag, attrs in page.elements:
        for value in attrs.values():
            assert "alert" not in (value or ""), (tag, value)
