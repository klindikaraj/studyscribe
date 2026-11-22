"""Hand-made `NotesDoc`s for the renderer tests: every section kind, every code state, an
appendix, and a version of all of it full of hostile text."""

from __future__ import annotations

import copy
from dataclasses import replace

from studyscribe.notes.doc import NotesDoc
from studyscribe.notes.templates import builtin_template
from studyscribe.notes.uncertain import UncertainWord


def r(n: int, ms: int) -> dict:
    return {"n": n, "segment_id": 100 + n, "start_ms": ms}


def _check(level: str, partial=(), missing=()) -> dict:
    return {"level": level, "partial": list(partial), "missing": list(missing)}


CONTENT = {
    "summary": "Prima frase. Seconda frase.",
    "outline": [{"title": "Introduzione", "ref": r(12, 66_980)},
                {"title": "Tensori", "ref": r(53, 268_800)}],
    "concepts": [{"name": "Scatola nera", "explanation": "Un modello riceve un input.",
                  "refs": [r(14, 76_740), r(15, 84_170)]}],
    "definitions": [{"term": "tensor", "definition": "un tipo di dato simile alla lista.",
                     "refs": [r(53, 268_800)]}],
    "examples": [
        {"code": "y_pred = model(x)", "explanation": "Calcola la previsione.",
         "refs": [r(100, 500_000)], "check": _check("ok")},
        {"code": "y_pred.detach()", "explanation": "Stacca il tensore.",
         "refs": [r(186, 851_792)], "check": _check("check", partial=["y_pred"])},
        {"code": "misure_notte = 1", "explanation": "Nome inventato.",
         "refs": [r(190, 900_000)], "check": _check("unsupported", missing=["misure_notte"])},
        {"code": "", "explanation": "Solo a parole.", "refs": [r(200, 950_000)]},
    ],
    "questions": [{"question": "Quanti sono gli step?", "answer": "Quattro step.",
                   "difficulty": "easy", "refs": [r(20, 111_170)]}],
    "open_points": [{"point": "TEC non è chiaro.", "refs": [r(186, 851_792)]}],
}

STATS = {
    "coverage_warnings": [{"block": 2, "last_cited": 225, "uncovered": [226, 241],
                           "start_ms": 1_032_722, "end_ms": 1_087_642}],
    "refs": {"map": {"invalid_refs": 2, "dropped_items": {"concepts": 1}}},
    "input_tokens": 12_167, "output_tokens": 3_584, "wall_s": 46.1,
}

LINES = {186: "stacchiamoli col metodo di TEC", 152: "usiamo Adam come ottimizzatore"}


def make_doc(**changes) -> NotesDoc:
    doc = NotesDoc(
        session_id="20261102-173400-e04c56", title="Lezione PyTorch", course="Reti neurali",
        recorded="2026-11-02 17:34", duration_ms=1_199_907, language="it", notes_id=2,
        version=2, total_versions=2, provider="ollama", model="qwen3.5:9b", effort="none",
        prompt_version="v5.1", finished="2026-11-16 20:39", template=builtin_template(),
        content=copy.deepcopy(CONTENT),
        uncertain=(UncertainWord("Adam", 152, 668_522, 0.34),
                   UncertainWord("TEC", 186, 851_792, 0.35)),
        stats=copy.deepcopy(STATS), line_texts=dict(LINES), time_mismatches=0,
        audio_present=True)
    return replace(doc, **changes)


HOSTILE = [
    "<script>alert(1)</script>",
    '"><img src=x onerror=alert(1)>',
    "</details><b>bold</b>",
    "&amp; &lt;i&gt; <!-- c -->",
    "[x](javascript:alert(1))",
    "`<u>tick</u>`",
    "a | b\nnew line",
]


def hostile_doc() -> NotesDoc:
    """Every piece of text the exports print comes from the doc's fields: all of them hostile."""
    pick = iter(range(10_000))

    def bad() -> str:
        return HOSTILE[next(pick) % len(HOSTILE)] + f" #{next(pick)}"

    content = copy.deepcopy(CONTENT)
    content["summary"] = bad()
    for key, fields in {"outline": ["title"], "concepts": ["name", "explanation"],
                        "definitions": ["term", "definition"],
                        "examples": ["code", "explanation"],
                        "questions": ["question", "answer"], "open_points": ["point"]}.items():
        for item in content[key]:
            for field in fields:
                item[field] = bad()
    for item in content["examples"]:
        if "check" in item:
            item["check"]["partial"] = [bad()] if item["check"]["partial"] else []
            item["check"]["missing"] = [bad()] if item["check"]["missing"] else []
    template = builtin_template()
    sections = tuple(s.model_copy(update={"heading": {"en": bad(), "it": bad()}})
                     for s in template.sections)
    return make_doc(
        title=bad(), course=bad(), model=bad(), effort=bad(), prompt_version=bad(),
        language="it", content=content,
        template=template.model_copy(update={"sections": sections}),
        uncertain=(UncertainWord(bad(), 186, 851_792, 0.35),),
        line_texts={186: HOSTILE[0] + " TEC " + HOSTILE[1]})
