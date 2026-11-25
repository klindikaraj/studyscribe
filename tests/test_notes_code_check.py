import pytest

from studyscribe.notes.code_check import annotate_code, check_code, cited_texts, vocabulary
from studyscribe.notes.transcript import Line


@pytest.mark.parametrize("code, cited", [
    # code form of words that were said
    ("y_pred = model(X)", ["calcoliamo ipsilon pred passando la X al model"]),
    ("torch.nn.Linear(1, 1)", ["usiamo PyTorch.neuralNetwork.Linear con uno e uno"]),
    ("firstName[firstName.length - 1]", ["first name bracket first name dot length minus one"]),
    ("var myName = \"Beau Carnes\"", ["keyword var", "so I can set my name to be Beau"]),
    ("nn.MSELoss()", ["la funzione mean squared error loss"]),  # MSE: initials
    ("x += 1", ["x plus equals one"]),  # nothing checkable
    ("Object.freeze(MATH_CONSTANTS)", ["object dot freeze", "math constants"]),
])
def test_code_form_of_said_words_is_ok(code, cited):
    assert check_code(code, cited).level == "ok"


def test_a_name_with_some_words_not_said_is_to_check():
    result = check_code("scores_total.push(x)", ["we push the new value onto scores"])
    assert result.level == "check" and result.partial == ("scores_total",)
    assert result.missing == ()


@pytest.mark.parametrize("code, cited, missing", [
    ("y_prediatti", ["e gli y previsti dal modello"], ("y_prediatti",)),
    ("misure_notte = torch.cat(a, b)", ["uniamo i due tensori"], ("misure_notte", "torch", "cat")),
    ("torch.optim.Adam(model.parameters(), lr=0.01)",
     ["uno dei più usati si chiama Adam"], ("torch", "optim", "model", "parameters")),
])
def test_names_with_no_word_said_are_unsupported(code, cited, missing):
    result = check_code(code, cited)
    assert result.level == "unsupported"
    assert result.missing == missing


def test_a_name_said_elsewhere_in_the_lecture_is_to_check_not_unsupported():
    cited = ["salviamoli sulla variabile y previsti"]
    lecture = vocabulary(["occhio che su YPRED ci sono gli output", *cited])
    assert check_code("y_pred = 1", cited).level == "unsupported"
    result = check_code("y_pred = 1", cited, lecture)
    assert result.level == "check" and result.partial == ("y_pred",)
    assert check_code("optimizer", cited, lecture | {"ottimizzatore"}).level == "unsupported"


def test_string_literals_and_keywords_are_not_checked():
    assert check_code('print("Zebra quokka")', ["stampiamo"]).level == "ok"
    assert check_code("return None if x else undefined", []).level == "ok"


def test_a_two_letter_word_is_not_a_three_letter_name():
    # "se" is 80% similar to "mse" (piece 5, it-copy)
    result = check_code("nn.MSELoss()", ["se la loss scende"])
    assert result.level == "check" and result.partial == ("MSELoss",)
    assert check_code("MSE", ["se scende"], vocabulary(["se"])).level == "unsupported"
    assert check_code("pred", ["ipsilon pre"]).level == "ok"  # three letters still count


@pytest.mark.parametrize("code, cited, level, partial", [
    ("const lookup = {}", "I'm going to create bar lookup", "check", ("const",)),
    ("const total = 1", "var total equals one", "check", ("const",)),
    ("var total = 1", "var total equals one", "ok", ()),
    ("let total = 1", "Let total equals one", "ok", ()),
    ("let total = 1", "let's say total equals one", "check", ("let",)),
    ("let total = 1", "let\u2019s say total equals one", "check", ("let",)),
    ("var total = 1", "the variable total is one", "check", ("var",)),
    ("for (var i = 0; i < 5; i++) {}", "a for loop", "check", ("var",)),
])
def test_var_let_and_const_must_have_been_said(code, cited, level, partial):
    result = check_code(code, [cited])
    assert (result.level, result.partial, result.missing) == (level, partial, ())


def test_a_declaration_said_elsewhere_in_the_lecture_is_still_to_check():
    result = check_code("const total = 1", ["total is one"], vocabulary(["use const"]))
    assert result.level == "check" and result.partial == ("const",)


def test_known_blind_spot_close_invented_names_pass():
    assert check_code("inputi", ["gli input del modello"]).level == "ok"


def test_cited_texts_take_one_line_on_each_side_once():
    lines = {n: Line(n, n, 0, 1, f"t{n}") for n in range(1, 8)}
    assert cited_texts([3, 4], lines) == ["t2", "t3", "t4", "t5"]
    assert cited_texts([1, 7], lines) == ["t1", "t2", "t6", "t7"]
    assert cited_texts([], lines) == []


def test_annotate_code_adds_a_check_to_code_items_and_counts_levels():
    lines = {1: Line(1, 1, 0, 1, "intro"), 2: Line(2, 2, 0, 1, "we call scores dot push"),
             3: Line(3, 3, 0, 1, "next"), 9: Line(9, 9, 0, 1, "unrelated words")}
    notes = {"summary": "s", "examples": [
        {"code": "scores.push(x)", "explanation": "e", "refs": [2]},
        {"code": "", "explanation": "a worked example", "refs": [2]},
        {"code": "zebraCount++", "explanation": "e", "refs": [9]},
    ]}
    annotated, counts = annotate_code(notes, ["examples"], lines)
    assert annotated["examples"][0]["check"] == {"level": "ok", "partial": [], "missing": []}
    assert "check" not in annotated["examples"][1]
    assert annotated["examples"][2]["check"]["level"] == "unsupported"
    assert counts == {"ok": 1, "check": 0, "unsupported": 1}
    assert "check" not in notes["examples"][0]  # the input is not changed
    assert annotated["summary"] == "s"
