from xai_llm.segment import render, segment


def words(text):
    units, _ = segment(text)
    return [u.text for u in units]


def test_basic_words():
    assert words("The capital of France is") == ["The", "capital", "of", "France", "is"]


def test_punctuation_is_its_own_unit():
    assert words("Hello, world!") == ["Hello", ",", "world", "!"]


def test_apostrophes_hyphens_and_decimals_stay_inside_words():
    assert words("don't stop-me 3.14") == ["don't", "stop-me", "3.14"]
    assert words("2+2=") == ["2", "+", "2", "="]


def test_indic_script_words_stay_whole():
    ws = words("भारत की राजधानी क्या है?")
    assert ws == ["भारत", "की", "राजधानी", "क्या", "है", "?"]
    assert words("ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು") == ["ಭಾರತದ", "ರಾಜಧಾನಿ", "ಯಾವುದು"]


def test_render_round_trip_keeps_original_text():
    for text in ["The capital of France is", "  leading and trailing  ", "Hello,  world!\nNext line "]:
        units, trailing = segment(text)
        assert render(units, [True] * len(units), trailing) == text


def test_removing_words():
    units, tr = segment("The capital of France is")
    keep = [u.text != "capital" for u in units]
    assert render(units, keep, tr) == "The of France is"
    keep = [u.text != "The" for u in units]
    assert render(units, keep, tr) == "capital of France is"  # no stray leading space


def test_trailing_space_is_preserved():
    units, tr = segment("The capital of France is ")
    assert tr == " "
    assert render(units, [True] * len(units), tr).endswith("is ")
