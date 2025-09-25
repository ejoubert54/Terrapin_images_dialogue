from story_image_tk_71_2_2_before_light2025_09_24_2 import DialogueExtractor


def _extract_lines(text: str, **kwargs):
    extractor = DialogueExtractor(**kwargs)
    return extractor.extract(text)


def _speech(utterances):
    return [u for u in utterances if u["character"] != "Narrator"]


def test_postposed_tag_assignment():
    text = '"We leave at dawn," said Alice.'
    utterances = _extract_lines(text, known_characters=["Alice"], aliases={}, mode="strict")
    speech = _speech(utterances)
    assert speech[0]["character"] == "Alice"
    assert speech[0]["attribution"]["method"] == "postposed_tag"
    assert speech[0]["attribution"]["score"] >= 0.90
    assert speech[0]["line"].startswith('"We leave at dawn')


def test_preposed_tag_assignment():
    text = 'Alice said, "We leave at dawn."'
    utterances = _extract_lines(text, known_characters=["Alice"], aliases={}, mode="strict")
    speech = _speech(utterances)
    assert speech[0]["character"] == "Alice"
    assert speech[0]["attribution"]["method"] in {"preposed_tag", "merged_narrator"}
    assert speech[0]["attribution"]["score"] >= 0.90


def test_interruption_assigns_both_quotes():
    text = '"Fine," Alice said, "but bring the maps."'
    utterances = _extract_lines(text, known_characters=["Alice"], aliases={}, mode="strict")
    speech = _speech(utterances)
    assert len(speech) == 2
    for item in speech:
        assert item["character"] == "Alice"
        assert item["attribution"]["method"] in {"interruption", "merged_narrator"}
        assert item["attribution"]["score"] >= 0.90


def test_emdash_and_script_label_detection():
    text = 'ALICE: Ready your gear.'
    utterances = _extract_lines(text, known_characters=["Alice"], aliases={}, mode="strict")
    alice_line = next(u for u in _speech(utterances) if u["character"] == "Alice")
    assert alice_line["attribution"]["method"] == "script_label"
    dash_text = '— Hold position.'
    dash_output = _extract_lines(dash_text, known_characters=["Alice"], aliases={}, mode="strict")
    dash_speech = _speech(dash_output)
    assert dash_speech[0]["character"] == "UNATTRIBUTED"
    assert dash_speech[0]["attribution"]["score"] == 0.0


def test_closed_set_enforcement():
    text = '"Hello," said Carol.'
    utterances = _extract_lines(text, known_characters=["Alice", "Bob"], aliases={}, mode="strict")
    speech = _speech(utterances)
    assert speech[0]["character"] == "UNATTRIBUTED"
    assert speech[0]["attribution"]["method"] == "none"


def test_narrator_wrapping_respects_limit():
    text = (
        "The night was long and cold. The fire burned low in the abandoned keep while the wind howled outside. "
        '"Hush," Alice whispered.'
    )
    utterances = _extract_lines(
        text,
        known_characters=["Alice"],
        aliases={},
        mode="strict",
        max_narrator_chars=40,
    )
    narrator_segments = [u for u in utterances if u["character"] == "Narrator"]
    assert len(narrator_segments) >= 2
    assert any(len(seg["line"]) <= 40 for seg in narrator_segments)
    speech = _speech(utterances)
    assert speech[0]["character"] == "Alice"
