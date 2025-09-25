from story_image_tk_71_2_2_before_light2025_09_24_2 import (
    clean_json_from_text,
    safe_json_loads,
)


def test_clean_json_removes_fences():
    raw = "```json\n{\"foo\": 1}\n```"
    cleaned = clean_json_from_text(raw)
    assert cleaned == '{"foo": 1}'
    data = safe_json_loads(raw)
    assert data == {"foo": 1}


def test_safe_json_handles_trailing_commas():
    data = safe_json_loads('{"foo": 1,}')
    assert data == {"foo": 1}


def test_control_characters_are_stripped():
    raw = '{"foo": "bar\x00baz"}'
    cleaned = clean_json_from_text(raw)
    assert "\x00" in cleaned
    data = safe_json_loads(cleaned.replace("\x00", ""))
    assert data == {"foo": "barbaz"}


def test_plain_string_promoted_to_title():
    data = safe_json_loads('"Victory"')
    assert data == {"title": "Victory"}
