from typing import Any, Dict, List

from story_image_tk_71_2_2_before_light2025_09_24_2 import (
    DialogueExtractor,
    LLMAssistedAttributor,
    _apply_llm_alternation,
    _compute_scene_image_priors,
    _split_narration_for_llm,
)


class _DummyClient:
    def __init__(self, responses: List[Dict[str, Any]]):
        self._responses = responses

    def chat_json(self, **_: Any) -> Dict[str, Any]:
        if not self._responses:
            return {"results": []}
        return self._responses.pop(0)


def test_llm_propose_filters_to_closed_set():
    text = 'Alice said, "Move!" "Hold," he rasped.'
    extractor = DialogueExtractor(known_characters=["Alice", "Bob"], aliases={"Alice": ["Al"]})
    utterances = extractor.extract(text)
    target = next(u for u in utterances if u["line"].startswith('"Hold'))
    start, end = target["char_span"]
    candidate = {
        "utterance_id": target["utterance_id"],
        "phase": "speech",
        "line": target["line"],
        "char_span": [start, end],
        "context_span": [max(0, start - 10), min(len(text), end + 10)],
        "scene_id": "S1",
        "scene_characters": ["Alice", "Bob"],
        "image_priors": {"Alice": 0.6},
    }
    name_idx = text.index("Alice")
    verb_idx = text.index("rasped")
    responses = [
        {
            "results": [
                {
                    "utterance_id": target["utterance_id"],
                    "character": "Alice",
                    "confidence": 0.91,
                    "evidence": {
                        "name_span": [name_idx, name_idx + 5],
                        "verb_span": [verb_idx, verb_idx + 6],
                        "rationale": "Narrator names Alice",
                    },
                },
                {
                    "utterance_id": target["utterance_id"],
                    "character": "Charlie",
                    "confidence": 0.95,
                    "evidence": {},
                },
            ]
        }
    ]
    client = _DummyClient(responses)
    agent = LLMAssistedAttributor(
        known_characters=["Alice", "Bob"],
        aliases={"Alice": ["Al"]},
        conf_threshold=0.8,
        batch_size=1,
        client=client,
        scene_rosters={"S1": ["Alice", "Bob"]},
        image_priors={"S1": {"Alice": 0.6}},
    )
    proposals = agent.propose(text, [candidate])
    assert len(proposals) == 1
    assert proposals[0]["character"] == "Alice"
    assert proposals[0]["confidence"] >= 0.9
    assert "name_span" in proposals[0]["evidence"]


def test_alternation_helper_fills_followups():
    utterances = [
        {"character": "Alice", "line": '"A"', "attribution": {"score": 0.95}},
        {"character": "Bob", "line": '"B"', "attribution": {"score": 0.94}},
        {"character": "UNATTRIBUTED", "line": '"C"', "attribution": {"score": 0.0}},
        {"character": "UNATTRIBUTED", "line": '"D"', "attribution": {"score": 0.0}},
    ]
    _apply_llm_alternation(utterances)
    assert utterances[2]["character"] == "Alice"
    assert utterances[3]["character"] == "Bob"
    assert utterances[2]["attribution"]["method"] == "alternation_llm"


def test_narration_split_caps_segment_length():
    text = ("The officer said they would hold the line. " * 12).strip()
    segments = _split_narration_for_llm(text, (0, len(text)), 240)
    assert segments
    for start, end in segments:
        segment_text = text[start:end]
        assert len(segment_text) <= 240 + 5  # allow minor whitespace overhead


def test_image_priors_aggregator_merges_sources():
    scenes = [
        {"id": "S1", "image_priors": {"Alice": 0.4, "Bob": 0.2}},
        {
            "id": "S2",
            "images": [
                {"character_scores": {"Alice": 0.6}},
                {"characters": ["Bob"]},
            ],
        },
    ]
    priors = _compute_scene_image_priors(scenes, {"Alice", "Bob"})
    assert priors["S1"]["Alice"] == 0.4
    assert 0.7 <= priors["S2"]["Bob"] <= 1.0
