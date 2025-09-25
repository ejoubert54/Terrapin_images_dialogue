from copy import deepcopy

from story_image_tk_71_2_2_before_light2025_09_24_2 import (
    distribute_extra_shots_after_final_plan,
)


def _base_analysis():
    return {
        "scenes": [
            {"id": "S1", "what_happens": "Words " * 30},
            {"id": "S2", "what_happens": "Words " * 15},
        ]
    }


def _base_captions():
    return {
        "scenes": [
            {"id": "S1", "shots": [], "what_happens": "Words " * 30},
            {"id": "S2", "shots": [], "what_happens": "Words " * 15},
        ]
    }


def test_proportional_allocation_by_words():
    analysis = _base_analysis()
    captions = _base_captions()
    updated = distribute_extra_shots_after_final_plan(
        analysis,
        captions,
        story_text="word " * 200,
        every_words=50,
        min_words=1,
        tolerance=0.0,
        max_total=10,
        dry_run=False,
    )
    scene_one = next(sc for sc in updated["scenes"] if sc["id"] == "S1")
    scene_two = next(sc for sc in updated["scenes"] if sc["id"] == "S2")
    assert sum(1 for shot in scene_one["shots"] if shot.get("extra")) == 3
    assert sum(1 for shot in scene_two["shots"] if shot.get("extra")) == 1


def test_zero_target_when_below_min_words():
    analysis = _base_analysis()
    captions = _base_captions()
    result = distribute_extra_shots_after_final_plan(
        analysis,
        captions,
        story_text="word " * 5,
        every_words=100,
        min_words=1000,
        tolerance=0.0,
        max_total=10,
        dry_run=False,
    )
    assert result is captions


def test_dry_run_returns_original_without_mutation():
    analysis = _base_analysis()
    captions = _base_captions()
    original = deepcopy(captions)
    result = distribute_extra_shots_after_final_plan(
        analysis,
        captions,
        story_text="word " * 200,
        every_words=50,
        min_words=1,
        tolerance=0.0,
        max_total=10,
        dry_run=True,
    )
    assert result is captions
    assert captions == original


def test_max_total_cap_is_enforced():
    analysis = _base_analysis()
    captions = _base_captions()
    updated = distribute_extra_shots_after_final_plan(
        analysis,
        captions,
        story_text="word " * 200,
        every_words=10,
        min_words=1,
        tolerance=0.0,
        max_total=2,
        dry_run=False,
    )
    total_extras = sum(
        1
        for scene in updated["scenes"]
        for shot in scene["shots"]
        if shot.get("extra")
    )
    assert total_extras <= 2
