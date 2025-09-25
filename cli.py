"""Command-line interface for story analysis utilities."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from story_image_tk_71_2_2_before_light2025_09_24_2 import (
    App,
    _maybe_expand_scenes,
    extract_and_save_dialogue,
)

LOGGER = logging.getLogger("storytools")


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def _parse_known_characters(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",")]
    cleaned = [p for p in parts if p]
    return cleaned or None


def _parse_aliases(raw: Optional[str]) -> Optional[Dict[str, List[str]]]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON for --aliases: {exc}")
    if not isinstance(data, dict):
        raise SystemExit("--aliases must decode to an object mapping")
    result: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            aliases = [str(item) for item in value if str(item).strip()]
        else:
            aliases = [str(value).strip()] if str(value).strip() else []
        if aliases:
            result[str(key)] = aliases
    return result or None


def _require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for this command")
    return api_key


def _command_dialogue_extract(args: argparse.Namespace) -> int:
    if args.llm_assist:
        _require_api_key()
    known = _parse_known_characters(args.known)
    aliases = _parse_aliases(args.aliases)
    story_path = Path(args.input)
    if not story_path.is_file():
        raise SystemExit(f"Story file not found: {story_path}")
    story_text = story_path.read_text(encoding="utf-8")
    base_path = Path(args.output)
    if base_path.suffix:
        base_path = base_path.with_suffix("")
    result = extract_and_save_dialogue(
        story_text=story_text,
        base_output_path=str(base_path),
        known_characters=known,
        character_aliases=aliases,
        voices_map=None,
        mode=args.mode,
        confidence_threshold=float(args.confidence_threshold),
        use_llm_assist=bool(args.llm_assist),
        llm_conf_threshold=float(args.llm_threshold),
        llm_batch_size=int(args.llm_batch),
        max_narrator_chars=(None if args.max_narrator is None else int(args.max_narrator)),
        story_analysis=None,
        llm_scene_bias=bool(getattr(args, "llm_scene_bias", True)),
        llm_image_bias=bool(getattr(args, "llm_image_bias", True)),
    )
    LOGGER.info("Dialogue sidecars written:")
    LOGGER.info("  %s", result["txt_path"])
    LOGGER.info("  %s", result["json_path"])
    return 0


def _command_scenes_expand(args: argparse.Namespace) -> int:
    analysis_path = Path(args.analysis)
    captions_path = Path(args.captions)
    if not analysis_path.is_file():
        raise SystemExit(f"Analysis JSON not found: {analysis_path}")
    if not captions_path.is_file():
        raise SystemExit(f"Captions map JSON not found: {captions_path}")
    _maybe_expand_scenes(
        str(analysis_path),
        str(captions_path),
        dry_run=bool(args.dry_run),
        extra_dry_run=bool(args.extra_dry_run),
    )
    from story_image_tk_71_2_2_before_light2025_09_24_2 import (
        LAST_EXPAND_SCENES_STATUS,
        LAST_EXTRA_SHOT_REPORT,
    )

    summary = LAST_EXPAND_SCENES_STATUS or "Scene expansion complete"
    extras = LAST_EXTRA_SHOT_REPORT or {}
    extras_tag = extras.get("target_extras") if extras.get("dry_run") else extras.get("added_total")
    if extras_tag is not None:
        summary += f"; extras={extras_tag}"
    LOGGER.info(summary)
    return 0


def _command_batch_run(args: argparse.Namespace) -> int:
    _require_api_key()
    app = App(root=None, headless=True)
    kwargs: Dict[str, Any] = {
        "stories_dir": args.stories_dir,
        "profiles_dir": args.profiles_dir,
        "out_root": args.out_root,
        "aspect": args.aspect,
        "render_n": int(args.render_n),
        "prompt_policy": args.policy,
        "char_views": tuple(_split_csv(args.char_views)) if args.char_views else ("front", "profile_left", "profile_right"),
        "char_per_view": int(args.char_per_view),
        "loc_views": tuple(_split_csv(args.loc_views)) if args.loc_views else ("establishing", "alt_angle"),
        "loc_per_view": int(args.loc_per_view),
        "min_words_per_image": (int(args.min_words_per_image) if args.min_words_per_image is not None else None),
    }
    app.run_batch_on_folder(**kwargs)
    LOGGER.info("Batch run completed")
    return 0


def _split_csv(value: Optional[str]) -> Iterable[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="storytools", description="Story tooling CLI")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command")

    dialogue_parser = subparsers.add_parser("dialogue", help="Dialogue utilities")
    dialogue_sub = dialogue_parser.add_subparsers(dest="dialogue_cmd")

    extract_parser = dialogue_sub.add_parser("extract", help="Extract dialogue sidecars")
    extract_parser.add_argument("--in", dest="input", required=True, help="Input story text file")
    extract_parser.add_argument("--out", dest="output", required=True, help="Output path stem")
    extract_parser.add_argument("--known", dest="known", help="Comma-separated list of known characters")
    extract_parser.add_argument("--aliases", dest="aliases", help="JSON mapping of aliases")
    extract_parser.add_argument("--mode", choices=("strict", "permissive"), default="strict")
    extract_parser.add_argument("--ct", dest="confidence_threshold", default=0.90, type=float)
    extract_parser.add_argument("--max-narrator", dest="max_narrator", type=int)
    extract_parser.add_argument("--llm-assist", action="store_true", dest="llm_assist")
    extract_parser.add_argument("--llm-threshold", dest="llm_threshold", default=0.83, type=float)
    extract_parser.add_argument("--llm-batch", dest="llm_batch", default=8, type=int)
    extract_parser.add_argument(
        "--no-llm-scene-bias",
        dest="llm_scene_bias",
        action="store_false",
        help="disable scene roster hints during the LLM pass",
        default=True,
    )
    extract_parser.add_argument(
        "--no-llm-image-bias",
        dest="llm_image_bias",
        action="store_false",
        help="disable image-derived hints during the LLM pass",
        default=True,
    )

    extract_parser.set_defaults(func=_command_dialogue_extract)

    scenes_parser = subparsers.add_parser("scenes", help="Scene tools")
    scenes_sub = scenes_parser.add_subparsers(dest="scenes_cmd")

    expand_parser = scenes_sub.add_parser("expand", help="Expand scenes using analysis")
    expand_parser.add_argument("--analysis", required=True, help="Path to *_analysis.json")
    expand_parser.add_argument("--captions", required=True, help="Path to captions_map.json")
    expand_parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    expand_parser.add_argument("--extra-dry-run", action="store_true", dest="extra_dry_run")
    expand_parser.set_defaults(func=_command_scenes_expand)

    batch_parser = subparsers.add_parser("batch", help="Batch rendering tools")
    batch_sub = batch_parser.add_subparsers(dest="batch_cmd")

    run_parser = batch_sub.add_parser("run", help="Run batch processing on a folder")
    run_parser.add_argument("--stories-dir", required=True)
    run_parser.add_argument("--profiles-dir", required=True)
    run_parser.add_argument("--out-root", required=True)
    run_parser.add_argument("--aspect", default="21:9")
    run_parser.add_argument("--render-n", default=1, type=int)
    run_parser.add_argument("--policy", choices=("final_prompt", "min_prompt"), default="final_prompt")
    run_parser.add_argument("--char-views", dest="char_views")
    run_parser.add_argument("--char-per-view", dest="char_per_view", default=1, type=int)
    run_parser.add_argument("--loc-views", dest="loc_views")
    run_parser.add_argument("--loc-per-view", dest="loc_per_view", default=1, type=int)
    run_parser.add_argument("--min-words-per-image", dest="min_words_per_image", type=int)
    run_parser.set_defaults(func=_command_batch_run)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_logging(verbose=getattr(args, "verbose", False))
    func = getattr(args, "func", None)
    if not func:
        parser.print_help()
        return 1
    return func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
