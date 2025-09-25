# Story Tools CLI

This repository packages the analysis helpers from `story_image_tk_71_2_2_before_light2025_09_24_2.py`
into a small command-line interface for headless workflows.

## Installation

```bash
pip install -e .
```

## Usage

Extract dialogue sidecars from a story text file:

```bash
python -m cli dialogue extract --in story.txt --out out/story --known "Alice,Bob"
```

Expand scenes for a processed analysis and preview extra-shot scheduling:

```bash
python -m cli scenes expand --analysis out/story_analysis.json --captions out/captions_map.json --dry-run
```

Run the batch image pipeline using existing stories and profile repositories:

```bash
python -m cli batch run --stories-dir ./stories --profiles-dir ./profiles --out-root ./out --aspect 21:9 --render-n 1
```

The same commands are also available via the `storytools` entry point after installation.
