#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import io
import re
import json
import base64
import time
import hashlib
import threading
import copy
import shutil
import glob
import math
import statistics
import bisect
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import sys


@dataclass
class DialogueCue:
    order: int
    speaker: str              # "Narrator" or character name
    text: str                 # exact line (no markers)
    emotion: str              # coarse tag
    speaker_conf: float       # 0..1 heuristic confidence
    emotion_conf: float       # 0..1 heuristic confidence
    start_time: Optional[float] = None  # filled later by Caption_App
    end_time: Optional[float] = None


_SAY_VERBS = r"(said|asked|whispered|yelled|shouted|murmured|replied|answered|cried|muttered|hissed|snapped|breathed|insisted|added|noted|admitted)"
_EMO_HINTS = {
    "angry":   ["angry","furious","irate","rage","snapped","shouted","yelled"],
    "sad":     ["sad","sorrow","tear","cry","sob","mourn","lament"],
    "happy":   ["happy","glad","smile","grin","joy","delight"],
    "fear":    ["afraid","scared","fear","terrified","panic"],
    "surprise":["surprise","astonish","shock","gasp"],
    "whisper": ["whisper","hushed","low voice","murmur"],
    "neutral": []
}


def _strip_quotes(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^[\“\"'`]+", "", s)
    s = re.sub(r"[\”\"'`]+$", "", s)
    return s.strip()


def _find_inline_emotion(text: str) -> Tuple[Optional[str], float]:
    # explicit author tags: [angry], (whisper), {sadly}
    m = re.search(r"[\[\(\{]\s*([a-z]{3,20})\s*[\]\)\}]", text.lower())
    if m:
        return m.group(1), 0.95
    # keyword heuristics
    tl = text.lower()
    best, score = None, 0.0
    for emo, keys in _EMO_HINTS.items():
        if not keys:
            continue
        hits = sum(1 for k in keys if k in tl)
        if hits > 0 and hits >= score:
            best, score = emo, float(min(1.0, 0.6 + 0.1 * hits))
    # punctuation cues
    if "!" in text and (best is None or score < 0.7):
        return (best or "angry"), max(score, 0.7)
    if "?" in text and (best is None or score < 0.6):
        return (best or "surprise"), max(score, 0.6)
    return (best or "neutral"), max(score, 0.4 if best else 0.5)


def _title_case_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).title()


def _guess_speaker_from_tail(tail: str) -> Optional[str]:
    # e.g., “Hello,” Alice said.  |  “Hello.” Bob asked.
    m = re.search(rf"\b([A-Z][A-Za-z\.\- ]{{1,40}})\s+{_SAY_VERBS}\b", tail)
    if m:
        return _title_case_name(m.group(1))
    # e.g., “Hello,” said Alice.
    m = re.search(rf"\b{_SAY_VERBS}\s+([A-Z][A-Za-z\.\- ]{{1,40}})\b", tail)
    if m:
        return _title_case_name(m.group(1))
    return None


def _guess_speaker_from_label(prefix: str) -> Optional[str]:
    # e.g., Alice: “Hello”   OR   ALICE — “Hello”
    m = re.search(r"^\s*([A-Z][A-Za-z\.\- ]{1,40})\s*[:\-—]\s*$", prefix)
    if m:
        return _title_case_name(m.group(1))
    return None


def _extract_quote_spans(text: str) -> List[Tuple[int, int, str, str]]:
    """
    Return list of (start, end, quote_text, trailing_context).
    Quotes use “ ” or " " or ' ' styles. Greedy enough for prose.
    """
    spans = []
    # Handle curly quotes first, then straight quotes
    patterns = [
        r"“([^”]+)”",           # curly
        r"\"([^\"]+)\"",        # double
        r"\'([^\']+)\'"          # single
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.DOTALL):
            q = m.group(1)
            tail = text[m.end(): m.end() + 120]  # a bit after quote to find 'X said'
            spans.append((m.start(), m.end(), q, tail))
    spans.sort(key=lambda t: t[0])
    return spans


def _segment_lines(text: str) -> List[str]:
    # A light segmenter that keeps paragraph breaks, but allows per-line tagging.
    parts = re.split(r"(\n{2,})", text)
    out = []
    for p in parts:
        if p.strip() == "":
            out.append(p)
        elif p.startswith("\n"):
            out.append(p)
        else:
            # split long paragraphs into sentence-ish lines
            sents = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9“\"'(])", p.strip())
            out.extend(sents)
    return out


# -------------------------------------------------------------
# Hybrid dialogue extraction & attribution (rule-first, LLM optional)
# -------------------------------------------------------------

def _now_utc() -> int:
    return int(time.time())


def _ensure_path_stem(path: str) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    else:
        os.makedirs(".", exist_ok=True)
    return path


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


class DialogueExtractor:
    """
    Deterministic extractor that segments dialogue/narration spans and
    attributes speakers via rule-based scoring. Optional permissive mode
    unlocks alternation/pronoun heuristics.
    """

    DIALOGUE_VERBS = {
        "said",
        "asked",
        "replied",
        "answered",
        "whispered",
        "shouted",
        "yelled",
        "murmured",
        "muttered",
        "exclaimed",
        "cried",
        "called",
        "noted",
        "added",
        "continued",
        "insisted",
        "agreed",
        "snapped",
        "barked",
        "growled",
        "sighed",
        "responded",
        "remarked",
        "stated",
    }

    QUOTE_PAIRS = [
        ("\"", "\""),
        ("“", "”"),
        ("„", "“"),
        ("«", "»"),
        ("‹", "›"),
        ("'", "'"),
    ]

    DASH_PREFIXES = ["—", "–"]

    NAME_RE = r"(?:(?:Dr|Mr|Mrs|Ms|Mx|Prof)\.\s+)?[A-Z][a-z]+(?:[-\s][A-Z][a-z]+){0,2}|[A-Z]{2,}(?:\s+[A-Z]{2,})?"

    def __init__(
        self,
        known_characters: Optional[List[str]] = None,
        aliases: Optional[Dict[str, List[str]]] = None,
        mode: str = "strict",
        confidence_threshold: float = 0.90,
        max_narrator_chars: Optional[int] = None,
    ) -> None:
        self.known = [*(known_characters or [])]
        self.aliases = {k: set(v) for k, v in (aliases or {}).items()}
        self.mode = (mode or "strict").lower()
        self.ct = float(confidence_threshold or 0.0)
        self.max_narrator = max_narrator_chars if max_narrator_chars is None else int(max_narrator_chars)
        self._build_regexes()
        self._line_starts: List[int] = []
        self._text: str = ""
        self._name_counts: Dict[str, int] = {}
        self.known_lookup = {name.lower(): name for name in self.known}
        self.alias_lookup: Dict[str, str] = {}
        for base, vals in self.aliases.items():
            canonical = base
            if canonical.lower() not in self.alias_lookup:
                self.alias_lookup[canonical.lower()] = canonical
            for alias in vals:
                if isinstance(alias, str):
                    self.alias_lookup[alias.lower()] = canonical
        for name in self.known:
            self.alias_lookup.setdefault(name.lower(), name)

    def _build_regexes(self) -> None:
        verb_alt = r"(?:" + "|".join(sorted(map(re.escape, self.DIALOGUE_VERBS))) + r")"
        self._verb_alt = verb_alt
        name = self.NAME_RE
        self.re_script = re.compile(r"^(?P<name>" + name + r")\s*:\s*(?P<body>.+)$", re.MULTILINE)
        self.re_emdash = re.compile(r"^(?:\t|\s)*(—|–)\s*(?P<body>.+)$", re.MULTILINE)
        self.re_post = re.compile(
            r"[\"“„«‹'](?P<q>.+?)[\"””“»›']\s*[\.,;:?!—-]*\s*(?:,?\s*)?(?:\\)?\s*(?:-\s*)?(?:"
            + verb_alt
            + r")\s+(?P<name>"
            + name
            + r")(?:\s+\w+)?",
            re.IGNORECASE | re.DOTALL,
        )
        self.re_intr = re.compile(
            r"(?:\"|“|„|«|‹)(?P<q1>.+?)(?:\"|”|“|»|›)\s*,\s*(?P<name>"
            + name
            + r")\s+(?:"
            + verb_alt
            + r")\s*,\s*(?:\"|“|„|«|‹)(?P<q2>.+?)(?:\"|”|“|»|›)",
            re.IGNORECASE | re.DOTALL,
        )

    def _compute_line_starts(self, text: str) -> List[int]:
        starts = [0]
        idx = text.find("\n")
        pos = 0
        while idx != -1:
            starts.append(idx + 1)
            pos = idx + 1
            idx = text.find("\n", pos)
        return starts

    def _line_for_index(self, index: int) -> int:
        if not self._line_starts:
            return 1
        pos = bisect.bisect_right(self._line_starts, index) - 1
        return max(1, pos + 1)

    @staticmethod
    def _clean_line(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    def _count_names(self, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for match in re.finditer(self.NAME_RE, text):
            name = self._clean_line(match.group(0))
            if not name:
                continue
            canonical = name if name.isupper() else name.title()
            counts[canonical] = counts.get(canonical, 0) + 1
        return counts

    def _canonicalize_name(self, raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        cleaned = self._clean_line(raw)
        if not cleaned:
            return None
        key = cleaned.lower()
        if self.known:
            if key in self.alias_lookup:
                return self.alias_lookup[key]
            if key in self.known_lookup:
                return self.known_lookup[key]
            for name in self.known:
                if name.lower() == key:
                    return name
            return None
        if key in self.alias_lookup:
            return self.alias_lookup[key]
        candidate = cleaned if cleaned.isupper() else cleaned.title()
        if self._name_counts.get(candidate, 0) >= 2:
            return candidate
        return None

    def extract(self, text: str) -> List[Dict[str, Any]]:
        self._text = text or ""
        self._line_starts = self._compute_line_starts(self._text)
        self._name_counts = self._count_names(self._text)

        speeches = self._detect_speech(self._text)
        utterances = self._interleave_narrator(self._text, speeches, self._line_starts)
        self._attribute_all(self._text, utterances)
        if self.mode == "permissive":
            self._apply_permissive_rules(self._text, utterances)
        self._enforce_closed_set(utterances)
        if isinstance(self.max_narrator, int):
            utterances = self._wrap_narrator_spans(utterances)
        for idx, utterance in enumerate(utterances, start=1):
            utterance["utterance_id"] = f"dlg_{idx:06d}"
        for utterance in utterances:
            utterance.pop("_speech_type", None)
            utterance.pop("_content", None)
            utterance.pop("_script_name", None)
        return utterances

    def _detect_speech(self, text: str) -> List[Dict[str, Any]]:
        speeches: List[Dict[str, Any]] = []
        used_ranges: List[Tuple[int, int]] = []

        def ranges_overlap(start: int, end: int) -> bool:
            for a, b in used_ranges:
                if start < b and end > a:
                    return True
            return False

        for match in self.re_script.finditer(text):
            body_start = match.start("body")
            body_end = match.end("body")
            line = self._clean_line(text[body_start:body_end])
            if not line:
                continue
            speeches.append(
                {
                    "start": body_start,
                    "end": body_end,
                    "text": line,
                    "_speech_type": "script",
                    "_script_name": match.group("name"),
                    "_content": line,
                }
            )
            used_ranges.append((match.start(), match.end()))

        for match in self.re_emdash.finditer(text):
            body_start = match.start("body")
            body_end = match.end("body")
            if ranges_overlap(body_start, body_end):
                continue
            line = self._clean_line(text[body_start:body_end])
            if not line:
                continue
            speeches.append(
                {
                    "start": body_start,
                    "end": body_end,
                    "text": line,
                    "_speech_type": "emdash",
                    "_content": line,
                }
            )
            used_ranges.append((match.start(), match.end()))

        for open_q, close_q in self.QUOTE_PAIRS:
            pattern = re.compile(re.escape(open_q) + r"(.*?)" + re.escape(close_q), re.DOTALL)
            pos = 0
            while True:
                match = pattern.search(text, pos)
                if not match:
                    break
                start = match.start()
                end = match.end()
                pos = match.start() + 1
                if ranges_overlap(start, end):
                    continue
                inner = self._clean_line(match.group(1))
                if not inner:
                    continue
                if " " not in inner and inner.lower() == inner and inner.isalpha():
                    continue
                speeches.append(
                    {
                        "start": start,
                        "end": end,
                        "text": self._clean_line(text[start:end]),
                        "_speech_type": "quote",
                        "_content": inner,
                    }
                )
                used_ranges.append((start, end))

        speeches.sort(key=lambda item: item["start"])
        for idx, item in enumerate(speeches):
            item["_index"] = idx
        return speeches

    def _interleave_narrator(
        self, text: str, speeches: List[Dict[str, Any]], line_starts: List[int]
    ) -> List[Dict[str, Any]]:
        utterances: List[Dict[str, Any]] = []
        cursor = 0
        for seg in speeches:
            start, end = seg["start"], seg["end"]
            if start > cursor:
                chunk = text[cursor:start]
                cleaned = self._clean_line(chunk)
                if cleaned:
                    utterances.append(
                        {
                            "character": "Narrator",
                            "line": cleaned,
                            "char_span": [cursor, start],
                            "source_line": self._line_for_index(cursor),
                            "attribution": {
                                "method": "none",
                                "score": 1.0,
                                "evidence": "narration",
                                "name_candidate": None,
                            },
                        }
                    )
            line = seg.get("text", "")
            cleaned_line = self._clean_line(line)
            if cleaned_line:
                utterances.append(
                    {
                        "character": "UNATTRIBUTED",
                        "line": cleaned_line,
                        "char_span": [start, end],
                        "source_line": self._line_for_index(start),
                        "attribution": {
                            "method": "none",
                            "score": 0.0,
                            "evidence": "",
                            "name_candidate": None,
                        },
                        "_speech_type": seg.get("_speech_type"),
                        "_content": seg.get("_content", cleaned_line),
                        "_script_name": seg.get("_script_name"),
                    }
                )
            cursor = max(cursor, end)
        if cursor < len(text):
            chunk = text[cursor:]
            cleaned = self._clean_line(chunk)
            if cleaned:
                utterances.append(
                    {
                        "character": "Narrator",
                        "line": cleaned,
                        "char_span": [cursor, len(text)],
                        "source_line": self._line_for_index(cursor),
                        "attribution": {
                            "method": "none",
                            "score": 1.0,
                            "evidence": "narration",
                            "name_candidate": None,
                        },
                    }
                )
        return utterances

    def _attribute_all(self, text: str, utterances: List[Dict[str, Any]]) -> None:
        for utterance in utterances:
            if utterance.get("character") == "Narrator":
                continue

            start, end = utterance["char_span"]
            content = utterance.get("_content") or utterance.get("line")
            pre_context = text[max(0, start - 240) : start]
            post_context = text[end : min(len(text), end + 240)]
            quote_plus_tail = text[start : min(len(text), end + 240)]
            pre_plus_quote = text[max(0, start - 240) : end]

            best_candidate: Optional[Tuple[float, str, str, str]] = None
            candidate_scores: Dict[str, Tuple[float, str, str]] = {}

            if utterance.get("_speech_type") == "script":
                raw_name = utterance.get("_script_name")
                canonical = self._canonicalize_name(raw_name)
                if canonical:
                    candidate_scores[canonical] = (0.95, "script_label", f"label:{raw_name}")

            interruption_match = self.re_intr.search(pre_context + text[start:end] + post_context)
            if interruption_match:
                q1 = self._clean_line(interruption_match.group("q1"))
                q2 = self._clean_line(interruption_match.group("q2"))
                chosen_q = self._clean_line(content)
                if chosen_q in (q1, q2):
                    name = interruption_match.group("name")
                    canonical = self._canonicalize_name(name)
                    if canonical:
                        candidate_scores.setdefault(
                            canonical,
                            (0.96, "interruption", f'interruption:{name.strip()}'),
                        )

            post_match = self.re_post.search(quote_plus_tail)
            if post_match and self._clean_line(post_match.group("q")) == self._clean_line(content):
                canonical = self._canonicalize_name(post_match.group("name"))
                if canonical:
                    candidate_scores.setdefault(
                        canonical,
                        (1.0, "postposed_tag", f'postposed:{post_match.group("name").strip()}'),
                    )

            pre_match = re.search(
                r"(?P<name>"
                + self.NAME_RE
                + r")\s+(?:"
                + self._verb_alt
                + r")(?:\s+\w+)?\s*[\.,;:?!—-]*\s*(?:\"|“|„|«|‹|')(?P<q>.+)$",
                pre_plus_quote,
                re.IGNORECASE | re.DOTALL,
            )
            if pre_match and self._clean_line(pre_match.group("q")) == self._clean_line(content):
                canonical = self._canonicalize_name(pre_match.group("name"))
                if canonical:
                    candidate_scores.setdefault(
                        canonical,
                        (0.98, "preposed_tag", f'preposed:{pre_match.group("name").strip()}'),
                    )

            for dash in self.DASH_PREFIXES:
                dash_pattern = re.compile(re.escape(dash) + r"\s*(?P<name>" + self.NAME_RE + r")")
                dash_match = dash_pattern.search(post_context)
                if dash_match:
                    candidate = dash_match.group("name")
                    window_start = max(0, end)
                    window_end = min(len(text), end + 120)
                    vicinity = text[window_start:window_end]
                    if re.search(self._verb_alt, vicinity, re.IGNORECASE):
                        canonical = self._canonicalize_name(candidate)
                        if canonical:
                            candidate_scores.setdefault(
                                canonical,
                                (0.93, "appositive_nearby", f"appositive:{candidate.strip()}"),
                            )

            if utterance.get("_speech_type") == "emdash":
                # If em-dash line contains inline "Name —" marker, treat similarly
                match_name = re.search(self.NAME_RE, pre_context, re.IGNORECASE)
                if match_name:
                    canonical = self._canonicalize_name(match_name.group(0))
                    if canonical:
                        candidate_scores.setdefault(
                            canonical,
                            (0.93, "appositive_nearby", f"emdash-context:{match_name.group(0).strip()}"),
                        )

            if candidate_scores:
                best_name, (best_score, method, evidence) = max(
                    candidate_scores.items(), key=lambda item: item[1][0]
                )
                competing = [score for (score, _, _) in candidate_scores.values() if score != best_score]
                if any(abs(best_score - score) <= 0.02 for score in competing):
                    best_candidate = None
                elif best_score >= self.ct:
                    best_candidate = (best_score, best_name, method, evidence)
            if best_candidate:
                score, name, method, evidence = best_candidate
                utterance["character"] = name
                utterance["attribution"] = {
                    "method": method,
                    "score": float(score),
                    "evidence": evidence,
                    "name_candidate": name,
                }
            else:
                utterance["character"] = "UNATTRIBUTED"
                utterance["attribution"] = {
                    "method": "none",
                    "score": 0.0,
                    "evidence": "",
                    "name_candidate": None,
                }

    def _apply_permissive_rules(self, text: str, utterances: List[Dict[str, Any]]) -> None:
        speech_indices = [idx for idx, u in enumerate(utterances) if u.get("_speech_type")]
        idx = 0
        while idx + 1 < len(speech_indices):
            first_idx = speech_indices[idx]
            second_idx = speech_indices[idx + 1]
            first = utterances[first_idx]
            second = utterances[second_idx]
            if (
                first.get("character") not in {"Narrator", "UNATTRIBUTED"}
                and second.get("character") not in {"Narrator", "UNATTRIBUTED"}
                and first["character"] != second["character"]
                and first["attribution"].get("score", 0.0) >= self.ct
                and second["attribution"].get("score", 0.0) >= self.ct
            ):
                a = first["character"]
                b = second["character"]
                expected = a
                for follow_idx in speech_indices[idx + 2 :]:
                    candidate = utterances[follow_idx]
                    current = candidate.get("character")
                    if current not in {"UNATTRIBUTED", a, b}:
                        break
                    if current == "UNATTRIBUTED":
                        candidate["character"] = expected
                        candidate["attribution"] = {
                            "method": "alternation",
                            "score": 0.85,
                            "evidence": f"alternating:{a}/{b}",
                            "name_candidate": expected,
                        }
                    expected = b if expected == a else a
                idx += 1
            idx += 1

        pronoun_map: Dict[str, str] = {}
        for canonical, aliases in self.aliases.items():
            for alias in aliases:
                token = (alias or "").strip().lower()
                if not token:
                    continue
                if token.isalpha():
                    if token not in pronoun_map:
                        pronoun_map[token] = canonical
        if not pronoun_map:
            return
        for utterance in utterances:
            if utterance.get("character") != "UNATTRIBUTED":
                continue
            start = utterance["char_span"][0]
            context = text[max(0, start - 240) : start]
            tokens = re.findall(r"[A-Za-z]+", context.lower())
            resolved: Optional[str] = None
            for token in reversed(tokens):
                if token in pronoun_map:
                    resolved = pronoun_map[token]
                    break
            if resolved:
                canonical = self._canonicalize_name(resolved)
                if canonical:
                    utterance["character"] = canonical
                    utterance["attribution"] = {
                        "method": "pronoun_chain",
                        "score": 0.82,
                        "evidence": f"pronoun:{resolved}",
                        "name_candidate": canonical,
                    }

    def _enforce_closed_set(self, utterances: List[Dict[str, Any]]) -> None:
        if not self.known:
            return
        valid = {name for name in self.known}
        for utterance in utterances:
            character = utterance.get("character")
            if character in {"Narrator", "UNATTRIBUTED"}:
                continue
            if character not in valid:
                utterance["character"] = "UNATTRIBUTED"
                utterance["attribution"] = {
                    "method": "none",
                    "score": 0.0,
                    "evidence": "closed_set_filter",
                    "name_candidate": None,
                }

    def _wrap_narrator_spans(self, utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(self.max_narrator, int) or self.max_narrator <= 0:
            return utterances
        wrapped: List[Dict[str, Any]] = []
        sentence_pattern = re.compile(r".+?(?:[\.\!\?][\"”’']?\s+|$)", re.DOTALL)
        for utterance in utterances:
            if utterance.get("character") != "Narrator":
                wrapped.append(utterance)
                continue
            text_span = self._text[utterance["char_span"][0] : utterance["char_span"][1]]
            if len(self._clean_line(text_span)) <= self.max_narrator:
                wrapped.append(utterance)
                continue
            start_offset = utterance["char_span"][0]
            accumulator: List[Tuple[int, int]] = []
            for match in sentence_pattern.finditer(text_span):
                seg_start = match.start()
                seg_end = match.end()
                if not accumulator:
                    accumulator.append((seg_start, seg_end))
                else:
                    last_start, last_end = accumulator[-1]
                    candidate_length = len(self._clean_line(text_span[last_start:seg_end]))
                    if candidate_length <= self.max_narrator:
                        accumulator[-1] = (last_start, seg_end)
                    else:
                        accumulator.append((seg_start, seg_end))
            for seg_start, seg_end in accumulator:
                segment_text = self._clean_line(text_span[seg_start:seg_end])
                if not segment_text:
                    continue
                absolute_start = start_offset + seg_start
                absolute_end = start_offset + seg_end
                wrapped.append(
                    {
                        "character": "Narrator",
                        "line": segment_text,
                        "char_span": [absolute_start, absolute_end],
                        "source_line": self._line_for_index(absolute_start),
                        "attribution": {
                            "method": "none",
                            "score": 1.0,
                            "evidence": "narration",
                            "name_candidate": None,
                        },
                    }
                )
        wrapped.sort(key=lambda item: item["char_span"][0])
        return wrapped


class LLMAssistedAttributor:
    """Optional hook for LLM-based attribution proposals."""

    def __init__(
        self,
        known_characters: Optional[List[str]],
        aliases: Optional[Dict[str, List[str]]],
        conf_threshold: float = 0.92,
        batch_size: int = 8,
    ) -> None:
        self.known = known_characters or []
        self.aliases = {k: set(v) for k, v in (aliases or {}).items()}
        self.conf_threshold = float(conf_threshold)
        self.batch_size = int(batch_size)

    def propose(self, full_text: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return []


def _apply_llm_assist(
    full_text: str,
    utterances: List[Dict[str, Any]],
    known_characters: Optional[List[str]],
    aliases: Optional[Dict[str, List[str]]],
    llm_conf_threshold: float,
    batch_size: int,
) -> None:
    pending: List[Dict[str, Any]] = []
    for utterance in utterances:
        if utterance.get("character") != "UNATTRIBUTED":
            continue
        start, end = utterance["char_span"]
        context_start = max(0, start - 240)
        context_end = min(len(full_text), end + 240)
        pending.append(
            {
                "utterance_id": utterance["utterance_id"],
                "char_span": [start, end],
                "line": utterance["line"],
                "context_span": [context_start, context_end],
            }
        )
    if not pending:
        return
    agent = LLMAssistedAttributor(known_characters, aliases, conf_threshold=llm_conf_threshold, batch_size=batch_size)
    proposals = agent.propose(full_text, pending) or []
    id_map = {u["utterance_id"]: u for u in utterances}
    name_re = re.compile(DialogueExtractor.NAME_RE)
    verb_re = re.compile(r"(?:" + "|".join(sorted(map(re.escape, DialogueExtractor.DIALOGUE_VERBS))) + r")", re.IGNORECASE)
    for proposal in proposals:
        uid = proposal.get("utterance_id")
        if not uid or uid not in id_map:
            continue
        utterance = id_map[uid]
        character = proposal.get("character")
        confidence = float(proposal.get("confidence") or 0.0)
        if confidence < llm_conf_threshold or not isinstance(character, str):
            continue
        if character != "UNATTRIBUTED" and known_characters and character not in known_characters:
            continue
        evidence = proposal.get("evidence") or {}
        valid = True
        for span_key, regex in (("name_span", name_re), ("verb_span", verb_re)):
            span = evidence.get(span_key)
            if not span:
                continue
            try:
                span_start, span_end = int(span[0]), int(span[1])
            except Exception:
                valid = False
                break
            if span_start < 0 or span_end > len(full_text) or span_start >= span_end:
                valid = False
                break
            if min(abs(span_start - utterance["char_span"][0]), abs(span_end - utterance["char_span"][1])) > 120:
                valid = False
                break
            snippet = full_text[span_start:span_end]
            if not regex.search(snippet):
                valid = False
                break
        if not valid:
            continue
        current_score = float(utterance["attribution"].get("score") or 0.0)
        if utterance.get("character") not in {"UNATTRIBUTED", "Narrator"} and confidence < current_score + 0.05:
            continue
        utterance["character"] = character
        utterance["attribution"] = {
            "method": "llm_verified",
            "score": confidence,
            "evidence": json.dumps(evidence, ensure_ascii=False),
            "name_candidate": character if character != "UNATTRIBUTED" else None,
        }


def _write_sidecars(
    base_output_path: str,
    utterances: List[Dict[str, Any]],
    voices_map: Optional[Dict[str, str]],
    llm_enabled: bool,
    llm_model: Optional[str],
    llm_conf_threshold: float,
) -> Dict[str, str]:
    base_output_path = _ensure_path_stem(base_output_path)
    txt_path = f"{base_output_path}_dialogue_marked.txt"
    json_path = f"{base_output_path}_analysis_dialogue.json"

    lines: List[str] = []
    if voices_map:
        lines.append("# voices_map: " + json.dumps(voices_map, ensure_ascii=False))
    for item in utterances:
        speaker = item.get("character") or "UNATTRIBUTED"
        lines.append(f"{speaker}: {item.get('line', '').strip()}")
    _write_text(txt_path, "\n".join(lines) + ("\n" if lines else ""))

    sanitized: List[Dict[str, Any]] = []
    confidence_values: List[float] = []
    summary: Dict[str, Any] = {
        "utterance_count": 0,
        "by_character": {},
        "confidence_stats": {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
        },
    }

    for item in utterances:
        entry = {
            "utterance_id": item.get("utterance_id"),
            "character": item.get("character"),
            "line": item.get("line"),
            "char_span": list(item.get("char_span", [])),
            "source_line": item.get("source_line"),
            "attribution": item.get("attribution", {}),
        }
        sanitized.append(entry)
        speaker = entry.get("character") or "UNATTRIBUTED"
        summary["by_character"].setdefault(speaker, {"count": 0, "char_total": 0})
        summary["by_character"][speaker]["count"] += 1
        summary["by_character"][speaker]["char_total"] += len(entry.get("line") or "")
        if entry.get("attribution"):
            confidence_values.append(float(entry["attribution"].get("score") or 0.0))

    summary["utterance_count"] = len(sanitized)
    for label in ("Narrator", "UNATTRIBUTED"):
        summary["by_character"].setdefault(label, {"count": 0, "char_total": 0})
    if confidence_values:
        summary["confidence_stats"] = {
            "mean": sum(confidence_values) / len(confidence_values),
            "median": statistics.median(confidence_values),
            "min": min(confidence_values),
            "max": max(confidence_values),
        }

    payload = {
        "version": "1.1-hybrid",
        "source": {"created_utc": _now_utc()},
        "llm_assist": {
            "enabled": bool(llm_enabled),
            "model": llm_model,
            "conf_threshold": llm_conf_threshold,
        },
        "voices_map": voices_map or {},
        "dialogue": sanitized,
        "summary": summary,
    }
    _write_json(json_path, payload)
    return {"txt_path": txt_path, "json_path": json_path}


def extract_and_save_dialogue(
    story_text: str,
    base_output_path: str,
    *,
    known_characters: Optional[List[str]] = None,
    character_aliases: Optional[Dict[str, List[str]]] = None,
    voices_map: Optional[Dict[str, str]] = None,
    mode: str = "strict",
    confidence_threshold: float = 0.90,
    use_llm_assist: bool = False,
    llm_conf_threshold: float = 0.92,
    llm_batch_size: int = 8,
    max_narrator_chars: Optional[int] = None,
) -> Dict[str, str]:
    extractor = DialogueExtractor(
        known_characters=known_characters,
        aliases=character_aliases,
        mode=mode,
        confidence_threshold=confidence_threshold,
        max_narrator_chars=max_narrator_chars,
    )
    utterances = extractor.extract(story_text or "")
    if use_llm_assist:
        _apply_llm_assist(
            story_text or "",
            utterances,
            known_characters,
            character_aliases,
            llm_conf_threshold,
            llm_batch_size,
        )
    return _write_sidecars(
        base_output_path,
        utterances,
        voices_map,
        llm_enabled=use_llm_assist,
        llm_model=None,
        llm_conf_threshold=llm_conf_threshold,
    )


# -----------------------------
# Config
# -----------------------------
DEFAULT_LLM_MODEL    = "gpt-5-chat-latest"
LLM_MODEL_CHOICES    = [DEFAULT_LLM_MODEL, "gpt-5", "gpt-5-mini"]
OPENAI_IMAGE_MODEL   = "gpt-image-1"
DEFAULT_IMAGE_SIZE   = "1024x1024"
IMAGE_SIZE_CHOICES   = ["1024x1024", "1536x1024", "1024x1536", "auto"]
GLOBAL_STYLE_DEFAULT = "Photorealistic cinematic still"

GLOBAL_STYLE_CHOICES = [GLOBAL_STYLE_DEFAULT, "3D cinematic render", "Graphic novel / inked", "No global style"]
NEGATIVE_TERMS_POLICY = (
    "no text, no watermark, no logos, no nudity, no sexual content, no gore, no gratuitous violence, "

    "no starwars references like Tie fighter shaped spaceships or emprerial star distroyer shaped spaceships, "
    "no xenomorph shaped aliens or xenos like in the movie Alien, "
    "no startrek references like the startrek comunicator ensignia or enterprise shaped spaceships"
)

NEGATIVE_TERMS_QUALITY = (
    "low-res, blurry, artifacts, watermark, text, logo, signature, jpeg noise, over-sharpened, posterized, nsfw, "
    "direct-to-camera gaze, selfie, posed portrait, centered face, head-and-shoulders mugshot, flat front lighting, "
    "beauty influencer lighting, ringlight catchlights, interview setup, goups standing having a meeting"
)

# The single constant used throughout the app:
NEGATIVE_TERMS = ", ".join([NEGATIVE_TERMS_POLICY, NEGATIVE_TERMS_QUALITY])


DEFAULT_ASPECT       = "21:9"

# --- Aspect presets (new) ---
ASPECT_CHOICES = ["1:1", "3:2", "2:3", "16:9", "21:9", "9:16"]

ASPECT_TO_SIZE = {
    "1:1":  "1024x1024",
    "3:2":  "1536x1024",
    "2:3":  "1024x1536",
    "9:16": "1024x1536",
    "16:9": "1536x1024",
    "21:9": "1536x1024",
}

# --- Extra image planning (word-gap) ---
EXTRA_IMAGES_MIN_WORDS_DEFAULT    = 500   # user-adjustable; 0 disables
EXTRA_IMAGES_MAX_PER_SCENE        = 5     # soft cap; script may raise this when large gaps demand it
EXTRA_IMAGES_ABS_MAX_PER_SCENE    = 40    # safety limit so worst-case bursts stay bounded

# Export size caps (MB)
MIN_SCENE_JSON_MB    = 0   # no lower bound enforced now; robustness/speed over padding
MAX_SCENE_JSON_MB    = 15  # cap to 15 MB
SIZE_HEADROOM_BYTES  = 150_000
# -------- New export layout (folder-based) --------
SCENE_SUBDIR_NAME    = "scenes"    # inside the chosen outdir
SCENE_REFS_DIR       = "refs"      # inside each scene folder
WRITE_ANALYSIS_FILE  = True        # write _analysis.json at the top of outdir
ANALYSIS_FILENAME    = "_analysis.json"
ALWAYS_EXTERNALIZE_IMAGES = True   # never leave data URIs in scene JSONs

# Reference selection caps (keep exports snappy & consistent)
MAX_REF_PER_CHAR     = 4
MAX_GALLERY_PER_CHAR = 3
MAX_REF_PER_LOC      = 4
MAX_GALLERY_PER_LOC  = 3

# Default encoding parameters
DEF_SIDE_PX          = 1280
DEF_JPEG_QUALITY     = 88

# === Exposure / Lighting Controls ===
EXPOSURE_BIAS: float = 0.20          # -1.0 (darker) … +1.0 (brighter); UI & CLI can override
EXPOSURE_POST_TONEMAP: bool = True   # If True, apply gentle gamma/contrast after generation
EMISSIVE_LEVEL: float = 0.0          # -1.0 … +1.0 prompt-only glow preference

# === Extra Renders Control ===
EXTRA_EVERY_WORDS: int = 500    # create ~1 extra shot per N words (overall story)
EXTRA_MIN_WORDS:   int = 200    # minimum words to start counting; below this -> 0 extras
EXTRA_TOLERANCE:   float = 0.10 # ±10% tolerance band to avoid off-by-one churn (optional)
EXTRA_MAX_TOTAL:   int = 999    # safety cap

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

# ---- Prompt richness & enrichment (scene-level fusion) ----
PROMPT_RICH_MIN_WORDS         = 160     # if storyline is thinner than this, we may ask for enrichment
PROMPT_ENRICH_ASK_IF_NEEDED   = True    # pop a small dialog with 4–5 questions only when needed
USE_LLM_FUSION                = True    # run a short LLM pass to polish the composed prompt
FUSION_TARGET_WORDS_MIN       = 170
FUSION_TARGET_WORDS_MAX       = 230

# SCENE_FUSION_SYSTEM = (
#     "You are a senior cinematographer and visual director. Fuse storyline, character DNA (for identity lock), "
#     "and location DNA into ONE richly detailed image-generation prompt. Requirements: "
#     "1) Include EVERY character given by name, each with 1–2 concise physical anchors that match the DNA; "
#     "2) Identity lock = anatomy only (face geometry, hair style/color, eye color, skin tone). Wardrobe is a RELAXED lock: "
#     "   keep signature accessories/palette motifs if they are essential to identity or continuity, but allow scene‑appropriate "
#     "   outfit changes that fit the era, role, weather, and action; "
#     "3) Set the scene: era/setting, layout and materials, mood/atmosphere, weather if present; "
#     "4) Cinematography: composition with depth layers, lens focal length (mm eq), camera height, blocking/gesture; "
#     "5) Lighting: key/fill/rim, direction & color temperature; "
#     "6) Palette & texture cues; "
#     "7) Aspect 21:9; "
#     "8) Avoid brand names; "
#     "9) Respect constraints like 'no text, no watermark, no logos'. "
#     f"Use {FUSION_TARGET_WORDS_MIN}–{FUSION_TARGET_WORDS_MAX} words. Return a SINGLE JSON object only: "
#     '{ "prompt": "..." }'
# )

# ---- Prompt richness & enrichment (scene-level fusion) ----
PROMPT_RICH_MIN_WORDS         = 160
PROMPT_ENRICH_ASK_IF_NEEDED   = True
USE_LLM_FUSION                = True
FUSION_TARGET_WORDS_MIN       = 170
FUSION_TARGET_WORDS_MAX       = 230

SCENE_FUSION_SYSTEM = (
    "You are a senior cinematographer and visual director. Fuse storyline, character DNA and location DNA into "
    "ONE richly detailed image-generation prompt.\n"
    "Identity & evolution rules:\n"
    "• Preserve stable facial geometry, eye color, hair color/texture, skin tone and build so each character stays recognizable across shots; these anchors are non-negotiable unless the story explicitly states a transformation.\n"
    "• Treat wardrobe, helmets/gear, hair styling tweaks and surface state (mud, dust, blood, fresh scars, bandages) as additive layers that follow the current scene text.\n"
    "• When new persistent changes appear (e.g., fresh scars, now wearing a helmet), include them from that scene onward when contextually relevant without discarding the underlying identity anchors.\n"
    "• Apply similar continuity to locations/vehicles: keep established architecture, proportions, palette and materials consistent while layering story-driven damage, debris, lighting shifts or activity.\n"
    "\n"
    "Compose the prompt with:\n"
    "1) EVERY named character with 1–2 concise physical anchors that match the DNA (avoid over-constraining hair/outfit details beyond what the scene demands).\n"
    "2) Setting: era/genre, layout and materials, atmosphere, weather if present.\n"
    "3) Cinematography: composition with depth layers, lens (mm eq), camera height, blocking/gesture.\n"
    "4) Lighting: key/fill/rim, direction & color temperature.\n"
    "5) Palette & texture cues.\n"
    "6) Aspect 21:9.\n"
    "7) Avoid brand names; respect constraints like 'no text, no watermark, no logos'.\n"
    f"Use {FUSION_TARGET_WORDS_MIN}–{FUSION_TARGET_WORDS_MAX} words. Return a SINGLE JSON object only: "
    '{ \"prompt\": \"...\" }'
)
# SCENE_FUSION_SYSTEM = (
#     "You are a senior cinematographer and visual director. Fuse storyline, character DNA (for identity lock), "
#     "and location DNA into ONE richly detailed image-generation prompt. Identity lock means persistent anatomy/face "
#     "geometry, hair color/texture, eye color, skin tone, and general build. DO NOT lock clothing, armor, helmets, "
#     "props, or temporary states. If the scene text mentions gear (e.g., helmet on), new scars/injuries, or damage "
#     "(e.g., dented hull), these override reference photos and MUST be depicted. Preserve character identity while "
#     "allowing story-driven variation in outfit, accessories, and condition.\n"
#     "Requirements: "
#     "1) Include EVERY named character, each with 1–2 concise physical anchors that match their DNA; "
#     "2) Respect the location DNA and reflect current state/condition if the scene indicates changes; "
#     "3) Set the scene: era/setting, layout/materials, mood/atmosphere, weather if present; "
#     "4) Cinematography: composition with depth layers, lens focal length (mm eq), camera height, blocking/gesture; "
#     "5) Lighting: key/fill/rim, direction & color temperature; "
#     "6) Palette & texture cues; "
#     "7) Use aspect 21:9; "
#     "8) Avoid brand names; "
#     "9) Respect constraints like 'no text, no watermark, no logos'. "
#     f"Use {FUSION_TARGET_WORDS_MIN}–{FUSION_TARGET_WORDS_MAX} words. Return a SINGLE JSON object only: "
#     '{ "prompt": "..." }'
# )

def scene_fusion_system(aspect_label: str) -> str:
    return (
        "You are a senior cinematographer and visual director. Fuse storyline, character DNA and location DNA into "
        "ONE richly detailed image-generation prompt.\n"
        "Identity & evolution rules:\n"
        "• Preserve stable facial geometry, eye color, hair color/texture, skin tone and build so each character stays recognizable across shots; these anchors are non‑negotiable unless the story explicitly states a transformation.\n"
        "• Treat wardrobe, helmets/gear, hair styling tweaks and surface state (mud, dust, blood, fresh scars, bandages) as additive layers that follow the current scene text.\n"
        "• When new persistent changes appear (e.g., fresh scars, now wearing a helmet), include them from that scene onward when contextually relevant without discarding the underlying identity anchors.\n"
        "• Apply similar continuity to locations/vehicles: keep established architecture, proportions, palette and materials consistent while layering story-driven damage, debris, lighting shifts or activity.\n"
        "\n"
        "Compose the prompt with:\n"
        "1) EVERY named character with 1–2 concise physical anchors that match the DNA (avoid over‑constraining hair/outfit details beyond what the scene demands).\n"
        "2) Setting: era/genre, layout and materials, atmosphere, weather if present.\n"
        "3) Cinematography: composition with depth layers, lens (mm eq), camera height, blocking/gesture. "
        "   When ships/vehicles are present or implied, an exterior vantage that emphasizes vehicle scale/motion is acceptable; "
        "   in that case, keep faces incidental or small in frame.\n"
        "4) Lighting: key/fill/rim, direction & color temperature.\n"
        "5) Palette & texture cues.\n"
        f"6) Aspect {aspect_label}.\n"

        "7) Avoid brand names; respect constraints like 'no text, no watermark, no logos'.\n"

        f"Use {FUSION_TARGET_WORDS_MIN}–{FUSION_TARGET_WORDS_MAX} words. Return a SINGLE JSON object only: "
        '{ \"prompt\": \"...\" }'
    )

def make_scene_fusion_user(ingredients: Dict[str, Any], global_style: str, negative_terms: str) -> str:
    """
    User content for the scene‑level fusion pass.
    Identity lock anchors face/hair/eyes/skin only; wardrobe/gear may vary per scene.
    Explicitly allow scene‑limited gear (helmets/visors/masks/spacesuits) and injuries/scars/bandages,
    and location damage/soot/dust/etc. Keep injuries non‑gory. Respect negative terms.
    Return only: { "prompt": "..." }.
    """
    import json
    parts = []
    parts.append("Storyline beat:\n")
    parts.append(ingredients.get("storyline","(none)")); parts.append("\n\n")
    parts.append("Cast (all must appear):\n")
    cast_block = ingredients.get("cast", [])
    parts.append(json.dumps(cast_block, ensure_ascii=False, indent=2)); parts.append("\n\n")

    hair_lines: List[str] = []
    ref_locked: List[str] = []
    for entry in cast_block:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("name") or "").strip()
        dna = entry.get("dna") or ""
        hair = extract_hair_descriptor(dna)
        if name and hair:
            hair_lines.append(f"{name}: {hair}")
        if name and entry.get("ref_paths"):
            ref_locked.append(name)
    if hair_lines:
        parts.append("Hair continuity anchors:\n")
        for line in hair_lines:
            parts.append(f"- {line}\n")
    if ref_locked:
        parts.append("Reference stills to respect: " + join_clause(ref_locked) + ".\n")
    if hair_lines or ref_locked:
        parts.append("\n")

    parts.append("Location:\n")
    parts.append(json.dumps(ingredients.get("location", {}), ensure_ascii=False, indent=2)); parts.append("\n\n")

    if global_style and global_style != "No global style":
        parts.append("Global style: " + global_style + "\n")
    parts.append("Wardrobe policy: relaxed lock — allow scene‑appropriate variations; keep only essential accessories/motifs.\n")

    parts.append(
        "Continuity / Overrides:\n"
        "- Identity lock anchors: face geometry, hair style/color, eye color, skin tone.\n"
        "- Wardrobe/gear may change per scene. If the beat calls for protective gear (helmets, visors, masks, space suits) or injuries/scars/bandages, include them as temporary overlays.\n"
        "- Reflect environment/damage cues (dust/soot/wetness, dents/scratches) when described. Keep injuries non‑gory.\n"
    )

    parts.append("Constraints: " + (negative_terms or "") + "\n")
    parts.append('Return JSON: { "prompt": "" }')
    return "".join(parts)


# -----------------------------
# Tk & Imaging
# -----------------------------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    TKDND_AVAILABLE = True
except Exception:
    TKDND_AVAILABLE = False

# -----------------------------
# Utilities (robust JSON + helpers)
# -----------------------------
def hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s).strip("_") or "item"

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ")

def b64_data_uri(image_bytes: bytes, mime: str = "image/png") -> str:
    return "data:" + mime + ";base64," + base64.b64encode(image_bytes).decode("utf-8")

def decode_data_uri(data_uri: str) -> Optional[bytes]:
    try:
        if not data_uri or not data_uri.startswith("data:"):
            return None
        _, b64 = data_uri.split(",", 1)
        return base64.b64decode(b64)
    except Exception:
        return None
def relpath_posix(path: str, start: str) -> str:
    return os.path.relpath(path, start=start).replace(os.sep, "/")

def mime_to_ext(mime: str) -> str:
    m = (mime or "").lower()
    if "jpeg" in m: return ".jpg"
    if "jpg"  in m: return ".jpg"
    if "png"  in m: return ".png"
    if "webp" in m: return ".webp"
    return ".png"

def sniff_ext_from_bytes(b: bytes, fallback: str=".png") -> str:
    try:
        im = Image.open(io.BytesIO(b))
        fmt = (im.format or "").lower()
        if fmt in ("jpeg","jpg"): return ".jpg"
        if fmt in ("png",): return ".png"
        if fmt in ("webp",): return ".webp"
    except Exception:
        pass
    return fallback

VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _ensure_assets_dir(world_path: str) -> str:
    try:
        base = os.path.dirname(os.path.abspath(world_path)) if world_path else os.getcwd()
    except Exception:
        base = os.getcwd()
    dest = os.path.join(base, "assets")
    os.makedirs(dest, exist_ok=True)
    return dest


def _sha1_short(path: str) -> str:
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except Exception:
        return hashlib.sha1(os.path.basename(path or "").encode("utf-8", errors="ignore")).hexdigest()[:10]
    return h.hexdigest()[:10]

def _script_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()


def _styles_store_path() -> str:
    return os.path.join(_script_dir(), "user_styles.json")


def _read_json_safely(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _style_export_minimal_dict(style: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "id",
        "name",
        "sample_asset_ids",
        "palette",
        "contrast",
        "colorfulness",
        "edge_density",
        "tone_bias",
        "grain_hint",
        "style_prompt",
    ]
    return {k: style.get(k) for k in keys}


def _style_preview_paths(app: "App", style: Dict[str, Any]) -> List[str]:
    reg = {}
    try:
        reg = (getattr(app, "world", {}) or {}).get("assets_registry") or []
    except Exception:
        reg = []
    by_id = {}
    try:
        by_id = {a.get("id"): a for a in reg if isinstance(a, dict)}
    except Exception:
        by_id = {}
    paths: List[str] = []
    for rid in (style.get("sample_asset_ids") or [])[:4]:
        rec = by_id.get(rid)
        path = (rec or {}).get("path") if isinstance(rec, dict) else ""
        if isinstance(path, str) and path and os.path.isfile(path):
            paths.append(path)
    return paths


def _save_thumb(path_in: str, path_out: str, max_side: int = 240) -> bool:
    try:
        with Image.open(path_in) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = max(1.0, max(w, h) / float(max_side))
            tw, th = max(1, int(w / scale)), max(1, int(h / scale))
            im = im.resize((tw, th), Image.LANCZOS)
            ensure_dir(os.path.dirname(path_out) or ".")
            im.save(path_out, "JPEG", quality=88)
        return True
    except Exception:
        return False


def _copy_into_assets(world_path: str, src: str) -> str:
    dst_dir = _ensure_assets_dir(world_path)
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)
    ext = (ext or "").lower()
    if ext not in VALID_IMAGE_EXTS:
        ext = ".png"
    h = _sha1_short(src)
    dst = os.path.join(dst_dir, f"{name}_{h}{ext}")
    try:
        if os.path.abspath(src) != os.path.abspath(dst):
            try:
                shutil.copy2(src, dst)
            except Exception:
                with open(src, "rb") as r, open(dst, "wb") as w:
                    w.write(r.read())
    except Exception:
        pass
    return dst


def _image_palette_and_luma(img_path: str, max_colors: int = 6) -> Tuple[List[str], float]:
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            small = im.resize((64, 64))
            data = list(small.getdata())
            if not data:
                return [], 0.5
            luma = sum(0.299 * r + 0.587 * g + 0.114 * b for r, g, b in data)
            luma /= (255.0 * len(data))
            pal_image = im.convert("P", palette=Image.ADAPTIVE, colors=max(1, max_colors))
            pal = pal_image.getpalette() or []
            colors: List[str] = []
            for idx in range(0, min(len(pal), max_colors * 3), 3):
                c = f"#{pal[idx]:02x}{pal[idx+1]:02x}{pal[idx+2]:02x}"
                if c not in colors:
                    colors.append(c)
            return colors[:max_colors], max(0.0, min(1.0, float(luma)))
    except Exception:
        return [], 0.5


def _register_asset(world_obj: Dict[str, Any], img_path: str,
                    *, entity_type: str = "", entity_name: str = "") -> Dict[str, Any]:
    reg = world_obj.setdefault("assets_registry", [])
    hsh = _sha1_short(img_path)
    aid = f"img_{hsh}"
    existing = None
    for item in reg:
        if isinstance(item, dict) and item.get("id") == aid:
            existing = item
            break
    if existing is None:
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            w = h = 0
        palette, luma = _image_palette_and_luma(img_path)
        existing = {
            "id": aid,
            "entity_type": entity_type or "",
            "entity_name": entity_name or "",
            "view": "",
            "path": img_path,
            "created_at": now_iso(),
            "hash": hsh,
            "w": int(w),
            "h": int(h),
            "palette": palette,
            "avg_luma": float(luma),
        }
        reg.append(existing)
    else:
        if entity_type and not existing.get("entity_type"):
            existing["entity_type"] = entity_type
        if entity_name and not existing.get("entity_name"):
            existing["entity_name"] = entity_name
        if not existing.get("palette") or not isinstance(existing.get("palette"), list):
            palette, luma = _image_palette_and_luma(img_path)
            existing["palette"] = palette
            existing["avg_luma"] = float(luma)
    return existing


def _image_dna_phrases(palette: List[str], luma: float) -> List[str]:
    cues: List[str] = []
    if palette:
        cues.append("palette " + ", ".join(palette[:4]))
    if luma < 0.33:
        cues.append("low ambient light; avoid total murk")
    elif luma > 0.66:
        cues.append("high-key cues; protect highlights")
    else:
        cues.append("balanced ambient light; preserve midtones")
    return cues


def _palette_from_image(img: Image.Image, k: int = 6) -> List[str]:
    try:
        pal = img.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=max(1, k)).getpalette() or []
        out: List[str] = []
        for idx in range(0, min(len(pal), k * 3), 3):
            color = f"#{pal[idx]:02x}{pal[idx+1]:02x}{pal[idx+2]:02x}"
            if color not in out:
                out.append(color)
        return out[:k]
    except Exception:
        return []


def _contrast_norm(img: Image.Image) -> float:
    try:
        im = img.convert("RGB").resize((128, 128))
        pixels = list(im.getdata())
        if not pixels:
            return 0.5
        luma_vals = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels]
        stdev = statistics.pstdev(luma_vals) if luma_vals else 0.0
        return max(0.0, min(1.0, stdev / 128.0))
    except Exception:
        return 0.5


def _colorfulness(img: Image.Image) -> float:
    try:
        im = img.convert("RGB").resize((128, 128))
        pixels = list(im.getdata())
        if not pixels:
            return 0.5
        rg = [abs(r - g) for r, g, b in pixels]
        yb = [abs(0.5 * (r + g) - b) for r, g, b in pixels]
        mr = statistics.mean(rg) if rg else 0.0
        sr = statistics.pstdev(rg) if rg else 0.0
        my = statistics.mean(yb) if yb else 0.0
        sy = statistics.pstdev(yb) if yb else 0.0
        val = math.sqrt(mr * mr + my * my) + 0.3 * math.sqrt(sr * sr + sy * sy)
        return max(0.0, min(1.0, val / 110.0))
    except Exception:
        return 0.5


def _edge_density(img: Image.Image) -> float:
    try:
        im = img.convert("L").filter(ImageFilter.FIND_EDGES).resize((128, 128))
        data = list(im.getdata())
        if not data:
            return 0.0
        threshold = 40
        ratio = sum(1 for v in data if v > threshold) / float(128 * 128)
        return max(0.0, min(1.0, ratio * 2.0))
    except Exception:
        return 0.0


def _tone_bias(palette: List[str]) -> str:
    try:
        if not palette:
            return "neutral"
        import colorsys

        hsv_vals: List[tuple] = []
        for hex_color in palette[:4]:
            if not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
                continue
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
            hsv_vals.append(colorsys.rgb_to_hsv(r, g, b))
        if not hsv_vals:
            return "neutral"
        avg_h = sum(h for h, _, _ in hsv_vals) / len(hsv_vals)
        avg_s = sum(s for _, s, _ in hsv_vals) / len(hsv_vals)
        if avg_s < 0.12:
            return "neutral"
        if avg_h < 0.12 or 0.08 < avg_h < 0.22:
            return "warm"
        if 0.5 < avg_h < 0.75:
            return "cool"
        return "neutral"
    except Exception:
        return "neutral"


def _grain_hint(contrast: float, edge: float) -> str:
    if edge > 0.55 and contrast < 0.45:
        return "stipple"
    if edge > 0.50 and contrast >= 0.45:
        return "linework"
    if contrast < 0.25:
        return "paper"
    return "clean"


def _summarize_style_prompt(palette: List[str], contrast: float, colorfulness: float,
                            edge: float, tone: str, grain: str) -> str:
    bits: List[str] = []
    if edge > 0.5:
        bits.append("intricate linework / cross-hatching")
    elif edge > 0.3:
        bits.append("visible ink contours, moderate hatching")
    else:
        bits.append("soft contours, painterly surfaces")
    if palette:
        bits.append("restrained palette " + ", ".join(palette[:4]))
    if tone != "neutral":
        bits.append(tone + " tone bias")
    bits.append(("low" if contrast < 0.35 else "medium" if contrast < 0.65 else "high") + " contrast")
    bits.append(("muted" if colorfulness < 0.35 else "balanced" if colorfulness < 0.65 else "vivid") + " colorfulness")
    if grain != "clean":
        bits.append(grain + " texture")
    return ", ".join(bits)


def _llm_style_summary(self, sample_paths: List[str], analysis_ctx: str, fallback: str) -> str:
    try:
        # Prefer the app’s OpenAIClient if connected
        client = getattr(self, "client", None)
        model = getattr(self, "llm_model", "gpt-5-chat-latest")
        if not client:
            # No client? fall back to the existing heuristic description
            return fallback

        # Build a compact, JSON-only instruction
        system = (
            "You are a cinematography/style taxonomist. Summarize the consistent visual style across the "
            "sample images into a concise descriptor suitable for a global prompt. Return a SINGLE JSON object "
            'with this shape: { "style_prompt": "..." }. Do NOT include camera brand names or watermark/text. '
            "Favor palette, contrast, texture/edge feel, lighting character, grain, and any recurring art direction."
        )

        # Attach up to 4 images as data URIs
        user_payload = []
        if analysis_ctx:
            user_payload.append({"type": "text", "text": "Project context:\n" + analysis_ctx[:800]})
        if sample_paths:
            from pathlib import Path

            imgs = []
            for raw_path in sample_paths[:4]:
                try:
                    p = Path(raw_path)
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    imgs.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}})
                except Exception:
                    continue
            if imgs:
                user_payload.extend(imgs)

        # Always include a text hint describing the metrics already computed offline
        if fallback:
            user_payload.append({"type": "text", "text": "Offline cues to incorporate: " + fallback})

        data = client.chat_json(
            model=model,
            system=system,
            user=user_payload if user_payload else 'Summarize the visual style into {"style_prompt":"..."}.',
            temperature=0.2,
        )
        prompt = (data.get("style_prompt") or data.get("prompt") or "").strip()
        return prompt or fallback
    except Exception:
        return fallback


def _normalize_visual_cues_value(value: Any) -> Tuple[str, List[str]]:
    if isinstance(value, list):
        cleaned = [str(x).strip() for x in value if str(x).strip()]
        unique = []
        for c in cleaned:
            if c not in unique:
                unique.append(c)
        return "; ".join(unique), unique
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "", []
        parts = [p.strip() for p in re.split(r"[;\n]+", text) if p.strip()]
        unique = []
        for p in parts:
            if p not in unique:
                unique.append(p)
        return text, unique
    return "", []

def parse_data_uri_to_bytes_and_ext(data_uri: str):
    """
    Returns (raw_bytes, extension) or None if parsing fails.
    """
    try:
        if not data_uri.startswith("data:"):
            return None
        head, b64 = data_uri.split(",", 1)
        mime = head.split(";")[0][5:]  # after 'data:'
        raw = base64.b64decode(b64)
        ext = mime_to_ext(mime) or sniff_ext_from_bytes(raw)
        return raw, ext
    except Exception:
        return None

SCENE_ID_RE = re.compile(r"^S(\d+)$")

LAST_EXPAND_SCENES_REPORT: Optional[Dict[str, Any]] = None
LAST_EXPAND_SCENES_STATUS: str = ""
LAST_EXTRA_SHOT_REPORT: Optional[Dict[str, Any]] = None

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root at {path} is not an object")
    return data

def _write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if os.path.isfile(path):
        bak = path + ".bak"
        try:
            if os.path.isfile(bak):
                os.remove(bak)
        except Exception:
            pass
        try:
            os.replace(path, bak)
        except Exception:
            pass
    os.replace(tmp, path)

def _scene_num(scene_id: str) -> int:
    m = SCENE_ID_RE.match(scene_id or "")
    return int(m.group(1)) if m else -1

def _next_scene_id(existing_ids: List[str]) -> str:
    mx = 0
    for sid in existing_ids or []:
        n = _scene_num(sid)
        if n > mx:
            mx = n
    return f"S{mx + 1}"

def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

_WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)

def _count_words(txt: str) -> int:
    if not txt:
        return 0
    return len(_WORD_RE.findall(str(txt)))

def _total_narrative_words(analysis: dict, story_text: str, captions_map: dict) -> int:
    """Estimate total words from story text, analysis summaries, or scene bodies."""
    total = 0
    if story_text:
        return _count_words(story_text)

    try:
        for mv in (analysis or {}).get("movements", []) or []:
            if isinstance(mv, dict):
                total += _count_words(mv.get("summary") or "")
                for ev in (mv.get("key_events") or []):
                    if isinstance(ev, dict):
                        total += _count_words(
                            ev.get("summary")
                            or ev.get("what_happens")
                            or ev.get("description")
                            or ""
                        )
    except Exception:
        pass

    try:
        cm = captions_map or {}
        sc_list = cm.get("scenes") if isinstance(cm.get("scenes"), list) else cm.get("items")
        for sc in sc_list or []:
            if not isinstance(sc, dict):
                continue
            anchor = (
                sc.get("what_happens")
                or sc.get("description")
                or sc.get("text_anchor")
                or ""
            )
            total += _count_words(anchor)
    except Exception:
        pass

    return total

def _scene_word_estimate(scene: dict) -> int:
    return _count_words((scene or {}).get("what_happens") or "")

def apply_exposure_tonemap(img: "Image.Image", bias: float) -> "Image.Image":
    """Apply a gentle gamma/contrast curve while preserving the incoming mode."""

    try:
        b = _clamp(float(bias), -1.0, 1.0)
    except Exception:
        b = 0.0
    if abs(b) < 0.05 or not isinstance(img, Image.Image):
        return img

    orig_mode = getattr(img, "mode", "RGB")
    work_mode = orig_mode if orig_mode in ("RGB", "RGBA") else "RGBA"
    work = img if work_mode == orig_mode else img.convert(work_mode)

    if b >= 0.0:
        gamma = _clamp(1.0 - 0.6 * b, 0.40, 1.80)
        cfac = 1.0 - 0.10 * b
    else:
        gamma = _clamp(1.0 - 0.8 * b, 0.40, 1.80)
        cfac = 1.0 + 0.12 * (-b)

    lut = [int(pow(i / 255.0, gamma) * 255 + 0.5) for i in range(256)]

    if work_mode == "RGBA":
        r, g, bch, a = work.split()
        rgb = Image.merge("RGB", (r, g, bch)).point(lut * 3)
        rgb = ImageEnhance.Contrast(rgb).enhance(cfac)
        r2, g2, b2 = rgb.split()
        out = Image.merge("RGBA", (r2, g2, b2, a))
    else:
        out = work.point(lut * 3)
        out = ImageEnhance.Contrast(out).enhance(cfac)

    if work_mode != orig_mode:
        try:
            out = out.convert(orig_mode)
        except Exception:
            # If the conversion fails, fall back to the processed work image.
            out = out.convert("RGBA") if work_mode == "RGBA" else out

    return out

def expand_scenes_to_analysis(analysis: Dict[str, Any],
                              cmap: Dict[str, Any],
                              *,
                              prefer_titles: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (updated_captions_map, report).

    Adds scene stubs for analysis key events missing from captions_map and normalizes
    related references. Inputs are not mutated; deep copies are returned.
    """
    report: Dict[str, Any] = {
        "existing_scenes": [],
        "new_scenes": [],
        "normalized_movements": [],
        "normalized_devices": [],
        "skipped": []
    }

    cmap_src = cmap if isinstance(cmap, dict) else {}
    analysis_src = analysis if isinstance(analysis, dict) else {}

    cmap2 = copy.deepcopy(cmap_src)
    analysis2 = copy.deepcopy(analysis_src)

    scene_key = "scenes" if isinstance(cmap2.get("scenes"), list) else "items"
    raw_scenes = _as_list(cmap2.get(scene_key))
    scenes = [s for s in raw_scenes if isinstance(s, dict)]

    existing_ids = [s.get("id", "") for s in scenes if s.get("id")]
    report["existing_scenes"] = existing_ids[:]

    have_ids = set(existing_ids)
    titles_lower: Dict[str, str] = {}
    for s in scenes:
        title = (s.get("title") or "").strip().lower()
        if title and s.get("id"):
            titles_lower[title] = s.get("id")

    beats: List[Dict[str, Any]] = []
    for mv in _as_list(analysis2.get("movements")):
        if not isinstance(mv, dict):
            continue
        for ev in _as_list(mv.get("key_events")):
            if isinstance(ev, dict):
                beats.append(ev)
    if not beats:
        beats = [ev for ev in _as_list(analysis2.get("key_events")) if isinstance(ev, dict)]

    def beat_title(ev: Dict[str, Any]) -> str:
        t = (ev.get("title") or ev.get("name") or ev.get("label") or "Untitled Beat").strip()
        if prefer_titles and t:
            return t
        desc = (ev.get("summary") or ev.get("what_happens") or ev.get("description") or "").strip()
        return desc or t

    def find_scene_for_title(title: str) -> str:
        low = (title or "").strip().lower()
        if low in titles_lower:
            return titles_lower[low]
        for t, sid in titles_lower.items():
            if low and (low in t or t in low):
                return sid
        return ""

    missing_beats: List[Dict[str, Any]] = []
    for ev in beats:
        t = beat_title(ev)
        sid_hint = (ev.get("scene_id") or ev.get("scene_hint") or "").strip()
        sid = sid_hint if sid_hint in have_ids else find_scene_for_title(t)
        if not sid:
            missing_beats.append(ev)

    def mk_stub(ev: Dict[str, Any]) -> Dict[str, Any]:
        nid = _next_scene_id(list(have_ids))
        have_ids.add(nid)

        title = beat_title(ev)
        desc = (ev.get("summary") or ev.get("what_happens") or ev.get("description") or title).strip()
        if not desc:
            desc = title or "Story beat"
        key_actions = _as_list(ev.get("key_actions"))
        caption = (desc[:160] or title or "Story beat").strip()
        shots = [{
            "id": f"{nid}_shot1",
            "caption": caption,
            "notes": "auto-generated from analysis beat"
        }]

        stub = {
            "id": nid,
            "title": title or nid,
            "what_happens": desc,
            "key_actions": key_actions,
            "shots": shots
        }
        report["new_scenes"].append(f"{nid} :: {title or nid}")
        return stub

    new_scenes = [mk_stub(ev) for ev in missing_beats]

    if new_scenes:
        scenes.extend(new_scenes)
        scenes.sort(key=lambda s: _scene_num(s.get("id", "")))
        cmap2[scene_key] = scenes
    else:
        cmap2[scene_key] = scenes

    max_sid = 0
    for sid in have_ids:
        n = _scene_num(sid)
        if n > max_sid:
            max_sid = n
    max_id = f"S{max_sid}" if max_sid > 0 else ""

    movements = _as_list(analysis2.get("movements"))
    for mv in movements:
        if not isinstance(mv, dict):
            continue
        end_sid = (mv.get("end_scene_id") or "").strip()
        if max_id and _scene_num(end_sid) > _scene_num(max_id):
            report["normalized_movements"].append({
                "movement": mv.get("name") or mv.get("id") or "",
                "from": end_sid,
                "to": max_id
            })
            mv["end_scene_id"] = max_id

    for dev in _as_list(analysis2.get("plot_devices")):
        if not isinstance(dev, dict):
            continue
        fs = (dev.get("first_scene_id") or "").strip()
        name = (dev.get("name") or "").strip()
        earliest = None
        if name:
            needle = name.lower()
            for s in scenes:
                if not isinstance(s, dict):
                    continue
                sid = s.get("id", "")
                body = ((s.get("title") or "") + " " + (s.get("what_happens") or s.get("description") or "")).lower()
                if needle in body and sid:
                    if earliest is None or _scene_num(sid) < _scene_num(earliest):
                        earliest = sid
        if earliest and earliest != fs:
            report["normalized_devices"].append({
                "device": name,
                "from": fs,
                "to": earliest
            })
            dev["first_scene_id"] = earliest

    report["analysis"] = analysis2
    return cmap2, report

def distribute_extra_shots_after_final_plan(analysis: dict,
                                            captions_map: dict,
                                            *,
                                            story_text: str = "",
                                            every_words: int = EXTRA_EVERY_WORDS,
                                            min_words: int = EXTRA_MIN_WORDS,
                                            tolerance: float = EXTRA_TOLERANCE,
                                            max_total: int = EXTRA_MAX_TOTAL,
                                            dry_run: bool = False) -> dict:
    """Add extra shot stubs proportionally across scenes once plan is final."""

    global LAST_EXTRA_SHOT_REPORT

    report: Dict[str, Any] = {
        "applied": False,
        "total_words": 0,
        "target_extras": 0,
        "per_scene_plan": {},
        "added_per_scene": {},
        "dry_run": bool(dry_run),
    }

    try:
        cmap_src = captions_map if isinstance(captions_map, dict) else {}
        cmap_work = copy.deepcopy(cmap_src)
        scene_key = "scenes" if isinstance(cmap_work.get("scenes"), list) else "items"
        scene_list = [s for s in _as_list(cmap_work.get(scene_key)) if isinstance(s, dict)]

        if not scene_list:
            print("[extras] skipped: no scenes")
            report["reason"] = "no_scenes"
            LAST_EXTRA_SHOT_REPORT = report
            return captions_map

        if every_words <= 0:
            print("[extras] every_words<=0; disabled")
            report["reason"] = "disabled"
            LAST_EXTRA_SHOT_REPORT = report
            return captions_map

        total_words = _total_narrative_words(analysis, story_text, cmap_work)
        report["total_words"] = total_words

        if total_words < max(1, int(min_words)):
            print(f"[extras] total_words={total_words} < min={min_words}; 0 extras")
            report["reason"] = "below_min"
            LAST_EXTRA_SHOT_REPORT = report
            return captions_map

        raw = total_words / float(every_words)
        target = int(raw + 1e-6)
        frac = raw - target
        if tolerance > 0 and abs(frac) < tolerance:
            pass
        target = max(0, min(int(target), int(max_total)))
        report["target_extras"] = target

        if target <= 0:
            print("[extras] computed 0 extras")
            report["reason"] = "zero_target"
            LAST_EXTRA_SHOT_REPORT = report
            return captions_map

        weights: List[int] = []
        for sc in scene_list:
            w = _scene_word_estimate(sc)
            weights.append(max(1, w))
        total_weight = float(sum(weights)) if weights else 1.0

        quotas = []
        for idx, weight in enumerate(weights):
            quotas.append((idx, (weight / total_weight) * target))

        base_alloc = [(i, int(qty)) for i, qty in quotas]
        assigned = sum(q for _, q in base_alloc)
        rema = sorted(
            ((i, quotas[i][1] - int(quotas[i][1])) for i in range(len(quotas))),
            key=lambda x: x[1],
            reverse=True,
        )
        leftover = target - assigned
        ri = 0
        while leftover > 0 and ri < len(rema):
            idx = rema[ri][0]
            base_alloc[idx] = (idx, base_alloc[idx][1] + 1)
            leftover -= 1
            ri += 1

        plan = {}
        for idx, qty in base_alloc:
            if qty > 0:
                sc = scene_list[idx]
                sid = sc.get("id") or sc.get("scene_id") or f"S{idx+1}"
                plan[sid] = qty
        report["per_scene_plan"] = plan

        if dry_run:
            print(f"[extras] DRY RUN: total_words={total_words}, target_extras={target}")
            for sid, qty in plan.items():
                print(f"  scene {sid}: +{qty} extra(s)")
            LAST_EXTRA_SHOT_REPORT = report
            return captions_map

        added_counts: Dict[str, int] = {}
        for idx, qty in base_alloc:
            if qty <= 0:
                continue
            sc = scene_list[idx]
            sid = sc.get("id") or sc.get("scene_id") or f"S{idx+1}"
            shots = sc.setdefault("shots", []) if isinstance(sc.get("shots"), list) else sc.setdefault("shots", [])
            existing_ids = {
                (sh.get("id") or "") for sh in shots if isinstance(sh, dict) and sh.get("id")
            }
            base_caption = (
                sc.get("what_happens")
                or sc.get("description")
                or sc.get("text_anchor")
                or sc.get("title")
                or "Extra detail"
            ).strip()
            created = 0
            suffix = 1
            while created < qty and suffix < 1000:
                shot_id = f"{sid}_extra{suffix}"
                suffix += 1
                if shot_id in existing_ids:
                    continue
                shot_entry = {
                    "id": shot_id,
                    "caption": (base_caption[:240] or "Extra detail"),
                    "notes": "auto-extra from words-per-image scheduler",
                    "extra": True,
                    "extra_index": created + 1,
                }
                shots.append(shot_entry)
                existing_ids.add(shot_id)
                created += 1
            if created:
                added_counts[sid] = created

        cmap_work[scene_key] = scene_list
        report["added_per_scene"] = added_counts
        report["applied"] = True
        report["added_total"] = sum(added_counts.values())
        print(f"[extras] added total extras: {report['added_total']} across {len(scene_list)} scene(s)")
        LAST_EXTRA_SHOT_REPORT = report
        return cmap_work

    except Exception as e:
        print("[extras] error:", e)
        report["error"] = str(e)
        LAST_EXTRA_SHOT_REPORT = report
        return captions_map

def _maybe_expand_scenes(analysis_path: str,
                         captions_path: str,
                         dry_run: bool = False,
                         extra_dry_run: bool = False):
    global LAST_EXPAND_SCENES_REPORT, LAST_EXPAND_SCENES_STATUS
    try:
        analysis = _read_json(analysis_path)
        cmap = _read_json(captions_path)
    except Exception as e:
        msg = f"[expand-scenes] skipped: load error: {e}"
        print(msg)
        LAST_EXPAND_SCENES_STATUS = msg
        LAST_EXPAND_SCENES_REPORT = None
        return

    try:
        updated_cmap, report = expand_scenes_to_analysis(analysis, cmap)
    except Exception as e:
        msg = f"[expand-scenes] failed: {e}"
        print(msg)
        LAST_EXPAND_SCENES_STATUS = msg
        LAST_EXPAND_SCENES_REPORT = None
        return

    updated_analysis = report.get("analysis") if isinstance(report.get("analysis"), dict) else analysis

    def _count_extra_shots(cmap_obj: Dict[str, Any]) -> int:
        if not isinstance(cmap_obj, dict):
            return 0
        key = "scenes" if isinstance(cmap_obj.get("scenes"), list) else "items"
        total = 0
        for sc in _as_list(cmap_obj.get(key)):
            if not isinstance(sc, dict):
                continue
            for shot in _as_list(sc.get("shots")):
                if isinstance(shot, dict) and shot.get("extra"):
                    total += 1
        return total

    story_text = ""
    analysis_dict = (
        updated_analysis if isinstance(updated_analysis, dict)
        else (analysis if isinstance(analysis, dict) else {})
    )
    updated_analysis = analysis_dict
    if isinstance(analysis_dict, dict):
        for key in ("story_text", "story", "full_text", "original_text"):
            val = analysis_dict.get(key)
            if val:
                story_text = str(val)
                break

    extras_before = _count_extra_shots(updated_cmap)
    updated_cmap = distribute_extra_shots_after_final_plan(
        analysis=analysis_dict,
        captions_map=updated_cmap,
        story_text=story_text,
        every_words=EXTRA_EVERY_WORDS,
        min_words=EXTRA_MIN_WORDS,
        tolerance=EXTRA_TOLERANCE,
        max_total=EXTRA_MAX_TOTAL,
        dry_run=bool(dry_run or extra_dry_run),
    )
    extras_after = _count_extra_shots(updated_cmap)

    cmap_changed = isinstance(updated_cmap, dict) and updated_cmap != cmap
    analysis_changed = isinstance(analysis_dict, dict) and analysis_dict != analysis

    if dry_run:
        print("[expand-scenes] DRY RUN")
    else:
        if cmap_changed:
            try:
                _write_json_atomic(captions_path, updated_cmap)
                print("[expand-scenes] captions_map.json updated")
            except Exception as e:
                print(f"[expand-scenes] write failed: {e}")
        else:
            print("[expand-scenes] captions_map.json unchanged")
        if analysis_changed:
            try:
                _write_json_atomic(analysis_path, updated_analysis)
                print("[expand-scenes] _analysis.json normalized")
            except Exception as e:
                print(f"[expand-scenes] analysis write failed: {e}")

    print(f"[expand-scenes] existing scenes: {len(report.get('existing_scenes', []))}")
    print(f"[expand-scenes] new scenes: {len(report.get('new_scenes', []))}")
    for s in report.get("new_scenes", []):
        print("  +", s)
    for mv in report.get("normalized_movements", []):
        print("  ~ movement end_scene_id:", mv)
    for dv in report.get("normalized_devices", []):
        print("  ~ device normalized:", dv)

    extras_report = LAST_EXTRA_SHOT_REPORT or {}
    if extras_report.get("dry_run"):
        extra_tag = extras_report.get("target_extras", 0)
    else:
        extra_tag = extras_report.get("added_total", extras_after)

    summary = (
        f"Scene expansion: +{len(report.get('new_scenes', []))} new"
        f"; movements {len(report.get('normalized_movements', []))} normalized"
        f"; devices {len(report.get('normalized_devices', []))}"
    )
    summary += f"; extras {'planned ' if extras_report.get('dry_run') else ''}{extra_tag}"
    if dry_run:
        summary += " (dry-run)"
    LAST_EXPAND_SCENES_STATUS = summary
    LAST_EXPAND_SCENES_REPORT = {
        "report": report,
        "dry_run": dry_run,
        "captions_changed": cmap_changed,
        "analysis_changed": analysis_changed,
        "extras_before": extras_before,
        "extras_after": extras_after,
        "extras": extras_report,
        "extra_dry_run": extra_dry_run,
    }

def clean_json_from_text(text):
    if text is None:
        return ""
    if isinstance(text, (dict, list)):
        return json.dumps(text, ensure_ascii=False)
    s = str(text).replace("\r", "")
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE | re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    s = s.lstrip("\ufeff").strip()
    s = re.sub(r'^(?:\\[nrtf]|\s)+', '', s)
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        s = s[l:r+1]
    return s.strip()

def safe_json_loads(payload):
    s = clean_json_from_text(payload)
    t = s.strip()
    if t[:1] in "{[":
        try:
            return json.loads(t)
        except Exception:
            pass
    token = t
    for _ in range(4):
        token = token.strip()
        m = re.match(r'^(["\'])(.*)\1$', token, flags=re.DOTALL)
        if not m:
            break
        token = m.group(2).strip()
    if token and not any(ch in token for ch in '{}[]:,'):
        return {"title": token}
    try:
        return json.loads(t)
    except Exception:
        pass
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    frag = s[start:i+1]
                    try:
                        return json.loads(frag)
                    except Exception:
                        frag2 = re.sub(r",\s*([}\]])", r"\1", frag)
                        return json.loads(frag2)
    cleaned = re.sub(r",\s*([}\]])", r"\1", s)
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", cleaned).strip()
    return json.loads(cleaned)

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class CharacterProfile:
    name: str
    initial_description: str
    role: str = ""
    goals: str = ""
    conflicts: str = ""
    refined_description: str = ""
    visual_cues_from_photos: str = ""
    visual_cues_from_photos_list: List[str] = field(default_factory=list)
    sheet_base_prompt: str = ""
    sheet_images: Dict[str, List[bytes]] = field(default_factory=dict)
    sheet_selected: Dict[str, List[bool]] = field(default_factory=dict)
    reference_images: List[str] = field(default_factory=list)
    primary_reference_id: str = ""
    dna_traits: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LocationProfile:
    name: str
    description: str = ""
    mood: str = ""
    lighting: str = ""
    key_props: str = ""
    visual_cues_from_photos: str = ""
    visual_cues_from_photos_list: List[str] = field(default_factory=list)
    sheet_base_prompt: str = ""
    sheet_images: Dict[str, List[bytes]] = field(default_factory=dict)
    sheet_selected: Dict[str, List[bool]] = field(default_factory=dict)
    reference_images: List[str] = field(default_factory=list)
    primary_reference_id: str = ""
    dna_traits: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ShotPrompt:
    id: str
    scene_id: str
    title: str
    shot_description: str
    prompt: str
    continuity_notes: str = ""


@dataclass
class AssetRecord:
    id: str
    entity_type: str
    entity_name: str
    view: str
    prompt_full: str
    model: str
    size: str
    file_path: str
    created_at: str
    notes: str = ""

# -----------------------------
# View definitions
# -----------------------------
CHAR_SHEET_VIEWS_DEF = {
    "front": {"label": "Front (head & shoulders)", "pose": "neutral pose, facing camera", "framing": "head-and-shoulders portrait", "camera": "85mm equivalent"},
    "three_quarter_left": {"label": "Three-quarter (left)", "pose": "neutral pose, 3/4 left angle", "framing": "head and shoulders", "camera": "85mm equivalent"},
    "profile_left": {"label": "Left profile", "pose": "neutral pose, left profile", "framing": "head and shoulders", "camera": "85mm equivalent"},
    "three_quarter_right": {"label": "Three-quarter (right)", "pose": "neutral pose, 3/4 right angle", "framing": "head and shoulders", "camera": "85mm equivalent"},
    "profile_right": {"label": "Right profile", "pose": "neutral pose, right profile", "framing": "head and shoulders", "camera": "85mm equivalent"},
    "back": {"label": "Back view", "pose": "standing, back to camera, arms relaxed", "framing": "waist-up", "camera": "50mm equivalent"},
    "full_body_tpose": {"label": "Full body (T-pose)", "pose": "standing full-body T-pose, feet shoulder-width", "framing": "full body", "camera": "35mm equivalent"},
}
LOC_VIEWS_DEF = {
    "establishing": {"label": "Establishing (wide)", "note": "wide establishing shot that shows layout and materials"},
    "alt_angle": {"label": "Alternate angle", "note": "alternate wide/medium angle for coverage"},
    "detail": {"label": "Detail close-up", "note": "close-up of telling texture/prop/signage"},
}

# -----------------------------
# OpenAI client (OpenAI-only)
# -----------------------------
def _require_openai():
    """
    Ensure the OpenAI SDK is importable. Raise a clear, actionable error if not.
    """
    try:
        import openai  # noqa: F401
        from openai import OpenAI  # noqa: F401
    except Exception as e:
        raise RuntimeError("Install the OpenAI SDK first: pip install openai")

class OpenAIClient:
    """
    Minimal wrapper around the OpenAI Python SDK that:
      • Accepts an API key and optional organization id.
      • Applies a per-request timeout via with_options(...) when supported.
      • Normalizes common result shapes.
    """
    def __init__(self, api_key: str, organization: str | None = None, timeout: float = 180.0):
        _require_openai()
        from openai import OpenAI
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("Valid OpenAI API key (sk-...) is required.")
        kwargs = {"api_key": api_key}
        if organization:
            kwargs["organization"] = organization
        # Older SDKs may not accept "timeout" at construction; keep it per-call.
        self.client = OpenAI(**kwargs)
        self._timeout = float(timeout)

    def _cx(self):
        """
        Return a client configured with a timeout if the SDK supports it.
        """
        try:
            return self.client.with_options(timeout=self._timeout)
        except Exception:
            return self.client

    def chat_json(self, model: str, system: str, user, temperature: float = 0.2) -> dict:
        cx = self._cx()
        res = cx.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        raw = (res.choices[0].message.content or "")
        try:
            return safe_json_loads(raw)
        except Exception:
            # Fallback: treat as a single token string
            token = str(raw).strip().strip('"')
            return {"prompt": token}

    def chat_text(self, model: str, system: str, user, temperature: float = 0.2) -> str:
        cx = self._cx()
        res = cx.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (res.choices[0].message.content or "").strip()

    def generate_images_b64(self, model: str, prompt: str, size: str, n: int = 1) -> list[bytes]:
        cx = self._cx()
        out = cx.images.generate(model=model, prompt=prompt, size=size, n=int(max(1, n)))
        blobs: list[bytes] = []
        for d in out.data:
            b64 = getattr(d, "b64_json", None) or (d.get("b64_json") if isinstance(d, dict) else None)
            if b64:
                blobs.append(base64.b64decode(b64))
        return blobs

    def generate_images_b64_with_refs(
        self,
        model: str,
        prompt: str,
        size: str,
        ref_data_uris: list[str] | None = None,
        n: int = 1,
    ) -> list[bytes]:
        """
        Attempt to call Images API with reference images; gracefully degrades to prompt-only.
        """
        if not ref_data_uris:
            return self.generate_images_b64(model=model, prompt=prompt, size=size, n=n)

        cx = self._cx()
        base_kwargs = {"model": model, "prompt": prompt, "size": size, "n": int(max(1, n))}
        for key in ("image", "images", "additional_input_images", "image[]"):
            try:
                kwargs = dict(base_kwargs); kwargs[key] = list(ref_data_uris)
                out = cx.images.generate(**kwargs)
                blobs: list[bytes] = []
                for d in out.data:
                    b64 = getattr(d, "b64_json", None) or (d.get("b64_json") if isinstance(d, dict) else None)
                    if b64:
                        blobs.append(base64.b64decode(b64))
                if blobs:
                    return blobs
            except Exception:
                # Try next key shape
                continue
        # Fallback
        return self.generate_images_b64(model=model, prompt=prompt, size=size, n=n)

# -----------------------------
# Prompt construction (NO .format on JSON examples)
# -----------------------------
ANALYZE_SYSTEM = (
    "You are a production designer and story analyst. Return a SINGLE JSON OBJECT. No markdown/code fences/prose. "
    "Read the user's story (it may be expository, a diary, a vignette, or a transcript without obvious script structure) "
    "and extract a structured world model. Required keys: title, logline, story_precis, story_summary, main_characters, "
    "locations, structure, plot_devices, scenes. "
    "ALWAYS return at least TWO items in main_characters and at least TWO items in locations. "
    "If the story lacks explicit names, create faithful descriptive handles (e.g., "
    "'Narrator — Planetary Real‑Estate Agent', 'Federation Official', 'Human Client', "
    "'Frontier Colony', 'Orbital Station', 'Ringworld Transit Hub'). "
    "For each character include: name, initial_description, role, goals, conflicts. "
    "For each location include: name, description. "
    "Also extract plot‑significant objects ('plot devices') — objects, artifacts, documents, or signals whose introduction changes stakes, creates goals, or solves obstacles (e.g., key, letter, ring, data core, virus sample). "
    "For each plot device include: name, description, first_scene_id (where it is introduced), significance, and whether it recurs. "
    "The 'structure' must include 'movements' (group scenes by movement/sequence). "
    "IMPORTANT: When a plot‑significant object is introduced, create a dedicated scene that centers the object and the immediate story around it (who introduces it, what it does, why it matters), even if there is no location or character change. "
    "Scenes should include: id, title, what_happens, description, characters_present (names), location (name), "
    "key_actions, tone, time_of_day, movement_id, beat_type, plot_devices (list of {name, event, notes}), is_plot_device_intro (bool), plot_device_focus (device name or \"\")."
)



def make_analyze_user(story: str) -> str:
    parts = []
    parts.append("Story (may be expository / no proper names):\n---\n")
    parts.append(story)
    parts.append("\n---\n\nInstructions:\n")
    parts.append("- Return at least TWO main_characters and TWO locations. If not named, invent faithful descriptive handles.\n")
    parts.append("- Characters: include name, initial_description, role, goals, conflicts.\n")
    parts.append("- Locations: include name and description that can be visualized.\n")
    parts.append("- Scenes can be lightweight but must include characters_present and location.\n")
    parts.append("\nReturn JSON exactly in this shape:\n")
    parts.append(
        "{\n"
        '  "title": "",\n'
        '  "logline": "",\n'
        '  "story_precis": "1–2 short paragraphs focusing on protagonist, goal, obstacles, stakes, change.",\n'
        '  "story_summary": "",\n'
        '  "main_characters": [\n'
        '    {"name":"","initial_description":"","role":"","goals":"","conflicts":""}\n'
        "  ],\n"
        '  "locations": [\n'
        '    {"name":"","description":""}\n'
        "  ],\n"
        '  "structure": {\n'
        '    "movements": [\n'
        '      {\n'
        '        "id":"M1",\n'
        '        "name":"",\n'
        '        "focus":"",\n'
        '        "start_scene_id":"S1",\n'
        '        "end_scene_id":"S3",\n'
        '        "key_events":[""],\n'
        '        "emotional_shift":"",\n'
        '        "stakes_change":""\n'
        '      }\n'
        '    ]\n'
        "  },\n"
        '  "scenes": [\n'
        "    {\n"
        '      "id":"S1",\n'
        '      "title":"",\n'
        '      "what_happens":"",\n'
        '      "description":"",\n'
        '      "characters_present":[""],\n'
        '      "location":"",\n'
        '      "key_actions":[""],\n'
        '      "tone":"",\n'
        '      "time_of_day":"",\n'
        '      "movement_id":"M1",\n'
        '      "beat_type":"opening|inciting|first_turn|midpoint|second_turn|crisis|climax|denouement"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )
    return "".join(parts)




SHOTS_SYSTEM = (
    "You are a seasoned storyboard artist and cinematographer. Respond with a SINGLE JSON object. "
    "Return a SINGLE JSON OBJECT. No markdown/code fences/prose. "
    "For each scene, propose 1–3 **expansive** shot prompts (120–220 words each) that combine:"
    " composition rules (framing, balance, depth layers), lens+camera height, blocking, gesture, "
    "light (key/fill/rim, color temperature, direction), atmosphere (haze, rain, dust, fog), palette, "
    "texture, dynamic range, contrast handling, and mood. "
    "Always assume WIDE aspect 21:9. Avoid proper-noun camera brand jargon. "
    "Respect constraints: no text, no watermark, no logos. Return JSON only."
)
def shots_system(aspect_label: str) -> str:
    return (
        "You are a seasoned storyboard artist and cinematographer. Respond with a SINGLE JSON object. "
        "Return a SINGLE JSON OBJECT. No markdown/code fences/prose. "
        "For each scene, propose 1–3 expansive shot prompts (120–220 words each) that combine: "
        "composition rules (framing, balance, depth layers), lens+camera height, blocking, gesture, "
        "light (key/fill/rim, color temperature, direction), atmosphere (haze, rain, dust, fog), palette, "
        "texture, dynamic range, contrast handling, and mood. "
        f"Assume aspect {aspect_label}. Avoid proper-noun camera brand jargon. "
        # Exterior/vehicle bias without dropping character coverage
        "When ships/vehicles/aircraft/rovers or large exterior scale are present or implied, ensure at least "
        "one shot is an EXTERIOR/ENVIRONMENTAL action view that favors the craft/environment over people; "
        "if people appear, keep them small in frame or silhouetted (do not center faces). "
        "Use motion cues when appropriate (thruster plumes, starfield parallax, contrails, dust trails, motion blur). "
        # Composition guardrails to avoid 'sheepish portraits'
        "Hard avoid: direct‑to‑camera gaze, selfie/posed portraits, centered head‑and‑shoulders, mugshot symmetry. "
        "Favor off‑axis or over‑the‑shoulder viewpoints, layered depth (foreground/midground/background), and "
        "environmental scale cues. Respect constraints: no text, no watermark, no logos. Return JSON only."
    )



def make_shots_user(
    story_summary: str,
    characters_ctx: List[Dict[str, Any]],
    locations_ctx: List[Dict[str, Any]],
    scenes: List[Dict[str, Any]],
    global_style: str,
    aspect_label: str,

) -> str:
    parts: List[str] = []

    summary = (story_summary or "").strip()
    parts.append("Story summary:\n")
    parts.append(summary if summary else "(none)")



    if characters_ctx:
        parts.append("\n\nCharacters (with visual DNA):\n")
        for c in characters_ctx:
            if not isinstance(c, dict):
                continue
            name = (c.get("name") or "").strip()
            dna = (c.get("dna") or c.get("description") or "").strip()
            if name or dna:
                parts.append(f"- {name or '(unnamed)'}: {dna or '(no description)'}\n")

    if locations_ctx:
        parts.append("\nLocations:\n")
        for l in locations_ctx:
            if not isinstance(l, dict):
                continue
            name = (l.get("name") or "").strip()
            desc = (l.get("description") or l.get("dna") or "").strip()
            if name or desc:
                parts.append(f"- {name or '(unnamed)'}: {desc or '(no description)'}\n")

    if scenes:
        parts.append("\nScenes (chronological):\n")

        for sc in scenes:
            if not isinstance(sc, dict):
                continue
            sid = (sc.get("id") or "").strip() or "(no id)"
            title = (sc.get("title") or sc.get("what_happens") or "").strip()
            where = (sc.get("location") or "").strip()
            who = ", ".join([c for c in (sc.get("characters_present") or []) if isinstance(c, str)])
            beat = (sc.get("beat_type") or "").strip()
            actions = ", ".join([a for a in (sc.get("key_actions") or []) if isinstance(a, str)])
            what = (sc.get("what_happens") or sc.get("description") or "").strip()


            parts.append(f"• {sid} — {title}\n")
            if where:
                parts.append(f"  Location: {where}\n")
            if who:
                parts.append(f"  Characters: {who}\n")
            if beat:
                parts.append(f"  Beat type: {beat}\n")
            if actions:
                parts.append(f"  Key actions: {actions}\n")
            if what:
                parts.append(f"  What happens: {what}\n")

    if global_style and global_style != "No global style":
        parts.append(f"\nGlobal visual style: {global_style}.")

    parts.append(f"\nAspect to assume: {aspect_label or DEFAULT_ASPECT}.")
    parts.append(f"\nForbidden elements: {NEGATIVE_TERMS}.")
    parts.append("\nFor each scene, deliver 1–3 cinematic shot entries covering different angles/coverage.")
    parts.append(
        "\nReturn JSON exactly as: {\n"
        "  \"shots\": [\n"
        "    {\"id\":\"\", \"scene_id\":\"\", \"title\":\"\", \"shot_description\":\"\", \"prompt\":\"\", \"continuity_notes\":\"\"}\n"
        "  ]\n"
        "}\n"
    )

    return "".join(parts)


def make_analyze_user(story: str) -> str:
    parts = []
    parts.append("Story (may be expository / no proper names):\n---\n")
    parts.append(story)
    parts.append("\n---\n\nInstructions:\n")
    parts.append("- Return at least TWO main_characters and TWO locations. If not named, invent faithful descriptive handles.\n")
    parts.append("- Characters: include name, initial_description, role, goals, conflicts.\n")
    parts.append("- Locations: include name and description that can be visualized.\n")
    parts.append("- Plot devices: extract objects/artifacts/documents/signals that materially affect the plot (MacGuffins, keys, letters, maps, rings, data cores, virus samples, etc.).\n")
    parts.append("- Scenes must include characters_present and location, and also plot_devices events when relevant.\n")
    parts.append("- IMPORTANT: If a plot‑significant object is introduced, create a dedicated scene that centers the object and the immediate story around it (who introduces it, why it matters), even if there is no character or location change.\n")
    parts.append("\nReturn JSON exactly in this shape:\n")
    parts.append(
        "{\n"
        '  "title": "",\n'
        '  "logline": "",\n'
        '  "story_precis": "1–2 short paragraphs focusing on protagonist, goal, obstacles, stakes, change.",\n'
        '  "story_summary": "",\n'
        '  "main_characters": [\n'
        '    {"name":"", "initial_description":"", "role":"", "goals":"", "conflicts":""}\n'
        '  ],\n'
        '  "locations": [\n'
        '    {"name":"", "description":""}\n'
        '  ],\n'
        '  "structure": {\n'
        '    "movements": [\n'
        '      {"id":"M1", "name":"", "focus":"", "start_scene_id":"S1", "end_scene_id":"S3", "key_events":[""], "emotional_shift":"", "stakes_change":""}\n'
        '    ]\n'
        '  },\n'
        '  "plot_devices": [\n'
        '    {"name":"", "description":"", "first_scene_id":"S1", "significance":"", "recurs":false}\n'
        '  ],\n'
        '  "scenes": [\n'
        "    {\n"
        '      "id":"S1",\n'
        '      "title":"",\n'
        '      "what_happens":"",\n'
        '      "description":"",\n'
        '      "characters_present":[""],\n'
        '      "location":"",\n'
        '      "key_actions":[""],\n'
        '      "tone":"",\n'
        '      "time_of_day":"",\n'
        '      "movement_id":"M1",\n'
        '      "beat_type":"opening|inciting|first_turn|midpoint|second_turn|crisis|climax|denouement",\n'
        '      "plot_devices":[{"name":"", "event":"introduction|use|reveal|loss|foreshadowing", "notes":""}],\n'
        '      "is_plot_device_intro": false,\n'
        '      "plot_device_focus": ""\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )
    return "".join(parts)





CHAR_BASELINE_SYSTEM = (
    "You are a senior character concept prompt writer. Based on the story summary, character details, "
    "and any visual cues from photos, write ONE concise baseline prompt that captures the character's "
    "visual identity for neutral studio reference. Do NOT include camera/angle; those are added elsewhere. "
    "Include: face geometry, hair, eyes (HEX if present), skin tone, age band, build/height, signature outfit/accessories, "
    "distinctive marks, palette cues, demeanor. Return a SINGLE JSON OBJECT. No markdown/code fences/prose. Return JSON only."
)
LOCATION_BASELINE_SYSTEM = (
    "You are a senior production designer. Based on the story summary and the location details, "
    "write ONE concise baseline prompt that captures the set/location identity for neutral reference (no camera/angle). "
    "Include: type/usage, era/time period, geography/setting, architectural style/structure, layout, "
    "materials/textures, color palette, signage/graphics (if any), wear/age, lighting qualities, atmosphere/mood, "
    "signature props/fixtures, and environmental storytelling cues. Return a SINGLE JSON OBJECT. No markdown/code fences/prose. Return JSON only."
)

def make_char_baseline_user(summary: str, name: str, initial: str, refined: str, cues: str) -> str:
    parts = []
    parts.append("Story summary:\n")
    parts.append(summary or "(none)")
    parts.append("\n\nCharacter:\n")
    parts.append("Name: "); parts.append(name); parts.append("\n")
    parts.append("Initial description: "); parts.append(initial or "(none)"); parts.append("\n")
    parts.append("Refined description: "); parts.append(refined or "(none)"); parts.append("\n")
    parts.append("Visual cues from photos: "); parts.append(cues or "(none)"); parts.append("\n\n")
    parts.append('Return JSON:\n{ "prompt": "" }')
    return "".join(parts)


def make_loc_baseline_user(summary: str, name: str, description: str, mood: str, lighting: str, key_props: str, cues: str) -> str:
    parts = []
    parts.append("Story summary:\n")
    parts.append(summary or "(none)")
    parts.append("\n\nLocation:\n")
    parts.append("Name: "); parts.append(name); parts.append("\n")
    parts.append("Description: "); parts.append(description or "(none)"); parts.append("\n")
    parts.append("Mood: "); parts.append(mood or "(none)"); parts.append("\n")
    parts.append("Lighting: "); parts.append(lighting or "(none)"); parts.append("\n")
    parts.append("Key props: "); parts.append(key_props or "(none)"); parts.append("\n")
    parts.append("Visual cues from photos: "); parts.append(cues or "(none)"); parts.append("\n\n")
    parts.append('Return JSON:\n{ "prompt": "" }')
    return "".join(parts)


IMAGE_DESC_SYSTEM = (
    "You are a character visual analyst. From the photo(s), extract compact 'visual DNA' suitable for generation prompts: "
    "age band, build/height, face geometry, skin tone, hair style/color, eye color (HEX if visible), distinctive marks, "
    "signature outfit/accessories, palette cues, demeanor. Return JSON only."
)
IMAGE_DESC_USER_TEXT = 'Describe visual DNA succinctly. Return JSON: {"visual_cues": ""}'

LOCATION_IMAGE_DESC_SYSTEM = (
    "You are a location visual analyst. From the photo(s), extract compact 'set DNA' for generation prompts: "
    "type/usage, era/time period, geography/setting, architectural style/structure, layout, materials/textures, color palette, "
    "signage/graphics, wear/age, lighting qualities, atmosphere/mood, signature props/fixtures. Return JSON only."
)
LOCATION_IMAGE_DESC_USER_TEXT = 'Describe location visual DNA succinctly. Return JSON: {"visual_cues": ""}'

# ---- Storyworld extraction (for persistent "world.json") ----
WORLD_EXTRACT_SYSTEM = (
    "You are a fiction continuity registrar. From the chapter/story text, extract persistent WORLD FACTS "
    "that should remain consistent across many chapters. Keep it concise and visual. "
    "Return ONE JSON object with these keys: "
    "species (list of {name, visual_dna, notes}), "
    "factions (list of {name, brief}), "
    "ships (list of {name, class, visual_dna}), "
    "planets (list of {name, type, visual_dna}), "
    "era (string), aesthetic (string), tech_level (string), tags (string[]), "
    "recurring_characters (string[]), recurring_locations (string[])."
)

from typing import Optional

def make_world_extract_user(story_text: str, prior_world: Optional[dict]) -> str:
    """
    Backwards-compatible signature for Python <3.10 (avoids the 'dict | None' union).
    """
    parts = []
    parts.append("Task: Extract a production-ready world model from the story text.")
    if prior_world:
        try:
            import json as _json
            parts.append("You MAY reuse/extend this prior world when consistent:\n" +
                         _json.dumps(prior_world, ensure_ascii=False, indent=2))
        except Exception:
            pass
    parts.append("Return JSON with keys: title, logline, story_precis, story_summary, main_characters, locations, structure, scenes.")
    return "\n\n".join(parts)


# -----------------------------
# Prompt helpers
# -----------------------------
def default_baseline_prompt(c: CharacterProfile) -> str:
    parts = []
    if c.initial_description: parts.append(c.initial_description)
    if c.refined_description: parts.append(c.refined_description)
    if c.visual_cues_from_photos: parts.append(c.visual_cues_from_photos)
    combo = " ".join(parts).strip() or (c.name + " — main character.")
    return (
        c.name + ": " + combo + ". "
        "Include exact hair geometry & color, eye color (HEX if known), skin tone, age band, physique/height, "
        "signature outfit and accessories, distinctive marks, palette cues, demeanor."
    ).strip()

def build_view_prompt_from_baseline(baseline: str, view_key: str, global_style: str) -> str:
    v = CHAR_SHEET_VIEWS_DEF[view_key]
    tail = (
        " Identity lock: keep the same face geometry, hair style/color, eye color, skin tone, and outfit as the baseline; "
        "do not change these across views. Neutral studio character reference. View: " + v["label"] + "; "
        + v["pose"] + "; framing " + v["framing"] + "; camera " + v["camera"] + ". "
        "Plain mid-gray background, even soft lighting, centered composition, neutral expression, no text, no watermark."
    )
    out = baseline.strip().rstrip(".") + ". " + tail
    if global_style and global_style != "No global style":
        out = out + " Global visual style: " + global_style + "."
    out = out.strip()
    try:
        bias_val = float(EXPOSURE_BIAS)
    except Exception:
        bias_val = 0.0
    try:
        emiss_val = float(EMISSIVE_LEVEL)
    except Exception:
        emiss_val = 0.0
    out = out + "\nExposure control: " + exposure_language(bias_val)
    if abs(emiss_val) >= 0.15:
        out = out + "\nEmissive lighting: " + emissive_language(emiss_val)
    return out

def build_loc_view_prompt_from_baseline(baseline: str, view_key: str, global_style: str) -> str:
    v = LOC_VIEWS_DEF[view_key]
    tail = (
        " Location reference • " + v["label"] + ": " + v["note"] + ". "
        "Accurately preserve architecture, layout, materials and color palette; realistic lighting; no people; "
        "no text, no watermark."
    )
    out = baseline.strip().rstrip(".") + ". " + tail
    if global_style and global_style != "No global style":
        out = out + " Global visual style: " + global_style + "."
    out = out.strip()
    try:
        bias_val = float(EXPOSURE_BIAS)
    except Exception:
        bias_val = 0.0
    try:
        emiss_val = float(EMISSIVE_LEVEL)
    except Exception:
        emiss_val = 0.0
    out = out + "\nExposure control: " + exposure_language(bias_val)
    if abs(emiss_val) >= 0.15:
        out = out + "\nEmissive lighting: " + emissive_language(emiss_val)
    return out

def exposure_language(level: float) -> str:
    """Short, model-friendly guidance for readable exposure."""
    try:
        l = _clamp(float(level), -1.0, 1.0)
    except Exception:
        l = 0.0
    if l >= 0.66:
        return "bright, open shadows; readable midtones; protect highlights; avoid overexposure"
    if l >= 0.33:
        return "balanced exposure; gently lifted shadows; preserve highlight detail"
    if l <= -0.66:
        return "very low-key, moody lighting; avoid total murk; preserve form"
    if l <= -0.33:
        return "low-key look; deeper shadows; avoid crushed blacks; controlled highlights"
    return "neutral exposure; natural contrast; avoid crushed blacks or clipped highlights"

def emissive_language(level: float) -> str:
    """Prompt-only hint for diegetic/practical light sources."""
    try:
        l = _clamp(float(level), -1.0, 1.0)
    except Exception:
        l = 0.0
    if l >= 0.66:
        return ("pronounced diegetic glows from instrument panels and screens; "
                "soft bloom; gentle rim-light; protect highlights")
    if l >= 0.33:
        return ("subtle diegetic glows and practicals; minimal bloom; balanced exposure")
    if l <= -0.33:
        return "no bloom; crisp practical lighting; avoid glows"
    return "naturalistic practical lighting with restrained bloom"

def compose_character_dna(c: CharacterProfile, max_len: int = 3000) -> str:
    parts = []
    if c.sheet_base_prompt: parts.append(c.sheet_base_prompt)
    elif c.refined_description or c.initial_description:
        parts.append((c.refined_description or c.initial_description))
    if c.visual_cues_from_photos: parts.append("Visual cues: " + c.visual_cues_from_photos)
    dna = " ".join(parts)
    dna = re.sub(r"\s+", " ", dna).strip()
    return dna[:max_len]

def compose_location_dna(l: LocationProfile, max_len: int = 3500) -> str:
    parts = []
    if l.sheet_base_prompt: parts.append(l.sheet_base_prompt)
    else:
        if l.description: parts.append(l.description)
        if l.mood: parts.append("Mood: " + l.mood)
        if l.lighting: parts.append("Lighting: " + l.lighting)
        if l.key_props: parts.append("Key props: " + l.key_props)
    if l.visual_cues_from_photos: parts.append("Visual cues: " + l.visual_cues_from_photos)
    dna = " ".join(parts)
    dna = re.sub(r"\s+", " ", dna).strip()
    return dna[:max_len]

# --- Identity helpers (hair continuity + readable lists) ---
HAIR_DESCRIPTOR_PATTERNS: List[tuple[str, str]] = [
    ("strawberry blonde hair", r"\bstrawberry[- ]blonde\b"),
    ("platinum blonde hair", r"\bplatinum[- ]blonde\b"),
    ("dirty blonde hair", r"\bdirty[- ]blonde\b"),
    ("honey blonde hair", r"\bhoney[- ]blonde\b"),
    ("sandy blonde hair", r"\bsandy[- ]blonde\b"),
    ("golden blonde hair", r"\bgolden[- ]blonde\b"),
    ("ash blonde hair", r"\bash[- ]blonde\b"),
    ("white blonde hair", r"\bwhite[- ]blonde\b"),
    ("light blonde hair", r"\blight[- ]blonde\b"),
    ("dark blonde hair", r"\bdark[- ]blonde\b"),
    ("blonde hair", r"\bblonde\b|\bblond\b"),
    ("jet black hair", r"\bjet[- ]black(?:\s+hair|[- ]haired)?\b"),
    ("raven black hair", r"\braven(?:\s+hair|[- ]haired)?\b"),
    ("black hair", r"\bblack(?:\s+hair|[- ]haired)\b"),
    ("chestnut brown hair", r"\bchestnut(?:\s+hair|[- ]haired)?\b"),
    ("dark brown hair", r"\bdark[- ]brown(?:\s+hair|[- ]haired)?\b"),
    ("light brown hair", r"\blight[- ]brown(?:\s+hair|[- ]haired)?\b"),
    ("brown hair", r"\bbrown(?:\s+hair|[- ]haired)\b|\bbrunette\b"),
    ("auburn hair", r"\bauburn(?:\s+hair|[- ]haired)?\b"),
    ("ginger hair", r"\bginger(?:\s+hair|[- ]haired)?\b"),
    ("red hair", r"\bred(?:\s+hair|[- ]haired)\b|\bredhead\b"),
    ("copper hair", r"\bcopper(?:\s+hair|[- ]haired)?\b"),
    ("silver hair", r"\bsilver(?:\s+hair|[- ]haired)?\b"),
    ("grey hair", r"\bgrey(?:\s+hair|[- ]haired)?\b"),
    ("gray hair", r"\bgray(?:\s+hair|[- ]haired)?\b"),
    ("white hair", r"\bwhite(?:\s+hair|[- ]haired)?\b"),
]

def extract_hair_descriptor(text: str) -> Optional[str]:
    """Return a concise hair descriptor (e.g., 'blonde hair') if one is present."""
    if not text:
        return None
    lower = text.lower()
    for label, pattern in HAIR_DESCRIPTOR_PATTERNS:
        if re.search(pattern, lower):
            return label
    return None

def join_clause(items: List[str]) -> str:
    """Join a list into natural language ('A', 'A and B', 'A, B, and C')."""
    filtered = [i for i in items if i]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) == 2:
        return f"{filtered[0]} and {filtered[1]}"
    return ", ".join(filtered[:-1]) + f", and {filtered[-1]}"


def compose_master_scene_prompt(base_prompt: str,
                                sc: Dict[str,Any],
                                movement_info: Dict[str,str],
                                char_dnas: Dict[str,str],
                                loc_dna: str,
                                global_style: str,
                                aspect_label: str) -> str:
    parts = []
    parts.append(base_prompt.strip())
    mv = movement_info or {}
    mv_label = (mv.get("id","") + (" — " + mv.get("name","") if mv.get("name") else "")).strip(" —")
    if mv_label:
        details = []
        if mv.get("focus"): details.append("focus: " + mv["focus"])
        if mv.get("emotional_shift"): details.append("emotional shift: " + mv["emotional_shift"])
        if mv.get("stakes_change"): details.append("stakes: " + mv["stakes_change"])
        parts.append("Movement " + mv_label + ((" (" + "; ".join(details) + ")") if details else ""))
    if char_dnas:
        lines = []
        for n, dna in char_dnas.items():
            if dna:

                lines.append(n + ": " + dna)

        if lines:
            parts.append("Character DNA — " + " | ".join(lines))
    if loc_dna:
        parts.append("Location DNA — " + loc_dna)
    continuity_bits: List[str] = []
    if char_dnas:
        continuity_bits.append(
            "Keep every character's facial structure, eye color, hair color/texture and skin tone consistent with their DNA; layer only explicit story-driven changes such as helmets, scars, grime or wardrobe tweaks without altering the underlying anatomy."
        )
    if loc_dna:
        continuity_bits.append(
            "Preserve the location's core architecture, scale and palette while reflecting story-noted lighting, weather, damage or clutter."
        )
    if continuity_bits:
        parts.append("Continuity guardrails: " + " ".join(continuity_bits))


    if global_style and global_style != "No global style":
        parts.append("Global visual style: " + global_style + ".")
    parts.append("Constraints: " + NEGATIVE_TERMS + ".")
    parts.append(f"Compose for {aspect_label or DEFAULT_ASPECT} aspect.")
    try:
        bias_val = float(EXPOSURE_BIAS)
    except Exception:
        bias_val = 0.0
    try:
        emiss_val = float(EMISSIVE_LEVEL)
    except Exception:
        emiss_val = 0.0
    parts.append("Exposure control: " + exposure_language(bias_val))
    if abs(emiss_val) >= 0.15:
        parts.append("Emissive lighting: " + emissive_language(emiss_val))
    return "\n\n".join([p for p in parts if p])


# -----------------------------
# LLM convenience
# -----------------------------
class LLM:
    @staticmethod
    def analyze_story(client: OpenAIClient, model: str, story: str) -> Dict[str, Any]:
        return client.chat_json(model=model, system=ANALYZE_SYSTEM, user=make_analyze_user(story), temperature=0.2)

    @staticmethod
    def expand_scout_shots(
        client: "OpenAIClient",
        model: str,
        story_summary: str | None = None,
        characters_ctx: List[Dict[str, Any]] | None = None,
        locations_ctx: List[Dict[str, Any]] | None = None,
        scenes: List[Dict[str, Any]] | None = None,
        global_style: str | None = None,
        aspect_label: str | None = None,
        **kwargs,
    ) -> List["ShotPrompt"]:
        """
        Compatibility version:
        - Accepts new names (characters_ctx/locations_ctx) and old names (character_blocks/location_blocks).
        - Accepts an explicit aspect_label; falls back to DEFAULT_ASPECT.
        """
        # Backward-compat keyword aliases
        if characters_ctx is None:
            characters_ctx = kwargs.get("character_blocks") or kwargs.get("characters") or []
        if locations_ctx is None:
            locations_ctx = kwargs.get("location_blocks") or kwargs.get("locations") or []
        if scenes is None:
            scenes = kwargs.get("scene_blocks") or kwargs.get("scenes") or []
        if global_style is None:
            global_style = kwargs.get("style") or ""
        if aspect_label is None:
            aspect_label = kwargs.get("aspect") or DEFAULT_ASPECT
    

        user = make_shots_user(
            story_summary or "(none)",
            characters_ctx, locations_ctx,
            scenes,
            global_style or "",
            aspect_label or DEFAULT_ASPECT

        )
        data = client.chat_json(
            model=model,
            system=shots_system(aspect_label or DEFAULT_ASPECT),
            user=user,
            temperature=0.35
        )

    
        shots: List[ShotPrompt] = []
        for s in data.get("shots", []) or []:
            shots.append(ShotPrompt(
                id=s.get("id",""),
                scene_id=s.get("scene_id",""),
                title=s.get("title",""),
                shot_description=s.get("shot_description",""),
                prompt=s.get("prompt",""),
                continuity_notes=s.get("continuity_notes","")
            ))

        return shots

        
    @staticmethod
    def extract_storyworld_facts(
        client: OpenAIClient,
        model: str,
        story_text: str,
        prior_world: dict | None = None
    ) -> Dict[str, Any]:
        try:
            payload = make_world_extract_user(story_text, prior_world or {})
            data = client.chat_json(
                model=model,
                system=WORLD_EXTRACT_SYSTEM,
                user=payload,
                temperature=0.15
            )
            # Normalize shape
            out = {
                "species": data.get("species", []) or [],
                "factions": data.get("factions", []) or [],
                "ships": data.get("ships", []) or [],
                "planets": data.get("planets", []) or [],
                "era": data.get("era", "") or "",
                "aesthetic": data.get("aesthetic", "") or "",
                "tech_level": data.get("tech_level", "") or "",
                "tags": data.get("tags", []) or [],
                "recurring_characters": data.get("recurring_characters", []) or [],
                "recurring_locations": data.get("recurring_locations", []) or [],
            }
            return out
        except Exception:
            return {
                "species": [], "factions": [], "ships": [], "planets": [],
                "era": "", "aesthetic": "", "tech_level": "", "tags": [],
                "recurring_characters": [], "recurring_locations": []
            }
        
    # @staticmethod
    # def fuse_scene_prompt(client: 'OpenAIClient', model: str,
    #                       ingredients: Dict[str, Any],
    #                       global_style: str,
    #                       negative_terms: str) -> str:
    #     try:
    #         data = client.chat_json(model=model,
    #                                 system=SCENE_FUSION_SYSTEM,
    #                                 user=make_scene_fusion_user(ingredients, global_style, negative_terms),
    #                                 temperature=0.35)
    #         return (data.get("prompt") or "").strip()
    #     except Exception:
    #         return ""
    @staticmethod
    def fuse_scene_prompt(client: 'OpenAIClient', model: str,
                          ingredients: Dict[str, Any],
                          global_style: str,
                          negative_terms: str,
                          aspect_label: str) -> str:
        try:
            data = client.chat_json(
                model=model,
                system=scene_fusion_system(aspect_label or DEFAULT_ASPECT),
                user=make_scene_fusion_user(ingredients, global_style, negative_terms),
                temperature=0.35
            )
            return (data.get("prompt") or "").strip()
        except Exception:
            return ""


    @staticmethod
    def extract_visual_cues_from_image(client: OpenAIClient, model: str, image_bytes: bytes) -> str:
        user_payload = [
            {"type": "text", "text": IMAGE_DESC_USER_TEXT},
            {"type": "image_url", "image_url": {"url": b64_data_uri(image_bytes)}},
        ]
        try:
            data = client.chat_json(model=model, system=IMAGE_DESC_SYSTEM, user=user_payload, temperature=0.1)
            return (data.get("visual_cues") or "").strip()
        except Exception:
            return ""

    @staticmethod
    def extract_visual_cues_from_location_image(client: OpenAIClient, model: str, image_bytes: bytes) -> str:
        user_payload = [
            {"type": "text", "text": LOCATION_IMAGE_DESC_USER_TEXT},
            {"type": "image_url", "image_url": {"url": b64_data_uri(image_bytes)}},
        ]
        try:
            data = client.chat_json(model=model, system=LOCATION_IMAGE_DESC_SYSTEM, user=user_payload, temperature=0.1)
            return (data.get("visual_cues") or "").strip()
        except Exception:
            return ""

    @staticmethod
    def propose_unified_character_prompt(client: OpenAIClient, model: str, summary: str, c: CharacterProfile, extra_cues: str = "") -> str:
        data = client.chat_json(model=model, system=CHAR_BASELINE_SYSTEM,
                                user=make_char_baseline_user(summary or "(none)",
                                                             c.name,
                                                             c.initial_description or "(none)",
                                                             c.refined_description or "(none)",
                                                             extra_cues or c.visual_cues_from_photos or "(none)"),
                                temperature=0.3)
        return (data.get("prompt") or "").strip()

    @staticmethod
    def propose_unified_location_prompt(client: OpenAIClient, model: str, summary: str, l: LocationProfile, extra_cues: str = "") -> str:
        kp = ", ".join([p.strip() for p in (l.key_props or "").split(",") if p.strip()])
        data = client.chat_json(model=model, system=LOCATION_BASELINE_SYSTEM,
                                user=make_loc_baseline_user(summary or "(none)",
                                                            l.name,
                                                            l.description or "(none)",
                                                            l.mood or "(none)",
                                                            l.lighting or "(none)",
                                                            kp or "(none)",
                                                            extra_cues or l.visual_cues_from_photos or "(none)"),
                                temperature=0.3)
        return (data.get("prompt") or "").strip()

# -----------------------------
# UI scaffolding
# -----------------------------
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Create window inside canvas
        self._win = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=vsb.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Resize inner width with the canvas width
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel: bind while cursor is over this ScrollableFrame
        self.scrollable_frame.bind("<Enter>", self._bind_mousewheel, add="+")
        self.scrollable_frame.bind("<Leave>", self._unbind_mousewheel, add="+")

    def _on_canvas_configure(self, event):
        # Keep inner frame the same width as the canvas
        self.canvas.itemconfig(self._win, width=event.width)

    # Cross‑platform wheel handling
    def _on_mousewheel(self, event):
        if event.delta:  # Windows / macOS
            self.canvas.yview_scroll(int(-event.delta/120), "units")
        else:            # Linux
            if event.num == 4:
                self.canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(3, "units")

    def _bind_mousewheel(self, _):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-4>",  self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-5>",  self._on_mousewheel, add="+")

    def _unbind_mousewheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def add_bottom_padding(self, pixels=80):
        """Optional: add some vertical breathing room at the end."""
        ttk.Frame(self.scrollable_frame, height=pixels).pack(fill="x")



class ProgressWindow:
    def __init__(self, root, title="Working…"):
        self.top = tk.Toplevel(root)
        self.top.title(title)
        self.top.geometry("720x380")
        self.var = tk.StringVar(value="Starting…")

        head = ttk.Frame(self.top); head.pack(fill="x", padx=10, pady=(10,4))
        ttk.Label(head, textvariable=self.var, anchor="w").pack(side="left", fill="x", expand=True)

        self.pbar = ttk.Progressbar(self.top, maximum=100, mode="determinate")
        self.pbar.pack(fill="x", padx=10, pady=(0,8))

        # Log area
        logf = ttk.Frame(self.top); logf.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.log = tk.Text(logf, wrap="word", height=12)
        ysb = ttk.Scrollbar(logf, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=ysb.set, state="disabled")
        self.log.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")

        self.top.transient(root)
        self.top.grab_set()
        self.top.protocol("WM_DELETE_WINDOW", lambda: None)

    def set_status(self, text: str):
        self.var.set(text)
        self.top.update_idletasks()

    def set_progress(self, value: float):
        self.pbar["value"] = max(0, min(100, value if value is not None else 0))
        self.top.update_idletasks()

    def append_log(self, text: str):
        try:
            self.log.configure(state="normal")
            self.log.insert("end", (text or "") + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        except Exception:
            pass
        self.top.update_idletasks()

    def close(self):
        try:
            self.top.grab_release()
        except Exception:
            pass
        self.top.destroy()


class EnrichmentDialog:
    """
    Minimal, blocking dialog that returns a dict of optional enrichments, or {} if skipped.
    """
    def __init__(self, root, scene_id: str):
        self.top = tk.Toplevel(root)
        self.top.title(f"Enrich scene {scene_id}")
        self.top.geometry("560x420")
        self.ans: Dict[str, str] = {}
        self._build()

    def _build(self):
        frm = ttk.Frame(self.top, padding=10); frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Add only what's missing. Leave blank to skip a field.").pack(anchor="w", pady=(0,8))

        self.vars = {
            "emotional_beat": tk.StringVar(),
            "atmosphere_weather": tk.StringVar(),
            "color_palette": tk.StringVar(),
            "props_motifs": tk.StringVar(),
            "camera_movement": tk.StringVar(),
        }
        fields = [
            ("Emotional beat / subtext", "emotional_beat"),
            ("Atmosphere or weather (e.g., mist, rain, dust, smoke, neon haze)", "atmosphere_weather"),
            ("Color palette cues (2–4 colors or adjectives)", "color_palette"),
            ("Signature props / motifs to include", "props_motifs"),
            ("Camera movement or style (e.g., slow dolly, handheld)", "camera_movement"),
        ]
        for label, key in fields:
            lf = ttk.Labelframe(frm, text=label); lf.pack(fill="x", pady=6)
            ttk.Entry(lf, textvariable=self.vars[key]).pack(fill="x", padx=6, pady=6)

        btns = ttk.Frame(frm); btns.pack(fill="x", pady=(12,0))
        ttk.Button(btns, text="Use these", command=self._ok).pack(side="left")
        ttk.Button(btns, text="Skip", command=self._skip).pack(side="left", padx=8)
        self.top.transient(self.top.master); self.top.grab_set()

    def _ok(self):
        out = {}
        for k, v in self.vars.items():
            t = v.get().strip()
            if t:
                out[k] = t
        self.ans = out
        self.top.destroy()

    def _skip(self):
        self.ans = {}
        self.top.destroy()

    def show(self) -> Dict[str, str]:
        self.top.wait_window()
        return self.ans
# =============================
# Profile matching & batch utils
# =============================
import unicodedata
from difflib import SequenceMatcher
from glob import glob

def _normalize_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii", errors="ignore")
    s = re.sub(r"[\W_]+", " ", s.lower()).strip()
    return s

def _char_ngrams(s: str, n: int = 4) -> set:
    s = re.sub(r"\s+", " ", s)
    if len(s) < n: return {s} if s else set()
    return {s[i:i+n] for i in range(0, len(s)-n+1)}

def _jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    inter = len(a & b); uni = len(a | b)
    return inter / max(1, uni)

# very light attribute extraction to anchor identity (hair/eyes/age/etc.)
_HAIR = r"(black|brown|blonde|blond|red|ginger|gray|grey|white|silver|auburn|pink|blue|green|purple)"
_EYES = r"(brown|blue|green|hazel|gray|grey|amber|violet)"
_GENDERISH = r"\b(male|man|boy|female|woman|girl|androgyn|nonbinary|non-binary)\b"
def _extract_identity_features(text: str) -> dict:
    t = _normalize_text(text)
    feats = {}
    m = re.search(rf"\b({_HAIR})\s+hair\b", t)
    if m: feats["hair"] = m.group(1)
    m = re.search(rf"\b({_EYES})\s+eyes\b", t)
    if m: feats["eyes"] = m.group(1)
    m = re.search(r"\b(\d{1,2})\s*(?:yo|y/o|years? old)\b", t)
    if m: feats["age_num"] = m.group(1)
    m = re.search(_GENDERISH, t)
    if m: feats["genderish"] = m.group(1)
    return feats

def _feature_overlap(a: dict, b: dict) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys: return 0.0
    hits = 0
    for k in keys:
        if k in a and k in b and a[k] == b[k]:
            hits += 1
    return hits / len(keys)

@dataclass
class ProfileEntry:
    kind: str             # "character" | "location"
    name: str
    prompt: str           # sheet_base_prompt or description DNA
    json_path: str
    image_paths: List[str] = field(default_factory=list)

class ProfileIndex:
    """
    Scans a directory tree for character/location profile JSONs and
    builds a lightweight text signature for robust cross-story matching.
    """
    def __init__(self):
        self.entries: List[ProfileEntry] = []

    def scan(self, root: str) -> None:
        if not root or not os.path.isdir(root): return
        for jp in glob(os.path.join(root, "**", "*.json"), recursive=True):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if not isinstance(data, dict): 
                continue
            t = (data.get("type") or "").strip()
            if t not in ("character_profile", "location_profile"): 
                continue

            kind = "character" if t == "character_profile" else "location"
            name = (data.get("name") or "").strip()
            if not name: 
                continue

            # pick best DNA-ish prompt to represent the profile
            dna = (data.get("sheet_base_prompt") or 
                   data.get("refined_description") or 
                   data.get("initial_description") or 
                   data.get("description") or 
                   "")

            # collect image paths (absolute)
            imgs = []
            imgdict = (data.get("images") or {}) if isinstance(data.get("images"), dict) else {}
            base = os.path.dirname(jp)
            for _, arr in imgdict.items():
                for item in (arr or []):
                    if isinstance(item, dict) and item.get("path"):
                        pth = item["path"]
                        if not os.path.isabs(pth):
                            pth = os.path.normpath(os.path.join(base, pth))
                        if os.path.isfile(pth):
                            imgs.append(pth)
            self.entries.append(ProfileEntry(kind=kind, name=name, prompt=dna, json_path=jp, image_paths=imgs))

    def _score(self, kind: str, cand_name: str, cand_prompt: str, entry: ProfileEntry) -> float:
        if entry.kind != kind:
            return 0.0
        # Name similarity
        a = sanitize_name(cand_name or "")
        b = sanitize_name(entry.name or "")
        name_sim = SequenceMatcher(None, a, b).ratio() if (a or b) else 0.0
        # Prompt similarity (character n-grams)
        A = _char_ngrams(_normalize_text(cand_prompt), 4)
        B = _char_ngrams(_normalize_text(entry.prompt), 4)
        j = _jaccard(A, B)
        # Identity anchors (helps when names drift)
        fa = _extract_identity_features(cand_prompt)
        fb = _extract_identity_features(entry.prompt)
        f = _feature_overlap(fa, fb)

        # blended score; prompt dominates
        return 0.55*j + 0.30*name_sim + 0.15*f

    def match(self, kind: str, cand_name: str, cand_prompt: str,
              threshold: float = 0.42, min_margin: float = 0.05) -> Optional[ProfileEntry]:
        if not self.entries:
            return None
        scored = [(self._score(kind, cand_name, cand_prompt, e), e) for e in self.entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return None
        top, second = scored[0], (scored[1] if len(scored) > 1 else (0.0, None))
        if top[0] >= threshold and (top[0] - (second[0] or 0.0)) >= min_margin:
            return top[1]
        return None

def _guess_view_from_filename(fn: str) -> str:
    s = fn.lower()
    m = re.search(r"_(front|profile_left|profile_right|three_quarter_left|three_quarter_right|back|full_body_tpose|establishing|alt_angle|detail)_", s)
    if m: return m.group(1)
    # weak fallbacks
    if "profile" in s and "left" in s: return "profile_left"
    if "profile" in s and "right" in s: return "profile_right"
    if "three" in s and "left" in s: return "three_quarter_left"
    if "three" in s and "right" in s: return "three_quarter_right"
    if "establish" in s: return "establishing"
    if "detail" in s: return "detail"
    return ""

def _asset_id_for_path(path: str) -> str:
    try:
        return "img_" + hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "img_" + hashlib.sha1((path or "x").encode("utf-8")).hexdigest()[:16]


class App:
    def __init__(self, root=None, headless=False):
        self.root = root
        self.headless = headless
        if self.root and not self.headless:
            self.root.title("Story → World JSON (OpenAI-only)")
            self.root.geometry("1380x1000")
    
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.llm_model = DEFAULT_LLM_MODEL
        self.image_model = OPENAI_IMAGE_MODEL
        self.image_size = DEFAULT_IMAGE_SIZE
        self.global_style = GLOBAL_STYLE_DEFAULT

        self.exposure_bias = float(EXPOSURE_BIAS)
        self.post_tonemap = bool(EXPOSURE_POST_TONEMAP)
        self.emissive_level = float(EMISSIVE_LEVEL)
        self.output_dir: str = ""
        self.input_text_path: str = ""
        self._last_story_path: str = ""
        self._last_export_dir: str = ""
        self._dialogue_story_text_cache: str = ""
        self.save_dialogue_btn: Optional[ttk.Button] = None

        # --- Aspect controls ---
        self.char_ref_aspect = "1:1"             # used for Character ref sheets
        self.loc_ref_aspect  = "1:1"             # used for Location ref sheets
        self.scene_render_aspect = DEFAULT_ASPECT  # used for shots/export/renders

        # --- Extra images control (word-gap) ---
        self.min_words_between_images = EXTRA_IMAGES_MIN_WORDS_DEFAULT

        # NEW: persistent world store (optional)
        self.world_store_path: str = ""          # if set, we read/write world.json here
        self.world: Dict[str, Any] = self._new_world_template()
        try:
            self.world.setdefault("style_presets", [])
        except Exception:
            self.world["style_presets"] = []
        self._user_styles: List[Dict[str, Any]] = self.world.get("style_presets", [])
        try:
            self._load_user_styles()
        except Exception:
            pass
        self.selected_style_id: str = ""
        self.selected_style_name: str = ""
        self._style_combo_mapping: Dict[str, Dict[str, Any]] = {}
        self._style_display_by_id: Dict[str, str] = {}
        self._style_manager_refs: List[Any] = []

        self.client: Optional[OpenAIClient] = None
        self.analysis: Optional[Dict[str, Any]] = None
        self.characters: Dict[str, CharacterProfile] = {}
        self.locations: Dict[str, LocationProfile] = {}
        self.scenes_by_id: Dict[str, Dict[str, Any]] = {}
        self.shots: List[ShotPrompt] = []
        self.assets: List[AssetRecord] = []
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._undo_scenes_stack: List[List[Dict[str, Any]]] = []
        self.scene_enrichment_vars: Dict[str, Dict[str, tk.Variable]] = {}
        # Encode cache: (sha1, side, qual, fmt) -> data_uri
        self._encode_cache: Dict[tuple, str] = {}
        self._scene_story_segments: Dict[str, str] = {}
        self._story_anchor_map_by_outdir: Dict[str, Dict[str, str]] = {}
        self._story_scene_titles_by_outdir: Dict[str, Dict[str, str]] = {}
        self._story_title_by_outdir: Dict[str, str] = {}
    
        if not self.headless and self.root:
            self._build_ui()
            # ✅ Install cross‑platform mouse‑wheel scrolling now that the UI exists
            self._install_global_mousewheel()
#############################################################################
    # ---------- Identity resolution helpers (drop-in) ----------
    
    def _name_tokens(self, s: str) -> set:
        """Lowercase, strip punctuation/underscores, drop filler tokens/synonyms."""
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        raw = [t for t in s.split() if t]
        # words to ignore when matching names (helps match Starship/Vessel etc.)
        FILLERS = {
            "the","a","an","of","and","room","hall","chamber","office","hq","headquarters",
            "ship","starship","vessel","craft","space","virtual","sector","department",
            "human","ai","xcc"
        }
        return {t for t in raw if t not in FILLERS}
    
    def _name_similarity(self, a: str, b: str) -> float:
        A, B = self._name_tokens(a), self._name_tokens(b)
        if not A and not B: return 0.0
        return len(A & B) / float(len(A | B))
    
    def _prompt_similarity(self, a: str, b: str) -> float:
        """Blend char-gram Jaccard with lightweight identity features."""
        A = _char_ngrams(_normalize_text(a or ""), 4)
        B = _char_ngrams(_normalize_text(b or ""), 4)
        j = _jaccard(A, B)
        fa = _extract_identity_features(a or "")
        fb = _extract_identity_features(b or "")
        f = _feature_overlap(fa, fb)
        return 0.70 * j + 0.30 * f
    
    def _gather_candidates(self, kind: str) -> list:
        """
        Collect known candidates from current state + loaded world.json.
        Returns list of tuples: (name, prompt_dna).
        """
        out = []
        if kind == "character":
            for nm, c in (self.characters or {}).items():
                dna = (getattr(c, "sheet_base_prompt", "") or
                       getattr(c, "refined_description", "") or
                       getattr(c, "initial_description", "") or "")
                out.append((nm, dna))
            src = (self.world or {}).get("characters") or {}
            if isinstance(src, dict):
                for nm, d in src.items():
                    dna = (d.get("sheet_base_prompt") or d.get("refined_description") or
                           d.get("initial_description") or d.get("description") or "")
                    out.append((nm, dna))
            elif isinstance(src, list):
                for d in src:
                    nm = (d.get("name") or "")
                    dna = (d.get("sheet_base_prompt") or d.get("refined_description") or
                           d.get("initial_description") or d.get("description") or "")
                    if nm:
                        out.append((nm, dna))
        else:
            for nm, l in (self.locations or {}).items():
                dna = (getattr(l, "sheet_base_prompt", "") or getattr(l, "description", "") or "")
                out.append((nm, dna))
            src = (self.world or {}).get("locations") or {}
            if isinstance(src, dict):
                for nm, d in src.items():
                    dna = (d.get("sheet_base_prompt") or d.get("description") or "")
                    out.append((nm, dna))
            elif isinstance(src, list):
                for d in src:
                    nm = (d.get("name") or "")
                    dna = (d.get("sheet_base_prompt") or d.get("description") or "")
                    if nm:
                        out.append((nm, dna))
        # Deduplicate by name keeping longest DNA
        by_name = {}
        for nm, dna in out:
            if nm in by_name and len(by_name[nm]) >= len(dna):
                continue
            by_name[nm] = dna
        return [(k, v) for k, v in by_name.items()]
    
    def _resolve_entity_name(self, kind: str, cand_name: str, cand_prompt: str,
                             threshold: float = 0.62) -> str:
        """
        Return canonical name if we find a strong match; otherwise return original.
        Mixes name-token and prompt-DNA similarities.
        """
        cand_name = (cand_name or "").strip()
        cand_prompt = cand_prompt or ""
        best_name, best = cand_name, 0.0
        for nm, dna in self._gather_candidates(kind):
            # don't compare to itself
            if sanitize_name(nm) == sanitize_name(cand_name):
                return nm
            name_sim = self._name_similarity(cand_name, nm)
            prompt_sim = self._prompt_similarity(cand_prompt, dna or "")
            score = 0.55 * name_sim + 0.45 * prompt_sim
            if score > best:
                best, best_name = score, nm
        return best_name if best >= threshold else cand_name
    
    def _merge_or_rename_profile(self, kind: str, old_name: str, new_name: str) -> None:
        """
        Merge `old_name` profile into `new_name` (or rename if new doesn't exist).
        Updates scenes to use canonical name. Non-destructive to existing fields.
        """
        if old_name == new_name:
            return
    
        if kind == "character":
            src = self.characters.get(old_name)
            dst = self.characters.get(new_name)
            if not dst:
                self.characters[new_name] = src
                del self.characters[old_name]
            else:
                # Fill gaps only
                for k in ("initial_description","refined_description","role","goals","conflicts"):
                    if getattr(dst, k, "") or not getattr(src, k, ""):
                        continue
                    setattr(dst, k, getattr(src, k, ""))
                if getattr(src, "sheet_base_prompt", "") and not getattr(dst, "sheet_base_prompt", ""):
                    dst.sheet_base_prompt = src.sheet_base_prompt
                # Union image refs
                dst.reference_images = list(dict.fromkeys((dst.reference_images or []) + (src.reference_images or [])))
                # Merge any in-memory sheet images/selections
                for vkey, imgs in (getattr(src, "sheet_images", {}) or {}).items():
                    dst.sheet_images.setdefault(vkey, []).extend(imgs or [])
                for vkey, flags in (getattr(src, "sheet_selected", {}) or {}).items():
                    dst.sheet_selected.setdefault(vkey, []).extend(flags or [])
                # Drop old
                del self.characters[old_name]
            # Update scenes
            for s in (self.analysis.get("scenes", []) or []):
                s["characters_present"] = [new_name if x == old_name else x for x in (s.get("characters_present", []) or [])]
    
        else:
            src = self.locations.get(old_name)
            dst = self.locations.get(new_name)
            if not dst:
                self.locations[new_name] = src
                del self.locations[old_name]
            else:
                if getattr(src, "description", "") and not getattr(dst, "description", ""):
                    dst.description = src.description
                if getattr(src, "sheet_base_prompt", "") and not getattr(dst, "sheet_base_prompt", ""):
                    dst.sheet_base_prompt = src.sheet_base_prompt
                dst.reference_images = list(dict.fromkeys((dst.reference_images or []) + (src.reference_images or [])))
                for vkey, imgs in (getattr(src, "sheet_images", {}) or {}).items():
                    dst.sheet_images.setdefault(vkey, []).extend(imgs or [])
                for vkey, flags in (getattr(src, "sheet_selected", {}) or {}).items():
                    dst.sheet_selected.setdefault(vkey, []).extend(flags or [])
                del self.locations[old_name]
            # Update scenes
            for s in (self.analysis.get("scenes", []) or []):
                if (s.get("location") or "") == old_name:
                    s["location"] = new_name

#######################################################################################
    def _match_or_create_character_profile(
        self,
        story_name: str,
        repo: dict,
        profiles_dir: str,
        generate_images: bool,
        match_threshold: float,
        min_margin: float,
        # NEW: creation controls (ignored on reuse path)
        views = None,
        per_view = None,
    ):
        """
        If a compatible repo profile exists, reuse it (adopt baseline if ours empty, attach assets).
        Otherwise generate refs now using the specified views/per_view, write profile JSON next to
        the images under profiles_dir/characters/<name>, and update the in‑memory repo map.
        """
        import json, os, hashlib
    
        c = self.characters.get(story_name)
        if not c:
            return
    
        # Prepare incoming signature (DNA) text
        if not (c.sheet_base_prompt or "").strip():
            try:
                c.sheet_base_prompt = (
                    LLM.propose_unified_character_prompt(
                        self.client, self.llm_model,
                        (self.analysis or {}).get("story_summary",""),
                        c,
                        extra_cues=c.visual_cues_from_photos
                    ) or compose_character_dna(c)
                )
            except Exception:
                c.sheet_base_prompt = compose_character_dna(c)
        incoming_sig = self._tokenize_signature(story_name + " " + compose_character_dna(c))
    
        # Repo candidates
        candidates = list((repo.get("characters") or {}).values())
        scored = [(self._score_candidate(story_name, incoming_sig, cand), cand) for cand in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        use_match = None
        if scored and scored[0][0] >= match_threshold:
            if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= min_margin:
                use_match = scored[0][1]
    
        if use_match:
            # Reuse: attach images as assets (de-dup by path), adopt baseline if ours empty
            try:
                with open(use_match["json_path"], "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
    
            if not c.sheet_base_prompt:
                c.sheet_base_prompt = data.get("sheet_base_prompt","") or c.sheet_base_prompt
    
            for vkey, arr in (use_match.get("images") or {}).items():
                for p in (arr or []):
                    if not os.path.isfile(p):
                        continue
                    aid = "img_" + hashlib.sha1(os.path.abspath(p).encode("utf-8")).hexdigest()[:16]
                    # de-dupe by path
                    if any(a.file_path == p for a in (self.assets or [])):
                        if aid not in c.reference_images:
                            c.reference_images.append(aid)
                        continue
                    self.assets.append(AssetRecord(
                        id=aid, entity_type="character", entity_name=story_name, view=vkey,
                        prompt_full=(c.sheet_base_prompt or ""), model=self.image_model, size=self.image_size,
                        file_path=p, created_at=now_iso(), notes="character sheet • " + vkey
                    ))
                    if aid not in c.reference_images:
                        c.reference_images.append(aid)
            print(f"[char] {story_name}: reused repo profile '{use_match.get('name','?')}'.")
            return
    
        # -------- No match → create refs now (honor per‑view knobs) --------
        if not generate_images:
            return
    
        char_folder = os.path.join(profiles_dir, "characters", sanitize_name(story_name))
        ensure_dir(char_folder)
    
        # Honor per‑view knobs (create path only)
        use_views = list(views) if views else ["front","profile_left","profile_right"]
        use_per_view = int(max(1, per_view if per_view is not None else 1))
    
        gen = self._generate_character_refs_now(
            name=story_name,
            dest_dir=char_folder,
            baseline=c.sheet_base_prompt,
            views=use_views,
            per_view=use_per_view,
            aspect=self.char_ref_aspect if hasattr(self, "char_ref_aspect") else "1:1"
        )
    
        # Persist JSON next to images and update repo in memory
        jp = self._save_character_profile_json(story_name, char_folder, c, gen["images"])
        repo.setdefault("characters", {})[story_name] = {
            "name": story_name,
            "json_path": jp,
            "folder": char_folder,
            "sheet_base_prompt": c.sheet_base_prompt,
            "visual_cues": c.visual_cues_from_photos,
            "description": c.refined_description or c.initial_description or "",
            "images": gen["images"],
            "signature_tokens": self._tokenize_signature(story_name + " " + compose_character_dna(c))
        }
        print(f"[char] {story_name}: created profile ({sum(len(v) for v in gen['images'].values())} image(s)).")

    def run_batch_on_folder(
        self,
        stories_dir: str,
        profiles_dir: str,
        out_root: str,
        prompt_policy: str = "final_prompt",
        render_n: int = 1,
        delay_s: int = 1,
        aspect: str = DEFAULT_ASPECT,
        match_threshold: float = 0.44,
        min_margin: float = 0.06,
        create_minimal_profiles_for_new: bool = True,
        # creation controls (CREATE paths only)
        char_views: tuple[str, ...] = ("front", "profile_left", "profile_right"),
        char_per_view: int = 1,
        loc_views:  tuple[str, ...] = ("establishing", "alt_angle"),
        loc_per_view: int = 1,
        # coverage for final images ("min" or "max")
        coverage_mode: str = "min",
        # extra coverage via word-gap (None → use UI/default)
        min_words_per_image: int | None = None,
        # callbacks
        progress_cb=None,
        log_cb=None,
    ) -> None:
        """
        Headless batch runner (repo‑first). For each *.txt in stories_dir:
          • Analyze → seed Characters/Locations
          • Apply world baselines/assets (from profiles_dir/world.json if present)
          • For each entity → repo match‑or‑create (auto‑saves refs + profile JSON on create)
          • Author shots (accept ShotPrompt objects; assign directly)
          • Export scene JSONs (coverage-aware) + update per-story world.json
          • Auto-render (coverage-aware)
        Word-gap extras can be controlled via `min_words_per_image` (None → respect UI/default).
        Profiles repo layout:
          profiles_dir/
            characters/<sanitize(name)>/{profile.json, images.*}
            locations/<sanitize(name)>/{profile.json, images.*}
        """
        # stable defaults for callbacks (fixes your 'log_cb' crash)
        progress_cb = progress_cb or (lambda pct, text=None: None)
        log_cb = log_cb or (lambda line: None)
    
        import os, json, traceback
        from pathlib import Path
    
        # Silence GUI popups while running headless
        try:
            import tkinter.messagebox as _mb
            _saved_mb = (getattr(_mb, "showinfo", None),
                         getattr(_mb, "showwarning", None),
                         getattr(_mb, "showerror", None))
            _mb.showinfo = _mb.showwarning = _mb.showerror = (lambda *a, **k: None)
        except Exception:
            _mb = None
            _saved_mb = None
    
        try:
            # -------- Ensure repo folders --------
            if not profiles_dir:
                profiles_dir = os.path.join(out_root, "_profiles_repo")
            os.makedirs(os.path.join(profiles_dir, "characters"), exist_ok=True)
            os.makedirs(os.path.join(profiles_dir, "locations"), exist_ok=True)
    
            # -------- Connect OpenAI if needed --------
            self.scene_render_aspect = aspect
            if not getattr(self, "client", None):
                api_key = os.environ.get("OPENAI_API_KEY", "") or getattr(self, "api_key", "")
                if not api_key:
                    raise RuntimeError("OpenAI API key missing (set OPENAI_API_KEY).")
                self.client = OpenAIClient(api_key)
    
            # -------- Load world.json (if present) & apply baselines/assets --------
            wpath = os.path.join(profiles_dir, "world.json")
            if os.path.isfile(wpath):
                self.world_store_path = wpath
                try:
                    self._refresh_world_from_path()
                except Exception as e:
                    print(f"[world] failed to load/apply {wpath}: {e}")
    
            # -------- Build repo index once --------
            repo = self._load_profiles_repo(profiles_dir)
    
            # -------- Pull UI batch knobs when present (UI overrides signature defaults) --------
            # Characters
            try:
                ui_char_views = [k for k, v in getattr(self, "batch_char_views_vars", {}).items() if bool(v.get())]
            except Exception:
                ui_char_views = []
            use_char_views = ui_char_views or list(char_views)
            try:
                ui_char_per = int(getattr(self, "batch_char_per_view_spin").get())
            except Exception:
                ui_char_per = int(char_per_view)
            use_char_per = max(1, ui_char_per)
    
            # Locations
            try:
                ui_loc_views = [k for k, v in getattr(self, "batch_loc_views_vars", {}).items() if bool(v.get())]
            except Exception:
                ui_loc_views = []
            use_loc_views = ui_loc_views or list(loc_views)
            try:
                ui_loc_per = int(getattr(self, "batch_loc_per_view_spin").get())
            except Exception:
                ui_loc_per = int(loc_per_view)
            use_loc_per = max(1, ui_loc_per)
            
            # Min words per image (batch override when present)
            effective_min_words = min_words_per_image
            if effective_min_words is None:
                try:
                    if (
                        threading.current_thread() is threading.main_thread()
                        and hasattr(self, "batch_min_words_var")
                    ):
                        effective_min_words = int(getattr(self, "batch_min_words_var").get())
                except Exception:
                    effective_min_words = None
            if effective_min_words is not None:
                try:
                    self.min_words_between_images = max(0, int(effective_min_words))
                except Exception:
                    pass
                
            # -------- Collect stories --------
            stories = sorted(Path(stories_dir).glob("*.txt"))
            total = len(stories)
            progress_cb(1.0, f"Found {total} stor{'y' if total == 1 else 'ies'}…")
            log_cb(f"[batch] Found {total} .txt file(s)")
    
            if not stories:
                print(f"[batch] No .txt files found in {stories_dir}")
                return
    
            # Adapter so render path (which uses msg,pct) reports into this (pct,text) API
            def _render_progress_adapter(msg: str, pct: float | None = None):
                progress_cb(0 if pct is None else pct, msg)
    
            for si, txt in enumerate(stories):
                slug = txt.stem
                per_story_steps = 5.0  # analyze, profiles, shots, export, render
                base = (si / max(1, total)) * 100.0
                step_size = (1.0 / max(1, total)) * (100.0 / per_story_steps)
    
                def _tick(step_no: float, label: str):
                    pct = min(99.0, base + step_size * step_no)
                    progress_cb(pct, f"[{slug}] {label}")
                    log_cb(f"[{slug}] {label}")
    
                _tick(0, "Starting…")
    
                print(f"\n=== [{slug}] ===")
                try:
                    # ---- Analyze story → seed state
                    story_text = txt.read_text(encoding="utf-8", errors="ignore")
                    self._last_story_text = story_text
                    self.input_text_path = str(txt)
                    self._last_story_path = str(txt)

                    try:
                        analysis = LLM.analyze_story(self.client, self.llm_model, story_text)
                    except Exception as e:
                        print(f"[{slug}] analyze failed: {e}")
                        continue
    
                    if not isinstance(analysis, dict):
                        analysis = {}
                    analysis = self._ensure_nonempty_entities(analysis, story_text)
    
                    scenes = analysis.get("scenes", []) or []
                    for idx, s in enumerate(scenes, 1):
                        if isinstance(s, dict):
                            s.setdefault("id", f"S{idx}" if not s.get("id") else s.get("id"))
                            s.setdefault("characters_present", list(s.get("characters_present", []) or []))
                            s.setdefault("key_actions", list(s.get("key_actions", []) or []))
    
                    # Mount fresh app state for this story
                    self.analysis = analysis
                    self.scenes_by_id = {s.get("id", ""): s for s in scenes if isinstance(s, dict) and s.get("id")}
                    self.characters = {}
                    self.locations = {}
                    self.shots = []
    
                    # ---- Seed characters/locations from analysis (strip names safely)
                    for c in (analysis.get("main_characters") or []):
                        nm = (c.get("name", "") or "").strip()
                        if nm:
                            self.characters[nm] = CharacterProfile(
                                name=nm,
                                initial_description=c.get("initial_description",""),
                                role=c.get("role",""),
                                goals=c.get("goals",""),
                                conflicts=c.get("conflicts",""),
                                refined_description=c.get("refined_description",""),
                                visual_cues_from_photos=c.get("visual_cues_from_photos","")
                            )
                    for l in (analysis.get("locations") or []):
                        nm = (l.get("name","") or "").strip()
                        if nm:
                            self.locations[nm] = LocationProfile(
                                name=nm,
                                description=l.get("description","")
                            )
    
                    # ---- Apply world baselines/assets
                    try:
                        self._apply_world_baselines_to_state(create_missing=True)
                    except TypeError:
                        # older builds
                        self._apply_world_baselines_to_state()
    
                    # ---- Repo-first: match or create (auto-saves refs + profile JSON on create)
                    for name in list(self.characters.keys()):
                        self._match_or_create_character_profile(
                            story_name=name,
                            repo=repo,
                            profiles_dir=profiles_dir,
                            generate_images=True,
                            match_threshold=match_threshold,
                            min_margin=min_margin,
                            views=use_char_views,
                            per_view=use_char_per,
                        )
                    for name in list(self.locations.keys()):
                        self._match_or_create_location_profile(
                            story_name=name,
                            repo=repo,
                            profiles_dir=profiles_dir,
                            generate_images=True,
                            match_threshold=match_threshold,
                            min_margin=min_margin,
                            views=use_loc_views,
                            per_view=use_loc_per,
                        )
    
                    # ---- Author shots (accept ShotPrompt objects as‑is)
                    try:
                        char_blocks = [
                            {"name": nm,
                             "dna": compose_character_dna(c),
                             "description": (c.refined_description or c.initial_description or "")}
                            for nm, c in self.characters.items()
                        ]
                        loc_blocks = [
                            {"name": nm,
                             "dna": compose_location_dna(l),
                             "description": (l.description or "")}
                            for nm, l in self.locations.items()
                        ]
                        shots_raw = LLM.expand_scout_shots(
                            self.client, self.llm_model,
                            story_summary=(analysis.get("story_summary","") or ""),
                            characters_ctx=char_blocks,
                            locations_ctx=loc_blocks,
                            scenes=scenes,
                            global_style=self.global_style,

                            aspect_label=self.scene_render_aspect

                        )
                        # Important: do NOT rewrap ShotPrompt objects
                        self.shots = shots_raw or []
                    except Exception as e:
                        print(f"[{slug}] shots failed: {e}")
                        self.shots = []
    
                    # ---- Export enriched scene JSONs (coverage-aware) + per‑story world.json
                    story_out = os.path.join(out_root, txt.stem); os.makedirs(story_out, exist_ok=True)
                    self.world_store_path = os.path.join(story_out, "world.json")
                    try:
                        self._auto_export_scene_jsons_sync(story_out, coverage_mode=(coverage_mode or "min"))
                    except Exception as e:
                        print(f"[{slug}] export failed: {e}")
    
                    # ---- Render images from exported scenes (coverage-aware)
                    try:
                        scenes_dir = os.path.join(story_out, SCENE_SUBDIR_NAME)
                        self._auto_render_from_scene_folder(
                            folder=scenes_dir,
                            n=int(max(1, render_n)),
                            policy_label=prompt_policy,
                            delay_s=int(max(0, delay_s)),
                            coverage_mode=(coverage_mode or "min"),
                            progress_cb=_render_progress_adapter  # keep UI progress in one place
                        )
                    except Exception as e:
                        print(f"[{slug}] render failed: {e}")
    
                except RuntimeError as e:
                    print(f"[{slug}] {e}")
                    continue
                except Exception as e:
                    print(f"[{slug}] unexpected error: {e}")
                    try:
                        traceback.print_exc()
                    except Exception:
                        pass
                    continue
    
            print("\n[batch] Done.")
        finally:
            # restore messagebox stubs
            if _mb and _saved_mb:
                try:
                    _mb.showinfo, _mb.showwarning, _mb.showerror = _saved_mb
                except Exception:
                    pass




    # ---------- Repo loader & matcher ----------
    
    def _load_profiles_repo(self, profiles_dir: str) -> dict:
        """
        Scans profiles_dir/{characters,locations}/**/*.json
        Returns:
          {
            'characters': { canonical_name: { 'name', 'json_path', 'folder', 'sheet_base_prompt', 'visual_cues',
                                              'description', 'images': {view:[abs paths]}, 'signature_tokens': set(...) } },
            'locations':  { ... same idea ... }
          }
        The loader tolerates 'images': {view: [{path: "..."}]} or simple lists.
        """
        import json, os, glob
        def _one(kind):
            base = os.path.join(profiles_dir, kind)
            items = {}
            for jp in glob.glob(os.path.join(base, "**", "*.json"), recursive=True):
                try:
                    with open(jp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                nm = (data.get("name") or os.path.splitext(os.path.basename(jp))[0]).strip()
                if not nm:
                    continue
                folder = os.path.dirname(jp)
                images = {}
                raw = data.get("images") or data.get("views") or {}
                if isinstance(raw, dict):
                    for vkey, arr in raw.items():
                        paths = []
                        for item in (arr or []):
                            if isinstance(item, dict) and item.get("path"):
                                p = os.path.join(folder, item["path"]) if not os.path.isabs(item["path"]) else item["path"]
                                paths.append(os.path.normpath(p))
                        if paths:
                            images[vkey] = paths
                sig_text = " ".join([
                    data.get("sheet_base_prompt","") or "",
                    data.get("visual_cues_from_photos","") or "",
                    data.get("initial_description","") or "",
                    data.get("refined_description","") or "",
                    data.get("description","") or "",
                ]).strip()
                items[nm] = {
                    "name": nm,
                    "json_path": jp,
                    "folder": folder,
                    "sheet_base_prompt": data.get("sheet_base_prompt","") or "",
                    "visual_cues": data.get("visual_cues_from_photos","") or "",
                    "description": data.get("description","") or "",
                    "images": images,
                    "signature_tokens": self._tokenize_signature(nm + " " + sig_text)
                }
            return items
        return {"characters": _one("characters"), "locations": _one("locations")}
    
    def _tokenize_signature(self, text: str) -> set:
        """
        Lowercases, strips punctuation, keeps alphanumerics + key hyphenated color terms.
        Returns a set of tokens for Jaccard overlap.
        """
        import re
        t = (text or "").lower()
        t = re.sub(r"[^a-z0-9#\-\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        toks = [w for w in t.split() if len(w) > 2]
        return set(toks)
    
    def _score_candidate(self, incoming_name: str, incoming_sig_tokens: set, candidate: dict) -> float:
        """
        Combines name similarity (SequenceMatcher) and token Jaccard on signatures.
        Returns a 0..1 score.
        """
        from difflib import SequenceMatcher
        def _norm(s): 
            import re
            return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
        name_sim = SequenceMatcher(None, _norm(incoming_name), _norm(candidate.get("name",""))).ratio()
        cand_tokens = candidate.get("signature_tokens") or set()
        jacc = 0.0
        if incoming_sig_tokens and cand_tokens:
            inter = len(incoming_sig_tokens & cand_tokens)
            union = len(incoming_sig_tokens | cand_tokens)
            if union:
                jacc = inter / union
        # Weighted blend—tilt slightly toward prompt DNA over the name text
        return 0.45 * name_sim + 0.55 * jacc
    
    # ---------- Character: match or create ----------
    # def _match_or_create_location_profile(
    #     self,
    #     story_name: str,
    #     repo: dict,
    #     profiles_dir: str,
    #     generate_images: bool,
    #     match_threshold: float,
    #     min_margin: float
    # ):
    #     import json, os, hashlib
    #     L = self.locations.get(story_name)
    #     if not L:
    #         return
    
    #     if not (L.sheet_base_prompt or "").strip():
    #         try:
    #             L.sheet_base_prompt = LLM.propose_unified_location_prompt(
    #                 self.client, self.llm_model, (self.analysis or {}).get("story_summary",""), L,
    #                 extra_cues=L.visual_cues_from_photos
    #             ) or ""
    #         except Exception:
    #             L.sheet_base_prompt = ""
    #     incoming_sig = self._tokenize_signature(story_name + " " + compose_location_dna(L))
    
    #     candidates = list((repo.get("locations") or {}).values())
    #     scored = [(self._score_candidate(story_name, incoming_sig, cand), cand) for cand in candidates]
    #     scored.sort(key=lambda x: x[0], reverse=True)
    #     use_match = None
    #     if scored and scored[0][0] >= match_threshold:
    #         if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= min_margin:
    #             use_match = scored[0][1]
    
    #     if use_match:
    #         try:
    #             with open(use_match["json_path"], "r", encoding="utf-8") as f:
    #                 data = json.load(f)
    #         except Exception:
    #             data = {}
    #         if not L.sheet_base_prompt:
    #             L.sheet_base_prompt = data.get("sheet_base_prompt","") or L.sheet_base_prompt
    #         if not L.description:
    #             L.description = data.get("description","") or L.description
    #         for vkey, arr in (use_match.get("images") or {}).items():
    #             for p in arr:
    #                 if not os.path.isfile(p):
    #                     continue
    #                 aid = "img_" + hashlib.sha1(os.path.abspath(p).encode("utf-8")).hexdigest()[:16]
    #                 if any(a.file_path == p for a in (self.assets or [])):
    #                     if aid not in L.reference_images:
    #                         L.reference_images.append(aid)
    #                     continue
    #                 self.assets.append(AssetRecord(
    #                     id=aid, entity_type="location", entity_name=story_name, view=vkey,
    #                     prompt_full=(L.sheet_base_prompt or ""), model=self.image_model, size=self.image_size,
    #                     file_path=p, created_at=now_iso(), notes="location ref • " + vkey
    #                 ))
    #                 if aid not in L.reference_images:
    #                     L.reference_images.append(aid)
    #         return
    
    #     # -------- No match → create refs now (HONOR batch controls) --------
    #     loc_folder = os.path.join(profiles_dir, "locations", sanitize_name(story_name))
    #     ensure_dir(loc_folder)
    
    #     if not (L.sheet_base_prompt or "").strip():
    #         try:
    #             L.sheet_base_prompt = LLM.propose_unified_location_prompt(
    #                 self.client, self.llm_model, (self.analysis or {}).get("story_summary",""), L,
    #                 extra_cues=L.visual_cues_from_photos
    #             ) or compose_location_dna(L)
    #         except Exception:
    #             L.sheet_base_prompt = compose_location_dna(L)
    
    #     # Read UI or default views/count
    #     try:
    #         loc_views = [k for k, v in getattr(self, "batch_loc_views_vars", {}).items() if bool(v.get())]
    #     except Exception:
    #         loc_views = []
    #     if not loc_views:
    #         loc_views = ["establishing","alt_angle"]
    #     try:
    #         loc_per = max(1, int(getattr(self, "batch_loc_per_view_spin").get()))
    #     except Exception:
    #         loc_per = 1
    
    #     gen = self._generate_location_refs_now(
    #         name=story_name,
    #         dest_dir=loc_folder,
    #         baseline=L.sheet_base_prompt,
    #         views=loc_views,
    #         per_view=int(loc_per),
    #         aspect=self.loc_ref_aspect if hasattr(self, "loc_ref_aspect") else "1:1"
    #     )
    #     jp = self._save_location_profile_json(story_name, loc_folder, L, gen["images"])
    #     repo.setdefault("locations", {})[story_name] = {
    #         "name": story_name,
    #         "json_path": jp,
    #         "folder": loc_folder,
    #         "sheet_base_prompt": L.sheet_base_prompt,
    #         "visual_cues": L.visual_cues_from_photos,
    #         "description": L.description or "",
    #         "images": gen["images"],
    #         "signature_tokens": self._tokenize_signature(story_name + " " + compose_location_dna(L))
    #     }
    def _match_or_create_location_profile(
        self,
        story_name: str,
        repo: dict,
        profiles_dir: str,
        generate_images: bool,
        match_threshold: float,
        min_margin: float,
        # NEW: creation controls (ignored on reuse path)
        views = None,
        per_view = None,
    ):
        """
        If a compatible repo profile exists, reuse it (adopt baseline if ours empty, attach assets).
        Otherwise generate refs now using the specified views/per_view, write profile JSON next to
        the images under profiles_dir/locations/<name>, and update the in‑memory repo map.
        """
        import json, os, hashlib
    
        L = self.locations.get(story_name)
        if not L:
            return
    
        # Prepare incoming signature (DNA) text
        if not (L.sheet_base_prompt or "").strip():
            try:
                L.sheet_base_prompt = (
                    LLM.propose_unified_location_prompt(
                        self.client, self.llm_model,
                        (self.analysis or {}).get("story_summary",""),
                        L,
                        extra_cues=L.visual_cues_from_photos
                    ) or compose_location_dna(L)
                )
            except Exception:
                L.sheet_base_prompt = compose_location_dna(L)
        incoming_sig = self._tokenize_signature(story_name + " " + compose_location_dna(L))
    
        # Repo candidates
        candidates = list((repo.get("locations") or {}).values())
        scored = [(self._score_candidate(story_name, incoming_sig, cand), cand) for cand in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        use_match = None
        if scored and scored[0][0] >= match_threshold:
            if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= min_margin:
                use_match = scored[0][1]
    
        if use_match:
            # Reuse: attach images as assets (de-dup by path), adopt baseline/description if ours empty
            try:
                with open(use_match["json_path"], "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
    
            if not L.sheet_base_prompt:
                L.sheet_base_prompt = data.get("sheet_base_prompt","") or L.sheet_base_prompt
            if not L.description:
                L.description = data.get("description","") or L.description
    
            for vkey, arr in (use_match.get("images") or {}).items():
                for p in (arr or []):
                    if not os.path.isfile(p):
                        continue
                    aid = "img_" + hashlib.sha1(os.path.abspath(p).encode("utf-8")).hexdigest()[:16]
                    # de-dupe by path
                    if any(a.file_path == p for a in (self.assets or [])):
                        if aid not in L.reference_images:
                            L.reference_images.append(aid)
                        continue
                    self.assets.append(AssetRecord(
                        id=aid, entity_type="location", entity_name=story_name, view=vkey,
                        prompt_full=(L.sheet_base_prompt or ""), model=self.image_model, size=self.image_size,
                        file_path=p, created_at=now_iso(), notes="location ref • " + vkey
                    ))
                    if aid not in L.reference_images:
                        L.reference_images.append(aid)
            print(f"[loc] {story_name}: reused repo profile '{use_match.get('name','?')}'.")
            return
    
        # -------- No match → create refs now (honor per‑view knobs) --------
        if not generate_images:
            return
    
        loc_folder = os.path.join(profiles_dir, "locations", sanitize_name(story_name))
        ensure_dir(loc_folder)
    
        # Honor per‑view knobs (create path only)
        use_views = list(views) if views else ["establishing","alt_angle"]
        use_per_view = int(max(1, per_view if per_view is not None else 1))
    
        gen = self._generate_location_refs_now(
            name=story_name,
            dest_dir=loc_folder,
            baseline=L.sheet_base_prompt,
            views=use_views,
            per_view=use_per_view,
            aspect=self.loc_ref_aspect if hasattr(self, "loc_ref_aspect") else "1:1"
        )
    
        # Persist JSON next to images and update repo in memory
        jp = self._save_location_profile_json(story_name, loc_folder, L, gen["images"])
        repo.setdefault("locations", {})[story_name] = {
            "name": story_name,
            "json_path": jp,
            "folder": loc_folder,
            "sheet_base_prompt": L.sheet_base_prompt,
            "visual_cues": L.visual_cues_from_photos,
            "description": L.description or "",
            "images": gen["images"],
            "signature_tokens": self._tokenize_signature(story_name + " " + compose_location_dna(L))
        }
        print(f"[loc] {story_name}: created profile ({sum(len(v) for v in gen['images'].values())} image(s)).")


    
    # ---------- Location: match or create ----------
    
    def _match_or_create_location_profile(
        self,
        story_name: str,
        repo: dict,
        profiles_dir: str,
        generate_images: bool,
        match_threshold: float,
        min_margin: float,
        # NEW: creation controls (optional; ignored on reuse path)
        views = None,
        per_view = None,
    ):
        import json, os, hashlib
        L = self.locations.get(story_name)
        if not L:
            return
    
        # Prepare incoming signature text
        if not (L.sheet_base_prompt or "").strip():
            try:
                L.sheet_base_prompt = LLM.propose_unified_location_prompt(
                    self.client, self.llm_model, (self.analysis or {}).get("story_summary",""), L,
                    extra_cues=L.visual_cues_from_photos
                ) or ""
            except Exception:
                L.sheet_base_prompt = ""
        incoming_sig = self._tokenize_signature(story_name + " " + compose_location_dna(L))
    
        candidates = list((repo.get("locations") or {}).values())
        scored = [(self._score_candidate(story_name, incoming_sig, cand), cand) for cand in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        use_match = None
        if scored and scored[0][0] >= match_threshold:
            if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= min_margin:
                use_match = scored[0][1]
    
        if use_match:
            try:
                with open(use_match["json_path"], "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
    
            if not L.sheet_base_prompt:
                L.sheet_base_prompt = data.get("sheet_base_prompt","") or L.sheet_base_prompt
            if not L.description:
                L.description = data.get("description","") or L.description
    
            for vkey, arr in (use_match.get("images") or {}).items():
                for p in arr:
                    if not os.path.isfile(p):
                        continue
                    aid = "img_" + hashlib.sha1(os.path.abspath(p).encode("utf-8")).hexdigest()[:16]
                    if any(a.file_path == p for a in (self.assets or [])):
                        if aid not in L.reference_images:
                            L.reference_images.append(aid)
                        continue
                    self.assets.append(AssetRecord(
                        id=aid, entity_type="location", entity_name=story_name, view=vkey,
                        prompt_full=(L.sheet_base_prompt or ""), model=self.image_model, size=self.image_size,
                        file_path=p, created_at=now_iso(), notes="location ref • " + vkey
                    ))
                    if aid not in L.reference_images:
                        L.reference_images.append(aid)
            return
    
        # -------- No match → create refs now --------
        loc_folder = os.path.join(profiles_dir, "locations", sanitize_name(story_name))
        ensure_dir(loc_folder)
    
        if not (L.sheet_base_prompt or "").strip():
            try:
                L.sheet_base_prompt = LLM.propose_unified_location_prompt(
                    self.client, self.llm_model, (self.analysis or {}).get("story_summary",""), L,
                    extra_cues=L.visual_cues_from_photos
                ) or compose_location_dna(L)
            except Exception:
                L.sheet_base_prompt = compose_location_dna(L)
    
        # Honor per‑view knobs on creation path only
        use_views = list(views) if views else ["establishing","alt_angle"]
        use_per_view = int(max(1, per_view if per_view is not None else 1))
    
        gen = self._generate_location_refs_now(
            name=story_name,
            dest_dir=loc_folder,
            baseline=L.sheet_base_prompt,
            views=use_views,
            per_view=use_per_view,
            aspect=self.loc_ref_aspect if hasattr(self, "loc_ref_aspect") else "1:1"
        )
        jp = self._save_location_profile_json(story_name, loc_folder, L, gen["images"])
        repo.setdefault("locations", {})[story_name] = {
            "name": story_name,
            "json_path": jp,
            "folder": loc_folder,
            "sheet_base_prompt": L.sheet_base_prompt,
            "visual_cues": L.visual_cues_from_photos,
            "description": L.description or "",
            "images": gen["images"],
            "signature_tokens": self._tokenize_signature(story_name + " " + compose_location_dna(L))
        }

    
    # ---------- Generators (no UI; direct Images API) ----------
    
    def _generate_character_refs_now(self, name: str, dest_dir: str, baseline: str,
                                     views: list[str], per_view: int, aspect: str) -> dict:
        """
        Generates character reference images for the given baseline & views, writes them to dest_dir,
        registers them as assets on self.assets, and links ids to self.characters[name].
        Returns { "images": {view:[abs_paths]} }.
        """
        import os, time, hashlib
        c = self.characters[name]
        out = {"images": {}}
        size_to_use = self.aspect_to_size(aspect)
        for vkey in views:
            prompt = build_view_prompt_from_baseline(baseline, vkey, self.global_style)
            imgs = self._try_images_generate(prompt, n=int(max(1, per_view)), size=size_to_use)
            paths = []
            for i, b in enumerate(imgs, 1):
                ext = sniff_ext_from_bytes(b, ".png")
                ts  = time.strftime("%Y%m%d_%H%M%S")
                fn  = f"{sanitize_name(name)}_{vkey}_{i}_{ts}{ext}"
                fp  = os.path.join(dest_dir, fn)
                processed_bytes, _ = self._process_generated_image(b, ext=ext, need_image=False)
                with open(fp, "wb") as f:
                    f.write(processed_bytes)
                paths.append(fp)
                aid = "img_" + hashlib.sha1(os.path.abspath(fp).encode("utf-8")).hexdigest()[:16]
                self.assets.append(AssetRecord(
                    id=aid, entity_type="character", entity_name=name, view=vkey,
                    prompt_full=(baseline or ""), model=self.image_model, size=self.image_size,
                    file_path=fp, created_at=now_iso(), notes="character sheet • " + vkey
                ))
                if aid not in c.reference_images:
                    c.reference_images.append(aid)
            if paths:
                out["images"][vkey] = paths
        return out
    
    def _generate_location_refs_now(self, name: str, dest_dir: str, baseline: str,
                                    views: list[str], per_view: int, aspect: str) -> dict:
        """
        Generates location reference images for the given baseline & views, writes to dest_dir,
        registers them as assets on self.assets, and links ids to self.locations[name].
        Returns { "images": {view:[abs_paths]} }.
        """
        import os, time, hashlib
        L = self.locations[name]
        out = {"images": {}}
        size_to_use = self.aspect_to_size(aspect)
        for vkey in views:
            prompt = build_loc_view_prompt_from_baseline(baseline, vkey, self.global_style)
            imgs = self._try_images_generate(prompt, n=int(max(1, per_view)), size=size_to_use)
            paths = []
            for i, b in enumerate(imgs, 1):
                ext = sniff_ext_from_bytes(b, ".png")
                ts  = time.strftime("%Y%m%d_%H%M%S")
                fn  = f"{sanitize_name(name)}_{vkey}_{i}_{ts}{ext}"
                fp  = os.path.join(dest_dir, fn)
                processed_bytes, _ = self._process_generated_image(b, ext=ext, need_image=False)
                with open(fp, "wb") as f:
                    f.write(processed_bytes)
                paths.append(fp)
                aid = "img_" + hashlib.sha1(os.path.abspath(fp).encode("utf-8")).hexdigest()[:16]
                self.assets.append(AssetRecord(
                    id=aid, entity_type="location", entity_name=name, view=vkey,
                    prompt_full=(baseline or ""), model=self.image_model, size=self.image_size,
                    file_path=fp, created_at=now_iso(), notes="location ref • " + vkey
                ))
                if aid not in L.reference_images:
                    L.reference_images.append(aid)
            if paths:
                out["images"][vkey] = paths
        return out
    
    # ---------- Writers (profile JSON next to images) ----------
    
    def _save_character_profile_json(self, name: str, folder: str, c: CharacterProfile, images: dict) -> str:
        """
        Writes a compact character_profile JSON that references images by relative path.
        Returns the absolute JSON path.
        """
        import json, os
        payload = {
            "type": "character_profile",
            "name": name,
            "initial_description": c.initial_description,
            "refined_description": c.refined_description,
            "role": c.role,
            "goals": c.goals,
            "conflicts": c.conflicts,
            "visual_cues_from_photos": c.visual_cues_from_photos,
            "sheet_base_prompt": c.sheet_base_prompt,
            "images": {},
            "created_at": now_iso()
        }
        for vkey, arr in (images or {}).items():
            for p in arr:
                payload["images"].setdefault(vkey, []).append({
                    "filename": os.path.basename(p),
                    "path": os.path.relpath(p, start=folder)
                })
        jp = os.path.join(folder, f"{sanitize_name(name)}.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return jp
    
    def _save_location_profile_json(self, name: str, folder: str, L: LocationProfile, images: dict) -> str:
        """
        Writes a compact location_profile JSON that references images by relative path.
        Returns the absolute JSON path.
        """
        import json, os
        payload = {
            "type": "location_profile",
            "name": name,
            "description": L.description,
            "mood": L.mood,
            "lighting": L.lighting,
            "key_props": [p.strip() for p in (L.key_props or "").split(",") if p.strip()],
            "visual_cues_from_photos": L.visual_cues_from_photos,
            "sheet_base_prompt": L.sheet_base_prompt,
            "images": {},
            "created_at": now_iso()
        }
        for vkey, arr in (images or {}).items():
            for p in arr:
                payload["images"].setdefault(vkey, []).append({
                    "filename": os.path.basename(p),
                    "path": os.path.relpath(p, start=folder)
                })
        jp = os.path.join(folder, f"{sanitize_name(name)}.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return jp

##########################################################
    # -----------------------------
    # Batch runner: stories folder → images
    # -----------------------------
    def _reset_session(self):
        """Clear state between stories (lightweight; no UI dependency)."""
        self.analysis = None
        self.characters = {}
        self.locations = {}
        self.scenes_by_id = {}
        self.shots = []
        # keep assets across stories? We’ll reuse/carry assets so IDs stay stable and copying is deduped
        # but we DO allow adding per-story “aliases” (same id reused under new entity_name)
        if not hasattr(self, "assets") or self.assets is None:
            self.assets = []

    def _ingest_profile_for_entity(self, kind: str, target_name: str, entry: ProfileEntry) -> Dict[str, Any]:
        """
        Link an existing profile to the current story entity:
          - copy its DNA/baseline text into the App profile
          - register its images as AssetRecord(s) under *this* target_name
          - attach their IDs to CharacterProfile/LocationProfile.reference_images
        Returns a small dict with bookkeeping (ids, json_path).
        """
        ids = []
        now = now_iso()
        if kind == "character":
            c = self.characters.get(target_name) or CharacterProfile(name=target_name, initial_description="")
            if not self.characters.get(target_name):
                self.characters[target_name] = c
            # adopt baseline-ish DNA (prefer sheet_base_prompt if available)
            if entry.prompt and not c.sheet_base_prompt:
                c.sheet_base_prompt = entry.prompt
            # add assets under the *current* name
            for pth in (entry.image_paths or []):
                aid = _asset_id_for_path(pth)
                view = _guess_view_from_filename(os.path.basename(pth))
                # avoid exact duplicate record (same id + same entity_name)
                if not any((a.id == aid and a.entity_type == "character" and a.entity_name == target_name) for a in (self.assets or [])):
                    self.assets.append(AssetRecord(
                        id=aid, entity_type="character", entity_name=target_name,
                        view=view, prompt_full="", model=self.image_model, size=self.image_size,
                        file_path=pth, created_at=now, notes=view
                    ))
                ids.append(aid)
            # attach
            if ids:
                c.reference_images = list(dict.fromkeys((c.reference_images or []) + ids))
            return {"ids": ids, "json": entry.json_path}

        else:  # location
            l = self.locations.get(target_name) or LocationProfile(name=target_name, description="")
            if not self.locations.get(target_name):
                self.locations[target_name] = l
            if entry.prompt and not l.sheet_base_prompt:
                l.sheet_base_prompt = entry.prompt
            for pth in (entry.image_paths or []):
                aid = _asset_id_for_path(pth)
                view = _guess_view_from_filename(os.path.basename(pth))
                if not any((a.id == aid and a.entity_type == "location" and a.entity_name == target_name) for a in (self.assets or [])):
                    self.assets.append(AssetRecord(
                        id=aid, entity_type="location", entity_name=target_name,
                        view=view, prompt_full="", model=self.image_model, size=self.image_size,
                        file_path=pth, created_at=now, notes=view
                    ))
                ids.append(aid)
            if ids:
                # stash on LocationProfile via a symmetric slot
                if not hasattr(l, "reference_images") or l.reference_images is None:
                    l.reference_images = []
                l.reference_images = list(dict.fromkeys((l.reference_images or []) + ids))
            return {"ids": ids, "json": entry.json_path}



    ##########################################################################

    def _new_world_template(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "story_world": {
                "species": [],
                "factions": [],
                "ships": [],
                "planets": [],
                "era": "",
                "aesthetic": "",
                "tech_level": "",
                "tags": []
            },
            "characters": {},   # name -> {"sheet_base_prompt": "..."}
            "locations": {},    # name -> {"sheet_base_prompt": "..."}
            "assets_registry": [],
            "style_presets": [],
            "default_style_id": ""
        }

    def _trim_world_for_size(self, world: dict, max_mb: int = 15) -> dict:
        """
        Ensure the scene JSON stays under max_mb. This is a conservative, loss-aware trimmer:
          1) Drop reference_gallery first (keeps identity lock in conditioning).
          2) Remove attachment_manifest if still too large.
          3) Keep only conditioning entries actually referenced (first item per entity if needed).
          4) Strip inline data_uri fields (prefer 'path' or id).
          5) Trim long text fields (what_happens/dna) mildly as last resort.
    
        Returns a new dict.
        """
        import copy, json
        limit = max(1, int(max_mb)) * 1024 * 1024
    
        def size_bytes(obj):
            try:
                return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
            except Exception:
                return 10**9
    
        w = copy.deepcopy(world)
        if size_bytes(w) <= limit:
            return w
    
        scene = w.get("scene", {}) or {}
    
        # 1) Remove reference_gallery entirely
        if "reference_gallery" in scene:
            scene["reference_gallery"] = {"characters": {}, "locations": {}}
            w["scene"] = scene
        if size_bytes(w) <= limit:
            return w
    
        # 2) Remove attachment_manifest (nice-to-have only)
        scene.pop("attachment_manifest", None)
        w["scene"] = scene
        if size_bytes(w) <= limit:
            return w
    
        # 3) Tighten conditioning to first item per entity
        def tighten_cond(group: dict) -> dict:
            out = {}
            for ent, arr in (group or {}).items():
                if not arr:
                    continue
                item = arr[0]
                if isinstance(item, dict):
                    kept = {"id": item.get("id")}
                    if item.get("path"):
                        kept["path"] = item["path"]
                    out[ent] = [kept]
                else:
                    out[ent] = [item]
            return out
    
        cond = scene.get("conditioning", {}) or {}
        chars = cond.get("characters", {})
        locs  = cond.get("locations", {})
        if chars or locs:
            scene["conditioning"] = {
                "characters": tighten_cond(chars),
                "locations":  tighten_cond(locs),
            }
            w["scene"] = scene
        if size_bytes(w) <= limit:
            return w
    
        # 4) Drop any remaining inline data_uris in gallery/conditioning (defensive)
        def strip_data_uris(group: dict) -> dict:
            out = {}
            for ent, arr in (group or {}).items():
                pruned = []
                for item in arr or []:
                    if isinstance(item, dict):
                        item = {"id": item.get("id"), **({"path": item["path"]} if item.get("path") else {})}
                    pruned.append(item)
                if pruned:
                    out[ent] = pruned
            return out
    
        refgal = scene.get("reference_gallery", {}) or {}
        rg_chars = refgal.get("characters", {})
        rg_locs  = refgal.get("locations", {})
        scene["reference_gallery"] = {
            "characters": strip_data_uris(rg_chars),
            "locations":  strip_data_uris(rg_locs),
        }
        w["scene"] = scene
        if size_bytes(w) <= limit:
            return w
    
        # 5) Mildly trim long text fields
        def trim_text(s: str, n: int) -> str:
            s = s or ""
            return (s[:n] + "…") if len(s) > n else s
    
        if isinstance(scene.get("what_happens"), str):
            scene["what_happens"] = trim_text(scene["what_happens"], 4000)
        if isinstance(scene.get("fused_prompt"), str):
            scene["fused_prompt"] = trim_text(scene["fused_prompt"], 4000)
        w["scene"] = scene
        return w

    def _build_scene_prompt_ingredients(self, world: dict, sc: dict, global_style: str, negative_terms: str):
        """
        Compatibility wrapper used by the export path:
          returns (ingredients_dict, fused_prompt_string)
    
        Internally uses the already-present _build_prompt_ingredients(...) and
        _compose_final_generation_prompt(...).
        """
        ingredients = self._build_prompt_ingredients(world, sc)
        fused = self._compose_final_generation_prompt(ingredients)
        return ingredients, fused

    def _install_global_mousewheel(self):
        # Windows / macOS
        self.root.bind_all("<MouseWheel>", self._on_global_mousewheel, add="+")
        # Linux
        self.root.bind_all("<Button-4>", self._on_global_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_global_mousewheel, add="+")
    
    def _on_global_mousewheel(self, event):
        """
        Robust global mousewheel handler:
          - Coerce string widget names to actual widgets (nametowidget)
          - Walk up to a scrollable ancestor safely
          - Ignore if nothing scrollable is found
        """
        try:
            # Normalize delta across platforms
            delta = getattr(event, "delta", 0)
            try:
                # On Windows, delta is multiples of 120
                step = 1 if delta > 0 else -1
                if abs(delta) >= 120:
                    step *= abs(delta) // 120
            except Exception:
                # Fallback if delta is weird/non-numeric
                step = 1 if str(delta).strip().startswith(("+", "1")) else -1

            # event.widget can be a widget OR a string path (e.g., ".!frame.!canvas")
            w = getattr(event, "widget", None)

            # If it's a string, try to resolve to a widget object; otherwise bail
            if isinstance(w, str):
                try:
                    w = self.root.nametowidget(w)
                except Exception:
                    return

            # Climb to something scrollable (yview/xview/canvas/tree)
            MAX_HOPS = 25
            hops = 0
            while w is not None:
                # If widget supports yview/xview, use it
                if hasattr(w, "yview"):
                    w.yview_scroll(-step, "units")
                    return
                if hasattr(w, "xview"):
                    w.xview_scroll(-step, "units")
                    return

                # If there's no parent to climb to, stop
                if not hasattr(w, "master"):
                    break

                w = w.master
                hops += 1
                if hops > MAX_HOPS:
                    break
            # Nothing scrollable found — silently ignore
        except Exception as e:
            # Don't spam Tk with tracebacks; log once per event is enough
            print("[wheel] suppressed:", e)


    # -------- OpenAI connect & image 403 helper --------
    def _on_connect(self):
        self.api_key = self.api_entry.get().strip() or os.environ.get("OPENAI_API_KEY","")
        self.llm_model = self.llm_combo.get().strip()
        self.image_model = self.img_combo.get().strip()
        self.image_size = self.size_combo.get().strip()
        display = self.style_combo.get().strip()
        info = self._style_combo_mapping.get(display) if hasattr(self, "_style_combo_mapping") else None
        if info and info.get("kind") == "user":
            self.global_style = self.selected_style_name or display
        else:
            self.global_style = display
        try:
            self.client = OpenAIClient(self.api_key)
            self._set_status("OpenAI ready.")
            # kick a usage refresh after a successful connect
            self._refresh_billing_snapshot(async_=True)
        except Exception as e:
            messagebox.showerror("OpenAI", str(e))

    # --- captions_todo writer (anchor) ---
    def _write_captions_todo(self, export_root: str, scenes: List[Dict[str,Any]], shot_prompt_map: Dict[str,str]) -> None:
        """
        Write captions_todo.txt to the export root:
          [S1] Title: ...
          Anchor: ...
          Primary prompt: ...
          Notes: ...
          ---
        """
        if not export_root:
            return
        lines: List[str] = []
        title = (self.analysis or {}).get("title", "Untitled")
        lines.append(f"Project: {title}")
        lines.append(f"Exported at: {now_iso()}")
        lines.append("")
        for sc in scenes:
            sid = sc.get("id","")
            if not sid:
                continue
            primary = self._choose_primary_shot(sid)
            primary_prompt = ""
            if primary:
                primary_prompt = (shot_prompt_map.get(primary.id, primary.prompt) or "").strip()
            if not primary_prompt:
                primary_prompt = (sc.get("what_happens") or sc.get("description") or "").strip()
    
            notes = ", ".join([v for v in [sc.get("tone",""), sc.get("time_of_day",""), sc.get("location","")] if v])
            lines.append(f"[{sid}] Title: {sc.get('title','')}")
            lines.append(f"Anchor: {(sc.get('what_happens') or sc.get('description') or '').strip()[:240]}")
            lines.append(f"Primary prompt: {primary_prompt.strip()[:600]}")
            lines.append(f"Notes: {notes}")
            lines.append("---")
        path = os.path.join(export_root, "captions_todo.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # --- captions_map writer (anchor) ---
    def _write_captions_map(self, export_root: str, scenes: List[Dict[str,Any]], shot_prompt_map: Dict[str,str]) -> None:
        """
        Write captions_map.json with one entry per scene:
          {
            "project": {"title":"...", "exported_at":"..."},
            "items": [ { "scene_id":"S1", ... } ]
          }
        """
        if not export_root:
            return
        items = []
        for sc in scenes:
            sid = sc.get("id","")
            if not sid:
                continue
            # collect shot prompts (use map first for any UI edits)
            shot_items = []
            for sh in (self.shots or []):
                if sh.scene_id == sid:
                    prompt = (shot_prompt_map.get(sh.id) or sh.prompt or "").strip()
                    if prompt:
                        shot_items.append({"id": sh.id, "prompt": prompt})
            primary = self._choose_primary_shot(sid)
            if primary and not shot_items:
                prompt = (shot_prompt_map.get(primary.id) or primary.prompt or "").strip()
                if prompt:
                    shot_items.append({"id": primary.id, "prompt": prompt})
            items.append({
                "scene_id": sid,
                "title": sc.get("title",""),
                "text_anchor": (sc.get("what_happens") or sc.get("description") or "").strip()[:240],
                "notes": ", ".join([v for v in [sc.get("tone",""), sc.get("time_of_day",""), sc.get("location","")] if v]),
                "shots": shot_items
            })
        payload = {
            "project": {"title": (self.analysis or {}).get("title","Untitled"), "exported_at": now_iso()},
            "items": items
        }
        path = os.path.join(export_root, "captions_map.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # def _world_json_path(self) -> str:
    #     """Return the active world.json path, or a default in CWD."""
    #     if getattr(self, "world_store_path", ""):
    #         return self.world_store_path
    #     return os.path.join(os.getcwd(), "world.json")

    def _world_json_path(self) -> str:
        """Return the active world.json path, or a default in CWD."""
        if getattr(self, "world_store_path", ""):
            return self.world_store_path
        return os.path.join(os.getcwd(), "world.json")
    
    def _world_json_load_if_any(self) -> dict:
        """Load world.json if present; return {} on error/missing."""
        p = self._world_json_path()
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.world = data if isinstance(data, dict) else self._new_world_template()
                self.world_store_path = p
                if hasattr(self, "world_path_var"):
                    self.world_path_var.set(p)
                return self.world
        except Exception:
            pass
        return {}

    def _world_json_merge_into_analysis(self, world: dict) -> None:
        """
        Merge character/location baselines from world.json into the current state.
        If no analysis is present yet, this will create CharacterProfile/LocationProfile
        entries so you can generate refs immediately.
        """
        if not isinstance(world, dict):
            return
        chars_map = (world.get("characters") or {})
        locs_map  = (world.get("locations") or {})

        # Ensure self.characters / self.locations exist
        if not hasattr(self, "characters"): self.characters = {}
        if not hasattr(self, "locations"):  self.locations = {}

        # Characters
        for name, rec in chars_map.items():
            if not name: 
                continue
            prof = self.characters.get(name)
            if not prof:
                prof = CharacterProfile(name=name, initial_description=rec.get("initial_description",""))
                self.characters[name] = prof
            # Prefer most specific baseline
            if rec.get("sheet_base_prompt"):
                prof.sheet_base_prompt = rec["sheet_base_prompt"]
            if rec.get("refined_description") and not prof.refined_description:
                prof.refined_description = rec["refined_description"]

        # Locations
        for name, rec in locs_map.items():
            if not name:
                continue
            prof = self.locations.get(name)
            if not prof:
                prof = LocationProfile(name=name, description=rec.get("description",""))
                self.locations[name] = prof
            if rec.get("sheet_base_prompt"):
                prof.sheet_base_prompt = rec["sheet_base_prompt"]

        # Refresh related UI panes if they exist
        try:
            self._rebuild_character_panels()
            self._rebuild_location_panels()
        except Exception:
            pass

    def _world_json_update_from_current(self, export_root: str) -> None:
        """
        Update or create world.json from the current session.
        Wrapper that forwards to the existing updater and writes the file next to exports if needed.
        """
        try:
            self._update_world_store_after_export(export_root)
        except Exception as e:
            messagebox.showerror("world.json", f"Failed to update world.json:\n{e}")

    def _open_world_dir(self) -> None:
        """Open the directory containing the current world.json (if any)."""
        p = self._world_json_path()
        try:
            folder = os.path.dirname(p) or os.getcwd()
            if os.name == "nt":
                os.startfile(folder)  # type: ignore[attr-defined]
            else:
                import subprocess, sys
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", folder])
        except Exception as e:
            messagebox.showerror("Open folder", str(e))

    def _format_usd(self, v):
        """Format numbers as USD without raising on odd inputs."""
        try:
            return "${:,.2f}".format(float(v))
        except Exception:
            return "—"

    def _billing_period_dates(self):
        # current calendar month
        import datetime as _dt
        today = _dt.date.today()
        start = today.replace(day=1)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    def _http_get_json(self, url, headers):
        import json as _json
        import urllib.request as _ureq, ssl as _ssl
        req = _ureq.Request(url, headers=headers, method="GET")
        ctx = _ssl.create_default_context()
        with _ureq.urlopen(req, timeout=30, context=ctx) as r:
            data = r.read().decode("utf-8", errors="ignore")
        return _json.loads(data or "{}")

    def _fetch_openai_billing_snapshot(self):
        """
        Returns a dict:
          { 'hard_limit_usd', 'usage_usd', 'credits_avail_usd', 'remaining_usd', 'start_date', 'end_date' }
        Uses official dashboard endpoints. Works for both paid and credit‑grant accounts.
        """
        if not self.api_key:
            raise RuntimeError("No API key; press Connect first.")

        start_date, end_date = self._billing_period_dates()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # Include organization if provided (helps on some org‑scoped keys)
        org = os.environ.get("OPENAI_ORGANIZATION") or os.environ.get("OPENAI_ORG_ID")
        if org:
            headers["OpenAI-Organization"] = org

        # 1) subscription (monthly hard cap)
        sub = {}
        try:
            sub = self._http_get_json("https://api.openai.com/v1/dashboard/billing/subscription", headers)
        except Exception:
            sub = {}
        hard_limit = sub.get("hard_limit_usd")

        # 2) usage (this period)
        usage = {}
        try:
            url = f"https://api.openai.com/v1/dashboard/billing/usage?start_date={start_date}&end_date={end_date}"
            usage = self._http_get_json(url, headers)
        except Exception:
            usage = {}
        # 'total_usage' is in USD cents on this endpoint
        usage_usd = float(usage.get("total_usage") or 0.0) / 100.0

        # 3) credit grants (trial/promo credit pools)
        credits = {}
        credits_avail = None
        try:
            credits = self._http_get_json("https://api.openai.com/v1/dashboard/billing/credit_grants", headers)
            # typical fields: total_granted, total_used, total_available
            if isinstance(credits, dict):
                credits_avail = credits.get("total_available")
        except Exception:
            credits_avail = None

        remaining = None
        if hard_limit is not None:
            remaining = max(0.0, float(hard_limit) - usage_usd)
        elif credits_avail is not None:
            # When no subscription cap is present, show available credits as the "remaining"
            remaining = max(0.0, float(credits_avail))

        return {
            "hard_limit_usd": hard_limit,
            "usage_usd": usage_usd,
            "credits_avail_usd": credits_avail,
            "remaining_usd": remaining,
            "start_date": start_date,
            "end_date": end_date,
        }

    def _refresh_billing_snapshot(self, async_: bool = True):
        """Refresh the billing label now; optional background thread to avoid UI stalls."""
        if async_:
            threading.Thread(target=self._billing_refresh_worker, daemon=True).start()
        else:
            self._billing_refresh_worker()

    def _billing_refresh_worker(self):
        try:
            snap = self._fetch_openai_billing_snapshot()
            text = []
            if snap.get("hard_limit_usd") is not None:
                text.append(f"Cap {self._format_usd(snap['hard_limit_usd'])}")
                text.append(f"Used {self._format_usd(snap['usage_usd'])}")
                if snap.get("remaining_usd") is not None:
                    text.append(f"Remaining {self._format_usd(snap['remaining_usd'])}")
            else:
                text.append(f"Used {self._format_usd(snap['usage_usd'])}")
                if snap.get("credits_avail_usd") is not None:
                    text.append(f"Credits {self._format_usd(snap['credits_avail_usd'])}")
                    if snap.get("remaining_usd") is not None:
                        text.append(f"Remaining {self._format_usd(snap['remaining_usd'])}")

            window = f"{snap.get('start_date','?')} → {snap.get('end_date','?')}"
            final = f"{'  •  '.join(text)}   ({window})"
        except Exception as e:
            final = f"Usage: —  (error: {e})"

        def _apply():
            var = getattr(self, "billing_var", None)
            if isinstance(var, tk.StringVar):
                try:
                    var.set(final)
                except Exception:
                    pass

        try:
            self.root.after(0, _apply)
        except Exception:
            # Safe no-op if we're shutting down
            pass


    def _on_choose_world_store(self):
        # Pick or create a world.json. If the chosen file exists, load it immediately.
        p = filedialog.asksaveasfilename(
            title="Select or create world.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="world.json"
        )
        if not p:
            return
        self.world_store_path = p
        self.world_path_var.set(p)
        self._load_world_store_if_exists()
        self._set_status("World store set: " + p)
        
    def _on_import_world_json(self):
        """
        Import an existing world.json and (if present) a sibling _analysis.json.
        - Applies character/location baselines (creating missing profiles).
        - Re-hydrates self.assets from assets_registry.
        - Attaches reference_images to characters/locations.
        - Refreshes UI panes.
        """
        p = filedialog.askopenfilename(title="Import world.json", filetypes=[("JSON files","*.json")])
        if not p:
            return
    
        try:
            with open(p, "r", encoding="utf-8") as f:
                world = json.load(f)
            if not isinstance(world, dict):
                raise ValueError("Top level of world.json must be a JSON object.")
        except Exception as e:
            messagebox.showerror("World import", f"Failed to read world.json:\n{e}")
            return
    
        # Record path + data and reflect the path in Settings
        self.world_store_path = p
        try:
            self.world_path_var.set(p)
        except Exception:
            pass
        self.world = world
    
        # Story-world facts first
        try:
            self._merge_world_facts(world.get("story_world") or {})
        except Exception:
            pass
    
        # Apply baselines; create missing profiles
        try:
            self._apply_world_baselines_to_state(create_missing=True)
        except TypeError:
            self._apply_world_baselines_to_state()
    
        # NEW: hydrate assets from assets_registry and attach ids to profiles
        try:
            self._hydrate_assets_from_world(attach_to_profiles=True)
        except Exception as e:
            # non-fatal
            print("hydrate assets from world.json failed:", e)
    
        # Try loading sibling _analysis.json (best-effort)
        imported_analysis = False
        try:
            anal = os.path.join(os.path.dirname(p), "_analysis.json")
            if os.path.exists(anal):
                with open(anal, "r", encoding="utf-8") as f:
                    self.analysis = json.load(f)
                # Rebuild tables if present
                try:
                    self._rebuild_scene_table()
                    self._render_precis_and_movements()
                except Exception:
                    pass
                imported_analysis = True
        except Exception:
            pass
    
        # Refresh UI panes
        try:
            self._rebuild_character_panels()
            self._rebuild_location_panels()
        except Exception:
            pass
    
        msg = "Imported world.json." + (" Prior _analysis.json loaded." if imported_analysis else "")
        self._set_status(msg)
        messagebox.showinfo("World import", msg)

  

    def _load_world_store_if_exists(self):
        if not self.world_store_path:
            return
        try:
            if os.path.exists(self.world_store_path):
                with open(self.world_store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.world = data
                else:
                    self.world = self._new_world_template()
            else:
                self.world = self._new_world_template()
        except Exception:
            self.world = self._new_world_template()
        try:
            self.world.setdefault("style_presets", [])
        except Exception:
            self.world["style_presets"] = []
        if "default_style_id" not in self.world:
            self.world["default_style_id"] = ""

    def _refresh_world_from_path(self):
        """
        Re-read the current world.json path (if set) and apply it to the UI.
        Safe to run multiple times (idempotent).
        """
        try:
            self._load_world_store_if_exists()
            # Apply baselines; create missing profiles if needed
            try:
                self._apply_world_baselines_to_state(create_missing=True)
            except TypeError:
                # older signature without the flag
                self._apply_world_baselines_to_state()
            # Refresh panes
            try:
                self._rebuild_character_panels()
                self._rebuild_location_panels()
            except Exception:
                pass
            self._set_status("World.json reloaded and applied.")
        except Exception as e:
            messagebox.showerror("World.json", str(e))

    def _save_world_store_to(self, path: Optional[str] = None):
        p = path or self.world_store_path
        if not p:
            return
        try:
            ensure_dir(os.path.dirname(p))
        except Exception:
            pass
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.world, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to write world store: {e}")

    def _merge_world_facts(self, frag: Dict[str, Any]):
        if not isinstance(frag, dict):
            return
        w = self.world.setdefault("story_world", {})
        for key in ("species","factions","ships","planets"):
            existing = { (x.get("name","") or "").lower(): x for x in (w.get(key) or []) if isinstance(x, dict) }
            for item in (frag.get(key) or []):
                nm = (item.get("name","") or "").lower()
                if nm and nm not in existing:
                    (w.setdefault(key, [])).append(item)
        for key in ("era","aesthetic","tech_level"):
            if not w.get(key) and frag.get(key):
                w[key] = frag[key]
        if frag.get("tags"):
            tags = set((w.get("tags") or []) + list(frag.get("tags") or []))
            w["tags"] = sorted(tags)

    def _apply_world_baselines_to_state(self, create_missing: bool = False):
        """
        Apply baselines from the loaded world store to current Characters & Locations.
        If create_missing=True, also create CharacterProfile/LocationProfile entries for
        any names present in the world file but not yet in the UI.
    
        NEW:
          - Hydrate self.assets from world.assets_registry so ID-based conditioning works
          - Pull visual_cues_from_photos and reference_image_ids back into profiles
          - Light fuzzy match on sanitized names to reduce name drift (e.g., 'Dr. R. Kline' vs 'R Kline')
        """
    
        # 0) Hydrate asset records from the world store (non-destructive)
        try:
            self.world.setdefault("assets_registry", [])
        except Exception:
            pass
        try:
            reg = (self.world or {}).get("assets_registry") or []
            have = {a.id for a in (self.assets or [])}
            for r in reg:
                rid = r.get("id"); pth = r.get("path") or ""
                if not rid or rid in have or not (pth and os.path.isfile(pth)):
                    continue
                self.assets.append(AssetRecord(
                    id=rid,
                    entity_type=r.get("entity_type",""),
                    entity_name=r.get("entity_name",""),
                    view=r.get("view",""),
                    prompt_full="",
                    model=self.image_model,
                    size=self.image_size,
                    file_path=pth,
                    created_at=now_iso(),
                    notes="world asset"
                ))
        except Exception:
            pass
    
        # 1) Build sanitized-name maps to soften naming drift
        state_chars_by_sn = {sanitize_name(k): k for k in (self.characters or {}).keys()}
        state_locs_by_sn  = {sanitize_name(k): k for k in (self.locations or {}).keys()}
    
        # 2) Characters
        raw_chars = (self.world.get("characters") or {})
        if isinstance(raw_chars, list):
            char_map = { (d.get("name","") or "").strip(): d for d in raw_chars if isinstance(d, dict) and d.get("name") }
        else:
            char_map = { (k or "").strip(): (v or {}) for k, v in raw_chars.items() }
    
        for nm_raw, entry in char_map.items():
            nm = (nm_raw or "").strip()
            if not nm:
                continue
            target_name = state_chars_by_sn.get(sanitize_name(nm)) or nm
            c = self.characters.get(target_name)
            if not c and create_missing:
                c = CharacterProfile(name=target_name, initial_description=entry.get("initial_description",""))
                self.characters[target_name] = c
            if not c:
                continue
            baseline = entry.get("sheet_base_prompt") or entry.get("dna") or entry.get("visual_dna") or entry.get("baseline") or ""
            if baseline and not c.sheet_base_prompt:
                c.sheet_base_prompt = baseline
            cues_raw = entry.get("visual_cues_from_photos")
            if cues_raw:
                txt, cues_list = _normalize_visual_cues_value(cues_raw)
                existing_list = getattr(c, "visual_cues_from_photos_list", []) or []
                combined = list(dict.fromkeys(existing_list + cues_list)) if cues_list else existing_list
                if combined:
                    c.visual_cues_from_photos_list = combined
                    c.visual_cues_from_photos = "; ".join(combined)
                elif txt and not c.visual_cues_from_photos:
                    c.visual_cues_from_photos = txt
            else:
                if not getattr(c, "visual_cues_from_photos_list", None):
                    c.visual_cues_from_photos_list = []
            # Lightly hydrate descriptors if missing
            for key in ("initial_description","refined_description","role","goals","conflicts"):
                if entry.get(key) and not getattr(c, key, ""):
                    setattr(c, key, entry[key])
            # NEW: adopt reference ids (union)
            ids = list(dict.fromkeys(entry.get("reference_image_ids") or []))
            for aid in ids:
                if aid not in c.reference_images:
                    c.reference_images.append(aid)
            pri = (entry.get("primary_reference_id") or "").strip()
            if pri:
                c.primary_reference_id = pri
            if not c.primary_reference_id and c.reference_images:
                c.primary_reference_id = c.reference_images[0]
            traits = entry.get("dna_traits")
            if isinstance(traits, dict):
                if not isinstance(c.dna_traits, dict):
                    c.dna_traits = {}
                for key, value in traits.items():
                    if key not in c.dna_traits:
                        c.dna_traits[key] = value
                    elif isinstance(c.dna_traits.get(key), dict) and isinstance(value, dict):
                        for subk, subv in value.items():
                            if subk not in c.dna_traits[key]:
                                c.dna_traits[key][subk] = subv

        # 3) Locations
        raw_locs = (self.world.get("locations") or {})
        if isinstance(raw_locs, list):
            loc_map = { (d.get("name","") or "").strip(): d for d in raw_locs if isinstance(d, dict) and d.get("name") }
        else:
            loc_map = { (k or "").strip(): (v or {}) for k, v in raw_locs.items() }
    
        for nm_raw, entry in loc_map.items():
            nm = (nm_raw or "").strip()
            if not nm:
                continue
            target_name = state_locs_by_sn.get(sanitize_name(nm)) or nm
            l = self.locations.get(target_name)
            if not l and create_missing:
                l = LocationProfile(name=target_name, description=entry.get("description",""))
                self.locations[target_name] = l
            if not l:
                continue
            baseline = entry.get("sheet_base_prompt") or entry.get("dna") or entry.get("visual_dna") or entry.get("baseline") or ""
            if baseline and not l.sheet_base_prompt:
                l.sheet_base_prompt = baseline
            if entry.get("description") and not l.description:
                l.description = entry["description"]
            cues_raw = entry.get("visual_cues_from_photos")
            if cues_raw:
                txt, cues_list = _normalize_visual_cues_value(cues_raw)
                existing_list = getattr(l, "visual_cues_from_photos_list", []) or []
                combined = list(dict.fromkeys(existing_list + cues_list)) if cues_list else existing_list
                if combined:
                    l.visual_cues_from_photos_list = combined
                    l.visual_cues_from_photos = "; ".join(combined)
                elif txt and not l.visual_cues_from_photos:
                    l.visual_cues_from_photos = txt
            else:
                if not getattr(l, "visual_cues_from_photos_list", None):
                    l.visual_cues_from_photos_list = []
            ids = list(dict.fromkeys(entry.get("reference_image_ids") or []))
            for aid in ids:
                if aid not in l.reference_images:
                    l.reference_images.append(aid)
            pri = (entry.get("primary_reference_id") or "").strip()
            if pri:
                l.primary_reference_id = pri
            if not l.primary_reference_id and l.reference_images:
                l.primary_reference_id = l.reference_images[0]
            traits = entry.get("dna_traits")
            if isinstance(traits, dict):
                if not isinstance(l.dna_traits, dict):
                    l.dna_traits = {}
                for key, value in traits.items():
                    if key not in l.dna_traits:
                        l.dna_traits[key] = value
                    elif isinstance(l.dna_traits.get(key), dict) and isinstance(value, dict):
                        for subk, subv in value.items():
                            if subk not in l.dna_traits[key]:
                                l.dna_traits[key][subk] = subv

        default_style_id = (self.world or {}).get("default_style_id") or ""
        preset, preset_name = (None, "")
        if default_style_id:
            self.selected_style_id = default_style_id
            preset, preset_name = self._resolve_selected_style()
            if not preset:
                self.selected_style_id = ""
        if preset and preset_name:
            self.selected_style_name = preset_name
            if preset_name:
                self.global_style = preset_name
        else:
            if not self.selected_style_name:
                self.selected_style_name = self.global_style
        if not self.headless and getattr(self, "style_combo", None):
            self._refresh_style_dropdown()

    def _hydrate_assets_from_world(self, attach_to_profiles: bool = True) -> int:
        """
        Rebuild self.assets from world['assets_registry'] and (optionally) attach
        reference_images to CharacterProfile/LocationProfile based on entity_name.
        Returns the number of new AssetRecord entries added.
        """
        reg = (self.world or {}).get("assets_registry") or []
        if not isinstance(reg, list):
            return 0
    
        existing_by_id = {a.id: a for a in (self.assets or [])}
        existing_by_path = {os.path.abspath(a.file_path): a for a in (self.assets or []) if a.file_path}
        added = 0
    
        for item in reg:
            if not isinstance(item, dict):
                continue
            aid = (item.get("id") or "").strip()
            pth = (item.get("path") or "").strip()
            if not pth:
                continue
            pabs = os.path.abspath(pth)
            if not os.path.isfile(pabs):
                continue
            if (aid and aid in existing_by_id) or (pabs in existing_by_path):
                continue
            ar = AssetRecord(
                id=aid or ("img_" + hashlib.sha1(pabs.encode("utf-8")).hexdigest()[:16]),
                entity_type=item.get("entity_type",""),
                entity_name=item.get("entity_name",""),
                view=item.get("view","") or "",
                prompt_full="",
                model=self.image_model,
                size=self.image_size,
                file_path=pabs,
                created_at=item.get("created_at") or now_iso(),
                notes="imported from world.json"
            )
            (self.assets or []).append(ar)
            existing_by_id[ar.id] = ar
            existing_by_path[pabs] = ar
            added += 1
    
        if attach_to_profiles:
            # Attach by name + type; also respect any explicit reference_images lists stored under entities
            # Characters
            raw_chars = (self.world.get("characters") or {})
            if isinstance(raw_chars, dict):
                for nm, entry in raw_chars.items():
                    c = self.characters.get(nm)
                    if not c:
                        c = CharacterProfile(name=nm, initial_description=entry.get("initial_description",""))
                        self.characters[nm] = c
                    listed = (entry.get("reference_images") or entry.get("reference_image_ids") or [])
                    mined  = [a.id for a in (self.assets or []) if a.entity_type == "character" and a.entity_name == nm]
                    c.reference_images = list(dict.fromkeys((c.reference_images or []) + list(listed) + mined))
    
            # Locations
            raw_locs = (self.world.get("locations") or {})
            if isinstance(raw_locs, dict):
                for nm, entry in raw_locs.items():
                    l = self.locations.get(nm)
                    if not l:
                        l = LocationProfile(name=nm, description=entry.get("description",""))
                        self.locations[nm] = l
                    listed = (entry.get("reference_images") or entry.get("reference_image_ids") or [])
                    mined  = [a.id for a in (self.assets or []) if a.entity_type == "location" and a.entity_name == nm]
                    l.reference_images = list(dict.fromkeys((l.reference_images or []) + list(listed) + mined))
    
        return added

    def _update_world_store_after_export(self, outdir: str):
        """
        Add/refresh persistent memory in world.json:
          - character/location baselines (existing behavior)
          - assets_registry [{id, entity_type, entity_name, view, path}]
          - NEW: persist reference_image_ids and visual_cues_from_photos per entity
        """
        # Ensure we have a destination
        if not self.world_store_path:
            self.world_store_path = os.path.join(outdir, "world.json")
            if hasattr(self, "world_path_var"):
                try:
                    self.world_path_var.set(self.world_store_path)
                except Exception:
                    pass
    
        # --- Characters (existing + NEW fields) ---
        wc = self.world.setdefault("characters", {})
        for name, c in self.characters.items():
            rec = wc.setdefault(name, {})
            if c.initial_description and not rec.get("initial_description"):
                rec["initial_description"] = c.initial_description
            if c.refined_description:
                rec["refined_description"] = c.refined_description
            if c.sheet_base_prompt:
                rec["sheet_base_prompt"] = c.sheet_base_prompt
            # NEW: also persist textual “image DNA” and the concrete ref IDs
            cues_list = getattr(c, "visual_cues_from_photos_list", []) or []
            if cues_list:
                rec["visual_cues_from_photos"] = list(cues_list)
            elif getattr(c, "visual_cues_from_photos", ""):
                rec["visual_cues_from_photos"] = c.visual_cues_from_photos
            if getattr(c, "reference_images", []):
                # IDs only; files live in assets/ and scene refs/
                rec["reference_image_ids"] = list(dict.fromkeys(c.reference_images))
            if getattr(c, "primary_reference_id", ""):
                rec["primary_reference_id"] = c.primary_reference_id
            if getattr(c, "dna_traits", None):
                if isinstance(c.dna_traits, dict) and c.dna_traits:
                    rec["dna_traits"] = c.dna_traits

        # --- Locations (existing + NEW fields) ---
        wl = self.world.setdefault("locations", {})
        for name, l in self.locations.items():
            rec = wl.setdefault(name, {})
            if l.description and not rec.get("description"):
                rec["description"] = l.description
            if l.sheet_base_prompt:
                rec["sheet_base_prompt"] = l.sheet_base_prompt
            # NEW: persist textual “image DNA” and ref IDs
            cues_list = getattr(l, "visual_cues_from_photos_list", []) or []
            if cues_list:
                rec["visual_cues_from_photos"] = list(cues_list)
            elif getattr(l, "visual_cues_from_photos", ""):
                rec["visual_cues_from_photos"] = l.visual_cues_from_photos
            if getattr(l, "reference_images", []):
                rec["reference_image_ids"] = list(dict.fromkeys(l.reference_images))
            if getattr(l, "primary_reference_id", ""):
                rec["primary_reference_id"] = l.primary_reference_id
            if getattr(l, "dna_traits", None):
                if isinstance(l.dna_traits, dict) and l.dna_traits:
                    rec["dna_traits"] = l.dna_traits

        # --- Assets registry (existing behavior preserved) ---
        # Use what's already in self.assets and also scan ./assets/**/*.png|jpg|jpeg|webp
        reg = self.world.setdefault("assets_registry", [])
        # De‑dupe by absolute path
        existing_by_path = {os.path.abspath(item.get("path","")): item for item in reg if isinstance(item, dict)}
    
        def _safe_add(aid, etype, ename, view, path_):
            abs_p = os.path.abspath(path_ or "")
            if not abs_p or not os.path.isfile(abs_p):
                return
            if abs_p in existing_by_path:
                # keep first record; we don't delete or overwrite
                return
            rec = {
                "id": aid,
                "entity_type": etype,
                "entity_name": ename,
                "view": view or "",
                "path": abs_p
            }
            reg.append(rec)
            existing_by_path[abs_p] = rec
    
        # from in‑memory asset records (fast path)
        try:
            for a in (getattr(self, "assets", []) or []):
                _safe_add(a.id, a.entity_type, a.entity_name, getattr(a, "view", ""), a.file_path)
        except Exception:
            pass
    
        # best‑effort disk scan (covers assets saved in earlier runs)
        try:
            for base, etype in [(os.path.join("assets","characters"), "character"),
                                (os.path.join("assets","locations"),  "location")]:
                if not os.path.isdir(base):
                    continue
                for name in os.listdir(base):
                    d = os.path.join(base, name)
                    if not os.path.isdir(d):
                        continue
                    for fn in os.listdir(d):
                        if not fn.lower().endswith((".png",".jpg",".jpeg",".webp")):
                            continue
                        fpath = os.path.join(d, fn)
                        # hash of absolute path to make a stable id
                        aid = "img_" + hashlib.sha1(os.path.abspath(fpath).encode("utf-8")).hexdigest()[:16]
                        # try to pull a view from the filename convention
                        view = ""
                        m = re.search(r"_(front|profile_left|profile_right|back|three_quarter_left|three_quarter_right|full_body_tpose|establishing|alt_angle|detail)_", fn, re.I)
                        if m: view = m.group(1).lower()
                        _safe_add(aid, etype, name, view, fpath)
        except Exception:
            pass
    
        # Touch updated_at & save
        try:
            self.world["updated_at"] = now_iso()
        except Exception:
            pass

        self._save_world_store_to(self.world_store_path)

    def _analysis_context_snippet(self) -> str:
        parts: List[str] = []
        try:
            if isinstance(self.analysis, dict):
                title = self.analysis.get("title")
                if title:
                    parts.append(str(title))
                summary = (
                    self.analysis.get("story_summary")
                    or self.analysis.get("story_precis")
                    or self.analysis.get("logline")
                )
                if summary:
                    parts.append(str(summary))
        except Exception:
            pass
        text = " \n".join(parts).strip()
        return text[:1200]

    def _merge_dna_maps(self, target: Optional[Dict[str, Any]], incoming: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(target, dict):
            target = {}
        if not isinstance(incoming, dict):
            return target
        for key, value in incoming.items():
            if key not in target:
                target[key] = value
            elif isinstance(target.get(key), dict) and isinstance(value, dict):
                target[key] = self._merge_dna_maps(target.get(key), value)
        return target

    def _llm_extract_relevant_traits(self, img_path: str, analysis_ctx: str) -> Dict[str, Any]:
        try:
            client = getattr(self, "client", None) or getattr(self, "llm", None)
            if client is None:
                return {}
            handler = getattr(client, "vision_describe", None)
            if not callable(handler):
                handler = getattr(client, "describe_image", None)
            if not callable(handler):
                return {}
            prompt = (
                "Extract story-relevant visual traits from this reference image. Return JSON with keys: "
                "{appearance:{age_band,hair_color,eye_color,face_shape,notable_features}, "
                "wardrobe:{style,colors,items}, environment:{type,materials,key_props,era}, "
                "quality:{lighting_notes,contrast_notes}}. Focus on traits that align with the story context: "
                + (analysis_ctx[:800] if analysis_ctx else "")
            )
            result = handler(image_path=img_path, prompt=prompt)
            if isinstance(result, dict):
                return result
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except Exception:
                    return {}
        except Exception:
            return {}
        return {}

    def _attach_refs_to_profile(self, profile, paths: List[str]) -> List[str]:
        if not profile or not paths:
            return []

        kind = None
        name = getattr(profile, "name", "")
        for nm, obj in (self.characters or {}).items():
            if obj is profile:
                kind = "character"
                name = nm
                break
        if kind is None:
            for nm, obj in (self.locations or {}).items():
                if obj is profile:
                    kind = "location"
                    name = nm
                    break
        if kind is None:
            return []

        world_path = getattr(self, "world_store_path", "") or ""
        bucket = self.world.setdefault("characters" if kind == "character" else "locations", {})
        entry = bucket.setdefault(name, {})
        entry.setdefault("reference_image_ids", [])
        entry.setdefault("visual_cues_from_photos", [])
        entry.setdefault("dna_traits", {})

        profile.reference_images = list(dict.fromkeys(profile.reference_images or []))
        if not isinstance(getattr(profile, "visual_cues_from_photos_list", None), list):
            profile.visual_cues_from_photos_list = []
        if not isinstance(getattr(profile, "dna_traits", None), dict):
            profile.dna_traits = {}

        added_ids: List[str] = []
        analysis_ctx = self._analysis_context_snippet()

        for src in paths:
            if not (isinstance(src, str) and os.path.isfile(src)):
                continue
            ext = os.path.splitext(src)[1].lower()
            if ext and ext not in VALID_IMAGE_EXTS:
                continue
            dst = _copy_into_assets(world_path, src)
            rec = _register_asset(self.world, dst, entity_type=kind, entity_name=name)
            rid = rec.get("id")
            if not rid:
                continue
            if rid not in profile.reference_images:
                profile.reference_images.append(rid)
                added_ids.append(rid)
            if not profile.primary_reference_id:
                profile.primary_reference_id = rid

            palette = rec.get("palette") or []
            luma = float(rec.get("avg_luma") or 0.5)
            cues = _image_dna_phrases(palette, luma)
            existing_cues = profile.visual_cues_from_photos_list or []
            for cue in cues:
                if cue not in existing_cues:
                    existing_cues.append(cue)
            profile.visual_cues_from_photos_list = existing_cues
            profile.visual_cues_from_photos = "; ".join(existing_cues)

            traits = self._llm_extract_relevant_traits(dst, analysis_ctx)
            if traits:
                profile.dna_traits = self._merge_dna_maps(profile.dna_traits, traits)

        entry["reference_image_ids"] = list(dict.fromkeys(profile.reference_images))
        entry["visual_cues_from_photos"] = list(profile.visual_cues_from_photos_list or [])
        if profile.primary_reference_id:
            entry["primary_reference_id"] = profile.primary_reference_id
        if profile.dna_traits:
            entry["dna_traits"] = profile.dna_traits

        try:
            if world_path:
                self._save_world_store_to(world_path)
        except Exception:
            pass

        return added_ids

    def _collect_reference_identity(self, profiles: List[Any], extra_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        seen: List[Any] = []
        for p in profiles or []:
            if p and p not in seen:
                seen.append(p)
        reg_items = {}
        for item in (self.world or {}).get("assets_registry", []) or []:
            if isinstance(item, dict) and item.get("id"):
                reg_items[item["id"]] = item

        ref_ids: List[str] = []
        palette: List[str] = []
        primary_map: Dict[str, str] = {}
        traits_map: Dict[str, Any] = {}

        for p in seen:
            nm = getattr(p, "name", "") or getattr(p, "description", "") or "profile"
            pri = getattr(p, "primary_reference_id", "") or ""
            if pri:
                primary_map[nm] = pri
            for rid in getattr(p, "reference_images", []) or []:
                if rid and rid not in ref_ids:
                    ref_ids.append(rid)
                rec = reg_items.get(rid)
                if rec:
                    for color in (rec.get("palette") or [])[:3]:
                        if color and color not in palette:
                            palette.append(color)
            if getattr(p, "dna_traits", None):
                traits_map[nm] = p.dna_traits

        for rid in extra_ids or []:
            if rid and rid not in ref_ids:
                ref_ids.append(rid)
            rec = reg_items.get(rid)
            if rec:
                for color in (rec.get("palette") or [])[:3]:
                    if color and color not in palette:
                        palette.append(color)

        bits: List[str] = []
        if ref_ids:
            bits.append("Identity-locked to reference images: " + ", ".join(ref_ids))
        if palette:
            bits.append("Honor color DNA: " + ", ".join(palette[:4]))
        trait_lines: List[str] = []
        for nm, trait in traits_map.items():
            if not isinstance(trait, dict):
                continue
            chunk: List[str] = []
            appearance = trait.get("appearance") if isinstance(trait.get("appearance"), dict) else {}
            if appearance:
                hair = appearance.get("hair_color")
                if hair:
                    chunk.append("hair " + str(hair))
                eyes = appearance.get("eye_color")
                if eyes:
                    chunk.append("eyes " + str(eyes))
                face = appearance.get("face_shape")
                if face:
                    chunk.append("face " + str(face))
                features = appearance.get("notable_features")
                if features:
                    if isinstance(features, list):
                        features = ", ".join(str(x) for x in features[:2] if x)
                    chunk.append("features " + str(features))
            wardrobe = trait.get("wardrobe") if isinstance(trait.get("wardrobe"), dict) else {}
            if wardrobe:
                items = wardrobe.get("items")
                if items:
                    if isinstance(items, list):
                        items = ", ".join(str(x) for x in items[:2] if x)
                    chunk.append("wardrobe " + str(items))
            if chunk:
                trait_lines.append(f"{nm}: " + "; ".join(chunk))
        if trait_lines:
            bits.append("Keep identity cues: " + " | ".join(trait_lines[:3]))

        return {
            "bits": bits,
            "ref_ids": ref_ids,
            "primary_map": primary_map,
            "traits": traits_map,
            "palette": palette,
        }


    def _create_style_from_images(self, name: str, paths: List[str]) -> Optional[Dict[str, Any]]:
        cleaned_name = (name or "").strip()
        if not cleaned_name:
            cleaned_name = "Untitled Style"
        if not paths or len(paths) < 5:
            print("[style] need ≥5 images to build a preset")
            return None

        world_path = getattr(self, "world_store_path", "") or ""
        asset_ids: List[str] = []
        palette_samples: List[List[str]] = []
        contrast_vals: List[float] = []
        color_vals: List[float] = []
        edge_vals: List[float] = []
        sample_paths: List[str] = []

        for src in paths:
            if not isinstance(src, str):
                continue
            ext = os.path.splitext(src)[1].lower()
            if ext and ext not in VALID_IMAGE_EXTS:
                continue
            if not os.path.isfile(src):
                continue
            try:
                dst = _copy_into_assets(world_path, src)
            except Exception:
                dst = src
            rec = _register_asset(self.world, dst)
            rid = rec.get("id")
            if not rid:
                continue
            if rid not in asset_ids:
                asset_ids.append(rid)
            sample_paths.append(dst)
            try:
                with Image.open(dst) as im:
                    im = im.convert("RGB")
                    palette_samples.append(_palette_from_image(im, 6))
                    contrast_vals.append(_contrast_norm(im))
                    color_vals.append(_colorfulness(im))
                    edge_vals.append(_edge_density(im))
            except Exception:
                continue

        if len(asset_ids) < 5:
            print("[style] <5 valid images after filtering")
            return None

        flat_palette: List[str] = []
        for pal in palette_samples:
            for color in pal:
                if color and color not in flat_palette:
                    flat_palette.append(color)
        flat_palette = flat_palette[:6]

        contrast = sum(contrast_vals) / len(contrast_vals) if contrast_vals else 0.5
        colorfulness = sum(color_vals) / len(color_vals) if color_vals else 0.5
        edge = sum(edge_vals) / len(edge_vals) if edge_vals else 0.4
        tone = _tone_bias(flat_palette)
        grain = _grain_hint(contrast, edge)
        base_prompt = _summarize_style_prompt(flat_palette, contrast, colorfulness, edge, tone, grain)
        analysis_ctx = ""
        try:
            analysis_ctx = self._analysis_context_snippet()
        except Exception:
            analysis_ctx = ""
        style_desc = _llm_style_summary(self, sample_paths, analysis_ctx, base_prompt)

        preset_id = f"style_{int(time.time())}_{hashlib.md5(cleaned_name.encode('utf-8')).hexdigest()[:6]}"
        preset = {
            "id": preset_id,
            "name": cleaned_name,
            "sample_asset_ids": asset_ids,
            "palette": flat_palette,
            "contrast": round(float(contrast), 3),
            "colorfulness": round(float(colorfulness), 3),
            "edge_density": round(float(edge), 3),
            "tone_bias": tone,
            "grain_hint": grain,
            "style_prompt": style_desc,
        }

        try:
            styles = (self.world or {}).setdefault("style_presets", [])
        except Exception:
            self.world["style_presets"] = []
            styles = self.world["style_presets"]
        styles.append(preset)

        try:
            if self.world_store_path:
                self._save_world_store_to(self.world_store_path)
        except Exception:
            pass

        print(f"[style] created preset '{cleaned_name}' with {len(asset_ids)} samples")
        return preset


    def _resolve_selected_style(self) -> Tuple[Optional[Dict[str, Any]], str]:
        sid = (getattr(self, "selected_style_id", "") or "").strip()
        styles = []
        try:
            styles = (self.world or {}).get("style_presets") or []
        except Exception:
            styles = []
        if sid:
            for preset in styles:
                if isinstance(preset, dict) and (preset.get("id") or "").strip() == sid:
                    name = (preset.get("name") or preset.get("id") or "").strip()
                    return preset, name
        return None, ""


    def _style_prompt_bits(self) -> List[str]:
        bits: List[str] = []
        preset, _ = self._resolve_selected_style()
        if not preset:
            return bits
        desc = (preset.get("style_prompt") or "").strip()
        if desc:
            bits.append("Style: " + desc)
        palette = [c for c in (preset.get("palette") or []) if c][:4]
        if palette:
            bits.append("Style palette: " + ", ".join(palette))
        return bits


    def _current_style_snapshot(self) -> Dict[str, Any]:
        preset, preset_name = self._resolve_selected_style()
        if preset:
            return {
                "preset": preset,
                "id": preset.get("id", ""),
                "name": preset_name or preset.get("name", ""),
            }
        fallback_name = getattr(self, "selected_style_name", "") or getattr(self, "global_style", "")
        return {"preset": None, "id": "", "name": fallback_name}


    def _load_user_styles(self) -> None:
        styles_list = []
        if isinstance(self.world, dict):
            raw = self.world.get("style_presets")
            if isinstance(raw, list):
                styles_list = raw
        self._user_styles = styles_list

        payload = _read_json_safely(_styles_store_path())
        if isinstance(payload, dict):
            source = payload.get("styles") or payload.get("style_presets") or []
        elif isinstance(payload, list):
            source = payload
        else:
            source = []

        existing_by_id: Dict[str, Dict[str, Any]] = {}
        for entry in self._user_styles:
            if isinstance(entry, dict):
                sid = (entry.get("id") or "").strip()
                if sid:
                    existing_by_id[sid] = entry

        for entry in source:
            if not isinstance(entry, dict):
                continue
            sid = (entry.get("id") or "").strip()
            if sid and sid in existing_by_id:
                try:
                    existing_by_id[sid].update(entry)
                except Exception:
                    pass
            else:
                self._user_styles.append(entry)
                if sid:
                    existing_by_id[sid] = entry

        if isinstance(self.world, dict):
            self.world["style_presets"] = self._user_styles


    def _save_user_styles(self) -> None:
        try:
            styles = [s for s in getattr(self, "_user_styles", []) if isinstance(s, dict)]
            payload = {"styles": styles}
            path = _styles_store_path()
            ensure_dir(os.path.dirname(path) or ".")
            _write_json_atomic(path, payload)
        except Exception as exc:
            try:
                print(f"[style] warning: failed to save user styles: {exc}")
            except Exception:
                pass


    def _merge_styles_for_dropdown(self) -> None:
        try:
            if not isinstance(self.world, dict):
                return
            styles = getattr(self, "_user_styles", None)
            if styles is None:
                styles = []
            current = self.world.setdefault("style_presets", [])
            if current is not styles:
                self.world["style_presets"] = styles
            self._refresh_style_dropdown(preserve_selection=False)
        except Exception:
            try:
                self._refresh_style_dropdown(preserve_selection=False)
            except Exception:
                pass


    def _load_thumb(self, path: str, max_side: int = 160):
        """
        Load a thumbnail ImageTk.PhotoImage, caching in self._thumb_cache.
        Returns (imgtk, (w, h)) or (None, (0, 0)) on failure.
        """
        try:
            if not hasattr(self, "_thumb_cache"):
                self._thumb_cache = {}
            key = (path, max_side)
            cached = self._thumb_cache.get(key)
            if cached:
                return cached
            with Image.open(path) as im:
                im = im.convert("RGBA")
                w, h = im.size
                scale = max(1.0, max(w, h) / float(max_side))
                tw = int(max(1, round(w / scale)))
                th = int(max(1, round(h / scale)))
                im = im.resize((tw, th), Image.LANCZOS)
                imtk = ImageTk.PhotoImage(im)
                cached = (imtk, (imtk.width(), imtk.height()))
                self._thumb_cache[key] = cached
                return cached
        except Exception:
            return None, (0, 0)

    def _asset_path_by_id(self, asset_id: str) -> str:
        """Resolve asset file path from world.assets_registry; return '' if not found."""
        try:
            reg = (self.world or {}).get("assets_registry") or []
            for item in reg:
                if isinstance(item, dict) and item.get("id") == asset_id:
                    pth = item.get("path")
                    return pth if isinstance(pth, str) else ""
        except Exception:
            return ""
        return ""


    def _find_profile_by_name(self, name: str, kind_hint: Optional[str] = None) -> Tuple[Optional[str], Optional[Any]]:
        target = (name or "").strip()
        if not target:
            return None, None

        lowered = target.lower()
        sanitized = sanitize_name(target)

        def _match(pool: Dict[str, Any]) -> Optional[str]:
            for nm in pool.keys():
                if (nm or "").lower() == lowered:
                    return nm
            for nm in pool.keys():
                if sanitize_name(nm) == sanitized:
                    return nm
            return None

        order: List[str]
        if kind_hint == "location":
            order = ["location", "character"]
        elif kind_hint == "character":
            order = ["character", "location"]
        else:
            order = ["character", "location"]

        for kind in order:
            pool = self.characters if kind == "character" else self.locations
            matched = _match(pool)
            if matched:
                return kind, pool.get(matched)

        return None, None

    def _analysis_context_snippet(self) -> str:
        parts: List[str] = []
        try:
            if isinstance(self.analysis, dict):
                title = self.analysis.get("title")
                if title:
                    parts.append(str(title))
                summary = (
                    self.analysis.get("story_summary")
                    or self.analysis.get("story_precis")
                    or self.analysis.get("logline")
                )
                if summary:
                    parts.append(str(summary))
        except Exception:
            pass
        text = " \n".join(parts).strip()
        return text[:1200]

    def _merge_dna_maps(self, target: Optional[Dict[str, Any]], incoming: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(target, dict):
            target = {}
        if not isinstance(incoming, dict):
            return target
        for key, value in incoming.items():
            if key not in target:
                target[key] = value
            elif isinstance(target.get(key), dict) and isinstance(value, dict):
                target[key] = self._merge_dna_maps(target.get(key), value)
        return target

    def _llm_extract_relevant_traits(self, img_path: str, analysis_ctx: str) -> Dict[str, Any]:
        try:
            client = getattr(self, "client", None) or getattr(self, "llm", None)
            if client is None:
                return {}
            handler = getattr(client, "vision_describe", None)
            if not callable(handler):
                handler = getattr(client, "describe_image", None)
            if not callable(handler):
                return {}
            prompt = (
                "Extract story-relevant visual traits from this reference image. Return JSON with keys: "
                "{appearance:{age_band,hair_color,eye_color,face_shape,notable_features}, "
                "wardrobe:{style,colors,items}, environment:{type,materials,key_props,era}, "
                "quality:{lighting_notes,contrast_notes}}. Focus on traits that align with the story context: "
                + (analysis_ctx[:800] if analysis_ctx else "")
            )
            result = handler(image_path=img_path, prompt=prompt)
            if isinstance(result, dict):
                return result
            if isinstance(result, str):
                try:
                    return json.loads(result)
                except Exception:
                    return {}
        except Exception:
            return {}
        return {}

    def _attach_refs_to_profile(self, profile, paths: List[str]) -> List[str]:
        if not profile or not paths:
            return []

        kind = None
        name = getattr(profile, "name", "")
        for nm, obj in (self.characters or {}).items():
            if obj is profile:
                kind = "character"
                name = nm
                break
        if kind is None:
            for nm, obj in (self.locations or {}).items():
                if obj is profile:
                    kind = "location"
                    name = nm
                    break
        if kind is None:
            return []

        world_path = getattr(self, "world_store_path", "") or ""
        bucket = self.world.setdefault("characters" if kind == "character" else "locations", {})
        entry = bucket.setdefault(name, {})
        entry.setdefault("reference_image_ids", [])
        entry.setdefault("visual_cues_from_photos", [])
        entry.setdefault("dna_traits", {})

        profile.reference_images = list(dict.fromkeys(profile.reference_images or []))
        if not isinstance(getattr(profile, "visual_cues_from_photos_list", None), list):
            profile.visual_cues_from_photos_list = []
        if not isinstance(getattr(profile, "dna_traits", None), dict):
            profile.dna_traits = {}

        added_ids: List[str] = []
        analysis_ctx = self._analysis_context_snippet()

        for src in paths:
            if not (isinstance(src, str) and os.path.isfile(src)):
                continue
            ext = os.path.splitext(src)[1].lower()
            if ext and ext not in VALID_IMAGE_EXTS:
                continue
            dst = _copy_into_assets(world_path, src)
            rec = _register_asset(self.world, dst, entity_type=kind, entity_name=name)
            rid = rec.get("id")
            if not rid:
                continue
            if rid not in profile.reference_images:
                profile.reference_images.append(rid)
                added_ids.append(rid)
            if not profile.primary_reference_id:
                profile.primary_reference_id = rid

            palette = rec.get("palette") or []
            luma = float(rec.get("avg_luma") or 0.5)
            cues = _image_dna_phrases(palette, luma)
            existing_cues = profile.visual_cues_from_photos_list or []
            for cue in cues:
                if cue not in existing_cues:
                    existing_cues.append(cue)
            profile.visual_cues_from_photos_list = existing_cues
            profile.visual_cues_from_photos = "; ".join(existing_cues)

            traits = self._llm_extract_relevant_traits(dst, analysis_ctx)
            if traits:
                profile.dna_traits = self._merge_dna_maps(profile.dna_traits, traits)

        entry["reference_image_ids"] = list(dict.fromkeys(profile.reference_images))
        entry["visual_cues_from_photos"] = list(profile.visual_cues_from_photos_list or [])
        if profile.primary_reference_id:
            entry["primary_reference_id"] = profile.primary_reference_id
        if profile.dna_traits:
            entry["dna_traits"] = profile.dna_traits

        try:
            if world_path:
                self._save_world_store_to(world_path)
        except Exception:
            pass

        return added_ids

    def _collect_reference_identity(self, profiles: List[Any], extra_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        seen: List[Any] = []
        for p in profiles or []:
            if p and p not in seen:
                seen.append(p)
        reg_items = {}
        for item in (self.world or {}).get("assets_registry", []) or []:
            if isinstance(item, dict) and item.get("id"):
                reg_items[item["id"]] = item

        ref_ids: List[str] = []
        palette: List[str] = []
        primary_map: Dict[str, str] = {}
        traits_map: Dict[str, Any] = {}

        for p in seen:
            nm = getattr(p, "name", "") or getattr(p, "description", "") or "profile"
            pri = getattr(p, "primary_reference_id", "") or ""
            if pri:
                primary_map[nm] = pri
            for rid in getattr(p, "reference_images", []) or []:
                if rid and rid not in ref_ids:
                    ref_ids.append(rid)
                rec = reg_items.get(rid)
                if rec:
                    for color in (rec.get("palette") or [])[:3]:
                        if color and color not in palette:
                            palette.append(color)
            if getattr(p, "dna_traits", None):
                traits_map[nm] = p.dna_traits

        for rid in extra_ids or []:
            if rid and rid not in ref_ids:
                ref_ids.append(rid)
            rec = reg_items.get(rid)
            if rec:
                for color in (rec.get("palette") or [])[:3]:
                    if color and color not in palette:
                        palette.append(color)

        bits: List[str] = []
        if ref_ids:
            bits.append("Identity-locked to reference images: " + ", ".join(ref_ids))
        if palette:
            bits.append("Honor color DNA: " + ", ".join(palette[:4]))
        trait_lines: List[str] = []
        for nm, trait in traits_map.items():
            if not isinstance(trait, dict):
                continue
            chunk: List[str] = []
            appearance = trait.get("appearance") if isinstance(trait.get("appearance"), dict) else {}
            if appearance:
                hair = appearance.get("hair_color")
                if hair:
                    chunk.append("hair " + str(hair))
                eyes = appearance.get("eye_color")
                if eyes:
                    chunk.append("eyes " + str(eyes))
                face = appearance.get("face_shape")
                if face:
                    chunk.append("face " + str(face))
                features = appearance.get("notable_features")
                if features:
                    if isinstance(features, list):
                        features = ", ".join(str(x) for x in features[:2] if x)
                    chunk.append("features " + str(features))
            wardrobe = trait.get("wardrobe") if isinstance(trait.get("wardrobe"), dict) else {}
            if wardrobe:
                items = wardrobe.get("items")
                if items:
                    if isinstance(items, list):
                        items = ", ".join(str(x) for x in items[:2] if x)
                    chunk.append("wardrobe " + str(items))
            if chunk:
                trait_lines.append(f"{nm}: " + "; ".join(chunk))
        if trait_lines:
            bits.append("Keep identity cues: " + " | ".join(trait_lines[:3]))

        return {
            "bits": bits,
            "ref_ids": ref_ids,
            "primary_map": primary_map,
            "traits": traits_map,
            "palette": palette,
        }


    def _create_style_from_images(self, name: str, paths: List[str]) -> Optional[Dict[str, Any]]:
        cleaned_name = (name or "").strip()
        if not cleaned_name:
            cleaned_name = "Untitled Style"
        if not paths or len(paths) < 5:
            print("[style] need ≥5 images to build a preset")
            return None

        world_path = getattr(self, "world_store_path", "") or ""
        asset_ids: List[str] = []
        palette_samples: List[List[str]] = []
        contrast_vals: List[float] = []
        color_vals: List[float] = []
        edge_vals: List[float] = []
        sample_paths: List[str] = []

        for src in paths:
            if not isinstance(src, str):
                continue
            ext = os.path.splitext(src)[1].lower()
            if ext and ext not in VALID_IMAGE_EXTS:
                continue
            if not os.path.isfile(src):
                continue
            try:
                dst = _copy_into_assets(world_path, src)
            except Exception:
                dst = src
            rec = _register_asset(self.world, dst)
            rid = rec.get("id")
            if not rid:
                continue
            if rid not in asset_ids:
                asset_ids.append(rid)
            sample_paths.append(dst)
            try:
                with Image.open(dst) as im:
                    im = im.convert("RGB")
                    palette_samples.append(_palette_from_image(im, 6))
                    contrast_vals.append(_contrast_norm(im))
                    color_vals.append(_colorfulness(im))
                    edge_vals.append(_edge_density(im))
            except Exception:
                continue

        if len(asset_ids) < 5:
            print("[style] <5 valid images after filtering")
            return None

        flat_palette: List[str] = []
        for pal in palette_samples:
            for color in pal:
                if color and color not in flat_palette:
                    flat_palette.append(color)
        flat_palette = flat_palette[:6]

        contrast = sum(contrast_vals) / len(contrast_vals) if contrast_vals else 0.5
        colorfulness = sum(color_vals) / len(color_vals) if color_vals else 0.5
        edge = sum(edge_vals) / len(edge_vals) if edge_vals else 0.4
        tone = _tone_bias(flat_palette)
        grain = _grain_hint(contrast, edge)
        base_prompt = _summarize_style_prompt(flat_palette, contrast, colorfulness, edge, tone, grain)
        analysis_ctx = ""
        try:
            analysis_ctx = self._analysis_context_snippet()
        except Exception:
            analysis_ctx = ""
        style_desc = _llm_style_summary(self, sample_paths, analysis_ctx, base_prompt)

        preset_id = f"style_{int(time.time())}_{hashlib.md5(cleaned_name.encode('utf-8')).hexdigest()[:6]}"
        preset = {
            "id": preset_id,
            "name": cleaned_name,
            "sample_asset_ids": asset_ids,
            "palette": flat_palette,
            "contrast": round(float(contrast), 3),
            "colorfulness": round(float(colorfulness), 3),
            "edge_density": round(float(edge), 3),
            "tone_bias": tone,
            "grain_hint": grain,
            "style_prompt": style_desc,
        }

        try:
            styles = (self.world or {}).setdefault("style_presets", [])
        except Exception:
            self.world["style_presets"] = []
            styles = self.world["style_presets"]
        styles.append(preset)

        try:
            if self.world_store_path:
                self._save_world_store_to(self.world_store_path)
        except Exception:
            pass

        print(f"[style] created preset '{cleaned_name}' with {len(asset_ids)} samples")
        return preset


    def _resolve_selected_style(self) -> Tuple[Optional[Dict[str, Any]], str]:
        sid = (getattr(self, "selected_style_id", "") or "").strip()
        styles = []
        try:
            prompt_for_api = self._augment_prompt_for_render(prompt)
            if refs:
                return self.client.generate_images_b64_with_refs(
                    model=self.image_model,
                    prompt=prompt_for_api,
                    size=target_size,
                    ref_data_uris=refs,
                    n=n
                )
            return self.client.generate_images_b64(
                model=self.image_model,
                prompt=prompt_for_api,
                size=target_size,
                n=n
            )
        except Exception as e:
            raise RuntimeError(f"Image generation failed ({target_size}): {e}") from e

        existing_by_id: Dict[str, Dict[str, Any]] = {}
        for entry in self._user_styles:
            if isinstance(entry, dict):
                sid = (entry.get("id") or "").strip()
                if sid:
                    existing_by_id[sid] = entry

    def _current_exposure_settings(self) -> tuple[float, bool, float]:
        try:
            bias = float(getattr(self, "exposure_bias", EXPOSURE_BIAS))
        except Exception:
            bias = float(EXPOSURE_BIAS)
        try:
            post = bool(getattr(self, "post_tonemap", EXPOSURE_POST_TONEMAP))
        except Exception:
            post = bool(EXPOSURE_POST_TONEMAP)
        try:
            emiss = float(getattr(self, "emissive_level", EMISSIVE_LEVEL))
        except Exception:
            emiss = float(EMISSIVE_LEVEL)
        return bias, post, emiss

    def _augment_prompt_for_render(self, prompt: str) -> str:
        bias, _, emiss = self._current_exposure_settings()
        try:
            base = str(prompt or "")
        except Exception:
            base = ""
        cleaned_lines: list[str] = []
        for line in base.splitlines():
            strip = line.strip().lower()
            if strip.startswith("exposure control:") or strip.startswith("emissive lighting:"):
                continue
            cleaned_lines.append(line)
        base_txt = "\n".join(cleaned_lines).strip()
        parts: list[str] = [base_txt] if base_txt else []
        parts.append("Exposure control: " + exposure_language(bias))
        if abs(emiss) >= 0.15:
            parts.append("Emissive lighting: " + emissive_language(emiss))
        augmented = "\n".join(parts)
        try:
            print(f"[prompt] exposure={bias:+.2f} emissive={emiss:+.2f}")
        except Exception:
            pass
        return augmented

    def _process_generated_image(
        self, raw: bytes, ext: str | None = None, *, need_image: bool = False
    ) -> tuple[bytes, Optional["Image.Image"]]:
        bias, post, _ = self._current_exposure_settings()
        processed = raw
        img_obj: Optional["Image.Image"] = None
        log_msg = None
        if post and abs(bias) >= 0.05:
            try:
                buf = io.BytesIO(raw)
                with Image.open(buf) as im:
                    im.load()
                    tonemapped = apply_exposure_tonemap(im, bias)
                    out_buf = io.BytesIO()
                    fmt = None
                    if ext:
                        fmt = {
                            ".png": "PNG",
                            ".jpg": "JPEG",
                            ".jpeg": "JPEG",
                            ".webp": "WEBP",
                        }.get(ext.lower())
                    if not fmt:
                        fmt = tonemapped.format or "PNG"
                    save_img = tonemapped
                    if fmt == "JPEG" and tonemapped.mode == "RGBA":
                        save_img = tonemapped.convert("RGB")
                    save_img.save(out_buf, format=fmt)
                    processed = out_buf.getvalue()
                    img_obj = tonemapped.copy()
                    log_msg = f"[exposure] post tone-map applied (bias={bias:+.2f})"
            except Exception as e:
                log_msg = f"[exposure] tone-map skipped: {e}"
                img_obj = None
                processed = raw
        if need_image and img_obj is None:
            try:
                buf = io.BytesIO(processed)
                with Image.open(buf) as im:
                    im.load()
                    img_obj = im.copy()
            except Exception:
                img_obj = None
        if log_msg:
            try:
                print(log_msg)
            except Exception:
                pass
        return processed, img_obj

    def _process_image_batch(self, imgs: list[bytes], ext: str | None = None) -> list[bytes]:
        processed: list[bytes] = []
        for b in imgs or []:
            pb, _ = self._process_generated_image(b, ext=ext, need_image=False)
            processed.append(pb)
        return processed


    def _build_ui(self):
        """
        Load a thumbnail ImageTk.PhotoImage, caching in self._thumb_cache.
        Returns (imgtk, (w, h)) or (None, (0, 0)) on failure.
        """
        try:
            if not hasattr(self, "_thumb_cache"):
                self._thumb_cache = {}
            key = (path, max_side)
            cached = self._thumb_cache.get(key)
            if cached:
                return cached
            with Image.open(path) as im:
                im = im.convert("RGBA")
                w, h = im.size
                scale = max(1.0, max(w, h) / float(max_side))
                tw = int(max(1, round(w / scale)))
                th = int(max(1, round(h / scale)))
                im = im.resize((tw, th), Image.LANCZOS)
                imtk = ImageTk.PhotoImage(im)
                cached = (imtk, (imtk.width(), imtk.height()))
                self._thumb_cache[key] = cached
                return cached
        except Exception:
            return None, (0, 0)

        # Tabs
        self._build_tab_settings()
        self._build_tab_story()
        self._build_tab_characters()
        self._build_tab_locations()
        self._build_tab_shots_export()

        try:
            if isinstance(self.analysis, dict):
                title = self.analysis.get("title")
                if title:
                    parts.append(str(title))
                summary = (
                    self.analysis.get("story_summary")
                    or self.analysis.get("story_precis")
                    or self.analysis.get("logline")
                )
                if summary:
                    parts.append(str(summary))
        except Exception:
            pass
        text = " \n".join(parts).strip()
        return text[:1200]

    def _merge_dna_maps(self, target: Optional[Dict[str, Any]], incoming: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(target, dict):
            target = {}
        if not isinstance(incoming, dict):
            return target
        for key, value in incoming.items():
            if key not in target:
                target[key] = value
            elif isinstance(target.get(key), dict) and isinstance(value, dict):
                target[key] = self._merge_dna_maps(target.get(key), value)
        return target

        if LAST_EXPAND_SCENES_STATUS:
            try:
                self._set_status(LAST_EXPAND_SCENES_STATUS)
            except Exception:
                pass

        # Initialize per-run budget line on the right
        self._init_run_budget()
        ttk.Label(bar, textvariable=self.budget_var, anchor="e").pack(side="right", padx=6)


        try:
            if world_path:
                self._save_world_store_to(world_path)
        except Exception:
            pass

        return added_ids

    def _collect_reference_identity(self, profiles: List[Any], extra_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        seen: List[Any] = []
        for p in profiles or []:
            if p and p not in seen:
                seen.append(p)
        reg_items = {}
        for item in (self.world or {}).get("assets_registry", []) or []:
            if isinstance(item, dict) and item.get("id"):
                reg_items[item["id"]] = item

        ref_ids: List[str] = []
        palette: List[str] = []
        primary_map: Dict[str, str] = {}
        traits_map: Dict[str, Any] = {}

        for p in seen:
            nm = getattr(p, "name", "") or getattr(p, "description", "") or "profile"
            pri = getattr(p, "primary_reference_id", "") or ""
            if pri:
                primary_map[nm] = pri
            for rid in getattr(p, "reference_images", []) or []:
                if rid and rid not in ref_ids:
                    ref_ids.append(rid)
                rec = reg_items.get(rid)
                if rec:
                    for color in (rec.get("palette") or [])[:3]:
                        if color and color not in palette:
                            palette.append(color)
            if getattr(p, "dna_traits", None):
                traits_map[nm] = p.dna_traits

        for rid in extra_ids or []:
            if rid and rid not in ref_ids:
                ref_ids.append(rid)
            rec = reg_items.get(rid)
            if rec:
                for color in (rec.get("palette") or [])[:3]:
                    if color and color not in palette:
                        palette.append(color)

        bits: List[str] = []
        if ref_ids:
            bits.append("Identity-locked to reference images: " + ", ".join(ref_ids))
        if palette:
            bits.append("Honor color DNA: " + ", ".join(palette[:4]))
        trait_lines: List[str] = []
        for nm, trait in traits_map.items():
            if not isinstance(trait, dict):
                continue
            chunk: List[str] = []
            appearance = trait.get("appearance") if isinstance(trait.get("appearance"), dict) else {}
            if appearance:
                hair = appearance.get("hair_color")
                if hair:
                    chunk.append("hair " + str(hair))
                eyes = appearance.get("eye_color")
                if eyes:
                    chunk.append("eyes " + str(eyes))
                face = appearance.get("face_shape")
                if face:
                    chunk.append("face " + str(face))
                features = appearance.get("notable_features")
                if features:
                    if isinstance(features, list):
                        features = ", ".join(str(x) for x in features[:2] if x)
                    chunk.append("features " + str(features))
            wardrobe = trait.get("wardrobe") if isinstance(trait.get("wardrobe"), dict) else {}
            if wardrobe:
                items = wardrobe.get("items")
                if items:
                    if isinstance(items, list):
                        items = ", ".join(str(x) for x in items[:2] if x)
                    chunk.append("wardrobe " + str(items))
            if chunk:
                trait_lines.append(f"{nm}: " + "; ".join(chunk))
        if trait_lines:
            bits.append("Keep identity cues: " + " | ".join(trait_lines[:3]))

        return {
            "bits": bits,
            "ref_ids": ref_ids,
            "primary_map": primary_map,
            "traits": traits_map,
            "palette": palette,
        }


    def _create_style_from_images(self, name: str, paths: List[str]) -> Optional[Dict[str, Any]]:
        cleaned_name = (name or "").strip()
        if not cleaned_name:
            cleaned_name = "Untitled Style"
        if not paths or len(paths) < 5:
            print("[style] need ≥5 images to build a preset")
            return None

        world_path = getattr(self, "world_store_path", "") or ""
        asset_ids: List[str] = []
        palette_samples: List[List[str]] = []
        contrast_vals: List[float] = []
        color_vals: List[float] = []
        edge_vals: List[float] = []
        sample_paths: List[str] = []

        for src in paths:
            if not isinstance(src, str):
                continue
            ext = os.path.splitext(src)[1].lower()
            if ext and ext not in VALID_IMAGE_EXTS:
                continue
            if not os.path.isfile(src):
                continue
            try:
                dst = _copy_into_assets(world_path, src)
            except Exception:
                dst = src
            rec = _register_asset(self.world, dst)
            rid = rec.get("id")
            if not rid:
                continue
            if rid not in asset_ids:
                asset_ids.append(rid)
            sample_paths.append(dst)
            try:
                with Image.open(dst) as im:
                    im = im.convert("RGB")
                    palette_samples.append(_palette_from_image(im, 6))
                    contrast_vals.append(_contrast_norm(im))
                    color_vals.append(_colorfulness(im))
                    edge_vals.append(_edge_density(im))
            except Exception:
                continue

        if len(asset_ids) < 5:
            print("[style] <5 valid images after filtering")
            return None

        flat_palette: List[str] = []
        for pal in palette_samples:
            for color in pal:
                if color and color not in flat_palette:
                    flat_palette.append(color)
        flat_palette = flat_palette[:6]

        contrast = sum(contrast_vals) / len(contrast_vals) if contrast_vals else 0.5
        colorfulness = sum(color_vals) / len(color_vals) if color_vals else 0.5
        edge = sum(edge_vals) / len(edge_vals) if edge_vals else 0.4
        tone = _tone_bias(flat_palette)
        grain = _grain_hint(contrast, edge)
        base_prompt = _summarize_style_prompt(flat_palette, contrast, colorfulness, edge, tone, grain)
        analysis_ctx = ""
        try:
            analysis_ctx = self._analysis_context_snippet()
        except Exception:
            analysis_ctx = ""
        style_desc = _llm_style_summary(self, sample_paths, analysis_ctx, base_prompt)

        preset_id = f"style_{int(time.time())}_{hashlib.md5(cleaned_name.encode('utf-8')).hexdigest()[:6]}"
        preset = {
            "id": preset_id,
            "name": cleaned_name,
            "sample_asset_ids": asset_ids,
            "palette": flat_palette,
            "contrast": round(float(contrast), 3),
            "colorfulness": round(float(colorfulness), 3),
            "edge_density": round(float(edge), 3),
            "tone_bias": tone,
            "grain_hint": grain,
            "style_prompt": style_desc,
        }

        try:
            styles = (self.world or {}).setdefault("style_presets", [])
        except Exception:
            self.world["style_presets"] = []
            styles = self.world["style_presets"]
        styles.append(preset)

        try:
            if self.world_store_path:
                self._save_world_store_to(self.world_store_path)
        except Exception:
            pass

        print(f"[style] created preset '{cleaned_name}' with {len(asset_ids)} samples")
        return preset


    def _resolve_selected_style(self) -> Tuple[Optional[Dict[str, Any]], str]:
        sid = (getattr(self, "selected_style_id", "") or "").strip()
        styles = []
        try:
            styles = (self.world or {}).get("style_presets") or []
        except Exception:
            styles = []
        if sid:
            for preset in styles:
                if isinstance(preset, dict) and (preset.get("id") or "").strip() == sid:
                    name = (preset.get("name") or preset.get("id") or "").strip()
                    return preset, name
        return None, ""


    def _style_prompt_bits(self) -> List[str]:
        bits: List[str] = []
        preset, _ = self._resolve_selected_style()
        if not preset:
            return bits
        desc = (preset.get("style_prompt") or "").strip()
        if desc:
            bits.append("Style: " + desc)
        palette = [c for c in (preset.get("palette") or []) if c][:4]
        if palette:
            bits.append("Style palette: " + ", ".join(palette))
        return bits


    def _current_style_snapshot(self) -> Dict[str, Any]:
        preset, preset_name = self._resolve_selected_style()
        if preset:
            return {
                "preset": preset,
                "id": preset.get("id", ""),
                "name": preset_name or preset.get("name", ""),
            }
        fallback_name = getattr(self, "selected_style_name", "") or getattr(self, "global_style", "")
        return {"preset": None, "id": "", "name": fallback_name}


    def _load_user_styles(self) -> None:
        styles_list = []
        if isinstance(self.world, dict):
            raw = self.world.get("style_presets")
            if isinstance(raw, list):
                styles_list = raw
        self._user_styles = styles_list

        payload = _read_json_safely(_styles_store_path())
        if isinstance(payload, dict):
            source = payload.get("styles") or payload.get("style_presets") or []
        elif isinstance(payload, list):
            source = payload
        else:
            source = []

        existing_by_id: Dict[str, Dict[str, Any]] = {}
        for entry in self._user_styles:
            if isinstance(entry, dict):
                sid = (entry.get("id") or "").strip()
                if sid:
                    existing_by_id[sid] = entry

        for entry in source:
            if not isinstance(entry, dict):
                continue
            sid = (entry.get("id") or "").strip()
            if sid and sid in existing_by_id:
                try:
                    existing_by_id[sid].update(entry)
                except Exception:
                    pass
            else:
                self._user_styles.append(entry)
                if sid:
                    existing_by_id[sid] = entry

        if isinstance(self.world, dict):
            self.world["style_presets"] = self._user_styles


    def _save_user_styles(self) -> None:
        try:
            styles = [s for s in getattr(self, "_user_styles", []) if isinstance(s, dict)]
            payload = {"styles": styles}
            path = _styles_store_path()
            ensure_dir(os.path.dirname(path) or ".")
            _write_json_atomic(path, payload)
        except Exception as exc:
            try:
                print(f"[style] warning: failed to save user styles: {exc}")
            except Exception:
                pass


    def _merge_styles_for_dropdown(self) -> None:
        try:
            if not isinstance(self.world, dict):
                return
            styles = getattr(self, "_user_styles", None)
            if styles is None:
                styles = []
            current = self.world.setdefault("style_presets", [])
            if current is not styles:
                self.world["style_presets"] = styles
            self._refresh_style_dropdown(preserve_selection=False)
        except Exception:
            try:
                self._refresh_style_dropdown(preserve_selection=False)
            except Exception:
                pass


    def _load_thumb(self, path: str, max_side: int = 160):
        """
        Load a thumbnail ImageTk.PhotoImage, caching in self._thumb_cache.
        Returns (imgtk, (w, h)) or (None, (0, 0)) on failure.
        """
        try:
            if not hasattr(self, "_thumb_cache"):
                self._thumb_cache = {}
            key = (path, max_side)
            cached = self._thumb_cache.get(key)
            if cached:
                return cached
            with Image.open(path) as im:
                im = im.convert("RGBA")
                w, h = im.size
                scale = max(1.0, max(w, h) / float(max_side))
                tw = int(max(1, round(w / scale)))
                th = int(max(1, round(h / scale)))
                im = im.resize((tw, th), Image.LANCZOS)
                imtk = ImageTk.PhotoImage(im)
                cached = (imtk, (imtk.width(), imtk.height()))
                self._thumb_cache[key] = cached
                return cached
        except Exception:
            return None, (0, 0)

    def _asset_path_by_id(self, asset_id: str) -> str:
        """Resolve asset file path from world.assets_registry; return '' if not found."""
        try:
            reg = (self.world or {}).get("assets_registry") or []
            for item in reg:
                if isinstance(item, dict) and item.get("id") == asset_id:
                    pth = item.get("path")
                    return pth if isinstance(pth, str) else ""
        except Exception:
            return ""
        return ""


    def _find_profile_by_name(self, name: str, kind_hint: Optional[str] = None) -> Tuple[Optional[str], Optional[Any]]:
        target = (name or "").strip()
        if not target:
            return None, None

        lowered = target.lower()
        sanitized = sanitize_name(target)

        def _match(pool: Dict[str, Any]) -> Optional[str]:
            for nm in pool.keys():
                if (nm or "").lower() == lowered:
                    return nm
            for nm in pool.keys():
                if sanitize_name(nm) == sanitized:
                    return nm
            return None

        order: List[str]
        if kind_hint == "location":
            order = ["location", "character"]
        elif kind_hint == "character":
            order = ["character", "location"]
        else:
            order = ["character", "location"]

        for kind in order:
            pool = self.characters if kind == "character" else self.locations
            matched = _match(pool)
            if matched:
                return kind, pool.get(matched)

        return None, None


    def _drop_org_and_reconnect(self) -> bool:
        try:
            os.environ.pop("OPENAI_ORG_ID", None)
            os.environ.pop("OPENAI_ORGANIZATION", None)
        except Exception:
            pass
        try:
            self.client = OpenAIClient(self.api_key)
            return True
        except Exception as e:
            messagebox.showerror("OpenAI", "Reconnect without Organization failed:\n" + str(e))
            return False


    def _attach_refs_to_profile(self, profile, paths: List[str]) -> List[str]:
        if not profile or not paths:
            return []

        kind = None
        name = getattr(profile, "name", "")
        for nm, obj in (self.characters or {}).items():
            if obj is profile:
                kind = "character"
                name = nm
                break
        if kind is None:
            for nm, obj in (self.locations or {}).items():
                if obj is profile:
                    kind = "location"
                    name = nm
                    break
        if kind is None:
            return []

        world_path = getattr(self, "world_store_path", "") or ""
        bucket = self.world.setdefault("characters" if kind == "character" else "locations", {})
        entry = bucket.setdefault(name, {})
        entry.setdefault("reference_image_ids", [])
        entry.setdefault("visual_cues_from_photos", [])
        entry.setdefault("dna_traits", {})

        profile.reference_images = list(dict.fromkeys(profile.reference_images or []))
        if not isinstance(getattr(profile, "visual_cues_from_photos_list", None), list):
            profile.visual_cues_from_photos_list = []
        if not isinstance(getattr(profile, "dna_traits", None), dict):
            profile.dna_traits = {}

        added_ids: List[str] = []
        analysis_ctx = self._analysis_context_snippet()

        for src in paths:
            if not (isinstance(src, str) and os.path.isfile(src)):
                continue
            ext = os.path.splitext(src)[1].lower()
            if ext and ext not in VALID_IMAGE_EXTS:
                continue
            dst = _copy_into_assets(world_path, src)
            rec = _register_asset(self.world, dst, entity_type=kind, entity_name=name)
            rid = rec.get("id")
            if not rid:
                continue
            if rid not in profile.reference_images:
                profile.reference_images.append(rid)
                added_ids.append(rid)
            if not profile.primary_reference_id:
                profile.primary_reference_id = rid

            palette = rec.get("palette") or []
            luma = float(rec.get("avg_luma") or 0.5)
            cues = _image_dna_phrases(palette, luma)
            existing_cues = profile.visual_cues_from_photos_list or []
            for cue in cues:
                if cue not in existing_cues:
                    existing_cues.append(cue)
            profile.visual_cues_from_photos_list = existing_cues
            profile.visual_cues_from_photos = "; ".join(existing_cues)

            traits = self._llm_extract_relevant_traits(dst, analysis_ctx)
            if traits:
                profile.dna_traits = self._merge_dna_maps(profile.dna_traits, traits)

        entry["reference_image_ids"] = list(dict.fromkeys(profile.reference_images))
        entry["visual_cues_from_photos"] = list(profile.visual_cues_from_photos_list or [])
        if profile.primary_reference_id:
            entry["primary_reference_id"] = profile.primary_reference_id
        if profile.dna_traits:
            entry["dna_traits"] = profile.dna_traits

        try:
            if world_path:
                self._save_world_store_to(world_path)
        except Exception:
            pass

        return added_ids

    def _collect_reference_identity(self, profiles: List[Any], extra_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        seen: List[Any] = []
        for p in profiles or []:
            if p and p not in seen:
                seen.append(p)
        reg_items = {}
        for item in (self.world or {}).get("assets_registry", []) or []:
            if isinstance(item, dict) and item.get("id"):
                reg_items[item["id"]] = item

        ref_ids: List[str] = []
        palette: List[str] = []
        primary_map: Dict[str, str] = {}
        traits_map: Dict[str, Any] = {}

        for p in seen:
            nm = getattr(p, "name", "") or getattr(p, "description", "") or "profile"
            pri = getattr(p, "primary_reference_id", "") or ""
            if pri:
                primary_map[nm] = pri
            for rid in getattr(p, "reference_images", []) or []:
                if rid and rid not in ref_ids:
                    ref_ids.append(rid)
                rec = reg_items.get(rid)
                if rec:
                    for color in (rec.get("palette") or [])[:3]:
                        if color and color not in palette:
                            palette.append(color)
            if getattr(p, "dna_traits", None):
                traits_map[nm] = p.dna_traits

        for rid in extra_ids or []:
            if rid and rid not in ref_ids:
                ref_ids.append(rid)
            rec = reg_items.get(rid)
            if rec:
                for color in (rec.get("palette") or [])[:3]:
                    if color and color not in palette:
                        palette.append(color)

        bits: List[str] = []
        if ref_ids:
            bits.append("Identity-locked to reference images: " + ", ".join(ref_ids))
        if palette:
            bits.append("Honor color DNA: " + ", ".join(palette[:4]))
        trait_lines: List[str] = []
        for nm, trait in traits_map.items():
            if not isinstance(trait, dict):
                continue
            chunk: List[str] = []
            appearance = trait.get("appearance") if isinstance(trait.get("appearance"), dict) else {}
            if appearance:
                hair = appearance.get("hair_color")
                if hair:
                    chunk.append("hair " + str(hair))
                eyes = appearance.get("eye_color")
                if eyes:
                    chunk.append("eyes " + str(eyes))
                face = appearance.get("face_shape")
                if face:
                    chunk.append("face " + str(face))
                features = appearance.get("notable_features")
                if features:
                    if isinstance(features, list):
                        features = ", ".join(str(x) for x in features[:2] if x)
                    chunk.append("features " + str(features))
            wardrobe = trait.get("wardrobe") if isinstance(trait.get("wardrobe"), dict) else {}
            if wardrobe:
                items = wardrobe.get("items")
                if items:
                    if isinstance(items, list):
                        items = ", ".join(str(x) for x in items[:2] if x)
                    chunk.append("wardrobe " + str(items))
            if chunk:
                trait_lines.append(f"{nm}: " + "; ".join(chunk))
        if trait_lines:
            bits.append("Keep identity cues: " + " | ".join(trait_lines[:3]))

        return {
            "bits": bits,
            "ref_ids": ref_ids,
            "primary_map": primary_map,
            "traits": traits_map,
            "palette": palette,
        }


    def _create_style_from_images(self, name: str, paths: List[str]) -> Optional[Dict[str, Any]]:
        cleaned_name = (name or "").strip()
        if not cleaned_name:
            cleaned_name = "Untitled Style"
        if not paths or len(paths) < 5:
            print("[style] need ≥5 images to build a preset")
            return None

        world_path = getattr(self, "world_store_path", "") or ""
        asset_ids: List[str] = []
        palette_samples: List[List[str]] = []
        contrast_vals: List[float] = []
        color_vals: List[float] = []
        edge_vals: List[float] = []
        sample_paths: List[str] = []

        for src in paths:
            if not isinstance(src, str):
                continue
            ext = os.path.splitext(src)[1].lower()
            if ext and ext not in VALID_IMAGE_EXTS:
                continue
            if not os.path.isfile(src):
                continue
            try:
                dst = _copy_into_assets(world_path, src)
            except Exception:
                dst = src
            rec = _register_asset(self.world, dst)
            rid = rec.get("id")
            if not rid:
                continue
            if rid not in asset_ids:
                asset_ids.append(rid)
            sample_paths.append(dst)
            try:
                with Image.open(dst) as im:
                    im = im.convert("RGB")
                    palette_samples.append(_palette_from_image(im, 6))
                    contrast_vals.append(_contrast_norm(im))
                    color_vals.append(_colorfulness(im))
                    edge_vals.append(_edge_density(im))
            except Exception:
                continue

        if len(asset_ids) < 5:
            print("[style] <5 valid images after filtering")
            return None

        flat_palette: List[str] = []
        for pal in palette_samples:
            for color in pal:
                if color and color not in flat_palette:
                    flat_palette.append(color)
        flat_palette = flat_palette[:6]

        contrast = sum(contrast_vals) / len(contrast_vals) if contrast_vals else 0.5
        colorfulness = sum(color_vals) / len(color_vals) if color_vals else 0.5
        edge = sum(edge_vals) / len(edge_vals) if edge_vals else 0.4
        tone = _tone_bias(flat_palette)
        grain = _grain_hint(contrast, edge)
        base_prompt = _summarize_style_prompt(flat_palette, contrast, colorfulness, edge, tone, grain)
        analysis_ctx = ""
        try:
            analysis_ctx = self._analysis_context_snippet()
        except Exception:
            analysis_ctx = ""
        style_desc = _llm_style_summary(self, sample_paths, analysis_ctx, base_prompt)

        preset_id = f"style_{int(time.time())}_{hashlib.md5(cleaned_name.encode('utf-8')).hexdigest()[:6]}"
        preset = {
            "id": preset_id,
            "name": cleaned_name,
            "sample_asset_ids": asset_ids,
            "palette": flat_palette,
            "contrast": round(float(contrast), 3),
            "colorfulness": round(float(colorfulness), 3),
            "edge_density": round(float(edge), 3),
            "tone_bias": tone,
            "grain_hint": grain,
            "style_prompt": style_desc,
        }

        try:
            styles = (self.world or {}).setdefault("style_presets", [])
        except Exception:
            self.world["style_presets"] = []
            styles = self.world["style_presets"]
        styles.append(preset)

        try:
            if self.world_store_path:
                self._save_world_store_to(self.world_store_path)
        except Exception:
            pass

        print(f"[style] created preset '{cleaned_name}' with {len(asset_ids)} samples")
        return preset


    def _resolve_selected_style(self) -> Tuple[Optional[Dict[str, Any]], str]:
        sid = (getattr(self, "selected_style_id", "") or "").strip()
        styles = []
        try:
            styles = (self.world or {}).get("style_presets") or []
        except Exception:
            styles = []
        if sid:
            for preset in styles:
                if isinstance(preset, dict) and (preset.get("id") or "").strip() == sid:
                    name = (preset.get("name") or preset.get("id") or "").strip()
                    return preset, name
        return None, ""


    def _style_prompt_bits(self) -> List[str]:
        bits: List[str] = []
        preset, _ = self._resolve_selected_style()
        if not preset:
            return bits
        desc = (preset.get("style_prompt") or "").strip()
        if desc:
            bits.append("Style: " + desc)
        palette = [c for c in (preset.get("palette") or []) if c][:4]
        if palette:
            bits.append("Style palette: " + ", ".join(palette))
        return bits


    def _current_style_snapshot(self) -> Dict[str, Any]:
        preset, preset_name = self._resolve_selected_style()
        if preset:
            return {
                "preset": preset,
                "id": preset.get("id", ""),
                "name": preset_name or preset.get("name", ""),
            }
        fallback_name = getattr(self, "selected_style_name", "") or getattr(self, "global_style", "")
        return {"preset": None, "id": "", "name": fallback_name}


    def _load_user_styles(self) -> None:
        styles_list = []
        if isinstance(self.world, dict):
            raw = self.world.get("style_presets")
            if isinstance(raw, list):
                styles_list = raw
        self._user_styles = styles_list

        payload = _read_json_safely(_styles_store_path())
        if isinstance(payload, dict):
            source = payload.get("styles") or payload.get("style_presets") or []
        elif isinstance(payload, list):
            source = payload
        else:
            source = []

        existing_by_id: Dict[str, Dict[str, Any]] = {}
        for entry in self._user_styles:
            if isinstance(entry, dict):
                sid = (entry.get("id") or "").strip()
                if sid:
                    existing_by_id[sid] = entry

        for entry in source:
            if not isinstance(entry, dict):
                continue
            sid = (entry.get("id") or "").strip()
            if sid and sid in existing_by_id:
                try:
                    existing_by_id[sid].update(entry)
                except Exception:
                    pass
            else:
                self._user_styles.append(entry)
                if sid:
                    existing_by_id[sid] = entry

        if isinstance(self.world, dict):
            self.world["style_presets"] = self._user_styles


    def _save_user_styles(self) -> None:
        try:
            styles = [s for s in getattr(self, "_user_styles", []) if isinstance(s, dict)]
            payload = {"styles": styles}
            path = _styles_store_path()
            ensure_dir(os.path.dirname(path) or ".")
            _write_json_atomic(path, payload)
        except Exception as exc:
            try:
                print(f"[style] warning: failed to save user styles: {exc}")
            except Exception:
                pass


    def _merge_styles_for_dropdown(self) -> None:
        try:
            if not isinstance(self.world, dict):
                return
            styles = getattr(self, "_user_styles", None)
            if styles is None:
                styles = []
            current = self.world.setdefault("style_presets", [])
            if current is not styles:
                self.world["style_presets"] = styles
            self._refresh_style_dropdown(preserve_selection=False)
        except Exception:
            try:
                self._refresh_style_dropdown(preserve_selection=False)
            except Exception:
                pass


    def _load_thumb(self, path: str, max_side: int = 160):
        """
        Load a thumbnail ImageTk.PhotoImage, caching in self._thumb_cache.
        Returns (imgtk, (w, h)) or (None, (0, 0)) on failure.
        """
        try:
            if not hasattr(self, "_thumb_cache"):
                self._thumb_cache = {}
            key = (path, max_side)
            cached = self._thumb_cache.get(key)
            if cached:
                return cached
            with Image.open(path) as im:
                im = im.convert("RGBA")
                w, h = im.size
                scale = max(1.0, max(w, h) / float(max_side))
                tw = int(max(1, round(w / scale)))
                th = int(max(1, round(h / scale)))
                im = im.resize((tw, th), Image.LANCZOS)
                imtk = ImageTk.PhotoImage(im)
                cached = (imtk, (imtk.width(), imtk.height()))
                self._thumb_cache[key] = cached
                return cached
        except Exception:
            return None, (0, 0)

    def _asset_path_by_id(self, asset_id: str) -> str:
        """Resolve asset file path from world.assets_registry; return '' if not found."""
        try:
            reg = (self.world or {}).get("assets_registry") or []
            for item in reg:
                if isinstance(item, dict) and item.get("id") == asset_id:
                    pth = item.get("path")
                    return pth if isinstance(pth, str) else ""
        except Exception:
            return ""
        return ""


    def _find_profile_by_name(self, name: str, kind_hint: Optional[str] = None) -> Tuple[Optional[str], Optional[Any]]:
        target = (name or "").strip()
        if not target:
            return None, None

        lowered = target.lower()
        sanitized = sanitize_name(target)

        def _match(pool: Dict[str, Any]) -> Optional[str]:
            for nm in pool.keys():
                if (nm or "").lower() == lowered:
                    return nm
            for nm in pool.keys():
                if sanitize_name(nm) == sanitized:
                    return nm
            return None

        order: List[str]
        if kind_hint == "location":
            order = ["location", "character"]
        elif kind_hint == "character":
            order = ["character", "location"]
        else:
            order = ["character", "location"]

        for kind in order:
            pool = self.characters if kind == "character" else self.locations
            matched = _match(pool)
            if matched:
                return kind, pool.get(matched)

        return None, None


    def _drop_org_and_reconnect(self) -> bool:
        try:
            os.environ.pop("OPENAI_ORG_ID", None)
            os.environ.pop("OPENAI_ORGANIZATION", None)
        except Exception:
            pass
        try:
            self.client = OpenAIClient(self.api_key)
            return True
        except Exception as e:
            messagebox.showerror("OpenAI", "Reconnect without Organization failed:\n" + str(e))
            return False

    def aspect_to_size(self, aspect: str) -> str:
        """
        Map an aspect (21:9, 16:9, 3:2, 2:3, 1:1) to a legal OpenAI Images size.
        Falls back to current self.image_size if the aspect is unrecognized.
        """
        return ASPECT_TO_SIZE.get((aspect or "").strip(), self.image_size)
    
    def _normalize_size(self, s: str) -> str:
        """
        Normalize user-visible size selections into an Images API-supported size.
        Special-cases "auto" to honor the currently selected scene aspect.
        """
        allowed = {"1024x1024", "1536x1024", "1024x1536"}
        s = (s or "").strip().lower()
        if s in allowed:
            return s
        if s == "auto":
            # Prefer an explicit aspect if available; otherwise keep our default.
            asp = getattr(self, "scene_render_aspect", None) or DEFAULT_ASPECT
            return self.aspect_to_size(asp)
        # Salvage orientation if a freeform string like "800x1200" appears.
        try:
            nums = [int(x) for x in re.findall(r"\d+", s)[:2]]
            if len(nums) == 2:
                w, h = nums
                return "1536x1024" if w >= h else "1024x1536"
        except Exception:
            pass
        return "1024x1024"

    def _pick_supported_size(self, aspect: str, requested: str) -> str:
        """
        Return an Images API-supported size for the desired aspect.
        Allowed: 1024x1024, 1536x1024, 1024x1536, or 'auto'.
    
        IMPORTANT: For scene renders we must ALWAYS honor the aspect selector.
        Mapping:
          • 21:9, 16:9, 3:2  -> 1536x1024 (wide)
          • 2:3              -> 1024x1536 (tall)
          • 1:1              -> 1024x1024 (square)
        """
        allowed = {"1024x1024", "1536x1024", "1024x1536", "auto"}
        asp = (aspect or "").strip()
        req = (requested or "").strip()
    
        # 1) Aspect mapping always wins when recognized
        if asp in {"21:9", "16:9", "3:2"}:
            return "1536x1024"
        if asp in {"2:3", "9:16"}:
            return "1024x1536"
        if asp == "1:1":
            return "1024x1024"
    
        # 2) Otherwise, respect an explicit valid size (non-auto)
        if req in allowed and req != "auto":
            return req
    
        # 3) Salvage orientation from any numeric hint in 'req'
        try:
            nums = [int(x) for x in re.findall(r"\d+", req)[:2]]
            if len(nums) == 2:
                w, h = nums
                return "1536x1024" if w >= h else "1024x1536"
        except Exception:
            pass
    
        # 4) Safe default
        return "1024x1024"


    def _try_images_generate(
        self,
        prompt: str,
        n: int,
        size: str | None = None,
        refs: list[str] | None = None,
    ) -> list[bytes]:
        """
        Centralized image generation helper that guarantees a legal size is passed.
        """
        if not self.client:
            self._on_connect()
            if not self.client:
                return []
        # Normalize size (and resolve "auto" against the active aspect)
        requested = size or self.image_size
        target_size = self._normalize_size(requested)
    
        try:
            prompt_for_api = self._augment_prompt_for_render(prompt)
            if refs:
                return self.client.generate_images_b64_with_refs(
                    model=self.image_model,
                    prompt=prompt_for_api,
                    size=target_size,
                    ref_data_uris=refs,
                    n=n
                )
            return self.client.generate_images_b64(
                model=self.image_model,
                prompt=prompt_for_api,
                size=target_size,
                n=n
            )
        except Exception as e:
            raise RuntimeError(f"Image generation failed ({target_size}): {e}") from e


    def _current_exposure_settings(self) -> tuple[float, bool, float]:
        try:
            bias = float(getattr(self, "exposure_bias", EXPOSURE_BIAS))
        except Exception:
            bias = float(EXPOSURE_BIAS)
        try:
            post = bool(getattr(self, "post_tonemap", EXPOSURE_POST_TONEMAP))
        except Exception:
            post = bool(EXPOSURE_POST_TONEMAP)
        try:
            emiss = float(getattr(self, "emissive_level", EMISSIVE_LEVEL))
        except Exception:
            emiss = float(EMISSIVE_LEVEL)
        return bias, post, emiss

    def _augment_prompt_for_render(self, prompt: str) -> str:
        bias, _, emiss = self._current_exposure_settings()
        try:
            base = str(prompt or "")
        except Exception:
            base = ""
        cleaned_lines: list[str] = []
        for line in base.splitlines():
            strip = line.strip().lower()
            if strip.startswith("exposure control:") or strip.startswith("emissive lighting:"):
                continue
            cleaned_lines.append(line)
        base_txt = "\n".join(cleaned_lines).strip()
        parts: list[str] = [base_txt] if base_txt else []
        parts.append("Exposure control: " + exposure_language(bias))
        if abs(emiss) >= 0.15:
            parts.append("Emissive lighting: " + emissive_language(emiss))
        augmented = "\n".join(parts)
        try:
            print(f"[prompt] exposure={bias:+.2f} emissive={emiss:+.2f}")
        except Exception:
            pass
        return augmented

    def _process_generated_image(
        self, raw: bytes, ext: str | None = None, *, need_image: bool = False
    ) -> tuple[bytes, Optional["Image.Image"]]:
        bias, post, _ = self._current_exposure_settings()
        processed = raw
        img_obj: Optional["Image.Image"] = None
        log_msg = None
        if post and abs(bias) >= 0.05:
            try:
                buf = io.BytesIO(raw)
                with Image.open(buf) as im:
                    im.load()
                    tonemapped = apply_exposure_tonemap(im, bias)
                    out_buf = io.BytesIO()
                    fmt = None
                    if ext:
                        fmt = {
                            ".png": "PNG",
                            ".jpg": "JPEG",
                            ".jpeg": "JPEG",
                            ".webp": "WEBP",
                        }.get(ext.lower())
                    if not fmt:
                        fmt = tonemapped.format or "PNG"
                    save_img = tonemapped
                    if fmt == "JPEG" and tonemapped.mode == "RGBA":
                        save_img = tonemapped.convert("RGB")
                    save_img.save(out_buf, format=fmt)
                    processed = out_buf.getvalue()
                    img_obj = tonemapped.copy()
                    log_msg = f"[exposure] post tone-map applied (bias={bias:+.2f})"
            except Exception as e:
                log_msg = f"[exposure] tone-map skipped: {e}"
                img_obj = None
                processed = raw
        if need_image and img_obj is None:
            try:
                buf = io.BytesIO(processed)
                with Image.open(buf) as im:
                    im.load()
                    img_obj = im.copy()
            except Exception:
                img_obj = None
        if log_msg:
            try:
                print(log_msg)
            except Exception:
                pass
        return processed, img_obj

    def _process_image_batch(self, imgs: list[bytes], ext: str | None = None) -> list[bytes]:
        processed: list[bytes] = []
        for b in imgs or []:
            pb, _ = self._process_generated_image(b, ext=ext, need_image=False)
            processed.append(pb)
        return processed


    def _build_ui(self):
        """
        Build the main notebook, all tabs, and a bottom status bar that includes:
          - Left: status text
          - Right: per-run "Run est." budget line (tokens/images)
        Also installs global mouse wheel handling.
        """
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)
        self.nb = nb

        # Tabs
        self._build_tab_settings()
        self._build_tab_story()
        self._build_tab_characters()
        self._build_tab_locations()
        self._build_tab_shots_export()
        try:
            self._build_tab_batch_run()
        except Exception:
            pass
        # Bottom status + per-run budget line
        bar = ttk.Frame(self.root)
        bar.pack(side="bottom", fill="x")
        self.status = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self.status, anchor="w").pack(side="left", fill="x", expand=True, padx=(6, 6))

        if LAST_EXPAND_SCENES_STATUS:
            try:
                self._set_status(LAST_EXPAND_SCENES_STATUS)
            except Exception:
                pass

        # Initialize per-run budget line on the right
        self._init_run_budget()
        ttk.Label(bar, textvariable=self.budget_var, anchor="e").pack(side="right", padx=6)

        # Global mouse wheel (so lists scroll without focusing them first)
        try:
            self._install_global_mousewheel()
        except Exception:
            pass
    def _build_tab_batch_run(self):
        t = ttk.Frame(self.nb)
        self.nb.add(t, text="Batch (Stories)")
        frm = ttk.Frame(t, padding=10)
        frm.pack(fill="both", expand=True)
    
        # --- Directory pickers ---
        self.batch_stories_dir  = tk.StringVar(value="")
        self.batch_profiles_dir = tk.StringVar(value="")
        self.batch_out_dir      = tk.StringVar(value="")
        self.batch_aspect       = tk.StringVar(value=DEFAULT_ASPECT)
        self.batch_policy       = tk.StringVar(value="final_prompt")
        self.batch_render_n     = tk.StringVar(value="1")
        self.batch_delay_s      = tk.StringVar(value="0")
    
        # NEW: coverage selector (min/max). Default: "min"
        self.batch_coverage_mode = tk.StringVar(value="min")
    
        def _row(label, var):
            row = ttk.Frame(frm)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=label, width=22).pack(side="left")
            ent = ttk.Entry(row, textvariable=var, width=70)
            ent.pack(side="left", fill="x", expand=True, padx=(6,6))
            return row
    
        # Stories folder
        r = _row("Stories folder (.txt)", self.batch_stories_dir)
        ttk.Button(r, text="Choose…",
                   command=lambda: self.batch_stories_dir.set(
                       filedialog.askdirectory() or self.batch_stories_dir.get()
                   )).pack(side="left")
    
        # Profiles repo (optional)
        r = _row("Profiles repo folder", self.batch_profiles_dir)
        ttk.Button(r, text="Choose…",
                   command=lambda: self.batch_profiles_dir.set(
                       filedialog.askdirectory() or self.batch_profiles_dir.get()
                   )).pack(side="left")
    
        # Output root (per‑story subfolders will be created)
        r = _row("Output root", self.batch_out_dir)
        ttk.Button(r, text="Choose…",
                   command=lambda: self.batch_out_dir.set(
                       filedialog.askdirectory() or self.batch_out_dir.get()
                   )).pack(side="left")
    
        # ---------- General options ----------
        opt = ttk.Labelframe(frm, text="Options")
        opt.pack(fill="x", pady=(8,6))
    
        r2 = ttk.Frame(opt); r2.pack(fill="x", pady=4)
        ttk.Label(r2, text="Render aspect:").pack(side="left")
        ttk.Combobox(r2, width=8, textvariable=self.batch_aspect,
                     values=ASPECT_CHOICES, state="readonly").pack(side="left", padx=(6,12))
        ttk.Label(r2, text="Prompt policy:").pack(side="left")
        ttk.Combobox(r2, width=14, textvariable=self.batch_policy,
                     values=["final_prompt","scene_fused","shot_prompt"], state="readonly").pack(side="left", padx=(6,12))
        ttk.Label(r2, text="Renders per shot:").pack(side="left")
        ttk.Entry(r2, width=4, textvariable=self.batch_render_n).pack(side="left", padx=(6,12))
        ttk.Label(r2, text="Delay (s) between renders:").pack(side="left")
        ttk.Entry(r2, width=4, textvariable=self.batch_delay_s).pack(side="left", padx=(6,12))
        ttk.Label(r2, text="Min words / image:").pack(side="left")
        self.batch_min_words_var = tk.StringVar(value=str(self.min_words_between_images))
        ttk.Entry(r2, width=6, textvariable=self.batch_min_words_var).pack(side="left", padx=(6,12))
        
        # ---------- Final image coverage ----------
        cov = ttk.Labelframe(frm, text="Final image coverage (shots per scene)")
        cov.pack(fill="x", pady=(6,6))
        rowc = ttk.Frame(cov); rowc.pack(fill="x", pady=(4,4))
        ttk.Radiobutton(rowc, text="Minimum (1 image per scene)",
                        variable=self.batch_coverage_mode, value="min").pack(side="left", padx=(4,8))
        ttk.Radiobutton(rowc, text="Maximum (all shots per scene)",
                        variable=self.batch_coverage_mode, value="max").pack(side="left", padx=(4,8))
    
        # ---------- Profiles — creation controls (used only when a profile is missing) ----------
        prof = ttk.Labelframe(frm, text="Profiles to create when missing (repo-first matching is always attempted)")
        prof.pack(fill="x", pady=(8,6))
    
        # Characters — views
        rowc1 = ttk.Frame(prof); rowc1.pack(fill="x", pady=(2,2))
        ttk.Label(rowc1, text="Character views:").pack(side="left")
        self.batch_char_views_vars = {}
    
        def _add_c(vkey, label, default_checked):
            var = tk.BooleanVar(value=default_checked)
            ttk.Checkbutton(rowc1, text=label, variable=var).pack(side="left", padx=(4,2))
            self.batch_char_views_vars[vkey] = var
    
        # Defaults: front + profile_left + profile_right checked
        _add_c("front",              CHAR_SHEET_VIEWS_DEF["front"]["label"], True)
        _add_c("three_quarter_left", CHAR_SHEET_VIEWS_DEF["three_quarter_left"]["label"], False)
        _add_c("profile_left",       CHAR_SHEET_VIEWS_DEF["profile_left"]["label"], True)
        _add_c("three_quarter_right",CHAR_SHEET_VIEWS_DEF["three_quarter_right"]["label"], False)
        _add_c("profile_right",      CHAR_SHEET_VIEWS_DEF["profile_right"]["label"], True)
        _add_c("back",               CHAR_SHEET_VIEWS_DEF["back"]["label"], False)
        _add_c("full_body_tpose",    CHAR_SHEET_VIEWS_DEF["full_body_tpose"]["label"], False)
    
        rowc2 = ttk.Frame(prof); rowc2.pack(fill="x", pady=(2,6))
        ttk.Label(rowc2, text="Images per char view:").pack(side="left")
        self.batch_char_per_view_spin = ttk.Spinbox(rowc2, from_=1, to=3, width=4)
        self.batch_char_per_view_spin.set("1")
        self.batch_char_per_view_spin.pack(side="left", padx=(4,12))
    
        # Locations — views
        rowl1 = ttk.Frame(prof); rowl1.pack(fill="x", pady=(2,2))
        ttk.Label(rowl1, text="Location views:").pack(side="left")
        self.batch_loc_views_vars = {}
    
        def _add_l(vkey, label, default_checked):
            var = tk.BooleanVar(value=default_checked)
            ttk.Checkbutton(rowl1, text=label, variable=var).pack(side="left", padx=(4,2))
            self.batch_loc_views_vars[vkey] = var
    
        # Defaults: establishing + alt_angle checked
        _add_l("establishing", LOC_VIEWS_DEF["establishing"]["label"], True)
        _add_l("alt_angle",    LOC_VIEWS_DEF["alt_angle"]["label"],    True)
        _add_l("detail",       LOC_VIEWS_DEF["detail"]["label"],       False)
    
        rowl2 = ttk.Frame(prof); rowl2.pack(fill="x", pady=(2,2))
        ttk.Label(rowl2, text="Images per loc view:").pack(side="left")
        self.batch_loc_per_view_spin = ttk.Spinbox(rowl2, from_=1, to=3, width=4)
        self.batch_loc_per_view_spin.set("1")
        self.batch_loc_per_view_spin.pack(side="left", padx=(4,12))
    
        # ---------- Run ----------
        run_row = ttk.Frame(frm); run_row.pack(fill="x", pady=(10,0))
        ttk.Button(run_row, text="Run Batch", command=self._on_run_batch_from_ui).pack(side="left")





    def _on_run_batch_from_ui(self):
        import os, threading
        stories_dir  = (self.batch_stories_dir.get()  or "").strip()
        profiles_dir = (self.batch_profiles_dir.get() or "").strip()
        out_dir      = (self.batch_out_dir.get()      or "").strip()
    
        if not stories_dir or not os.path.isdir(stories_dir):
            messagebox.showerror("Batch", "Choose a valid stories folder with .txt files."); return
        if not out_dir:
            messagebox.showerror("Batch", "Choose an output folder."); return
    
        if not self.client:
            self._on_connect()
            if not self.client: return
    
        # Honor aspect from the picker
        try:
            self.scene_render_aspect = (self.batch_aspect.get() or self.scene_render_aspect).strip()
        except Exception:
            pass
    
        # Robust coverage fallback
        try:
            coverage = (self.batch_coverage_mode.get() or "min").strip().lower()
        except Exception:
            coverage = "min"

        # Snapshot min-words knob on the UI thread so worker threads don't touch Tk state
        batch_min_words = None
        try:
            if hasattr(self, "batch_min_words_var"):
                raw = (self.batch_min_words_var.get() or "").strip()
                if raw != "":
                    batch_min_words = int(raw)
        except Exception:
            batch_min_words = None
        if batch_min_words is None:
            try:
                batch_min_words = int(getattr(self, "min_words_between_images", 0) or 0)
            except Exception:
                batch_min_words = 0

        prog = ProgressWindow(self.root, title="Batch")
        prog.set_status("Running…")
        prog.set_progress(2)
    
        def cb(pct: float, text: str | None = None):
            def _u():
                if text:
                    prog.set_status(text); prog.append_log(text)
                prog.set_progress(0 if pct is None else pct)
            self.root.after(0, _u)
    
        def gui_log(line: str):
            self.root.after(0, lambda: prog.append_log(line))
    
        def worker():
            err = None
            try:
                self.run_batch_on_folder(
                    stories_dir=stories_dir,
                    profiles_dir=profiles_dir or os.path.join(out_dir, "_profiles_repo"),
                    out_root=out_dir,
                    prompt_policy=(self.batch_policy.get() or "final_prompt"),
                    render_n=max(1, int(self.batch_render_n.get())),
                    delay_s=max(0, int(self.batch_delay_s.get() or 0)),
                    aspect=self.scene_render_aspect,
                    coverage_mode=coverage,  # "min" or "max"
                    min_words_per_image=batch_min_words,
                    progress_cb=cb,
                    log_cb=gui_log,
                )
                # NEW: Collate everything into out_dir/_COLLATED
                try:
                    collated = self._collate_all_renders(out_dir)
                    if collated:
                        gui_log(f"Collated renders → {collated}")
                except Exception as ce:
                    gui_log(f"Collate step failed: {ce}")
            except Exception as e:
                err = e
            finally:
                self.root.after(0, prog.close)
                if err:
                    self.root.after(0, lambda: messagebox.showerror("Batch", f"Batch failed:\n{err}"))
                else:
                    self.root.after(0, lambda: messagebox.showinfo("Batch", f"Done.\nOutput: {out_dir}"))
    
        threading.Thread(target=worker, daemon=True).start()




    def _init_run_budget(self):
        """Initialize per-run counters and the readout StringVar."""
        # Counters (persist on self)
        try:
            self.run_tokens_prompt = int(getattr(self, "run_tokens_prompt", 0))
            self.run_tokens_completion = int(getattr(self, "run_tokens_completion", 0))
            self.run_images = int(getattr(self, "run_images", 0))
        except Exception:
            self.run_tokens_prompt = 0
            self.run_tokens_completion = 0
            self.run_images = 0
        # Readout
        self.budget_var = getattr(self, "budget_var", tk.StringVar(value="Run est.: tokens 0 (0/0)  •  images 0"))
        self._budget_update_label()

    def _inc_tokens(self, prompt: int = 0, completion: int = 0):
        """Add token counts and refresh the budget readout."""
        try:
            self.run_tokens_prompt += int(max(0, prompt))
            self.run_tokens_completion += int(max(0, completion))
            self._budget_update_label()
        except Exception:
            pass

    def _inc_images_rendered(self, n: int = 1):
        """Bump the count of rendered images for this run and refresh the readout."""
        try:
            self.run_images += int(max(0, n))
            self._budget_update_label()
        except Exception:
            pass

    def _budget_update_label(self):
        """Format the per-run budget label text."""
        try:
            p = int(getattr(self, "run_tokens_prompt", 0))
            c = int(getattr(self, "run_tokens_completion", 0))
            total = p + c
            imgs = int(getattr(self, "run_images", 0))
            txt = f"Run est.: tokens {total:,} ({p:,}/{c:,})  •  images {imgs:,}"
            # If you later add $/1k token + per-image estimates in Settings,
            # you can append "  •  ≈ $X.YY" here.
            self.budget_var.set(txt)
        except Exception:
            # Be defensive: never crash UI updates
            try:
                self.budget_var.set("Run est.: tokens —  •  images —")
            except Exception:
                pass

    def _on_import_analysis_json(self):
        """
        Let the user pick any _analysis.json and merge it into the current session.
        - Updates self.analysis and scene table
        - Seeds characters/locations if they are missing
        - Reapplies world baselines so profiles auto‑load
        """
        p = filedialog.askopenfilename(
            title="Choose an _analysis.json",
            filetypes=[("JSON","*_analysis.json"), ("JSON","*.json")]
        )
        if not p:
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                ana = json.load(f)
            if not isinstance(ana, dict):
                raise ValueError("File is not a JSON object.")
        except Exception as e:
            messagebox.showerror("Import analysis", f"Could not read JSON:\n{e}")
            return

        # Normalize the few fields we rely on
        self.analysis = {
            "story_precis": ana.get("story_precis", ana.get("story_summary","")),
            "story_summary": ana.get("story_summary",""),
            "main_characters": ana.get("main_characters", []),
            "locations": ana.get("locations", []),
            "structure": ana.get("structure", {}),
            "plot_devices": ana.get("plot_devices", []),
            "scenes": ana.get("scenes", []),

        }

        # Seed entities if missing
        for c in self.analysis.get("main_characters", []):
            nm = (c.get("name","") or "").strip()
            if nm and nm not in self.characters:
                self.characters[nm] = CharacterProfile(name=nm, initial_description=c.get("initial_description",""))
        for l in self.analysis.get("locations", []):
            nm = (l.get("name","") or "").strip()
            if nm and nm not in self.locations:
                self.locations[nm] = LocationProfile(name=nm, description=l.get("description",""))

        # Reapply baselines from world.json so the profiles show up immediately
        try:
            self._apply_world_baselines_to_state(create_missing=False)
        except Exception:
            pass

        # Refresh Story/Scenes tab UI
        try:
            self.scenes_by_id = {s.get("id",""): s for s in (self.analysis.get("scenes") or []) if s.get("id")}
            self._render_scene_table()
            self._render_precis_and_movements()
        except Exception:
            pass

        # Refresh Characters & Locations panes
        try:
            self._rebuild_character_panels()
            self._rebuild_location_panels()
        except Exception:
            pass

        self._set_status("Imported prior analysis.")
        messagebox.showinfo("Analysis import", "Analysis loaded and applied.")

    def _build_style_combo_options(self) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, str]]:
        values: List[str] = []
        mapping: Dict[str, Dict[str, Any]] = {}
        id_map: Dict[str, str] = {}

        seen: set[str] = set()
        for name in GLOBAL_STYLE_CHOICES:
            label = name
            values.append(label)
            mapping[label] = {"kind": "builtin", "name": label}
            seen.add(label)

        styles: List[Dict[str, Any]]
        try:
            styles = [s for s in (self.world or {}).get("style_presets", []) if isinstance(s, dict)]
        except Exception:
            styles = []

        if styles:
            separator = "— User Styles —"
            values.append(separator)
            mapping[separator] = {"kind": "separator"}
            seen.add(separator)
            for preset in styles:
                base = (preset.get("name") or preset.get("id") or "User style").strip() or "User style"
                display = base
                suffix = 2
                while display in seen:
                    display = f"{base} ({suffix})"
                    suffix += 1
                values.append(display)
                mapping[display] = {"kind": "user", "preset": preset}
                seen.add(display)
                pid = (preset.get("id") or "").strip()
                if pid:
                    id_map[pid] = display

        return values, mapping, id_map

    def _apply_style_selection_from_display(self, display: str, *, quiet: bool = False) -> None:
        info = self._style_combo_mapping.get(display)
        if not info:
            self.selected_style_id = ""
            self.selected_style_name = display
            if display:
                self.global_style = display
            return

        kind = info.get("kind")
        if kind == "separator":
            return
        if kind == "user":
            preset = info.get("preset") or {}
            pid = (preset.get("id") or "").strip()
            self.selected_style_id = pid
            self.selected_style_name = (preset.get("name") or preset.get("id") or "").strip()
            if self.selected_style_name:
                self.global_style = self.selected_style_name
            if not quiet:
                try:
                    self._set_status(f"Style preset: {self.selected_style_name}")
                except Exception:
                    pass
        else:
            self.selected_style_id = ""
            self.selected_style_name = display
            if display:
                self.global_style = display
            if not quiet:
                try:
                    self._set_status(f"Global style: {display}")
                except Exception:
                    pass

    def _refresh_style_dropdown(self, preserve_selection: bool = True) -> None:
        combo = getattr(self, "style_combo", None)
        if combo is None:
            return

        values, mapping, id_map = self._build_style_combo_options()
        self._style_combo_mapping = mapping
        self._style_display_by_id = id_map
        try:
            combo.configure(values=values)
        except Exception:
            combo["values"] = values

        desired_display = None

        # 1) If we were told to preserve, try the currently selected style id
        if preserve_selection and self.selected_style_id:
            desired_display = id_map.get(self.selected_style_id)

        # 2) If no selection yet, prefer default_style_id from world.json
        if not desired_display:
            try:
                dsid = (self.world or {}).get("default_style_id") or ""
                if dsid:
                    desired_display = id_map.get(dsid)
                    if desired_display:
                        self.selected_style_id = dsid
            except Exception:
                pass

        # 3) Next, prefer a previously selected style name when it’s builtin
        if not desired_display and not self.selected_style_id and self.global_style:
            info = mapping.get(self.global_style)
            if info and info.get("kind") == "builtin":
                desired_display = self.global_style

        # 4) Otherwise pick the first non-separator item
        if not desired_display:
            for candidate in values:
                info = mapping.get(candidate)
                if info and info.get("kind") != "separator":
                    desired_display = candidate
                    break
        if desired_display:
            try:
                combo.set(desired_display)
            except Exception:
                pass
            self._apply_style_selection_from_display(desired_display, quiet=True)

    def _on_style_selected(self, _event=None):
        combo = getattr(self, "style_combo", None)
        if combo is None:
            return
        current = combo.get().strip()
        info = self._style_combo_mapping.get(current)
        if info and info.get("kind") == "separator":
            previous_display = None
            if self.selected_style_id:
                previous_display = self._style_display_by_id.get(self.selected_style_id)
            if not previous_display:
                previous_display = self.selected_style_name or self.global_style or GLOBAL_STYLE_DEFAULT
            if previous_display:
                try:
                    combo.set(previous_display)
                except Exception:
                    pass
                self._apply_style_selection_from_display(previous_display, quiet=True)
            return
        self._apply_style_selection_from_display(current, quiet=False)
        self._refresh_style_dropdown(preserve_selection=True)

    def _export_style_dialog(self):
        if getattr(self, "root", None) is None:
            print("[style] export not available in headless mode")
            return
        sel_id = (getattr(self, "selected_style_id", "") or "").strip()
        target = None
        for preset in getattr(self, "_user_styles", []) or []:
            if isinstance(preset, dict) and (preset.get("id") or "").strip() == sel_id:
                target = preset
                break
        if not target:
            try:
                messagebox.showinfo("Export Style", "Select a user style to export.")
            except Exception:
                print("[style] select a user style to export")
            return

        default_name = (target.get("name") or target.get("id") or "style").strip() or "style"
        default_name = sanitize_name(default_name) or "style"
        out_path = filedialog.asksaveasfilename(
            title="Export Style",
            defaultextension=".style.json",
            initialfile=f"{default_name}.style.json",
            filetypes=[("Style JSON", "*.style.json"), ("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not out_path:
            return

        payload = _style_export_minimal_dict(target)
        try:
            ensure_dir(os.path.dirname(out_path) or ".")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            try:
                messagebox.showerror("Export failed", str(exc))
            except Exception:
                print(f"[style] export failed: {exc}")
            return

        try:
            base = os.path.splitext(out_path)[0]
            for idx, path in enumerate(_style_preview_paths(self, target), start=1):
                thumb_path = f"{base}.preview{idx}.jpg"
                _save_thumb(path, thumb_path, max_side=320)
        except Exception:
            pass

        try:
            messagebox.showinfo("Export Style", f"Exported '{target.get('name', '(unnamed)')}'.")
        except Exception:
            print(f"[style] exported '{target.get('name', '(unnamed)')}' → {out_path}")

    def _import_style_dialog(self):
        if getattr(self, "root", None) is None:
            print("[style] import not available in headless mode")
            return
        in_path = filedialog.askopenfilename(
            title="Import Style",

            filetypes=[("Style JSON", "*.style.json *.json"), ("All Files", "*.*")],

        )
        if not in_path:
            return

        try:
            data = _read_json_safely(in_path)
            if not isinstance(data, dict):
                raise ValueError("Not a JSON object")
            style = data.get("style") if isinstance(data.get("style"), dict) else data
            if not isinstance(style, dict):
                raise ValueError("Invalid style structure")
            sid = (style.get("id") or "").strip()
            name = (style.get("name") or "").strip()
            if not name:
                name = sid or f"Imported_{int(time.time())}"
                style["name"] = name
            if not sid:
                sid = f"style_{int(time.time())}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:6]}"
                style["id"] = sid

            user_styles = getattr(self, "_user_styles", []) or []
            conflict_idx = -1
            for idx, preset in enumerate(user_styles):
                if isinstance(preset, dict) and (preset.get("id") or "").strip() == sid:
                    conflict_idx = idx
                    break

            if conflict_idx >= 0:
                try:
                    replace = messagebox.askyesno(
                        "Import Style",
                        f"Style id '{sid}' already exists.\nYes = Replace, No = Keep both.",
                    )
                except Exception:
                    replace = True
                if replace:
                    user_styles[conflict_idx] = style
                else:
                    sid = f"{sid}_dup{int(time.time())}"
                    style["id"] = sid
                    user_styles.append(style)
            else:
                user_styles.append(style)

            self.selected_style_id = sid
            self.selected_style_name = style.get("name", "")
            try:
                self._save_user_styles()
            except Exception:
                pass

            try:
                if self.world_store_path:
                    # persist default selection so the app doesn't revert to builtin on restart
                    self.world["default_style_id"] = sid
                    self._save_world_store_to(self.world_store_path)
            except Exception:
                pass

            # Rebuild dropdown and select this style
            try:
                self._merge_styles_for_dropdown()
            except Exception:
                self._refresh_style_dropdown(preserve_selection=True)

            # Show it selected in the combobox
            try:
                combo = getattr(self, "style_combo", None)
                # _build_style_combo_options() keeps a map id->display; use it if present
                display = getattr(self, "_style_display_by_id", {}).get(sid) or self.selected_style_name
                if combo and display:
                    combo.set(display)
                    # notify selection logic
                    self._on_style_selected()
            except Exception:
                pass

            try:
                messagebox.showinfo("Import Style", f"Imported '{self.selected_style_name}'.")
            except Exception:

                print(f"[style] imported '{self.selected_style_name}'")
        except Exception as exc:
            try:
                messagebox.showerror("Import failed", str(exc))
            except Exception:
                print(f"[style] import failed: {exc}")

    def import_style_from_path(self, path: str) -> str:
        """
        Import a style preset from a .style.json or .json file and make it the default selection.
        Returns the preset id, or '' on failure.
        """
        try:
            data = _read_json_safely(path)
            if not isinstance(data, dict):
                raise ValueError("Not a JSON object")
            style = data.get("style") if isinstance(data.get("style"), dict) else data
            if not isinstance(style, dict):
                raise ValueError("Invalid style structure")
            sid = (style.get("id") or "").strip()
            name = (style.get("name") or "").strip()
            if not name:
                name = sid or f"Imported_{int(time.time())}"
                style["name"] = name
            if not sid:
                sid = f"style_{int(time.time())}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:6]}"
                style["id"] = sid

            user_styles = getattr(self, "_user_styles", []) or []
            # Replace if id exists, else append
            for idx, preset in enumerate(user_styles):
                if isinstance(preset, dict) and (preset.get("id") or "").strip() == sid:
                    user_styles[idx] = style
                    break
            else:
                user_styles.append(style)

            self.selected_style_id = sid
            self.selected_style_name = style.get("name", "")
            self.global_style = self.selected_style_name or self.global_style

            try:
                self._save_user_styles()
            except Exception:
                pass
            try:
                if self.world_store_path:
                    self.world["default_style_id"] = sid
                    self._save_world_store_to(self.world_store_path)
            except Exception:
                pass

            try:
                self._merge_styles_for_dropdown()
            except Exception:
                self._refresh_style_dropdown(preserve_selection=True)
            return sid
        except Exception as exc:
            print(f"[style] import failed: {exc}")
            return ""

    def _open_style_manager(self):
        if getattr(self, "root", None) is None:
            return
        existing = getattr(self, "_style_mgr_win", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title("Manage Styles")
        win.geometry("860x520")
        self._style_mgr_win = win

        container = ttk.Frame(win, padding=10)
        container.pack(fill="both", expand=True)

        left = ttk.Frame(container)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ttk.Label(left, text="User styles").pack(anchor="w")
        style_list = tk.Listbox(left, height=12)
        style_list.pack(fill="both", expand=True)

        info_var = tk.StringVar(value="Select a style to preview")
        info_label = ttk.Label(left, textvariable=info_var, wraplength=260, justify="left")
        info_label.pack(fill="x", pady=(6, 6))

        preview_frame = ttk.Frame(left)
        preview_frame.pack(fill="x", pady=(0, 10))
        preview_canvases: List[tk.Canvas] = []
        for r in range(2):
            row = ttk.Frame(preview_frame)
            row.pack(fill="x")
            for _ in range(2):
                canvas = tk.Canvas(row, width=120, height=120, highlightthickness=1, relief="solid")
                canvas.pack(side="left", padx=4, pady=4)
                preview_canvases.append(canvas)

        rename_frame = ttk.Frame(left)
        rename_frame.pack(fill="x", pady=(4, 4))
        ttk.Label(rename_frame, text="Rename selected:").pack(anchor="w")
        rename_var = tk.StringVar()
        rename_entry = ttk.Entry(rename_frame, textvariable=rename_var)
        rename_entry.pack(fill="x", pady=(2, 4))

        buttons_row = ttk.Frame(left)
        buttons_row.pack(fill="x", pady=(0, 6))
        default_btn = ttk.Button(buttons_row, text="Set default")
        default_btn.pack(side="left", padx=(0, 6))
        rename_btn = ttk.Button(buttons_row, text="Rename")
        rename_btn.pack(side="left", padx=(0, 6))
        delete_btn = ttk.Button(buttons_row, text="Delete")
        delete_btn.pack(side="left", padx=(0, 6))

        right = ttk.Frame(container)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(right, text="Create new style").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(right, text="Name").grid(row=1, column=0, sticky="w")
        new_name_var = tk.StringVar()
        ttk.Entry(right, textvariable=new_name_var).grid(row=1, column=1, columnspan=2, sticky="we", padx=4, pady=2)

        ttk.Label(right, text="Sample images (≥5)").grid(row=2, column=0, sticky="w")
        file_list = tk.Listbox(right, height=12)
        file_list.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(0, 6))
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(1, weight=1)

        selected_samples: List[str] = []

        def _update_file_list():
            file_list.delete(0, tk.END)
            for path in selected_samples:
                file_list.insert(tk.END, os.path.basename(path))

        def _add_files():
            paths = filedialog.askopenfilenames(parent=win, title="Select style images",
                                                filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff")])
            if not paths:
                return
            for path in paths:
                if path and os.path.isfile(path):
                    ext = os.path.splitext(path)[1].lower()
                    if ext in VALID_IMAGE_EXTS:
                        selected_samples.append(os.path.abspath(path))
            seen = list(dict.fromkeys(selected_samples))
            selected_samples[:] = seen
            _update_file_list()

        def _add_folder():
            folder = filedialog.askdirectory(parent=win, title="Select folder with images")
            if not folder:
                return
            pattern = os.path.join(folder, "**", "*")
            matches = glob.glob(pattern, recursive=True)
            added = False
            for path in matches:
                ext = os.path.splitext(path)[1].lower()
                if ext in VALID_IMAGE_EXTS and os.path.isfile(path):
                    selected_samples.append(os.path.abspath(path))
                    added = True
            if not added:
                messagebox.showinfo("No images", "No compatible images found in that folder.", parent=win)
            selected_samples[:] = list(dict.fromkeys(selected_samples))
            _update_file_list()

        def _clear_files():
            selected_samples.clear()
            _update_file_list()

        ttk.Button(right, text="Add Files…", command=_add_files).grid(row=4, column=0, sticky="w", padx=(0, 6), pady=(0, 6))
        ttk.Button(right, text="Add Folder…", command=_add_folder).grid(row=4, column=1, sticky="w", pady=(0, 6))
        ttk.Button(right, text="Clear", command=_clear_files).grid(row=4, column=2, sticky="w", pady=(0, 6))

        def _create_style():
            paths = list(dict.fromkeys(selected_samples))
            if len(paths) < 5:
                messagebox.showwarning("Need more images", "Select at least five images to build a style.", parent=win)
                return
            preset = self._create_style_from_images(new_name_var.get(), paths)
            if not preset:
                return
            self.selected_style_id = preset.get("id", "")
            self.selected_style_name = preset.get("name", "")
            self.global_style = self.selected_style_name or self.global_style
            selected_samples.clear()
            _update_file_list()
            new_name_var.set("")
            rebuild_list(select_id=self.selected_style_id)
            self._refresh_style_dropdown(preserve_selection=False)

        ttk.Button(right, text="Create Style", command=_create_style).grid(row=5, column=0, columnspan=3, sticky="we", pady=(4, 0))

        display_to_preset: Dict[str, Dict[str, Any]] = {}

        def rebuild_list(select_id: str | None = None):
            style_list.delete(0, tk.END)
            display_to_preset.clear()
            default_id = (self.world or {}).get("default_style_id") or ""
            for preset in (self.world or {}).get("style_presets", []) or []:
                if not isinstance(preset, dict):
                    continue
                name = (preset.get("name") or preset.get("id") or "User style").strip() or "User style"
                label = ("★ " if preset.get("id") == default_id else "  ") + name
                display_to_preset[label] = preset
                style_list.insert(tk.END, label)
            if select_id:
                for idx, lbl in enumerate(style_list.get(0, tk.END)):
                    preset = display_to_preset.get(lbl)
                    if preset and preset.get("id") == select_id:
                        style_list.selection_set(idx)
                        style_list.see(idx)
                        break
            _update_preview()

        def _update_preview():
            self._style_manager_refs = []
            for canvas in preview_canvases:
                canvas.delete("all")
            selection = style_list.curselection()
            if not selection:
                info_var.set("Select a style to preview")
                rename_var.set("")
                return
            label = style_list.get(selection[0])
            preset = display_to_preset.get(label)
            if not preset:
                info_var.set("Select a style to preview")
                rename_var.set("")
                return
            rename_var.set(preset.get("name", ""))
            desc = (preset.get("style_prompt") or "").strip()
            palette = ", ".join((preset.get("palette") or [])[:4])
            detail_lines = []
            if desc:
                detail_lines.append(desc)
            if palette:
                detail_lines.append("Palette: " + palette)
            detail_lines.append(f"Contrast {preset.get('contrast', 0):0.2f} | Color {preset.get('colorfulness',0):0.2f} | Edge {preset.get('edge_density',0):0.2f}")
            info_var.set("\n".join(detail_lines))
            sample_ids = preset.get("sample_asset_ids") or []
            for idx, aid in enumerate(sample_ids[:len(preview_canvases)]):
                path = self._asset_path_by_id(aid)
                if not path or not os.path.isfile(path):
                    continue
                thumb, _ = self._load_thumb(path, max_side=110)
                if thumb:
                    canvas = preview_canvases[idx]
                    canvas.create_image(60, 60, image=thumb)
                    self._style_manager_refs.append(thumb)

        style_list.bind("<<ListboxSelect>>", lambda _e: _update_preview())

        def _rename_selected():
            selection = style_list.curselection()
            if not selection:
                return
            label = style_list.get(selection[0])
            preset = display_to_preset.get(label)
            if not preset:
                return
            new_name = rename_var.get().strip()
            if not new_name:
                messagebox.showwarning("Rename", "Enter a new name for the style.", parent=win)
                return
            preset["name"] = new_name
            try:
                self._save_user_styles()
            except Exception:
                pass
            try:
                if self.world_store_path:
                    self._save_world_store_to(self.world_store_path)
            except Exception:
                pass
            rebuild_list(select_id=preset.get("id"))
            self._refresh_style_dropdown(preserve_selection=False)

        def _delete_selected():
            selection = style_list.curselection()
            if not selection:
                return
            label = style_list.get(selection[0])
            preset = display_to_preset.get(label)
            if not preset:
                return
            if not messagebox.askyesno("Delete style", f"Delete '{preset.get('name','style')}'?", parent=win):
                return
            sid = preset.get("id")
            styles = (self.world or {}).get("style_presets") or []
            self.world["style_presets"] = [p for p in styles if not (isinstance(p, dict) and p.get("id") == sid)]
            if self.world.get("default_style_id") == sid:
                self.world["default_style_id"] = ""
            if self.selected_style_id == sid:
                self.selected_style_id = ""
                self.selected_style_name = self.global_style
            try:
                self._save_user_styles()
            except Exception:
                pass
            try:
                if self.world_store_path:
                    self._save_world_store_to(self.world_store_path)
            except Exception:
                pass
            rebuild_list()
            self._refresh_style_dropdown(preserve_selection=False)


        # Refresh Story/Scenes tab UI
        try:
            self.scenes_by_id = {s.get("id",""): s for s in (self.analysis.get("scenes") or []) if s.get("id")}
            self._render_scene_table()
            self._render_precis_and_movements()
        except Exception:
            pass

        # Refresh Characters & Locations panes
        try:
            self._rebuild_character_panels()
            self._rebuild_location_panels()
        except Exception:
            pass

        self._set_status("Imported prior analysis.")
        messagebox.showinfo("Analysis import", "Analysis loaded and applied.")

    def _build_style_combo_options(self) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, str]]:
        values: List[str] = []
        mapping: Dict[str, Dict[str, Any]] = {}
        id_map: Dict[str, str] = {}

        seen: set[str] = set()
        for name in GLOBAL_STYLE_CHOICES:
            label = name
            values.append(label)
            mapping[label] = {"kind": "builtin", "name": label}
            seen.add(label)

        styles: List[Dict[str, Any]]
        try:
            styles = [s for s in (self.world or {}).get("style_presets", []) if isinstance(s, dict)]
        except Exception:
            styles = []

        if styles:
            separator = "— User Styles —"
            values.append(separator)
            mapping[separator] = {"kind": "separator"}
            seen.add(separator)
            for preset in styles:
                base = (preset.get("name") or preset.get("id") or "User style").strip() or "User style"
                display = base
                suffix = 2
                while display in seen:
                    display = f"{base} ({suffix})"
                    suffix += 1
                values.append(display)
                mapping[display] = {"kind": "user", "preset": preset}
                seen.add(display)
                pid = (preset.get("id") or "").strip()
                if pid:
                    id_map[pid] = display

        return values, mapping, id_map

    def _apply_style_selection_from_display(self, display: str, *, quiet: bool = False) -> None:
        info = self._style_combo_mapping.get(display)
        if not info:
            self.selected_style_id = ""
            self.selected_style_name = display
            if display:
                self.global_style = display
            return

        kind = info.get("kind")
        if kind == "separator":
            return
        if kind == "user":
            preset = info.get("preset") or {}
            pid = (preset.get("id") or "").strip()
            self.selected_style_id = pid
            self.selected_style_name = (preset.get("name") or preset.get("id") or "").strip()
            if self.selected_style_name:
                self.global_style = self.selected_style_name
            if not quiet:
                try:
                    self._set_status(f"Style preset: {self.selected_style_name}")
                except Exception:
                    pass
        else:
            self.selected_style_id = ""
            self.selected_style_name = display
            if display:
                self.global_style = display
            if not quiet:
                try:
                    self._set_status(f"Global style: {display}")
                except Exception:
                    pass

    def _refresh_style_dropdown(self, preserve_selection: bool = True) -> None:
        combo = getattr(self, "style_combo", None)
        if combo is None:
            return

        values, mapping, id_map = self._build_style_combo_options()
        self._style_combo_mapping = mapping
        self._style_display_by_id = id_map
        try:
            combo.configure(values=values)
        except Exception:
            combo["values"] = values

        desired_display = None
        # 1) If we were told to preserve, try the currently selected style id
        if preserve_selection and self.selected_style_id:
            desired_display = id_map.get(self.selected_style_id)

        # 2) If no selection yet, prefer default_style_id from world.json
        if not desired_display:
            try:
                dsid = (self.world or {}).get("default_style_id") or ""
                if dsid:
                    desired_display = id_map.get(dsid)
                    if desired_display:
                        self.selected_style_id = dsid
            except Exception:
                pass

        # 3) Next, prefer a previously selected style name when it’s builtin
        if not desired_display and not self.selected_style_id and self.global_style:
            info = mapping.get(self.global_style)
            if info and info.get("kind") == "builtin":
                desired_display = self.global_style

        # 4) Otherwise pick the first non-separator item
        if not desired_display:
            for candidate in values:
                info = mapping.get(candidate)
                if info and info.get("kind") != "separator":
                    desired_display = candidate
                    break
        if desired_display:
            try:
                combo.set(desired_display)
            except Exception:
                pass
            self._apply_style_selection_from_display(desired_display, quiet=True)

    def _on_style_selected(self, _event=None):
        combo = getattr(self, "style_combo", None)
        if combo is None:
            return
        current = combo.get().strip()
        info = self._style_combo_mapping.get(current)
        if info and info.get("kind") == "separator":
            previous_display = None
            if self.selected_style_id:
                previous_display = self._style_display_by_id.get(self.selected_style_id)
            if not previous_display:
                previous_display = self.selected_style_name or self.global_style or GLOBAL_STYLE_DEFAULT
            if previous_display:
                try:
                    combo.set(previous_display)
                except Exception:
                    pass
                self._apply_style_selection_from_display(previous_display, quiet=True)
            return
        self._apply_style_selection_from_display(current, quiet=False)
        self._refresh_style_dropdown(preserve_selection=True)

    def _export_style_dialog(self):
        if getattr(self, "root", None) is None:
            print("[style] export not available in headless mode")
            return
        sel_id = (getattr(self, "selected_style_id", "") or "").strip()
        target = None
        for preset in getattr(self, "_user_styles", []) or []:
            if isinstance(preset, dict) and (preset.get("id") or "").strip() == sel_id:
                target = preset
                break
        if not target:
            try:
                messagebox.showinfo("Export Style", "Select a user style to export.")
            except Exception:
                print("[style] select a user style to export")
            return

        default_name = (target.get("name") or target.get("id") or "style").strip() or "style"
        default_name = sanitize_name(default_name) or "style"
        out_path = filedialog.asksaveasfilename(
            title="Export Style",
            defaultextension=".style.json",
            initialfile=f"{default_name}.style.json",
            filetypes=[("Style JSON", "*.style.json"), ("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not out_path:
            return

        payload = _style_export_minimal_dict(target)
        try:
            ensure_dir(os.path.dirname(out_path) or ".")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            try:
                messagebox.showerror("Export failed", str(exc))
            except Exception:
                print(f"[style] export failed: {exc}")
            return

        try:
            base = os.path.splitext(out_path)[0]
            for idx, path in enumerate(_style_preview_paths(self, target), start=1):
                thumb_path = f"{base}.preview{idx}.jpg"
                _save_thumb(path, thumb_path, max_side=320)
        except Exception:
            pass

        try:
            messagebox.showinfo("Export Style", f"Exported '{target.get('name', '(unnamed)')}'.")
        except Exception:
            print(f"[style] exported '{target.get('name', '(unnamed)')}' → {out_path}")

    def _import_style_dialog(self):
        if getattr(self, "root", None) is None:
            print("[style] import not available in headless mode")
            return
        in_path = filedialog.askopenfilename(
            title="Import Style",
            filetypes=[("Style JSON", "*.style.json *.json"), ("All Files", "*.*")],
        )
        if not in_path:
            return

        try:
            data = _read_json_safely(in_path)
            if not isinstance(data, dict):
                raise ValueError("Not a JSON object")
            style = data.get("style") if isinstance(data.get("style"), dict) else data
            if not isinstance(style, dict):
                raise ValueError("Invalid style structure")
            sid = (style.get("id") or "").strip()
            name = (style.get("name") or "").strip()
            if not name:
                name = sid or f"Imported_{int(time.time())}"
                style["name"] = name
            if not sid:
                sid = f"style_{int(time.time())}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:6]}"
                style["id"] = sid

            user_styles = getattr(self, "_user_styles", []) or []
            conflict_idx = -1
            for idx, preset in enumerate(user_styles):
                if isinstance(preset, dict) and (preset.get("id") or "").strip() == sid:
                    conflict_idx = idx
                    break

            if conflict_idx >= 0:
                try:
                    replace = messagebox.askyesno(
                        "Import Style",
                        f"Style id '{sid}' already exists.\nYes = Replace, No = Keep both.",
                    )
                except Exception:
                    replace = True
                if replace:
                    user_styles[conflict_idx] = style
                else:
                    sid = f"{sid}_dup{int(time.time())}"
                    style["id"] = sid
                    user_styles.append(style)
            else:
                user_styles.append(style)

            self.selected_style_id = sid
            self.selected_style_name = style.get("name", "")
            try:
                self._save_user_styles()
            except Exception:
                pass

            try:
                if self.world_store_path:
                    # persist default selection so the app doesn't revert to builtin on restart
                    self.world["default_style_id"] = sid
                    self._save_world_store_to(self.world_store_path)
            except Exception:
                pass

            # Rebuild dropdown and select this style
            try:
                self._merge_styles_for_dropdown()
            except Exception:
                self._refresh_style_dropdown(preserve_selection=True)

            # Show it selected in the combobox
            try:
                combo = getattr(self, "style_combo", None)
                # _build_style_combo_options() keeps a map id->display; use it if present
                display = getattr(self, "_style_display_by_id", {}).get(sid) or self.selected_style_name
                if combo and display:
                    combo.set(display)
                    # notify selection logic
                    self._on_style_selected()
            except Exception:
                pass

            try:
                messagebox.showinfo("Import Style", f"Imported '{self.selected_style_name}'.")
            except Exception:
                print(f"[style] imported '{self.selected_style_name}'")
        except Exception as exc:
            try:
                messagebox.showerror("Import failed", str(exc))
            except Exception:
                print(f"[style] import failed: {exc}")

    def import_style_from_path(self, path: str) -> str:
        """
        Import a style preset from a .style.json or .json file and make it the default selection.
        Returns the preset id, or '' on failure.
        """
        try:
            data = _read_json_safely(path)
            if not isinstance(data, dict):
                raise ValueError("Not a JSON object")
            style = data.get("style") if isinstance(data.get("style"), dict) else data
            if not isinstance(style, dict):
                raise ValueError("Invalid style structure")
            sid = (style.get("id") or "").strip()
            name = (style.get("name") or "").strip()
            if not name:
                name = sid or f"Imported_{int(time.time())}"
                style["name"] = name
            if not sid:
                sid = f"style_{int(time.time())}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:6]}"
                style["id"] = sid

            user_styles = getattr(self, "_user_styles", []) or []
            # Replace if id exists, else append
            for idx, preset in enumerate(user_styles):
                if isinstance(preset, dict) and (preset.get("id") or "").strip() == sid:
                    user_styles[idx] = style
                    break
            else:
                user_styles.append(style)

            self.selected_style_id = sid
            self.selected_style_name = style.get("name", "")
            self.global_style = self.selected_style_name or self.global_style

            try:
                self._save_user_styles()
            except Exception:
                pass
            try:
                if self.world_store_path:
                    self.world["default_style_id"] = sid
                    self._save_world_store_to(self.world_store_path)
            except Exception:
                pass

            try:
                self._merge_styles_for_dropdown()
            except Exception:
                self._refresh_style_dropdown(preserve_selection=True)
            return sid
        except Exception as exc:
            print(f"[style] import failed: {exc}")
            return ""

    def _open_style_manager(self):
        if getattr(self, "root", None) is None:
            return
        existing = getattr(self, "_style_mgr_win", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return

        # Refresh Story/Scenes tab UI
        try:
            self.scenes_by_id = {s.get("id",""): s for s in (self.analysis.get("scenes") or []) if s.get("id")}
            self._render_scene_table()
            self._render_precis_and_movements()
        except Exception:
            pass

        # Refresh Characters & Locations panes
        try:
            self._rebuild_character_panels()
            self._rebuild_location_panels()
        except Exception:
            pass

        self._set_status("Imported prior analysis.")
        messagebox.showinfo("Analysis import", "Analysis loaded and applied.")

    def _build_style_combo_options(self) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, str]]:
        values: List[str] = []
        mapping: Dict[str, Dict[str, Any]] = {}
        id_map: Dict[str, str] = {}

        seen: set[str] = set()
        for name in GLOBAL_STYLE_CHOICES:
            label = name
            values.append(label)
            mapping[label] = {"kind": "builtin", "name": label}
            seen.add(label)

        styles: List[Dict[str, Any]]
        try:
            styles = [s for s in (self.world or {}).get("style_presets", []) if isinstance(s, dict)]
        except Exception:
            styles = []

        if styles:
            separator = "— User Styles —"
            values.append(separator)
            mapping[separator] = {"kind": "separator"}
            seen.add(separator)
            for preset in styles:
                base = (preset.get("name") or preset.get("id") or "User style").strip() or "User style"
                display = base
                suffix = 2
                while display in seen:
                    display = f"{base} ({suffix})"
                    suffix += 1
                values.append(display)
                mapping[display] = {"kind": "user", "preset": preset}
                seen.add(display)
                pid = (preset.get("id") or "").strip()
                if pid:
                    id_map[pid] = display

        return values, mapping, id_map

    def _apply_style_selection_from_display(self, display: str, *, quiet: bool = False) -> None:
        info = self._style_combo_mapping.get(display)
        if not info:
            self.selected_style_id = ""
            self.selected_style_name = display
            if display:
                self.global_style = display
            return

        kind = info.get("kind")
        if kind == "separator":
            return
        if kind == "user":
            preset = info.get("preset") or {}
            pid = (preset.get("id") or "").strip()
            self.selected_style_id = pid
            self.selected_style_name = (preset.get("name") or preset.get("id") or "").strip()
            if self.selected_style_name:
                self.global_style = self.selected_style_name
            if not quiet:
                try:
                    self._set_status(f"Style preset: {self.selected_style_name}")
                except Exception:
                    pass
        else:
            self.selected_style_id = ""
            self.selected_style_name = display
            if display:
                self.global_style = display
            if not quiet:
                try:
                    self._set_status(f"Global style: {display}")
                except Exception:
                    pass

    def _refresh_style_dropdown(self, preserve_selection: bool = True) -> None:
        combo = getattr(self, "style_combo", None)
        if combo is None:
            return

        values, mapping, id_map = self._build_style_combo_options()
        self._style_combo_mapping = mapping
        self._style_display_by_id = id_map
        try:
            combo.configure(values=values)
        except Exception:
            combo["values"] = values

        desired_display = None
        # 1) If we were told to preserve, try the currently selected style id
        if preserve_selection and self.selected_style_id:
            desired_display = id_map.get(self.selected_style_id)

        # 2) If no selection yet, prefer default_style_id from world.json
        if not desired_display:
            try:
                dsid = (self.world or {}).get("default_style_id") or ""
                if dsid:
                    desired_display = id_map.get(dsid)
                    if desired_display:
                        self.selected_style_id = dsid
            except Exception:
                pass

        # 3) Next, prefer a previously selected style name when it’s builtin
        if not desired_display and not self.selected_style_id and self.global_style:
            info = mapping.get(self.global_style)
            if info and info.get("kind") == "builtin":
                desired_display = self.global_style

        # 4) Otherwise pick the first non-separator item
        if not desired_display:
            for candidate in values:
                info = mapping.get(candidate)
                if info and info.get("kind") != "separator":
                    desired_display = candidate
                    break
        if desired_display:
            try:
                combo.set(desired_display)
            except Exception:
                pass
            self._apply_style_selection_from_display(desired_display, quiet=True)

    def _on_style_selected(self, _event=None):
        combo = getattr(self, "style_combo", None)
        if combo is None:
            return
        current = combo.get().strip()
        info = self._style_combo_mapping.get(current)
        if info and info.get("kind") == "separator":
            previous_display = None
            if self.selected_style_id:
                previous_display = self._style_display_by_id.get(self.selected_style_id)
            if not previous_display:
                previous_display = self.selected_style_name or self.global_style or GLOBAL_STYLE_DEFAULT
            if previous_display:
                try:
                    combo.set(previous_display)
                except Exception:
                    pass
                self._apply_style_selection_from_display(previous_display, quiet=True)
            return
        self._apply_style_selection_from_display(current, quiet=False)
        self._refresh_style_dropdown(preserve_selection=True)

    def _export_style_dialog(self):
        if getattr(self, "root", None) is None:
            print("[style] export not available in headless mode")
            return
        sel_id = (getattr(self, "selected_style_id", "") or "").strip()
        target = None
        for preset in getattr(self, "_user_styles", []) or []:
            if isinstance(preset, dict) and (preset.get("id") or "").strip() == sel_id:
                target = preset
                break
        if not target:
            try:
                messagebox.showinfo("Export Style", "Select a user style to export.")
            except Exception:
                print("[style] select a user style to export")
            return

        default_name = (target.get("name") or target.get("id") or "style").strip() or "style"
        default_name = sanitize_name(default_name) or "style"
        out_path = filedialog.asksaveasfilename(
            title="Export Style",
            defaultextension=".style.json",
            initialfile=f"{default_name}.style.json",
            filetypes=[("Style JSON", "*.style.json"), ("JSON", "*.json"), ("All Files", "*.*")],
        )
        if not out_path:
            return

        payload = _style_export_minimal_dict(target)
        try:
            ensure_dir(os.path.dirname(out_path) or ".")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            try:
                messagebox.showerror("Export failed", str(exc))
            except Exception:
                print(f"[style] export failed: {exc}")
            return

        try:
            base = os.path.splitext(out_path)[0]
            for idx, path in enumerate(_style_preview_paths(self, target), start=1):
                thumb_path = f"{base}.preview{idx}.jpg"
                _save_thumb(path, thumb_path, max_side=320)
        except Exception:
            pass

        try:
            messagebox.showinfo("Export Style", f"Exported '{target.get('name', '(unnamed)')}'.")
        except Exception:
            print(f"[style] exported '{target.get('name', '(unnamed)')}' → {out_path}")

    def _import_style_dialog(self):
        if getattr(self, "root", None) is None:
            print("[style] import not available in headless mode")
            return
        in_path = filedialog.askopenfilename(
            title="Import Style",
            filetypes=[("Style JSON", "*.style.json *.json"), ("All Files", "*.*")],
        )
        if not in_path:
            return

        try:
            data = _read_json_safely(in_path)
            if not isinstance(data, dict):
                raise ValueError("Not a JSON object")
            style = data.get("style") if isinstance(data.get("style"), dict) else data
            if not isinstance(style, dict):
                raise ValueError("Invalid style structure")
            sid = (style.get("id") or "").strip()
            name = (style.get("name") or "").strip()
            if not name:
                name = sid or f"Imported_{int(time.time())}"
                style["name"] = name
            if not sid:
                sid = f"style_{int(time.time())}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:6]}"
                style["id"] = sid

            user_styles = getattr(self, "_user_styles", []) or []
            conflict_idx = -1
            for idx, preset in enumerate(user_styles):
                if isinstance(preset, dict) and (preset.get("id") or "").strip() == sid:
                    conflict_idx = idx
                    break

            if conflict_idx >= 0:
                try:
                    replace = messagebox.askyesno(
                        "Import Style",
                        f"Style id '{sid}' already exists.\nYes = Replace, No = Keep both.",
                    )
                except Exception:
                    replace = True
                if replace:
                    user_styles[conflict_idx] = style
                else:
                    sid = f"{sid}_dup{int(time.time())}"
                    style["id"] = sid
                    user_styles.append(style)
            else:
                user_styles.append(style)

            self.selected_style_id = sid
            self.selected_style_name = style.get("name", "")
            try:
                self._save_user_styles()
            except Exception:
                pass

            try:
                if self.world_store_path:
                    # persist default selection so the app doesn't revert to builtin on restart
                    self.world["default_style_id"] = sid
                    self._save_world_store_to(self.world_store_path)
            except Exception:
                pass

            # Rebuild dropdown and select this style
            try:
                self._merge_styles_for_dropdown()
            except Exception:
                self._refresh_style_dropdown(preserve_selection=True)

            # Show it selected in the combobox
            try:
                combo = getattr(self, "style_combo", None)
                # _build_style_combo_options() keeps a map id->display; use it if present
                display = getattr(self, "_style_display_by_id", {}).get(sid) or self.selected_style_name
                if combo and display:
                    combo.set(display)
                    # notify selection logic
                    self._on_style_selected()
            except Exception:
                pass

            try:
                messagebox.showinfo("Import Style", f"Imported '{self.selected_style_name}'.")
            except Exception:
                print(f"[style] imported '{self.selected_style_name}'")
        except Exception as exc:
            try:
                messagebox.showerror("Import failed", str(exc))
            except Exception:
                print(f"[style] import failed: {exc}")

    def import_style_from_path(self, path: str) -> str:
        """
        Import a style preset from a .style.json or .json file and make it the default selection.
        Returns the preset id, or '' on failure.
        """
        try:
            data = _read_json_safely(path)
            if not isinstance(data, dict):
                raise ValueError("Not a JSON object")
            style = data.get("style") if isinstance(data.get("style"), dict) else data
            if not isinstance(style, dict):
                raise ValueError("Invalid style structure")
            sid = (style.get("id") or "").strip()
            name = (style.get("name") or "").strip()
            if not name:
                name = sid or f"Imported_{int(time.time())}"
                style["name"] = name
            if not sid:
                sid = f"style_{int(time.time())}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:6]}"
                style["id"] = sid

            user_styles = getattr(self, "_user_styles", []) or []
            # Replace if id exists, else append
            for idx, preset in enumerate(user_styles):
                if isinstance(preset, dict) and (preset.get("id") or "").strip() == sid:
                    user_styles[idx] = style
                    break
            else:
                user_styles.append(style)

            self.selected_style_id = sid
            self.selected_style_name = style.get("name", "")
            self.global_style = self.selected_style_name or self.global_style

            try:
                self._save_user_styles()
            except Exception:
                pass
            try:
                if self.world_store_path:
                    self.world["default_style_id"] = sid
                    self._save_world_store_to(self.world_store_path)
            except Exception:
                pass

            try:
                self._merge_styles_for_dropdown()
            except Exception:
                self._refresh_style_dropdown(preserve_selection=True)
            return sid
        except Exception as exc:
            print(f"[style] import failed: {exc}")
            return ""

    def _open_style_manager(self):
        if getattr(self, "root", None) is None:
            return
        existing = getattr(self, "_style_mgr_win", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title("Manage Styles")
        win.geometry("860x520")
        self._style_mgr_win = win

        container = ttk.Frame(win, padding=10)
        container.pack(fill="both", expand=True)

        left = ttk.Frame(container)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ttk.Label(left, text="User styles").pack(anchor="w")
        style_list = tk.Listbox(left, height=12)
        style_list.pack(fill="both", expand=True)

        info_var = tk.StringVar(value="Select a style to preview")
        info_label = ttk.Label(left, textvariable=info_var, wraplength=260, justify="left")
        info_label.pack(fill="x", pady=(6, 6))

        preview_frame = ttk.Frame(left)
        preview_frame.pack(fill="x", pady=(0, 10))
        preview_canvases: List[tk.Canvas] = []
        for r in range(2):
            row = ttk.Frame(preview_frame)
            row.pack(fill="x")
            for _ in range(2):
                canvas = tk.Canvas(row, width=120, height=120, highlightthickness=1, relief="solid")
                canvas.pack(side="left", padx=4, pady=4)
                preview_canvases.append(canvas)

        rename_frame = ttk.Frame(left)
        rename_frame.pack(fill="x", pady=(4, 4))
        ttk.Label(rename_frame, text="Rename selected:").pack(anchor="w")
        rename_var = tk.StringVar()
        rename_entry = ttk.Entry(rename_frame, textvariable=rename_var)
        rename_entry.pack(fill="x", pady=(2, 4))

        buttons_row = ttk.Frame(left)
        buttons_row.pack(fill="x", pady=(0, 6))
        default_btn = ttk.Button(buttons_row, text="Set default")
        default_btn.pack(side="left", padx=(0, 6))
        rename_btn = ttk.Button(buttons_row, text="Rename")
        rename_btn.pack(side="left", padx=(0, 6))
        delete_btn = ttk.Button(buttons_row, text="Delete")
        delete_btn.pack(side="left", padx=(0, 6))

        right = ttk.Frame(container)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(right, text="Create new style").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(right, text="Name").grid(row=1, column=0, sticky="w")
        new_name_var = tk.StringVar()
        ttk.Entry(right, textvariable=new_name_var).grid(row=1, column=1, columnspan=2, sticky="we", padx=4, pady=2)

        ttk.Label(right, text="Sample images (≥5)").grid(row=2, column=0, sticky="w")
        file_list = tk.Listbox(right, height=12)
        file_list.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(0, 6))
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(1, weight=1)

        selected_samples: List[str] = []

        def _update_file_list():
            file_list.delete(0, tk.END)
            for path in selected_samples:
                file_list.insert(tk.END, os.path.basename(path))

        def _add_files():
            paths = filedialog.askopenfilenames(parent=win, title="Select style images",
                                                filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff")])
            if not paths:
                return
            for path in paths:
                if path and os.path.isfile(path):
                    ext = os.path.splitext(path)[1].lower()
                    if ext in VALID_IMAGE_EXTS:
                        selected_samples.append(os.path.abspath(path))
            seen = list(dict.fromkeys(selected_samples))
            selected_samples[:] = seen
            _update_file_list()

        def _add_folder():
            folder = filedialog.askdirectory(parent=win, title="Select folder with images")
            if not folder:
                return
            pattern = os.path.join(folder, "**", "*")
            matches = glob.glob(pattern, recursive=True)
            added = False
            for path in matches:
                ext = os.path.splitext(path)[1].lower()
                if ext in VALID_IMAGE_EXTS and os.path.isfile(path):
                    selected_samples.append(os.path.abspath(path))
                    added = True
            if not added:
                messagebox.showinfo("No images", "No compatible images found in that folder.", parent=win)
            selected_samples[:] = list(dict.fromkeys(selected_samples))
            _update_file_list()

        def _clear_files():
            selected_samples.clear()
            _update_file_list()

        ttk.Button(right, text="Add Files…", command=_add_files).grid(row=4, column=0, sticky="w", padx=(0, 6), pady=(0, 6))
        ttk.Button(right, text="Add Folder…", command=_add_folder).grid(row=4, column=1, sticky="w", pady=(0, 6))
        ttk.Button(right, text="Clear", command=_clear_files).grid(row=4, column=2, sticky="w", pady=(0, 6))

        def _create_style():
            paths = list(dict.fromkeys(selected_samples))
            if len(paths) < 5:
                messagebox.showwarning("Need more images", "Select at least five images to build a style.", parent=win)
                return
            preset = self._create_style_from_images(new_name_var.get(), paths)
            if not preset:
                return
            self.selected_style_id = preset.get("id", "")
            self.selected_style_name = preset.get("name", "")
            self.global_style = self.selected_style_name or self.global_style
            selected_samples.clear()
            _update_file_list()
            new_name_var.set("")
            rebuild_list(select_id=self.selected_style_id)
            self._refresh_style_dropdown(preserve_selection=False)

        ttk.Button(right, text="Create Style", command=_create_style).grid(row=5, column=0, columnspan=3, sticky="we", pady=(4, 0))

        display_to_preset: Dict[str, Dict[str, Any]] = {}

        def rebuild_list(select_id: str | None = None):
            style_list.delete(0, tk.END)
            display_to_preset.clear()
            default_id = (self.world or {}).get("default_style_id") or ""
            for preset in (self.world or {}).get("style_presets", []) or []:
                if not isinstance(preset, dict):
                    continue
                name = (preset.get("name") or preset.get("id") or "User style").strip() or "User style"
                label = ("★ " if preset.get("id") == default_id else "  ") + name
                display_to_preset[label] = preset
                style_list.insert(tk.END, label)
            if select_id:
                for idx, lbl in enumerate(style_list.get(0, tk.END)):
                    preset = display_to_preset.get(lbl)
                    if preset and preset.get("id") == select_id:
                        style_list.selection_set(idx)
                        style_list.see(idx)
                        break
            _update_preview()

        def _update_preview():
            self._style_manager_refs = []
            for canvas in preview_canvases:
                canvas.delete("all")
            selection = style_list.curselection()
            if not selection:
                info_var.set("Select a style to preview")
                rename_var.set("")
                return
            label = style_list.get(selection[0])
            preset = display_to_preset.get(label)
            if not preset:
                info_var.set("Select a style to preview")
                rename_var.set("")
                return
            rename_var.set(preset.get("name", ""))
            desc = (preset.get("style_prompt") or "").strip()
            palette = ", ".join((preset.get("palette") or [])[:4])
            detail_lines = []
            if desc:
                detail_lines.append(desc)
            if palette:
                detail_lines.append("Palette: " + palette)
            detail_lines.append(f"Contrast {preset.get('contrast', 0):0.2f} | Color {preset.get('colorfulness',0):0.2f} | Edge {preset.get('edge_density',0):0.2f}")
            info_var.set("\n".join(detail_lines))
            sample_ids = preset.get("sample_asset_ids") or []
            for idx, aid in enumerate(sample_ids[:len(preview_canvases)]):
                path = self._asset_path_by_id(aid)
                if not path or not os.path.isfile(path):
                    continue
                thumb, _ = self._load_thumb(path, max_side=110)
                if thumb:
                    canvas = preview_canvases[idx]
                    canvas.create_image(60, 60, image=thumb)
                    self._style_manager_refs.append(thumb)

        style_list.bind("<<ListboxSelect>>", lambda _e: _update_preview())

        def _rename_selected():
            selection = style_list.curselection()
            if not selection:
                return
            label = style_list.get(selection[0])
            preset = display_to_preset.get(label)
            if not preset:
                return
            new_name = rename_var.get().strip()
            if not new_name:
                messagebox.showwarning("Rename", "Enter a new name for the style.", parent=win)
                return
            preset["name"] = new_name
            try:
                self._save_user_styles()
            except Exception:
                pass
            try:
                if self.world_store_path:
                    self._save_world_store_to(self.world_store_path)
            except Exception:
                pass
            rebuild_list(select_id=preset.get("id"))
            self._refresh_style_dropdown(preserve_selection=False)

        def _delete_selected():
            selection = style_list.curselection()
            if not selection:
                return
            label = style_list.get(selection[0])
            preset = display_to_preset.get(label)
            if not preset:
                return
            if not messagebox.askyesno("Delete style", f"Delete '{preset.get('name','style')}'?", parent=win):
                return
            sid = preset.get("id")
            styles = (self.world or {}).get("style_presets") or []
            self.world["style_presets"] = [p for p in styles if not (isinstance(p, dict) and p.get("id") == sid)]
            if self.world.get("default_style_id") == sid:
                self.world["default_style_id"] = ""
            if self.selected_style_id == sid:
                self.selected_style_id = ""
                self.selected_style_name = self.global_style
            try:
                self._save_user_styles()
            except Exception:
                pass
            try:
                if self.world_store_path:
                    self._save_world_store_to(self.world_store_path)
            except Exception:
                pass
            rebuild_list()
            self._refresh_style_dropdown(preserve_selection=False)

        def _set_default_style():
            selection = style_list.curselection()
            if not selection:
                return
            label = style_list.get(selection[0])
            preset = display_to_preset.get(label)
            if not preset:
                return
            sid = preset.get("id") or ""
            self.world["default_style_id"] = sid
            try:
                if self.world_store_path:
                    self._save_world_store_to(self.world_store_path)
            except Exception:
                pass
            self.selected_style_id = sid
            self.selected_style_name = preset.get("name", "")
            if self.selected_style_name:
                self.global_style = self.selected_style_name
            rebuild_list(select_id=sid)
            self._refresh_style_dropdown(preserve_selection=False)

        rename_btn.configure(command=_rename_selected)
        delete_btn.configure(command=_delete_selected)
        default_btn.configure(command=_set_default_style)

        rebuild_list(select_id=self.selected_style_id or (self.world or {}).get("default_style_id"))

    def _build_tab_settings(self):
        """
        Settings tab:
          - API + model selectors
          - Default image size & global style
          - Monthly usage snapshot (billing_var)
          - Aspect pickers
          - world.json path with: Choose…  Import…  Refresh  Open…  Import analysis…
        """
        t = ttk.Frame(self.nb); self.nb.add(t, text="Settings")
        frm = ttk.Frame(t, padding=10); frm.pack(fill="both", expand=True)

        # --- API / Models ---
        ttk.Label(frm, text="OpenAI API key").grid(row=0, column=0, sticky="w")
        self.api_entry = ttk.Entry(frm, width=60); self.api_entry.insert(0, getattr(self, "api_key", "") or "")
        self.api_entry.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(frm, text="LLM model").grid(row=1, column=0, sticky="w")
        self.llm_combo = ttk.Combobox(frm, values=LLM_MODEL_CHOICES, state="readonly")
        self.llm_combo.set(self.llm_model); self.llm_combo.grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(frm, text="Image model").grid(row=2, column=0, sticky="w")
        self.img_combo = ttk.Combobox(frm, values=[OPENAI_IMAGE_MODEL], state="readonly")
        self.img_combo.set(self.image_model); self.img_combo.grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(frm, text="Default image size").grid(row=3, column=0, sticky="w")
        self.size_combo = ttk.Combobox(frm, values=IMAGE_SIZE_CHOICES, state="readonly")
        self.size_combo.set(self.image_size); self.size_combo.grid(row=3, column=1, sticky="w", padx=6)

        ttk.Label(frm, text="Global style").grid(row=4, column=0, sticky="w")
        self.style_combo = ttk.Combobox(frm, state="readonly")
        self.style_combo.grid(row=4, column=1, sticky="we", padx=6)
        self.style_combo.bind("<<ComboboxSelected>>", self._on_style_selected)
        ttk.Button(frm, text="Manage Styles…", command=self._open_style_manager).grid(row=4, column=2, sticky="w", padx=(0,6))
        ttk.Button(frm, text="Export Style…", command=self._export_style_dialog).grid(row=4, column=3, sticky="w", padx=(0,6))
        ttk.Button(frm, text="Import Style…", command=self._import_style_dialog).grid(row=4, column=4, sticky="w", padx=(0,6))
        self._refresh_style_dropdown(preserve_selection=False)

        ttk.Button(frm, text="Connect", command=self._on_connect).grid(row=5, column=0, pady=(8,0))
        frm.grid_columnconfigure(1, weight=1)
        frm.grid_columnconfigure(2, weight=1)

        # --- Monthly usage snapshot (shows after Connect) ---
        self.billing_var = getattr(self, "billing_var", tk.StringVar(value="Usage: —"))
        ttk.Label(frm, text="OpenAI usage (this month)").grid(row=6, column=0, sticky="w")
        ttk.Label(frm, textvariable=self.billing_var).grid(row=6, column=1, sticky="w", padx=6)

        # --- Aspects ---
        ttk.Label(frm, text="Character ref aspect").grid(row=7, column=0, sticky="w")
        char_ref_cmb = ttk.Combobox(frm, values=ASPECT_CHOICES, state="readonly", width=10)
        char_ref_cmb.set(self.char_ref_aspect)
        char_ref_cmb.grid(row=7, column=1, sticky="w", padx=6)
        char_ref_cmb.bind("<<ComboboxSelected>>", lambda e: setattr(self, "char_ref_aspect", char_ref_cmb.get()))

        ttk.Label(frm, text="Location ref aspect").grid(row=8, column=0, sticky="w")
        loc_ref_cmb = ttk.Combobox(frm, values=ASPECT_CHOICES, state="readonly", width=10)
        loc_ref_cmb.set(self.loc_ref_aspect)
        loc_ref_cmb.grid(row=8, column=1, sticky="w", padx=6)
        loc_ref_cmb.bind("<<ComboboxSelected>>", lambda e: setattr(self, "loc_ref_aspect", loc_ref_cmb.get()))

        ttk.Label(frm, text="Scene render aspect").grid(row=9, column=0, sticky="w")
        scene_cmb = ttk.Combobox(frm, values=ASPECT_CHOICES, state="readonly", width=10)
        scene_cmb.set(self.scene_render_aspect)
        scene_cmb.grid(row=9, column=1, sticky="w", padx=6)
        scene_cmb.bind("<<ComboboxSelected>>", lambda e: setattr(self, "scene_render_aspect", scene_cmb.get()))

        # --- world.json (persistent memory) ---
        ttk.Label(frm, text="World store (world.json)").grid(row=10, column=0, sticky="w")
        self.world_path_var = tk.StringVar(value=getattr(self, "world_store_path", "") or "")
        path_row = ttk.Frame(frm); path_row.grid(row=10, column=1, sticky="we", padx=6)
        ttk.Entry(path_row, textvariable=self.world_path_var, width=60).pack(side="left", fill="x", expand=True)
        ttk.Button(path_row, text="Choose…", command=self._on_choose_world_store).pack(side="left", padx=(6,0))
        ttk.Button(path_row, text="Import…", command=self._on_import_world_json).pack(side="left", padx=(6,0))
        # “Refresh from world.json” re-applies baselines from current path
        ttk.Button(path_row, text="Refresh", command=self._refresh_world_from_path).pack(side="left", padx=(6,0))
        ttk.Button(path_row, text="Open…", command=self._open_world_dir).pack(side="left", padx=(6,0))
        # Optional: import a prior _analysis.json to seed scenes/characters/locations
        ttk.Button(path_row, text="Import analysis…", command=self._on_import_analysis_json).pack(side="left", padx=(6,0))

        # ---- Exposure bias slider ----
        ttk.Label(frm, text="Exposure bias (dark ↔ bright)").grid(row=11, column=0, sticky="w", padx=6, pady=(8,0))
        self.exposure_bias_var = tk.DoubleVar(value=float(getattr(self, "exposure_bias", EXPOSURE_BIAS)))

        def _on_exposure_change(_v=None):
            try:
                val = float(self.exposure_bias_var.get())
            except Exception:
                val = float(EXPOSURE_BIAS)
            val = max(-1.0, min(1.0, val))
            self.exposure_bias = val
            globals()["EXPOSURE_BIAS"] = val
            try:
                self._set_status(f"Exposure bias {val:+.2f}")
            except Exception:
                pass

        exposure_scale = ttk.Scale(frm, from_=-1.0, to=1.0, orient="horizontal",
                                   variable=self.exposure_bias_var, command=_on_exposure_change)
        exposure_scale.grid(row=11, column=1, sticky="we", padx=6, pady=(8,0))

        # ---- Post tone-map toggle ----
        ttk.Label(frm, text="Post tone-map").grid(row=12, column=0, sticky="w", padx=6)
        self.post_tonemap_var = tk.BooleanVar(value=bool(getattr(self, "post_tonemap", EXPOSURE_POST_TONEMAP)))

        def _on_post_tonemap_toggle():
            val = bool(self.post_tonemap_var.get())
            self.post_tonemap = val
            globals()["EXPOSURE_POST_TONEMAP"] = val

        ttk.Checkbutton(frm, text="Apply after generation",
                         variable=self.post_tonemap_var, command=_on_post_tonemap_toggle).grid(row=12, column=1, sticky="w", padx=6, pady=(4,0))

        # ---- Emissive / glow slider ----
        ttk.Label(frm, text="Emissive / glow").grid(row=13, column=0, sticky="w", padx=6)
        self.emissive_level_var = tk.DoubleVar(value=float(getattr(self, "emissive_level", EMISSIVE_LEVEL)))

        def _on_emissive_change(_v=None):
            try:
                val = float(self.emissive_level_var.get())
            except Exception:
                val = float(EMISSIVE_LEVEL)
            val = max(-1.0, min(1.0, val))
            self.emissive_level = val
            globals()["EMISSIVE_LEVEL"] = val

        emissive_scale = ttk.Scale(frm, from_=-1.0, to=1.0, orient="horizontal",
                                   variable=self.emissive_level_var, command=_on_emissive_change)
        emissive_scale.grid(row=13, column=1, sticky="we", padx=6, pady=(4,0))

    def _build_tab_story(self):
        t = ttk.Frame(self.nb); self.nb.add(t, text="Story & Scenes")
        outer = ttk.Frame(t, padding=10); outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer); left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Story text:").pack(anchor="w")
        self.story_text = tk.Text(left, wrap="word", height=18)
        self.story_text.pack(fill="both", expand=True, pady=6)
        if TKDND_AVAILABLE:
            self.story_text.drop_target_register(DND_FILES)
            self.story_text.dnd_bind("<<Drop>>", self._on_drop_story)

        row = ttk.Frame(left); row.pack(fill="x")
        ttk.Button(row, text="Load .txt...", command=self._on_load_story).pack(side="left")
        ttk.Button(row, text="Analyze Story", command=self._on_analyze_story).pack(side="left", padx=8)
        self.save_dialogue_btn = ttk.Button(row, text="Save Dialogue Files…", command=self._on_save_dialogue_files)
        self.save_dialogue_btn.state(["disabled"])
        self.save_dialogue_btn.pack(side="left")
        
        self._set_dialogue_save_enabled(False)

        right = ttk.Frame(outer); right.pack(side="left", fill="both", expand=True, padx=10)

        prec_lf = ttk.Labelframe(right, text="Story précis")
        prec_lf.pack(fill="x", padx=0, pady=(0,6))
        self.precis_text = tk.Text(prec_lf, height=6, wrap="word")
        self.precis_text.configure(state="disabled")
        self.precis_text.pack(fill="x", padx=6, pady=6)

        cols = ("id","movement","title","location","time","tone","characters","what")
        self.scene_tree = ttk.Treeview(right, columns=cols, show="headings", height=12, selectmode="browse")
        for c, w in [("id",60),("movement", 90),("title",220),("location",160),("time",90),("tone",120),("characters",240),("what",360)]:
            self.scene_tree.heading(c, text=("Movement" if c=="movement" else ("What happens" if c=="what" else c.title())))
            self.scene_tree.column(c, width=w, anchor="w")
        self.scene_tree.pack(fill="both", expand=True, pady=(0,6))

        mov_lf = ttk.Labelframe(right, text="Movements")
        mov_lf.pack(fill="both", expand=False)
        mov_cols = ("id","name","span","focus","emotional","stakes")
        self.mov_tree = ttk.Treeview(mov_lf, columns=mov_cols, show="headings", height=6, selectmode="browse")
        for c, w in [("id",60),("name",200),("span",120),("focus",260),("emotional",220),("stakes",220)]:
            self.mov_tree.heading(c, text=c.title())
            self.mov_tree.column(c, width=w, anchor="w")
        self.mov_tree.pack(fill="both", expand=True, padx=0, pady=0)

        split_lf = ttk.Labelframe(right, text="Refine / Split scenes")
        split_lf.pack(fill="x", padx=0, pady=(8,0))
        ttk.Button(split_lf, text="Split by movement beat", command=self._on_split_by_movement).pack(side="left", padx=4, pady=6)
        ttk.Button(split_lf, text="Split by location change", command=self._on_split_by_location_change).pack(side="left", padx=4, pady=6)
        ttk.Button(split_lf, text="Split by character intro", command=self._on_split_by_character_intro).pack(side="left", padx=4, pady=6)
        ttk.Button(split_lf, text="Undo last split", command=self._on_undo_split).pack(side="left", padx=10, pady=6)

    def _set_status(self, msg: str):
        self.status.set(msg)

    def _set_dialogue_save_enabled(self, enabled: bool) -> None:
        btn = getattr(self, "save_dialogue_btn", None)
        if not btn:
            return
        try:
            if enabled:
                btn.state(["!disabled"])
            else:
                btn.state(["disabled"])
        except Exception:
            try:
                btn.configure(state=(tk.NORMAL if enabled else tk.DISABLED))
            except Exception:
                pass

    def _on_drop_story(self, event):
        paths = self.root.splitlist(event.data)
        for p in paths:
            if os.path.isfile(p) and p.lower().endswith(".txt"):
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    data = f.read()
                self.story_text.delete("1.0","end")
                self.story_text.insert("1.0", data)
                self._last_story_text = data
                self.input_text_path = p
                self._last_story_path = p
                self._set_status("Loaded: " + os.path.basename(p))
                self._set_dialogue_save_enabled(False)
                break

    def _on_load_story(self):
        p = filedialog.askopenfilename(filetypes=[("Text files","*.txt")])
        if not p: return
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read()
        self.story_text.delete("1.0","end")
        self.story_text.insert("1.0", data)
        self._last_story_text = data
        self.input_text_path = p
        self._last_story_path = p
        self._set_status("Loaded: " + os.path.basename(p))
        self._set_dialogue_save_enabled(False)

    def _fallback_extract_entities(self, text: str):
        import re
        chars = []
        locs = []

        t = text or ""
        t_low = t.lower()

        # Characters (heuristics)
        saw_first_person = bool(re.search(r'\b(I|my|we|our)\b', t, re.IGNORECASE))
        if saw_first_person and ("real estate" in t_low or "estate agent" in t_low or "catalogue star systems" in t_low):
            chars.append({
                "name": "Narrator — Planetary Real‑Estate Agent",
                "initial_description": "An experienced alien registrar who catalogs star systems and rates worlds for settlement; pragmatic, meticulous, wry observer.",
                "role": "narrator / surveyor",
                "goals": "Assess worlds, prevent disputes, facilitate lawful colonization",
                "conflicts": "Bureaucratic constraints, cultural clashes, dangerous frontier"
            })
        elif saw_first_person:
            chars.append({
                "name": "Narrator",
                "initial_description": "First‑person storyteller; pragmatic observer involved in classification of worlds.",
                "role": "narrator",
                "goals": "Document and navigate interstellar processes",
                "conflicts": "Conflicting interests among factions"
            })

        if "human" in t_low or "humans" in t_low:
            chars.append({
                "name": "Human Client",
                "initial_description": "Representative human buyer/settler; ambitious and impulsive.",
                "role": "client / settler",
                "goals": "Acquire land or rights on promising worlds",
                "conflicts": "Regulatory limits; ethical constraints; competition"
            })

        if "federation" in t_low:
            chars.append({
                "name": "Federation Official",
                "initial_description": "Administrator enforcing settlement law and sacred‑site restrictions.",
                "role": "bureaucrat",
                "goals": "Maintain order and adhere to doctrine",
                "conflicts": "Pressure from settlers and coalition worlds"
            })

        # If still fewer than 2, add generic complements
        if len(chars) < 2:
            pool = [
                {"name":"Colonist Prospector","initial_description":"Frontier opportunist seeking claims.","role":"settler","goals":"Stake valuable land","conflicts":"Rivals, harsh environments"},
                {"name":"Corporate Surveyor","initial_description":"Corporate scout balancing profit and law.","role":"surveyor","goals":"Secure resources for corp","conflicts":"Red tape, ethics"}
            ]
            for p in pool:
                if len(chars) >= 2: break
                chars.append(p)

        # Locations (heuristics)
        kws = [
            ("ringworld", "Ringworld Transit Hub"),
            ("dyson", "Dyson Shell Panel Plain"),
            ("station", "Orbital Waystation"),
            ("moon", "Frozen Moon Surface"),
            ("colony", "Frontier Colony Outpost"),
            ("planet", "Temperate Exoplanet Basin"),
            ("archive", "Planetary Registry Archive"),
            ("office", "Interstellar Title Office")
        ]
        seen = set()
        for k, name in kws:
            if k in t_low and name not in seen:
                seen.add(name)
                locs.append({"name": name, "description": f"{name} — canonical set with readable materials, palette and layout suitable for visualization."})

        # Ensure minimum of 2 locations
        if len(locs) < 2:
            defaults = [
                {"name": "Interstellar Title Office", "description": "Bureaucratic hall of counters, terminals and holographic maps; cool lighting; brushed alloy and glass."},
                {"name": "Frontier Colony Outpost", "description": "Makeshift settlement on a raw world; prefab modules, dust, antennae, fuel drums; hard sidelight."},
                {"name": "Orbital Waystation", "description": "Docking ring with windows to the planet; gantries, signage, cargo canisters; sodium‑vapor spill."}
            ]
            for d in defaults:
                if len(locs) >= 2: break
                # Avoid duplicate
                if all(x["name"] != d["name"] for x in locs):
                    locs.append(d)

        return chars, locs

    def _ensure_nonempty_entities(self, analysis: dict, story_text: str) -> dict:
        """
        Guarantee at least 2 characters and 2 locations; backfill minimal scene fields.
        NEW: renumber scenes to contiguous S1..Sn and rewrite movement spans accordingly.
        """
        if not isinstance(analysis, dict):
            return analysis

        text_for_cache = (story_text or "").strip()
        if text_for_cache:
            try:
                analysis["_story_text_cache"] = text_for_cache
                analysis["_story_word_count_cache"] = self._count_words(text_for_cache)
            except Exception:
                analysis["_story_text_cache"] = text_for_cache

        chars = analysis.get("main_characters") or []
        locs  = analysis.get("locations") or []

        # If we already have enough entities, keep going; otherwise backfill from text
        if len(chars) < 2 or len(locs) < 2:
            fb_chars, fb_locs = self._fallback_extract_entities(story_text)
            if len(chars) < 2:
                names = {(c.get("name") or "").strip() for c in chars if isinstance(c, dict)}
                for c in fb_chars:
                    if c["name"] not in names:
                        chars.append(c); names.add(c["name"])
                    if len(chars) >= 2: break
                analysis["main_characters"] = chars
            if len(locs) < 2:
                names = {(l.get("name") or "").strip() for l in locs if isinstance(l, dict)}
                for l in fb_locs:
                    if l["name"] not in names:
                        locs.append(l); names.add(l["name"])
                    if len(locs) >= 2: break
                analysis["locations"] = locs
    
        # Light scene backfill
        seeded_char_names = [c.get("name","") for c in analysis.get("main_characters", []) if isinstance(c, dict)]
        seeded_loc_names  = [l.get("name","") for l in analysis.get("locations", []) if isinstance(l, dict)]
        new_scenes = []
        for s in (analysis.get("scenes") or []):
            if not isinstance(s, dict):
                new_scenes.append(s); continue
            cp = s.get("characters_present") or []
            loc = s.get("location") or ""
            if not cp:
                s["characters_present"] = seeded_char_names[: min(3, len(seeded_char_names))]
            if not loc and seeded_loc_names:
                s["location"] = seeded_loc_names[0]
            new_scenes.append(s)
        analysis["scenes"] = new_scenes
    
        # --- NEW: contiguous renumber (S1..Sn)
        sid_map = {}
        for idx, s in enumerate(analysis.get("scenes", []), 1):
            old_id = (s.get("id") or "").strip()
            new_id = f"S{idx}"
            if not old_id or old_id != new_id:
                sid_map[old_id or new_id] = new_id
                s["id"] = new_id
    
        # Rewrite movement start/end ids if present
        try:
            struct = analysis.get("structure") or {}
            for m in struct.get("movements", []) or []:
                s0 = (m.get("start_scene_id") or "").strip()
                s1 = (m.get("end_scene_id") or "").strip()
                if s0 in sid_map: m["start_scene_id"] = sid_map[s0]
                if s1 in sid_map: m["end_scene_id"] = sid_map[s1]
            analysis["structure"] = struct
        except Exception:
            pass
    
        return analysis


    def _on_analyze_story(self):
        """
        Run story analysis off the UI thread and keep the window responsive.
        Mirrors your existing logic but moves the slow calls into a worker.
        """
        if not self.client:
            self._on_connect()
            if not self.client:
                return
    
        story = self.story_text.get("1.0", "end").strip()
        self._last_story_text = story

        if not story:
            messagebox.showinfo("Story", "Paste or load a story first.")
            return
    
        prog = ProgressWindow(self.root, title="Analyze Story")
        prog.set_status("Analyzing story…")
        prog.set_progress(1)

        self._set_dialogue_save_enabled(False)

        def worker():
            import traceback
            try:
                story_hash = hash_str(story)
    
                # 1) LLM analysis (with memoization)
                if story_hash in self._analysis_cache:
                    analysis = self._analysis_cache[story_hash]
                    if isinstance(analysis, dict):
                        try:
                            analysis["_story_text_cache"] = story
                            analysis["_story_word_count_cache"] = self._count_words(story)
                        except Exception:
                            pass
                else:
                    analysis = LLM.analyze_story(self.client, self.llm_model, story)
                    if isinstance(analysis, dict):
                        analysis = {
                            "title": analysis.get("title", "Untitled"),
                            "logline": analysis.get("logline", ""),
                            "story_precis": analysis.get("story_precis",""),
                            "story_summary": analysis.get("story_summary", ""),
                            "main_characters": analysis.get("main_characters", []) or [],
                            "locations": analysis.get("locations", []) or [],
                            "structure": analysis.get("structure", {}) or {},
                            "plot_devices": analysis.get("plot_devices", []) or [],
                            "scenes": analysis.get("scenes", []) or [],

                        }
                    else:
                        analysis = {
                              "title": "Untitled",
                            "logline": "",
                            "story_precis": "",
                            "story_summary": "",
                            "main_characters": [],
                            "locations": [],
                            "structure": {"movements":[]},
                            "plot_devices": [],
                            "scenes": [],

                        }
    
                    # Ensure at least 2 chars/locs and backfill scene fields
                    analysis = self._ensure_nonempty_entities(analysis, story)
                    self._analysis_cache[story_hash] = analysis
    
                # 2) Normalize scenes (ids, empty fields)
                scenes = analysis.get("scenes", []) or []
                for idx, s in enumerate(scenes, 1):
                    if not s.get("id"):
                        s["id"] = f"S{idx}"
                    if not s.get("what_happens"):
                        k = s.get("key_actions", []) or []
                        s["what_happens"] = s.get("description") or (", ".join(k) if k else "")
                    s["characters_present"] = list(s.get("characters_present", []) or [])
                    s["key_actions"] = list(s.get("key_actions", []) or [])
    
                # 3) Store on app state
                self.analysis = analysis
                self.scenes_by_id = {s.get("id",""): s for s in scenes if s.get("id")}
                self.characters, self.locations, self.shots = {}, {}, []
    
                for c in analysis.get("main_characters", []) or []:
                    nm = (c.get("name","") or "").strip()
                    if nm:
                        self.characters[nm] = CharacterProfile(
                            name=nm,
                            initial_description=c.get("initial_description",""),
                            role=c.get("role",""),
                            goals=c.get("goals",""),
                            conflicts=c.get("conflicts",""),
                        )
                for l in analysis.get("locations", []) or []:
                    nm = (l.get("name","") or "").strip()
                    if nm:
                        self.locations[nm] = LocationProfile(name=nm, description=l.get("description",""))
    
                # 4) Apply baselines from world.json (if any)
                try:
                    self._apply_world_baselines_to_state()
                except Exception:
                    pass
    
                # 5) Refresh UI on main thread
                def _finish_ok():
                    self._render_scene_table()
                    self._render_precis_and_movements()
                    self._rebuild_character_panels()
                    self._rebuild_location_panels()
                    self._render_shot_panels(clear=True)
                    prog.set_status("Story analyzed.")
                    prog.set_progress(100.0)
                    prog.close()
                    self._set_dialogue_save_enabled(True)
                self.root.after(0, _finish_ok)

            except Exception as e:
                traceback.print_exc()
                def _finish_err():
                    prog.close()
                    messagebox.showerror("Analyze", str(e))
                    self._set_dialogue_save_enabled(False)
                self.root.after(0, _finish_err)

        import threading
        threading.Thread(target=worker, daemon=True).start()

    def _on_save_dialogue_files(self):
        story = ""
        try:
            story = self.story_text.get("1.0", "end").strip()
        except Exception:
            story = getattr(self, "_last_story_text", "") or ""
        if not (story or "").strip():
            messagebox.showinfo("Dialogue", "Load and analyze a story first.")
            return

        out_dir = filedialog.askdirectory(title="Choose a folder for dialogue files")
        if not out_dir:
            return

        src_path = getattr(self, "_last_story_path", "") or getattr(self, "input_text_path", "")

        try:
            self._set_status("Saving dialogue files…")
        except Exception:
            pass
        try:
            self.root.config(cursor="watch")
        except Exception:
            pass
        try:
            result = self.analyze_and_emit_dialogue(
                text=story,
                out_dir=out_dir,
                source_text_path=src_path,
            )
        except Exception as exc:
            try:
                messagebox.showerror("Dialogue", f"Failed to save dialogue files:\n{exc}")
            except Exception:
                pass
            finally:
                try:
                    self.root.config(cursor="")
                except Exception:
                    pass
            return

        finally:
            try:
                self.root.config(cursor="")
            except Exception:
                pass

        marked = (result or {}).get("marked") if isinstance(result, dict) else None
        js = (result or {}).get("json") if isinstance(result, dict) else None
        try:
            self._last_export_dir = out_dir
        except Exception:
            pass
        try:
            if marked or js:
                lines = ["Dialogue files saved:"]
                if marked:
                    lines.append("- " + marked)
                if js:
                    lines.append("- " + js)
                messagebox.showinfo("Dialogue", "\n".join(lines))
            else:
                messagebox.showinfo("Dialogue", "Dialogue extraction completed, but no files were reported.")
        except Exception:
            pass
        try:
            self._set_status("Dialogue files saved.")
        except Exception:
            pass


    def _render_precis_and_movements(self):
        self.precis_text.configure(state="normal")
        self.precis_text.delete("1.0","end")
        if self.analysis:
            precis = (self.analysis.get("story_precis") or self.analysis.get("story_summary") or "").strip()
            self.precis_text.insert("1.0", precis)
        self.precis_text.configure(state="disabled")

        for row in self.mov_tree.get_children():
            self.mov_tree.delete(row)
        struct = (self.analysis or {}).get("structure", {}) or {}
        moves = struct.get("movements", []) or []
        for m in moves:
            mid = m.get("id","")
            s0 = (m.get("start_scene_id") or "").strip()
            s1 = (m.get("end_scene_id") or "").strip()
            span = (s0 + (" → " if (s0 or s1) else "") + s1) if (s0 or s1) else ""
            self.mov_tree.insert("", "end", iid=mid or None, values=(
                mid, m.get("name",""), span, m.get("focus",""),
                m.get("emotional_shift",""), m.get("stakes_change","")
            ))

    def _render_scene_table(self):
        for row in self.scene_tree.get_children():
            self.scene_tree.delete(row)
        if not self.analysis:
            return
        for s in self.analysis.get("scenes", []):
            sid = s.get("id","")
            if not sid: continue
            chars = ", ".join(s.get("characters_present", []))
            self.scene_tree.insert("", "end", iid=sid, values=(
                sid,
                s.get("movement_id",""),
                s.get("title",""),
                s.get("location",""),
                s.get("time_of_day",""),
                s.get("tone",""),
                chars,
                s.get("what_happens", s.get("description",""))
            ))

    def _build_tab_characters(self):
        t = ttk.Frame(self.nb); self.nb.add(t, text="Characters")
        top = ttk.Frame(t, padding=10); top.pack(fill="both", expand=True)

        bulk = ttk.Frame(top); bulk.pack(fill="x")
        ttk.Label(bulk, text="Tick characters below →").pack(side="left")
        ttk.Button(bulk, text="Import Character JSON…", command=self._on_bulk_import_characters).pack(side="left", padx=6)
        ttk.Button(bulk, text="Propose baseline (GPT) for ticked", command=self._on_bulk_propose_char_baselines).pack(side="left", padx=6)
        ttk.Button(bulk, text="Generate selected views for ticked", command=self._on_bulk_generate_char_views).pack(side="left", padx=6)

        # --- Aspect controls for Character ref sheets (new) ---
        aspect_row = ttk.Frame(top); aspect_row.pack(fill="x", pady=(6,0))
        ttk.Label(aspect_row, text="Aspect for character refs:").pack(side="left")
        self.char_ref_aspect_var = tk.StringVar(value=self.char_ref_aspect)
        for a in ASPECT_CHOICES:
            ttk.Radiobutton(aspect_row, text=a, value=a, variable=self.char_ref_aspect_var).pack(side="left", padx=2)

        self.char_scroll = ScrollableFrame(top); self.char_scroll.pack(fill="both", expand=True, pady=6)
        self.char_panels: Dict[str, Dict[str, Any]] = {}

    def _rebuild_character_panels(self):
        for w in self.char_scroll.scrollable_frame.winfo_children(): w.destroy()
        self.char_panels.clear()

        for name, c in self.characters.items():
            lf = ttk.Labelframe(self.char_scroll.scrollable_frame, text="👤 " + name)
            lf.pack(fill="x", pady=8, padx=8)

            row0 = ttk.Frame(lf); row0.pack(fill="x", padx=6, pady=(2,2))
            select_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(row0, text="Include in bulk actions", variable=select_var).pack(side="left")
            ttk.Button(row0, text="Load JSON…", command=lambda n=name: self._on_load_character_json(n)).pack(side="left", padx=6)
            ttk.Button(row0, text="Export Character Folder…", command=lambda n=name: self._on_export_character_folder(n)).pack(side="left", padx=6)

            view_vars: Dict[str, tk.BooleanVar] = {}
            views_row = ttk.Frame(lf); views_row.pack(fill="x", padx=6, pady=(2,2))
            for vkey in ["front","three_quarter_left","profile_left","three_quarter_right","profile_right","back","full_body_tpose"]:
                var = tk.BooleanVar(value=(vkey != "back"))
                ttk.Checkbutton(views_row, text=CHAR_SHEET_VIEWS_DEF[vkey]["label"], variable=var).pack(side="left", padx=4)
                view_vars[vkey] = var
            ctrls = ttk.Frame(lf); ctrls.pack(fill="x", padx=6, pady=(2,4))
            ttk.Label(ctrls, text="Images per view:").pack(side="left")
            per_view_spin = ttk.Spinbox(ctrls, from_=1, to=3, increment=1, width=4); per_view_spin.set("1")
            per_view_spin.pack(side="left", padx=(4,12))
            ttk.Label(lf, text="Baseline prompt (shared across views; editable):").pack(anchor="w", padx=6)
            base_prompt_text = tk.Text(lf, height=4, width=140)
            seed = c.sheet_base_prompt or default_baseline_prompt(c)
            base_prompt_text.insert("1.0", seed)
            base_prompt_text.pack(fill="x", padx=6, pady=(0,6))
            btns = ttk.Frame(lf); btns.pack(fill="x", padx=6, pady=(2,4))
            ttk.Button(btns, text="Propose baseline (GPT)", command=lambda n=name: self._on_propose_char_baseline(n)).pack(side="left")
            ttk.Button(btns, text="Baseline from imported photos (GPT)", command=lambda n=name: self._on_baseline_from_imported_char(n)).pack(side="left", padx=6)
            ttk.Button(btns, text="Generate selected views", command=lambda n=name: self._on_generate_char_views(n)).pack(side="left", padx=8)
            ttk.Button(btns, text="Save selected refs to assets", command=lambda n=name: self._on_save_selected_refs_char(n)).pack(side="left", padx=8)

            ref_frame = ttk.Labelframe(lf, text="Reference Images")
            ref_frame.pack(fill="x", padx=6, pady=(0,6))
            ref_frame.columnconfigure(0, weight=1)
            ref_frame.rowconfigure(0, weight=1)
            ref_list = tk.Listbox(ref_frame, height=6)
            ref_list.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            ref_scroll = ttk.Scrollbar(ref_frame, orient="vertical", command=ref_list.yview)
            ref_scroll.grid(row=0, column=1, sticky="ns", pady=4)
            ref_list.configure(yscrollcommand=ref_scroll.set)
            thumb_canvas = tk.Canvas(ref_frame, width=176, height=176, highlightthickness=1, relief="solid")
            thumb_canvas.grid(row=0, column=2, rowspan=2, sticky="ne", padx=(8,4), pady=4)
            thumb_label = ttk.Label(ref_frame, text="", width=32)
            thumb_label.grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=(0,4))

            def _char_add_files(nm=name):
                c_obj = self.characters.get(nm)
                if not c_obj:
                    messagebox.showinfo("References", "Select a character first.")
                    return
                paths = filedialog.askopenfilenames(
                    title=f"Add reference images for {nm}",
                    filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff")]
                )
                if not paths:
                    return
                added = self._attach_refs_to_profile(c_obj, list(paths))
                self._refresh_character_ref_list(nm)
                self._render_character_panel_thumb(nm)
                try:
                    self._set_status(f"Added {len(added)} reference image(s) to {nm}")
                except Exception:
                    pass

            def _char_add_folder(nm=name):
                c_obj = self.characters.get(nm)
                if not c_obj:
                    messagebox.showinfo("References", "Select a character first.")
                    return
                folder = filedialog.askdirectory(title=f"Import folder for {nm}")
                if not folder:
                    return
                paths = [
                    p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                    if os.path.splitext(p)[1].lower() in VALID_IMAGE_EXTS and os.path.isfile(p)
                ]
                if not paths:
                    try:
                        messagebox.showinfo("References", "No compatible images found in that folder.")
                    except Exception:
                        pass
                    return
                added = self._attach_refs_to_profile(c_obj, paths)
                self._refresh_character_ref_list(nm)
                self._render_character_panel_thumb(nm)
                try:
                    self._set_status(f"Imported {len(added)} image(s) for {nm}")
                except Exception:
                    pass

            def _char_set_primary(nm=name):
                c_obj = self.characters.get(nm)
                if not c_obj:
                    return
                sel = ref_list.curselection()
                if not sel:
                    return
                if sel[0] < len(c_obj.reference_images):
                    c_obj.primary_reference_id = c_obj.reference_images[sel[0]]
                    entry = self.world.setdefault("characters", {}).setdefault(nm, {})
                    entry["primary_reference_id"] = c_obj.primary_reference_id
                    try:
                        if self.world_store_path:
                            self._save_world_store_to(self.world_store_path)
                    except Exception:
                        pass
                    self._refresh_character_ref_list(nm)
                    self._render_character_panel_thumb(nm)

            def _char_remove_ref(nm=name):
                c_obj = self.characters.get(nm)
                if not c_obj:
                    return
                sel = ref_list.curselection()
                if not sel:
                    return
                if sel[0] >= len(c_obj.reference_images):
                    return
                rid = c_obj.reference_images[sel[0]]
                c_obj.reference_images = [x for x in c_obj.reference_images if x != rid]
                if c_obj.primary_reference_id == rid:
                    c_obj.primary_reference_id = c_obj.reference_images[0] if c_obj.reference_images else ""
                entry = self.world.setdefault("characters", {}).setdefault(nm, {})
                entry["reference_image_ids"] = list(dict.fromkeys(c_obj.reference_images))
                entry["primary_reference_id"] = c_obj.primary_reference_id
                try:
                    if self.world_store_path:
                        self._save_world_store_to(self.world_store_path)
                except Exception:
                    pass
                self._refresh_character_ref_list(nm)
                self._render_character_panel_thumb(nm)

            btn_row = ttk.Frame(ref_frame)
            btn_row.grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=(0,4))
            ttk.Button(btn_row, text="Add Images…", command=_char_add_files).pack(side="left", padx=(0,4))
            ttk.Button(btn_row, text="Add Folder…", command=_char_add_folder).pack(side="left", padx=(0,4))
            ttk.Button(btn_row, text="Set Primary", command=_char_set_primary).pack(side="left", padx=(0,4))
            ttk.Button(btn_row, text="Remove", command=_char_remove_ref).pack(side="left")

            ref_list.bind("<<ListboxSelect>>", lambda e, nm=name: self._render_character_panel_thumb(nm))

            thumb_rows: Dict[str, ttk.Frame] = {}
            for vkey in CHAR_SHEET_VIEWS_DEF.keys():
                row = ttk.Labelframe(lf, text=CHAR_SHEET_VIEWS_DEF[vkey]["label"])
                row.pack(fill="x", padx=6, pady=(4,4))
                head = ttk.Frame(row); head.pack(fill="x", padx=4, pady=(2,2))
                ttk.Button(head, text="Import…", command=lambda n=name, vk=vkey: self._on_import_char_images(n, vk)).pack(side="left")
                thumb_area = ttk.Frame(row); thumb_area.pack(fill="x", padx=4, pady=(2,6))
                thumb_rows[vkey] = thumb_area

            self.char_panels[name] = {
                "select_var": select_var,
                "view_vars": view_vars,
                "per_view_spin": per_view_spin,
                "base_prompt_text": base_prompt_text,
                "thumb_rows": thumb_rows,
                "ref_list": ref_list,
                "thumb_canvas": thumb_canvas,
                "thumb_label": thumb_label,
            }

            self._refresh_character_ref_list(name)
            self._render_character_panel_thumb(name)
            for vkey, imgs in c.sheet_images.items():
                self._refresh_char_view_thumbs(name, vkey)

    def _selected_char_views(self, name: str) -> List[str]:
        vp = self.char_panels.get(name, {}).get("view_vars", {})
        return [k for k, var in vp.items() if var.get()]

    def _images_per_view(self, name: str) -> int:
        spin = self.char_panels.get(name, {}).get("per_view_spin")
        try:
            return max(1, min(3, int(spin.get())))
        except Exception:
            return 1

    def _get_char_baseline(self, name: str) -> str:
        c = self.characters[name]
        box = self.char_panels[name]["base_prompt_text"]
        txt = box.get("1.0","end").strip()
        if not txt:
            txt = c.sheet_base_prompt or default_baseline_prompt(c)
        c.sheet_base_prompt = txt
        return txt

    def _on_propose_char_baseline(self, name: str):
        if not self.client:
            self._on_connect()
            if not self.client: return
        c = self.characters[name]
        baseline = LLM.propose_unified_character_prompt(
            self.client, self.llm_model,
            (self.analysis or {}).get("story_summary",""),
            c,
            extra_cues=c.visual_cues_from_photos
        ) or default_baseline_prompt(c)
        # Inject current global style/preset details so ref sheets follow the chosen look.
        try:
            style_name = (self.selected_style_name or self.global_style or "").strip()
            style_bits = []
            try:
                style_bits = list(self._style_prompt_bits() or [])
            except Exception:
                style_bits = []
            lines = [baseline.strip()]
            if style_name and "global visual style:" not in baseline.lower():
                lines.append(f"Global visual style: {style_name}.")
            # Avoid duplicating style detail lines if already present
            if style_bits:
                for bit in style_bits:
                    if bit and bit not in baseline:
                        lines.append(bit)
            baseline = "\n".join([ln for ln in lines if ln]).strip()
        except Exception:
            pass
        c.sheet_base_prompt = baseline
        box = self.char_panels[name]["base_prompt_text"]
        box.delete("1.0","end"); box.insert("1.0", baseline)
        self._set_status("Baseline proposed for " + name)

    def _collect_ticked_images_for_char(self, name: str, max_count: int = 50) -> List[bytes]:
        c = self.characters[name]
        out: List[bytes] = []
        # Prioritize views for DNA: front → profiles → back → 3/4 → t-pose
        view_order = ["front","profile_left","profile_right","back","three_quarter_left","three_quarter_right","full_body_tpose"]
        for vk in view_order:
            imgs = c.sheet_images.get(vk, [])
            flags = c.sheet_selected.get(vk, [])
            for i, b in enumerate(imgs):
                if i < len(flags) and flags[i]:
                    out.append(b)
                    if len(out) >= max_count:
                        return out
        return out

    def _on_baseline_from_imported_char(self, name: str):
        if not self.client:
            self._on_connect()
            if not self.client: return
        c = self.characters[name]
        imgs = self._collect_ticked_images_for_char(name, max_count=4)
        if not imgs:
            messagebox.showinfo("Baseline from photos", "Tick ('Use') one or more images first, then try again.")
            return

        self._set_status("Analyzing selected photos → baseline for " + name + "…")
        self.root.config(cursor="watch"); self.root.update_idletasks()
        try:
            cues_all = []
            for b in imgs:
                cues = LLM.extract_visual_cues_from_image(self.client, self.llm_model, b)
                if cues:
                    cues_all.append(cues)
            if cues_all:
                c.visual_cues_from_photos = " ".join(cues_all).strip()
            baseline = LLM.propose_unified_character_prompt(
                self.client, self.llm_model,
                (self.analysis or {}).get("story_summary",""),
                c,
                extra_cues=c.visual_cues_from_photos
            ) or default_baseline_prompt(c)
            # Inject current global style/preset details
            try:
                style_name = (self.selected_style_name or self.global_style or "").strip()
                style_bits = []
                try:
                    style_bits = list(self._style_prompt_bits() or [])
                except Exception:
                    style_bits = []
                lines = [baseline.strip()]
                if style_name and "global visual style:" not in baseline.lower():
                    lines.append(f"Global visual style: {style_name}.")
                if style_bits:
                    for bit in style_bits:
                        if bit and bit not in baseline:
                            lines.append(bit)
                baseline = "\n".join([ln for ln in lines if ln]).strip()
            except Exception:
                pass
            c.sheet_base_prompt = baseline
            box = self.char_panels[name]["base_prompt_text"]
            box.delete("1.0","end"); box.insert("1.0", baseline)
            self._set_status("Baseline updated from selected photos for " + name + ".")
        except Exception as e:
            messagebox.showerror("Baseline from photos", str(e))
        finally:
            self.root.config(cursor="")

    def _on_generate_char_views(self, name: str):
        if not self.client:
            self._on_connect()
            if not self.client: return
        c = self.characters[name]
        baseline = self._get_char_baseline(name)
        views = self._selected_char_views(name)
        if not views:
            messagebox.showinfo("Character", "Select at least one view.")
            return
        n = self._images_per_view(name)
        size_to_use = self.aspect_to_size(self.char_ref_aspect_var.get() if hasattr(self, "char_ref_aspect_var") else self.char_ref_aspect)
    
        prog = ProgressWindow(self.root, title=f"Generate views — {name}")
        prog.set_status("Starting…"); prog.set_progress(1)
    
        def worker():
            errs = []
            try:
                for idx, vkey in enumerate(views, 1):
                    prompt = build_view_prompt_from_baseline(baseline, vkey, self.global_style)
                    def _tick():
                        prog.set_status(f"{vkey} ({idx}/{len(views)})…")
                        prog.set_progress(100.0 * (idx-1) / max(1,len(views)))
                    self.root.after(0, _tick)
                    imgs = self._try_images_generate(prompt, n, size=size_to_use)
                    processed_imgs = self._process_image_batch(imgs)
                    c.sheet_images.setdefault(vkey, []).extend(processed_imgs)
                    c.sheet_selected.setdefault(vkey, []).extend([False] * len(processed_imgs))
                    self.root.after(0, lambda vk=vkey: self._refresh_char_view_thumbs(name, vk))
            except Exception as e:
                errs.append(str(e))
            finally:
                def _done():
                    prog.close()
                    if errs:
                        messagebox.showerror("Generate", "\n".join(errs))
                    else:
                        self._set_status(f"Generated images for {name}.")
                self.root.after(0, _done)
    
        import threading
        threading.Thread(target=worker, daemon=True).start()


    def _on_import_char_images(self, name: str, view_key: str):
        paths = filedialog.askopenfilename(
            title="Import images for " + name + " — " + CHAR_SHEET_VIEWS_DEF[view_key]["label"],
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.webp")],
            multiple=True
        )
        if not paths:
            return
        c = self.characters[name]
        added: List[bytes] = []
        for p in paths:
            try:
                with open(p, "rb") as f:
                    b = f.read()
                added.append(b)
            except Exception as e:
                print("Import error:", e)
        if not added:
            return
        c.sheet_images.setdefault(view_key, []).extend(added)
        c.sheet_selected.setdefault(view_key, [])
        c.sheet_selected[view_key].extend([False]*len(added))
        self._refresh_char_view_thumbs(name, view_key)
        self._set_status("Imported " + str(len(added)) + " image(s) to " + name + " • " + CHAR_SHEET_VIEWS_DEF[view_key]['label'] + ".")

    def _refresh_char_view_thumbs(self, name: str, view_key: str):
        row = self.char_panels[name]["thumb_rows"][view_key]
        for w in row.winfo_children():
            w.destroy()
        c = self.characters[name]
        images = c.sheet_images.get(view_key, []) or []
        c.sheet_selected.setdefault(view_key, [])
        sel = c.sheet_selected[view_key]
        if len(sel) < len(images):
            sel.extend([False] * (len(images) - len(sel)))
        elif len(sel) > len(images):
            del sel[len(images):]

        batch = ttk.Frame(row); batch.pack(fill="x", padx=4, pady=4)
        for i, b in enumerate(images):
            col = ttk.Frame(batch); col.pack(side="left", padx=4)
            im = Image.open(io.BytesIO(b)); im.thumbnail((220, 220))
            tkimg = ImageTk.PhotoImage(im)
            lbl = ttk.Label(col, image=tkimg); lbl.image = tkimg; lbl.pack()
            var = tk.BooleanVar(value=c.sheet_selected[view_key][i])
            def _bind_toggle(iv=i, vk=view_key, nm=name, v=var):
                def _t():
                    self.characters[nm].sheet_selected[vk][iv] = v.get()
                return _t
            ttk.Checkbutton(col, text="Use", variable=var, command=_bind_toggle()).pack(anchor="center")
            ttk.Button(col, text="Delete", command=lambda nm=name, vk=view_key, iv=i: self._on_delete_char_image(nm, vk, iv)).pack(pady=(2,0))

    def _on_delete_char_image(self, name: str, view_key: str, index: int):
        c = self.characters.get(name)
        if not c: return
        imgs = c.sheet_images.get(view_key, [])
        sels = c.sheet_selected.get(view_key, [])
        if 0 <= index < len(imgs):
            del imgs[index]
            if 0 <= index < len(sels):
                del sels[index]
            self._refresh_char_view_thumbs(name, view_key)
            self._set_status("Deleted one image from " + name + " • " + view_key)

    def _on_save_selected_refs_char(self, name: str):
        c = self.characters[name]
        any_sel = any(any(flags) for flags in c.sheet_selected.values())
        if not any_sel:
            messagebox.showinfo("Save", "Tick at least one image to save as a reference.")
            return
        base_dir = os.path.join("assets","characters", sanitize_name(name))
        ensure_dir(base_dir)
        saved = 0
        for vkey, imgs in c.sheet_images.items():
            flags = c.sheet_selected.get(vkey, [])
            for i, b in enumerate(imgs):
                if i < len(flags) and flags[i]:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = sanitize_name(name) + "_" + vkey + "_" + str(i+1) + "_" + ts + ".png"
                    fpath = os.path.join(base_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(b)
                    aid = "img_" + sanitize_name(name) + "_" + vkey + "_" + str(i+1) + "_" + ts
                    self.assets.append(AssetRecord(
                        id=aid, entity_type="character", entity_name=name, view=vkey,
                        prompt_full=(c.sheet_base_prompt or ""), model=self.image_model, size=self.image_size,
                        file_path=fpath, created_at=now_iso(), notes="character sheet • " + vkey
                    ))
                    if aid not in c.reference_images:
                        c.reference_images.append(aid)
                    saved += 1
        messagebox.showinfo("Saved", str(saved) + " selected image(s) written to " + base_dir + " and linked to " + name + ".")

    def _on_load_character_json(self, name: str):
        p = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not p: return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Import", "Failed to read JSON:\n" + str(e)); return

        c = self.characters.get(name) or CharacterProfile(name=name, initial_description="")
        self.characters[name] = c

        profiles = []
        if isinstance(data, dict) and (data.get("type") == "character_profile"):
            profiles = [data]
        elif isinstance(data, list):
            profiles = [d for d in data if isinstance(d, dict) and (d.get("type") == "character_profile")]
        else:
            messagebox.showinfo("Import", "Unsupported character JSON structure.")
            return

        if not profiles:
            messagebox.showinfo("Import", "No character_profile objects found.")
            return

        d = profiles[0]
        c.initial_description = d.get("initial_description","") or c.initial_description
        c.refined_description = d.get("refined_description","") or c.refined_description
        c.role = d.get("role","") or c.role
        c.goals = d.get("goals","") or c.goals
        c.conflicts = d.get("conflicts","") or c.conflicts
        c.visual_cues_from_photos = d.get("visual_cues_from_photos","") or c.visual_cues_from_photos
        c.sheet_base_prompt = d.get("sheet_base_prompt","") or c.sheet_base_prompt

        imgs = d.get("images") or d.get("views") or {}
        for view_key, arr in imgs.items():
            if view_key not in CHAR_SHEET_VIEWS_DEF:
                continue
            for item in (arr or []):
                b = None
                if isinstance(item, dict):
                    b = decode_data_uri(item.get("data_uri",""))
                    if (b is None) and item.get("path"):
                        try:
                            with open(item["path"], "rb") as fh: b = fh.read()
                        except Exception:
                            b = None
                elif isinstance(item, str) and item.startswith("data:"):
                    b = decode_data_uri(item)
                if b:
                    c.sheet_images.setdefault(view_key, []).append(b)
                    c.sheet_selected.setdefault(view_key, []).append(False)

        self._rebuild_character_panels()
        self._set_status("Imported character JSON for " + name)

    def _on_export_character_folder(self, name: str):
        c = self.characters[name]
        outdir = filedialog.askdirectory(title="Choose a folder to export this character")
        if not outdir: return
        char_dir = os.path.join(outdir, sanitize_name(name))
        ensure_dir(char_dir)

        exported = []
        for vkey, imgs in c.sheet_images.items():
            flags = c.sheet_selected.get(vkey, [])
            for i, b in enumerate(imgs):
                if i < len(flags) and flags[i]:
                    fname = sanitize_name(name) + "_" + vkey + "_" + str(i+1) + ".png"
                    fpath = os.path.join(char_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(b)
                    exported.append({"view": vkey, "filename": fname, "path": fpath, "data_uri": b64_data_uri(b)})

        if not exported:
            messagebox.showinfo("Export", "Tick at least one image to export.")
            return

        payload = {
            "type": "character_profile",
            "name": c.name,
            "initial_description": c.initial_description,
            "refined_description": c.refined_description,
            "role": c.role,
            "goals": c.goals,
            "conflicts": c.conflicts,
            "visual_cues_from_photos": c.visual_cues_from_photos,
            "sheet_base_prompt": c.sheet_base_prompt,
            "images": {},
            "created_at": now_iso()
        }
        for item in exported:
            payload["images"].setdefault(item["view"], []).append({
                "filename": item["filename"],
                "path": os.path.relpath(item["path"], start=char_dir),
                "data_uri": item["data_uri"]
            })

        out_json = os.path.join(char_dir, sanitize_name(name) + ".json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # NEW: also mirror for batch under <picked>/characters/<Name>
        repo_root = self._repo_root_for(outdir)
        repo_json = self._export_character_profile_to(name, repo_root)
        repo_dir = os.path.dirname(repo_json)

        # Friendly message showing both destinations
        try:
            import os
            same = os.path.samefile(char_dir, repo_dir)
        except Exception:
            same = (os.path.abspath(char_dir) == os.path.abspath(repo_dir))

        if same:
            messagebox.showinfo("Export", "Character exported to:\n" + char_dir)
        else:
            messagebox.showinfo(
                "Export",
                "Character exported to:\n" + char_dir + "\n\nAlso mirrored for batch to:\n" + repo_dir
            )

    def _on_bulk_propose_char_baselines(self):
        if not self.client:
            self._on_connect()
            if not self.client: return
        todo = [n for n, p in self.char_panels.items() if p["select_var"].get()]
        if not todo:
            messagebox.showinfo("Characters", "Tick at least one character.")
            return
        for n in todo:
            self._on_propose_char_baseline(n)

    def _on_bulk_generate_char_views(self):
        todo = [n for n, p in self.char_panels.items() if p["select_var"].get()]
        if not todo:
            messagebox.showinfo("Characters", "Tick at least one character.")
            return
        for n in todo:
            self._on_generate_char_views(n)

    def _build_tab_locations(self):
        t = ttk.Frame(self.nb); self.nb.add(t, text="Locations")
        top = ttk.Frame(t, padding=10); top.pack(fill="both", expand=True)

        cmd = ttk.Frame(top); cmd.pack(fill="x")
        ttk.Button(cmd, text="Import Location JSON…", command=self._on_bulk_import_locations).pack(side="left")
        ttk.Label(cmd, text="Generate/import multiple angles; tick 'Use' then save/export.").pack(side="left", padx=10)
        # --- Aspect controls for Location ref sheets (new) ---
        aspect_row = ttk.Frame(top); aspect_row.pack(fill="x", pady=(6,0))
        ttk.Label(aspect_row, text="Aspect for location refs:").pack(side="left")
        self.loc_ref_aspect_var = tk.StringVar(value=self.loc_ref_aspect)
        for a in ASPECT_CHOICES:
            ttk.Radiobutton(aspect_row, text=a, value=a, variable=self.loc_ref_aspect_var).pack(side="left", padx=2)

        self.loc_scroll = ScrollableFrame(top); self.loc_scroll.pack(fill="both", expand=True, pady=6)
        self.loc_panels: Dict[str, Dict[str, Any]] = {}

    def _rebuild_location_panels(self):
        for w in self.loc_scroll.scrollable_frame.winfo_children(): w.destroy()
        self.loc_panels.clear()

        for name, l in self.locations.items():
            lf = ttk.Labelframe(self.loc_scroll.scrollable_frame, text="📍 " + name)
            lf.pack(fill="x", pady=8, padx=8)

            row0 = ttk.Frame(lf); row0.pack(fill="x", padx=6, pady=(2,2))
            ttk.Button(row0, text="Load JSON…", command=lambda n=name: self._on_load_location_json(n)).pack(side="left")
            ttk.Button(row0, text="Export Location Folder…", command=lambda n=name: self._on_export_location_folder(n)).pack(side="left", padx=6)

            ttk.Label(lf, text="Initial: " + (l.description or ""), wraplength=1100, justify="left").pack(anchor="w", padx=6, pady=(2,4))

            view_vars: Dict[str, tk.BooleanVar] = {}
            row = ttk.Frame(lf); row.pack(fill="x", padx=6, pady=(4,2))
            for vkey in ["establishing","alt_angle","detail"]:
                var = tk.BooleanVar(value=(vkey != "detail"))
                ttk.Checkbutton(row, text=LOC_VIEWS_DEF[vkey]["label"], variable=var).pack(side="left", padx=4)
                view_vars[vkey] = var

            ctrls = ttk.Frame(lf); ctrls.pack(fill="x", padx=6, pady=(2,4))
            ttk.Label(ctrls, text="Images per view:").pack(side="left")
            per_view_spin = ttk.Spinbox(ctrls, from_=1, to=3, increment=1, width=4); per_view_spin.set("1")
            per_view_spin.pack(side="left", padx=(4,12))

            ttk.Label(lf, text="Baseline (concise set identity; editable):").pack(anchor="w", padx=6)
            base_prompt_text = tk.Text(lf, height=3, width=140)
            seed = l.sheet_base_prompt or (l.description or name)
            base_prompt_text.insert("1.0", seed)
            base_prompt_text.pack(fill="x", padx=6, pady=(0,6))

            btns = ttk.Frame(lf); btns.pack(fill="x", padx=6, pady=(2,4))
            ttk.Button(btns, text="Propose baseline (GPT)", command=lambda n=name: self._on_propose_loc_baseline(n)).pack(side="left")
            ttk.Button(btns, text="Baseline from imported photos (GPT)", command=lambda n=name: self._on_baseline_from_imported_loc(n)).pack(side="left", padx=6)
            ttk.Button(btns, text="Generate selected views", command=lambda n=name: self._on_generate_loc_views(n)).pack(side="left", padx=8)
            ttk.Button(btns, text="Save selected refs to assets", command=lambda n=name: self._on_save_selected_refs_loc(n)).pack(side="left", padx=8)

            ref_frame = ttk.Labelframe(lf, text="Reference Images")
            ref_frame.pack(fill="x", padx=6, pady=(0,6))
            ref_frame.columnconfigure(0, weight=1)
            ref_frame.rowconfigure(0, weight=1)
            ref_list = tk.Listbox(ref_frame, height=6)
            ref_list.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
            ref_scroll = ttk.Scrollbar(ref_frame, orient="vertical", command=ref_list.yview)
            ref_scroll.grid(row=0, column=1, sticky="ns", pady=4)
            ref_list.configure(yscrollcommand=ref_scroll.set)
            thumb_canvas = tk.Canvas(ref_frame, width=176, height=176, highlightthickness=1, relief="solid")
            thumb_canvas.grid(row=0, column=2, rowspan=2, sticky="ne", padx=(8,4), pady=4)
            thumb_label = ttk.Label(ref_frame, text="", width=32)
            thumb_label.grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=(0,4))

            def _loc_add_files(nm=name):
                loc = self.locations.get(nm)
                if not loc:
                    messagebox.showinfo("References", "Select a location first.")
                    return
                paths = filedialog.askopenfilenames(
                    title=f"Add reference images for {nm}",
                    filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff")]
                )
                if not paths:
                    return
                added = self._attach_refs_to_profile(loc, list(paths))
                self._refresh_location_ref_list(nm)
                self._render_location_panel_thumb(nm)
                try:
                    self._set_status(f"Added {len(added)} location reference(s) to {nm}")
                except Exception:
                    pass

            def _loc_add_folder(nm=name):
                loc = self.locations.get(nm)
                if not loc:
                    messagebox.showinfo("References", "Select a location first.")
                    return
                folder = filedialog.askdirectory(title=f"Import folder for {nm}")
                if not folder:
                    return
                paths = [
                    p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                    if os.path.splitext(p)[1].lower() in VALID_IMAGE_EXTS and os.path.isfile(p)
                ]
                if not paths:
                    try:
                        messagebox.showinfo("References", "No compatible images found in that folder.")
                    except Exception:
                        pass
                    return
                added = self._attach_refs_to_profile(loc, paths)
                self._refresh_location_ref_list(nm)
                self._render_location_panel_thumb(nm)
                try:
                    self._set_status(f"Imported {len(added)} location image(s) for {nm}")
                except Exception:
                    pass

            def _loc_set_primary(nm=name):
                loc = self.locations.get(nm)
                if not loc:
                    return
                sel = ref_list.curselection()
                if not sel:
                    return
                if sel[0] < len(loc.reference_images):
                    loc.primary_reference_id = loc.reference_images[sel[0]]
                    entry = self.world.setdefault("locations", {}).setdefault(nm, {})
                    entry["primary_reference_id"] = loc.primary_reference_id
                    try:
                        if self.world_store_path:
                            self._save_world_store_to(self.world_store_path)
                    except Exception:
                        pass
                    self._refresh_location_ref_list(nm)
                    self._render_location_panel_thumb(nm)

            def _loc_remove_ref(nm=name):
                loc = self.locations.get(nm)
                if not loc:
                    return
                sel = ref_list.curselection()
                if not sel:
                    return
                if sel[0] >= len(loc.reference_images):
                    return
                rid = loc.reference_images[sel[0]]
                loc.reference_images = [x for x in loc.reference_images if x != rid]
                if loc.primary_reference_id == rid:
                    loc.primary_reference_id = loc.reference_images[0] if loc.reference_images else ""
                entry = self.world.setdefault("locations", {}).setdefault(nm, {})
                entry["reference_image_ids"] = list(dict.fromkeys(loc.reference_images))
                entry["primary_reference_id"] = loc.primary_reference_id
                try:
                    if self.world_store_path:
                        self._save_world_store_to(self.world_store_path)
                except Exception:
                    pass
                self._refresh_location_ref_list(nm)
                self._render_location_panel_thumb(nm)

            ref_btns = ttk.Frame(ref_frame)
            ref_btns.grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=(0,4))
            ttk.Button(ref_btns, text="Add Images…", command=_loc_add_files).pack(side="left", padx=(0,4))
            ttk.Button(ref_btns, text="Add Folder…", command=_loc_add_folder).pack(side="left", padx=(0,4))
            ttk.Button(ref_btns, text="Set Primary", command=_loc_set_primary).pack(side="left", padx=(0,4))
            ttk.Button(ref_btns, text="Remove", command=_loc_remove_ref).pack(side="left")

            ref_list.bind("<<ListboxSelect>>", lambda e, nm=name: self._render_location_panel_thumb(nm))

            thumb_rows: Dict[str, ttk.Frame] = {}
            for vkey in LOC_VIEWS_DEF.keys():
                r = ttk.Labelframe(lf, text=LOC_VIEWS_DEF[vkey]["label"])
                r.pack(fill="x", padx=6, pady=(4,4))
                head = ttk.Frame(r); head.pack(fill="x", padx=4, pady=(2,2))
                ttk.Button(head, text="Import…", command=lambda n=name, vk=vkey: self._on_import_loc_images(n, vk)).pack(side="left")
                area = ttk.Frame(r); area.pack(fill="x", padx=4, pady=(2,6))
                thumb_rows[vkey] = area

            self.loc_panels[name] = {
                "view_vars": view_vars,
                "per_view_spin": per_view_spin,
                "base_prompt_text": base_prompt_text,
                "thumb_rows": thumb_rows,
                "ref_list": ref_list,
                "thumb_canvas": thumb_canvas,
                "thumb_label": thumb_label,
            }

            self._refresh_location_ref_list(name)
            self._render_location_panel_thumb(name)
            for vkey, imgs in l.sheet_images.items():
                self._refresh_loc_view_thumbs(name, vkey)

    def _render_character_panel_thumb(self, name: str, primary_first: bool = True) -> None:
        panel = self.char_panels.get(name, {})
        canvas: Optional[tk.Canvas] = panel.get("thumb_canvas") if panel else None
        label: Optional[ttk.Label] = panel.get("thumb_label") if panel else None
        lb: Optional[tk.Listbox] = panel.get("ref_list") if panel else None
        if not canvas or not label:
            return
        canvas.delete("all")
        panel["thumb_image"] = None

        c_obj = self.characters.get(name)
        if not c_obj:
            label.config(text="")
            return

        pri = (c_obj.primary_reference_id or "").strip()
        ids = list(c_obj.reference_images or [])
        ordered = list(ids)
        if primary_first and pri and pri in ids:
            ordered = [pri] + [x for x in ids if x != pri]

        sel = lb.curselection() if lb else ()
        if sel and 0 <= sel[0] < len(ids):
            sel_id = ids[sel[0]]
            if sel_id in ordered:
                ordered = ([pri] if pri else []) + [sel_id] + [x for x in ids if x not in {pri, sel_id}]

        path = ""
        used_id = ""
        for rid in ordered:
            pth = self._asset_path_by_id(rid)
            if pth and os.path.isfile(pth):
                path = pth
                used_id = rid
                break

        if not path:
            label.config(text="(No valid image on disk)")
            return

        imtk, _ = self._load_thumb(path, max_side=168)
        if not imtk:
            label.config(text="(Failed to load image)")
            return

        try:
            cw = int(canvas.cget("width"))
            ch = int(canvas.cget("height"))
        except Exception:
            cw = ch = 176
        canvas.create_image(cw // 2, ch // 2, image=imtk)
        panel["thumb_image"] = imtk
        star = " ★" if used_id == pri and pri else ""
        label.config(text=f"{used_id}{star}")

    def _render_location_panel_thumb(self, name: str, primary_first: bool = True) -> None:
        panel = self.loc_panels.get(name, {})
        canvas: Optional[tk.Canvas] = panel.get("thumb_canvas") if panel else None
        label: Optional[ttk.Label] = panel.get("thumb_label") if panel else None
        lb: Optional[tk.Listbox] = panel.get("ref_list") if panel else None
        if not canvas or not label:
            return
        canvas.delete("all")
        panel["thumb_image"] = None

        loc = self.locations.get(name)
        if not loc:
            label.config(text="")
            return

        pri = (loc.primary_reference_id or "").strip()
        ids = list(loc.reference_images or [])
        ordered = list(ids)
        if primary_first and pri and pri in ids:
            ordered = [pri] + [x for x in ids if x != pri]

        sel = lb.curselection() if lb else ()
        if sel and 0 <= sel[0] < len(ids):
            sel_id = ids[sel[0]]
            if sel_id in ordered:
                ordered = ([pri] if pri else []) + [sel_id] + [x for x in ids if x not in {pri, sel_id}]

        path = ""
        used_id = ""
        for rid in ordered:
            pth = self._asset_path_by_id(rid)
            if pth and os.path.isfile(pth):
                path = pth
                used_id = rid
                break

        if not path:
            label.config(text="(No valid image on disk)")
            return

        imtk, _ = self._load_thumb(path, max_side=168)
        if not imtk:
            label.config(text="(Failed to load image)")
            return

        try:
            cw = int(canvas.cget("width"))
            ch = int(canvas.cget("height"))
        except Exception:
            cw = ch = 176
        canvas.create_image(cw // 2, ch // 2, image=imtk)
        panel["thumb_image"] = imtk
        star = " ★" if used_id == pri and pri else ""
        label.config(text=f"{used_id}{star}")

    def _refresh_character_ref_list(self, name: str) -> None:
        panel = self.char_panels.get(name, {})
        lb = panel.get("ref_list")
        if not lb:
            return
        lb.delete(0, tk.END)
        c = self.characters.get(name)
        if not c:
            return
        pri = c.primary_reference_id or ""
        for rid in c.reference_images or []:
            mark = " (primary)" if pri and rid == pri else ""
            lb.insert(tk.END, rid + mark)
        self._render_character_panel_thumb(name)

    def _refresh_location_ref_list(self, name: str) -> None:
        panel = self.loc_panels.get(name, {})
        lb = panel.get("ref_list")
        if not lb:
            return
        lb.delete(0, tk.END)
        loc = self.locations.get(name)
        if not loc:
            return
        pri = loc.primary_reference_id or ""
        for rid in loc.reference_images or []:
            mark = " (primary)" if pri and rid == pri else ""
            lb.insert(tk.END, rid + mark)
        self._render_location_panel_thumb(name)

    def _selected_loc_views(self, name: str) -> List[str]:
        vp = self.loc_panels.get(name, {}).get("view_vars", {})
        return [k for k, var in vp.items() if var.get()]

    def _images_per_view_loc(self, name: str) -> int:
        spin = self.loc_panels.get(name, {}).get("per_view_spin")
        try:
            return max(1, min(3, int(spin.get())))
        except Exception:
            return 1

    def _get_loc_baseline(self, name: str) -> str:
        l = self.locations[name]
        box = self.loc_panels[name]["base_prompt_text"]
        txt = box.get("1.0","end").strip()
        l.sheet_base_prompt = txt or l.sheet_base_prompt or l.description or name
        return l.sheet_base_prompt

    def _on_propose_loc_baseline(self, name: str):
        if not self.client:
            self._on_connect()
            if not self.client: return
        l = self.locations[name]
        baseline = LLM.propose_unified_location_prompt(
            self.client, self.llm_model,
            (self.analysis or {}).get("story_summary",""),
            l,
            extra_cues=l.visual_cues_from_photos
        ) or (l.sheet_base_prompt or l.description or name)
        # Inject current global style/preset details
        try:
            style_name = (self.selected_style_name or self.global_style or "").strip()
            style_bits = []
            try:
                style_bits = list(self._style_prompt_bits() or [])
            except Exception:
                style_bits = []
            lines = [baseline.strip()]
            if style_name and "global visual style:" not in baseline.lower():
                lines.append(f"Global visual style: {style_name}.")
            if style_bits:
                for bit in style_bits:
                    if bit and bit not in baseline:
                        lines.append(bit)
            baseline = "\n".join([ln for ln in lines if ln]).strip()
        except Exception:
            pass
        l.sheet_base_prompt = baseline
        box = self.loc_panels[name]["base_prompt_text"]
        box.delete("1.0","end"); box.insert("1.0", baseline)
        self._set_status("Location baseline proposed for " + name)

    def _collect_ticked_images_for_loc(self, name: str, max_count: int = 50) -> List[bytes]:
        l = self.locations[name]
        out: List[bytes] = []
        for vk in ["establishing","alt_angle","detail"]:
            imgs = l.sheet_images.get(vk, [])
            flags = l.sheet_selected.get(vk, [])
            for i, b in enumerate(imgs):
                if i < len(flags) and flags[i]:
                    out.append(b)
                    if len(out) >= max_count:
                        return out
        return out

    def _on_baseline_from_imported_loc(self, name: str):
        if not self.client:
            self._on_connect()
            if not self.client: return
        l = self.locations[name]
        imgs = self._collect_ticked_images_for_loc(name, max_count=4)
        if not imgs:
            messagebox.showinfo("Baseline from photos", "Tick ('Use') one or more images first, then try again.")
            return

        self._set_status("Analyzing selected location photos → baseline for " + name + "…")
        self.root.config(cursor="watch"); self.root.update_idletasks()
        try:
            cues_all = []
            for b in imgs:
                cues = LLM.extract_visual_cues_from_location_image(self.client, self.llm_model, b)
                if cues:
                    cues_all.append(cues)
            if cues_all:
                l.visual_cues_from_photos = " ".join(cues_all).strip()
            baseline = LLM.propose_unified_location_prompt(
                self.client, self.llm_model,
                (self.analysis or {}).get("story_summary",""),
                l,
                extra_cues=l.visual_cues_from_photos
            ) or (l.sheet_base_prompt or l.description or name)
            # Inject current global style/preset details
            try:
                style_name = (self.selected_style_name or self.global_style or "").strip()
                style_bits = []
                try:
                    style_bits = list(self._style_prompt_bits() or [])
                except Exception:
                    style_bits = []
                lines = [baseline.strip()]
                if style_name and "global visual style:" not in baseline.lower():
                    lines.append(f"Global visual style: {style_name}.")
                if style_bits:
                    for bit in style_bits:
                        if bit and bit not in baseline:
                            lines.append(bit)
                baseline = "\n".join([ln for ln in lines if ln]).strip()
            except Exception:
                pass
            l.sheet_base_prompt = baseline
            box = self.loc_panels[name]["base_prompt_text"]
            box.delete("1.0","end"); box.insert("1.0", baseline)
            self._set_status("Location baseline updated from selected photos for " + name + ".")
        except Exception as e:
            messagebox.showerror("Baseline from photos", str(e))
        finally:
            self.root.config(cursor="")

    def _on_generate_loc_views(self, name: str):
        if not self.client:
            self._on_connect()
            if not self.client: return
        l = self.locations[name]
        baseline = self._get_loc_baseline(name)
        views = self._selected_loc_views(name)
        if not views:
            messagebox.showinfo("Location", "Select at least one view.")
            return
        n = self._images_per_view_loc(name)
        size_to_use = self.aspect_to_size(self.loc_ref_aspect_var.get() if hasattr(self, "loc_ref_aspect_var") else self.loc_ref_aspect)
    
        prog = ProgressWindow(self.root, title=f"Generate views — {name}")
        prog.set_status("Starting…"); prog.set_progress(1)
    
        def worker():
            errs = []
            try:
                for idx, vkey in enumerate(views, 1):
                    prompt = build_loc_view_prompt_from_baseline(baseline, vkey, self.global_style)
                    def _tick():
                        prog.set_status(f"{vkey} ({idx}/{len(views)})…")
                        prog.set_progress(100.0 * (idx-1) / max(1,len(views)))
                    self.root.after(0, _tick)
                    imgs = self._try_images_generate(prompt, n, size=size_to_use)
                    processed_imgs = self._process_image_batch(imgs)
                    l.sheet_images.setdefault(vkey, []).extend(processed_imgs)
                    l.sheet_selected.setdefault(vkey, []).extend([False]*len(processed_imgs))
                    self.root.after(0, lambda vk=vkey: self._refresh_loc_view_thumbs(name, vk))
            except Exception as e:
                errs.append(str(e))
            finally:
                def _done():
                    prog.close()
                    if errs:
                        messagebox.showerror("Generate", "\n".join(errs))
                    else:
                        self._set_status(f"Generated location refs for {name}.")
                self.root.after(0, _done)
    
        import threading
        threading.Thread(target=worker, daemon=True).start()


    def _on_import_loc_images(self, name: str, view_key: str):
        paths = filedialog.askopenfilename(
            title="Import images for " + name + " — " + LOC_VIEWS_DEF[view_key]["label"],
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.webp")],
            multiple=True
        )
        if not paths:
            return
        l = self.locations[name]
        added: List[bytes] = []
        for p in paths:
            try:
                with open(p, "rb") as f:
                    added.append(f.read())
            except Exception as e:
                print("Import loc error:", e)
        if not added: return
        l.sheet_images.setdefault(view_key, []).extend(added)
        l.sheet_selected.setdefault(view_key, [])
        l.sheet_selected[view_key].extend([False]*len(added))
        self._refresh_loc_view_thumbs(name, view_key)
        self._set_status("Imported " + str(len(added)) + " image(s) to " + name + " • " + LOC_VIEWS_DEF[view_key]['label'] + ".")

    def _refresh_loc_view_thumbs(self, name: str, view_key: str):
        row = self.loc_panels[name]["thumb_rows"][view_key]
        for w in row.winfo_children():
            w.destroy()
        l = self.locations[name]
        images = l.sheet_images.get(view_key, []) or []
        l.sheet_selected.setdefault(view_key, [])
        sel = l.sheet_selected[view_key]
        if len(sel) < len(images):
            sel.extend([False] * (len(images) - len(sel)))
        elif len(sel) > len(images):
            del sel[len(images):]

        batch = ttk.Frame(row); batch.pack(fill="x", padx=4, pady=4)
        for i, b in enumerate(images):
            col = ttk.Frame(batch); col.pack(side="left", padx=4)
            im = Image.open(io.BytesIO(b)); im.thumbnail((220, 220))
            tkimg = ImageTk.PhotoImage(im)
            lbl = ttk.Label(col, image=tkimg); lbl.image = tkimg; lbl.pack()
            var = tk.BooleanVar(value=l.sheet_selected[view_key][i])
            def _bind_toggle(iv=i, vk=view_key, nm=name, v=var):
                def _t():
                    self.locations[nm].sheet_selected[vk][iv] = v.get()
                return _t
            ttk.Checkbutton(col, text="Use", variable=var, command=_bind_toggle()).pack(anchor="center")
            ttk.Button(col, text="Delete", command=lambda nm=name, vk=view_key, iv=i: self._on_delete_loc_image(nm, vk, iv)).pack(pady=(2,0))

    def _on_delete_loc_image(self, name: str, view_key: str, index: int):
        l = self.locations.get(name)
        if not l: return
        imgs = l.sheet_images.get(view_key, [])
        sels = l.sheet_selected.get(view_key, [])
        if 0 <= index < len(imgs):
            del imgs[index]
            if 0 <= index < len(sels):
                del sels[index]
            self._refresh_loc_view_thumbs(name, view_key)
            self._set_status("Deleted one image from " + name + " • " + view_key)

    def _on_save_selected_refs_loc(self, name: str):
        l = self.locations[name]
        any_sel = any(any(flags) for flags in l.sheet_selected.values())
        if not any_sel:
            messagebox.showinfo("Save", "Tick at least one image to save as a reference.")
            return
        base_dir = os.path.join("assets","locations", sanitize_name(name))
        ensure_dir(base_dir)
        saved = 0
        for vkey, imgs in l.sheet_images.items():
            flags = l.sheet_selected.get(vkey, [])
            for i, b in enumerate(imgs):
                if i < len(flags) and flags[i]:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = sanitize_name(name) + "_" + vkey + "_" + str(i+1) + "_" + ts + ".png"
                    fpath = os.path.join(base_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(b)
                    aid = "img_" + sanitize_name(name) + "_" + vkey + "_" + str(i+1) + "_" + ts
                    self.assets.append(AssetRecord(
                        id=aid, entity_type="location", entity_name=name, view=vkey,
                        prompt_full=(l.sheet_base_prompt or ""), model=self.image_model, size=self.image_size,
                        file_path=fpath, created_at=now_iso(), notes="location ref • " + vkey
                    ))
                    if aid not in l.reference_images:
                        l.reference_images.append(aid)
                    saved += 1
        messagebox.showinfo("Saved", str(saved) + " selected image(s) written to " + base_dir + " and linked to " + name + ".")
############################################
    # --- Silent utilities: detect, save, export, import (no UI) -----------------
    
    def _entity_has_any_refs(self, entity_type: str, name: str) -> bool:
        """Return True if we already have refs for this entity (in memory, assets, or on disk)."""
        if entity_type == "character":
            c = self.characters.get(name)
            if c and (c.reference_images or any((c.sheet_images or {}).values())):
                return True
        else:
            l = self.locations.get(name)
            if l and (l.reference_images or any((l.sheet_images or {}).values())):
                return True
        # in-memory assets list
        for a in (self.assets or []):
            if a.entity_type == entity_type and a.entity_name == name:
                return True
        # disk check (assets/<type>/<name_slug>/*.png|jpg|webp)
        base = os.path.join("assets", "characters" if entity_type == "character" else "locations", sanitize_name(name))
        if os.path.isdir(base):
            for fn in os.listdir(base):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    return True
        return False

    def _repo_root_for(self, picked_dir: str) -> str:
        """Normalize a user-chosen directory to a repo root.
        If they picked .../characters or .../locations directly, use its parent to avoid nesting."""
        import os

        try:
            base = os.path.basename(os.path.normpath(picked_dir)).lower()
            if base in ("characters", "locations"):
                return os.path.dirname(os.path.normpath(picked_dir))
        except Exception:
            pass
        return picked_dir

    def _save_selected_refs_char_silent(self, name: str) -> int:
        """Save currently 'selected' character images to assets and link to profile (no dialogs)."""
        c = self.characters[name]
        base_dir = os.path.join("assets", "characters", sanitize_name(name))
        ensure_dir(base_dir)
        saved = 0
        for vkey, imgs in (c.sheet_images or {}).items():
            flags = (c.sheet_selected or {}).get(vkey, [])
            for i, b in enumerate(imgs):
                if i < len(flags) and flags[i]:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"{sanitize_name(name)}_{vkey}_{i+1}_{ts}.png"
                    fpath = os.path.join(base_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(b)
                    aid = "img_" + sanitize_name(name) + "_" + vkey + "_" + str(i+1) + "_" + ts
                    self.assets.append(AssetRecord(
                        id=aid, entity_type="character", entity_name=name, view=vkey,
                        prompt_full=(c.sheet_base_prompt or ""), model=self.image_model, size=self.image_size,
                        file_path=fpath, created_at=now_iso(), notes="character sheet • " + vkey
                    ))
                    if aid not in c.reference_images:
                        c.reference_images.append(aid)
                    saved += 1
        return saved
    
    def _save_selected_refs_loc_silent(self, name: str) -> int:
        """Save currently 'selected' location images to assets and link to profile (no dialogs)."""
        l = self.locations[name]
        base_dir = os.path.join("assets", "locations", sanitize_name(name))
        ensure_dir(base_dir)
        saved = 0
        for vkey, imgs in (l.sheet_images or {}).items():
            flags = (l.sheet_selected or {}).get(vkey, [])
            for i, b in enumerate(imgs):
                if i < len(flags) and flags[i]:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"{sanitize_name(name)}_{vkey}_{i+1}_{ts}.png"
                    fpath = os.path.join(base_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(b)
                    aid = "img_" + sanitize_name(name) + "_" + vkey + "_" + str(i+1) + "_" + ts
                    self.assets.append(AssetRecord(
                        id=aid, entity_type="location", entity_name=name, view=vkey,
                        prompt_full=(l.sheet_base_prompt or ""), model=self.image_model, size=self.image_size,
                        file_path=fpath, created_at=now_iso(), notes="location ref • " + vkey
                    ))
                    if aid not in l.reference_images:
                        l.reference_images.append(aid)
                    saved += 1
        return saved
    
    def _export_character_profile_to(self, name: str, library_root: str) -> str:
        """
        Write a character_profile folder with JSON and copies of selected images (no dialogs).
        Returns path to the written JSON.
        """
        c = self.characters[name]
        root = os.path.join(library_root, "characters", sanitize_name(name))
        ensure_dir(root)
        exported = []
        for vkey, imgs in (c.sheet_images or {}).items():
            flags = (c.sheet_selected or {}).get(vkey, [])
            for i, b in enumerate(imgs):
                if i < len(flags) and flags[i]:
                    fname = f"{sanitize_name(name)}_{vkey}_{i+1}.png"
                    fpath = os.path.join(root, fname)
                    with open(fpath, "wb") as f:
                        f.write(b)
                    exported.append({"view": vkey, "filename": fname, "path": fpath, "data_uri": b64_data_uri(b)})
    
        payload = {
            "type": "character_profile",
            "name": c.name,
            "initial_description": c.initial_description,
            "refined_description": c.refined_description,
            "role": c.role, "goals": c.goals, "conflicts": c.conflicts,
            "visual_cues_from_photos": c.visual_cues_from_photos,
            "sheet_base_prompt": c.sheet_base_prompt,
            "images": {},
            "created_at": now_iso()
        }
        for item in exported:
            payload["images"].setdefault(item["view"], []).append({
                "filename": item["filename"],
                "path": os.path.relpath(item["path"], start=root),
                "data_uri": item["data_uri"]
            })
        out_json = os.path.join(root, f"{sanitize_name(name)}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_json
    
    def _export_location_profile_to(self, name: str, library_root: str) -> str:
        """Write a location_profile folder with JSON and copies of selected images (no dialogs)."""
        l = self.locations[name]
        root = os.path.join(library_root, "locations", sanitize_name(name))
        ensure_dir(root)
        exported = []
        for vkey, imgs in (l.sheet_images or {}).items():
            flags = (l.sheet_selected or {}).get(vkey, [])
            for i, b in enumerate(imgs):
                if i < len(flags) and flags[i]:
                    fname = f"{sanitize_name(name)}_{vkey}_{i+1}.png"
                    fpath = os.path.join(root, fname)
                    with open(fpath, "wb") as f:
                        f.write(b)
                    exported.append({"view": vkey, "filename": fname, "path": fpath, "data_uri": b64_data_uri(b)})
    
        payload = {
            "type": "location_profile",
            "name": l.name,
            "description": l.description,
            "mood": l.mood, "lighting": l.lighting,
            "key_props": [p.strip() for p in (l.key_props or "").split(",") if p.strip()],
            "visual_cues_from_photos": l.visual_cues_from_photos,
            "sheet_base_prompt": l.sheet_base_prompt,
            "images": {},
            "created_at": now_iso()
        }
        for item in exported:
            payload["images"].setdefault(item["view"], []).append({
                "filename": item["filename"],
                "path": os.path.relpath(item["path"], start=root),
                "data_uri": item["data_uri"]
            })
        out_json = os.path.join(root, f"{sanitize_name(name)}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_json
    
    def _import_character_json_from_path(self, name: str, path: str) -> None:
        """Programmatic version of Load Character JSON (no dialogs)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        c = self.characters.get(name) or CharacterProfile(name=name, initial_description="")
        self.characters[name] = c
        profiles = []
        if isinstance(data, dict) and data.get("type") == "character_profile":
            profiles = [data]
        elif isinstance(data, list):
            profiles = [d for d in data if isinstance(d, dict) and d.get("type") == "character_profile"]
        if not profiles:
            return
        d = profiles[0]
        c.initial_description = d.get("initial_description","") or c.initial_description
        c.refined_description = d.get("refined_description","") or c.refined_description
        c.role = d.get("role","") or c.role
        c.goals = d.get("goals","") or c.goals
        c.conflicts = d.get("conflicts","") or c.conflicts
        c.visual_cues_from_photos = d.get("visual_cues_from_photos","") or c.visual_cues_from_photos
        c.sheet_base_prompt = d.get("sheet_base_prompt","") or c.sheet_base_prompt
        c.sheet_images = c.sheet_images or {}
        c.sheet_selected = c.sheet_selected or {}
        imgs = d.get("images") or d.get("views") or {}
        for view_key, arr in imgs.items():
            for item in (arr or []):
                b = None
                if isinstance(item, dict):
                    if item.get("data_uri"):
                        b = decode_data_uri(item["data_uri"])
                    elif item.get("path"):
                        try:
                            with open(os.path.join(os.path.dirname(path), item["path"]), "rb") as fh:
                                b = fh.read()
                        except Exception:
                            b = None
                elif isinstance(item, str) and item.startswith("data:"):
                    b = decode_data_uri(item)
                if b:
                    c.sheet_images.setdefault(view_key, []).append(b)
                    c.sheet_selected.setdefault(view_key, []).append(False)
    
    def _import_location_json_from_path(self, name: str, path: str) -> None:
        """Programmatic version of Load Location JSON (no dialogs)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        l = self.locations.get(name) or LocationProfile(name=name, description="")
        self.locations[name] = l
        profiles = []
        if isinstance(data, dict) and data.get("type") == "location_profile":
            profiles = [data]
        elif isinstance(data, list):
            profiles = [d for d in data if isinstance(d, dict) and d.get("type") == "location_profile"]
        if not profiles:
            return
        d = profiles[0]
        l.description = d.get("description","") or l.description
        l.mood = d.get("mood","") or l.mood
        l.lighting = d.get("lighting","") or l.lighting
        kp = d.get("key_props", [])
        if isinstance(kp, list):
            l.key_props = ", ".join(kp)
        l.visual_cues_from_photos = d.get("visual_cues_from_photos","") or l.visual_cues_from_photos
        l.sheet_base_prompt = d.get("sheet_base_prompt","") or l.sheet_base_prompt
        l.sheet_images = l.sheet_images or {}
        l.sheet_selected = l.sheet_selected or {}
        imgs = d.get("images") or d.get("views") or {}
        for view_key, arr in imgs.items():
            for item in (arr or []):
                b = None
                if isinstance(item, dict):
                    if item.get("data_uri"):
                        b = decode_data_uri(item["data_uri"])
                    elif item.get("path"):
                        try:
                            with open(os.path.join(os.path.dirname(path), item["path"]), "rb") as fh:
                                b = fh.read()
                        except Exception:
                            b = None
                elif isinstance(item, str) and item.startswith("data:"):
                    b = decode_data_uri(item)
                if b:
                    l.sheet_images.setdefault(view_key, []).append(b)
                    l.sheet_selected.setdefault(view_key, []).append(False)
    
    def _select_first_per_view(self, entity_type: str, name: str, n: int = 1):
        """Mark first n images per view as selected (useful after importing a profile)."""
        if entity_type == "character":
            obj = self.characters[name]
        else:
            obj = self.locations[name]
        for vkey, imgs in (obj.sheet_images or {}).items():
            sel = obj.sheet_selected.setdefault(vkey, [])
            # Ensure sel length
            if len(sel) < len(imgs):
                sel.extend([False] * (len(imgs) - len(sel)))
            # Flip first n
            up_to = min(n, len(imgs))
            for i in range(up_to):
                sel[i] = True
    
    # --- Autogeneration when a profile is missing -------------------------------
    
    def _autogen_character_profile_if_missing(
        self, name: str, profiles_dir: str, views: list[str] | None = None,
        per_view: int = 1, aspect: str | None = None
    ) -> str | None:
        if self._entity_has_any_refs("character", name):
            return None
        if not self.client:
            self._on_connect()
            if not self.client:
                return None
        c = self.characters[name]
        # Baseline
        try:
            if not (c.sheet_base_prompt or "").strip():
                c.sheet_base_prompt = LLM.propose_unified_character_prompt(
                    self.client, self.llm_model,
                    (self.analysis or {}).get("story_summary",""), c,
                    extra_cues=c.visual_cues_from_photos
                ) or default_baseline_prompt(c)
        except Exception:
            c.sheet_base_prompt = c.sheet_base_prompt or default_baseline_prompt(c)
        views = views or ["front", "profile_left", "profile_right"]
        size = self.aspect_to_size((aspect or getattr(self, "char_ref_aspect", "1:1")))
        # Generate + select
        for vkey in views:
            prompt = build_view_prompt_from_baseline(c.sheet_base_prompt, vkey, self.global_style)
            imgs = self._try_images_generate(prompt, n=int(max(1, per_view)), size=size)
            if imgs:
                processed_imgs = self._process_image_batch(imgs)
                c.sheet_images.setdefault(vkey, []).extend(processed_imgs)
                c.sheet_selected.setdefault(vkey, []).extend([True] * len(processed_imgs))
        # Save to assets and export a profile JSON
        self._save_selected_refs_char_silent(name)
        return self._export_character_profile_to(name, profiles_dir)
    
    def _autogen_location_profile_if_missing(
        self, name: str, profiles_dir: str, views: list[str] | None = None,
        per_view: int = 1, aspect: str | None = None
    ) -> str | None:
        if self._entity_has_any_refs("location", name):
            return None
        if not self.client:
            self._on_connect()
            if not self.client:
                return None
        l = self.locations[name]
        # Baseline
        try:
            if not (l.sheet_base_prompt or "").strip():
                l.sheet_base_prompt = LLM.propose_unified_location_prompt(
                    self.client, self.llm_model,
                    (self.analysis or {}).get("story_summary",""), l,
                    extra_cues=l.visual_cues_from_photos
                ) or (l.description or name)
        except Exception:
            l.sheet_base_prompt = l.sheet_base_prompt or (l.description or name)
        views = views or ["establishing", "alt_angle", "detail"]
        size = self.aspect_to_size((aspect or getattr(self, "loc_ref_aspect", "3:2")))
        # Generate + select
        for vkey in views:
            prompt = build_loc_view_prompt_from_baseline(l.sheet_base_prompt, vkey, self.global_style)
            imgs = self._try_images_generate(prompt, n=int(max(1, per_view)), size=size)
            if imgs:
                processed_imgs = self._process_image_batch(imgs)
                l.sheet_images.setdefault(vkey, []).extend(processed_imgs)
                l.sheet_selected.setdefault(vkey, []).extend([True] * len(processed_imgs))
        # Save to assets and export a profile JSON
        self._save_selected_refs_loc_silent(name)
        return self._export_location_profile_to(name, profiles_dir)

#################################################
    def _on_bulk_import_locations(self):
        p = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not p: return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Import", "Failed to read JSON:\n" + str(e)); return

        imported = 0
        profiles = []
        if isinstance(data, dict) and (data.get("type") == "location_profile"):
            profiles = [data]
        elif isinstance(data, list):
            profiles = [d for d in data if isinstance(d, dict) and (d.get("type") == "location_profile")]
        else:
            messagebox.showinfo("Import", "Unsupported JSON structure (expect location_profile objects).")
            return

        for d in profiles:
            nm = d.get("name","") or "Location"
            if nm not in self.locations:
                self.locations[nm] = LocationProfile(name=nm, description="")
            l = self.locations[nm]
            l.description = d.get("description","") or l.description
            l.mood = d.get("mood","") or l.mood
            l.lighting = d.get("lighting","") or l.lighting
            kp = d.get("key_props", [])
            if isinstance(kp, list):
                l.key_props = ", ".join(kp)
            l.visual_cues_from_photos = d.get("visual_cues_from_photos","") or l.visual_cues_from_photos
            l.sheet_base_prompt = d.get("sheet_base_prompt","") or l.sheet_base_prompt

            imgs = d.get("images") or d.get("views") or {}
            for view_key, arr in imgs.items():
                if view_key not in LOC_VIEWS_DEF:
                    continue
                for item in (arr or []):
                    b = None
                    if isinstance(item, dict):
                        b = decode_data_uri(item.get("data_uri",""))
                        if (b is None) and item.get("path"):
                            try:
                                with open(item["path"], "rb") as fh: b = fh.read()
                            except Exception:
                                b = None
                    elif isinstance(item, str) and item.startswith("data:"):
                        b = decode_data_uri(item)
                    if b:
                        l.sheet_images.setdefault(view_key, []).append(b)
                        l.sheet_selected.setdefault(view_key, []).append(False)
            imported += 1

        self._rebuild_location_panels()
        self._set_status("Imported " + str(imported) + " location(s).")

    def _on_load_location_json(self, name: str):
        p = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not p: return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Import", "Failed to read JSON:\n" + str(e)); return

        l = self.locations.get(name) or LocationProfile(name=name, description="")
        self.locations[name] = l

        profiles = []
        if isinstance(data, dict) and (data.get("type") == "location_profile"):
            profiles = [data]
        elif isinstance(data, list):
            profiles = [d for d in data if isinstance(d, dict) and (d.get("type") == "location_profile")]
        else:
            messagebox.showinfo("Import", "Unsupported location JSON structure.")
            return

        if not profiles:
            messagebox.showinfo("Import", "No location_profile objects found.")
            return

        d = profiles[0]
        l.description = d.get("description","") or l.description
        l.mood = d.get("mood","") or l.mood
        l.lighting = d.get("lighting","") or l.lighting
        kp = d.get("key_props", [])
        if isinstance(kp, list):
            l.key_props = ", ".join(kp)
        l.visual_cues_from_photos = d.get("visual_cues_from_photos","") or l.visual_cues_from_photos
        l.sheet_base_prompt = d.get("sheet_base_prompt","") or l.sheet_base_prompt

        imgs = d.get("images") or d.get("views") or {}
        for view_key, arr in imgs.items():
            if view_key not in LOC_VIEWS_DEF:
                continue
            for item in (arr or []):
                b = None
                if isinstance(item, dict):
                    b = decode_data_uri(item.get("data_uri",""))
                    if (b is None) and item.get("path"):
                        try:
                            with open(item["path"], "rb") as fh: b = fh.read()
                        except Exception:
                            b = None
                elif isinstance(item, str) and item.startswith("data:"):
                    b = decode_data_uri(item)
                if b:
                    l.sheet_images.setdefault(view_key, []).append(b)
                    l.sheet_selected.setdefault(view_key, []).append(False)

        self._rebuild_location_panels()
        self._set_status("Imported location JSON for " + name)

    def _on_export_location_folder(self, name: str):
        l = self.locations[name]
        outdir = filedialog.askdirectory(title="Choose a folder to export this location")
        if not outdir: return
        loc_dir = os.path.join(outdir, sanitize_name(name))
        ensure_dir(loc_dir)

        exported = []
        for vkey, imgs in l.sheet_images.items():
            flags = l.sheet_selected.get(vkey, [])
            for i, b in enumerate(imgs):
                if i < len(flags) and flags[i]:
                    fname = sanitize_name(name) + "_" + vkey + "_" + str(i+1) + ".png"
                    fpath = os.path.join(loc_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(b)
                    exported.append({"view": vkey, "filename": fname, "path": fpath, "data_uri": b64_data_uri(b)})

        if not exported:
            messagebox.showinfo("Export", "Tick at least one image to export.")
            return

        payload = {
            "type": "location_profile",
            "name": l.name,
            "description": l.description,
            "mood": l.mood,
            "lighting": l.lighting,
            "key_props": [p.strip() for p in (l.key_props or "").split(",") if p.strip()],
            "visual_cues_from_photos": l.visual_cues_from_photos,
            "sheet_base_prompt": l.sheet_base_prompt,
            "images": {},
            "created_at": now_iso()
        }
        for item in exported:
            payload["images"].setdefault(item["view"], []).append({
                "filename": item["filename"],
                "path": os.path.relpath(item["path"], start=loc_dir),
                "data_uri": item["data_uri"]
            })

        out_json = os.path.join(loc_dir, sanitize_name(name) + ".json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # NEW: also mirror for batch under <picked>/locations/<Name>
        repo_root = self._repo_root_for(outdir)
        repo_json = self._export_location_profile_to(name, repo_root)
        repo_dir = os.path.dirname(repo_json)

        try:
            import os
            same = os.path.samefile(loc_dir, repo_dir)
        except Exception:
            same = (os.path.abspath(loc_dir) == os.path.abspath(repo_dir))

        if same:
            messagebox.showinfo("Export", "Location exported to:\n" + loc_dir)
        else:
            messagebox.showinfo(
                "Export",
                "Location exported to:\n" + loc_dir + "\n\nAlso mirrored for batch to:\n" + repo_dir
            )

    def _build_tab_shots_export(self):
        t = ttk.Frame(self.nb); self.nb.add(t, text="Shots & Export")
    
        top = ttk.Frame(t, padding=10); top.pack(fill="x")
        ttk.Button(top, text="Generate Shot Prompts", command=self._on_generate_shots).pack(side="left")
        ttk.Button(top, text="Export Scene Folders…", command=self._on_export_scene_jsons).pack(side="left", padx=10)
    
        # --- Render controls (per-shot count, prompt source, delay, aspect, coverage, grouping) ---
        ctrl = ttk.Frame(t); ctrl.pack(fill="x", pady=(0,6))
    
        ttk.Label(ctrl, text="Per-shot:").pack(side="left")
        self.render_n_spin = ttk.Spinbox(ctrl, from_=1, to=6, width=4); self.render_n_spin.set("1")
        self.render_n_spin.pack(side="left", padx=(4,12))
    
        ttk.Label(ctrl, text="Prompt source:").pack(side="left")
        self.prompt_source_combo = ttk.Combobox(
            ctrl,
            values=["auto (shot→fused→final)", "shot prompts", "fused_prompt", "final_prompt"],
            state="readonly", width=26
        )
        self.prompt_source_combo.set("auto (shot→fused→final)")
        self.prompt_source_combo.pack(side="left", padx=(4,12))
    
        ttk.Label(ctrl, text="Delay (s):").pack(side="left")
        self.render_delay_spin = ttk.Spinbox(ctrl, from_=0, to=60, width=4); self.render_delay_spin.set("1")
        self.render_delay_spin.pack(side="left", padx=(4,12))
    
        ttk.Label(ctrl, text="Aspect:").pack(side="left")
        self.scene_render_aspect_var = tk.StringVar(value=self.scene_render_aspect)
        cmb_aspect = ttk.Combobox(ctrl, values=ASPECT_CHOICES, state="readonly", width=8, textvariable=self.scene_render_aspect_var)
        cmb_aspect.pack(side="left", padx=(4,12))
        cmb_aspect.bind("<<ComboboxSelected>>", lambda e: setattr(self, "scene_render_aspect", self.scene_render_aspect_var.get()))

        # Extra images (word-gap) control
        ttk.Label(ctrl, text="Min words / image:").pack(side="left")
        self.min_words_between_images_var = tk.StringVar(value=str(self.min_words_between_images))
        ttk.Entry(ctrl, width=6, textvariable=self.min_words_between_images_var).pack(side="left", padx=(4,12))
    
        # NEW: coverage choice (so "Maximum" works from this tab too)
        cov_box = ttk.Frame(ctrl); cov_box.pack(side="left", padx=(4,12))
        ttk.Label(cov_box, text="Final image coverage:").pack(side="left")
        self.scene_coverage_mode_var = getattr(self, "scene_coverage_mode_var", tk.StringVar(value="min"))
        ttk.Radiobutton(cov_box, text="Primary only", value="min", variable=self.scene_coverage_mode_var).pack(side="left", padx=3)
        ttk.Radiobutton(cov_box, text="Maximum (all shots)", value="max", variable=self.scene_coverage_mode_var).pack(side="left", padx=3)
    
        # NEW: grouping toggle (per-shot subfolders after render)
        self.group_renders_by_shot_var = getattr(self, "group_renders_by_shot_var", tk.BooleanVar(value=True))
        ttk.Checkbutton(ctrl, text="Group renders by shot", variable=self.group_renders_by_shot_var).pack(side="left", padx=(4,12))
    
        ttk.Button(ctrl, text="Render from Scene JSON(s)…", command=self._on_render_from_scene_jsons).pack(side="left", padx=6)
        ttk.Button(ctrl, text="Render Scene Folder…", command=self._on_render_from_scene_folder).pack(side="left", padx=6)
    
        self.shots_scroll = ScrollableFrame(t); self.shots_scroll.pack(fill="both", expand=True, pady=8)
        self.shot_panels: Dict[str, Dict[str, Any]] = {}


    # ------- Extra images by word-gap: planning + prompt mutation -------

    def _count_words(self, text: str) -> int:
        import re
        return len(re.findall(r"\b\w+\b", text or ""))


    def _excerpt_text(self, text: str, min_words: int, frac: float) -> str:
        """
        Pick a compact excerpt around a fractional position in the text.
        frac in [0..1]. Ensures ~min_words words (±).
        """
        import re, math
        words = re.findall(r"\S+", text or "")
        if not words:
            return ""
        total = len(words)
        # aim: a chunk roughly >= min_words but not longer than ~1.4x
        chunk = max(min_words, min(total, int(math.ceil(min_words * 1.1))))
        center = max(0, min(total - 1, int(total * max(0.0, min(1.0, frac)))))
        start = max(0, min(center - chunk // 2, total - chunk))
        end = min(total, start + chunk)
        return " ".join(words[start:end]).strip()

    def _ensure_sentence_anchor(self, text: str) -> str:
        """Ensure the anchor text ends on a sentence boundary."""
        import re

        raw = re.sub(r"\s+", " ", (text or "").strip())
        if not raw:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", raw)
        collected: List[str] = []
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            collected.append(s)
            if re.search(r"[.!?][\"'\)]*$", s):
                break
        if collected:
            return " ".join(collected).strip()
        return raw



    def _pick_variant_label(self, scene: dict, index: int) -> str:
        try:
            # Gather source-specific variants (these names already exist upstream)
            char_variants = []
            env_variants = []
            try:
                # If scene has profiles/cast/environment fields, derive buckets (keep existing helpers if present)
                if hasattr(self, "_character_variant_lines"):
                    char_variants = list(self._character_variant_lines(scene)) or []
            except Exception:
                char_variants = []
            try:
                if hasattr(self, "_environment_variant_lines"):
                    env_variants = list(self._environment_variant_lines(scene)) or []
            except Exception:
                env_variants = []

            # Fallback to previous inline generation if helper buckets are empty
            if not char_variants:
                try:
                    def _norm_list(items):
                        return [i.strip() for i in items if isinstance(i, str) and i.strip()]

                    chars = _norm_list((scene or {}).get("characters_present") or [])
                    if len(chars) >= 2:
                        pov_index = max(0, index - 1)
                        pov_char = chars[pov_index % len(chars)]
                        opposite_char = chars[(pov_index + 1) % len(chars)]
                        char_variants = [
                            f"from {pov_char}'s eye‑line, framing {opposite_char} in midground",
                            f"tight beside {pov_char}, tracking their hands interacting with props",
                            f"shoulder‑level over {pov_char}, background crowd defocused",
                        ]
                    elif len(chars) == 1:
                        solo = chars[0]
                        char_variants = [
                            f"dramatic low three‑quarter on {solo}, environment looming",
                            f"profile silhouette of {solo} against the environment lighting",
                            f"macro detail on {solo}'s hands/gear with the setting blurred",
                        ]
                except Exception:
                    char_variants = []

            if not env_variants:
                try:
                    env_variants = [
                        "bird's‑eye plan view that maps out the scene geography",
                        "wide architectural establishing shot with characters as small accents",
                        "reflection in glossy surface or window while action plays beyond",
                        "wide cinematic dolly from corridor/doorway threshold",
                        "low prow/ground‑hugging tracking shot emphasizing depth",
                    ]
                except Exception:
                    env_variants = []

            # Fallback/wildcards to keep variety even when cast/setting are sparse
            wildcard_variants: list[str] = [
                "wide reverse angle from the far side of the space",
                "low angle from floor level to exaggerate scale",
                "long-lens compression from the back of the room",
                "wide lens very close to foreground geometry for parallax",
                "reflected viewpoint off glass/metal surfaces",
                "over-the-shoulder from the secondary character",
                "high angle from overhead gantry or balcony",
                "oblique angle through doorway or bulkhead framing",
            ]

            # Build a unique pool while preserving order
            variant_pool: list[str] = []
            for bucket in (char_variants, env_variants, wildcard_variants):
                if not bucket:
                    continue
                for item in bucket:
                    item = (item or "").strip()
                    if item and item not in variant_pool:
                        variant_pool.append(item)

            # Absolute last-chance fallback
            if not variant_pool:
                return "alternate angle — shift camera position for fresh coverage"

            # Deterministic pick tied to index to avoid repeats across extras
            pick = variant_pool[index % len(variant_pool)]
            return pick

        except Exception:
            # Never let extras fail due to variant selection
            return "alternate angle — change lensing or camera height"

        except Exception:
            # Never let extras fail due to variant selection
            return "alternate angle — change lensing or camera height"

    def _mutate_prompt_for_extra_image(self, base_prompt: str, variant: str, excerpt: str, scene: dict) -> str:
        """
        Derive a new prompt from an existing shot prompt by changing perspective and
        weaving in the currently narrated text excerpt.
        """
        aspect = getattr(self, "scene_render_aspect", DEFAULT_ASPECT)
        txt = (excerpt or "").strip()
        hair_guard = self._hair_guard_sentence(scene) if hasattr(self, "_hair_guard_sentence") else ""
        ref_guard = self._reference_guard_sentence(scene) if hasattr(self, "_reference_guard_sentence") else ""
        identity_bits = ["Maintain each character's face geometry, eye color, and hair exactly as established."]

        if hair_guard:
            identity_bits.append(hair_guard)
        if ref_guard:
            identity_bits.append(ref_guard)

        identity_sentence = " ".join(identity_bits).strip()
        if identity_sentence and not identity_sentence.endswith("."):
            identity_sentence += "."
        set_sentence = "Keep continuity of set architecture, props, and lighting while embracing the new angle."
        continuity_clause = " ".join([s for s in [identity_sentence, set_sentence] if s]).strip()

        if base_prompt:
            return (
                base_prompt.strip().rstrip(".")
                + f". Alternate coverage — shift the camera to {variant}. "
                  f"Integrate narrative detail: {txt}. "

            ).strip()
        # Fallback composition if no base prompt available
        loc = scene.get("location", "")
        chars = ", ".join(scene.get("characters_present") or [])
        return (
            f"{loc}: {chars}. {txt}. Alternate coverage — {variant}. "
            f"Cinematic composition with layered depth and motivated lighting. "
            f"{continuity_clause} "
            f"Avoid brand names; no text, no watermark. Aspect {aspect}."
        ).strip()


    def _analysis_for_export(self) -> dict:
        if isinstance(self.analysis, dict):
            try:
                return {
                    k: v
                    for k, v in self.analysis.items()

                    if k not in ("_story_text_cache", "_story_word_count_cache")

                }
            except Exception:
                return dict(self.analysis)
        return self.analysis or {}


    def build_dialogue_cues(self, story_text: str) -> List[DialogueCue]:
        """Extract dialogue via the hybrid rule-based engine and map to DialogueCue objects."""

        text = story_text or ""
        if not text.strip():
            self._dialogue_last_utterances = []
            self._dialogue_last_metadata = []
            self._dialogue_last_known_characters = []
            self._dialogue_last_aliases = {}
            return []

        analysis_obj = getattr(self, "analysis", {})
        analysis = analysis_obj if isinstance(analysis_obj, dict) else {}

        def _collect_names() -> List[str]:
            names: List[str] = []
            seen: set = set()

            def _push(value: Optional[str]) -> None:
                if not value:
                    return
                key = _title_case_name(value)
                if key and key not in seen:
                    seen.add(key)
                    names.append(key)

            for key in ("characters", "main_characters", "character_list", "cast", "dramatis_personae"):
                val = analysis.get(key)
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            _push(item.get("name") or item.get("character"))
                        elif isinstance(item, str):
                            _push(item)

            scenes = analysis.get("scenes")
            if isinstance(scenes, list):
                for scene in scenes:
                    if isinstance(scene, dict):
                        chars = scene.get("characters_present") or scene.get("characters")
                        if isinstance(chars, list):
                            for char in chars:
                                _push(char)
            return names

        known_characters = _collect_names()
        alias_map: Dict[str, List[str]] = {}
        alias_data = analysis.get("character_aliases")
        if isinstance(alias_data, dict):
            for base, aliases in alias_data.items():
                if not isinstance(base, str):
                    continue
                clean_base = _title_case_name(base)
                collected: List[str] = []
                if isinstance(aliases, list):
                    for alias in aliases:
                        if isinstance(alias, str) and alias.strip():
                            collected.append(alias.strip())
                alias_map[clean_base] = collected
                if clean_base and clean_base not in known_characters:
                    known_characters.append(clean_base)

        extractor_mode = getattr(self, "dialogue_extraction_mode", "strict")
        confidence_threshold = float(getattr(self, "dialogue_confidence_threshold", 0.90))
        max_narrator_chars = getattr(self, "dialogue_max_narrator_chars", None)
        if max_narrator_chars is not None:
            try:
                max_narrator_chars = int(max_narrator_chars)
            except Exception:
                max_narrator_chars = None

        extractor = DialogueExtractor(
            known_characters=known_characters or None,
            aliases=alias_map or None,
            mode=extractor_mode,
            confidence_threshold=confidence_threshold,
            max_narrator_chars=max_narrator_chars,
        )
        utterances = extractor.extract(text)

        self._dialogue_last_utterances = copy.deepcopy(utterances)
        self._dialogue_last_metadata = copy.deepcopy(utterances)
        self._dialogue_last_known_characters = list(known_characters)
        self._dialogue_last_aliases = dict(alias_map)

        cues: List[DialogueCue] = []
        for order, item in enumerate(utterances, start=1):
            line = (item.get("line") or "").strip()
            if not line:
                continue
            speaker = item.get("character") or "UNATTRIBUTED"
            emotion, emotion_conf = _find_inline_emotion(line)
            score = float((item.get("attribution") or {}).get("score") or 0.0)
            speaker_conf = 1.0 if speaker == "Narrator" else (score if score > 0 else 0.5)
            cues.append(
                DialogueCue(
                    order=order,
                    speaker=speaker,
                    text=line,
                    emotion=emotion,
                    speaker_conf=speaker_conf,
                    emotion_conf=emotion_conf,
                )
            )

        return cues

    def emit_marked_text(self, story_text: str, cues: List[DialogueCue]) -> str:
        """Return a simple "Character: line" transcript from DialogueCue records."""
        lines: List[str] = []
        for cue in cues:
            speaker = cue.speaker or "Narrator"
            text = (cue.text or "").strip()
            if not text:
                continue
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines).strip() + ("\n" if lines else "")

    def save_dialogue_artifacts(self, source_text_path: str, out_dir: str, cues: List[DialogueCue]) -> Dict[str, str]:
        """Persist the dialogue sidecars using the most recent extractor output."""
        src = Path(source_text_path) if source_text_path else None
        base = src.stem if src else "story"
        outp = Path(out_dir or (src.parent if src else Path.cwd()))
        outp.mkdir(parents=True, exist_ok=True)

        story_text = getattr(self, "_dialogue_story_text_cache", "")
        if not story_text:
            story_text = getattr(self, "_last_story_text", "")
        if not story_text:
            story_text = ""

        known_chars = getattr(self, "_dialogue_last_known_characters", [])
        alias_map = getattr(self, "_dialogue_last_aliases", {})
        utterances = copy.deepcopy(getattr(self, "_dialogue_last_utterances", []))
        if not utterances and story_text:
            max_chars = getattr(self, "dialogue_max_narrator_chars", None)
            if max_chars is not None:
                try:
                    max_chars = int(max_chars)
                except Exception:
                    max_chars = None
            extractor = DialogueExtractor(
                known_characters=known_chars or None,
                aliases=alias_map or None,
                mode=getattr(self, "dialogue_extraction_mode", "strict"),
                confidence_threshold=float(getattr(self, "dialogue_confidence_threshold", 0.90)),
                max_narrator_chars=max_chars,
            )
            utterances = extractor.extract(story_text)

        llm_enabled = bool(getattr(self, "enable_dialogue_llm", False))
        llm_model = getattr(self, "dialogue_llm_model", None)
        llm_threshold = float(getattr(self, "dialogue_llm_threshold", 0.92))
        llm_batch = int(getattr(self, "dialogue_llm_batch_size", 8))
        if llm_enabled and story_text and utterances:
            _apply_llm_assist(story_text, utterances, known_chars, alias_map, llm_threshold, llm_batch)

        voices_map = getattr(self, "dialogue_voices_map", None)
        if not isinstance(voices_map, dict):
            voices_map = None

        result = _write_sidecars(
            str(outp / base),
            utterances,
            voices_map,
            llm_enabled,
            llm_model,
            llm_threshold,
        )
        self._dialogue_last_metadata = copy.deepcopy(utterances)
        return {"json": result["json_path"], "marked": result["txt_path"]}


    def _inject_extra_scenes_by_word_gap(self, min_words: int) -> int:
        """
        NEW IMPLEMENTATION:
        Measure the actual words in the story text between consecutive scene markers.
        For each base scene S_i, consider the story segment from its marker to the next marker
        (first scene also includes any preface). Insert evenly spaced "extra" scenes so that
        the average density is at least `min_words` words per image, capped by
        EXTRA_IMAGES_MAX_PER_SCENE.
    
        Returns: number of extra scenes created.
        """
        import math
    

        self._scene_story_segments = {}

        if not self.analysis or not isinstance(self.analysis.get("scenes"), list):
            return 0

        try:
            min_words = int(max(1, min_words or 0))
        except Exception:
            min_words = 1

    
        scenes_in = list(self.analysis.get("scenes") or [])
        if not scenes_in:
            return 0
    
        # Separate base scenes from any previously inserted extras
        base_scenes = [s for s in scenes_in if isinstance(s, dict)]
        existing_extras_by_base: dict[str, list[dict]] = {}
        for s in base_scenes:
            if not isinstance(s, dict):
                continue
            meta = s.get("meta") or {}
            if meta.get("is_auto_extra") and meta.get("source_scene_id"):
                existing_extras_by_base.setdefault(meta.get("source_scene_id"), []).append(s)

        anchors = [s for s in base_scenes if not ((s.get("meta") or {}).get("is_auto_extra"))]

        # --- Build per-base-scene story segments using your marker builder ---
        # Prefer a stored copy of the story to avoid UI access off the main thread.

        cached_story = ""
        cached_word_total = 0
        if isinstance(self.analysis, dict):
            cached_story = (self.analysis.get("_story_text_cache") or "")
            try:
                cached_word_total = int(self.analysis.get("_story_word_count_cache") or 0)
            except Exception:
                cached_word_total = 0

        story_text = ""
        try:
            story_text = getattr(self, "_last_story_text", "") or ""
        except Exception:
            story_text = ""

        if not story_text and cached_story:
            story_text = cached_story
        if not story_text:
            try:
                if getattr(self, "story_text", None):

                    # Fallback only if available; may be called from a worker thread.

                    story_text = self.story_text.get("1.0", "end").strip()
            except Exception:
                story_text = ""
        if not story_text and cached_story:
            story_text = cached_story
        if story_text and not getattr(self, "_last_story_text", ""):
            try:
                self._last_story_text = story_text
            except Exception:
                pass

        segment_by_sid: dict[str, str] = {}

        story_word_total = self._count_words(story_text) if story_text else 0
        if story_word_total <= 0 and cached_word_total:
            story_word_total = int(cached_word_total)



        def _slice_story_evenly(anchor_scenes, text):
            matches = list(re.finditer(r"\S+", text or ""))
            total_tokens = len(matches)
            if total_tokens <= 0:
                return {}
            valid_ids = []
            for sc in anchor_scenes:
                sid = (sc.get("id") or "").strip()
                if sid:
                    valid_ids.append(sid)
            count = len(valid_ids)
            if count <= 0:
                return {}
            boundaries: List[int] = []
            for i in range(count + 1):
                val = int(round(i * total_tokens / count))
                if boundaries:
                    val = max(val, boundaries[-1])
                val = min(val, total_tokens)
                boundaries.append(val)
            if boundaries:
                boundaries[-1] = total_tokens
            segments: Dict[str, str] = {}
            for idx, sid in enumerate(valid_ids):
                start_idx = boundaries[idx]
                end_idx = boundaries[idx + 1]
                if end_idx <= start_idx or start_idx >= total_tokens:
                    continue
                start_char = 0 if start_idx == 0 else matches[start_idx].start()
                end_char = len(text) if end_idx >= total_tokens else matches[end_idx].start()
                seg_text = text[start_char:end_char].strip()
                if seg_text:
                    segments[sid] = seg_text
            return segments

        if story_text and anchors:
            try:
                txt_with_markers, markers = self._build_marked_story_and_index(story_text, anchors)
                lines = txt_with_markers.splitlines()


                # sid -> 1-based marker line number

                mline = {}
                for m in (markers or []):
                    sid = (m.get("scene_id") or "").strip()
                    ml = m.get("marker_line_number")
                    if sid and isinstance(ml, int):
                        mline[sid] = ml

    
                ordered = anchors[:]
    
                # Preface (text before the first marker) goes to the first scene

                preface_text = ""
                if ordered:
                    first_sid = (ordered[0].get("id") or "").strip()
                    first_ml = mline.get(first_sid)
                    if isinstance(first_ml, int) and first_ml > 1:
                        preface_text = "\n".join(lines[:max(0, first_ml - 1)])

                for i, sc in enumerate(ordered):
                    sid = (sc.get("id") or "").strip()
                    if not sid:
                        continue
                    this_ml = mline.get(sid)

    
                    # find next scene with a valid marker

                    nxt_ml = None
                    for k in range(i + 1, len(ordered)):
                        ml = mline.get((ordered[k].get("id") or "").strip())
                        if isinstance(ml, int):
                            nxt_ml = ml
                            break

    
                    seg_parts = []
                    if i == 0 and preface_text:
                        seg_parts.append(preface_text)
    
                    if isinstance(this_ml, int):
                        # [this_ml + 1 .. nxt_ml - 1], 1-based indices in lines[]
                        start = min(len(lines), max(1, this_ml + 1))
                    else:
                        start = None  # no anchor for this scene; will fall back later
    
                    end = len(lines) if nxt_ml is None else max(1, nxt_ml - 1)
    
                    if start is not None:
                        chunk = "\n".join(lines[start - 1 : end])
                        seg_parts.append(chunk)
    

                    text_seg = "\n".join([p for p in seg_parts if p]).strip()
                    if text_seg:
                        segment_by_sid[sid] = text_seg
            except Exception:

                # If anything goes wrong, we fall back to per-scene summaries/fractional slices below.

                segment_by_sid = {}

        if story_text and anchors:
            missing_segment_ids = [
                (sc.get("id") or "").strip()
                for sc in anchors
                if (sc.get("id") or "").strip() and not (segment_by_sid.get((sc.get("id") or "").strip()) or "").strip()
            ]
            if missing_segment_ids:
                fallback_segments = _slice_story_evenly(anchors, story_text)
                for sid, txt in fallback_segments.items():
                    if (txt or "").strip():
                        segment_by_sid[sid] = txt
                for sid, txt in list(segment_by_sid.items()):
                    if not (txt or "").strip():
                        segment_by_sid.pop(sid, None)



            # If the marker-based slices barely cover the story text, fall back to even spacing.

            if story_word_total > 0 and segment_by_sid:
                seg_word_total = sum(self._count_words(txt) for txt in segment_by_sid.values())
                coverage_ratio = seg_word_total / float(story_word_total)
                if coverage_ratio < 0.6:
                    even_segments = _slice_story_evenly(anchors, story_text)
                    if even_segments:
                        segment_by_sid = {
                            sid: txt for sid, txt in even_segments.items() if (txt or "").strip()
                        }
                        seg_word_total = sum(self._count_words(txt) for txt in segment_by_sid.values())
                        coverage_ratio = seg_word_total / float(story_word_total) if story_word_total else 1.0
                        try:
                            print(
                                "[extras] coverage fallback: using evenly sliced story text "
                                f"({coverage_ratio:.0%} of {story_word_total} words)"
                            )
                        except Exception:
                            pass




        # Fallback for any scene missing a segment: use its summary (legacy behavior)
        for sc in base_scenes:
            sid = (sc.get("id") or "").strip()
            if sid and sid not in segment_by_sid and not ((sc.get("meta") or {}).get("is_auto_extra")):
                segment_by_sid[sid] = (sc.get("what_happens") or sc.get("description") or "").strip()

        base_segments_for_anchor: Dict[str, str] = {}
        for sc in base_scenes:
            if not isinstance(sc, dict):
                continue

            sid = (sc.get("id") or "").strip()
            if not sid:
                continue
            txt = (segment_by_sid.get(sid) or "").strip()
            if txt:
                base_segments_for_anchor[sid] = txt


        if base_segments_for_anchor:
            self._scene_story_segments = dict(base_segments_for_anchor)


        new_scenes: list[dict] = []
        created = 0
        seen_extra_ids = {
            s.get("id", "")
            for s in base_scenes
            if (s.get("meta") or {}).get("is_auto_extra")
        }

        # Insert extras immediately after their base scenes
        for sc in base_scenes:
            if not isinstance(sc, dict):
                new_scenes.append(sc)
                continue

            sid = (sc.get("id") or "").strip()
            if not sid:
                new_scenes.append(sc)
                continue

    
            # Always keep the original (even if it was an extra)
            new_scenes.append(sc)
    
            # Skip: we only generate extras for base scenes
            if (sc.get("meta") or {}).get("is_auto_extra"):
                continue
    
            text = (segment_by_sid.get(sid) or "").strip()
            total_words = self._count_words(text)
            if total_words <= 0:
                continue
    

            needed_images = max(1, math.ceil(total_words / float(min_words)))
            extras_target = max(0, needed_images - 1)
            if extras_target > EXTRA_IMAGES_MAX_PER_SCENE:
                extras_target = min(EXTRA_IMAGES_ABS_MAX_PER_SCENE, extras_target)
            extras_needed = extras_target

    
            if extras_needed <= 0:
                continue
    
            # Don't duplicate existing extras linked to this base scene
            existing_for_base = existing_extras_by_base.get(sid, [])
            to_make = max(0, extras_needed - len(existing_for_base))
            if to_make <= 0:
                continue
    
            # Base prompt from primary shot

            base_prompt = ""
            try:
                prim = self._choose_primary_shot(sid)
                if prim and prim.prompt:
                    base_prompt = prim.prompt
            except Exception:
                base_prompt = ""

            # Evenly spaced extras across the story segment
            for j in range(len(existing_for_base) + 1, len(existing_for_base) + to_make + 1):
                new_id = f"{sid}x{j}"
                if new_id in seen_extra_ids:
                    continue

                # pick an excerpt around the j/(extras_needed+1) position of the segment
                frac = (j / (extras_needed + 1)) if extras_needed > 0 else 0.5
                excerpt = self._excerpt_text(text, min_words, frac=frac)
                variant = self._pick_variant_label(sc, j)

    
                ns = {
                    "id": new_id,
                    "title": (sc.get("title") or sid) + f" — extra image {j}",
                    "what_happens": excerpt or (sc.get("what_happens") or sc.get("description") or ""),

                    "description": sc.get("description", ""),
                    "characters_present": list(sc.get("characters_present") or []),
                    "location": sc.get("location", ""),
                    "key_actions": list(sc.get("key_actions") or []),

                    "tone": sc.get("tone", ""),
                    "time_of_day": sc.get("time_of_day", ""),
                    "movement_id": sc.get("movement_id", ""),
                    "beat_type": sc.get("beat_type", ""),

                    "meta": {
                        **(sc.get("meta") or {}),
                        "is_auto_extra": True,
                        "source_scene_id": sid,
                        "variant": variant
                    }

                }
                new_scenes.append(ns)
                try:
                    self.scenes_by_id[new_id] = ns
                except Exception:
                    pass

    
                prompt = self._mutate_prompt_for_extra_image(base_prompt, variant, excerpt, sc)
                sh_id = "sh_" + hash_str(new_id + variant + prompt)[:10]
                try:

                    self.shots.append(ShotPrompt(
                        id=sh_id,
                        scene_id=new_id,
                        title=ns["title"],

                        shot_description=f"Auto extra: {variant}",
                        prompt=prompt,
                        continuity_notes="auto extra image (word-gap by story markers)"

                    ))
                except Exception:
                    pass


                seen_extra_ids.add(new_id)
                if excerpt:
                    try:
                        self._scene_story_segments[new_id] = excerpt

                    except Exception:
                        pass
                created += 1

        self.analysis["scenes"] = new_scenes
        return created


    def _current_scene_anchor_map(self) -> Dict[str, str]:
        anchors: Dict[str, str] = {}
        segments = getattr(self, "_scene_story_segments", None)
        if isinstance(segments, dict):
            for sid, txt in segments.items():
                sid_norm = (sid or "").strip()
                if sid_norm and (txt or "").strip():
                    anchors[sid_norm] = txt.strip()

        if isinstance(self.analysis, dict):
            for sc in (self.analysis.get("scenes") or []):
                if not isinstance(sc, dict):
                    continue
                sid = (sc.get("id") or "").strip()
                if not sid:
                    continue
                if sid not in anchors or not anchors[sid].strip():
                    txt = (sc.get("what_happens") or sc.get("description") or "").strip()
                    if txt:
                        anchors[sid] = txt
        return anchors



    def _auto_export_scene_jsons_sync(self, outdir: str, coverage_mode: str = "min") -> None:
        """
        Build enriched scene JSONs (no image generation).
        NEW: 'coverage_mode' controls how many shots get written into each scene:
             - 'min' → one shot (primary) per scene
             - 'max' → all available shot prompts for the scene
        """
        ensure_dir(outdir)
        self.output_dir = outdir
        self._last_export_dir = outdir
        self._scene_story_segments = {}

        story_text = getattr(self, "_last_story_text", "") or ""
        if not story_text:
            story_text = getattr(self, "_dialogue_story_text_cache", "") or ""
        if not story_text and isinstance(self.analysis, dict):
            try:
                story_text = (self.analysis.get("_story_text_cache") or "").strip()
            except Exception:
                story_text = ""
        if not story_text and hasattr(self, "story_text"):
            try:
                story_text = self.story_text.get("1.0", "end").strip()
            except Exception:
                story_text = ""
        if story_text:
            try:
                self._dialogue_story_text_cache = story_text
            except Exception:
                pass

        # --- Auto-insert extra scenes based on word-gap threshold ---
        try:
            # If the Shots & Export UI field exists, prefer its current value
            if (
                threading.current_thread() is threading.main_thread()
                and hasattr(self, "min_words_between_images_var")
            ):
                self.min_words_between_images = int(self.min_words_between_images_var.get() or "0")
        except Exception:
            pass
        try:
            mw = int(getattr(self, "min_words_between_images", 0) or 0)
        except Exception:
            mw = 0
        if mw and mw > 0:
            try:
                created = self._inject_extra_scenes_by_word_gap(min_words=mw)
                if created:
                    print(f"[extras] inserted {created} extra scene(s) based on {mw} words/image")
            except Exception as _e:
                print("extras planning failed:", _e)

        anchor_map = self._current_scene_anchor_map()
        try:
            self._story_anchor_map_by_outdir[os.path.abspath(outdir)] = dict(anchor_map)
        except Exception:
            self._story_anchor_map_by_outdir[os.path.abspath(outdir)] = anchor_map

        titles_map: Dict[str, str] = {}
        story_title = ""
        if isinstance(self.analysis, dict):
            story_title = (self.analysis.get("title") or "").strip()
            for sc in (self.analysis.get("scenes") or []):
                if not isinstance(sc, dict):
                    continue
                sid = (sc.get("id") or "").strip()
                if not sid:
                    continue
                title_txt = (sc.get("title") or sid).strip()
                titles_map[sid] = title_txt or sid
        abs_outdir = os.path.abspath(outdir)
        self._story_scene_titles_by_outdir[abs_outdir] = titles_map
        if story_title:
            self._story_title_by_outdir[abs_outdir] = story_title

        scenes_dir = os.path.join(outdir, SCENE_SUBDIR_NAME)

        ensure_dir(scenes_dir)
    
        # Persist analysis at top level
        if WRITE_ANALYSIS_FILE:
            try:
                with open(os.path.join(outdir, ANALYSIS_FILENAME), "w", encoding="utf-8") as f:
                    json.dump(self._analysis_for_export(), f, indent=2, ensure_ascii=False)
            except Exception:
                pass
    
        # Map of shot prompts (include any edits if a UI is present)
        shot_prompt_map: Dict[str, str] = {}
        for s in (self.shots or []):
            txt = None
            try:
                panel = self.shot_panels.get(s.id) if hasattr(self, "shot_panels") else None
                if panel and "prompt_text" in panel:
                    txt = panel["prompt_text"].get("1.0", "end").strip()
            except Exception:
                txt = None
            shot_prompt_map[s.id] = (txt or s.prompt or "").strip()
    
        errors: List[str] = []
        cov = (coverage_mode or "min").strip().lower()
        for sc in (self.analysis.get("scenes", []) if self.analysis else []):
            sid = sc.get("id", "").strip()
            if not sid:
                continue
            try:
                scene_dir = os.path.join(scenes_dir, sanitize_name(sid))
                ensure_dir(scene_dir)
                ensure_dir(os.path.join(scene_dir, SCENE_REFS_DIR))  # make refs/ upfront
    
                # Choose primary shot & prompt
                primary_shot = self._choose_primary_shot(sid)
                primary_prompt = ""
                if primary_shot:
                    primary_prompt = (shot_prompt_map.get(primary_shot.id, primary_shot.prompt) or "").strip()
                if not primary_prompt:
                    primary_prompt = (sc.get("what_happens") or sc.get("description") or "").strip()
    
                # Build world & conditioning (keeps refs externalized)
                world = self._build_scene_world_enriched(sc, primary_shot, primary_prompt, shot_prompt_map)
                final_world = self._apply_conditioning_with_budget(world, sc, primary_shot, primary_prompt)
    
                # Externalize refs to disk
                if ALWAYS_EXTERNALIZE_IMAGES:
                    final_world = self._externalize_images_to_disk(final_world, scene_dir)
    
                # Optional backlink to _analysis.json
                if WRITE_ANALYSIS_FILE:
                    try:
                        rel_analysis = relpath_posix(os.path.join(outdir, ANALYSIS_FILENAME), start=scene_dir)
                        final_world.setdefault("source", {})["analysis_file"] = rel_analysis
                    except Exception:
                        pass
    
                # Compose ingredients + fused prompt
                ingredients = self._build_prompt_ingredients(final_world, sc)
                fused = self._compose_final_generation_prompt(ingredients)
                used_boost = False
                if USE_LLM_FUSION and self.client:
                    try:
                        #boosted = LLM.fuse_scene_prompt(self.client, self.llm_model, ingredients, self.global_style, NEGATIVE_TERMS)
                        boosted = LLM.fuse_scene_prompt(self.client, self.llm_model, ingredients, self.global_style, NEGATIVE_TERMS, self.scene_render_aspect)
                        if boosted:
                            fused = boosted
                            used_boost = True
                    except Exception:
                        pass
                style_bits = self._style_prompt_bits()
                if style_bits:
                    if used_boost:
                        base = (fused or "").strip()
                        parts = [base] if base else []
                        parts.extend(style_bits)
                        fused = "\n\n".join(parts)
                    elif not any(bit in (fused or "") for bit in style_bits):
                        fused = "\n\n".join([fused] + style_bits) if fused else "\n\n".join(style_bits)
                final_world["scene"]["ingredients"] = ingredients
                final_world["scene"]["fused_prompt"] = fused

                # NEW: write shots according to coverage mode
                shot_entries: List[Dict[str, Any]] = []
                if cov == "max":
                    for sh in (self.shots or []):
                        if sh.scene_id == sid:
                            ptxt = (shot_prompt_map.get(sh.id) or sh.prompt or "").strip()
                            if ptxt:
                                entry = {"id": sh.id, "title": sh.title, "prompt": ptxt}
                                if sh.continuity_notes:
                                    entry["notes"] = sh.continuity_notes

                                shot_entries.append(entry)
                    if not shot_entries and primary_prompt:
                        # Fallback: at least one shot
                        shot_entries = [{"id": (primary_shot.id if primary_shot else f"{sid}-A"),
                                         "title": (primary_shot.title if primary_shot else "Primary"),
                                         "prompt": primary_prompt}]
                else:  # 'min'
                    if primary_prompt:

                        shot_entries = [{"id": (primary_shot.id if primary_shot else f"{sid}-A"),
                                         "title": (primary_shot.title if primary_shot else "Primary"),
                                         "prompt": primary_prompt}]


                final_world["scene"]["shots"] = shot_entries

                # Trim & write
                final_world = self._trim_world_for_size(final_world, max_mb=MAX_SCENE_JSON_MB)
                out_json = os.path.join(scene_dir, f"{sid}.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(final_world, f, indent=2, ensure_ascii=False)

            except Exception as e:
                errors.append(f"{sid}: {e}")

        # Captions + world.json
        try:
            self._write_captions_todo(outdir, self.analysis.get("scenes", []), shot_prompt_map)
        except Exception as me:
            errors.append("captions_todo: " + str(me))
        try:
            self._write_captions_map(outdir, self.analysis.get("scenes", []), shot_prompt_map)
        except Exception as me:
            errors.append("captions_map: " + str(me))
        try:
            self._world_json_update_from_current(outdir)
        except Exception as we:
            errors.append("world.json: " + str(we))

        try:
            self.analyze_and_emit_dialogue(
                out_dir=outdir,
                text=story_text,
                source_text_path=getattr(self, "_last_story_path", "") or getattr(self, "input_text_path", ""),
            )
        except Exception as exc:
            print(f"[DIALOGUE] analysis/emit failed: {exc}")

        if errors:
            messagebox.showerror("Export finished with errors", "Some scenes failed:\n- " + "\n- ".join(errors))
        else:
            messagebox.showinfo("Export", "Export completed to:\n" + outdir)



    def analyze_and_emit_dialogue(self, **kw) -> Dict[str, str]:
        """
        Entry point you can call after your main analysis step.
        Expects self.story_text and self.output_dir (or kw overrides).
        """
        text = kw.get("text") or getattr(self, "_last_story_text", "") or ""
        if not text and hasattr(self, "story_text"):
            try:
                text = self.story_text.get("1.0", "end").strip()
            except Exception:
                text = ""
        if not text:
            cache = (self.analysis or {}).get("_story_text_cache") if isinstance(self.analysis, dict) else ""
            text = cache or ""
        if not text.strip():
            print("[DIALOGUE] No story text available; skipping.")
            return {}

        outd = kw.get("out_dir") or getattr(self, "_last_export_dir", "") or getattr(self, "output_dir", "")
        if not outd:
            world_path = getattr(self, "world_store_path", "") or ""
            if world_path:
                outd = os.path.dirname(world_path)
        if not outd:
            outd = os.getcwd()

        srcp = kw.get("source_text_path") or getattr(self, "_last_story_path", "") or getattr(self, "input_text_path", "")

        self._dialogue_story_text_cache = text
        cues = self.build_dialogue_cues(text)
        try:
            return self.save_dialogue_artifacts(srcp, outd, cues)
        finally:
            self._dialogue_story_text_cache = ""


    def _inject_extra_scenes_by_word_gap(self, min_words: int) -> int:
        """
        NEW IMPLEMENTATION:
        Measure the actual words in the story text between consecutive scene markers.
        For each base scene S_i, consider the story segment from its marker to the next marker
        (first scene also includes any preface). Insert evenly spaced "extra" scenes so that
        the average density is at least `min_words` words per image, capped by
        EXTRA_IMAGES_MAX_PER_SCENE.
    
        Returns: number of extra scenes created.
        """
        import math
    

        self._scene_story_segments = {}

        if not self.analysis or not isinstance(self.analysis.get("scenes"), list):
            return 0

        try:
            min_words = int(max(1, min_words or 0))
        except Exception:
            min_words = 1

    
        scenes_in = list(self.analysis.get("scenes") or [])
        if not scenes_in:
            return 0
    
        # Separate base scenes from any previously inserted extras
        base_scenes = [s for s in scenes_in if isinstance(s, dict)]
        existing_extras_by_base: dict[str, list[dict]] = {}
        for s in base_scenes:
            if not isinstance(s, dict):
                continue
            meta = s.get("meta") or {}
            if meta.get("is_auto_extra") and meta.get("source_scene_id"):
                existing_extras_by_base.setdefault(meta.get("source_scene_id"), []).append(s)

        anchors = [s for s in base_scenes if not ((s.get("meta") or {}).get("is_auto_extra"))]

        # --- Build per-base-scene story segments using your marker builder ---
        # Prefer a stored copy of the story to avoid UI access off the main thread.

        cached_story = ""
        cached_word_total = 0
        if isinstance(self.analysis, dict):
            cached_story = (self.analysis.get("_story_text_cache") or "")
            try:
                cached_word_total = int(self.analysis.get("_story_word_count_cache") or 0)
            except Exception:
                cached_word_total = 0

        story_text = ""
        try:
            story_text = getattr(self, "_last_story_text", "") or ""
        except Exception:
            story_text = ""

        if not story_text and cached_story:
            story_text = cached_story
        if not story_text:
            try:
                if getattr(self, "story_text", None):

                    # Fallback only if available; may be called from a worker thread.

                    story_text = self.story_text.get("1.0", "end").strip()
            except Exception:
                story_text = ""
        if not story_text and cached_story:
            story_text = cached_story
        if story_text and not getattr(self, "_last_story_text", ""):
            try:
                self._last_story_text = story_text
            except Exception:
                pass

        segment_by_sid: dict[str, str] = {}

        story_word_total = self._count_words(story_text) if story_text else 0
        if story_word_total <= 0 and cached_word_total:
            story_word_total = int(cached_word_total)



        def _slice_story_evenly(anchor_scenes, text):
            matches = list(re.finditer(r"\S+", text or ""))
            total_tokens = len(matches)
            if total_tokens <= 0:
                return {}
            valid_ids = []
            for sc in anchor_scenes:
                sid = (sc.get("id") or "").strip()
                if sid:
                    valid_ids.append(sid)
            count = len(valid_ids)
            if count <= 0:
                return {}
            boundaries: List[int] = []
            for i in range(count + 1):
                val = int(round(i * total_tokens / count))
                if boundaries:
                    val = max(val, boundaries[-1])
                val = min(val, total_tokens)
                boundaries.append(val)
            if boundaries:
                boundaries[-1] = total_tokens
            segments: Dict[str, str] = {}
            for idx, sid in enumerate(valid_ids):
                start_idx = boundaries[idx]
                end_idx = boundaries[idx + 1]
                if end_idx <= start_idx or start_idx >= total_tokens:
                    continue
                start_char = 0 if start_idx == 0 else matches[start_idx].start()
                end_char = len(text) if end_idx >= total_tokens else matches[end_idx].start()
                seg_text = text[start_char:end_char].strip()
                if seg_text:
                    segments[sid] = seg_text
            return segments

        if story_text and anchors:
            try:
                txt_with_markers, markers = self._build_marked_story_and_index(story_text, anchors)
                lines = txt_with_markers.splitlines()


                # sid -> 1-based marker line number

                mline = {}
                for m in (markers or []):
                    sid = (m.get("scene_id") or "").strip()
                    ml = m.get("marker_line_number")
                    if sid and isinstance(ml, int):
                        mline[sid] = ml

    
                ordered = anchors[:]
    
                # Preface (text before the first marker) goes to the first scene

                preface_text = ""
                if ordered:
                    first_sid = (ordered[0].get("id") or "").strip()
                    first_ml = mline.get(first_sid)
                    if isinstance(first_ml, int) and first_ml > 1:
                        preface_text = "\n".join(lines[:max(0, first_ml - 1)])

                for i, sc in enumerate(ordered):
                    sid = (sc.get("id") or "").strip()
                    if not sid:
                        continue
                    this_ml = mline.get(sid)

    
                    # find next scene with a valid marker

                    nxt_ml = None
                    for k in range(i + 1, len(ordered)):
                        ml = mline.get((ordered[k].get("id") or "").strip())
                        if isinstance(ml, int):
                            nxt_ml = ml
                            break

    
                    seg_parts = []
                    if i == 0 and preface_text:
                        seg_parts.append(preface_text)
    
                    if isinstance(this_ml, int):
                        # [this_ml + 1 .. nxt_ml - 1], 1-based indices in lines[]
                        start = min(len(lines), max(1, this_ml + 1))
                    else:
                        start = None  # no anchor for this scene; will fall back later
    
                    end = len(lines) if nxt_ml is None else max(1, nxt_ml - 1)
    
                    if start is not None:
                        chunk = "\n".join(lines[start - 1 : end])
                        seg_parts.append(chunk)
    

                    text_seg = "\n".join([p for p in seg_parts if p]).strip()
                    if text_seg:
                        segment_by_sid[sid] = text_seg
            except Exception:

                # If anything goes wrong, we fall back to per-scene summaries/fractional slices below.

                segment_by_sid = {}

        if story_text and anchors:
            missing_segment_ids = [
                (sc.get("id") or "").strip()
                for sc in anchors
                if (sc.get("id") or "").strip() and not (segment_by_sid.get((sc.get("id") or "").strip()) or "").strip()
            ]
            if missing_segment_ids:
                fallback_segments = _slice_story_evenly(anchors, story_text)
                for sid, txt in fallback_segments.items():
                    if (txt or "").strip():
                        segment_by_sid[sid] = txt
                for sid, txt in list(segment_by_sid.items()):
                    if not (txt or "").strip():
                        segment_by_sid.pop(sid, None)



            # If the marker-based slices barely cover the story text, fall back to even spacing.

            if story_word_total > 0 and segment_by_sid:
                seg_word_total = sum(self._count_words(txt) for txt in segment_by_sid.values())
                coverage_ratio = seg_word_total / float(story_word_total)
                if coverage_ratio < 0.6:
                    even_segments = _slice_story_evenly(anchors, story_text)
                    if even_segments:
                        segment_by_sid = {
                            sid: txt for sid, txt in even_segments.items() if (txt or "").strip()
                        }
                        seg_word_total = sum(self._count_words(txt) for txt in segment_by_sid.values())
                        coverage_ratio = seg_word_total / float(story_word_total) if story_word_total else 1.0
                        try:
                            print(
                                "[extras] coverage fallback: using evenly sliced story text "
                                f"({coverage_ratio:.0%} of {story_word_total} words)"
                            )
                        except Exception:
                            pass




        # Fallback for any scene missing a segment: use its summary (legacy behavior)
        for sc in base_scenes:
            sid = (sc.get("id") or "").strip()
            if sid and sid not in segment_by_sid and not ((sc.get("meta") or {}).get("is_auto_extra")):
                segment_by_sid[sid] = (sc.get("what_happens") or sc.get("description") or "").strip()

        base_segments_for_anchor: Dict[str, str] = {}
        for sc in base_scenes:
            if not isinstance(sc, dict):
                continue

            sid = (sc.get("id") or "").strip()
            if not sid:
                continue
            txt = (segment_by_sid.get(sid) or "").strip()
            if txt:
                base_segments_for_anchor[sid] = txt


        if base_segments_for_anchor:
            self._scene_story_segments = dict(base_segments_for_anchor)


        new_scenes: list[dict] = []
        created = 0
        seen_extra_ids = {
            s.get("id", "")
            for s in base_scenes
            if (s.get("meta") or {}).get("is_auto_extra")
        }

        # Insert extras immediately after their base scenes
        for sc in base_scenes:
            if not isinstance(sc, dict):
                new_scenes.append(sc)
                continue

            sid = (sc.get("id") or "").strip()
            if not sid:
                new_scenes.append(sc)
                continue

    
            # Always keep the original (even if it was an extra)
            new_scenes.append(sc)
    
            # Skip: we only generate extras for base scenes
            if (sc.get("meta") or {}).get("is_auto_extra"):
                continue
    
            text = (segment_by_sid.get(sid) or "").strip()
            total_words = self._count_words(text)
            if total_words <= 0:
                continue
    

            needed_images = max(1, math.ceil(total_words / float(min_words)))
            extras_target = max(0, needed_images - 1)
            if extras_target > EXTRA_IMAGES_MAX_PER_SCENE:
                extras_target = min(EXTRA_IMAGES_ABS_MAX_PER_SCENE, extras_target)
            extras_needed = extras_target

    
            if extras_needed <= 0:
                continue
    
            # Don't duplicate existing extras linked to this base scene
            existing_for_base = existing_extras_by_base.get(sid, [])
            to_make = max(0, extras_needed - len(existing_for_base))
            if to_make <= 0:
                continue
    
            # Base prompt from primary shot

            base_prompt = ""
            try:
                prim = self._choose_primary_shot(sid)
                if prim and prim.prompt:
                    base_prompt = prim.prompt
            except Exception:
                base_prompt = ""

            # Evenly spaced extras across the story segment
            for j in range(len(existing_for_base) + 1, len(existing_for_base) + to_make + 1):
                new_id = f"{sid}x{j}"
                if new_id in seen_extra_ids:
                    continue

                # pick an excerpt around the j/(extras_needed+1) position of the segment
                frac = (j / (extras_needed + 1)) if extras_needed > 0 else 0.5
                excerpt = self._excerpt_text(text, min_words, frac=frac)
                variant = self._pick_variant_label(sc, j)

    
                ns = {
                    "id": new_id,
                    "title": (sc.get("title") or sid) + f" — extra image {j}",
                    "what_happens": excerpt or (sc.get("what_happens") or sc.get("description") or ""),

                    "description": sc.get("description", ""),
                    "characters_present": list(sc.get("characters_present") or []),
                    "location": sc.get("location", ""),
                    "key_actions": list(sc.get("key_actions") or []),

                    "tone": sc.get("tone", ""),
                    "time_of_day": sc.get("time_of_day", ""),
                    "movement_id": sc.get("movement_id", ""),
                    "beat_type": sc.get("beat_type", ""),

                    "meta": {
                        **(sc.get("meta") or {}),
                        "is_auto_extra": True,
                        "source_scene_id": sid,
                        "variant": variant
                    }

                }
                new_scenes.append(ns)
                try:
                    self.scenes_by_id[new_id] = ns
                except Exception:
                    pass

    
                prompt = self._mutate_prompt_for_extra_image(base_prompt, variant, excerpt, sc)
                sh_id = "sh_" + hash_str(new_id + variant + prompt)[:10]
                try:

                    self.shots.append(ShotPrompt(
                        id=sh_id,
                        scene_id=new_id,
                        title=ns["title"],

                        shot_description=f"Auto extra: {variant}",
                        prompt=prompt,
                        continuity_notes="auto extra image (word-gap by story markers)"

                    ))
                except Exception:
                    pass


                seen_extra_ids.add(new_id)
                if excerpt:
                    try:
                        self._scene_story_segments[new_id] = excerpt

                    except Exception:
                        pass
                created += 1

        self.analysis["scenes"] = new_scenes
        return created


    def _current_scene_anchor_map(self) -> Dict[str, str]:
        anchors: Dict[str, str] = {}
        segments = getattr(self, "_scene_story_segments", None)
        if isinstance(segments, dict):
            for sid, txt in segments.items():
                sid_norm = (sid or "").strip()
                if sid_norm and (txt or "").strip():
                    anchors[sid_norm] = txt.strip()

        if isinstance(self.analysis, dict):
            for sc in (self.analysis.get("scenes") or []):
                if not isinstance(sc, dict):
                    continue
                sid = (sc.get("id") or "").strip()
                if not sid:
                    continue
                if sid not in anchors or not anchors[sid].strip():
                    txt = (sc.get("what_happens") or sc.get("description") or "").strip()
                    if txt:
                        anchors[sid] = txt
        return anchors



    def _auto_export_scene_jsons_sync(self, outdir: str, coverage_mode: str = "min") -> None:
        """
        Build enriched scene JSONs (no image generation).
        NEW: 'coverage_mode' controls how many shots get written into each scene:
             - 'min' → one shot (primary) per scene
             - 'max' → all available shot prompts for the scene
        """
        ensure_dir(outdir)
        self.output_dir = outdir
        self._last_export_dir = outdir
        self._scene_story_segments = {}

        story_text = getattr(self, "_last_story_text", "") or ""
        if not story_text:
            story_text = getattr(self, "_dialogue_story_text_cache", "") or ""
        if not story_text and isinstance(self.analysis, dict):
            try:
                story_text = (self.analysis.get("_story_text_cache") or "").strip()
            except Exception:
                story_text = ""
        if not story_text and hasattr(self, "story_text"):
            try:
                story_text = self.story_text.get("1.0", "end").strip()
            except Exception:
                story_text = ""
        if story_text:
            try:
                self._dialogue_story_text_cache = story_text
            except Exception:
                pass

        # --- Auto-insert extra scenes based on word-gap threshold ---
        try:
            # If the Shots & Export UI field exists, prefer its current value
            if (
                threading.current_thread() is threading.main_thread()
                and hasattr(self, "min_words_between_images_var")
            ):
                self.min_words_between_images = int(self.min_words_between_images_var.get() or "0")
        except Exception:
            pass
        try:
            mw = int(getattr(self, "min_words_between_images", 0) or 0)
        except Exception:
            mw = 0
        if mw and mw > 0:
            try:
                created = self._inject_extra_scenes_by_word_gap(min_words=mw)
                if created:
                    print(f"[extras] inserted {created} extra scene(s) based on {mw} words/image")
            except Exception as _e:
                print("extras planning failed:", _e)

        anchor_map = self._current_scene_anchor_map()
        try:
            self._story_anchor_map_by_outdir[os.path.abspath(outdir)] = dict(anchor_map)
        except Exception:
            self._story_anchor_map_by_outdir[os.path.abspath(outdir)] = anchor_map

        titles_map: Dict[str, str] = {}
        story_title = ""
        if isinstance(self.analysis, dict):
            story_title = (self.analysis.get("title") or "").strip()
            for sc in (self.analysis.get("scenes") or []):
                if not isinstance(sc, dict):
                    continue
                sid = (sc.get("id") or "").strip()
                if not sid:
                    continue
                title_txt = (sc.get("title") or sid).strip()
                titles_map[sid] = title_txt or sid
        abs_outdir = os.path.abspath(outdir)
        self._story_scene_titles_by_outdir[abs_outdir] = titles_map
        if story_title:
            self._story_title_by_outdir[abs_outdir] = story_title

        scenes_dir = os.path.join(outdir, SCENE_SUBDIR_NAME)

        ensure_dir(scenes_dir)
    
        # Persist analysis at top level
        if WRITE_ANALYSIS_FILE:
            try:
                with open(os.path.join(outdir, ANALYSIS_FILENAME), "w", encoding="utf-8") as f:
                    json.dump(self._analysis_for_export(), f, indent=2, ensure_ascii=False)
            except Exception:
                pass
    
        # Map of shot prompts (include any edits if a UI is present)
        shot_prompt_map: Dict[str, str] = {}
        for s in (self.shots or []):
            txt = None
            try:
                panel = self.shot_panels.get(s.id) if hasattr(self, "shot_panels") else None
                if panel and "prompt_text" in panel:
                    txt = panel["prompt_text"].get("1.0", "end").strip()
            except Exception:
                txt = None
            shot_prompt_map[s.id] = (txt or s.prompt or "").strip()
    
        errors: List[str] = []
        cov = (coverage_mode or "min").strip().lower()
        for sc in (self.analysis.get("scenes", []) if self.analysis else []):
            sid = sc.get("id", "").strip()
            if not sid:
                continue
            try:
                scene_dir = os.path.join(scenes_dir, sanitize_name(sid))
                ensure_dir(scene_dir)
                ensure_dir(os.path.join(scene_dir, SCENE_REFS_DIR))  # make refs/ upfront
    
                # Choose primary shot & prompt
                primary_shot = self._choose_primary_shot(sid)
                primary_prompt = ""
                if primary_shot:
                    primary_prompt = (shot_prompt_map.get(primary_shot.id, primary_shot.prompt) or "").strip()
                if not primary_prompt:
                    primary_prompt = (sc.get("what_happens") or sc.get("description") or "").strip()
    
                # Build world & conditioning (keeps refs externalized)
                world = self._build_scene_world_enriched(sc, primary_shot, primary_prompt, shot_prompt_map)
                final_world = self._apply_conditioning_with_budget(world, sc, primary_shot, primary_prompt)
    
                # Externalize refs to disk
                if ALWAYS_EXTERNALIZE_IMAGES:
                    final_world = self._externalize_images_to_disk(final_world, scene_dir)
    
                # Optional backlink to _analysis.json
                if WRITE_ANALYSIS_FILE:
                    try:
                        rel_analysis = relpath_posix(os.path.join(outdir, ANALYSIS_FILENAME), start=scene_dir)
                        final_world.setdefault("source", {})["analysis_file"] = rel_analysis
                    except Exception:
                        pass
    
                # Compose ingredients + fused prompt
                ingredients = self._build_prompt_ingredients(final_world, sc)
                fused = self._compose_final_generation_prompt(ingredients)
                used_boost = False
                if USE_LLM_FUSION and self.client:
                    try:
                        #boosted = LLM.fuse_scene_prompt(self.client, self.llm_model, ingredients, self.global_style, NEGATIVE_TERMS)
                        boosted = LLM.fuse_scene_prompt(self.client, self.llm_model, ingredients, self.global_style, NEGATIVE_TERMS, self.scene_render_aspect)
                        if boosted:
                            fused = boosted
                            used_boost = True
                    except Exception:
                        pass
                style_bits = self._style_prompt_bits()
                if style_bits:
                    if used_boost:
                        base = (fused or "").strip()
                        parts = [base] if base else []
                        parts.extend(style_bits)
                        fused = "\n\n".join(parts)
                    elif not any(bit in (fused or "") for bit in style_bits):
                        fused = "\n\n".join([fused] + style_bits) if fused else "\n\n".join(style_bits)
                final_world["scene"]["ingredients"] = ingredients
                final_world["scene"]["fused_prompt"] = fused
    
                # NEW: write shots according to coverage mode
                shot_entries: List[Dict[str, Any]] = []
                if cov == "max":
                    for sh in (self.shots or []):
                        if sh.scene_id == sid:
                            ptxt = (shot_prompt_map.get(sh.id) or sh.prompt or "").strip()
                            if ptxt:
                                entry = {"id": sh.id, "title": sh.title, "prompt": ptxt}
                                if sh.continuity_notes:
                                    entry["notes"] = sh.continuity_notes

                                shot_entries.append(entry)
                    if not shot_entries and primary_prompt:
                        # Fallback: at least one shot
                        shot_entries = [{"id": (primary_shot.id if primary_shot else f"{sid}-A"),
                                         "title": (primary_shot.title if primary_shot else "Primary"),
                                         "prompt": primary_prompt}]
                else:  # 'min'
                    if primary_prompt:

                        shot_entries = [{"id": (primary_shot.id if primary_shot else f"{sid}-A"),
                                         "title": (primary_shot.title if primary_shot else "Primary"),
                                         "prompt": primary_prompt}]
    

                final_world["scene"]["shots"] = shot_entries
    
                # Trim & write
                final_world = self._trim_world_for_size(final_world, max_mb=MAX_SCENE_JSON_MB)
                out_json = os.path.join(scene_dir, f"{sid}.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(final_world, f, indent=2, ensure_ascii=False)
    
            except Exception as e:
                errors.append(f"{sid}: {e}")
    
        # Captions + world.json
        try:
            self._write_captions_todo(outdir, self.analysis.get("scenes", []), shot_prompt_map)
        except Exception as me:
            errors.append("captions_todo: " + str(me))
        try:
            self._write_captions_map(outdir, self.analysis.get("scenes", []), shot_prompt_map)
        except Exception as me:
            errors.append("captions_map: " + str(me))
        try:
            self._world_json_update_from_current(outdir)
        except Exception as we:
            errors.append("world.json: " + str(we))

        try:
            self.analyze_and_emit_dialogue(
                out_dir=outdir,
                text=story_text,
                source_text_path=getattr(self, "_last_story_path", "") or getattr(self, "input_text_path", ""),
            )
        except Exception as exc:
            print(f"[DIALOGUE] analysis/emit failed: {exc}")

        if errors:
            messagebox.showerror("Export finished with errors", "Some scenes failed:\n- " + "\n- ".join(errors))
        else:
            messagebox.showinfo("Export", "Export completed to:\n" + outdir)



    def _auto_render_from_scene_folder(
        self,
        folder: str,
        n: int = 1,
        policy_label: str = "final_prompt",
        delay_s: int = 0,
        coverage_mode: str = "min",
        progress_cb=None
    ) -> None:
        """
        Render images from a folder of scene JSONs.
        - Honors self.scene_render_aspect (incl. portrait ratios).
        - policy_label: "final_prompt" | "scene_fused" | "shot_prompt" | "auto"
        - coverage_mode: "min" (primary prompt) | "max" (all shot prompts if available)
        - Writes a .json sidecar per image with scene_id, shot_id, prompt_used, size, model.
        """
        import os, json, time, glob
    
        if not folder or not os.path.isdir(folder):
            if progress_cb: progress_cb("Folder not found.", 100.0)
            return
    
        cov = (coverage_mode or "min").strip().lower()
        label = (policy_label or "final_prompt").strip().lower()
    
        # Collect tasks across scenes: (scene_dir, scene_id, shot_title, shot_id, prompt_text, identity_ctx, scene_identity)
        tasks: list[tuple[str, str, str, str, str, Dict[str, Any], Dict[str, Any]]] = []
        scene_dirs = sorted([d for d in glob.glob(os.path.join(folder, "*")) if os.path.isdir(d)])
        for scene_dir in scene_dirs:
            sid = os.path.basename(scene_dir)
            jpath = os.path.join(scene_dir, f"{sid}.json")
            if not os.path.isfile(jpath):
                continue
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    scene = json.load(f)
            except Exception:
                continue

            sc = (scene.get("scene") or {})
            fused = (sc.get("fused_prompt") or "").strip()
            finalp = (sc.get("final_prompt") or "").strip()
            shots = list(sc.get("shots") or [])
            scene_identity = sc.get("reference_identity") or {}

            # Title→id map to preserve shot IDs in sidecars
            title_to_id = {}
            for s in shots:
                tl = (s.get("title") or "").strip()
                if tl and s.get("id"):
                    title_to_id[sanitize_name(tl)] = s["id"]

            def primary_fallback() -> tuple[str, str, str, Dict[str, Any]] | None:
                if shots:
                    first = shots[0]
                    ident = first.get("reference_identity") or scene_identity or {}
                    return (
                        first.get("title") or "Shot",
                        (first.get("prompt") or "").strip(),
                        first.get("id") or "",
                        ident,
                    )
                if finalp:
                    return ("Final", finalp, "", scene_identity or {})
                if fused:
                    return ("Fused", fused, "", scene_identity or {})
                return None

            prompts: list[tuple[str, str, str, Dict[str, Any]]] = []  # (title, prompt_text, shot_id, identity)

            if label in ("shot_prompt", "shot prompts"):
                if cov == "max" and shots:
                    for s in shots:
                        ptxt = (s.get("prompt") or "").strip()
                        if ptxt:
                            prompts.append(
                                (
                                    s.get("title") or "Shot",
                                    ptxt,
                                    s.get("id") or "",
                                    s.get("reference_identity") or scene_identity or {},
                                )
                            )
                else:
                    fb = primary_fallback()
                    if fb and fb[1]:
                        prompts.append(fb)
            elif label in ("scene_fused", "fused"):
                if fused:
                    prompts.append(("Fused", fused, "", scene_identity or {}))
                else:
                    fb = primary_fallback()
                    if fb and fb[1]:
                        prompts.append(fb)
            elif label in ("auto", "auto (shot→fused→final)"):
                fb = primary_fallback()
                if fb and fb[1]:
                    prompts.append(fb)
                elif finalp:
                    prompts.append(("Final", finalp, "", scene_identity or {}))
            else:  # "final_prompt"
                if finalp:
                    prompts.append(("Final", finalp, "", scene_identity or {}))
                else:
                    fb = primary_fallback()
                    if fb and fb[1]:
                        prompts.append(fb)

            for (ptitle, ptxt, pid, ident_ctx) in prompts:
                shot_id = pid or title_to_id.get(sanitize_name(ptitle)) or ""
                if ptxt:
                    tasks.append((scene_dir, sid, ptitle, shot_id, ptxt, ident_ctx or {}, scene_identity or {}))
    
        if not tasks:
            if progress_cb: progress_cb("No renderable prompts found.", 100.0)
            return
    
        total = max(1, len(tasks) * max(1, int(n)))
        done = 0
    
        # Honor the selected aspect (includes 9:16 portrait)
        size = self._pick_supported_size(self.scene_render_aspect, self.image_size)
    
        for (scene_dir, sid, ptitle, shot_id, ptxt, identity_ctx, scene_identity_ctx) in tasks:
            renders_dir = os.path.join(scene_dir, "renders")
            ensure_dir(renders_dir)
            for i in range(max(1, int(n))):
                if progress_cb:
                    progress_cb(f"[{sid}] {ptitle} — render {i+1}/{n}", 100.0 * done / total)
                try:
                    prompt_used = self._augment_prompt_for_render(ptxt)
                    imgs = self._try_images_generate(prompt_used, n=1, size=size)
                    for b in imgs:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        ext = sniff_ext_from_bytes(b, ".png")
                        fn = f"{sid}_{sanitize_name(ptitle)}_{i+1}_{ts}{ext}"
                        fp = os.path.join(renders_dir, fn)
                        processed_bytes, _ = self._process_generated_image(b, ext=ext, need_image=False)
                        with open(fp, "wb") as f:
                            f.write(processed_bytes)

                        # Sidecar for later organization
                        bias, post, emiss = self._current_exposure_settings()
                        effective_identity = identity_ctx or scene_identity_ctx or {}
                        sidecar = {
                            "scene_id": sid,
                            "shot_id": shot_id or "",
                            "prompt_used": prompt_used,
                            "image_model": self.image_model,
                            "image_size": size,
                            "created_at": now_iso(),
                            "exposure_bias": float(bias),
                            "post_tonemap": bool(post),
                            "exposure_guidance": exposure_language(bias),
                            "emissive_level": float(emiss),
                        }
                        style_snapshot = self._current_style_snapshot()
                        sidecar["style_id"] = style_snapshot.get("id", "")
                        sidecar["style_name"] = style_snapshot.get("name", "")
                        preset = style_snapshot.get("preset")
                        if isinstance(preset, dict):
                            if preset.get("style_prompt"):
                                sidecar["style_prompt"] = preset.get("style_prompt")
                            if preset.get("palette"):
                                sidecar["style_palette"] = list(preset.get("palette") or [])
                        if effective_identity.get("ref_ids"):
                            sidecar["reference_image_ids"] = list(effective_identity.get("ref_ids", []))
                        if effective_identity.get("primary_map"):
                            sidecar["primary_reference_ids"] = effective_identity.get("primary_map")
                        if effective_identity.get("traits"):
                            sidecar["dna_traits"] = effective_identity.get("traits")
                        if effective_identity.get("palette"):
                            sidecar["reference_palette"] = list(effective_identity.get("palette", []))
                        if abs(sidecar["emissive_level"]) >= 0.15:
                            sidecar["emissive_guidance"] = emissive_language(sidecar["emissive_level"])
                        sidecar_path = os.path.join(renders_dir, os.path.splitext(fn)[0] + ".json")
                        with open(sidecar_path, "w", encoding="utf-8") as jf:
                            json.dump(sidecar, jf, indent=2, ensure_ascii=False)
                except Exception as e:
                    if progress_cb:
                        progress_cb(f"[{sid}] ERROR: {e}", 100.0 * done / total)
                finally:
                    done += 1
                    if delay_s > 0:
                        time.sleep(int(delay_s))
    
            # Optional: reorganize per scene as we go (safe/no‑op if user disabled later)
            try:
                if getattr(self, "group_renders_by_shot_var", None) and self.group_renders_by_shot_var.get():
                    self._reorganize_scene_renders(scene_dir)
            except Exception:
                pass
    
        if progress_cb:
            progress_cb("Rendering complete.", 100.0)


    def _reorganize_scene_renders(self, scene_dir: str) -> None:
        """
        Move images in scene_dir/renders into per‑shot subfolders:
          renders/<shot_id>/<filename>.png (+ sidecar .json)
        Falls back to title‑based buckets if no shot_id can be found.
        """
        import os, glob, json, shutil
    
        renders_dir = os.path.join(scene_dir, "renders")
        if not os.path.isdir(renders_dir):
            return
    
        sid = os.path.basename(scene_dir)
        scene_json = os.path.join(scene_dir, f"{sid}.json")
    
        # Build title→shot_id map from the scene JSON
        title_to_id = {}
        try:
            with open(scene_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            for s in (data.get("scene", {}).get("shots") or []):
                tl = (s.get("title") or "").strip()
                if tl and s.get("id"):
                    title_to_id[sanitize_name(tl)] = s["id"]
        except Exception:
            pass
    
        # Gather renders
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        imgs = []
        for pat in exts:
            imgs.extend(glob.glob(os.path.join(renders_dir, pat)))
        imgs = sorted(imgs)
    
        for img in imgs:
            base = os.path.basename(img)
            root, _ = os.path.splitext(img)
            sidecar = root + ".json"
            shot_id = None
            title_key = ""
    
            # Prefer explicit sidecar
            if os.path.isfile(sidecar):
                try:
                    with open(sidecar, "r", encoding="utf-8") as jf:
                        meta = json.load(jf)
                    shot_id = (meta.get("shot_id") or "").strip() or None
                except Exception:
                    shot_id = None
    
            # Fallback: derive from filename and map via title
            if not shot_id:
                # filename pattern is typically: {scene_id}_{sanitized_title}_{k}_{timestamp}.png
                try:
                    parts = os.path.splitext(os.path.basename(img))[0].split("_")
                    if parts and parts[0].lower() == sid.lower():
                        parts = parts[1:]
                    # drop trailing counters/timestamp
                    if len(parts) >= 2 and parts[-2].isdigit():
                        title_key = "_".join(parts[:-2])
                    elif len(parts) >= 1:
                        title_key = "_".join(parts[:-1])
                    if title_key:
                        shot_id = title_to_id.get(title_key)
                except Exception:
                    shot_id = None
    
            bucket = shot_id or (("by_title_" + title_key) if title_key else "misc")
            dest_dir = os.path.join(renders_dir, bucket)
            ensure_dir(dest_dir)
            try:
                shutil.move(img, os.path.join(dest_dir, os.path.basename(img)))
            except Exception:
                pass
            if os.path.isfile(sidecar):
                try:
                    shutil.move(sidecar, os.path.join(dest_dir, os.path.basename(sidecar)))
                except Exception:
                    pass

    def _collate_all_renders(self, batch_root: str, collated_dir_name: str = "_COLLATED") -> str:
        """
        Sweep all scene renders (png/jpg/webp) under <batch_root>/*/scenes/*/renders/** into
        <batch_root>/_COLLATED/<story>/<scene>/<shot_bucket>/, copying sidecar .json files when present.
        Also produce a flat sibling folder with all renders in one directory and a manifest mapping
        filenames to scenes and narrative anchors.
        Returns the path to the collated folder.
        """
        import glob, shutil, time
        from pathlib import Path

        if not batch_root or not os.path.isdir(batch_root):
            return ""

        base = os.path.join(batch_root, collated_dir_name)
        target = base
        # If an existing non-empty _COLLATED exists, create a timestamped sibling
        try:
            if os.path.isdir(base) and os.listdir(base):
                target = os.path.join(batch_root, f"{collated_dir_name}_{time.strftime('%Y%m%d_%H%M%S')}")
        except Exception:
            pass

        ensure_dir(target)

        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
        patterns = [os.path.join(batch_root, "*", SCENE_SUBDIR_NAME, "*", "renders", "**", e) for e in exts]

        files: list[str] = []
        for pat in patterns:
            files.extend(glob.glob(pat, recursive=True))

        flat_entries: List[Dict[str, str]] = []

        def _safe_copy(src: str, dstdir: str, sid: str, shot_bucket: str) -> str:
            ensure_dir(dstdir)
            base_name = os.path.basename(src)
            prefix = "_".join([p for p in [sid, sanitize_name(shot_bucket)] if p])
            dst_name = (prefix + "_" + base_name) if prefix else base_name
            dst = os.path.join(dstdir, dst_name)
            try:
                shutil.copy2(src, dst)
            except Exception:
                stem, ext = os.path.splitext(dst_name)
                i = 2
                while True:
                    cand = os.path.join(dstdir, f"{stem}__{i}{ext}")
                    if not os.path.exists(cand):
                        shutil.copy2(src, cand)
                        dst = cand
                        break
                    i += 1
            sidecar = os.path.splitext(src)[0] + ".json"
            if os.path.isfile(sidecar):
                try:
                    shutil.copy2(sidecar, os.path.join(dstdir, os.path.basename(os.path.splitext(dst)[0]) + ".json"))
                except Exception:
                    pass
            return dst

        for src in files:
            src_path = Path(src)
            parts = list(src_path.parts)
            try:
                sc_idx = parts.index(SCENE_SUBDIR_NAME)
            except ValueError:
                continue
            story_slug = parts[sc_idx - 1] if sc_idx - 1 >= 0 else "story"
            sid = parts[sc_idx + 1] if sc_idx + 1 < len(parts) else "S?"
            try:
                r_idx = parts.index("renders", sc_idx + 1)
                shot_bucket = parts[r_idx + 1] if r_idx + 2 < len(parts) else ""
            except ValueError:
                shot_bucket = ""

            out_dir = os.path.join(target, sanitize_name(story_slug), sanitize_name(sid))
            if shot_bucket:
                out_dir = os.path.join(out_dir, sanitize_name(shot_bucket))
            _safe_copy(src, out_dir, sid, shot_bucket)

            story_root_path = Path(*parts[:sc_idx]) if sc_idx > 0 else src_path.parent
            flat_entries.append({
                "src": src,
                "scene_id": sid,
                "shot_bucket": shot_bucket,
                "story_slug": story_slug,
                "story_root": str(story_root_path),
            })

        if flat_entries:
            try:
                flat_dir = self._flatten_collated_renders(target, flat_entries)
                if flat_dir:
                    try:
                        print(f"[collate] flattened renders → {flat_dir}")
                    except Exception:
                        pass
            except Exception:
                pass

        return target

    def _flatten_collated_renders(self, collated_root: str, entries: List[Dict[str, str]]) -> str:
        import os, shutil

        if not collated_root:
            return ""

        flat_dir = collated_root.rstrip(os.sep) + "_FLAT"
        ensure_dir(flat_dir)

        manifest_lines: List[str] = []
        used_names: set[str] = set()

        for entry in sorted(entries, key=lambda e: (
            sanitize_name(e.get("story_slug", "")),
            e.get("scene_id", ""),
            sanitize_name(e.get("shot_bucket", "")),
            e.get("src", "")
        )):
            src = entry.get("src", "")
            if not src or not os.path.isfile(src):
                continue

            story_root = os.path.abspath(entry.get("story_root", "")) if entry.get("story_root") else ""
            scene_id = entry.get("scene_id", "").strip() or "?"
            shot_bucket = entry.get("shot_bucket", "").strip()
            story_slug = entry.get("story_slug", "").strip()

            prefix_parts = [story_slug, scene_id]
            if shot_bucket:
                prefix_parts.append(shot_bucket)
            prefix = "_".join([sanitize_name(p) for p in prefix_parts if p])
            base_name = os.path.basename(src)
            candidate = f"{prefix}_{base_name}" if prefix else base_name

            dest_name = candidate
            stem, ext = os.path.splitext(candidate)
            counter = 2
            while dest_name in used_names or os.path.exists(os.path.join(flat_dir, dest_name)):
                dest_name = f"{stem}__{counter}{ext}"
                counter += 1
            used_names.add(dest_name)

            dest_path = os.path.join(flat_dir, dest_name)
            try:
                shutil.copy2(src, dest_path)
            except Exception:
                continue

            anchor_map = self._story_anchor_map_by_outdir.get(story_root, {})
            raw_anchor = (anchor_map.get(scene_id) or "").strip()
            titles_map = self._story_scene_titles_by_outdir.get(story_root, {})
            scene_title = titles_map.get(scene_id) or scene_id
            story_title = self._story_title_by_outdir.get(story_root, "")

            anchor = self._ensure_sentence_anchor(raw_anchor)
            if not anchor:
                anchor = "(no anchor available)"

            line_parts = [f"Image: {dest_name}", f"Scene: {scene_title}"]
            if story_title:
                line_parts.insert(1, f"Story: {story_title}")
            line_parts.append(f"Anchor: {anchor}")
            manifest_lines.append(" | ".join(line_parts))

        if manifest_lines:
            manifest_path = os.path.join(flat_dir, "_image_manifest.txt")
            try:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(manifest_lines))
            except Exception:
                pass

        return flat_dir

    def _on_render_from_scene_folder(self):
        # Pick the folder that contains per‑scene subfolders (S1, S2, …) with {scene_id}.json inside.
        folder = filedialog.askdirectory(title="Choose a 'scenes' folder (contains S1, S2, …)")
        if not folder:
            return
    
        # Read UI knobs safely
        try:
            n = max(1, min(6, int(self.render_n_spin.get())))
        except Exception:
            n = 1
        policy = (self.prompt_source_combo.get() or "final_prompt").strip().lower()
        try:
            delay = max(0, int(self.render_delay_spin.get()))
        except Exception:
            delay = 0
        try:
            coverage = (self.scene_coverage_mode_var.get() or "min").strip().lower()
        except Exception:
            coverage = "min"
        group_after = bool(self.group_renders_by_shot_var.get()) if hasattr(self, "group_renders_by_shot_var") else True
    
        # Progress UI
        prog = ProgressWindow(self.root, title="Render from Scene Folder")
        prog.set_status("Starting…"); prog.set_progress(1)
    
        # Progress callback used by the engine
        def cb(msg: str, pct: float | None = None):
            self.root.after(0, lambda: (
                prog.set_status(msg),
                prog.append_log(msg),
                prog.set_progress(0 if pct is None else pct)
            ))
    
        def worker():
            err = None
            try:
                self._auto_render_from_scene_folder(
                    folder=folder,
                    n=n,
                    policy_label=policy,
                    delay_s=delay,
                    coverage_mode=coverage,            # ← now honors your choice
                    progress_cb=cb
                )
                if group_after:
                    # Sweep into per‑shot subfolders inside each scene
                    for name in sorted(os.listdir(folder)):
                        scene_dir = os.path.join(folder, name)
                        if os.path.isdir(scene_dir):
                            try:
                                self._reorganize_scene_renders(scene_dir)
                            except Exception:
                                pass
            except Exception as e:
                err = e
            finally:
                self.root.after(0, prog.close)
                if err:
                    self.root.after(0, lambda: messagebox.showerror("Render", str(err)))
                else:
                    self.root.after(0, lambda: messagebox.showinfo("Render", "Done."))
    
        threading.Thread(target=worker, daemon=True).start()



    def _on_generate_shots(self):
        if not self.client:
            self._on_connect()
            if not self.client: return
        if not self.analysis:
            messagebox.showinfo("Shots", "Analyze the story first."); return

        char_blocks = []
        for n, c in self.characters.items():
            char_blocks.append({
                "name": n,
                "baseline": (c.refined_description or c.initial_description),
                "dna": compose_character_dna(c)
            })
        loc_blocks = []
        for n, l in self.locations.items():
            loc_blocks.append({
                "name": n,
                "baseline": l.description,
                "dna": compose_location_dna(l)
            })
        scenes = self.analysis.get("scenes", [])
        self._set_status("Authoring shot prompts…")

        try:
            self.shots = LLM.expand_scout_shots(self.client, self.llm_model,
                                                self.analysis.get("story_summary",""),
                                                char_blocks, loc_blocks, scenes, self.global_style,
                                                aspect_label=self.scene_render_aspect)

        except Exception as e:
            messagebox.showerror("Shots", str(e)); self._set_status("Error."); return
        self._render_shot_panels(clear=False)
        self._set_status(str(len(self.shots)) + " shots ready.")

    def _build_scene_enrichment_inline(self, parent, sid: str):
        """
        Render the small inline enrichment panel just below the primary shot prompt.
        Values persist via self.scene_enrichment_vars[sid].
        """
        sc = self.scenes_by_id.get(sid, {}) if hasattr(self, "scenes_by_id") else {}
        # ensure vars exist
        vars_map = self.scene_enrichment_vars.get(sid)
        if not vars_map:
            vars_map = {
                "use": tk.BooleanVar(value=bool(sc.get("export_enrichment_enabled")) or bool(sc.get("export_enrichment"))),
                "emotional_beat": tk.StringVar(value=(sc.get("export_enrichment", {}) or {}).get("emotional_beat", "")),
                "atmosphere_weather": tk.StringVar(value=(sc.get("export_enrichment", {}) or {}).get("atmosphere_weather", "")),
                "color_palette": tk.StringVar(value=(sc.get("export_enrichment", {}) or {}).get("color_palette", "")),
                "props_motifs": tk.StringVar(value=(sc.get("export_enrichment", {}) or {}).get("props_motifs", "")),
                "camera_movement": tk.StringVar(value=(sc.get("export_enrichment", {}) or {}).get("camera_movement", "")),
            }
            self.scene_enrichment_vars[sid] = vars_map

        lf = ttk.Labelframe(parent, text=f"Scene enrichment for {sid} (optional)")
        lf.pack(fill="x", padx=8, pady=(0,8))

        top = ttk.Frame(lf); top.pack(fill="x", padx=8, pady=(6,4))
        ttk.Checkbutton(top, text="Use these details for this scene",
                        variable=vars_map["use"]).pack(side="left")

        grid = ttk.Frame(lf); grid.pack(fill="x", padx=8, pady=(2,8))
        rows = [
            ("Emotional beat / subtext", "emotional_beat"),
            ("Atmosphere / weather", "atmosphere_weather"),
            ("Color palette cues", "color_palette"),
            ("Signature props / motifs", "props_motifs"),
            ("Camera movement / style", "camera_movement"),
        ]
        for r, (label, key) in enumerate(rows):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", padx=(0,8), pady=2)
            entry = ttk.Entry(grid, textvariable=vars_map[key], width=100)
            entry.grid(row=r, column=1, sticky="we", pady=2)
        grid.grid_columnconfigure(1, weight=1)

    def _render_shot_panels(self, clear: bool):
        for w in self.shots_scroll.scrollable_frame.winfo_children(): w.destroy()
        self.shot_panels.clear()
        if clear: return
        # Determine primary shot id per scene so we can attach the enrichment panel there
        primary_by_scene: Dict[str, str] = {}
        seen_scenes = set(s.scene_id for s in self.shots) if self.shots else set()
        for sid in seen_scenes:
            p = self._choose_primary_shot(sid)
            if p:
                primary_by_scene[sid] = p.id

        for s in self.shots:
            lf = ttk.Labelframe(self.shots_scroll.scrollable_frame, text=s.id + " — " + s.title)
            lf.pack(fill="x", padx=8, pady=6)
            ttk.Label(lf, text=s.shot_description, wraplength=1000, justify="left").pack(anchor="w", padx=8, pady=(2,4))
            ttk.Label(lf, text="Prompt:").pack(anchor="w", padx=8)
            prompt_text = tk.Text(lf, height=8, width=120)
            merged = s.prompt.strip()

            if self.global_style and self.global_style != "No global style":
                if "Global visual style:" not in merged:
                    merged = merged + "\n\nGlobal visual style: " + self.global_style + "."
            # if "Aspect:" not in merged and "21:9" not in merged:
            #     merged = merged + "\n\nAspect: " + DEFAULT_ASPECT + "."
            if "Aspect:" not in merged:
                merged = merged + "\n\nAspect: " + self.scene_render_aspect + "."


            if NEGATIVE_TERMS not in merged:
                merged = merged + "\nConstraints: " + NEGATIVE_TERMS + "."
            prompt_text.insert("1.0", merged)
            prompt_text.pack(fill="x", padx=8, pady=(0,6))
            self.shot_panels[s.id] = {"prompt_text": prompt_text}

            # ---- NEW: Enrichment panel directly under the scene’s primary shot prompt ----
            if primary_by_scene.get(s.scene_id) == s.id:
                self._build_scene_enrichment_inline(lf, s.scene_id)

    def _encode_data_uri_cached(self, raw_bytes: bytes, max_side: int, quality: int, use_png: bool) -> str:
        sha = hashlib.sha1(raw_bytes).hexdigest()
        key = (sha, max_side, quality if not use_png else 0, "png" if use_png else "jpg")
        if key in self._encode_cache:
            return self._encode_cache[key]
        im = Image.open(io.BytesIO(raw_bytes))
        if im.mode not in ("RGB", "L", "RGBA"):
            im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            if w >= h:
                new_w = max_side
                new_h = int(h * (max_side / float(w)))
            else:
                new_h = max_side
                new_w = int(w * (max_side / float(h)))
            im = im.resize((max(1, new_w), max(1, new_h)), Image.LANCZOS)
        buf = io.BytesIO()
        if use_png:
            im.save(buf, format="PNG", optimize=True)
            uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            if im.mode == "RGBA":
                im = im.convert("RGB")
            im.save(buf, format="JPEG", quality=int(quality), optimize=True, progressive=True)
            uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        self._encode_cache[key] = uri
        return uri

    def _choose_primary_shot(self, scene_id: str) -> Optional[ShotPrompt]:
        candidates = self.shots if self.shots else []
        c = [s for s in candidates if s.scene_id == scene_id]
        if not c:
            return None
        pref = []
        for s in c:
            p = (s.title + " " + s.shot_description + " " + s.prompt).lower()
            score = 0
            # Existing preferences
            if "establishing" in p: score += 3
            if "master" in p:       score += 2
            if "wide" in p or "ws" in p: score += 1
            # NEW: strongly prefer clear exterior/vehicle language
            veh_tokens = [
                "ship","spacecraft","shuttle","freighter","carrier","hull","exterior","flyby","formation",
                "dogfight","docking","undock","launch","landing","re-entry","reentry","thruster","ion trail",
                "contrail","hangar exterior","rover","crawler","convoy","sailplane","glider","hovercraft",
                "orbital","planet limb","atmosphere entry","apoapsis","periapsis","burn"
            ]
            if any(t in p for t in veh_tokens):
                score += 4
            pref.append((score, s))
        pref.sort(key=lambda x: (-x[0], c.index(x[1])))
        return pref[0][1] if pref else c[0]


    def _classify_shot_prompt(self, title: str, prompt: str, scene_chars: List[str]) -> str:
        """
        Return one of: 'establishing', 'character', 'mixed'.
        Heuristic: favour character/mixed when people are clearly present so continuity locks stay active.
        """
        txt = ((title or "") + " " + (prompt or "")).lower()
        veh_keys = [
            "ship","spacecraft","shuttle","freighter","carrier","frigate","hull","airframe","cockpit exterior",
            "rover","crawler","hovercraft","speeder","convoy","launch","re-entry","reentry","landing","docking",
            "flyby","formation","dogfight","burn","thruster","ion trail","contrail","hangar exterior","orbital",
            "planet limb","atmosphere entry","descent","ascent","space"
        ]
        exterior_phrasings = [
            "exterior","outside","overhead","over the","above the","in space","wide landscape","canyon","open desert",
            "skyline","dorsal view","three-quarter top","silhouette against","far below","vast scale"
        ]
        interior_phrasings = [
            "interior","inside","within","command deck","bridge","control room","ops center","operations center",
            "strategy room","war room","briefing room","situation room","command table","holo table","conference room"
        ]

        has_vehicle = any(k in txt for k in veh_keys)
        is_exterior = any(k in txt for k in exterior_phrasings)
        mentions_interior = any(k in txt for k in interior_phrasings)

        clean_txt = re.sub(r"[^a-z0-9]+", " ", txt)
        mentions_person = False
        for raw_name in (scene_chars or []):
            name_clean = re.sub(r"[^a-z0-9]+", " ", (raw_name or "").lower()).strip()
            if not name_clean:
                continue
            if f" {name_clean} " in f" {clean_txt} ":
                mentions_person = True
                break

        if mentions_person:
            # If people are present, keep identity locks by treating the shot as character/mixed coverage.
            return "character" if not has_vehicle else "mixed"

        if has_vehicle:
            if is_exterior and not mentions_interior:
                return "establishing"
            if not mentions_interior:
                return "establishing"

        if is_exterior and not mentions_interior:
            return "establishing"

        return "mixed"


    def _assets_for_scene(self, scene_chars: List[str], location_name: str) -> List[AssetRecord]:
        out: List[AssetRecord] = []
        for a in self.assets:
            if (a.entity_type == "character" and a.entity_name in (scene_chars or [])) \
               or (a.entity_type == "location" and a.entity_name == (location_name or "")):
                out.append(a)
        return out

    def _asset_ids_for(self, entity_type: str, entity_name: str) -> List[str]:
        return [a.id for a in self.assets if a.entity_type == entity_type and a.entity_name == entity_name]

    def _asset_filename(self, a: AssetRecord) -> str:
        """Return the basename we expect to upload to ChatGPT for this asset."""
        try:
            bn = os.path.basename(a.file_path or "")
        except Exception:
            bn = ""
        # Fallback if path is missing
        if not bn:
            bn = f"{sanitize_name(a.entity_type)}_{sanitize_name(a.entity_name)}_{a.view}_{a.id}.png"
        return bn

    def _derive_view_prefs(self, prompt_text: str) -> Dict[str, str]:
        p = (prompt_text or "").lower()
        char_view = "front"
        loc_view = "establishing"
        if "profile right" in p or ("profile" in p and "right" in p):
            char_view = "profile_right"
        elif "profile left" in p or ("profile" in p and "left" in p):
            char_view = "profile_left"
        elif "three-quarter right" in p or "three quarter right" in p:
            char_view = "three_quarter_right"
        elif "three-quarter left" in p or "three quarter left" in p:
            char_view = "three_quarter_left"
        elif "back" in p:
            char_view = "back"
        elif "full body" in p or "t-pose" in p or "t pose" in p:
            char_view = "full_body_tpose"
        if "detail" in p or "close-up" in p or "close up" in p:
            loc_view = "detail"
        elif "alternate angle" in p or "alt angle" in p:
            loc_view = "alt_angle"
        else:
            loc_view = "establishing"
        return {"char_view": char_view, "loc_view": loc_view}

    def _infer_shot_cast(self, text: str, scene_chars: List[str]) -> List[str]:
        """
        Return the subset of scene_chars that are explicitly called for in a shot.
        If the text clearly indicates an establishing/environment-only shot, return [].
        """
        t = (text or "").lower()
        # Common "no-people" indicators:
        establishing_cues = [
            "establishing", "wide establishing", "empty landscape", "no characters",
            "no people", "exterior only", "environment only", "nobody in frame"
        ]
        if any(k in t for k in establishing_cues):
            return []

        found = []
        for name in scene_chars or []:
            if not name:
                continue
            # word-boundary match, case-insensitive
            if re.search(r"\b" + re.escape(name) + r"\b", text, flags=re.IGNORECASE):
                found.append(name)

        # de-dup, preserve order
        seen = set()
        cast = []
        for n in found:
            if n not in seen:
                cast.append(n)
                seen.add(n)
        return cast

    def _compose_shot_fused_prompt(self,
                                   shot_prompt: str,
                                   storyline: str,
                                   cast_in_frame: List[str],
                                   char_dnas: Dict[str, str],
                                   loc_dna: str,
                                   location_name: str,
                                   tone: str,
                                   global_style: str) -> str:
        """
        Fuse the creative shot prompt with only the DNA that actually appears in frame.
        For establishing shots (empty cast_in_frame), no character DNA is attached.
        """
        parts = []
        parts.append((shot_prompt or "").strip())
        if storyline:
            parts.append("Storyline focus: " + storyline.strip())
        if tone:
            parts.append("Mood: " + tone.strip())
        if cast_in_frame:
            lines = []
            for n in cast_in_frame:
                dna = (char_dnas.get(n) or "").strip()
                if dna:

                    lines.append(n + ": " + dna)

            if lines:
                parts.append("Character DNA — " + " | ".join(lines))
        if loc_dna:
            parts.append("Location DNA — " + loc_dna)
        profiles: List[Any] = []
        for nm in cast_in_frame or []:
            prof = self.characters.get(nm)
            if prof:
                profiles.append(prof)
        if location_name and (location_name in self.locations):
            profiles.append(self.locations[location_name])
        ref_ctx = self._collect_reference_identity(profiles)
        if ref_ctx.get("bits"):
            parts.append("Identity DNA — " + " | ".join(ref_ctx["bits"]))
        continuity_bits: List[str] = []
        if cast_in_frame:
            continuity_bits.append(
                "Keep each listed character's face geometry, eye color, hair color/texture and skin tone identical to their DNA while layering only the scene-specified gear/injuries (helmets, scars, grime) on top; do not turn them metallic or synthetic unless the script states it."
            )
        if loc_dna:
            continuity_bits.append(
                "Maintain the location's established architecture, materials and palette while applying only story-driven lighting, weather, debris or damage cues."
            )
        if continuity_bits:
            parts.append("Continuity guardrails: " + " ".join(continuity_bits))


        if global_style and global_style != "No global style":
            parts.append("Global visual style: " + global_style + ".")
        parts.append("Constraints: " + NEGATIVE_TERMS + ".")
        parts.append(f"Compose for {self.scene_render_aspect} aspect.")

        return "\n\n".join([p for p in parts if p])

    def _load_asset_bytes_by_id(self, asset_id: str) -> Optional[bytes]:
        for a in self.assets:
            if a.id == asset_id:
                try:
                    with open(a.file_path, "rb") as f:
                        return f.read()
                except Exception:
                    return None
        return None

    def _gather_char_candidate_bytes(self, char_name: str) -> List[bytes]:
        c = self.characters.get(char_name)
        if not c: return []
        bytes_list: List[bytes] = []
        seen = set()
        def add_unique(bs: bytes):
            h = hashlib.sha256(bs).hexdigest()
            if h not in seen:
                seen.add(h); bytes_list.append(bs)
        # Selected first
        for vk in ["front","three_quarter_left","three_quarter_right","profile_left","profile_right","back","full_body_tpose"]:
            arr = c.sheet_images.get(vk, [])
            flags = c.sheet_selected.get(vk, [])
            for i, b in enumerate(arr):
                if i < len(flags) and flags[i]:
                    add_unique(b)
        # Assets
        for a in self.assets:
            if a.entity_type == "character" and a.entity_name == char_name:
                try:
                    with open(a.file_path, "rb") as f:
                        add_unique(f.read())
                except Exception:
                    pass
        # Others
        for vkey, arr in c.sheet_images.items():
            flags = c.sheet_selected.get(vkey, [])
            for i, b in enumerate(arr):
                if not (i < len(flags) and flags[i]):
                    add_unique(b)
        # Limit pool size for speed
        return bytes_list[: (MAX_REF_PER_CHAR + MAX_GALLERY_PER_CHAR + 6)]

    def _gather_loc_candidate_bytes(self, loc_name: str) -> List[bytes]:
        l = self.locations.get(loc_name)
        if not l: return []
        bytes_list: List[bytes] = []
        seen = set()
        def add_unique(bs: bytes):
            h = hashlib.sha256(bs).hexdigest()
            if h not in seen:
                seen.add(h); bytes_list.append(bs)
        # Selected
        for vk in ["establishing","alt_angle","detail"]:
            arr = l.sheet_images.get(vk, [])
            flags = l.sheet_selected.get(vk, [])
            for i, b in enumerate(arr):
                if i < len(flags) and flags[i]:
                    add_unique(b)
        # Assets
        for a in self.assets:
            if a.entity_type == "location" and a.entity_name == loc_name:
                try:
                    with open(a.file_path, "rb") as f:
                        add_unique(f.read())
                except Exception:
                    pass
        # Others
        for vkey, arr in l.sheet_images.items():
            flags = l.sheet_selected.get(vkey, [])
            for i, b in enumerate(arr):
                if not (i < len(flags) and flags[i]):
                    add_unique(b)
        return bytes_list[: (MAX_REF_PER_LOC + MAX_GALLERY_PER_LOC + 6)]

    def _movement_info_for_scene(self, movement_id: str) -> Dict[str,str]:
        struct = (self.analysis or {}).get("structure", {}) or {}
        moves = struct.get("movements", []) or []
        for m in moves:
            if m.get("id") == movement_id:
                return {
                    "id": m.get("id",""),
                    "name": m.get("name",""),
                    "focus": m.get("focus",""),
                    "emotional_shift": m.get("emotional_shift",""),
                    "stakes_change": m.get("stakes_change","")
                }
        return {}

    def _build_scene_world_enriched(self, sc: Dict[str,Any],
                                    primary_shot: Optional[ShotPrompt],
                                    provided_primary_prompt: Optional[str],
                                    shot_prompt_map: Dict[str,str]) -> Dict[str,Any]:
        scene_chars = sc.get("characters_present", [])
        location_name = sc.get("location","")
        sid = sc.get("id","S?")

        # Assets scoped to this scene (for registry and ID-based conditioning)
        scene_assets = self._assets_for_scene(scene_chars, location_name)
        scene_asset_ids = {a.id for a in scene_assets}

        if provided_primary_prompt is not None:
            base_prompt = provided_primary_prompt
        else:
            if primary_shot:
                base_prompt = shot_prompt_map.get(primary_shot.id, primary_shot.prompt)
            else:
                what = sc.get("what_happens") or sc.get("description") or ""
                tone = sc.get("tone","")
                tod  = sc.get("time_of_day","")
                loc  = location_name
                combo = what
                if tone: combo += " • tone: " + tone
                if tod:  combo += " • time of day: " + tod
                if loc:  combo += " • location: " + loc
                base_prompt = combo or "Wide cinematic shot."

        char_dnas: Dict[str,str] = {}
        for n in scene_chars:
            if n in self.characters:
                char_dnas[n] = compose_character_dna(self.characters[n], max_len=3000)
        loc_dna = ""
        if location_name and (location_name in self.locations):
            loc_dna = compose_location_dna(self.locations[location_name], max_len=3500)

        scene_profiles: List[Any] = []
        for n in scene_chars:
            prof = self.characters.get(n)
            if prof:
                scene_profiles.append(prof)
        if location_name and (location_name in self.locations):
            scene_profiles.append(self.locations[location_name])

        # mv_info = self._movement_info_for_scene(sc.get("movement_id",""))
        # final_prompt = compose_master_scene_prompt(base_prompt, sc, mv_info, char_dnas, loc_dna, self.global_style)
        mv_info = self._movement_info_for_scene(sc.get("movement_id",""))

        # NEW: if the primary shot is establishing, do NOT inject Character DNA at scene level
        primary_kind = self._classify_shot_prompt(primary_shot.title if primary_shot else "",
                                                  base_prompt,
                                                  scene_chars)
        char_dnas_for_prompt = {} if primary_kind == "establishing" else char_dnas
        # final_prompt = compose_master_scene_prompt(base_prompt, sc, mv_info, char_dnas_for_prompt, loc_dna, self.global_style)
        final_prompt = compose_master_scene_prompt(
            base_prompt, sc, mv_info, char_dnas_for_prompt, loc_dna, self.global_style, self.scene_render_aspect
        )

        scene_ref_ctx = self._collect_reference_identity(scene_profiles)
        if scene_ref_ctx.get("bits"):
            final_prompt = final_prompt + "\n\nIdentity DNA — " + " | ".join(scene_ref_ctx["bits"])

        char_blocks = []
        for n in scene_chars:
            if n in self.characters:
                c = self.characters[n]
                base = (c.refined_description or c.initial_description or "").strip()
                ref_ids = [aid for aid in (c.reference_images or []) if aid in scene_asset_ids]
                char_blocks.append({
                    "id": "char_" + sanitize_name(n),
                    "name": n,
                    "baseline": base[:1200],
                    "dna": char_dnas.get(n, ""),
                    "reference_image_ids": ref_ids
                })

        loc_blocks = []
        if location_name:
            if location_name in self.locations:
                L = self.locations[location_name]
                ref_ids_loc = [aid for aid in (L.reference_images or []) if aid in scene_asset_ids]
                loc_blocks.append({
                    "id": "loc_" + sanitize_name(location_name),
                    "name": location_name,
                    "baseline": (L.description or "")[:1600],
                    "mood": L.mood,
                    "lighting": L.lighting,
                    "dna": loc_dna,
                    "reference_image_ids": ref_ids_loc
                })
            else:
                loc_blocks.append({
                    "id": "loc_" + sanitize_name(location_name),
                    "name": location_name,
                    "baseline": "",
                    "dna": "",
                    "reference_image_ids": []
                })

        shots_small = []
        storyline_text = (sc.get("what_happens") or sc.get("description") or "").strip()
        if self.shots:
            for sh in [s for s in self.shots if s.scene_id == sid]:
                merged_prompt = shot_prompt_map.get(sh.id, sh.prompt)
                # infer who is actually in this shot (from title + description + prompt)
                probe_text = (sh.title or "") + " • " + (sh.shot_description or "") + " • " + (merged_prompt or "")
                cast_in_frame = self._infer_shot_cast(probe_text, scene_chars)
                shot_profiles: List[Any] = []
                for nm in cast_in_frame or []:
                    prof = self.characters.get(nm)
                    if prof:
                        shot_profiles.append(prof)
                if location_name and (location_name in self.locations):
                    shot_profiles.append(self.locations[location_name])
                shot_ref_ctx = self._collect_reference_identity(shot_profiles)
                # fuse only the relevant DNA
                shot_fused = self._compose_shot_fused_prompt(
                    merged_prompt,
                    storyline_text,
                    cast_in_frame,
                    char_dnas,
                    loc_dna,
                    location_name,
                    sc.get("tone",""),
                    self.global_style
                )
                shots_small.append({
                    "id": sh.id,
                    "title": sh.title,
                    "prompt": merged_prompt,
                    "fused_prompt": shot_fused,
                    "cast_in_frame": cast_in_frame,
                    "requires_characters": bool(cast_in_frame),
                    "required_character_names": cast_in_frame,
                    "aspect": self.scene_render_aspect,
                    "mood": sc.get("tone",""),
                    "constraints": NEGATIVE_TERMS,
                    "reference_identity": shot_ref_ctx,
                    "reference_image_ids": list(shot_ref_ctx.get("ref_ids", []))
                })

        world = {
            "version": "1.3",
            "project": {
                "title": self.analysis.get("title","Untitled") if self.analysis else "Untitled",
                "logline": (self.analysis.get("logline","") if self.analysis else ""),
                "style": self.global_style,
                "image_model": self.image_model,
                "default_aspect": self.scene_render_aspect,
                "negative_terms": NEGATIVE_TERMS
            },
            "source": {  # NEW: tie scene JSON to the story analysis
                "scene_id": sid,
                "movement_id": sc.get("movement_id",""),
                "movement_name": (mv_info.get("name","") if mv_info else ""),
                "analysis_file": ""  # filled later with a relative path like ../_analysis.json
            },

            "registry": {
                "images": [
                    {
                        "id": a.id,
                        "path": a.file_path,
                        "filename": self._asset_filename(a),
                        "prompt_used": a.prompt_full,
                        "notes": (a.notes or a.view or ""),
                        "created_at": a.created_at
                    } for a in scene_assets
                ]
            },

            "characters": char_blocks,
            "locations": loc_blocks,
            "scene": {
                "id": sid,
                "title": sc.get("title",""),
                "what_happens": sc.get("what_happens", sc.get("description","")),
                "aspect": self.scene_render_aspect,
                "final_prompt": final_prompt,
                "fused_prompt": final_prompt,
                "primary_shot_kind": primary_kind,
                "conditioning": {
                    "characters": {},
                    "locations": {}
                },
                "reference_gallery": {
                    "characters": {},
                    "locations": {}
                },
                "shots": shots_small,
                "reference_identity": scene_ref_ctx,
                "reference_image_ids": list(scene_ref_ctx.get("ref_ids", []))
            }
        }
        return world

    def _apply_conditioning_with_budget(self, world: Dict[str,Any], sc: Dict[str,Any],
                                        primary_shot: Optional[ShotPrompt],
                                        primary_prompt: Optional[str]) -> Dict[str,Any]:
        """
        Unifies signature: the caller MUST pass primary_prompt for this scene.
        If the primary is an establishing shot, suppress character conditioning at scene level.
        Prefer ID-based references; fall back to embedded data-URIs when needed.
        """
        sid = sc.get("id","S?")
        scene_chars = sc.get("characters_present", [])
        location_name = sc.get("location","")

        # If the primary is an establishing shot, keep scene-level conditioning focused on location
        primary_kind = (primary_shot.title or "").lower() if primary_shot and primary_shot.title else ""
        skip_char_conditioning = ("establish" in primary_kind) or ("wide" in primary_kind and "no people" in primary_kind)

        # Collect asset ids available for this scene (for ID-based path)
        scene_assets = self._assets_for_scene(scene_chars, location_name)
        scene_asset_ids = {a.id for a in scene_assets}

        # Build mapping character name -> asset ids
        char_asset_ids: Dict[str, List[str]] = {}
        if not skip_char_conditioning:
            for n in scene_chars:
                if n in self.characters:
                    c = self.characters[n]
                    # use only asset ids present in this scene
                    char_asset_ids[n] = [aid for aid in (c.reference_images or []) if aid in scene_asset_ids]

        # Locations
        loc_asset_ids: Dict[str, List[str]] = {}
        if location_name and (location_name in self.locations):
            L = self.locations[location_name]
            loc_asset_ids[location_name] = [aid for aid in (L.reference_images or []) if aid in scene_asset_ids]

        any_assets = (any(char_asset_ids.values()) if char_asset_ids else False) or \
                     (any(loc_asset_ids.values()) if loc_asset_ids else False)

        if any_assets:
            # ---------- ID-based path ----------
            tmp = json.loads(json.dumps(world))
            cond_chars_ids: Dict[str, List[str]] = {}
            gallery_chars_ids: Dict[str, List[str]] = {}
            for n, ids in char_asset_ids.items():
                if not ids:
                    continue
                cond_chars_ids["char_" + sanitize_name(n)] = list(ids[:MAX_REF_PER_CHAR])
                if len(ids) > MAX_REF_PER_CHAR:
                    gallery_chars_ids["char_" + sanitize_name(n)] = list(ids[MAX_REF_PER_CHAR:MAX_REF_PER_CHAR+MAX_GALLERY_PER_CHAR])

            cond_locs_ids: Dict[str, List[str]] = {}
            gallery_locs_ids: Dict[str, List[str]] = {}
            for loc, ids in loc_asset_ids.items():
                if not ids:
                    continue
                cond_locs_ids["loc_" + sanitize_name(loc)] = list(ids[:MAX_REF_PER_LOC])
                if len(ids) > MAX_REF_PER_LOC:
                    gallery_locs_ids["loc_" + sanitize_name(loc)] = list(ids[MAX_REF_PER_LOC:MAX_REF_PER_LOC+MAX_GALLERY_PER_LOC])

            # Build filename mirrors using the registry
            reg_list = (world.get("registry", {}) or {}).get("images", []) or []
            id_to_filename = {}
            for it in reg_list:
                iid = it.get("id")
                fn = it.get("filename") or os.path.basename(it.get("path","") or "")
                if iid and fn:
                    id_to_filename[iid] = fn

            def _map_names(d: Dict[str, List[str]]) -> Dict[str, List[str]]:
                out = {}
                for k, ids in (d or {}).items():
                    names = [id_to_filename[i] for i in ids if i in id_to_filename]
                    if names:
                        out[k] = names
                return out

            tmp["scene"]["conditioning"]["characters"] = cond_chars_ids
            tmp["scene"]["conditioning"]["locations"]  = cond_locs_ids
            tmp["scene"]["reference_gallery"]["characters"] = gallery_chars_ids
            tmp["scene"]["reference_gallery"]["locations"]  = gallery_locs_ids
            tmp["scene"]["conditioning_by_filename"] = {
                "characters": _map_names(cond_chars_ids),
                "locations":  _map_names(cond_locs_ids)
            }
            tmp["scene"]["reference_gallery_by_filename"] = {
                "characters": _map_names(gallery_chars_ids),
                "locations":  _map_names(gallery_locs_ids)
            }
            required = sorted(set(sum(tmp["scene"]["conditioning_by_filename"]["characters"].values(), []) +
                                  sum(tmp["scene"]["conditioning_by_filename"]["locations"].values(), [])))
            optional = sorted(set(sum(tmp["scene"]["reference_gallery_by_filename"]["characters"].values(), []) +
                                  sum(tmp["scene"]["reference_gallery_by_filename"]["locations"].values(), [])))
            tmp["scene"]["attachment_manifest"] = {
                "required_filenames": required,
                "optional_filenames": optional,
                "notes": "Upload these exact filenames with this JSON in the same ChatGPT message. "
                         "Use 'required_filenames' for identity lock; 'optional_filenames' are secondary gallery context."
            }
            return tmp

        # ---------- Embedded (data URI) fallback ----------
        use_png = False  # keep as-is unless you need PNG encoding
        char_candidates: Dict[str, List[bytes]] = {}
        if not skip_char_conditioning:
            for n in scene_chars:
                char_candidates[n] = self._gather_char_candidate_bytes(n)
        loc_candidates: Dict[str, List[bytes]] = {}
        if location_name:
            loc_candidates[location_name] = self._gather_loc_candidate_bytes(location_name)

        # Determine initial budgets
        per_char_ref = MAX_REF_PER_CHAR
        per_loc_ref  = MAX_REF_PER_LOC
        per_char_gallery = MAX_GALLERY_PER_CHAR
        per_loc_gallery  = MAX_GALLERY_PER_LOC
        side = DEF_SIDE_PX
        qual = DEF_JPEG_QUALITY

        def make_data_uri(bs: bytes) -> str:
            return b64_data_uri(bs, png=use_png, side=side, jpeg_quality=qual)

        def build_payload():
            tmp = json.loads(json.dumps(world))
            cond_chars: Dict[str, List[str]] = {}
            gal_chars: Dict[str, List[str]] = {}
            if not skip_char_conditioning:
                for n, arr in char_candidates.items():
                    if not arr: 
                        continue
                    key = "char_" + sanitize_name(n)
                    cond_chars[key] = [make_data_uri(b) for b in arr[:per_char_ref]]
                    rest = arr[per_char_ref:per_char_ref+per_char_gallery]
                    if rest:
                        gal_chars[key] = [make_data_uri(b) for b in rest]

            cond_locs: Dict[str, List[str]] = {}
            gal_locs: Dict[str, List[str]] = {}
            for loc, arr in loc_candidates.items():
                if not arr:
                    continue
                key = "loc_" + sanitize_name(loc)
                cond_locs[key] = [make_data_uri(b) for b in arr[:per_loc_ref]]
                rest = arr[per_loc_ref:per_loc_ref+per_loc_gallery]
                if rest:
                    gal_locs[key] = [make_data_uri(b) for b in rest]

            tmp["scene"]["conditioning"]["characters"] = cond_chars
            tmp["scene"]["conditioning"]["locations"]  = cond_locs
            tmp["scene"]["reference_gallery"]["characters"] = gal_chars
            tmp["scene"]["reference_gallery"]["locations"]  = gal_locs
            return tmp

        # Keep it simple: try once; if file size guards exist elsewhere, they’ll kick in.
        return build_payload()

    def _maybe_enrich_scene_with_questions(self, sc: Dict[str, Any]) -> None:
        """
        If the scene's storyline is too thin, ask minimal questions once at export time.
        Answers are stored under sc['export_enrichment'] and used in prompt fusion.
        """
        if not PROMPT_ENRICH_ASK_IF_NEEDED:
            return
        text = (sc.get("what_happens") or sc.get("description") or "").strip()
        need = (len(text) < PROMPT_RICH_MIN_WORDS) or (not sc.get("tone")) or (not sc.get("time_of_day"))
        if not need:
            return
        dlg = EnrichmentDialog(self.root, sc.get("id","S?"))
        answers = dlg.show() or {}
        if answers:
            sc.setdefault("export_enrichment", {}).update(answers)

    def _build_prompt_ingredients(self, world: Dict[str, Any], sc: Dict[str, Any]) -> Dict[str, Any]:
        sid = sc.get("id","S?")
        storyline_bits = []

        # Base storyline
        storyline_bits.append((sc.get("what_happens") or sc.get("description") or "").strip())

        # Plot device emphasis (if present)
        pd_names = []
        pd_notes = []
        _pds = sc.get("plot_devices")
        if isinstance(_pds, list):
            for _pd in _pds:
                if isinstance(_pd, dict):
                    n = (_pd.get("name") or "").strip()
                    if n:
                        pd_names.append(n)
                    ev = (_pd.get("event") or "").strip()
                    nt = (_pd.get("notes") or "").strip()
                    if ev and n:
                        pd_notes.append(f"{n} — {ev}")
                    elif nt and n:
                        pd_notes.append(f"{n} — {nt}")
                elif isinstance(_pd, str):
                    if _pd.strip():
                        pd_names.append(_pd.strip())
        f = (sc.get("plot_device_focus") or "").strip()
        if f:
            pd_names = [f] + [n for n in pd_names if n != f]
        if pd_names:
            storyline_bits.append("Plot device: " + ", ".join(pd_names) + (" (introduction)" if sc.get("is_plot_device_intro") else ""))
        if pd_notes:
            storyline_bits.append("Device beats: " + "; ".join(pd_notes))


        # Tone / time / movement tag
        if sc.get("tone"):
            storyline_bits.append("Tone: " + sc["tone"])
        if sc.get("time_of_day"):
            storyline_bits.append("Time of day: " + sc["time_of_day"])
        mv = self._movement_info_for_scene(sc.get("movement_id",""))
        if mv:
            mv_tags = []
            if mv.get("focus"): mv_tags.append("focus: " + mv["focus"])
            if mv.get("emotional_shift"): mv_tags.append("emotional shift: " + mv["emotional_shift"])
            if mv.get("stakes_change"): mv_tags.append("stakes: " + mv["stakes_change"])
            if mv_tags:
                storyline_bits.append("Movement — " + mv.get("id","") + (" " + mv.get("name","") if mv.get("name") else "") + " (" + "; ".join(mv_tags) + ")")

        # Use inline enrichment only if user ticked "Use these"
        enr_enabled = bool(sc.get("export_enrichment_enabled"))
        enr = sc.get("export_enrichment", {}) if enr_enabled else {}
        if enr.get("emotional_beat"): storyline_bits.append("Emotional beat: " + enr["emotional_beat"])
        if enr.get("atmosphere_weather"): storyline_bits.append("Atmosphere: " + enr["atmosphere_weather"])
        if enr.get("color_palette"): storyline_bits.append("Palette: " + enr["color_palette"])
        if enr.get("props_motifs"): storyline_bits.append("Motifs/props: " + enr["props_motifs"])
        if enr.get("camera_movement"): storyline_bits.append("Camera movement: " + enr["camera_movement"])

        storyline = " ".join([x for x in storyline_bits if x]).strip()

        # Build a quick registry map for ID→path (works before or after your externalizer)
        reg = {r.get("id"): r for r in (world.get("registry", {}) or {}).get("images", [])}

        # Character block
        cast = []
        for name in sc.get("characters_present", []) or []:
            char_id = "char_" + sanitize_name(name)
            dna = ""
            if name in self.characters:
                dna = compose_character_dna(self.characters[name])
            ref_paths: List[str] = []
            def _accumulate(group):
                for ent_id, arr in (group or {}).items():
                    if ent_id != char_id:
                        continue
                    for item in arr or []:
                        if isinstance(item, dict) and item.get("path"):
                            ref_paths.append(item["path"])
                        elif isinstance(item, str) and item in reg and reg[item].get("path"):
                            ref_paths.append(reg[item]["path"])
            scene = world.get("scene", {})
            _accumulate((scene.get("conditioning") or {}).get("characters"))
            _accumulate((scene.get("reference_gallery") or {}).get("characters"))
            cast.append({"name": name, "dna": dna, "ref_paths": ref_paths[:3]})

        # Location block
        loc_name = sc.get("location","") or ""
        loc_id = "loc_" + sanitize_name(loc_name) if loc_name else ""
        loc_dna = compose_location_dna(self.locations[loc_name]) if (loc_name and loc_name in self.locations) else ""
        loc_ref_paths: List[str] = []
        def _accumulate_loc(group):
            for ent_id, arr in (group or {}).items():
                if ent_id != loc_id:
                    continue
                for item in arr or []:
                    if isinstance(item, dict) and item.get("path"):
                        loc_ref_paths.append(item["path"])
                    elif isinstance(item, str) and item in reg and reg[item].get("path"):
                        loc_ref_paths.append(reg[item]["path"])
        if loc_name:
            scene = world.get("scene", {})
            _accumulate_loc((scene.get("conditioning") or {}).get("locations"))
            _accumulate_loc((scene.get("reference_gallery") or {}).get("locations"))

        return {
            "storyline": storyline,
            "cast": cast,
            "location": {"name": loc_name, "dna": loc_dna, "ref_paths": loc_ref_paths[:3]},
        }

    # def _compose_final_generation_prompt(self, ingredients: Dict[str, Any]) -> str:
    #     parts = []
    #     # Storyline (front)
    #     if ingredients.get("storyline"):
    #         parts.append(ingredients["storyline"].strip())

    #     # Cast (guarantee naming)
    #     cast = ingredients.get("cast", [])
    #     if cast:
    #         lines = []
    #         for c in cast:
    #             anchor = (c.get("dna") or "")
    #             # keep anchor tight—first ~18 words
    #             anchor = " ".join(anchor.split()[:18])
    #             lines.append(f"{c.get('name')}: {anchor}".strip(": "))
    #         parts.append("Cast in frame — " + " | ".join(lines))

    #     # Location
    #     loc = ingredients.get("location") or {}
    #     if loc.get("name") or loc.get("dna"):
    #         ldna = " ".join((loc.get("dna") or "").split()[:28])
    #         label = (loc.get("name") + " — " if loc.get("name") else "")
    #         parts.append("Setting — " + label + ldna)

    #     # Shot craft (standardized so prompts are consistently cinematic)
    #     parts.append(
    #         "Cinematography — wide 21:9 composition with foreground‑midground‑background depth; specify lens "
    #         "(e.g., 32–50mm equivalent), camera height (eye‑level or low angle), natural blocking and gesture; "
    #         "light with key/fill/rim and believable directionality and color temperature; subtle atmosphere if appropriate."
    #     )

    #     # Constraints and style hooks
    #     if self.global_style and self.global_style != "No global style":
    #         parts.append("Global visual style: " + self.global_style + ".")
    #     parts.append("Constraints: " + NEGATIVE_TERMS + ".")
    #     parts.append("Identity lock: keep each character’s face geometry, hair, eye color, skin tone, and signature outfit consistent with references.")
    #     parts.append("Do not use camera brand names. No text or watermarks.")

    #     return "\n\n".join([p for p in parts if p])
    def _compose_final_generation_prompt(self, ingredients: Dict[str, Any]) -> str:
        """
        Build the final per-shot generation prompt directly (no LLM fusion).
        Identity lock anchors face/hair/eyes/skin only. Wardrobe/gear may vary per scene.
        If the beat mentions protective gear (helmets/visors/masks/spacesuits) or injuries/scars/damage,
        include them as temporary, scene-limited overlays that do not change identity anchors.
        """
        import re
    
        def _kw_hit(text: str, kws: list[str]) -> bool:
            t = (text or "").lower()
            return any(k in t for k in kws)
    
        storyline = (ingredients.get("storyline") or "").strip()
        parts = []
        if storyline:
            parts.append(storyline)
    
        # Cast (guarantee naming)
        cast = ingredients.get("cast", []) or []
        if cast:
            lines = []
            hair_guards: List[str] = []
            ref_locked: List[str] = []
            for c in cast:
                full_dna = (c.get("dna") or "")
                anchor = " ".join(full_dna.split()[:18])  # keep anchor tight
                nm = (c.get("name") or "").strip()
                if nm:
                    lines.append(f"{nm}: {anchor}".strip(": "))
                    hair_desc = extract_hair_descriptor(full_dna)
                    if hair_desc:
                        hair_guards.append(f"{nm}'s {hair_desc}")
                    if c.get("ref_paths"):
                        ref_locked.append(nm)
            if lines:
                parts.append("Cast in frame — " + " | ".join(lines))
            if hair_guards:
                parts.append("Hair continuity — keep " + join_clause(hair_guards) + " unchanged.")
            if ref_locked:
                parts.append(
                    "Reference stills — use the attached images for "
                    + join_clause(ref_locked)
                    + " to match face geometry, hair color, and skin tone exactly."
                )

        # Location
        loc = ingredients.get("location") or {}
        if loc.get("name") or loc.get("dna"):
            ldna = " ".join((loc.get("dna") or "").split()[:28])
            label = (loc.get("name") + " — " if loc.get("name") else "")
            parts.append("Setting — " + label + ldna)
    
        # Shot craft (consistent cinematography scaffold)
        parts.append(
            "Cinematography — wide 21:9 composition with foreground‑midground‑background depth; specify lens "
            "(e.g., 32–50mm equivalent), camera height (eye‑level or low angle), natural blocking and gesture; "
            "light with key/fill/rim and believable directionality and color temperature; subtle atmosphere if appropriate."
        )

        style_bits = self._style_prompt_bits()
        if style_bits:
            parts.extend(style_bits)

        # Scene‑specific overrides (detect common cues; but allow even if not detected)
        gear_kws   = ["helmet", "helmets", "visor", "visors", "mask", "respirator", "rebreather", "space suit", "spacesuit",
                      "hazmat", "goggles", "breathing apparatus", "oxygen mask", "hard hat", "armor", "armour"]
        injury_kws = ["scar", "scarred", "bandage", "bandaged", "bruise", "cut", "wound", "stitches", "burn", "gash"]
        damage_kws = ["dent", "dented", "damaged", "cracked", "scratched", "shattered", "battle damage", "impact damage"]
    
        mention_overrides = (
            _kw_hit(storyline, gear_kws) or
            _kw_hit(storyline, injury_kws) or
            _kw_hit(storyline, damage_kws)
        )
    
        override_lines = [
            "Continuity: identity lock anchors face geometry, hair style/color, eye color, and skin tone.",
            "Wardrobe/gear may change per scene. If the beat calls for protective gear (helmets, visors, masks, space suits) or injuries/scars/bandages, include them; treat as temporary overlays.",
            "Reflect environmental/damage cues on people and sets (e.g., soot/dust/wetness, dents/scratches) when described. Keep injuries non‑gory.",
        ]
        # Always include the rule; it's harmless if not needed.
        parts.append("Overrides — " + " ".join(override_lines))
    
        # Global style + constraints
        if self.global_style and self.global_style != "No global style":
            parts.append("Global visual style: " + self.global_style + ".")
        parts.append("Constraints: " + NEGATIVE_TERMS + ".")
        parts.append("Do not use camera brand names. No text or watermarks.")
    
        return "\n\n".join([p for p in parts if p])

    def _externalize_images_to_disk(self, world: Dict[str,Any], scene_dir: str) -> Dict[str,Any]:
        """
        Copy/emit all images referenced in the world dict into scene_dir/refs/,
        rewrite conditioning & gallery arrays to [{id, path}], and rebuild registry with relative paths.
        Works for both ID-based references and embedded data URIs.
        """
        refs_dir = os.path.join(scene_dir, SCENE_REFS_DIR)
        ensure_dir(refs_dir)

        # Map original registry by id (may contain absolute paths)
        orig_reg = {r.get("id"): r for r in world.get("registry",{}).get("images", [])}

        copied: Dict[str, str] = {}  # key -> rel path (keys: "path:<abs>" and "b64:<sha>")
        def copy_asset(asset_id: str) -> Optional[str]:
            r = orig_reg.get(asset_id)
            if not r:
                return None
            src = r.get("path") or r.get("abs_path") or ""
            if not src:
                return None
            src_abs = os.path.abspath(src)
            key = "path:" + src_abs
            if key in copied:
                return copied[key]
            ext = os.path.splitext(src_abs)[1].lower()
            if ext not in [".png",".jpg",".jpeg",".webp"]:
                try:
                    with open(src_abs,"rb") as fh:
                        ext = sniff_ext_from_bytes(fh.read(), fallback=".png")
                except Exception:
                    ext = ".png"
            fname = sanitize_name(asset_id) + ext
            dest = os.path.join(refs_dir, fname)
            if os.path.exists(dest):
                stem, e = os.path.splitext(fname)
                k = 2
                while os.path.exists(os.path.join(refs_dir, f"{stem}_{k}{e}")):
                    k += 1
                dest = os.path.join(refs_dir, f"{stem}_{k}{e}")
            try:
                with open(src_abs, "rb") as inf, open(dest, "wb") as outf:
                    outf.write(inf.read())
            except Exception:
                return None
            rel = relpath_posix(dest, start=scene_dir)
            copied[key] = rel
            return rel

        def write_data_uri(item_id: str, data_uri: str) -> Optional[str]:
            parsed = parse_data_uri_to_bytes_and_ext(data_uri)
            if not parsed:
                return None
            raw, ext = parsed
            key = "b64:" + hashlib.sha256(raw).hexdigest()
            if key in copied:
                return copied[key]
            fname = sanitize_name(item_id) + "_" + key[-8:] + ext
            dest = os.path.join(refs_dir, fname)
            with open(dest, "wb") as f:
                f.write(raw)
            rel = relpath_posix(dest, start=scene_dir)
            copied[key] = rel
            return rel

        def transform_group(group_map: Dict[str, Any], id_based: bool) -> Dict[str, Any]:
            new_map: Dict[str, Any] = {}
            for ent_id, arr in (group_map or {}).items():
                out_list = []
                for item in (arr or []):
                    if id_based and isinstance(item, str):
                        rel = copy_asset(item)
                        if rel:
                            out_list.append({"id": item, "path": rel})
                    elif isinstance(item, dict) and "data_uri" in item:
                        rel = write_data_uri(item.get("id") or "ref", item["data_uri"])
                        if rel:
                            out_list.append({"id": item.get("id","ref"), "path": rel})
                    elif isinstance(item, dict) and "path" in item:
                        out_list.append(item)  # already externalized
                if out_list:
                    new_map[ent_id] = out_list
            return new_map

        # Decide mode by inspecting a conditioning value
        chars_group = world["scene"]["conditioning"].get("characters", {})
        is_id_based = False
        for v in chars_group.values():
            if v and isinstance(v[0], str):
                is_id_based = True
                break

        world["scene"]["conditioning"]["characters"] = transform_group(world["scene"]["conditioning"].get("characters", {}), id_based=is_id_based)
        world["scene"]["conditioning"]["locations"]  = transform_group(world["scene"]["conditioning"].get("locations", {}),  id_based=is_id_based)
        world["scene"]["reference_gallery"]["characters"] = transform_group(world["scene"]["reference_gallery"].get("characters", {}), id_based=is_id_based)
        world["scene"]["reference_gallery"]["locations"]  = transform_group(world["scene"]["reference_gallery"].get("locations", {}),  id_based=is_id_based)

        # Build a compact registry for just-this-scene with relative paths
        used_entries = []
        seen_ids = set()
        def collect_from(group):
            for _, arr in (group or {}).items():
                for it in arr:
                    iid = it.get("id"); pth = it.get("path")
                    if iid and pth and iid not in seen_ids:
                        meta = orig_reg.get(iid, {})
                        used_entries.append({
                            "id": iid,
                            "path": pth,  # relative to scene_dir
                            "prompt_used": meta.get("prompt_used",""),
                            "notes": meta.get("notes",""),
                            "created_at": meta.get("created_at","")
                        })
                        seen_ids.add(iid)

        collect_from(world["scene"]["conditioning"]["characters"])
        collect_from(world["scene"]["conditioning"]["locations"])
        collect_from(world["scene"]["reference_gallery"]["characters"])
        collect_from(world["scene"]["reference_gallery"]["locations"])
        world["registry"]["images"] = used_entries

        return world

    def _build_marked_story_and_index(self, story_text: str, scenes: List[Dict[str,Any]]):
        """
        Returns (txt_with_markers:str, markers:list[dict]).
        Strategy:
          • Find a paragraph for each scene by keyword hit (characters/location/what_happens).
          • Insert a visible marker line before that paragraph.
          • If no obvious paragraph is found, fall back to even spacing.
        """
        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "")).strip().lower()

        lines = story_text.splitlines()
        # Build paragraph blocks (start_line, end_line, text)
        blocks = []
        i = 0
        while i < len(lines):
            if lines[i].strip() == "":
                i += 1; continue
            j = i
            chunk = []
            while j < len(lines) and lines[j].strip() != "":
                chunk.append(lines[j]); j += 1
            blocks.append((i, j-1, "\n".join(chunk)))
            i = j

        block_lower = [norm(b[2]) for b in blocks]
        used_block_idx = set()

        markers = []
        total_lines_emitted = 0
        inserts: Dict[int, List[str]] = {}

        # Keyword-based placement per scene
        for si, sc in enumerate(scenes):
            sid = sc.get("id", f"S{si+1}")
            title = sc.get("title", "")
            loc = sc.get("location", "")
            chars = [c for c in (sc.get("characters_present") or []) if isinstance(c, str)]
            what = sc.get("what_happens") or sc.get("description") or ""
            keys = [k for k in [loc] + chars + [what] if k]
            keys = [k.strip() for k in keys if k and k.strip()]

            # find first unused block with any keyword
            pick = None
            if keys:
                nkeys = [norm(k) for k in keys]
                for bi, bl in enumerate(block_lower):
                    if bi in used_block_idx: 
                        continue
                    if any(k in bl for k in nkeys):
                        pick = bi; break

            if pick is None:
                # fallback: space roughly evenly across story
                if blocks:
                    pick = min(len(blocks)-1, max(0, round((si+1) * (len(blocks) / (len(scenes)+1))) - 1))
                    # nudge forward if already used
                    while pick in used_block_idx and pick < len(blocks)-1:
                        pick += 1
            if pick is None:
                # story has no paragraphs; anchor at top (line 0)
                anchor_line = 0
            else:
                used_block_idx.add(pick)
                anchor_line = blocks[pick][0]  # start line of block

            marker_line = f"<<< IMAGE MARKER [{sid}] {title or loc or 'Scene'} >>>"
            inserts.setdefault(anchor_line, []).append(marker_line)

        # produce output text and record final line numbers
        out_lines = []
        current_line = 0
        line_to_marker_texts = {k: v for k, v in inserts.items()}
        # iterate through original lines and inject markers before the anchor line
        for idx, ln in enumerate(lines):
            if idx in line_to_marker_texts:
                for m in line_to_marker_texts[idx]:
                    out_lines.append(m)
            out_lines.append(ln)

        # compute marker line numbers in the produced text (1-based)
        marker_line_numbers: Dict[str, int] = {}
        line_idx = 0
        for ln in out_lines:
            line_idx += 1
            if ln.startswith("<<< IMAGE MARKER ["):
                try:
                    sid = ln.split("[",1)[1].split("]",1)[0]
                    marker_line_numbers[sid] = line_idx
                except Exception:
                    pass

        for sc in scenes:
            sid = sc.get("id","")
            if not sid: 
                continue
            markers.append({
                "scene_id": sid,
                "title": sc.get("title",""),
                "location": sc.get("location",""),
                "characters_present": sc.get("characters_present", []),
                "what_happens": sc.get("what_happens", sc.get("description","")),
                "marker_text": f"<<< IMAGE MARKER [{sid}] {sc.get('title','') or sc.get('location','') or 'Scene'} >>>",
                "marker_line_number": marker_line_numbers.get(sid, None)
            })

        return "\n".join(out_lines), markers

    def _write_analysis_artifacts(self, outdir: Optional[str], story_text: str, analysis: Dict[str,Any], story_hash: str):
        """
        Write:
          • <title>__<hash>__<timestamp>__story_with_markers.txt
          • <title>__<hash>__<timestamp>__scene_markers.json
        If outdir is None → ./analysis_drops
        Returns (txt_path, json_path)
        """
        if not (story_text and analysis and isinstance(analysis, dict)):
            return ("","")

        base_dir = outdir or os.path.join(os.getcwd(), "analysis_drops")
        ensure_dir(base_dir)

        title = (analysis.get("title") or "Untitled")
        title_slug = sanitize_name(title) or "Untitled"
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"{title_slug}__{story_hash[:8]}__{ts}"

        txt_with_markers, markers = self._build_marked_story_and_index(
            story_text, analysis.get("scenes", []) or []
        )

        txt_path = os.path.join(base_dir, base + "__story_with_markers.txt")
        with open(txt_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(txt_with_markers)

        json_path = os.path.join(base_dir, base + "__scene_markers.json")
        payload = {
            "story_title": title,
            "story_hash": story_hash,
            "created_at": now_iso(),
            "total_scenes": len(analysis.get("scenes", []) or []),
            "notes": "Manual placement: open the TXT and locate the <<< IMAGE MARKER […] >>> lines.",
            "markers": markers
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, indent=2, ensure_ascii=False)

        return (txt_path, json_path)

    def _on_export_scene_jsons(self):
        if not self.analysis:
            messagebox.showinfo("Export", "Analyze the story first."); return
        outdir = filedialog.askdirectory(title="Select output directory for enriched scene JSON files (≤ 15 MB)")
        if not outdir: return

        self.output_dir = outdir
        self._last_export_dir = outdir

        # Pre-collect shot prompt edits in main thread (Text widgets not thread-safe)
        shot_prompt_map: Dict[str,str] = {}
        for s in self.shots:
            ptxt = self.shot_panels.get(s.id, {}).get("prompt_text")
            shot_prompt_map[s.id] = (ptxt.get("1.0","end").strip() if ptxt else s.prompt)

        # ---- NEW: Pre-collect inline enrichment choices into the scenes ----
        scenes = list(self.analysis.get("scenes", []))
        for sc in scenes:
            sid = sc.get("id","")
            vars_map = self.scene_enrichment_vars.get(sid, {})
            use = bool(vars_map.get("use").get()) if vars_map.get("use") else False

            if use:
                answers = {
                    "emotional_beat": (vars_map.get("emotional_beat").get().strip() if vars_map.get("emotional_beat") else ""),
                    "atmosphere_weather": (vars_map.get("atmosphere_weather").get().strip() if vars_map.get("atmosphere_weather") else ""),
                    "color_palette": (vars_map.get("color_palette").get().strip() if vars_map.get("color_palette") else ""),
                    "props_motifs": (vars_map.get("props_motifs").get().strip() if vars_map.get("props_motifs") else ""),
                    "camera_movement": (vars_map.get("camera_movement").get().strip() if vars_map.get("camera_movement") else ""),
                }
                answers = {k: v for k, v in answers.items() if v}  # keep only non-empty
                sc["export_enrichment_enabled"] = bool(answers)     # <-- only true if something provided
                sc["export_enrichment"] = answers
            else:
                sc["export_enrichment_enabled"] = False
                sc["export_enrichment"] = {}

        self._export_in_thread(outdir, scenes, shot_prompt_map)
        
    def export_scenes(self, outdir: str) -> None:
        """
        Back-compat wrapper for old docs that referenced `export_scenes(outdir)`.
        Mirrors the UI 'Export Scene Folders…' flow without file dialogs.
        """
        if not outdir:
            return
        if not self.analysis:
            raise RuntimeError("Analyze the story first before exporting.")

        self.output_dir = outdir
        self._last_export_dir = outdir

        # Pre-collect shot prompt edits in main thread (Text widgets are not thread-safe)
        shot_prompt_map: Dict[str, str] = {}
        for s in self.shots:
            ptxt = self.shot_panels.get(s.id, {}).get("prompt_text")
            shot_prompt_map[s.id] = (ptxt.get("1.0", "end").strip() if ptxt else s.prompt)
    
        # Include any inline scene-enrichment answers from the Shots tab
        scenes = list(self.analysis.get("scenes", []))
        for sc in scenes:
            sid = sc.get("id", "")
            vars_map = self.scene_enrichment_vars.get(sid, {}) if hasattr(self, "scene_enrichment_vars") else {}
            use = bool(vars_map.get("use").get()) if vars_map.get("use") else False
            if use:
                answers = {
                    "emotional_beat": (vars_map.get("emotional_beat").get().strip() if vars_map.get("emotional_beat") else ""),
                    "atmosphere_weather": (vars_map.get("atmosphere_weather").get().strip() if vars_map.get("atmosphere_weather") else ""),
                    "color_palette": (vars_map.get("color_palette").get().strip() if vars_map.get("color_palette") else ""),
                    "props_motifs": (vars_map.get("props_motifs").get().strip() if vars_map.get("props_motifs") else ""),
                    "camera_movement": (vars_map.get("camera_movement").get().strip() if vars_map.get("camera_movement") else ""),
                }
                answers = {k: v for k, v in answers.items() if v}
                sc["export_enrichment_enabled"] = bool(answers)
                sc["export_enrichment"] = answers if answers else {}
            else:
                sc["export_enrichment_enabled"] = False
                sc["export_enrichment"] = {}
    
        # Kick the exact same worker the button uses
        self._export_in_thread(outdir, scenes, shot_prompt_map)

    def _on_render_from_scene_jsons(self):
        paths = filedialog.askopenfilenames(
            title="Pick one or more scene JSONs",
            filetypes=[("JSON files","*.json")]
        )
        if not paths:
            return
    
        outdir = filedialog.askdirectory(title="Choose output folder for renders")
        if not outdir:
            return
    
        try:
            n = max(1, min(6, int(self.render_n_spin.get())))
        except Exception:
            n = 1
        policy = self.prompt_source_combo.get() or "auto (shot→fused→final)"
        try:
            delay_s = max(0, int(self.render_delay_spin.get()))
        except Exception:
            delay_s = 0
    
        # Honor selected aspect
        try:
            desired_size = self.aspect_to_size(self.scene_render_aspect)
        except Exception:
            desired_size = ASPECT_TO_SIZE.get(getattr(self, "scene_render_aspect", DEFAULT_ASPECT), self.image_size)
    
        prog = ProgressWindow(self.root, title="Rendering images…")
        prog.set_status("Scanning scenes…"); prog.set_progress(0)
    
        def worker():
            errors: List[str] = []
            total_tasks = 0
    
            # Pre-count tasks
            for spath in paths:
                try:
                    with open(spath, "r", encoding="utf-8") as f:
                        w = json.load(f)
                    shots = (w.get("scene", {}) .get("shots") or []) or [None]
                    total_tasks += (len(shots) * n)
                except Exception:
                    total_tasks += n
    
            ensure_dir(outdir)
            log_path = os.path.join(outdir, "_render_log.txt")
    
            def _log(line: str):
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{now_iso()}] {line}\n")
                except Exception:
                    pass
    
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"Render started {now_iso()}\n")
                    f.write(f"image_model={self.image_model}  desired_size={desired_size}  n={n}  policy={policy}\n\n")
            except Exception:
                pass
    
            done = 0
            for spath in paths:
                base_dir = os.path.dirname(spath)
                try:
                    with open(spath, "r", encoding="utf-8") as f:
                        world = json.load(f)
                except Exception as e:
                    msg = f"{os.path.basename(spath)}: {e}"
                    errors.append(msg); _log("ERROR " + msg)
                    continue
    
                scene = (world.get("scene") or {})
                scene_id = scene.get("id","SCENE")
                shots = (scene.get("shots") or [])
                scene_identity = scene.get("reference_identity") or {}
                target_shots = shots if shots else [None]
    
                # Resolve reference images (if present in world)
                refs = self._resolve_world_ref_images(world, base_dir=base_dir, max_images=8)
    
                for sh in target_shots:
                    shot_id = (sh or {}).get("id","SCENE")
                    shot_identity = (sh or {}).get("reference_identity") if isinstance(sh, dict) else {}
                    try:
                        prompt_base = self._choose_prompt_from_world(world, sh, policy)
                        prompt_used = self._augment_prompt_for_render(prompt_base)
                        if not self.client:
                            self._on_connect()
                            if not self.client:
                                raise RuntimeError("OpenAI client not initialized.")
    
                        # Output folder: <outdir>/<scene_id>/<shot_id>/
                        dest_root = os.path.join(outdir, scene_id, shot_id)
                        ensure_dir(dest_root)
    
                        # Try one retry for size issues, one for timeout (same behavior you already use)
                        curr_size = desired_size
                        imgs = None
                        attempts = 0
                        last_exc = None
                        while attempts <= 1:
                            try:
                                prog_text = f"{scene_id} • {shot_id} — size {curr_size} (try {attempts+1}/{2})"
                                self.root.after(0, lambda t=prog_text: prog.set_status(t))
                                imgs = self.client.generate_images_b64_with_refs(
                                    model=self.image_model,
                                    prompt=prompt_used,
                                    size=curr_size,
                                    ref_data_uris=refs,
                                    n=n
                                )
                                break
                            except Exception as e:
                                last_exc = e
                                emsg = str(e).lower()
                                if ("invalid" in emsg and "size" in emsg) or ("invalid_request_error" in emsg and "size" in emsg):
                                    _log(f"WARN {scene_id}/{shot_id}: size {curr_size} rejected → retry 'auto'")
                                    curr_size = "auto"; attempts += 1; continue
                                if "timed out" in emsg and attempts == 0:
                                    _log(f"WARN {scene_id}/{shot_id}: request timed out → retry once")
                                    time.sleep(3); attempts += 1; continue
                                raise
                        if not imgs and last_exc:
                            raise last_exc
    
                        for i, b in enumerate(imgs):
                            # Robust extension sniff
                            try:
                                ext = sniff_ext_from_bytes(b, fallback=".png")
                            except Exception:
                                ext = ".png"
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            fname = f"{scene_id}_{shot_id}_{i+1}_{ts}{ext}"
                            fpath = os.path.join(dest_root, fname)
                            processed_bytes, _ = self._process_generated_image(b, ext=ext, need_image=False)
                            with open(fpath, "wb") as wf:
                                wf.write(processed_bytes)

                            # Sidecar (kept exactly as before, now next to the image)
                            bias, post, emiss = self._current_exposure_settings()
                            effective_identity = shot_identity or scene_identity or {}
                            sidecar = {
                                "scene_id": scene_id,
                                "shot_id": shot_id,
                                "prompt_used": prompt_used,
                                "image_model": self.image_model,
                                "image_size": curr_size,
                                "ref_images_count": len(refs),
                                "created_at": now_iso(),
                                "exposure_bias": float(bias),
                                "post_tonemap": bool(post),
                                "exposure_guidance": exposure_language(bias),
                                "emissive_level": float(emiss),
                            }
                            style_snapshot = self._current_style_snapshot()
                            sidecar["style_id"] = style_snapshot.get("id", "")
                            sidecar["style_name"] = style_snapshot.get("name", "")
                            preset = style_snapshot.get("preset")
                            if isinstance(preset, dict):
                                if preset.get("style_prompt"):
                                    sidecar["style_prompt"] = preset.get("style_prompt")
                                if preset.get("palette"):
                                    sidecar["style_palette"] = list(preset.get("palette") or [])
                            if effective_identity.get("ref_ids"):
                                sidecar["reference_image_ids"] = list(effective_identity.get("ref_ids", []))
                            if effective_identity.get("primary_map"):
                                sidecar["primary_reference_ids"] = effective_identity.get("primary_map")
                            if effective_identity.get("traits"):
                                sidecar["dna_traits"] = effective_identity.get("traits")
                            if effective_identity.get("palette"):
                                sidecar["reference_palette"] = list(effective_identity.get("palette", []))
                            if abs(sidecar["emissive_level"]) >= 0.15:
                                sidecar["emissive_guidance"] = emissive_language(sidecar["emissive_level"])
                            sidecar_path = os.path.join(dest_root, os.path.splitext(fname)[0] + ".json")
                            with open(sidecar_path, "w", encoding="utf-8") as jf:
                                json.dump(sidecar, jf, indent=2, ensure_ascii=False)
                            done += 1
                            _log(f"SAVED {os.path.join(scene_id, shot_id, fname)}  bytes={len(b)}  size={curr_size}")
                            self.root.after(0, lambda d=done, t=total_tasks, sid=scene_id, shid=shot_id, sz=curr_size:
                                (prog.set_status(f"{sid} • {shid} → saved ({d}/{t}) [size {sz}]"),
                                 prog.set_progress(100.0 * d / max(1, t)))
                            )
                            if delay_s > 0:
                                time.sleep(delay_s)
    
                    except Exception as e:
                        done += n
                        err = f"{os.path.basename(spath)} / {shot_id}: {e}"
                        errors.append(err); _log("ERROR " + err)
                        self.root.after(0, lambda d=done, t=total_tasks:
                            (prog.set_status(f"Error; continuing ({d}/{t})"),
                             prog.set_progress(100.0 * d / max(1, t)))
                        )
    
            def _finish():
                prog.close()
                _log(f"Finished. errors={len(errors)} of {total_tasks} tasks.")
                top = tk.Toplevel(self.root)
                top.title("Render finished" + (" with errors" if errors else ""))
                txt = ("All images rendered to:\n" + outdir) if not errors else \
                      (f"Some renders failed ({len(errors)} of {total_tasks}).\nA log has been written to:\n{log_path}")
                ttk.Label(top, text=txt, justify="left", wraplength=520).pack(fill="x", padx=12, pady=(12,8))
                btns = ttk.Frame(top); btns.pack(pady=(0,12))
                ttk.Button(btns, text="OK", command=top.destroy).pack(side="left", padx=6)
                if errors:
                    def _open_log():
                        p = log_path
                        try:
                            if os.name == "nt":
                                os.startfile(p)  # type: ignore[attr-defined]
                            else:
                                import subprocess, sys
                                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", p])
                        except Exception as ee:
                            messagebox.showerror("Open log", str(ee))
                    ttk.Button(btns, text="Show details…", command=_open_log).pack(side="left", padx=6)
                top.transient(self.root); top.grab_set(); top.focus_set()
                self._set_status("Render complete." if not errors else "Render completed with errors.")
    
            self.root.after(0, _finish)
    
        threading.Thread(target=worker, daemon=True).start()




    def _choose_prompt_from_world(self, world: Dict[str, Any], shot, policy_label: str) -> str:
        """
        Accepts either a dict or a ShotPrompt dataclass for `shot`.
        Builds a prompt according to the selected policy and appends global style,
        negative terms, and aspect if missing.
        """
        scene = (world.get("scene") or {})
        project = (world.get("project") or {})
    
        negatives = project.get("negative_terms") or (self.analysis and self.analysis.get("negative_terms")) or ""
        style = project.get("style") or self.global_style
    
        def _shot_field(s, key: str):
            if not s:
                return None
            # dict
            if isinstance(s, dict):
                return s.get(key)
            # dataclass or simple object with attributes
            try:
                return getattr(s, key, None)
            except Exception:
                return None
    
        def ensure_tails(p: str) -> str:
            out = (p or "").strip()
            if style and "global visual style:" not in out.lower() and style != "No global style":
                out += ("\n\nGlobal visual style: " + style + ".")
            if negatives and negatives not in out:
                out += ("\nConstraints: " + negatives + ".")
            # Append aspect if missing (exports already target 21:9; keep explicit here)
            if self.image_size and "aspect" not in out.lower():
                asp = project.get("aspect") or self.scene_render_aspect or "21:9"
                out += ("\nAspect: " + asp + ".")
            return out
    
        policy = (policy_label or "").lower().strip()
    
        # Explicit policy selections
        if "shot_prompt" in policy or ("shot" in policy and "prompt" in policy):
            return ensure_tails(_shot_field(shot, "prompt") or "")
        if "fused_prompt" in policy or "scene_fused" in policy or ("fused" in policy and "prompt" in policy):
            return ensure_tails(scene.get("fused_prompt", ""))
        if "final_prompt" in policy:
            return ensure_tails(scene.get("final_prompt", ""))
    
        # Auto policy: prefer shot → fused → final → storyline fallback
        base = _shot_field(shot, "prompt") or ""
        if not base:
            base = scene.get("fused_prompt", "") or scene.get("final_prompt", "")
        if not base:
            ing = (scene.get("ingredients") or {})
            base = ing.get("storyline", "") or "Wide cinematic shot."
        return ensure_tails(base)


    def _resolve_world_ref_images(self, world: Dict[str,Any], base_dir: Optional[str], max_images: int = 8) -> List[str]:
        """
        Collects reference images in priority:
          1) scene.conditioning characters/locations (IDs → registry, or path → disk, or data_uri)
          2) If nothing, empty list.
        Produces compact JPEG data URIs using the app's encoder cache.
        """
        reg = {}
        for rec in (world.get("registry",{}).get("images",[]) or []):
            rid = rec.get("id"); 
            if not rid: continue
            pth = rec.get("path") or ""
            # Make relative to the JSON location if needed
            if base_dir and pth and not os.path.isabs(pth):
                pth = os.path.normpath(os.path.join(base_dir, pth))
            reg[rid] = {"path": pth}

        def _iter_blocks():
            cond = (world.get("scene",{}).get("conditioning") or {})
            for blk in ("characters","locations"):
                data = cond.get(blk, {})
                if isinstance(data, dict):
                    for _, arr in data.items():
                        for item in (arr or []):
                            yield item

        refs: List[str] = []
        for item in _iter_blocks():
            if isinstance(item, str):
                # ID only → look up in registry
                rec = reg.get(item)
                if rec and rec.get("path") and os.path.isfile(rec["path"]):
                    try:
                        with open(rec["path"], "rb") as f: raw = f.read()
                        refs.append(self._encode_data_uri_cached(raw, DEF_SIDE_PX, DEF_JPEG_QUALITY, use_png=False))
                    except Exception:
                        pass
            elif isinstance(item, dict):
                # dict could be {'id':..}, {'path':..}, or {'data_uri':..}
                if item.get("data_uri"):
                    refs.append(item["data_uri"])
                elif item.get("path"):
                    pth = item["path"]
                    if base_dir and pth and not os.path.isabs(pth):
                        pth = os.path.normpath(os.path.join(base_dir, pth))
                    if os.path.isfile(pth):
                        try:
                            with open(pth, "rb") as f: raw = f.read()
                            refs.append(self._encode_data_uri_cached(raw, DEF_SIDE_PX, DEF_JPEG_QUALITY, use_png=False))
                        except Exception:
                            pass
                elif item.get("id"):
                    rec = reg.get(item["id"])
                    if rec and rec.get("path") and os.path.isfile(rec["path"]):
                        try:
                            with open(rec["path"], "rb") as f: raw = f.read()
                            refs.append(self._encode_data_uri_cached(raw, DEF_SIDE_PX, DEF_JPEG_QUALITY, use_png=False))
                        except Exception:
                            pass
            if len(refs) >= max_images:
                break

        return refs[:max_images]

    def _export_in_thread(self, outdir: str, scenes: List[Dict[str,Any]], shot_prompt_map: Dict[str,str]):
        self.output_dir = outdir
        self._last_export_dir = outdir
        prog = ProgressWindow(self.root, title="Exporting scene JSONs…")
        prog.set_status("Preparing…")
        prog.set_progress(0)
    
        def worker():
            errors: List[str] = []
            story_text = getattr(self, "_last_story_text", "") or ""
            try:
                scenes_root = os.path.join(outdir, SCENE_SUBDIR_NAME)
                ensure_dir(scenes_root)
                if WRITE_ANALYSIS_FILE:
                    try:
                        analysis_path = os.path.join(outdir, ANALYSIS_FILENAME)
                        with open(analysis_path, "w", encoding="utf-8") as f:
                            json.dump(self._analysis_for_export(), f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        errors.append("analysis: " + str(e))
    
                total = max(1, len(scenes))
                for idx, sc in enumerate(scenes, 1):
                    sid = sc.get("id","S?")
                    def _tick(msg):
                        self.root.after(0, lambda m=msg, i=idx, t=total: (prog.set_status(m), prog.set_progress(100.0*i/t)))
                    _tick("Building " + sid + "…")
    
                    try:
                        scene_dir = os.path.join(scenes_root, sanitize_name(sid))
                        ensure_dir(scene_dir)
                        ensure_dir(os.path.join(scene_dir, SCENE_REFS_DIR))
    
                        primary = self._choose_primary_shot(sid)
                        primary_prompt = None
                        if primary:
                            primary_prompt = shot_prompt_map.get(primary.id, primary.prompt)
    
                        # Build world + apply conditioning **WITH primary_prompt**
                        world = self._build_scene_world_enriched(sc, primary, primary_prompt, shot_prompt_map)
                        _tick("Conditioning " + sid + "…")
                        final_world = self._apply_conditioning_with_budget(world, sc, primary, primary_prompt)
    
                        # --- NEW: Externalize into scenes/<sid>/refs and re-point world
                        if ALWAYS_EXTERNALIZE_IMAGES:
                            final_world = self._externalize_images_to_disk(final_world, scene_dir)
    
                        # Backlink to analysis file (optional)
                        if WRITE_ANALYSIS_FILE:
                            try:
                                rel_analysis = relpath_posix(os.path.join(outdir, ANALYSIS_FILENAME), start=scene_dir)
                                final_world.setdefault("source", {})["analysis_file"] = rel_analysis
                            except Exception:
                                pass
    
                        # Compose ingredients + fused prompt
                        ingredients = self._build_prompt_ingredients(final_world, sc)
                        fused = self._compose_final_generation_prompt(ingredients)
                        used_boost = False
                        if USE_LLM_FUSION and self.client:
                            try:
                                boosted = LLM.fuse_scene_prompt(self.client, self.llm_model, ingredients, self.global_style, NEGATIVE_TERMS)
                                if boosted:
                                    fused = boosted
                                    used_boost = True
                            except Exception:
                                pass
                        style_bits = self._style_prompt_bits()
                        if style_bits:
                            if used_boost:
                                base = (fused or "").strip()
                                merged = [base] if base else []
                                merged.extend(style_bits)
                                fused = "\n\n".join(merged)
                            elif not any(bit in (fused or "") for bit in style_bits):
                                fused = "\n\n".join([fused] + style_bits) if fused else "\n\n".join(style_bits)
                        final_world["scene"]["ingredients"] = ingredients
                        final_world["scene"]["fused_prompt"] = fused
    
                        # Trim & save
                        out_path = os.path.join(scene_dir, sid + ".json")
                        final_world = self._trim_world_for_size(final_world, max_mb=MAX_SCENE_JSON_MB)
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(final_world, f, indent=2, ensure_ascii=False)
    
                        _tick(f"Wrote scenes/{sid}/{sid}.json")
                    except Exception as se:
                        errors.append(sid + ": " + str(se))
                        _tick("Skipped " + sid + " (error)")
    
                # ---- Extra artifacts: captions + markers
                try:
                    self._write_captions_todo(outdir, scenes, shot_prompt_map)
                except Exception as me:
                    errors.append("captions_todo: " + str(me))
                try:
                    self._write_captions_map(outdir, scenes, shot_prompt_map)
                except Exception as me:
                    errors.append("captions_map: " + str(me))
    
                # story_with_markers + scene_markers for manual placement
                try:
                    if not story_text and hasattr(self, "story_text"):
                        story_text = self.story_text.get("1.0", "end").strip()
                    basis_hash = hash_str(story_text) if story_text else hash_str(json.dumps(self.analysis or {}, sort_keys=True))
                    self._write_analysis_artifacts(outdir, story_text, self.analysis or {}, basis_hash)
                except Exception as me:
                    errors.append("markers: " + str(me))
    
                # Update world.json (bridge to your persistent store)
                try:
                    self._world_json_update_from_current(outdir)
                except Exception as we:
                    errors.append("world.json: " + str(we))

                try:
                    self.analyze_and_emit_dialogue(
                        out_dir=outdir,
                        text=story_text,
                        source_text_path=getattr(self, "_last_story_path", "") or getattr(self, "input_text_path", ""),
                    )
                except Exception as exc:
                    print(f"[DIALOGUE] analysis/emit failed: {exc}")

                self.root.after(0, lambda: prog.set_status("All scenes processed."))
            finally:
                def _finish():
                    prog.close()
                    if errors:
                        messagebox.showerror("Export finished with errors", "Some scenes failed:\n- " + "\n- ".join(errors))
                    else:
                        messagebox.showinfo("Export", "Export completed to:\n" + outdir)
                    self._set_status("Export complete." if not errors else "Export completed with errors.")
                self.root.after(0, _finish)
    
        t = threading.Thread(target=worker, daemon=True)
        t.start()



    def _on_bulk_import_characters(self):
        p = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not p: return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Import", "Failed to read JSON:\n" + str(e)); return

        imported = 0
        profiles = []
        if isinstance(data, list):
            profiles = [d for d in data if isinstance(d, dict) and (d.get("type") == "character_profile")]
        elif isinstance(data, dict) and (data.get("type") == "character_profile"):
            profiles = [data]
        else:
            messagebox.showinfo("Import", "Unsupported JSON structure (expect character_profile objects).")
            return

        for d in profiles:
            nm = d.get("name","") or "Character"
            if nm not in self.characters:
                self.characters[nm] = CharacterProfile(name=nm, initial_description="")
            c = self.characters[nm]
            c.initial_description = d.get("initial_description","") or c.initial_description
            c.refined_description = d.get("refined_description","") or c.refined_description
            c.role = d.get("role","") or c.role
            c.goals = d.get("goals","") or c.goals
            c.conflicts = d.get("conflicts","") or c.conflicts
            c.visual_cues_from_photos = d.get("visual_cues_from_photos","") or c.visual_cues_from_photos
            c.sheet_base_prompt = d.get("sheet_base_prompt","") or c.sheet_base_prompt

            imgs = d.get("images") or d.get("views") or {}
            for view_key, arr in imgs.items():
                if view_key not in CHAR_SHEET_VIEWS_DEF:
                    continue
                for item in (arr or []):
                    b = None
                    if isinstance(item, dict):
                        b = decode_data_uri(item.get("data_uri",""))
                        if (b is None) and item.get("path"):
                            try:
                                with open(item["path"], "rb") as fh: b = fh.read()
                            except Exception:
                                b = None
                    elif isinstance(item, str) and item.startswith("data:"):
                        b = decode_data_uri(item)
                    if b:
                        c.sheet_images.setdefault(view_key, []).append(b)
                        c.sheet_selected.setdefault(view_key, []).append(False)
            imported += 1

        self._rebuild_character_panels()
        self._set_status("Imported " + str(imported) + " character(s).")

    def _on_split_by_movement(self):
        self._propose_and_preview_splits(strategy="movement")

    def _on_split_by_location_change(self):
        self._propose_and_preview_splits(strategy="location")

    def _on_split_by_character_intro(self):
        self._propose_and_preview_splits(strategy="character")

    def _on_undo_split(self):
        if not self._undo_scenes_stack:
            messagebox.showinfo("Undo", "Nothing to undo.")
            return
        prev = self._undo_scenes_stack.pop()
        if not self.analysis:
            return
        self.analysis["scenes"] = prev
        self.scenes_by_id = {s.get("id",""): s for s in prev if s.get("id")}
        self.shots.clear()
        self._render_scene_table()
        self._render_shot_panels(clear=True)
        self._set_status("Reverted last split; cleared shots.")

    def _propose_and_preview_splits(self, strategy: str):
        if not self.analysis or not self.analysis.get("scenes"):
            messagebox.showinfo("Split scenes", "Analyze the story first.")
            return

        new_scenes, changes = self._propose_splits(strategy)
        if changes == 0:
            messagebox.showinfo("Split scenes", "No split opportunities found for this strategy.")
            return

        top = tk.Toplevel(self.root)
        top.title("Preview scene splits — " + strategy)
        top.geometry("1100x520")
        ttk.Label(top, text="Proposed scenes (" + str(len(new_scenes)) + ") — review and Apply").pack(anchor="w", padx=8, pady=6)
        cols = ("id","title","location","chars","what")
        tree = ttk.Treeview(top, columns=cols, show="headings", height=18)
        for c, w in [("id",80),("title",240),("location",160),("chars",240),("what",420)]:
            tree.heading(c, text=c.title() if c != "what" else "What happens")
            tree.column(c, width=w, anchor="w")
        tree.pack(fill="both", expand=True, padx=8, pady=6)
        for s in new_scenes:
            tree.insert("", "end", values=(
                s.get("id",""),
                s.get("title",""),
                s.get("location",""),
                ", ".join(s.get("characters_present", [])),
                (s.get("what_happens","") or s.get("description",""))[:200]
            ))

        btns = ttk.Frame(top); btns.pack(fill="x", padx=8, pady=(0,10))
        def apply_now():
            self._undo_scenes_stack.append([dict(x) for x in (self.analysis.get("scenes") or [])])
            self.analysis["scenes"] = new_scenes
            self.scenes_by_id = {s.get("id",""): s for s in new_scenes if s.get("id")}
            self.shots.clear()
            self._render_scene_table()
            self._render_shot_panels(clear=True)
            self._set_status("Applied split (" + strategy + "); cleared shots.")
            top.destroy()
        ttk.Button(btns, text="Apply", command=apply_now).pack(side="left")
        ttk.Button(btns, text="Cancel", command=top.destroy).pack(side="left", padx=8)

    def _propose_splits(self, strategy: str):
        old = self.analysis.get("scenes", []) if self.analysis else []
        if not old:
            return [], 0

        new_list: List[Dict[str, Any]] = []
        changes = 0

        loc_names = list((self.locations or {}).keys())
        char_names = list((self.characters or {}).keys())

        for s in old:
            base_id = s.get("id","S?")
            text = (s.get("what_happens") or s.get("description") or "").strip()
            if not text or len(text) < 30:
                new_list.append(s); continue

            if strategy == "movement":
                segs = self._segment_by_movement(text)
                new_list.extend(self._materialize_segments(s, base_id, segs))
                changes += max(0, len(segs) - 1)
            elif strategy == "location":
                segs = self._segment_by_location_mentions(text, loc_names)
                new_list.extend(self._materialize_segments(s, base_id, segs, infer_location=True, loc_names=loc_names))
                changes += max(0, len(segs) - 1)
            elif strategy == "character":
                segs = self._segment_by_character_intro(text, char_names)
                new_list.extend(self._materialize_segments(s, base_id, segs))
                changes += max(0, len(segs) - 1)
            else:
                new_list.append(s)

        return new_list, changes

    def _segment_by_movement(self, text: str) -> List[str]:
        pattern = re.compile(
            r'\b(?:then|after that|later|meanwhile|'
            r'enters?|leaves?|exits?|arrives?|'
            r'heads? to|goes? to|returns? to|moves? to|walks? to|runs? to|drives? to|'
            r'back at|cut to|smash cut|montage)\b',
            re.IGNORECASE
        )
        parts = pattern.split(text)
        markers = pattern.findall(text)
        if not markers:
            rough = re.split(r'(?:\.\s+|\;\s+|\s+\bthen\b\s+)', text, flags=re.IGNORECASE)
            segs = [seg.strip() for seg in rough if seg and seg.strip()]
            return segs if len(segs) > 1 else [text]
        segs = []
        buf = parts[0].strip()
        for i, m in enumerate(markers):
            if buf:
                segs.append(buf)
            nxt = parts[i+1]
            buf = (m + " " + nxt).strip()
        if buf:
            segs.append(buf)
        segs = [s for s in segs if len(s.strip()) >= 12]
        return segs if segs else [text]

    def _segment_by_location_mentions(self, text: str, loc_names: List[str]) -> List[Dict[str, Any]]:
        names = [n for n in (loc_names or []) if n]
        if not names:
            return [{"text": text}]
        escaped = [re.escape(n) for n in names]
        patt = re.compile(r'(' + "|".join(escaped) + r')', re.IGNORECASE)

        segs: List[Dict[str, Any]] = []
        last = 0
        for m in patt.finditer(text):
            start = m.start()
            if start > last:
                chunk = text[last:start].strip()
                if chunk:
                    segs.append({"text": chunk})
            last = start
        tail = text[last:].strip()
        if tail:
            segs.append({"text": tail})

        if len(segs) <= 1:
            return [{"text": text}]

        for seg in segs:
            match = patt.search(seg["text"])
            if match:
                seg["loc"] = match.group(1)
        return segs

    def _segment_by_character_intro(self, text: str, char_names: List[str]) -> List[str]:
        names = [n for n in (char_names or []) if n]
        if not names:
            return [text]
        joined = "|".join(re.escape(n) for n in names)
        patt = re.compile(
            r'(?:\b(?:meets|meets with|meets up with|joins|joined by|introduces|arrives|enters|appears|shows up|with)\s+(' + joined + r')\b)'
            r'|(?:\b(' + joined + r')\s+(?:arrives|enters|joins|appears|shows up)\b)',
            re.IGNORECASE
        )
        pos = [m.start() for m in patt.finditer(text)]
        if not pos:
            simple = []
            for n in names:
                m = re.search(r'\b' + re.escape(n) + r'\b', text, flags=re.IGNORECASE)
                if m:
                    simple.append(m.start())
            pos = sorted(set(simple))[:3]
        if not pos:
            return [text]
        segs: List[str] = []
        last = 0
        for p in pos:
            if p > last:
                chunk = text[last:p].strip()
                if chunk:
                    segs.append(chunk)
            last = p
        tail = text[last:].strip()
        if tail:
            segs.append(tail)
        segs = [s for s in segs if len(s) >= 12]
        return segs if segs else [text]

    def _materialize_segments(self, s: Dict[str, Any], base_id: str, segs, infer_location: bool=False, loc_names: List[str]=None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if isinstance(segs, list) and not segs:
            return [s]

        def sub_letter(idx: int) -> str:
            letters = "abcdefghijklmnopqrstuvwxyz"
            res = ""
            i = idx
            while True:
                res = letters[i % 26] + res
                i = i // 26 - 1
                if i < 0: break
            return res

        def make_sub_id(base: str, idx: int) -> str:
            suf = sub_letter(idx)
            if re.match(r'^S\d+$', base):
                return base + suf
            return base + "-" + suf

        for i, segment in enumerate(segs):
            text = segment["text"] if (infer_location and isinstance(segment, dict)) else (segment if isinstance(segment, str) else "")
            text = text.strip()
            if not text:
                continue
            new_scene = dict(s)
            new_id = make_sub_id(base_id, i)
            new_scene["id"] = new_id
            new_scene["what_happens"] = text
            if infer_location and isinstance(segment, dict) and segment.get("loc"):
                new_scene["location"] = segment["loc"]
            base_title = s.get("title","").strip()
            label = sub_letter(i).upper()
            if base_title:
                new_scene["title"] = base_title + " — " + label
            else:
                new_scene["title"] = new_id
            out.append(new_scene)

        return out if len(out) >= 2 else [s]


# -----------------------------
# Headless batch runner for --batch
# -----------------------------
def _run_batch_cli(stories_dir: str, out_root: str, render: bool = True,
                   aspect: str = None, world_path: str = None):
    """
    Minimal headless pipeline:
      1) analyze each .txt story
      2) merge baselines from world.json (if present)
      3) author shots
      4) export scene JSONs
      5) (optional) render images
    """
    # Silence GUI popups in headless mode
    try:
        import tkinter.messagebox as _mb
        _mb.showinfo = _mb.showwarning = _mb.showerror = lambda *a, **k: None
    except Exception:
        pass

    # Hidden root + app (we won't call mainloop)
    root = TkinterDnD.Tk() if TKDND_AVAILABLE else tk.Tk()
    root.withdraw()
    app = App(root)

    # Connect OpenAI (uses env var OPENAI_API_KEY if no UI value)
    from os import environ
    app.client = OpenAIClient(environ.get("OPENAI_API_KEY", app.api_key))
    if aspect:
        app.scene_render_aspect = aspect  # export/render honors this

    # If a world store was given, remember it (load if present)
        if world_path:
            app.world_store_path = world_path
            try:
                if os.path.exists(world_path):
                    with open(world_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        app.world = data
                        try:
                            app.world.setdefault("style_presets", [])
                        except Exception:
                            app.world["style_presets"] = []
                        if "default_style_id" not in app.world:
                            app.world["default_style_id"] = ""
            except Exception:
                pass

# -----------------------------
# Boot (GUI by default; CLI if --batch)
# -----------------------------
# --- GUI by default; batch when --batch is present ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Story → Image workstation")
    parser.add_argument("--batch", nargs=3, metavar=("STORIES_DIR", "PROFILES_DIR", "OUT_ROOT"),
                        help="Run headless batch export/render for stories in STORIES_DIR")
    parser.add_argument("--exposure-bias", type=float, default=None,
                        help="[-1..+1] negative=darker, positive=brighter")
    parser.add_argument("--no-post-tonemap", action="store_true",
                        help="Disable post-generation tone-mapping")
    parser.add_argument("--emissive-level", type=float, default=None,
                        help="[-1..+1] prompt-only glow level")
    parser.add_argument("--extra-every-words", type=int, default=None,
                        help="Create ~1 extra per N words after final planning")
    parser.add_argument("--extra-min-words", type=int, default=None,
                        help="Minimum words before scheduling extras")
    parser.add_argument("--extra-max-total", type=int, default=None,
                        help="Max number of extras")
    parser.add_argument("--extra-dry-run", action="store_true",
                        help="Plan extras but do not modify shots")
    parser.add_argument("--expand-scenes", action="store_true",
                        help="Expand captions_map scenes from analysis beats")
    parser.add_argument("--expand-scenes-dry-run", action="store_true",
                        help="Plan scene expansion without writing")
    parser.add_argument("--world-store", type=str, default=None,
                        help="Path to world.json for CLI/headless reference operations")
    parser.add_argument("--add-ref", action="append",
                        help="Attach reference images to a profile: NAME:PATH_OR_GLOB. May repeat.")
    parser.add_argument("--add-ref-folder", action="append",
                        help="Attach all images from a folder (recursive): NAME:FOLDER")
    parser.add_argument("--create-style", action="append",
                        help="Create a user style from ≥5 images: NAME:PATH_OR_GLOB_OR_FOLDER. May repeat.")
    parser.add_argument("--export-style", action="append", metavar="ID_OR_NAME:OUTPATH",
                        help="Export a user style to a .style.json file")
    parser.add_argument("--import-style", action="append", metavar="PATH",
                        help="Import a style from a .style.json")
    args = parser.parse_args()

    def _normalize_world_store_path(raw: Optional[str]) -> str:
        if not raw:
            return ""
        expanded = os.path.expanduser(os.path.expandvars(raw))
        return os.path.abspath(expanded)

    def _parse_profile_label(label: str) -> Tuple[str, Optional[str]]:
        raw = (label or "").strip()
        if not raw:
            return "", None
        low = raw.lower()
        mappings = [
            ("char=", "character"),
            ("character=", "character"),
            ("char/", "character"),
            ("character/", "character"),
            ("loc=", "location"),
            ("location=", "location"),
            ("loc/", "location"),
            ("location/", "location"),
        ]
        for prefix, kind in mappings:
            if low.startswith(prefix):
                return raw[len(prefix):].strip(), kind
        return raw, None

    def _gather_cli_images(path_spec: str, recursive: bool) -> List[str]:
        spec = (path_spec or "").strip()
        if not spec:
            return []
        expanded = os.path.expanduser(os.path.expandvars(spec))
        expanded = os.path.abspath(expanded)
        use_recursive = recursive or any(ch in expanded for ch in ("*", "?", "["))
        matches: List[str]
        if os.path.isfile(expanded):
            matches = [expanded]
        else:
            pattern = expanded
            if os.path.isdir(expanded):
                pattern = os.path.join(expanded, "**", "*") if use_recursive else os.path.join(expanded, "*")
            matches = glob.glob(pattern, recursive=use_recursive)
        files: List[str] = []
        for candidate in matches:
            if not candidate:
                continue
            if os.path.isdir(candidate):
                continue
            ext = os.path.splitext(candidate)[1].lower()
            if ext in VALID_IMAGE_EXTS and os.path.isfile(candidate):
                files.append(os.path.abspath(candidate))
        return list(dict.fromkeys(files))

    def _run_cli_reference_ops(app: App,
                               file_specs: List[str],
                               folder_specs: List[str],
                               world_path: str) -> None:
        if not file_specs and not folder_specs:
            return

        if not world_path:
            print("[add-ref] No world store path provided; skipping reference import.")
            return

        app.world_store_path = world_path
        try:
            app._load_world_store_if_exists()
        except Exception as exc:
            print(f"[add-ref] warning: failed to load world store: {exc}")
        try:
            app._apply_world_baselines_to_state(create_missing=True)
        except TypeError:
            app._apply_world_baselines_to_state()
        except Exception as exc:
            print(f"[add-ref] warning: failed to apply world baselines: {exc}")

        print(f"[add-ref] world store: {world_path}")
        operations: List[Tuple[str, bool]] = []
        for spec in file_specs or []:
            operations.append((spec, False))
        for spec in folder_specs or []:
            operations.append((spec, True))

        total_added = 0
        processed = 0
        for raw_spec, recursive in operations:
            processed += 1
            try:
                name_part, path_part = raw_spec.split(":", 1)
            except ValueError:
                print(f"[add-ref] invalid format (expected NAME:PATH): {raw_spec}")
                continue

            name_clean, hint = _parse_profile_label(name_part)
            if not name_clean:
                print(f"[add-ref] empty profile name in spec: {raw_spec}")
                continue

            kind, profile = app._find_profile_by_name(name_clean, hint)
            if not profile:
                print(f"[add-ref] profile not found: {name_clean}")
                continue

            paths = _gather_cli_images(path_part, recursive=recursive)
            if not paths:
                print(f"[add-ref] no images matched for {raw_spec}")
                continue

            added_ids = app._attach_refs_to_profile(profile, paths)
            label = getattr(profile, "name", name_clean) or name_clean
            print(f"[add-ref] {label}: attached {len(added_ids)} of {len(paths)} image(s)")
            total_added += len(added_ids)

        try:
            if app.world_store_path:
                app._save_world_store_to(app.world_store_path)
        except Exception as exc:
            print(f"[add-ref] warning: failed to save world store: {exc}")

        print(f"[add-ref] completed ({processed} spec(s), {total_added} image(s) linked)")

    def _run_cli_style_creation(app: App,
                                specs: List[str],
                                world_path: str) -> None:
        if not specs:
            return
        if not world_path:
            print("[create-style] No world store path provided; skipping style creation.")
            return

        app.world_store_path = world_path
        try:
            app._load_world_store_if_exists()
        except Exception as exc:
            print(f"[create-style] warning: failed to load world store: {exc}")

        created = 0
        for raw in specs:
            try:
                name, src = raw.split(":", 1)
            except ValueError:
                print(f"[create-style] invalid format (expected NAME:PATH): {raw}")
                continue
            name = name.strip()
            paths = _gather_cli_images(src, recursive=True)
            if len(paths) < 5:
                print(f"[create-style] need ≥5 images (got {len(paths)}) for {raw}")
                continue
            preset = app._create_style_from_images(name, paths)
            if preset:
                created += 1
                print(f"[create-style] created '{preset.get('name','style')}' ({len(preset.get('sample_asset_ids') or [])} samples)")
            else:
                print(f"[create-style] failed to create style for {raw}")

        try:
            if app.world_store_path:
                app._save_world_store_to(app.world_store_path)
        except Exception as exc:
            print(f"[create-style] warning: failed to save world store: {exc}")

        print(f"[create-style] completed ({created} style(s) added)")

    def _run_cli_style_export(app: App,
                               specs: List[str],
                               world_path: str) -> None:
        if not specs:
            return

        if world_path:
            app.world_store_path = world_path
            try:
                app._load_world_store_if_exists()
            except Exception as exc:
                print(f"[style] export warning: failed to load world store: {exc}")

        exported = 0
        for raw in specs:
            try:
                ident, out_raw = raw.split(":", 1)
            except ValueError:
                print(f"[style] export invalid format (expected ID_OR_NAME:OUTPATH): {raw}")
                continue
            ident = (ident or "").strip()
            out_path = os.path.abspath(os.path.expanduser(out_raw.strip()))
            if not ident or not out_path:
                print(f"[style] export invalid spec: {raw}")
                continue

            target = None
            for preset in getattr(app, "_user_styles", []) or []:
                if not isinstance(preset, dict):
                    continue
                sid = (preset.get("id") or "").strip()
                name = (preset.get("name") or "").strip()
                if ident == sid or ident == name:
                    target = preset
                    break
            if not target:
                print(f"[style] export: user style not found: {ident}")
                continue

            payload = _style_export_minimal_dict(target)
            try:
                ensure_dir(os.path.dirname(out_path) or ".")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                try:
                    base = os.path.splitext(out_path)[0]
                    for idx, path in enumerate(_style_preview_paths(app, target), start=1):
                        _save_thumb(path, f"{base}.preview{idx}.jpg", max_side=320)
                except Exception:
                    pass
                print(f"[style] exported '{target.get('name', '(unnamed)')}' → {out_path}")
                exported += 1
            except Exception as exc:
                print(f"[style] export failed: {exc}")

        if exported == 0:
            print("[style] export complete (no files written)")
        else:
            print(f"[style] export complete ({exported} file(s) written)")


    def _run_cli_style_import(app: App,
                               paths: List[str],
                               world_path: str) -> None:
        if not paths:
            return

        if world_path:
            app.world_store_path = world_path
            try:
                app._load_world_store_if_exists()
            except Exception as exc:
                print(f"[style] import warning: failed to load world store: {exc}")

        imported = 0
        for raw in paths:
            path = os.path.abspath(os.path.expanduser((raw or "").strip()))
            if not path or not os.path.isfile(path):
                print(f"[style] import failed (missing file): {raw}")
                continue
            data = _read_json_safely(path)
            if not isinstance(data, dict):
                print(f"[style] import failed (not JSON object): {path}")
                continue
            style = data.get("style") if isinstance(data.get("style"), dict) else data
            if not isinstance(style, dict):
                print(f"[style] import failed (bad structure): {path}")
                continue
            sid = (style.get("id") or "").strip()
            name = (style.get("name") or "").strip()
            if not name:
                name = sid or f"Imported_{int(time.time())}"
                style["name"] = name
            if not sid:
                sid = f"style_{int(time.time())}_{hashlib.md5(name.encode('utf-8')).hexdigest()[:6]}"
                style["id"] = sid

            user_styles = getattr(app, "_user_styles", []) or []
            conflict = False
            for preset in user_styles:
                if isinstance(preset, dict) and (preset.get("id") or "").strip() == sid:
                    conflict = True
                    break
            if conflict:
                new_id = f"{sid}_dup{int(time.time())}"
                style["id"] = new_id
                print(f"[style] import: id '{sid}' exists; keeping both as '{new_id}'")
            user_styles.append(style)
            imported += 1

        if imported:
            try:
                app._save_user_styles()
            except Exception as exc:
                print(f"[style] import warning: failed to save user styles: {exc}")
            try:
                app._merge_styles_for_dropdown()
            except Exception:
                pass
            if world_path:
                try:
                    app._save_world_store_to(world_path)
                except Exception as exc:
                    print(f"[style] import warning: failed to save world store: {exc}")
        print(f"[style] import complete ({imported} style(s) added)")


    if args.exposure_bias is not None:
        EXPOSURE_BIAS = _clamp(float(args.exposure_bias), -1.0, 1.0)
    if args.no_post_tonemap:
        EXPOSURE_POST_TONEMAP = False
    if args.emissive_level is not None:
        EMISSIVE_LEVEL = _clamp(float(args.emissive_level), -1.0, 1.0)
    if args.extra_every_words is not None:
        EXTRA_EVERY_WORDS = max(1, int(args.extra_every_words))
    if args.extra_min_words is not None:
        EXTRA_MIN_WORDS = max(0, int(args.extra_min_words))
    if args.extra_max_total is not None:
        EXTRA_MAX_TOTAL = max(0, int(args.extra_max_total))
    if getattr(args, "expand_scenes", False) or getattr(args, "expand_scenes_dry_run", False):
        _maybe_expand_scenes(
            analysis_path="/mnt/data/_analysis.json",
            captions_path="/mnt/data/captions_map.json",
            dry_run=bool(args.expand_scenes_dry_run),
            extra_dry_run=bool(args.extra_dry_run)
        )

    add_ref_specs = args.add_ref or []
    add_ref_folder_specs = args.add_ref_folder or []
    create_style_specs = args.create_style or []
    export_style_specs = args.export_style or []
    import_style_specs = args.import_style or []
    world_store_cli = _normalize_world_store_path(args.world_store)
    if args.batch and not world_store_cli and args.batch[1]:
        world_store_cli = _normalize_world_store_path(os.path.join(args.batch[1], "world.json"))
    if (add_ref_specs or add_ref_folder_specs or create_style_specs or export_style_specs or import_style_specs) and not world_store_cli:
        world_store_cli = _normalize_world_store_path(os.path.join(os.getcwd(), "world.json"))
    if add_ref_specs or add_ref_folder_specs or create_style_specs or export_style_specs or import_style_specs:
        tmp_app = App(root=None, headless=True)
        if add_ref_specs or add_ref_folder_specs:
            _run_cli_reference_ops(tmp_app, add_ref_specs, add_ref_folder_specs, world_store_cli)
        if create_style_specs:
            _run_cli_style_creation(tmp_app, create_style_specs, world_store_cli)
        if export_style_specs:
            _run_cli_style_export(tmp_app, export_style_specs, world_store_cli)
        if import_style_specs:
            _run_cli_style_import(tmp_app, import_style_specs, world_store_cli)
        if not args.batch:
            sys.exit(0)
    if args.batch:
        stories_dir, profiles_dir, out_root = args.batch
        app = App(root=None, headless=True)
        if world_store_cli:
            app.world_store_path = world_store_cli
            try:
                app._load_world_store_if_exists()
            except Exception:
                pass
        app.run_batch_on_folder(
            stories_dir=stories_dir,
            profiles_dir=profiles_dir,
            out_root=out_root,
            prompt_policy="final_prompt",
            render_n=1,
            delay_s=1,
            aspect=DEFAULT_ASPECT,
            match_threshold=0.44,
            min_margin=0.06,
            create_minimal_profiles_for_new=True
        )
    else:
        root = TkinterDnD.Tk() if 'TKDND_AVAILABLE' in globals() and TKDND_AVAILABLE else tk.Tk()
        App(root)
        root.mainloop()


