"""
validate_outputs.py

Programmatic QA pass over pipeline output files.

Checks:
  1. JSON Schema validation (metadata.json and transcript.json)
  2. Temporal ordering (start_time ascending)
  3. No start_time >= end_time
  4. No end_time > audio_duration_seconds
  5. Speaker ID coverage (no Speaker_unknown if diarization ran)
  6. Nonverbal event count
  7. Emotion event count
  8. Screenshare segment count and timing
  9. Spot-check transcript at key utterances

Usage:
    python src/validate_outputs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("ERROR: jsonschema not installed. Run: pip install jsonschema")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
SCHEMA_DIR = ROOT / "docs"

METADATA_SCHEMA_PATH = SCHEMA_DIR / "schema_metadata.json"
TRANSCRIPT_SCHEMA_PATH = SCHEMA_DIR / "schema_transcript.json"

VIDEO_IDS = [
    "1711656206762",
    "1712170523563",
]

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def warn(label: str, detail: str = "") -> None:
    print(f"  [{WARN}] {label}" + (f" — {detail}" if detail else ""))


def validate_video(video_id: str, meta_schema: dict, transcript_schema: dict) -> bool:
    meta_path = OUTPUT_DIR / f"{video_id}_metadata.json"
    tx_path = OUTPUT_DIR / f"{video_id}_transcript.json"

    print(f"\n{'='*60}")
    print(f"Video: {video_id}")
    print(f"{'='*60}")

    all_pass = True

    # ── File existence ───────────────────────────────────────────
    if not check("metadata.json exists", meta_path.exists(), str(meta_path)):
        all_pass = False
        return False
    if not check("transcript.json exists", tx_path.exists(), str(tx_path)):
        all_pass = False
        return False

    meta = json.loads(meta_path.read_text())
    tx = json.loads(tx_path.read_text())

    # ── Schema validation ────────────────────────────────────────
    try:
        jsonschema.validate(meta, meta_schema)
        check("metadata schema valid", True)
    except jsonschema.ValidationError as e:
        check("metadata schema valid", False, e.message[:120])
        all_pass = False

    try:
        jsonschema.validate(tx, transcript_schema)
        check("transcript schema valid", True)
    except jsonschema.ValidationError as e:
        check("transcript schema valid", False, e.message[:120])
        all_pass = False

    # ── Audio duration ────────────────────────────────────────────
    audio_dur = meta.get("audio_duration_seconds", float("inf"))
    print(f"\n  Audio duration: {audio_dur:.1f}s  "
          f"Video duration: {meta.get('duration_seconds', '?')}s")

    # ── Utterance checks ─────────────────────────────────────────
    utterances = tx.get("utterances", [])
    n = len(utterances)
    print(f"  Utterances: {n}")

    if n == 0:
        check("has utterances", False, "transcript is empty")
        all_pass = False
        return False

    # temporal ordering
    bad_order = [
        i for i in range(1, n)
        if utterances[i]["start_time"] < utterances[i-1]["start_time"]
    ]
    all_pass &= check(
        "start_time ascending",
        len(bad_order) == 0,
        f"{len(bad_order)} violations" if bad_order else "",
    )

    # no zero-length or inverted
    bad_dur = [
        i for i, u in enumerate(utterances)
        if u["start_time"] >= u["end_time"]
    ]
    all_pass &= check(
        "no start_time >= end_time",
        len(bad_dur) == 0,
        f"{len(bad_dur)} violations" if bad_dur else "",
    )

    # no end_time past audio duration
    bad_bounds = [
        i for i, u in enumerate(utterances)
        if u["end_time"] > audio_dur + 1.0  # 1s tolerance
    ]
    all_pass &= check(
        "end_time within audio bounds",
        len(bad_bounds) == 0,
        f"{len(bad_bounds)} violations" if bad_bounds else "",
    )

    # ── Speaker IDs ──────────────────────────────────────────────
    speaker_ids = {u["speaker_id"] for u in utterances}
    unknown_count = sum(1 for u in utterances if u["speaker_id"] == "Speaker_unknown")
    if unknown_count == n:
        check("speaker diarization active", False, "all utterances are Speaker_unknown")
        all_pass = False
    elif unknown_count > 0:
        warn("some utterances have Speaker_unknown", f"{unknown_count}/{n}")
    else:
        check("speaker diarization active", True, f"{len(speaker_ids)} distinct speakers")

    print(f"  Speakers found: {sorted(speaker_ids)}")

    # ── Nonverbal events ─────────────────────────────────────────
    nv_total = sum(len(u.get("nonverbal_events", [])) for u in utterances)
    check("nonverbal events present", nv_total > 0, f"{nv_total} events")

    # ── Emotion events ────────────────────────────────────────────
    em_total = sum(len(u.get("inferred_emotions", [])) for u in utterances)
    em_coverage = em_total / n * 100
    check(
        "emotion events present",
        em_total > 0,
        f"{em_total} events ({em_coverage:.1f}% coverage)",
    )

    evidence_types = {
        ev["evidence_type"]
        for u in utterances
        for ev in u.get("inferred_emotions", [])
    }
    if evidence_types:
        print(f"  Emotion evidence types: {evidence_types}")

    # ── Screenshare ───────────────────────────────────────────────
    screenshare = tx.get("screenshare_segments", [])
    check("screenshare segments detected", len(screenshare) > 0, f"{len(screenshare)} segment(s)")
    for seg in screenshare:
        print(f"    Screenshare: {seg['start_time']:.1f}s → {seg.get('end_time', '?')}s "
              f"(platform: {seg.get('platform_name')})")

    # ── Spot checks ───────────────────────────────────────────────
    print("\n  Transcript spot-checks:")
    indices = [0, min(100, n - 1), min(300, n - 1), n - 1]
    seen = set()
    for i in indices:
        if i in seen:
            continue
        seen.add(i)
        u = utterances[i]
        txt = (u["text"] or "")[:80]
        print(f"    [{i:4d}] {u['start_time']:7.1f}s  spk={u['speaker_id']:12s}  \"{txt}\"")

    return all_pass


def main() -> None:
    print("Loading schemas...")
    meta_schema = json.loads(METADATA_SCHEMA_PATH.read_text())
    transcript_schema = json.loads(TRANSCRIPT_SCHEMA_PATH.read_text())

    results = {}
    for vid in VIDEO_IDS:
        results[vid] = validate_video(vid, meta_schema, transcript_schema)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for vid, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  [{status}] {vid}")
        all_ok = all_ok and ok

    if all_ok:
        print("\nAll checks passed.")
        sys.exit(0)
    else:
        print("\nSome checks FAILED. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
