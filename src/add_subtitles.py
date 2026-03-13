"""
Merges an SRT subtitle file into a scenes JSON file.

For each scene the script collects every subtitle whose time range
overlaps with [start_sec, end_sec] and concatenates the text into
the 'transcript' field.

Usage:
    python add_subtitles.py --srt data/shrek.srt --json output/shrek_scenes.json
    python add_subtitles.py --srt data/shrek.srt --json output/shrek_scenes.json --out output/shrek_scenes_sub.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

_TIMECODE_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _tc_to_sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _read_text_autoenc(path: str) -> str:
    """Try common encodings; return decoded text of the first that works cleanly."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1251", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1")  # last-resort, never raises


def parse_srt(path: str) -> list[dict]:
    """Return list of {start, end, text} dicts parsed from an SRT file."""
    text = _read_text_autoenc(path)
    # Normalize line endings (\r\n and \r -> \n) so block-split works on any OS
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", text.strip())
    subtitles = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        # Find the timecode line (may not be the second line in malformed files)
        tc_line = None
        tc_idx = -1
        for i, line in enumerate(lines):
            m = _TIMECODE_RE.search(line)
            if m:
                tc_line = m
                tc_idx = i
                break
        if tc_line is None:
            continue
        start = _tc_to_sec(*tc_line.group(1, 2, 3, 4))
        end = _tc_to_sec(*tc_line.group(5, 6, 7, 8))
        text_lines = lines[tc_idx + 1 :]
        text = " ".join(l.strip() for l in text_lines if l.strip())
        # Strip simple HTML/ASS tags like <i>, {\\an8}, etc.
        text = re.sub(r"<[^>]+>|\{[^}]+\}", "", text).strip()
        if text:
            subtitles.append({"start": start, "end": end, "text": text})
    return subtitles


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge(scenes: list[dict], subtitles: list[dict]) -> list[dict]:
    """
    For each scene collect subtitles that overlap with the scene window.
    Adds:
      - 'subtitles': list of {start, end, text} with original timecodes
      - 'transcript': all subtitle texts joined into one string
    Overlap condition: subtitle.start < scene.end AND subtitle.end > scene.start
    """
    for scene in scenes:
        s_start = scene.get("start_sec", 0)
        s_end = scene.get("end_sec", 0)
        matched = [
            {"start": sub["start"], "end": sub["end"], "text": sub["text"]}
            for sub in subtitles
            if sub["start"] < s_end and sub["end"] > s_start
        ]
        scene["subtitles"] = matched
        scene["transcript"] = " ".join(s["text"] for s in matched)
    return scenes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Add SRT subtitles to scenes JSON.")
    parser.add_argument("--srt", required=True, help="Path to the .srt subtitle file")
    parser.add_argument("--json", required=True, help="Path to the scenes JSON file")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: overwrite --json in-place)",
    )
    args = parser.parse_args()

    srt_path = args.srt
    json_path = args.json
    out_path = args.out or json_path

    if not Path(srt_path).exists():
        print(f"ERROR: SRT file not found: {srt_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(json_path).exists():
        print(f"ERROR: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    subtitles = parse_srt(srt_path)
    print(f"Parsed {len(subtitles)} subtitle cues from '{srt_path}'")

    with open(json_path, encoding="utf-8") as f:
        scenes = json.load(f)
    print(f"Loaded {len(scenes)} scenes from '{json_path}'")

    scenes = merge(scenes, subtitles)

    filled = sum(1 for s in scenes if s.get("transcript"))
    print(f"Scenes with transcript: {filled}/{len(scenes)}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    print(f"Saved to '{out_path}'")


if __name__ == "__main__":
    main()
