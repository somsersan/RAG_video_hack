"""
Metadata normalization and enrichment helpers.

This module keeps metadata evolution backward-compatible:
- legacy field: transcript
- new field:    transcript_text
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any


SCENE_METADATA_FIELDS = {
    "season_number",
    "episode_number",
    "characters_in_frame",
    "actors_in_frame",
    "transcript_text",
}


def _as_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def transcript_text(scene: Dict[str, Any]) -> str:
    """Return transcript text with legacy fallback."""
    t = scene.get("transcript_text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    legacy = scene.get("transcript")
    if isinstance(legacy, str):
        return legacy.strip()
    return ""


def normalize_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize one scene dict in-place and return it.

    Ensures new metadata fields exist while preserving compatibility.
    """
    txt = transcript_text(scene)
    scene["transcript_text"] = txt
    scene["transcript"] = txt  # keep old field used by existing code
    scene["season_number"] = _as_optional_int(scene.get("season_number"))
    scene["episode_number"] = _as_optional_int(scene.get("episode_number"))
    scene["characters_in_frame"] = _as_list_of_str(scene.get("characters_in_frame"))
    scene["actors_in_frame"] = _as_list_of_str(scene.get("actors_in_frame"))
    return scene


def normalize_scenes(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for scene in scenes:
        normalize_scene(scene)
    return scenes


def _make_scene_key(scene: Dict[str, Any]) -> tuple[str, int | None]:
    return (str(scene.get("video", "")), _as_optional_int(scene.get("scene_id")))


def merge_scene_metadata_overrides(
    scenes: List[Dict[str, Any]],
    overrides_path: str | None,
) -> List[Dict[str, Any]]:
    """
    Merge optional scene-level overrides into detected scene metadata.

    Expected JSON formats:
    1) list[dict]
    2) {"scenes": list[dict]}
    Each row should contain at least:
    - video
    - scene_id
    Optional fields:
    - season_number, episode_number
    - characters_in_frame, actors_in_frame
    - transcript_text
    """
    if not overrides_path:
        return scenes

    path = Path(overrides_path)
    if not path.exists():
        raise FileNotFoundError(f"Scene metadata file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = raw.get("scenes", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(
            "Scene metadata JSON must be a list or {'scenes': [...]} structure."
        )

    index = {_make_scene_key(scene): scene for scene in scenes}
    merged = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("video", "")), _as_optional_int(row.get("scene_id")))
        target = index.get(key)
        if target is None:
            continue
        for field in SCENE_METADATA_FIELDS:
            if field in row:
                target[field] = row[field]
        normalize_scene(target)
        merged += 1

    print(f"Merged metadata overrides for {merged} scene(s) from {path}")
    return scenes


def load_film_metadata_map(path: str | None) -> Dict[str, Dict[str, Any]]:
    """
    Load optional film-level metadata mapping by video filename.

    Supported formats:
    1) {"video.mp4": {"plot_summary": "...", "cast_mapping": {...}}, ...}
    2) {"films": [{"video": "video.mp4", "plot_summary": "...", "cast_mapping": {...}}]}
    3) [{"video": "video.mp4", "plot_summary": "...", "cast_mapping": {...}}]
    """
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Film metadata file not found: {p}")

    with open(p, encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict) and "films" in raw and isinstance(raw["films"], list):
        rows = raw["films"]
    elif isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        for video, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            out[str(video)] = {
                "plot_summary": str(payload.get("plot_summary", "")).strip(),
                "cast_mapping": _normalize_cast(payload.get("cast_mapping", {})),
            }
        return out
    else:
        raise ValueError("Unsupported film metadata JSON format.")

    for row in rows:
        if not isinstance(row, dict):
            continue
        video = str(row.get("video", "")).strip()
        if not video:
            continue
        out[video] = {
            "plot_summary": str(row.get("plot_summary", "")).strip(),
            "cast_mapping": _normalize_cast(row.get("cast_mapping", {})),
        }
    return out


def _normalize_cast(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    cast: Dict[str, str] = {}
    for actor, character in value.items():
        actor_s = str(actor).strip()
        character_s = str(character).strip()
        if actor_s:
            cast[actor_s] = character_s
    return cast


def build_film_metadata_for_videos(
    video_names: List[str],
    external_map: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a complete mapping for all indexed videos.
    Missing videos get empty defaults to keep schema consistent.
    """
    external_map = external_map or {}
    out: Dict[str, Dict[str, Any]] = {}
    for video in video_names:
        payload = external_map.get(video, {})
        out[video] = {
            "plot_summary": str(payload.get("plot_summary", "")).strip(),
            "cast_mapping": _normalize_cast(payload.get("cast_mapping", {})),
        }
    return out


def save_film_metadata_json(
    output_dir: str,
    film_metadata: Dict[str, Dict[str, Any]],
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "film_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(film_metadata, f, ensure_ascii=False, indent=2)
    return path
