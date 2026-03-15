"""
Face detection/identification utilities powered by InsightFace.

Gallery layout (default):
  data/actors/
    Mike Myers/
      ref1.jpg
      ref2.png
    Eddie Murphy/
      ref1.jpg

Also supported:
  data/actors/Mike Myers.jpg
  data/actors/Eddie Murphy.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _l2_norm(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def _import_insightface_face_analysis():
    try:
        from insightface.app import FaceAnalysis  # type: ignore
        return FaceAnalysis
    except Exception as exc:
        raise RuntimeError(
            "Face detection is enabled, but InsightFace is unavailable. "
            "Install dependencies: pip install insightface onnxruntime"
        ) from exc


def _discover_actor_images(actors_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if not actors_dir.exists():
        return out

    for child in sorted(actors_dir.iterdir()):
        if child.is_dir():
            files = sorted(
                p for p in child.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            if files:
                out[child.name] = files
            continue

        if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
            actor = child.stem.strip()
            if actor:
                out.setdefault(actor, []).append(child)

    return out


def _extract_face_rows(face_analyzer: Any, frame_bgr: np.ndarray, max_faces: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for face in face_analyzer.get(frame_bgr):
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        if emb is None:
            continue

        emb_norm = _l2_norm(np.asarray(emb, dtype=np.float32))
        if emb_norm.size == 0:
            continue

        bbox_raw = getattr(face, "bbox", None)
        if bbox_raw is not None:
            bbox = [float(x) for x in np.asarray(bbox_raw).reshape(-1).tolist()[:4]]
            if len(bbox) == 4:
                area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
            else:
                bbox = []
                area = 0.0
        else:
            bbox = []
            area = 0.0

        rows.append({"embedding": emb_norm, "bbox": bbox, "area": area})

    rows.sort(key=lambda r: -float(r["area"]))
    if max_faces > 0:
        rows = rows[:max_faces]
    return rows


def _make_cast_lookup(cast_mapping: dict[str, str] | None) -> tuple[dict[str, str], dict[str, str]]:
    cast_mapping = cast_mapping or {}
    exact: dict[str, str] = {}
    folded: dict[str, str] = {}
    for actor, character in cast_mapping.items():
        actor_s = str(actor).strip()
        character_s = str(character).strip()
        if not actor_s:
            continue
        exact[actor_s] = character_s
        folded[actor_s.casefold()] = character_s
    return exact, folded


def build_people_context_text(actors: list[str], characters: list[str]) -> str:
    parts: list[str] = []
    if actors:
        parts.append(f"Актеры в кадре: {', '.join(actors)}")
    if characters:
        parts.append(f"Персонажи в кадре: {', '.join(characters)}")
    return ". ".join(parts).strip()


def inject_people_context_into_transcripts(scenes: list[dict[str, Any]]) -> int:
    """
    Prefix scene transcripts with detected actor/character names.
    Keeps original transcript in transcript_raw.
    """
    injected = 0
    for scene in scenes:
        raw = str(
            scene.get("transcript_raw")
            or scene.get("transcript_text")
            or scene.get("transcript")
            or ""
        ).strip()
        scene["transcript_raw"] = raw

        actors = [str(x).strip() for x in (scene.get("actors_in_frame") or []) if str(x).strip()]
        characters = [str(x).strip() for x in (scene.get("characters_in_frame") or []) if str(x).strip()]
        context = build_people_context_text(actors, characters)
        scene["face_context_text"] = context

        if context:
            merged = f"{context}. {raw}".strip()
            injected += 1
        else:
            merged = raw

        scene["transcript_text"] = merged
        scene["transcript"] = merged
    return injected


@dataclass
class FaceEnrichmentStats:
    actors_in_gallery: int
    gallery_images: int
    scenes_with_detected_actors: int


def enrich_scenes_with_insightface(
    scenes: list[dict[str, Any]],
    faces_dir: str,
    film_metadata_by_video: dict[str, dict[str, Any]],
    device: str = "cpu",
    model_name: str = "buffalo_l",
    similarity_threshold: float = 0.4,
    det_size: int = 640,
    max_faces_per_frame: int = 5,
) -> FaceEnrichmentStats:
    """
    Detect/identify actors on scene keyframes and fill:
      - actors_in_frame
      - characters_in_frame
      - face_matches
    """
    faces_path = Path(faces_dir)
    actor_images = _discover_actor_images(faces_path)
    if not actor_images:
        raise RuntimeError(f"No actor reference images found in: {faces_path}")

    FaceAnalysis = _import_insightface_face_analysis()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(int(det_size), int(det_size)))

    actor_names: list[str] = []
    actor_embs: list[np.ndarray] = []
    total_gallery_images = 0

    for actor, paths in actor_images.items():
        actor_face_embs: list[np.ndarray] = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            rows = _extract_face_rows(app, img, max_faces=max_faces_per_frame)
            if not rows:
                continue
            actor_face_embs.append(np.asarray(rows[0]["embedding"], dtype=np.float32))
            total_gallery_images += 1

        if not actor_face_embs:
            continue

        centroid = _l2_norm(np.mean(np.vstack(actor_face_embs), axis=0))
        if centroid.size == 0:
            continue
        actor_names.append(actor)
        actor_embs.append(centroid.astype(np.float32))

    if not actor_embs:
        raise RuntimeError(
            f"Could not build actor gallery embeddings from: {faces_path}. "
            "Check reference images and face visibility."
        )

    gallery_matrix = np.vstack(actor_embs).astype(np.float32)
    scenes_with_detected = 0

    for scene in scenes:
        score_by_actor: dict[str, float] = {}
        all_matches: list[dict[str, Any]] = []

        for kf in [str(p) for p in (scene.get("keyframe_paths") or []) if str(p).strip()]:
            frame = cv2.imread(kf)
            if frame is None:
                continue
            rows = _extract_face_rows(app, frame, max_faces=max_faces_per_frame)
            for row in rows:
                emb = np.asarray(row["embedding"], dtype=np.float32).reshape(-1)
                scores = gallery_matrix @ emb
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])
                if best_score < similarity_threshold:
                    continue
                actor = actor_names[best_idx]
                score_by_actor[actor] = max(score_by_actor.get(actor, -1.0), best_score)
                all_matches.append(
                    {
                        "actor": actor,
                        "similarity": round(best_score, 4),
                        "bbox": row.get("bbox", []),
                        "keyframe_path": kf,
                    }
                )

        ordered_actors = sorted(score_by_actor.keys(), key=lambda a: (-score_by_actor[a], a.casefold()))
        cast_mapping = (film_metadata_by_video.get(str(scene.get("video", "")), {}) or {}).get("cast_mapping", {})
        cast_exact, cast_folded = _make_cast_lookup(cast_mapping if isinstance(cast_mapping, dict) else {})

        characters: list[str] = []
        for actor in ordered_actors:
            character = cast_exact.get(actor) or cast_folded.get(actor.casefold()) or ""
            if character and character not in characters:
                characters.append(character)

        scene["actors_in_frame"] = ordered_actors
        scene["characters_in_frame"] = characters
        scene["face_matches"] = all_matches

        if ordered_actors:
            scenes_with_detected += 1

    return FaceEnrichmentStats(
        actors_in_gallery=len(actor_names),
        gallery_images=total_gallery_images,
        scenes_with_detected_actors=scenes_with_detected,
    )
