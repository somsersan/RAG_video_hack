"""
Optional PostgreSQL persistence for film/scene metadata.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Any

from src.metadata_schema import transcript_text


def _import_psycopg():
    try:
        import psycopg  # type: ignore
        return psycopg
    except Exception as exc:
        raise RuntimeError(
            "PostgreSQL sync requested, but psycopg is not installed. "
            "Install dependency: pip install psycopg[binary]"
        ) from exc


def _scene_uid(scene: Dict[str, Any]) -> str:
    video = str(scene.get("video", ""))
    scene_id = scene.get("scene_id")
    start_sec = float(scene.get("start_sec", 0.0))
    end_sec = float(scene.get("end_sec", 0.0))
    return f"{video}:{scene_id}:{start_sec:.3f}:{end_sec:.3f}"


def sync_metadata_to_postgres(
    dsn: str,
    scenes: List[Dict[str, Any]],
    film_metadata: Dict[str, Dict[str, Any]],
    schema: str = "public",
) -> None:
    """
    Persist metadata to PostgreSQL. Idempotent via UPSERT.
    """
    psycopg = _import_psycopg()
    schema = schema.strip() or "public"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
        raise ValueError(
            f"Invalid PostgreSQL schema name: {schema!r}. "
            "Use letters, numbers and underscores only."
        )

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {schema}.film_metadata (
                    video_id TEXT PRIMARY KEY,
                    plot_summary TEXT NOT NULL DEFAULT '',
                    cast_mapping JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {schema}.scene_metadata (
                    scene_uid TEXT PRIMARY KEY,
                    faiss_id INTEGER,
                    video_id TEXT NOT NULL,
                    scene_id INTEGER,
                    season_number INTEGER,
                    episode_number INTEGER,
                    start_sec DOUBLE PRECISION NOT NULL,
                    end_sec DOUBLE PRECISION NOT NULL,
                    keyframe_path TEXT,
                    transcript_text TEXT NOT NULL DEFAULT '',
                    characters_in_frame TEXT[] NOT NULL DEFAULT '{{}}',
                    actors_in_frame TEXT[] NOT NULL DEFAULT '{{}}',
                    raw_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_scene_metadata_video ON {schema}.scene_metadata (video_id)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_scene_metadata_faiss ON {schema}.scene_metadata (faiss_id)"
            )

            film_sql = f"""
                INSERT INTO {schema}.film_metadata (video_id, plot_summary, cast_mapping, updated_at)
                VALUES (%s, %s, %s::jsonb, NOW())
                ON CONFLICT (video_id) DO UPDATE SET
                    plot_summary = EXCLUDED.plot_summary,
                    cast_mapping = EXCLUDED.cast_mapping,
                    updated_at = NOW()
            """
            for video_id, payload in film_metadata.items():
                cur.execute(
                    film_sql,
                    (
                        video_id,
                        str(payload.get("plot_summary", "")),
                        json.dumps(payload.get("cast_mapping", {}), ensure_ascii=False),
                    ),
                )

            scene_sql = f"""
                INSERT INTO {schema}.scene_metadata (
                    scene_uid, faiss_id, video_id, scene_id, season_number, episode_number,
                    start_sec, end_sec, keyframe_path, transcript_text,
                    characters_in_frame, actors_in_frame, raw_json, updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s::jsonb, NOW()
                )
                ON CONFLICT (scene_uid) DO UPDATE SET
                    faiss_id = EXCLUDED.faiss_id,
                    video_id = EXCLUDED.video_id,
                    scene_id = EXCLUDED.scene_id,
                    season_number = EXCLUDED.season_number,
                    episode_number = EXCLUDED.episode_number,
                    start_sec = EXCLUDED.start_sec,
                    end_sec = EXCLUDED.end_sec,
                    keyframe_path = EXCLUDED.keyframe_path,
                    transcript_text = EXCLUDED.transcript_text,
                    characters_in_frame = EXCLUDED.characters_in_frame,
                    actors_in_frame = EXCLUDED.actors_in_frame,
                    raw_json = EXCLUDED.raw_json,
                    updated_at = NOW()
            """
            for scene in scenes:
                cur.execute(
                    scene_sql,
                    (
                        _scene_uid(scene),
                        scene.get("faiss_id"),
                        scene.get("video", ""),
                        scene.get("scene_id"),
                        scene.get("season_number"),
                        scene.get("episode_number"),
                        float(scene.get("start_sec", 0.0)),
                        float(scene.get("end_sec", 0.0)),
                        scene.get("keyframe_path"),
                        transcript_text(scene),
                        scene.get("characters_in_frame", []) or [],
                        scene.get("actors_in_frame", []) or [],
                        json.dumps(scene, ensure_ascii=False),
                    ),
                )
        conn.commit()
