"""
Video-RAG pipeline: scene detection → ASR → embeddings → FAISS DB.

Visual embeddings:  SigLIP2 (google/siglip2-large-patch16-384)
                    3 keyframes per scene, mean-pooled → visual.index
Text embeddings:    BGE-M3  (BAAI/bge-m3)
                    cleaned transcript → CLS-pooled  → text.index

Run:
    python build_vectordb.py

Options:
    --data_dir        Directory with .mp4 files            [data]
    --output_dir      Intermediate artefacts (keyframes, scenes JSON) [output]
    --db_dir          Final FAISS + metadata output        [db]
    --scene_threshold ContentDetector threshold            [27.0]
    --min_scene_len   Min scene length in frames           [15]
    --keyframe_method Keyframe extraction strategy         [fixed3]
    --sample_fps      Uniform sampling FPS (new method)    [1.0]
    --keyframes_per_scene Representatives per scene        [3]
    --cluster_model   HF model for frame embeddings        [apple/MobileCLIP-S2]
    --cluster_batch   Batch size for frame embedding       [16]
    --cluster_seed    Random seed for k-means++            [42]
    --scene_batch_size Number of videos per scene batch    [8]
    --cluster_verbose Detailed logs for kmeans++ selection  [False]
    --hf_cache_dir    HuggingFace cache directory           [.model_cache/hf]
    --offline_models  Load models only from local cache     [False]
    --transcripts_pkl Precomputed transcripts.pkl           [None]
    --whisper_model   faster-whisper size: tiny|base|small [base]
    --language        Whisper language code or None        [None = auto]
    --device          "cpu" or "cuda"                      [cpu]
    --img_batch       Batch size for image embedding       [8]
    --txt_batch       Batch size for text embedding        [32]
    --skip_asr        Skip ASR step (useful for re-runs)   [False]
    --skip_scenes     Reuse existing output/all_scenes.json[False]
    --scene_metadata_json Optional scene-level overrides    [None]
    --film_metadata_json  Optional film metadata JSON       [None]
    --enable_face_detection Run InsightFace on keyframes    [False]
    --faces_dir          Actor references dir               [data/actors]
"""

import json
import argparse
import traceback
import pickle
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import torch

from src.scene_detect import detect_scenes, detect_scenes_batch
from src.asr import transcribe_video
from src.embed import (
    load_visual_model,
    load_text_model,
    build_all_visual_embeddings,
    build_all_text_embeddings,
    save_db,
)
from src.metadata_schema import (
    build_film_metadata_for_videos,
    load_film_metadata_map,
    merge_scene_metadata_overrides,
    normalize_scenes,
    save_film_metadata_json,
)
from src.metadata_store import sync_metadata_to_postgres
from src.face_recognition import (
    enrich_scenes_with_insightface,
    inject_people_context_into_transcripts,
)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


def _discover_videos(data_dir: Path) -> list[Path]:
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Video-RAG FAISS vector database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",        default="data")
    p.add_argument("--output_dir",      default="output")
    p.add_argument("--db_dir",          default="db")
    p.add_argument("--scene_threshold", type=float, default=27.0)
    p.add_argument("--min_scene_len",   type=int,   default=15)
    p.add_argument(
        "--keyframe_method",
        default="fixed3",
        choices=["fixed3", "uniform_clip_kmeans"],
        help="Keyframe extraction strategy",
    )
    p.add_argument("--sample_fps", type=float, default=1.0,
                   help="Candidate frame sampling FPS for uniform_clip_kmeans")
    p.add_argument("--keyframes_per_scene", type=int, default=3,
                   help="Representative keyframes per scene for uniform_clip_kmeans")
    p.add_argument("--cluster_model", default="apple/MobileCLIP-S2",
                   help="HF model id for frame embeddings in uniform_clip_kmeans")
    p.add_argument("--cluster_batch", type=int, default=16,
                   help="Batch size for frame embedding model in uniform_clip_kmeans")
    p.add_argument("--cluster_seed", type=int, default=42,
                   help="Random seed for k-means++ in uniform_clip_kmeans")
    p.add_argument("--scene_batch_size", type=int, default=8,
                   help="Number of videos processed per shared frame-embedder batch")
    p.add_argument("--cluster_verbose", action="store_true",
                   help="Print detailed logs for kmeans++ keyframe selection")
    p.add_argument("--hf_cache_dir", default=".model_cache/hf",
                   help="Local cache dir for HuggingFace models")
    p.add_argument("--offline_models", action="store_true",
                   help="Load models from local cache only (no internet downloads)")
    p.add_argument("--transcripts_pkl", default=None,
                   help="Path to precomputed transcripts.pkl. If set, ASR stage is skipped.")
    p.add_argument("--whisper_model",   default="base",
                   choices=["tiny", "base", "small", "medium", "large-v3"])
    p.add_argument("--language",        default=None,
                   help="ISO-639-1 code, e.g. 'ru'. None = auto-detect.")
    p.add_argument("--device",          default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--img_batch",       type=int, default=8)
    p.add_argument("--txt_batch",       type=int, default=32)
    p.add_argument("--skip_asr",        action="store_true",
                   help="Skip ASR; use transcript='' for all scenes.")
    p.add_argument("--skip_scenes",     action="store_true",
                   help="Reuse output/all_scenes.json if it exists.")
    p.add_argument("--scene_metadata_json", default=None,
                   help="Optional JSON with scene-level metadata overrides.")
    p.add_argument("--film_metadata_json", default=None,
                   help="Optional JSON with film-level metadata (plot_summary, cast_mapping).")
    p.add_argument("--pg_dsn", default=None,
                   help="Optional PostgreSQL DSN for metadata sync.")
    p.add_argument("--pg_schema", default="public",
                   help="PostgreSQL schema for metadata tables.")
    p.add_argument("--enable_face_detection", action="store_true",
                   help="Enable face detection/identification with InsightFace.")
    p.add_argument("--faces_dir", default="data/actors",
                   help="Directory with actor reference photos for InsightFace gallery.")
    p.add_argument("--face_model", default="buffalo_l",
                   help="InsightFace model pack name.")
    p.add_argument("--face_similarity_threshold", type=float, default=0.4,
                   help="Cosine similarity threshold for actor identification.")
    p.add_argument("--face_det_size", type=int, default=640,
                   help="InsightFace detection input size (pixels).")
    p.add_argument("--face_max_per_frame", type=int, default=5,
                   help="Maximum detected faces considered per keyframe.")
    return p.parse_args()


def _resolve_requested_device(requested: str) -> str:
    """Resolve requested device to an available torch device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("  [device] CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"

    if requested == "mps":
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_ok:
            print("  [device] MPS requested but unavailable. Falling back to CPU.")
            return "cpu"

    return requested


def _resolve_stage_devices(requested: str) -> dict[str, str]:
    """
    Resolve per-stage devices:
      - frame_embed: keyframe selection model (Transformers)
      - index_embed: visual/text embedding models (Transformers)
      - asr: faster-whisper (ctranslate2; no MPS support => CPU fallback)
    """
    base = _resolve_requested_device(requested)
    asr_device = "cuda" if base == "cuda" else "cpu"
    return {
        "frame_embed": base,
        "index_embed": base,
        "asr": asr_device,
    }


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _build_transcript_map(transcripts: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_stem: dict[str, list[dict[str, Any]]] = {}
    for raw_key, segs in transcripts.items():
        if not isinstance(segs, list):
            continue
        stem = Path(PurePosixPath(str(raw_key)).name).stem
        cleaned: list[dict[str, Any]] = []
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            start = _to_float(seg.get("start"), 0.0)
            end = _to_float(seg.get("end"), start)
            if end <= start:
                continue
            text = str(seg.get("text") or "").strip()
            cleaned.append({"start": start, "end": end, "text": text})
        cleaned.sort(key=lambda x: (x["start"], x["end"]))
        if cleaned:
            by_stem[stem] = cleaned
    return by_stem


def _attach_transcript_segments_to_scenes(
    scenes: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> int:
    """
    Attach transcript by scene interval overlap.
    If no overlap, fallback to the next transcript block by scene midpoint.
    """
    if not scenes:
        return 0

    scene_refs = sorted(range(len(scenes)), key=lambda i: _to_float(scenes[i].get("start_sec"), 0.0))
    seg_idx = 0
    filled = 0

    for i in scene_refs:
        scene = scenes[i]
        s_start = _to_float(scene.get("start_sec"), 0.0)
        s_end = _to_float(scene.get("end_sec"), s_start)
        if s_end < s_start:
            s_end = s_start

        while seg_idx < len(segments) and segments[seg_idx]["end"] <= s_start:
            seg_idx += 1

        k = seg_idx
        matched: list[str] = []
        while k < len(segments) and segments[k]["start"] < s_end:
            if segments[k]["end"] > s_start and segments[k]["text"]:
                matched.append(segments[k]["text"])
            k += 1

        if matched:
            text = " ".join(matched).strip()
        else:
            midpoint = (s_start + s_end) * 0.5
            k = seg_idx
            while k < len(segments) and segments[k]["end"] <= midpoint:
                k += 1
            if k < len(segments):
                text = segments[k]["text"]
            elif segments:
                text = segments[-1]["text"]
            else:
                text = ""

        scene["transcript"] = text
        scene["transcript_text"] = text
        if text:
            filled += 1

    return filled


def _attach_precomputed_transcripts(
    all_scenes: list[dict[str, Any]],
    transcripts_pkl: str,
) -> tuple[int, int]:
    pkl_path = Path(transcripts_pkl)
    if not pkl_path.exists():
        raise FileNotFoundError(f"transcripts.pkl not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict in transcripts.pkl, got {type(raw)}")

    transcript_map = _build_transcript_map(raw)
    if not transcript_map:
        raise RuntimeError(f"No valid transcript segments found in: {pkl_path}")

    scenes_by_stem: dict[str, list[dict[str, Any]]] = {}
    for scene in all_scenes:
        stem = Path(str(scene.get("video", ""))).stem
        scenes_by_stem.setdefault(stem, []).append(scene)

    total_filled = 0
    videos_with_transcripts = 0
    for stem, scenes in scenes_by_stem.items():
        segs = transcript_map.get(stem, [])
        if not segs:
            for scene in scenes:
                scene["transcript"] = ""
                scene["transcript_text"] = ""
            continue

        videos_with_transcripts += 1
        total_filled += _attach_transcript_segments_to_scenes(scenes, segs)

    return total_filled, videos_with_transcripts


def main() -> None:
    args = parse_args()
    stage_device = _resolve_stage_devices(args.device)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    db_dir     = Path(args.db_dir)
    scenes_json = output_dir / "all_scenes.json"

    video_files = _discover_videos(data_dir)
    if not video_files:
        raise FileNotFoundError(
            f"No supported video files found in '{data_dir}'. "
            f"Supported extensions: {sorted(VIDEO_EXTS)}"
        )
    print(f"Found {len(video_files)} video(s): {[v.name for v in video_files]}")
    print(
        "Resolved devices: "
        f"frame_embed={stage_device['frame_embed']}, "
        f"index_embed={stage_device['index_embed']}, "
        f"asr={stage_device['asr']}"
    )
    film_metadata_external = load_film_metadata_map(args.film_metadata_json)
    film_metadata = build_film_metadata_for_videos(
        [v.name for v in video_files],
        external_map=film_metadata_external,
    )

    # ── Step 1 : Scene detection ─────────────────────────────────────────────
    if args.skip_scenes and scenes_json.exists():
        print(f"\n[1/4] Reusing existing scenes: {scenes_json}")
        with open(scenes_json, encoding="utf-8") as f:
            all_scenes = json.load(f)
    else:
        print("\n[1/4] Detecting scenes …")
        all_scenes = []
        if args.keyframe_method == "uniform_clip_kmeans":
            all_scenes = detect_scenes_batch(
                video_paths=[str(vp) for vp in video_files],
                output_dir=str(output_dir),
                threshold=args.scene_threshold,
                min_scene_len=args.min_scene_len,
                keyframe_method=args.keyframe_method,
                sample_fps=args.sample_fps,
                keyframes_per_scene=args.keyframes_per_scene,
                cluster_model=args.cluster_model,
                cluster_device=stage_device["frame_embed"],
                cluster_batch=args.cluster_batch,
                cluster_seed=args.cluster_seed,
                hf_cache_dir=args.hf_cache_dir,
                offline_models=args.offline_models,
                cluster_verbose=args.cluster_verbose,
                video_batch_size=args.scene_batch_size,
            )
        else:
            for vp in video_files:
                scenes = detect_scenes(
                    str(vp), str(output_dir),
                    threshold=args.scene_threshold,
                    min_scene_len=args.min_scene_len,
                    keyframe_method=args.keyframe_method,
                    sample_fps=args.sample_fps,
                    keyframes_per_scene=args.keyframes_per_scene,
                    cluster_model=args.cluster_model,
                    cluster_device=stage_device["frame_embed"],
                    cluster_batch=args.cluster_batch,
                    cluster_seed=args.cluster_seed,
                    cluster_verbose=args.cluster_verbose,
                    hf_cache_dir=args.hf_cache_dir,
                    offline_models=args.offline_models,
                )
                all_scenes.extend(scenes)

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(scenes_json, "w", encoding="utf-8") as f:
            json.dump(all_scenes, f, ensure_ascii=False, indent=2)
        print(f"  Total: {len(all_scenes)} scenes → {scenes_json}")

    print(f"  Scenes in memory: {len(all_scenes)} (from {len(video_files)} video(s))")

    # ── Step 2 : ASR or precomputed transcripts ─────────────────────────────
    if args.transcripts_pkl:
        print("\n[2/4] Attaching precomputed transcripts from transcripts.pkl …")
        filled, covered_videos = _attach_precomputed_transcripts(all_scenes, args.transcripts_pkl)
        with open(scenes_json, "w", encoding="utf-8") as f:
            json.dump(all_scenes, f, ensure_ascii=False, indent=2)
        print(
            f"  Attached transcripts: {filled}/{len(all_scenes)} scene(s), "
            f"videos covered={covered_videos}"
        )
        print(f"  Checkpoint saved → {scenes_json}")
    elif args.skip_asr:
        print("\n[2/4] Skipping ASR (--skip_asr)")
    else:
        print("\n[2/4] Transcribing audio …")
        for i, vp in enumerate(video_files, 1):
            video_scenes = [s for s in all_scenes if s["video"] == vp.name]
            if not video_scenes:
                print(f"  [{vp.name}] No scenes found, skipping")
                continue
            print(f"  ({i}/{len(video_files)}) Processing {vp.name} — {len(video_scenes)} scenes")
            try:
                transcribe_video(
                    str(vp), video_scenes,
                    model_size=args.whisper_model,
                    language=args.language,
                    device=stage_device["asr"],
                )
            except Exception:
                print(f"  ERROR transcribing {vp.name}, skipping:")
                traceback.print_exc()
                continue

            # Save after every video so partial results are never lost
            with open(scenes_json, "w", encoding="utf-8") as f:
                json.dump(all_scenes, f, ensure_ascii=False, indent=2)
            filled = sum(1 for s in all_scenes if s.get("transcript"))
            print(f"  Checkpoint saved → {scenes_json} ({filled}/{len(all_scenes)} scenes with text)")

    # ── Step 3 : Metadata enrichment ─────────────────────────────────────────
    print("\n[3/4] Enriching scene metadata …")
    normalize_scenes(all_scenes)

    if args.enable_face_detection:
        faces_dir = Path(args.faces_dir)
        if not faces_dir.exists():
            raise FileNotFoundError(
                f"Face gallery directory not found: {faces_dir}. "
                "Create it and add actor reference images."
            )
        stats = enrich_scenes_with_insightface(
            scenes=all_scenes,
            faces_dir=str(faces_dir),
            film_metadata_by_video=film_metadata,
            device=stage_device["frame_embed"],
            model_name=args.face_model,
            similarity_threshold=args.face_similarity_threshold,
            det_size=args.face_det_size,
            max_faces_per_frame=args.face_max_per_frame,
        )
        print(
            "  InsightFace done: "
            f"gallery actors={stats.actors_in_gallery}, "
            f"gallery images={stats.gallery_images}, "
            f"scenes with detected actors={stats.scenes_with_detected_actors}/{len(all_scenes)}"
        )
    else:
        print("  Face detection disabled (enable via --enable_face_detection).")

    all_scenes = merge_scene_metadata_overrides(all_scenes, args.scene_metadata_json)
    normalize_scenes(all_scenes)
    for scene in all_scenes:
        if not (scene.get("characters_in_frame") or []):
            cast_mapping = (
                (film_metadata.get(str(scene.get("video", "")), {}) or {}).get("cast_mapping", {})
            )
            if isinstance(cast_mapping, dict):
                cast_exact = {str(k).strip(): str(v).strip() for k, v in cast_mapping.items() if str(k).strip()}
                cast_folded = {k.casefold(): v for k, v in cast_exact.items()}
                inferred_characters: list[str] = []
                for actor in scene.get("actors_in_frame", []) or []:
                    actor_s = str(actor).strip()
                    if not actor_s:
                        continue
                    character = cast_exact.get(actor_s) or cast_folded.get(actor_s.casefold()) or ""
                    if character and character not in inferred_characters:
                        inferred_characters.append(character)
                if inferred_characters:
                    scene["characters_in_frame"] = inferred_characters

        raw_text = str(
            scene.get("transcript_text") or scene.get("transcript") or ""
        ).strip()
        old_ctx = str(scene.get("face_context_text") or "").strip()
        if old_ctx and raw_text.startswith(old_ctx):
            raw_text = raw_text[len(old_ctx):].lstrip(" .")
        scene["transcript_raw"] = raw_text

    injected = inject_people_context_into_transcripts(all_scenes)
    print(f"  Injected actor/character context into {injected}/{len(all_scenes)} scene transcripts")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(scenes_json, "w", encoding="utf-8") as f:
        json.dump(all_scenes, f, ensure_ascii=False, indent=2)
    print(f"  Enriched scenes saved → {scenes_json}")

    film_metadata_path = save_film_metadata_json(str(db_dir), film_metadata)
    print(f"  Film metadata saved → {film_metadata_path}")

    if args.pg_dsn:
        print(f"  Syncing metadata to PostgreSQL (schema={args.pg_schema}) …")
        sync_metadata_to_postgres(
            dsn=args.pg_dsn,
            scenes=all_scenes,
            film_metadata=film_metadata,
            schema=args.pg_schema,
        )
        print("  PostgreSQL sync done.")

    # ── Step 4 : Embeddings + FAISS ──────────────────────────────────────────
    print("\n[4/4] Building embeddings and FAISS index …")
    print("  Loading visual model (SigLIP2) …")
    vis_model, vis_proc = load_visual_model(
        stage_device["index_embed"],
        cache_dir=args.hf_cache_dir,
        local_files_only=args.offline_models,
    )
    print("  Loading text model (BGE-M3) …")
    txt_model, txt_tok  = load_text_model(
        stage_device["index_embed"],
        cache_dir=args.hf_cache_dir,
        local_files_only=args.offline_models,
    )

    print(f"  Encoding {len(all_scenes)} scenes (visual) …")
    vis_embs = build_all_visual_embeddings(
        all_scenes, vis_model, vis_proc, stage_device["index_embed"]
    )

    print(f"  Encoding {len(all_scenes)} scenes (text) …")
    txt_embs = build_all_text_embeddings(
        all_scenes, txt_model, txt_tok, stage_device["index_embed"], args.txt_batch
    )

    save_db(all_scenes, vis_embs, txt_embs, str(db_dir))

    print("\n✓ Done! Run `python search.py --query \"<your query>\"` to test retrieval.")


if __name__ == "__main__":
    main()
