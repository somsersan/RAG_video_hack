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
    --whisper_model   faster-whisper size: tiny|base|small [base]
    --language        Whisper language code or None        [None = auto]
    --device          "cpu" or "cuda"                      [cpu]
    --img_batch       Batch size for image embedding       [8]
    --txt_batch       Batch size for text embedding        [32]
    --skip_asr        Skip ASR step (useful for re-runs)   [False]
    --skip_scenes     Reuse existing output/all_scenes.json[False]
"""

import json
import argparse
import traceback
from pathlib import Path

from src.scene_detect import detect_scenes
from src.asr import transcribe_video
from src.embed import (
    load_visual_model,
    load_text_model,
    build_all_visual_embeddings,
    build_all_text_embeddings,
    save_db,
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
    p.add_argument("--whisper_model",   default="base",
                   choices=["tiny", "base", "small", "medium", "large-v3"])
    p.add_argument("--language",        default=None,
                   help="ISO-639-1 code, e.g. 'ru'. None = auto-detect.")
    p.add_argument("--device",          default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--img_batch",       type=int, default=8)
    p.add_argument("--txt_batch",       type=int, default=32)
    p.add_argument("--skip_asr",        action="store_true",
                   help="Skip ASR; use transcript='' for all scenes.")
    p.add_argument("--skip_scenes",     action="store_true",
                   help="Reuse output/all_scenes.json if it exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    db_dir     = Path(args.db_dir)
    scenes_json = output_dir / "all_scenes.json"

    video_files = sorted(data_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in '{data_dir}'")
    print(f"Found {len(video_files)} video(s): {[v.name for v in video_files]}")

    # ── Step 1 : Scene detection ─────────────────────────────────────────────
    if args.skip_scenes and scenes_json.exists():
        print(f"\n[1/3] Reusing existing scenes: {scenes_json}")
        with open(scenes_json, encoding="utf-8") as f:
            all_scenes = json.load(f)
    else:
        print("\n[1/3] Detecting scenes …")
        all_scenes = []
        for vp in video_files:
            scenes = detect_scenes(
                str(vp), str(output_dir),
                threshold=args.scene_threshold,
                min_scene_len=args.min_scene_len,
            )
            all_scenes.extend(scenes)

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(scenes_json, "w", encoding="utf-8") as f:
            json.dump(all_scenes, f, ensure_ascii=False, indent=2)
        print(f"  Total: {len(all_scenes)} scenes → {scenes_json}")

    print(f"  Scenes in memory: {len(all_scenes)} (from {len(video_files)} video(s))")

    # ── Step 2 : ASR ────────────────────────────────────────────────────────
    if args.skip_asr:
        print("\n[2/3] Skipping ASR (--skip_asr)")
    else:
        print("\n[2/3] Transcribing audio …")
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
                    device=args.device,
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

    # ── Step 3 : Embeddings + FAISS ──────────────────────────────────────────
    print("\n[3/3] Building embeddings and FAISS index …")
    print("  Loading visual model (SigLIP2) …")
    vis_model, vis_proc = load_visual_model(args.device)
    print("  Loading text model (BGE-M3) …")
    txt_model, txt_tok  = load_text_model(args.device)

    print(f"  Encoding {len(all_scenes)} scenes (visual) …")
    vis_embs = build_all_visual_embeddings(
        all_scenes, vis_model, vis_proc, args.device
    )

    print(f"  Encoding {len(all_scenes)} scenes (text) …")
    txt_embs = build_all_text_embeddings(
        all_scenes, txt_model, txt_tok, args.device, args.txt_batch
    )

    save_db(all_scenes, vis_embs, txt_embs, str(db_dir))

    print("\n✓ Done! Run `python search.py --query \"<your query>\"` to test retrieval.")


if __name__ == "__main__":
    main()
