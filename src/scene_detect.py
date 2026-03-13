"""
Scene detection using PySceneDetect.

Detects scene boundaries, extracts 3 keyframes per scene
(beginning, middle, end), and returns scene metadata as a list of dicts.

Usage (standalone):
    python -m src.scene_detect --video data/shrek.mp4 --output_dir output
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

import cv2
from tqdm import tqdm
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def _save_frame_at_sec(
    cap: cv2.VideoCapture,
    sec: float,
    out_path: Path,
    fallback_sec: float | None = None,
) -> bool:
    """
    Try to save a frame at a given timestamp in seconds.
    If reading fails, optionally tries fallback_sec.
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(str(out_path), frame)
        return True

    if fallback_sec is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, fallback_sec * 1000)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(str(out_path), frame)
            return True

    return False


def detect_scenes(
    video_path: str,
    output_dir: str,
    threshold: float = 27.0,
    min_scene_len: int = 15,  # frames
) -> List[Dict]:
    """
    Detect scenes in a video file and extract 3 keyframes per scene.

    Args:
        video_path:     Path to the input .mp4 file.
        output_dir:     Root output directory. Keyframes go into
                        <output_dir>/keyframes/<video_stem>/.
        threshold:      ContentDetector sensitivity (lower = more splits).
        min_scene_len:  Minimum scene length in frames.

    Returns:
        List of scene dicts::

            {
                "video": "shrek.mp4",
                "scene_id": 0,
                "start_sec": 0.0,
                "end_sec": 4.17,
                "keyframe_paths": [
                    "output/keyframes/shrek/scene_0000_00.jpg",
                    "output/keyframes/shrek/scene_0000_01.jpg",
                    "output/keyframes/shrek/scene_0000_02.jpg"
                ],
                "transcript": "",
            }
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    keyframes_dir = output_dir / "keyframes" / video_path.stem
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    # ── Scene detection ──────────────────────────────────────────────────────
    print(f"  [{video_path.name}] Detecting scene boundaries …")
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    print(f"  [{video_path.name}] Found {len(scene_list)} scene boundaries")

    if not scene_list:
        # Fallback: treat the whole video as one scene
        cap_tmp = cv2.VideoCapture(str(video_path))
        fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 25
        total_frames = cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)
        cap_tmp.release()

        from scenedetect import FrameTimecode
        scene_list = [
            (FrameTimecode(0, fps), FrameTimecode(max(int(total_frames) - 1, 0), fps))
        ]

    # ── Keyframe extraction ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    scenes: List[Dict] = []

    for i, (start_tc, end_tc) in tqdm(enumerate(scene_list), total=len(scene_list)):
        start_sec = start_tc.get_seconds()
        end_sec = end_tc.get_seconds()

        # Иногда end_sec может быть равен start_sec на очень коротких сценах
        duration = max(end_sec - start_sec, 0.001)

        # Берём кадры не на самых границах сцены, а чуть внутри:
        # 15%, 50%, 85% от длины сцены.
        sample_secs = [
            start_sec + duration * 0.15,
            start_sec + duration * 0.50,
            start_sec + duration * 0.85,
        ]

        keyframe_paths = []
        for j, sec in enumerate(sample_secs):
            frame_path = keyframes_dir / f"scene_{i:04d}_{j:02d}.jpg"

            # fallback: если не удалось прочитать, пробуем середину сцены
            ok = _save_frame_at_sec(
                cap=cap,
                sec=sec,
                out_path=frame_path,
                fallback_sec=start_sec + duration * 0.50,
            )

            if ok:
                keyframe_paths.append(str(frame_path))
            else:
                keyframe_paths.append("")

        scenes.append(
            {
                "video": video_path.name,
                "scene_id": i,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "keyframe_paths": keyframe_paths,
                "transcript": "",
            }
        )

    cap.release()
    print(f"  [{video_path.name}] {len(scenes)} scenes detected")
    return scenes


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect scenes and extract keyframes")
    parser.add_argument("--video", required=True, help="Path to .mp4")
    parser.add_argument("--output_dir", default="output", help="Output root dir")
    parser.add_argument("--threshold", type=float, default=27.0)
    parser.add_argument("--min_scene_len", type=int, default=15)
    args = parser.parse_args()

    scenes = detect_scenes(
        args.video, args.output_dir, args.threshold, args.min_scene_len
    )

    out_json = Path(args.output_dir) / f"{Path(args.video).stem}_scenes.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)

    print(f"Saved → {out_json}")