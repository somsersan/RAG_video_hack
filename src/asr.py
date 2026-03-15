"""
ASR transcription with faster-whisper.

Strategy:
  1. Extract full mono-16kHz audio from video once via ffmpeg.
  2. Spawn src/_whisper_worker.py as a subprocess that streams JSONL to stdout.
     Running Whisper in a subprocess fully isolates ctranslate2's sys.exit()
     from the main pipeline process — this was the root cause of silent crashes.
  3. Assign each segment to the scene whose window contains the midpoint.

Usage (standalone):
    python -m src.asr --video data/shrek.mp4 \
                      --scenes output/all_scenes.json \
                      --out    output/all_scenes.json \
                      --model  base --device cuda \
                      --whisper_cache_dir .model_cache/whisper
"""

import json
import os
import shutil
import subprocess
import tempfile
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional

from tqdm import tqdm


# ── Audio extraction ──────────────────────────────────────────────────────────

def _extract_audio(video_path: str, audio_path: str) -> None:
    """Use ffmpeg to extract mono 16 kHz WAV from video."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg binary not found in PATH. "
            "Install ffmpeg and ensure it is available from terminal.\n"
            "macOS (Homebrew): brew install ffmpeg\n"
            "Ubuntu/Debian: sudo apt-get install ffmpeg"
        )
    subprocess.run(
        [
            ffmpeg_bin, "-y",
            "-i", video_path,
            "-ac", "1",
            "-ar", "16000",
            "-vn",
            audio_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ── Segment → scene assignment ────────────────────────────────────────────────

def _assign_segments_to_scenes(
    segments: List[Dict],
    scenes: List[Dict],
) -> None:
    """
    Assign each segment to the scene whose [start_sec, end_sec) contains
    the segment midpoint. Fills scene["transcript"] in-place.
    """
    for scene in scenes:
        scene["transcript"] = ""
        scene["transcript_text"] = ""

    for seg in segments:
        seg_mid = (seg["start"] + seg["end"]) / 2.0
        for scene in scenes:
            if scene["start_sec"] <= seg_mid < scene["end_sec"]:
                t = seg["text"].strip()
                if t:
                    prev = scene["transcript"]
                    scene["transcript"] = (prev + " " + t).strip()
                    scene["transcript_text"] = scene["transcript"]
                break


# ── Subprocess Whisper ────────────────────────────────────────────────────────

def _run_whisper_subprocess(
    audio_path: str,
    model_size: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    label: str,
    download_root: Optional[str] = None,
    local_files_only: bool = False,
) -> List[Dict]:
    """
    Spawn _whisper_worker.py and collect segments from its JSONL stdout.
    Returns list of {"start", "end", "text"} dicts.
    Raises RuntimeError on worker failure.
    """
    worker = Path(__file__).parent / "_whisper_worker.py"
    cmd = [
        sys.executable, str(worker),
        "--audio",        audio_path,
        "--model_size",   model_size,
        "--language",     str(language),   # worker handles "None" string
        "--device",       device,
        "--compute_type", compute_type,
    ]
    if download_root:
        cmd.extend(["--download_root", download_root])
    if local_files_only:
        cmd.append("--local_files_only")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONLEGACYWINDOWSSTDIO"] = "0"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    segments: List[Dict] = []
    pbar = None
    received_done = False

    for raw_line in proc.stdout:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        if obj["type"] == "info":
            speech_dur = obj.get("duration_after_vad") or obj["duration"]
            print(
                f"  [{label}] Language: {obj['language']} "
                f"(prob={obj['language_probability']:.2f}), "
                f"speech: {speech_dur:.0f}s"
            )
            pbar = tqdm(
                total=speech_dur,
                desc=f"  [{label}] whisper",
                unit="sec",
                dynamic_ncols=True,
            )

        elif obj["type"] == "segment":
            segments.append(obj)
            if pbar is not None:
                pbar.update(obj["end"] - obj["start"])

        elif obj["type"] == "done":
            if pbar is not None:
                pbar.n = pbar.total
                pbar.refresh()
                pbar.close()
                pbar = None
            received_done = True
            print(f"  [{label}] {obj['total']} segments received")

        elif obj["type"] == "error":
            if pbar is not None:
                pbar.close()
            raise RuntimeError(f"Whisper worker error: {obj['message']}")

    proc.wait()

    if pbar is not None:
        pbar.close()

    # ctranslate2 / CRT sometimes crashes during C++ cleanup with codes like
    # 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN) after all data is already sent.
    # If we received the "done" message, treat any exit code as success.
    if proc.returncode != 0 and not received_done:
        stderr_out = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(
            f"Whisper worker exited with code {proc.returncode}.\n{stderr_out}"
        )

    return segments


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe_video(
    video_path: str,
    scenes: List[Dict],
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "cpu",
    compute_type: str = "int8",
    whisper_cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> List[Dict]:
    """
    Transcribe *video_path* and attach transcripts to each entry in *scenes*.
    Modifies scenes in-place and returns them.
    """
    video_path = Path(video_path)
    label = video_path.name

    # int8 is CPU-only in ctranslate2
    if compute_type == "int8" and device == "cuda":
        compute_type = "float16"

    print(f"  [{label}] Extracting audio …")
    if whisper_cache_dir:
        Path(whisper_cache_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        _extract_audio(str(video_path), audio_path)
        print(f"  [{label}] Starting Whisper-{model_size} subprocess (device={device}) …")
        segments = _run_whisper_subprocess(
            audio_path,
            model_size,
            language,
            device,
            compute_type,
            label,
            download_root=whisper_cache_dir,
            local_files_only=local_files_only,
        )
    finally:
        Path(audio_path).unlink(missing_ok=True)

    print(f"  [{label}] Assigning {len(segments)} segments to {len(scenes)} scenes …")
    _assign_segments_to_scenes(segments, scenes)
    filled = sum(1 for s in scenes if s["transcript"])
    print(f"  [{label}] Done — {filled}/{len(scenes)} scenes have transcript text")

    return scenes


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe video and attach to scenes")
    parser.add_argument("--video",    required=True)
    parser.add_argument("--scenes",   required=True)
    parser.add_argument("--out",      default=None)
    parser.add_argument("--model",    default="base")
    parser.add_argument("--language", default=None)
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--whisper_cache_dir", default=".model_cache/whisper")
    parser.add_argument("--offline_models", action="store_true")
    args = parser.parse_args()

    with open(args.scenes, encoding="utf-8") as f:
        scenes = json.load(f)

    video_name   = Path(args.video).name
    video_scenes = [s for s in scenes if s["video"] == video_name]

    transcribe_video(
        args.video,
        video_scenes,
        args.model,
        args.language,
        args.device,
        whisper_cache_dir=args.whisper_cache_dir,
        local_files_only=args.offline_models,
    )

    out_path = args.out or args.scenes
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}")
