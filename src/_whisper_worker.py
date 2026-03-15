"""
Worker script: runs faster-whisper and streams results as JSONL to stdout.

Called by src/asr.py as a subprocess — this isolates ctranslate2's
aggressive sys.exit() behaviour from the main pipeline process.

Output format (one JSON object per line):
  {"type": "info",    "language": "ru", "language_probability": 0.95,
   "duration": 4859.3, "duration_after_vad": 4200.1}
  {"type": "segment", "start": 0.0, "end": 2.4, "text": "Привет"}
  ...
  {"type": "done", "total": 2156}

Any exception is written as:
  {"type": "error", "message": "..."}
and the process exits with code 1.
"""

import json
import sys
import io
import argparse

# Force UTF-8 stdout regardless of Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio",        required=True)
    p.add_argument("--model_size",   default="base")
    p.add_argument("--language",     default=None)
    p.add_argument("--device",       default="cpu")
    p.add_argument("--compute_type", default="int8")
    p.add_argument("--download_root", default=None)
    p.add_argument("--local_files_only", action="store_true")
    args = p.parse_args()

    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            args.model_size,
            device=args.device,
            compute_type=args.compute_type,
            download_root=args.download_root,
            local_files_only=args.local_files_only,
        )
        segments_gen, info = model.transcribe(
            args.audio,
            language=args.language if args.language != "None" else None,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        # ── Header ────────────────────────────────────────────────────────────
        print(json.dumps({
            "type":                 "info",
            "language":             info.language,
            "language_probability": info.language_probability,
            "duration":             info.duration,
            "duration_after_vad":   getattr(info, "duration_after_vad", info.duration),
        }, ensure_ascii=False), flush=True)

        # ── Segments (streamed) ───────────────────────────────────────────────
        count = 0
        for seg in segments_gen:
            print(json.dumps({
                "type":  "segment",
                "start": seg.start,
                "end":   seg.end,
                "text":  seg.text,
            }, ensure_ascii=False), flush=True)
            count += 1

        print(json.dumps({"type": "done", "total": count}), flush=True)

    except Exception as exc:
        print(json.dumps({"type": "error", "message": str(exc)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
