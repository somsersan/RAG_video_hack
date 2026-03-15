"""
VLM-based re-ranking via OpenRouter (OpenAI-compatible API).

For each candidate scene:
  1. Extract N keyframes uniformly from [start_sec, end_sec] via OpenCV.
  2. Send frames + query to the model as base64 image_url content items.
  3. Ask the model to score relevance 1-10 and give a brief explanation.
  4. Re-rank candidates by VLM score (descending).

Usage:
    from src.vlm_rerank import rerank
    hits = rerank(hits, metadata, query, api_key="sk-or-...",
                  video_dir="data", model="google/gemini-2.5-flash-lite")
"""

import base64
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL       = "qwen/qwen3-vl-235b-a22b-instruct"
FRAMES_PER_SCENE    = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

_SCORE_LINE_RE = re.compile(
    r"^\s*SCORE\s*:\s*([1-9]|10)(?:\s*/\s*10)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _parse_score(text: str) -> float:
    m = _SCORE_LINE_RE.search(_strip_thinking(text))
    return float(m.group(1)) if m else 0.0


def _extract_frames_b64(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int = FRAMES_PER_SCENE,
) -> List[str]:
    """Extract n_frames uniformly from [start_sec, end_sec] as base64 JPEGs."""
    cap = cv2.VideoCapture(video_path)
    duration = max(end_sec - start_sec, 0.1)
    timestamps = [
        start_sec + duration * i / max(n_frames - 1, 1)
        for i in range(n_frames)
    ]

    frames_b64: List[str] = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

    cap.release()
    return frames_b64


# ── OpenRouter call ───────────────────────────────────────────────────────────

def _call_openrouter(
    query: str,
    transcript: str,
    frames_b64: List[str],
    client: OpenAI,
    model: str,
    provider: str,
    use_vision: bool,
) -> Tuple[float, str]:
    """Send one scene to OpenRouter and return (score, explanation)."""
    text_prompt = (
        f"Ты помощник для поиска ключевых моментов из видео. ОТВЕЧАЙ ОЧЕНЬ КОРОТКО.\n"
        f"Query: \"{query}\"\n"
        f"Scene transcript: \"{transcript or 'no speech'}\"\n\n"
    )
    if use_vision and frames_b64:
        text_prompt += (
            "Посмотри на кадры из фильма и реши, насколько они соответствуют query.\n"
            "Отвечай СТРОГО в формате (две строки, ничего лишнего):\n"
            "REASON: <одно предложение — почему сцена подходит или не подходит>\n"
            "SCORE: <целое число от 1 до 10>"
        )
    else:
        text_prompt += (
            "Decide how relevant this scene is to the query.\n"
            "Reply STRICTLY in this format (two lines, nothing else):\n"
            "REASON: <one sentence why the scene matches or does not match>\n"
            "SCORE: <integer 1-10>"
        )

    # Build content list: text first, then images
    content: List[Dict] = [{"type": "text", "text": text_prompt}]
    if use_vision and frames_b64:
        for b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=256,
        )
        raw   = resp.choices[0].message.content or ""
        text  = _strip_thinking(raw)
        score = _parse_score(text)
        m = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
        reason = m.group(1).strip() if m else text.replace("\n", " ")[:120]
        if score == 0.0:
            print(f"    [WARN] Could not parse SCORE from: {text[:200]!r}")
        return score, reason
    except Exception as exc:
        return 0.0, f"[error: {exc}]"


# ── Public API ────────────────────────────────────────────────────────────────

def rerank(
    hits: List[Tuple[float, int]],
    metadata: List[Dict],
    query: str,
    api_key: str,
    video_dir: str = "data",
    model: str = DEFAULT_MODEL,
    provider: str = "ollama",
    use_vision: bool = True,
    n_frames: int = FRAMES_PER_SCENE,
    top_k: int = 5,
) -> List[Dict]:
    """
    Re-rank *hits* with a VLM via OpenRouter and return enriched result dicts.
    """
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    video_dir = Path(video_dir)
    results: List[Dict] = []

    for fid, score in tqdm(hits, desc="  VLM re-rank", unit="scene"):
        scene = dict(metadata[int(fid)])
        scene["faiss_score"] = float(score)

        video_path  = str(video_dir / scene["video"])
        frames_b64: List[str] = []

        if use_vision and Path(video_path).exists():
            frames_b64 = _extract_frames_b64(
                video_path, scene["start_sec"], scene["end_sec"], n_frames
            )
        elif use_vision:
            kf = scene.get("keyframe_path") or (
                scene.get("keyframe_paths") or [""]
            )[0]
            if kf and Path(kf).exists():
                with open(kf, "rb") as f:
                    frames_b64 = [base64.b64encode(f.read()).decode("utf-8")]

        vlm_score, vlm_reason = _call_openrouter(
            query, scene.get("transcript", ""),
            frames_b64, client, model, provider, use_vision,
        )
        print(
            f"    Scene {scene['video']} [{scene['start_sec']:.1f}s]: "
            f"VLM score={vlm_score:.1f}, reason={vlm_reason}"
        )
        scene["vlm_score"]  = vlm_score
        scene["vlm_reason"] = vlm_reason
        results.append(scene)

    results.sort(key=lambda x: -x["vlm_score"])
    return results[:top_k]


# ── Frame extraction ──────────────────────────────────────────────────────────

def _extract_frames_b64(
    video_path: str,
    start_sec: float,
    end_sec: float,
    n_frames: int = FRAMES_PER_SCENE,
) -> List[str]:
    """
    Extract *n_frames* uniformly sampled from [start_sec, end_sec].
    Returns list of base64-encoded JPEG strings.
    """
    cap = cv2.VideoCapture(video_path)
    duration = max(end_sec - start_sec, 0.1)
    timestamps = [
        start_sec + duration * i / max(n_frames - 1, 1)
        for i in range(n_frames)
    ]

    frames_b64: List[str] = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        # Encode as JPEG in memory → base64
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))

    cap.release()
    return frames_b64


# ── Ollama call ───────────────────────────────────────────────────────────────

# Only match an explicit "SCORE: N" or "SCORE: N/10" line – nothing else.
# This prevents the regex from firing on numbers inside the REASON text.
_SCORE_LINE_RE = re.compile(
    r"^\s*SCORE\s*:\s*([1-9]|10)(?:\s*/\s*10)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (qwen3, etc)."""
    return _THINK_RE.sub("", text).strip()


def _parse_score(text: str) -> float:
    """
    Extract a 1-10 score from the *explicit* SCORE: line.
    Deliberately does NOT fall back to scanning arbitrary digits –
    that fallback was the main source of score/reason mismatches.
    Returns 0.0 if the line is absent (caller can decide what to do).
    """
    clean = _strip_thinking(text)
    m = _SCORE_LINE_RE.search(clean)
    if m:
        return float(m.group(1))
    return 0.0
