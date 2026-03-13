"""
Gradio chat interface for Video-RAG search with VLM re-ranking.

Features
--------
* Query → LLM decomposition (visual_query + text_query + weights) via OpenRouter
* SigLIP2 visual search + BGE-M3 text search over FAISS
* Weighted RRF fusion → VLM re-ranking (OpenRouter) → top-K scenes
* Click any result to watch the extracted video clip right in the browser

Run:
    python app.py
    python app.py --db_dir db --data_dir data --device cpu

API key is read from OPENROUTER_API_KEY env var or .env file.
"""

import argparse
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import gradio as gr
from openai import OpenAI
from transformers import AutoProcessor, AutoModel, AutoTokenizer

from src.embed import (
    load_visual_model, load_text_model,
    encode_visual_query, encode_text_query,
    load_db,
)
from src.vlm_rerank import rerank, _strip_thinking
from search import search_index, weighted_reciprocal_rank_fusion


# ── Load .env ───────────────────────────────────────────────────────────────────────

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL  = "google/gemini-2.5-flash-lite"


# ── LLM query rewriting ─────────────────────────────────────────────────────

QUERY_REWRITE_PROMPT = (
    "You are a search query analyst for a video archive. "
    "Given a user query, decompose it into two sub-queries with weights.\n\n"
    "Rules:\n"
    "- visual_query: short English phrase describing VISIBLE content (actions, objects, setting)\n"
    "- text_query: short Russian phrase for DIALOGUE, speech, or plot keywords\n"
    "- visual_weight + text_weight must sum to 1.0\n"
    "- If the query is mostly visual, set visual_weight higher; if dialogue/plot — text_weight higher\n\n"
    "Return ONLY a JSON object with exactly these keys: "
    "visual_query, text_query, visual_weight, text_weight"
)


def _fallback_query_plan(query: str) -> dict:
    return {
        "visual_query": query,
        "text_query": query,
        "visual_weight": 0.5,
        "text_weight": 0.5,
    }


def _extract_json_candidate(text: str) -> str:
    """Extract first {...} JSON block from a model response string."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text.strip()


def _normalize_query_plan(result: dict, query: str) -> dict:
    """Validate and normalise a parsed query plan dict; falls back gracefully."""
    base = _fallback_query_plan(query)

    visual_query = result.get("visual_query") or base["visual_query"]
    text_query   = result.get("text_query")   or base["text_query"]

    try:
        visual_weight = float(result.get("visual_weight", 0.5))
    except Exception:
        visual_weight = 0.5
    try:
        text_weight = float(result.get("text_weight", 0.5))
    except Exception:
        text_weight = 0.5

    if not math.isfinite(visual_weight):
        visual_weight = 0.5
    if not math.isfinite(text_weight):
        text_weight = 0.5

    visual_weight = max(0.0, min(1.0, visual_weight))
    text_weight   = max(0.0, min(1.0, text_weight))

    s = visual_weight + text_weight
    if s <= 0:
        visual_weight, text_weight = 0.5, 0.5
    else:
        visual_weight /= s
        text_weight   /= s

    if not isinstance(visual_query, str) or not visual_query.strip():
        visual_query = base["visual_query"]
    if not isinstance(text_query, str) or not text_query.strip():
        text_query = base["text_query"]

    return {
        "visual_query":  visual_query.strip(),
        "text_query":    text_query.strip(),
        "visual_weight": round(visual_weight, 4),
        "text_weight":   round(text_weight, 4),
    }


def rewrite_query(
    query: str,
    model: str,
    api_key: str,
) -> dict:
    """
    Ask an OpenRouter LLM to decompose the query into visual/text sub-queries.
    Returns a query-plan dict: {visual_query, text_query, visual_weight, text_weight}.
    Falls back to equal-weight copies of the original query on any error.
    """
    fallback = _fallback_query_plan(query)
    if not api_key:
        return fallback
    try:
        client  = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        resp    = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_PROMPT},
                {"role": "user",   "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=120,
        )
        content   = resp.choices[0].message.content or ""
        json_text = _extract_json_candidate(_strip_thinking(content))
        parsed    = json.loads(json_text)
        if not isinstance(parsed, dict):
            return fallback
        return _normalize_query_plan(parsed, query)
    except Exception:
        return fallback


# ── Globals (loaded once at startup) ─────────────────────────────────────────

VIS_IDX    = None
TXT_IDX    = None
METADATA   = None
VIS_MODEL  = None   # SigLIP2 for visual search
VIS_PROC   = None
TXT_MODEL  = None   # BGE-M3 for text search
TXT_TOK    = None
DEVICE     = "cpu"
DATA_DIR   = Path("data")
VLM_MODEL  = DEFAULT_LLM_MODEL
TOP_K      = 2
CANDIDATES = 5

_TMP_CLIPS: List[str] = []   # track tmp files to clean on exit


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _extract_clip(video_path: str, start_sec: float, end_sec: float,
                  padding: float = 1.5) -> Optional[str]:
    """Write a short MP4 clip to a temp file and return its path."""
    clip_start = max(0.0, start_sec - padding)
    clip_end   = end_sec + padding

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    _TMP_CLIPS.append(tmp.name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_MSEC, clip_start * 1000)
    while True:
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms / 1000 > clip_end:
            break
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    return tmp.name


def _generate_summary(query: str, results: List[Dict], model: str, api_key: str) -> str:
    """Ask OpenRouter for an overall answer based on all per-scene reasons."""
    if not api_key:
        return "(OPENROUTER_API_KEY not set — сводка недоступна)"
    lines = [
        f"  {i+1}. [{r['video']} {_fmt_time(r['start_sec'])}–{_fmt_time(r['end_sec'])}] "
        f"{r.get('vlm_reason', '—')}"
        for i, r in enumerate(results)
    ]
    joined_lines = "\n".join(lines)
    prompt = (
        f"Пользователь искал в видеоархиве: '{query}'\n\n"
        f"Найденные сцены и пояснения:\n{joined_lines}\n\n"
        f"На основе этих данных дай краткий связный ответ на запрос пользователя "
        f"(2-3 предложения, на русском языке). "
        f"Если информации недостаточно — так и скажи."
    )
    try:
        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        resp   = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
        )
        return _strip_thinking(resp.choices[0].message.content or "")
    except Exception as exc:
        return f"[Не удалось сгенерировать сводку: {exc}]"


# ── Core search pipeline ──────────────────────────────────────────────────────

def run_search(
    query: str,
    vlm_model: str,
    top_k: int,
    candidates: int,
    data_dir: str,
    use_vlm: bool = True,
    use_llm_rewrite: bool = True,
    api_key: str = "",
) -> Tuple[List[Dict], str, dict]:
    """
    Full pipeline:
      1. LLM decomposes query → visual_query + text_query + weights
      2. Encode each sub-query separately with SigLIP
      3. Search visual.index with visual_query, text.index with text_query
      4. Weighted RRF using LLM-decided weights
      5. VLM re-rank → summary

    Returns (results_list, summary_text, query_plan_dict).
    """
    global VIS_IDX, TXT_IDX, METADATA, VIS_MODEL, VIS_PROC, TXT_MODEL, TXT_TOK, DEVICE

    api_key = api_key or OPENROUTER_API_KEY

    # ── Step 1: LLM query decomposition ──────────────────────────────────────
    if use_llm_rewrite:
        plan = rewrite_query(query, vlm_model, api_key)
    else:
        plan = _fallback_query_plan(query)

    visual_query = plan["visual_query"]
    text_query   = plan["text_query"]
    vis_weight   = plan["visual_weight"]
    txt_weight   = plan["text_weight"]

    # ── Step 2: Encode sub-queries with their respective models ───────────────
    vis_vec = encode_visual_query(visual_query, VIS_MODEL, VIS_PROC, DEVICE)
    txt_vec = encode_text_query(text_query,     TXT_MODEL, TXT_TOK,  DEVICE)

    # ── Step 3: Search both indexes ───────────────────────────────────────────
    _, vis_ids = search_index(VIS_IDX, vis_vec, candidates)
    _, txt_ids = search_index(TXT_IDX, txt_vec, candidates)

    # ── Step 4: Weighted RRF ──────────────────────────────────────────────────
    hits = weighted_reciprocal_rank_fusion(
        [vis_ids.tolist(), txt_ids.tolist()],
        [vis_weight, txt_weight],
    )

    if not use_vlm:
        results = []
        for score, fid in hits[:top_k]:
            scene = dict(METADATA[int(fid)])
            scene["vlm_score"]  = float(score)
            scene["vlm_reason"] = scene.get("transcript") or "—"
            results.append(scene)
        summary = "(VLM отключён)"
        return results, summary, plan

    # ── Step 5: VLM re-rank + summary ─────────────────────────────────────────
    results = rerank(
        hits, METADATA, query,
        api_key=api_key,
        video_dir=data_dir,
        model=vlm_model,
        provider=vlm_provider,
        use_vision=True,
        n_frames=4,
        top_k=top_k,
    )
    summary = _generate_summary(query, results, model=vlm_model, api_key=api_key)
    return results, summary, plan


# ── Gradio callbacks ──────────────────────────────────────────────────────────

def chat_fn(message: str, history: list,
            vlm_model: str, top_k: int, candidates: int,
            data_dir: str, use_vlm: bool, use_llm_rewrite: bool,
            api_key: str = ""):
    """Main chat handler – returns (updated_history, results_state, choices_update)."""
    if not message.strip():
        yield history, [], gr.update(choices=[], value=None), None
        return

    # Acknowledge immediately
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "⏳ Ищу…"},
    ]
    yield history, [], gr.update(choices=[], value=None), None

    try:
        results, summary, plan = run_search(
            message, vlm_model, int(top_k), int(candidates),
            data_dir, use_vlm, use_llm_rewrite, api_key,
        )
    except Exception as exc:
        history[-1]["content"] = f"❌ Ошибка: {exc}"
        yield history, [], gr.update(choices=[], value=None), None
        return

    # Show query decomposition if LLM rewrite was used
    lines = []
    if use_llm_rewrite:
        lines.append(
            f"🔍 **Декомпозиция запроса:**  "
            f"visual=`{plan['visual_query']}` ({plan['visual_weight']:.0%}) | "
            f"text=`{plan['text_query']}` ({plan['text_weight']:.0%})\n"
        )

    # Build markdown response
    lines.append("### Топ результатов\n")
    for i, r in enumerate(results, 1):
        ts    = f"{_fmt_time(r['start_sec'])} – {_fmt_time(r['end_sec'])}"
        score = r.get("vlm_score", r.get("faiss_score", 0))
        reason = r.get("vlm_reason", "—")
        lines.append(
            f"**{i}. {r['video']}** — `{ts}`  "
            f"(VLM: {score:.1f}/10)\n> {reason}\n"
        )

    lines.append(f"\n---\n### 💬 Общий ответ\n{summary}")
    response = "\n".join(lines)

    history[-1]["content"] = response

    # Build radio choices
    choices = [
        f"{i}. {r['video']}  {_fmt_time(r['start_sec'])}–{_fmt_time(r['end_sec'])}"
        for i, r in enumerate(results, 1)
    ]

    yield history, results, gr.update(choices=choices, value=None), None


def show_clip(choice: Optional[str], results: list, data_dir: str):
    """Extract and return video clip for the selected result."""
    if not choice or not results:
        return None

    # Parse index from "1. video.mp4 ..."
    try:
        idx = int(choice.split(".")[0]) - 1
    except (ValueError, IndexError):
        return None

    if idx < 0 or idx >= len(results):
        return None

    scene     = results[idx]
    video_path = str(Path(data_dir) / scene["video"])

    if not Path(video_path).exists():
        return None

    return _extract_clip(video_path, scene["start_sec"], scene["end_sec"])


# ── Build UI ──────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Video RAG Search") as demo:
        gr.Markdown(
            "# 🎬 Video RAG Search\n"
            "Ищи моменты в видеоархиве на естественном языке. "
            "Система использует SigLIP + FAISS + VLM (Ollama/OpenRouter) для поиска и ранжирования."
        )

        results_state = gr.State([])

        with gr.Row():
            # ── Left column: chat ──────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Диалог",
                    height=520,
                )
                with gr.Row():
                    query_box = gr.Textbox(
                        placeholder="Введи запрос, например: «Шрек бежит по болоту»",
                        show_label=False,
                        scale=5,
                    )
                    send_btn = gr.Button("🔍 Найти", variant="primary", scale=1)

                with gr.Accordion("⚙️ Настройки", open=False):
                    with gr.Row():
                        vlm_provider_box = gr.Dropdown(
                            choices=["ollama", "openrouter"],
                            value=VLM_PROVIDER,
                            label="VLM provider",
                        )
                    with gr.Row():
                        vlm_model_box = gr.Textbox(
                            value=VLM_MODEL, label="OpenRouter модель"
                        )
                        data_dir_box = gr.Textbox(
                            value=str(DATA_DIR), label="Папка с видео (data_dir)"
                        )
                    api_key_box = gr.Textbox(
                        value=OPENROUTER_API_KEY,
                        label="OpenRouter API Key",
                        type="password",
                        placeholder="sk-or-v1-...",
                    )
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            1, 10, value=TOP_K, step=1, label="Top-K результатов"
                        )
                        cand_slider = gr.Slider(
                            5, 40, value=CANDIDATES, step=5,
                            label="Кандидатов для VLM"
                        )
                    with gr.Row():
                        use_vlm_chk = gr.Checkbox(
                            value=True, label="Использовать VLM re-ranking (OpenRouter)"
                        )
                        use_rewrite_chk = gr.Checkbox(
                            value=True, label="LLM декомпозиция запроса"
                        )

            # ── Right column: clip viewer ──────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 🎞️ Просмотр клипа")
                scene_radio = gr.Radio(
                    choices=[],
                    label="Выбери сцену для просмотра",
                    interactive=True,
                )
                video_player = gr.Video(
                    label="Фрагмент видео",
                    height=340,
                    interactive=False,
                )
                gr.Markdown(
                    "_Нажми на сцену слева, чтобы посмотреть соответствующий клип._",
                    elem_id="hint",
                )

        # ── Wiring ────────────────────────────────────────────────────────
        search_inputs = [
            query_box, chatbot,
            vlm_model_box, top_k_slider, cand_slider, data_dir_box,
            use_vlm_chk, use_rewrite_chk, api_key_box,
        ]
        search_outputs = [chatbot, results_state, scene_radio, video_player]

        send_btn.click(
            fn=chat_fn,
            inputs=search_inputs,
            outputs=search_outputs,
        )
        query_box.submit(
            fn=chat_fn,
            inputs=search_inputs,
            outputs=search_outputs,
        )

        scene_radio.change(
            fn=show_clip,
            inputs=[scene_radio, results_state, data_dir_box],
            outputs=[video_player],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db_dir",    default="db")
    p.add_argument("--data_dir",  default="data")
    p.add_argument("--device",    default="cpu")
    p.add_argument("--vlm_model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--top_k",     type=int, default=5)
    p.add_argument("--candidates",type=int, default=15)
    p.add_argument("--port",      type=int, default=7860)
    p.add_argument("--share",     action="store_true")
    return p.parse_args()


def main():
    global VIS_IDX, TXT_IDX, METADATA, VIS_MODEL, VIS_PROC, TXT_MODEL, TXT_TOK
    global DEVICE, DATA_DIR, VLM_MODEL, TOP_K, CANDIDATES, OPENROUTER_API_KEY

    args = parse_args()
    DEVICE     = args.device
    DATA_DIR   = Path(args.data_dir)
    VLM_PROVIDER = args.vlm_provider
    VLM_MODEL  = args.vlm_model
    TOP_K      = args.top_k
    CANDIDATES = args.candidates

    # Allow overriding the API key via CLI (useful for quick tests)
    if hasattr(args, "api_key") and args.api_key:
        OPENROUTER_API_KEY = args.api_key

    if not OPENROUTER_API_KEY:
        print(
            "[WARN] OPENROUTER_API_KEY is not set. "
            "LLM query rewrite and VLM re-ranking will be skipped.\n"
            "Set it in .env or as an env var: OPENROUTER_API_KEY=sk-or-..."
        )

    print("Loading FAISS database …")
    VIS_IDX, TXT_IDX, METADATA = load_db(args.db_dir)

    VIS_MODEL, VIS_PROC = load_visual_model(DEVICE)
    TXT_MODEL, TXT_TOK  = load_text_model(DEVICE)
    print("Ready! Starting Gradio …\n")

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share, theme=gr.themes.Soft())

    # Cleanup temp clips
    for f in _TMP_CLIPS:
        try:
            os.unlink(f)
        except OSError:
            pass


if __name__ == "__main__":
    main()
