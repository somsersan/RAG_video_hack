"""
ANN search over the Video-RAG FAISS database.

Supports two retrieval modes (can be combined via Reciprocal Rank Fusion):
  - visual   : text query → SigLIP text encoder → search in image-embedding index
  - speech   : text query → SigLIP text encoder → search in transcript-embedding index
  - fused    : RRF over both (default)

Optional VLM re-ranking via Ollama/OpenRouter:
  python search.py --query "Шрек бежит" --vlm
  python search.py --query "погоня" --vlm --vlm_model qwen3.5 --vlm_candidates 15

Usage:
    python search.py --query "Шрек бежит по болоту"
    python search.py --query "герой разговаривает с ослом" --mode visual --top_k 5
    python search.py --query "погоня" --mode speech
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
import torch
from transformers import AutoProcessor, AutoModel

from src.embed import SIGLIP_MODEL, _l2_norm, _extract_tensor, load_db, load_model
from src.metadata_schema import transcript_text


# ── Query encoding ────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_query(
    query: str,
    model: AutoModel,
    processor: AutoProcessor,
    device: str = "cpu",
) -> np.ndarray:
    """Encode a text query with SigLIP text tower. Returns (1, D) float32."""
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    ).to(device)
    txt_inputs = {k: v for k, v in inputs.items() if "pixel" not in k}
    feats = _extract_tensor(model.text_model(**txt_inputs))
    feats = _l2_norm(feats)
    return feats.cpu().float().numpy()


# ── Retrieval ─────────────────────────────────────────────────────────────────

def search_index(
    index: faiss.Index,
    query_vec: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, faiss_ids) arrays of length top_k."""
    scores, ids = index.search(query_vec, top_k)
    return scores[0], ids[0]


def reciprocal_rank_fusion(
    rankings: List[List[int]],
    k: int = 60,
) -> List[Tuple[float, int]]:
    """
    Fuse multiple ranked lists via RRF.

    Args:
        rankings: List of lists of faiss IDs (each ordered best→worst).
        k:        RRF constant (default 60 from Cormack 2009).
    Returns:
        List of (rrf_score, faiss_id) sorted descending.
    """
    scores: Dict[int, float] = {}
    for ranked_list in rankings:
        for rank, fid in enumerate(ranked_list):
            if fid < 0:   # FAISS returns -1 for missing
                continue
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (k + rank + 1)
    fused_hits = [(score, fid) for fid, score in scores.items()]
    return sorted(fused_hits, key=lambda x: -x[0])


def weighted_reciprocal_rank_fusion(
    rankings: List[List[int]],
    weights: List[float],
    k: int = 60,
) -> List[Tuple[float, int]]:
    """
    Weighted RRF over multiple ranked lists.

    Args:
        rankings: List of lists of faiss IDs (each ordered best→worst).
        weights:  Per-list weights (same length as rankings). Auto-normalised.
        k:        RRF constant (default 60).
    Returns:
        List of (weighted_rrf_score, faiss_id) sorted descending.
    """
    total = sum(weights)
    if total <= 0:
        norm_w = [1.0 / len(weights)] * len(weights)
    else:
        norm_w = [w / total for w in weights]

    scores: Dict[int, float] = {}
    for ranked_list, w in zip(rankings, norm_w):
        for rank, fid in enumerate(ranked_list):
            if fid < 0:
                continue
            scores[fid] = scores.get(fid, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


# ── Pretty print ──────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def print_results(
    hits: List[Tuple[float, int]],
    metadata: List[Dict],
    top_k: int,
) -> None:
    print(f"\n{'─'*70}")
    print(f"{'#':<4} {'Score':>7}  {'Video':<34} {'Time':<13} Transcript")
    print(f"{'─'*70}")
    for rank, (score, fid) in enumerate(hits[:top_k], 1):
        m = metadata[int(fid)]
        time_str = f"{_fmt_time(m['start_sec'])}–{_fmt_time(m['end_sec'])}"
        transcript = (transcript_text(m) or "—")[:55]
        print(f"{rank:<4} {score:>7.4f}  {m['video']:<34} {time_str:<13} {transcript}")
    print(f"{'─'*70}\n")


def print_vlm_results(results: List[Dict]) -> None:
    print(f"\n{'─'*70}")
    print(f"{'#':<4} {'VLM':>5}  {'Video':<34} {'Time':<13} Reason")
    print(f"{'─'*70}")
    for rank, r in enumerate(results, 1):
        time_str = f"{_fmt_time(r['start_sec'])}–{_fmt_time(r['end_sec'])}"
        reason = (r.get("vlm_reason") or "—")[:55]
        print(f"{rank:<4} {r['vlm_score']:>5.1f}  {r['video']:<34} {time_str:<13} {reason}")
    print(f"{'─'*70}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Video-RAG FAISS DB")
    parser.add_argument("--query",           required=True)
    parser.add_argument("--db_dir",          default="db")
    parser.add_argument("--data_dir",        default="data",
                        help="Directory with video files (used by VLM for frame extraction)")
    parser.add_argument("--mode",            default="fused",
                        choices=["visual", "speech", "fused"])
    parser.add_argument("--top_k",           type=int, default=5)
    parser.add_argument("--device",          default="cpu")
    parser.add_argument("--hf_cache_dir",    default=".model_cache/hf",
                        help="Local cache dir for HuggingFace models")
    parser.add_argument("--offline_models",  action="store_true",
                        help="Load models from local cache only (no internet downloads)")
    # VLM re-ranking
    parser.add_argument("--vlm",             action="store_true",
                        help="Re-rank FAISS results with a VLM provider")
    parser.add_argument("--vlm_provider",    default="ollama",
                        choices=["ollama", "openrouter"],
                        help="VLM provider: ollama|openrouter")
    parser.add_argument("--vlm_model",       default="qwen3.5",
                        help="Model name for selected provider")
    parser.add_argument("--vlm_candidates",  type=int, default=15,
                        help="How many FAISS candidates to send to VLM")
    parser.add_argument("--vlm_frames",      type=int, default=4,
                        help="Frames per scene for VLM")
    parser.add_argument("--vlm_text_only",   action="store_true",
                        help="Use transcript only (no frames), for text-only models")
    args = parser.parse_args()

    # Load DB
    vis_idx, txt_idx, metadata = load_db(args.db_dir)

    # Load model
    print(f"Loading SigLIP …")
    model, proc = load_model(
        args.device,
        cache_dir=args.hf_cache_dir,
        local_files_only=args.offline_models,
    )

    # Encode query
    q_vec = encode_query(args.query, model, proc, args.device)
    print(f"Query: \"{args.query}\"  (mode={args.mode})")

    candidate_k = args.vlm_candidates if args.vlm else args.top_k * 3

    if args.mode == "visual":
        scores, ids = search_index(vis_idx, q_vec, candidate_k)
        hits = list(zip(scores.tolist(), ids.tolist()))

    elif args.mode == "speech":
        scores, ids = search_index(txt_idx, q_vec, candidate_k)
        hits = list(zip(scores.tolist(), ids.tolist()))

    else:   # fused
        _, vis_ids = search_index(vis_idx, q_vec, candidate_k)
        _, txt_ids = search_index(txt_idx, q_vec, candidate_k)
        hits = reciprocal_rank_fusion([vis_ids.tolist(), txt_ids.tolist()])

    print(hits)
    if args.vlm:
        from src.vlm_rerank import rerank
        print(
            f"Re-ranking top-{len(hits)} with "
            f"{args.vlm_provider}/{args.vlm_model} …"
        )
        vlm_results = rerank(
            hits, metadata, args.query,
            video_dir=args.data_dir,
            model=args.vlm_model,
            provider=args.vlm_provider,
            use_vision=not args.vlm_text_only,
            n_frames=args.vlm_frames,
            top_k=args.top_k,
        )
        print_vlm_results(vlm_results)
    else:
        print_results(hits, metadata, args.top_k)


if __name__ == "__main__":
    main()
