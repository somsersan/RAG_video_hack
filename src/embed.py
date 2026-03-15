"""
Embedding generation and FAISS index construction for Video-RAG.

Two separate FAISS IndexFlatIP indices are built:
  - visual.index : L2-normalised SigLIP2 image embeddings
                   (mean-pooled from 3 keyframes per scene)
  - text.index   : L2-normalised BGE-M3 text embeddings
                   (CLS-pooled from cleaned scene transcript)

Both indices share the same integer IDs (= row index in metadata.json).
Cosine similarity = inner product after L2 normalisation.

Layout of db/ after save_db():
    db/
    ├── visual.index      ← FAISS binary  (SigLIP2 space)
    ├── text.index        ← FAISS binary  (BGE-M3 space)
    └── metadata.json     ← list of scene dicts (faiss_id, video, timestamps, …)

Usage (standalone):
    python -m src.embed --scenes output/all_scenes.json --db_dir db
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, AutoTokenizer

# ── Model identifiers ─────────────────────────────────────────────────────────
SIGLIP_MODEL      = "google/siglip2-large-patch16-384"   # visual encoder
BGE_M3_MODEL      = "BAAI/bge-m3"                        # text encoder

SIGLIP_IMAGE_SIZE = 384
PLACEHOLDER_TEXT  = "[без речи]"   # fallback for silent scenes


# ── Tensor utilities ──────────────────────────────────────────────────────────

def _l2_norm(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _extract_tensor(out) -> torch.Tensor:
    """Extract a pooled embedding tensor from a HuggingFace model output."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]   # CLS token fallback
    raise TypeError(f"Cannot extract embedding tensor from {type(out)}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_visual_model(
    device: str = "cpu",
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Tuple[AutoModel, AutoProcessor]:
    """Load SigLIP2 vision+text model for visual embedding and query encoding."""
    print(f"Loading visual model: {SIGLIP_MODEL} …")
    processor = AutoProcessor.from_pretrained(
        SIGLIP_MODEL,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        SIGLIP_MODEL,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    ).to(device).eval()
    return model, processor


def load_model(
    device: str = "cpu",
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Tuple[AutoModel, AutoProcessor]:
    """Backward-compatible alias for visual query model loading."""
    return load_visual_model(
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def load_text_model(
    device: str = "cpu",
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Tuple[AutoModel, AutoTokenizer]:
    """Load BGE-M3 encoder for text embedding and query encoding."""
    print(f"Loading text model: {BGE_M3_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(
        BGE_M3_MODEL,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        BGE_M3_MODEL,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    ).to(device).eval()
    return model, tokenizer


# ── Scene text builder ────────────────────────────────────────────────────────

def _clean_transcript(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"www\.\S+",     " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDVD\s*rip\b","", text, flags=re.IGNORECASE)
    text = re.sub(r"\[.*?\]",      " ", text)
    text = re.sub(r"\s+",          " ", text).strip()
    return text


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _scene_people_context(scene: dict) -> str:
    ready = _clean_transcript(str(scene.get("face_context_text") or "")).strip()
    if ready:
        return ready

    actors = _as_str_list(scene.get("actors_in_frame"))
    characters = _as_str_list(scene.get("characters_in_frame"))
    parts: list[str] = []
    if actors:
        parts.append(f"Actors in frame: {', '.join(actors)}")
    if characters:
        parts.append(f"Characters in frame: {', '.join(characters)}")
    return ". ".join(parts).strip()


def build_scene_text(scene: dict, max_chars: int = 1500) -> str:
    """Build a text string for embedding from a scene metadata dict."""
    transcript = _clean_transcript(
        str(scene.get("transcript_text") or scene.get("transcript") or "")
    ).strip()
    people_ctx = _scene_people_context(scene)

    parts: list[str] = []
    if people_ctx:
        parts.append(people_ctx)
    if transcript:
        parts.append(f"Subtitles: {transcript}")

    text = " ".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    if not text:
        return PLACEHOLDER_TEXT
    return text


# ── Visual embedding ──────────────────────────────────────────────────────────

@torch.no_grad()
def embed_images(
    image_paths: List[str],
    model: AutoModel,
    processor: AutoProcessor,
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """
    Encode a flat list of image paths with the SigLIP2 vision tower.

    Returns float32 array of shape (N, D), L2-normalised.
    """
    all_embs: List[np.ndarray] = []

    with tqdm(total=len(image_paths), desc="  SigLIP2 images", unit="img",
              dynamic_ncols=True) as pbar:
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images: List[Image.Image] = []
            for p in batch_paths:
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    images.append(
                        Image.new("RGB", (SIGLIP_IMAGE_SIZE, SIGLIP_IMAGE_SIZE), color=0)
                    )

            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            img_inputs = {k: v for k, v in inputs.items() if "pixel" in k}
            out   = model.vision_model(**img_inputs)
            feats = _l2_norm(_extract_tensor(out))
            all_embs.append(feats.cpu().float().numpy())
            pbar.update(len(batch_paths))

    return np.vstack(all_embs).astype(np.float32)


@torch.no_grad()
def embed_scene_keyframes(
    keyframe_paths: List[str],
    model: AutoModel,
    processor: AutoProcessor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Encode up to 3 keyframes for a single scene and return the mean-pooled,
    L2-normalised embedding of shape (D,).
    """
    embs = embed_images(
        image_paths=keyframe_paths,
        model=model,
        processor=processor,
        device=device,
        batch_size=len(keyframe_paths),
    )
    pooled = embs.mean(axis=0)
    pooled = pooled / (np.linalg.norm(pooled) + 1e-12)
    return pooled.astype(np.float32)


def build_all_visual_embeddings(
    scenes: List[Dict],
    model: AutoModel,
    processor: AutoProcessor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Build one visual embedding per scene (mean-pooled over keyframes).
    Returns float32 array of shape (N, D).
    """
    all_embs: List[np.ndarray] = []
    for scene in tqdm(scenes, desc="Visual scene embeddings"):
        paths = [str(p) for p in scene.get("keyframe_paths", [])]
        if not paths:
            # Fallback: single keyframe_path field
            kp = scene.get("keyframe_path", "")
            paths = [kp] if kp else []
        emb = embed_scene_keyframes(
            keyframe_paths=paths,
            model=model,
            processor=processor,
            device=device,
        )
        all_embs.append(emb)
    return np.vstack(all_embs).astype(np.float32)


# ── Text embedding (BGE-M3) ───────────────────────────────────────────────────

@torch.no_grad()
def embed_texts_bge(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode texts with BGE-M3 using CLS-token pooling.

    Returns float32 array of shape (N, D), L2-normalised.
    """
    all_embs: List[np.ndarray] = []

    with tqdm(total=len(texts), desc="  BGE-M3 texts", unit="text",
              dynamic_ncols=True) as pbar:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch = [t.strip() if t.strip() else PLACEHOLDER_TEXT for t in batch]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            out   = model(**inputs)
            feats = out.last_hidden_state[:, 0]   # CLS pooling
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            all_embs.append(feats.cpu().float().numpy())
            pbar.update(len(batch))

    return np.vstack(all_embs).astype(np.float32)


def build_all_text_embeddings(
    scenes: List[Dict],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Build one BGE-M3 text embedding per scene from the scene transcript.
    Returns float32 array of shape (N, D).
    """
    texts = [build_scene_text(scene) for scene in scenes]
    embs  = embed_texts_bge(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    # ensure L2-normalised (embed_texts_bge already does this, but be safe)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.clip(norms, 1e-12, None)
    return embs.astype(np.float32)


# ── Query encoding ────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_visual_query(
    query: str,
    model: AutoModel,
    processor: AutoProcessor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Encode a text query with the SigLIP2 text tower for searching visual.index.
    Returns float32 array of shape (1, D), L2-normalised.
    """
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


@torch.no_grad()
def encode_text_query(
    query: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str = "cpu",
) -> np.ndarray:
    """
    Encode a text query with BGE-M3 for searching text.index.
    Returns float32 array of shape (1, D), L2-normalised.
    """
    inputs = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    out   = model(**inputs)
    feats = out.last_hidden_state[:, 0]
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return feats.cpu().float().numpy()


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat inner-product FAISS index (IP == cosine for L2-normed vecs)."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# ── Persistence ───────────────────────────────────────────────────────────────

def save_db(
    scenes: List[Dict],
    visual_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    output_dir: str,
) -> None:
    """Persist both FAISS indices and scene metadata to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert len(scenes) == len(visual_embeddings) == len(text_embeddings), (
        f"Length mismatch: scenes={len(scenes)}, "
        f"visual={len(visual_embeddings)}, text={len(text_embeddings)}"
    )

    faiss.write_index(build_index(visual_embeddings), str(output_dir / "visual.index"))
    faiss.write_index(build_index(text_embeddings),   str(output_dir / "text.index"))

    metadata = [{"faiss_id": i, **scene} for i, scene in enumerate(scenes)]
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(
        f"\nVector DB saved to: {output_dir}\n"
        f"  visual.index  {visual_embeddings.shape}\n"
        f"  text.index    {text_embeddings.shape}\n"
        f"  metadata.json {len(metadata)} records"
    )


def load_db(db_dir: str) -> Tuple[faiss.Index, faiss.Index, List[Dict]]:
    """Load previously saved DB. Returns (visual_index, text_index, metadata)."""
    db_dir = Path(db_dir)
    vis_idx = faiss.read_index(str(db_dir / "visual.index"))
    txt_idx = faiss.read_index(str(db_dir / "text.index"))
    with open(db_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return vis_idx, txt_idx, metadata


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS vector DB from scene JSON")
    parser.add_argument("--scenes",    required=True, help="Path to all_scenes.json")
    parser.add_argument("--db_dir",    default="db",  help="Output DB directory")
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--img_batch", type=int, default=8)
    parser.add_argument("--txt_batch", type=int, default=32)
    args = parser.parse_args()

    with open(args.scenes, encoding="utf-8") as f:
        scenes = json.load(f)

    vis_model, vis_proc   = load_visual_model(args.device)
    txt_model, txt_tok    = load_text_model(args.device)

    print(f"\nEmbedding {len(scenes)} scenes …")
    vis_embs = build_all_visual_embeddings(scenes, vis_model, vis_proc, args.device)
    txt_embs = build_all_text_embeddings(scenes, txt_model, txt_tok,
                                         args.device, args.txt_batch)

    save_db(scenes, vis_embs, txt_embs, args.db_dir)
