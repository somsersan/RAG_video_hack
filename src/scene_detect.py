"""
Scene detection using PySceneDetect.

Detects scene boundaries and extracts keyframes per scene.

Available keyframe methods:
  - fixed3: 3 fixed points per scene (15%, 50%, 85%)
  - uniform_clip_kmeans:
      1) uniform frame sampling (default 1 FPS)
      2) frame embeddings with CLIP-like model
      3) k-means++ clustering, pick representative frames

Usage (standalone):
    python -m src.scene_detect --video data/shrek.mp4 --output_dir output
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoProcessor


SUPPORTED_KEYFRAME_METHODS = ("fixed3", "uniform_clip_kmeans")
DEFAULT_CLUSTER_MODEL = "apple/MobileCLIP-S2"
MAX_UNIFORM_CANDIDATES = 256


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _extract_feature_tensor(out: object) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    raise TypeError(f"Cannot extract tensor from model output: {type(out)}")


def _save_frame_at_sec(
    cap: cv2.VideoCapture,
    sec: float,
    out_path: Path,
    fallback_sec: float | None = None,
) -> bool:
    """Try to save a frame at sec. On failure, optionally try fallback_sec."""
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
    ok, frame = cap.read()
    if ok:
        return bool(cv2.imwrite(str(out_path), frame))

    if fallback_sec is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, fallback_sec * 1000.0)
        ok, frame = cap.read()
        if ok:
            return bool(cv2.imwrite(str(out_path), frame))
    return False


def _read_frame_at_sec(cap: cv2.VideoCapture, sec: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _uniform_candidate_count(start_sec: float, end_sec: float) -> int:
    """
    1 frame per 1 second of scene duration with an upper cap.
    Examples:
      - 1 sec   -> 1 frame
      - 256 sec -> 256 frames
      - >256    -> 256 frames
    """
    duration = max(end_sec - start_sec, 0.0)
    if duration <= 1e-6:
        return 1
    return min(MAX_UNIFORM_CANDIDATES, max(1, int(math.ceil(duration))))


def _uniform_sample_seconds(start_sec: float, end_sec: float, sample_fps: float) -> list[float]:
    """
    Uniformly sample timestamps inside [start_sec, end_sec).
    sample_fps is kept for API compatibility and is intentionally ignored.
    """
    _ = sample_fps
    duration = max(end_sec - start_sec, 0.0)
    if duration <= 1e-6:
        return [start_sec]

    n_samples = _uniform_candidate_count(start_sec, end_sec)
    if n_samples == 1:
        return [start_sec + duration * 0.5]

    step = duration / n_samples
    return [float(start_sec + step * (i + 0.5)) for i in range(n_samples)]


def _collect_uniform_candidates(
    cap: cv2.VideoCapture,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
) -> list[tuple[float, np.ndarray]]:
    sample_secs = _uniform_sample_seconds(start_sec, end_sec, sample_fps)
    candidates: list[tuple[float, np.ndarray]] = []

    for sec in sample_secs:
        frame = _read_frame_at_sec(cap, sec)
        if frame is not None:
            candidates.append((sec, frame))

    if not candidates:
        midpoint = (start_sec + end_sec) * 0.5
        frame = _read_frame_at_sec(cap, midpoint)
        if frame is not None:
            candidates.append((midpoint, frame))

    return candidates


def _cluster_representative_indices_with_labels(
    vectors: np.ndarray,
    n_select: int,
    seed: int = 42,
) -> tuple[list[int], np.ndarray]:
    """
    k-means++ over frame embeddings + nearest frame to each centroid.
    Returns indices of representative frames.
    """
    n_frames, dim = vectors.shape
    if n_frames <= n_select:
        labels = np.arange(n_frames, dtype=np.int32)
        return list(range(n_frames)), labels

    k = max(1, min(n_select, n_frames))
    data = np.ascontiguousarray(vectors.astype(np.float32))

    labels, centroids = _kmeans_pp_numpy(data, k=k, niter=30, seed=seed)

    selected: list[int] = []
    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue
        centroid = centroids[cluster_id]
        dists = np.sum((data[members] - centroid) ** 2, axis=1)
        selected.append(int(members[int(np.argmin(dists))]))

    if len(selected) < k:
        selected_set = set(selected)
        selected.extend(i for i in range(n_frames) if i not in selected_set)
        selected = selected[:k]

    return sorted(set(selected)), labels.astype(np.int32)


def _kmeans_pp_numpy(
    data: np.ndarray,
    k: int,
    niter: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numpy-only k-means++ for small per-scene candidate sets.
    Avoids native FAISS clustering crashes on some macOS setups.
    """
    n, d = data.shape
    rng = np.random.default_rng(seed)

    # k-means++ init
    centers = np.empty((k, d), dtype=np.float32)
    first = int(rng.integers(0, n))
    centers[0] = data[first]
    closest_d2 = np.sum((data - centers[0]) ** 2, axis=1)

    for i in range(1, k):
        total = float(closest_d2.sum())
        if total <= 1e-12:
            idx = int(rng.integers(0, n))
        else:
            probs = closest_d2 / total
            idx = int(rng.choice(n, p=probs))
        centers[i] = data[idx]
        d2_new = np.sum((data - centers[i]) ** 2, axis=1)
        closest_d2 = np.minimum(closest_d2, d2_new)

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max(1, niter)):
        # assign
        dists = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        # update
        for cid in range(k):
            members = np.where(labels == cid)[0]
            if len(members) == 0:
                # re-seed empty centroid to the farthest sample
                farthest = int(np.argmax(np.min(dists, axis=1)))
                centers[cid] = data[farthest]
            else:
                centers[cid] = data[members].mean(axis=0)

        # keep centroids on the unit sphere to mirror cosine-space behavior
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.clip(norms, 1e-12, None)

    return labels, centers


def _cluster_representative_indices(
    vectors: np.ndarray,
    n_select: int,
    seed: int = 42,
) -> list[int]:
    selected, _ = _cluster_representative_indices_with_labels(vectors, n_select, seed)
    return selected


class ClipFrameEmbedder:
    """Lightweight image embedder for keyframe selection."""

    def __init__(
        self,
        model_name: str = DEFAULT_CLUSTER_MODEL,
        device: str = "cpu",
        batch_size: int = 16,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ):
        self.device = device
        self.batch_size = batch_size
        print(f"  Loading frame-selection model: {model_name} ({device}) …")
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except Exception:
            # Some lightweight CLIP checkpoints expose AutoProcessor only.
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(device).eval()

    @torch.no_grad()
    def encode_frames(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        all_embs: list[np.ndarray] = []

        for start in range(0, len(frames_bgr), self.batch_size):
            batch = frames_bgr[start:start + self.batch_size]
            images = [
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for frame in batch
            ]

            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None:
                raise RuntimeError("Image processor did not return pixel_values")

            if hasattr(self.model, "get_image_features"):
                feats = self.model.get_image_features(pixel_values=pixel_values)
            elif hasattr(self.model, "vision_model"):
                out = self.model.vision_model(pixel_values=pixel_values)
                feats = _extract_feature_tensor(out)
            else:
                out = self.model(pixel_values=pixel_values)
                # CLIPModel(...) returns BaseModelOutputWithPooling in vision_model_output.
                if hasattr(out, "vision_model_output") and out.vision_model_output is not None:
                    feats = _extract_feature_tensor(out.vision_model_output)
                else:
                    feats = _extract_feature_tensor(out)

            # Compatibility guard for transformers variants that may return
            # model-output objects instead of raw tensors.
            if not isinstance(feats, torch.Tensor):
                feats = _extract_feature_tensor(feats)

            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_embs.append(feats.cpu().float().numpy())

        return _l2_normalize(np.vstack(all_embs).astype(np.float32))


def _extract_fixed3_keyframes(
    cap: cv2.VideoCapture,
    start_sec: float,
    end_sec: float,
    scene_idx: int,
    keyframes_dir: Path,
) -> list[str]:
    duration = max(end_sec - start_sec, 0.001)
    sample_secs = [
        start_sec + duration * 0.15,
        start_sec + duration * 0.50,
        start_sec + duration * 0.85,
    ]

    keyframe_paths: list[str] = []
    for j, sec in enumerate(sample_secs):
        frame_path = keyframes_dir / f"scene_{scene_idx:04d}_{j:02d}.jpg"
        ok = _save_frame_at_sec(
            cap=cap,
            sec=sec,
            out_path=frame_path,
            fallback_sec=start_sec + duration * 0.50,
        )
        keyframe_paths.append(str(frame_path) if ok else "")
    return keyframe_paths


def _extract_uniform_cluster_keyframes(
    cap: cv2.VideoCapture,
    start_sec: float,
    end_sec: float,
    scene_idx: int,
    keyframes_dir: Path,
    embedder: ClipFrameEmbedder,
    sample_fps: float = 1.0,
    keyframes_per_scene: int = 3,
    seed: int = 42,
    log_prefix: str = "",
    verbose: bool = False,
) -> list[str]:
    t0 = time.perf_counter()
    candidates = _collect_uniform_candidates(
        cap=cap,
        start_sec=start_sec,
        end_sec=end_sec,
        sample_fps=sample_fps,
    )
    if verbose:
        print(
            f"{log_prefix} sampled candidates: {len(candidates)} "
            f"(fps={sample_fps}, range={start_sec:.2f}-{end_sec:.2f}s)"
        )

    if not candidates:
        if verbose:
            print(f"{log_prefix} no readable frames, returning empty keyframe slots")
        return ["" for _ in range(max(keyframes_per_scene, 1))]

    _, frames = zip(*candidates)
    t_emb = time.perf_counter()
    embeddings = embedder.encode_frames(list(frames))
    n_select = max(1, min(keyframes_per_scene, len(candidates)))
    if verbose:
        print(
            f"{log_prefix} embedding done: shape={embeddings.shape}, "
            f"target_clusters={n_select}"
        )

    t_cluster = time.perf_counter()
    selected_idx = _cluster_representative_indices(embeddings, n_select, seed=seed)
    selected_idx = sorted(selected_idx, key=lambda idx: candidates[idx][0])
    if verbose:
        print(
            f"{log_prefix} kmeans++ done: selected={selected_idx} "
            f"(embed={t_cluster - t_emb:.2f}s, cluster={time.perf_counter() - t_cluster:.2f}s)"
        )

    keyframe_paths: list[str] = []
    for j, idx in enumerate(selected_idx):
        _, frame = candidates[idx]
        frame_path = keyframes_dir / f"scene_{scene_idx:04d}_{j:02d}.jpg"
        ok = cv2.imwrite(str(frame_path), frame)
        keyframe_paths.append(str(frame_path) if ok else "")
    if verbose:
        print(f"{log_prefix} scene done in {time.perf_counter() - t0:.2f}s")
    return keyframe_paths


def inspect_scene_clusters(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_dir: str = "output/cluster_debug",
    sample_fps: float = 1.0,
    keyframes_per_scene: int = 3,
    cluster_model: str = DEFAULT_CLUSTER_MODEL,
    cluster_device: str = "cpu",
    cluster_batch: int = 16,
    cluster_seed: int = 42,
    hf_cache_dir: str | None = None,
    offline_models: bool = False,
) -> dict:
    """
    Build clustering debug artefacts for one scene segment.
    Returns JSON-serializable data for interactive visualization.
    """
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video}")

    candidates = _collect_uniform_candidates(
        cap=cap,
        start_sec=start_sec,
        end_sec=end_sec,
        sample_fps=sample_fps,
    )
    cap.release()

    if not candidates:
        raise RuntimeError("No frames could be extracted for the provided scene range.")

    embedder = ClipFrameEmbedder(
        model_name=cluster_model,
        device=cluster_device,
        batch_size=cluster_batch,
        cache_dir=hf_cache_dir,
        local_files_only=offline_models,
    )
    _, frames = zip(*candidates)
    embeddings = embedder.encode_frames(list(frames))
    n_select = max(1, min(keyframes_per_scene, len(candidates)))
    selected_idx, labels = _cluster_representative_indices_with_labels(
        embeddings, n_select, seed=cluster_seed,
    )
    selected_set = set(selected_idx)

    scene_tag = f"{start_sec:.3f}_{end_sec:.3f}".replace(".", "p")
    scene_dir = Path(output_dir) / video.stem / scene_tag
    scene_dir.mkdir(parents=True, exist_ok=True)

    frame_rows: list[dict] = []
    for idx, ((sec, frame), cluster_id) in enumerate(zip(candidates, labels)):
        path = scene_dir / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(path), frame)
        frame_rows.append(
            {
                "index": idx,
                "timestamp_sec": round(float(sec), 3),
                "cluster_id": int(cluster_id),
                "selected": idx in selected_set,
                "image_path": str(path),
            }
        )

    return {
        "video_path": str(video),
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "sample_fps": float(sample_fps),
        "requested_keyframes": int(keyframes_per_scene),
        "n_candidates": len(candidates),
        "n_selected": len(selected_idx),
        "selected_indices": [int(i) for i in selected_idx],
        "cluster_ids": sorted({int(x) for x in labels.tolist()}),
        "frames": frame_rows,
        "output_dir": str(scene_dir),
    }


def detect_scenes(
    video_path: str,
    output_dir: str,
    threshold: float = 27.0,
    min_scene_len: int = 15,  # frames
    keyframe_method: str = "fixed3",
    sample_fps: float = 1.0,
    keyframes_per_scene: int = 3,
    cluster_model: str = DEFAULT_CLUSTER_MODEL,
    cluster_device: str = "cpu",
    cluster_batch: int = 16,
    cluster_seed: int = 42,
    hf_cache_dir: str | None = None,
    offline_models: bool = False,
    cluster_verbose: bool = False,
    frame_embedder: ClipFrameEmbedder | None = None,
) -> List[Dict]:
    """
    Detect scenes in a video file and extract keyframes.

    Args:
        video_path:         Path to input .mp4.
        output_dir:         Root output directory.
        threshold:          ContentDetector sensitivity.
        min_scene_len:      Minimum scene length in frames.
        keyframe_method:    "fixed3" or "uniform_clip_kmeans".
        sample_fps:         Uniform sampling FPS for uniform_clip_kmeans.
        keyframes_per_scene:
                            Representative frame count for uniform_clip_kmeans.
        cluster_model:      HF model id for frame embeddings.
        cluster_device:     Device for frame embedding model.
        cluster_batch:      Batch size for frame embedding model.
        cluster_seed:       Seed for k-means++ init.
        hf_cache_dir:       HuggingFace cache directory.
        offline_models:     Load HF models only from local cache.
        cluster_verbose:    Detailed per-scene logs for uniform_clip_kmeans.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    keyframes_dir = output_dir / "keyframes" / video_path.stem
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    if keyframe_method not in SUPPORTED_KEYFRAME_METHODS:
        raise ValueError(
            f"Unknown keyframe_method={keyframe_method!r}. "
            f"Supported: {SUPPORTED_KEYFRAME_METHODS}"
        )

    # Scene detection
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
        # Fallback: treat whole video as one scene
        cap_tmp = cv2.VideoCapture(str(video_path))
        fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 25
        total_frames = cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)
        cap_tmp.release()

        from scenedetect import FrameTimecode

        scene_list = [
            (FrameTimecode(0, fps), FrameTimecode(max(int(total_frames) - 1, 0), fps))
        ]

    cap = cv2.VideoCapture(str(video_path))
    scenes: List[Dict] = []

    embedder: ClipFrameEmbedder | None = frame_embedder
    if keyframe_method == "uniform_clip_kmeans":
        print(
            f"  [{video_path.name}] Keyframe mode: uniform_clip_kmeans "
            f"(sampling=1 frame/sec, max={MAX_UNIFORM_CANDIDATES}, "
            f"target={max(1, keyframes_per_scene)}, seed={cluster_seed})"
        )
        if embedder is None:
            embedder = ClipFrameEmbedder(
                model_name=cluster_model,
                device=cluster_device,
                batch_size=cluster_batch,
                cache_dir=hf_cache_dir,
                local_files_only=offline_models,
            )

    total_scenes = len(scene_list)
    for i, (start_tc, end_tc) in tqdm(enumerate(scene_list), total=total_scenes):
        start_sec = start_tc.get_seconds()
        end_sec = end_tc.get_seconds()

        if keyframe_method == "fixed3":
            keyframe_paths = _extract_fixed3_keyframes(
                cap=cap,
                start_sec=start_sec,
                end_sec=end_sec,
                scene_idx=i,
                keyframes_dir=keyframes_dir,
            )
        else:
            if embedder is None:
                raise RuntimeError("Embedder is not initialized")
            should_log_scene = cluster_verbose and (
                i < 3 or (i + 1) % 100 == 0 or i == total_scenes - 1
            )
            if should_log_scene:
                print(
                    f"  [{video_path.name}] scene {i + 1}/{total_scenes} "
                    f"({start_sec:.2f}-{end_sec:.2f}s): clustering keyframes ..."
                )
            keyframe_paths = _extract_uniform_cluster_keyframes(
                cap=cap,
                start_sec=start_sec,
                end_sec=end_sec,
                scene_idx=i,
                keyframes_dir=keyframes_dir,
                embedder=embedder,
                sample_fps=sample_fps,
                keyframes_per_scene=max(1, keyframes_per_scene),
                seed=cluster_seed,
                log_prefix=f"    [{video_path.name} scene {i + 1}/{total_scenes}]",
                verbose=should_log_scene,
            )

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


def detect_scenes_batch(
    video_paths: Sequence[str | Path],
    output_dir: str,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    keyframe_method: str = "fixed3",
    sample_fps: float = 1.0,
    keyframes_per_scene: int = 3,
    cluster_model: str = DEFAULT_CLUSTER_MODEL,
    cluster_device: str = "cpu",
    cluster_batch: int = 16,
    cluster_seed: int = 42,
    hf_cache_dir: str | None = None,
    offline_models: bool = False,
    cluster_verbose: bool = False,
    video_batch_size: int = 8,
) -> List[Dict]:
    """
    Process many videos while reusing the frame embedder per video-batch.
    This avoids reloading the frame model for each single video.
    """
    all_scenes: List[Dict] = []
    videos = [Path(v) for v in video_paths]
    if not videos:
        return all_scenes

    batch_size = max(1, int(video_batch_size))

    if keyframe_method != "uniform_clip_kmeans":
        for vp in videos:
            all_scenes.extend(
                detect_scenes(
                    video_path=str(vp),
                    output_dir=output_dir,
                    threshold=threshold,
                    min_scene_len=min_scene_len,
                    keyframe_method=keyframe_method,
                    sample_fps=sample_fps,
                    keyframes_per_scene=keyframes_per_scene,
                    cluster_model=cluster_model,
                    cluster_device=cluster_device,
                    cluster_batch=cluster_batch,
                    cluster_seed=cluster_seed,
                    hf_cache_dir=hf_cache_dir,
                    offline_models=offline_models,
                    cluster_verbose=cluster_verbose,
                )
            )
        return all_scenes

    total = len(videos)
    for start in range(0, total, batch_size):
        batch = videos[start:start + batch_size]
        print(
            f"  [batch] Scene detection batch {start // batch_size + 1}: "
            f"{len(batch)} video(s), shared frame model"
        )
        embedder = ClipFrameEmbedder(
            model_name=cluster_model,
            device=cluster_device,
            batch_size=cluster_batch,
            cache_dir=hf_cache_dir,
            local_files_only=offline_models,
        )
        for vp in batch:
            all_scenes.extend(
                detect_scenes(
                    video_path=str(vp),
                    output_dir=output_dir,
                    threshold=threshold,
                    min_scene_len=min_scene_len,
                    keyframe_method=keyframe_method,
                    sample_fps=sample_fps,
                    keyframes_per_scene=keyframes_per_scene,
                    cluster_model=cluster_model,
                    cluster_device=cluster_device,
                    cluster_batch=cluster_batch,
                    cluster_seed=cluster_seed,
                    hf_cache_dir=hf_cache_dir,
                    offline_models=offline_models,
                    cluster_verbose=cluster_verbose,
                    frame_embedder=embedder,
                )
            )
    return all_scenes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect scenes and extract keyframes")
    parser.add_argument("--video", required=True, help="Path to .mp4")
    parser.add_argument("--output_dir", default="output", help="Output root dir")
    parser.add_argument("--threshold", type=float, default=27.0)
    parser.add_argument("--min_scene_len", type=int, default=15)
    parser.add_argument(
        "--keyframe_method",
        default="fixed3",
        choices=list(SUPPORTED_KEYFRAME_METHODS),
        help="Keyframe extraction strategy",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=1.0,
        help="Candidate sampling FPS for uniform_clip_kmeans",
    )
    parser.add_argument(
        "--keyframes_per_scene",
        type=int,
        default=3,
        help="Representative keyframes per scene for uniform_clip_kmeans",
    )
    parser.add_argument(
        "--cluster_model",
        default=DEFAULT_CLUSTER_MODEL,
        help="HF model id used for frame embeddings in uniform_clip_kmeans",
    )
    parser.add_argument("--cluster_device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--cluster_batch", type=int, default=16)
    parser.add_argument("--cluster_seed", type=int, default=42)
    parser.add_argument(
        "--cluster_verbose",
        action="store_true",
        help="Print detailed logs for kmeans++ keyframe selection",
    )
    parser.add_argument("--hf_cache_dir", default=".model_cache/hf")
    parser.add_argument(
        "--offline_models",
        action="store_true",
        help="Load models from local cache only (no internet downloads)",
    )
    args = parser.parse_args()

    scenes = detect_scenes(
        args.video,
        args.output_dir,
        args.threshold,
        args.min_scene_len,
        keyframe_method=args.keyframe_method,
        sample_fps=args.sample_fps,
        keyframes_per_scene=args.keyframes_per_scene,
        cluster_model=args.cluster_model,
        cluster_device=args.cluster_device,
        cluster_batch=args.cluster_batch,
        cluster_seed=args.cluster_seed,
        cluster_verbose=args.cluster_verbose,
        hf_cache_dir=args.hf_cache_dir,
        offline_models=args.offline_models,
    )

    out_json = Path(args.output_dir) / f"{Path(args.video).stem}_scenes.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)

    print(f"Saved -> {out_json}")
