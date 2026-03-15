# AGENTS.md

## Project
Video-RAG for semantic scene search in one or more video files.

Current implementation:
- scene detection: `src/scene_detect.py` (PySceneDetect)
- ASR per scene: `src/asr.py` (faster-whisper)
- embeddings + indices: `src/embed.py` (SigLIP + FAISS)
- retrieval and fusion: `search.py` (visual/speech/fused via RRF)
- VLM re-ranking: `src/vlm_rerank.py` (Ollama/OpenRouter)
- metadata schema/enrichment: `src/metadata_schema.py`
- optional PostgreSQL sync: `src/metadata_store.py`
- UI and serving: `app.py` (Gradio)
- evaluation: `eval/` (`dataset.py`, `evaluate.py`)

## Goal
Return relevant video scenes for a natural-language query with:
- timestamped evidence (`start_sec`, `end_sec`)
- transcript/context for the scene
- optional VLM explanation and score

System focus is experimentation and reproducible local runs, not production scaling.

---

## Scope

### In scope
- Multi-video indexing from `data/` (`.mp4`, `.mkv`)
- Scene-based segmentation (content-aware, not fixed windows)
- Keyframe extraction per scene
- Speech transcription per scene
- Dual retrieval (visual + speech) and fused ranking
- Optional VLM re-ranking and concise user-facing answer
- Scene metadata enrichment: season/episode, characters, actors
- Film metadata enrichment: plot summary, cast mapping
- Optional PostgreSQL persistence for metadata
- Offline model cache support
- Quantitative evaluation on labeled queries

### Out of scope
- Real-time streaming ingestion
- Distributed/cluster search infrastructure
- Fine-tuning foundation models
- Full OCR-first pipeline
- Strong production SLAs/observability

---

## Main pipeline
1. Load videos from `data/`.
2. Detect scene boundaries and extract keyframes.
3. Transcribe scene audio (unless `--skip_asr`).
4. Merge optional metadata overrides for scenes/films.
5. Build visual and text embeddings with SigLIP.
6. Store FAISS indices (`visual.index`, `text.index`) + `metadata.json` + `film_metadata.json`.
7. Optionally sync metadata to PostgreSQL.
8. For a query:
   - encode text query with SigLIP text tower
   - retrieve candidates from visual and/or speech index
   - optionally fuse rankings via RRF
   - optionally re-rank top candidates with VLM
9. Return ranked scenes with timestamps and rationale.

---

## Agents

### 1. Ingestion Agent (`build_vectordb.py`)
Responsible for:
- scanning input videos in `data/`
- validating supported formats
- orchestrating full build pipeline

Input:
- directory with videos

Output:
- intermediate artifacts in `output/`
- searchable DB in `db/`

### 2. Scene Detection Agent (`src/scene_detect.py`)
Responsible for:
- content-based scene split
- keyframe extraction
- scene metadata (`video`, `start_sec`, `end_sec`, `keyframe_path`)

### 3. ASR Agent (`src/asr.py`)
Responsible for:
- scene-level speech transcription via faster-whisper
- language/device/caching handling
- filling `transcript` field in scene metadata

### 4. Embedding Agent (`src/embed.py`)
Responsible for:
- loading SigLIP model and processor
- encoding keyframes (visual vectors)
- encoding transcripts/query text (text vectors)
- building/saving/loading FAISS indices

### 5. Retrieval Agent (`search.py`)
Responsible for:
- query encoding
- ANN search in selected mode: `visual`, `speech`, `fused`
- rank fusion using Reciprocal Rank Fusion (RRF)
- returning top-k timestamped scene candidates

### 6. Re-ranking / Answer Agent (`src/vlm_rerank.py`, `app.py`)
Responsible for:
- VLM-based scoring of retrieved candidates
- per-scene reason generation
- short summary answer for the user

Provider options:
- `ollama`
- `openrouter`

### 7. Evaluation Agent (`eval/`)
Responsible for:
- dataset format and loading (`EvalDataset`)
- retrieval metrics: Recall@K, MRR, nDCG, tIoU, offset
- optional generation quality via VLM-as-judge

### 8. Metadata Storage Agent (`src/metadata_store.py`)
Responsible for:
- persisting film-level metadata to PostgreSQL (`film_metadata`)
- persisting scene-level metadata to PostgreSQL (`scene_metadata`)
- upserting records without breaking local FAISS workflow

---

## Grounding rules
- Final answer must rely on retrieved scenes and their metadata.
- Use timestamps from metadata; do not invent intervals.
- If evidence is weak/conflicting, state low confidence explicitly.
- Do not fabricate dialogue/details absent in transcript or visual evidence.

---

## Data schema

### Scene metadata (`db/metadata.json`)
Required fields:
- `faiss_id`
- `video`
- `scene_id`
- `start_sec`
- `end_sec`
- `keyframe_path`
- `transcript`
- `transcript_text`
- `season_number`
- `episode_number`
- `characters_in_frame`
- `actors_in_frame`

### Film metadata (`db/film_metadata.json`)
Fields:
- `video` key
- `plot_summary`
- `cast_mapping` (`actor -> character`)

### Retrieval result
Typical fields:
- `video`
- `start_sec`
- `end_sec`
- `faiss_id`
- `score` or `rrf_score`
- optional `vlm_score`
- optional `vlm_reason`

### Eval dataset entry (`eval/eval_dataset.json`)
Fields:
- `query`
- `query_type`
- `video_id` (optional)
- `time_spans` (optional)
- `expected_answer` (optional)

---

## Success criteria
- End-to-end DB build completes for provided videos.
- Text query returns semantically relevant scenes with timestamps.
- Fused search is not worse than single-modality baselines on eval set.
- Evaluation metrics are reproducible for fixed config/checkpoints.

---

## Development principle
Prefer simple, inspectable components with clear CLI entry points:
- `python build_vectordb.py`
- `python search.py --query "..."`
- `python app.py`

When in doubt, prioritize reproducibility and debuggability over architectural complexity.
