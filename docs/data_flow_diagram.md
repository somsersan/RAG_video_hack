# Data Flow Diagram (Video-RAG)

Ниже описан полный поток данных и трансформации в проекте: от входного видео до выдачи таймкода найденного момента.

## 1) End-to-end диаграмма (Mermaid)

```mermaid
flowchart TD
    A[data/*.mp4] --> B[build_vectordb.py]

    subgraph OFFLINE["Offline indexing pipeline"]
        B --> C[scene_detect.detect_scenes]
        C --> C1[Scene boundaries<br/>start_sec/end_sec]
        C --> C2[Keyframe extraction<br/>scene_i.jpg]
        C1 --> D[all_scenes: List[scene]]
        C2 --> D

        D --> E[asr.transcribe_video]
        E --> E1[ffmpeg -> mono 16k wav]
        E1 --> E2[_whisper_worker.py<br/>segments: start/end/text]
        E2 --> E3[Assign ASR segments to scenes<br/>by segment midpoint]
        E3 --> F[all_scenes + transcript]

        F --> G[src.embed.load_model<br/>SigLIP]
        G --> H[embed_images(keyframe_path)]
        G --> I[embed_texts(transcript)]

        H --> H1[visual_embeddings: N x D<br/>L2-normalized float32]
        I --> I1[text_embeddings: N x D<br/>L2-normalized float32]

        H1 --> J[FAISS IndexFlatIP]
        I1 --> K[FAISS IndexFlatIP]

        F --> L[metadata.json<br/>faiss_id + scene fields]
        J --> M[db/visual.index]
        K --> N[db/text.index]
        L --> O[db/metadata.json]
    end

    subgraph ONLINE["Online retrieval pipeline"]
        P[query text] --> Q[search.encode_query<br/>SigLIP text tower]
        Q --> Q1[q_vec: 1 x D, L2-normalized]

        Q1 --> R[Search visual.index]
        Q1 --> S[Search text.index]
        R --> T[visual ranking]
        S --> U[text ranking]
        T --> V[RRF fusion]
        U --> V
        V --> W[top candidate scene IDs]

        W --> X[metadata lookup<br/>video, start_sec, end_sec, transcript]
        X --> Y{VLM rerank enabled?}
        Y -->|No| Z[Return top-k with timestamps]
        Y -->|Yes| AA[src.vlm_rerank.rerank]
        AA --> AB[extract N frames per candidate]
        AB --> AC[Ollama/OpenRouter scoring + reason]
        AC --> AD[Sort by vlm_score]
        AD --> AE[Return top-k with timestamps + reason]
    end

    M -. used by .-> R
    N -. used by .-> S
    O -. used by .-> X
```

## 2) Ключевые сущности данных и их трансформации

1. `scene` (базовая запись после scene detection)
- Поля: `video`, `scene_id`, `start_sec`, `end_sec`, `keyframe_path`, `transcript`.
- Источник: `src/scene_detect.py`.
- Трансформация: сначала `transcript=""`, затем заполняется после ASR.

2. `all_scenes.json`
- Список всех `scene` по всем видео.
- Используется как checkpoint между этапами scene detection, ASR и embedding.

3. `ASR segments`
- Временные сегменты речи `{"start","end","text"}` от faster-whisper.
- Не хранятся отдельно в финальной БД, а агрегируются в `scene["transcript"]`.

4. `visual_embeddings` и `text_embeddings`
- Матрицы `N x D` (`float32`, L2-нормализованные).
- `N` = число сцен, `D` = размерность SigLIP.
- Визуальные векторы строятся по keyframe, текстовые по transcript.

5. `FAISS indices`
- `visual.index`: эмбеддинги кадров.
- `text.index`: эмбеддинги транскриптов.
- Тип: `IndexFlatIP` (из-за L2-нормализации IP эквивалентен cosine similarity).

6. `metadata.json`
- Строка метаданных на каждый `faiss_id`.
- Гарантирует связь `faiss_id -> scene timestamps + video path`.

7. `query vector (q_vec)`
- Текст запроса -> SigLIP text embedding -> L2 normalization.
- Ищется одновременно в `visual.index` и `text.index` (режим `fused`).

8. `retrieval results`
- До VLM: `(score, faiss_id)` из FAISS/RRF.
- После VLM: сцены с `vlm_score`, `vlm_reason`, `start_sec/end_sec`.

## 3) Где это в коде

- Индексация: `build_vectordb.py`
- Детекция сцен/ключкадров: `src/scene_detect.py`
- ASR и привязка текста к сценам: `src/asr.py`, `src/_whisper_worker.py`
- Эмбеддинги и FAISS: `src/embed.py`
- Поиск и RRF: `search.py`
- VLM rerank: `src/vlm_rerank.py`
- UI-пайплайн (поиск + клип): `app.py`
