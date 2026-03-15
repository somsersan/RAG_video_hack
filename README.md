# Video-RAG — поиск по видео

Система семантического поиска сцен в видеофайлах с помощью мультимодальных эмбеддингов и VLM-ранжирования.

---

## Как это работает

1. **Детекция сцен** — видео нарезается на сцены по изменению контента (PySceneDetect).
2. **ASR** — для каждой сцены извлекается транскрипт речи (faster-whisper).
3. **Метаданные** — сцены можно обогатить сезонами/эпизодами, персонажами и актёрами; фильм — summary и cast mapping.
4. **Эмбеддинги** — ключевые кадры и транскрипты кодируются через SigLIP в два FAISS-индекса.
5. **Поиск** — текстовый запрос кодируется SigLIP; поиск по визуальному и речевому индексу объединяется через Reciprocal Rank Fusion (RRF).
6. **VLM-ранжирование** — топ-кандидаты переранжируются VLM-моделью через Ollama или OpenRouter.

---

## Модели

| Компонент | Модель |
|---|---|
| Эмбеддинги (текст + кадры) | `google/siglip-so400m-patch14-384` |
| ASR (речь → текст) | `faster-whisper` (`tiny` / `base` / `small` / `medium` / `large-v3`) |
| VLM-ранжирование | `qwen3.5:9b` (Ollama) или совместимая модель в OpenRouter |

---

## Архитектура файлов

```
├── app.py               # Gradio веб-интерфейс
├── build_vectordb.py    # Пайплайн: видео → FAISS БД
├── search.py            # CLI-поиск по готовой БД
├── docker-compose.postgres.yml # PostgreSQL для метаданных (опционально)
├── requirements.txt
├── data/                # Видео + (опционально) референсы лиц актеров
│   └── actors/          # data/actors/<Actor Name>/*.jpg
├── output/
│   ├── all_scenes.json  # Найденные сцены
│   └── keyframes/       # Ключевые кадры по видео
├── db/
│   ├── visual.index     # FAISS-индекс по кадрам
│   ├── text.index       # FAISS-индекс по транскриптам
│   ├── metadata.json    # Метаданные сцен
│   └── film_metadata.json # Метаданные фильмов
└── src/
    ├── scene_detect.py  # Детекция сцен
    ├── asr.py           # Транскрибация (Whisper)
    ├── embed.py         # SigLIP/BGE-M3 эмбеддинги + FAISS
    ├── face_recognition.py # InsightFace: лица актеров в keyframes
    └── vlm_rerank.py    # VLM-ранжирование через Ollama/OpenRouter
```

---

## Установка

```bash
pip install -r requirements.txt
```

> Для GPU замените `faiss-cpu` на `faiss-gpu` в requirements.txt.  
> Для VLM-ранжирования через Ollama нужен запущенный [Ollama](https://ollama.com) с моделью:
> ```bash
> ollama pull qwen3.5:9b
> ```
>
> Для OpenRouter задайте ключ в переменной окружения (ключ не храните в коде):
> ```bash
> export OPENROUTER_API_KEY="sk-or-..."
> ```
> или положите в локальный `.env` (файл уже исключён из git):
> ```env
> OPENROUTER_API_KEY=sk-or-...
> ```
>
> Для локального PostgreSQL:
> ```bash
> docker compose -f docker-compose.postgres.yml up -d
> ```

---

## Использование

### 1. Положить видео (и при необходимости лица актеров)

Скопируйте видеофайлы (`.mp4` или `.mkv`) в папку `data/`.
Для распознавания актеров добавьте референсные фото в `data/actors/`:
`data/actors/<Имя актера>/<фото>.jpg`.

### 2. Построить базу данных

```bash
python build_vectordb.py
```

Основные опции:

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--data_dir` | `data` | Папка с видео (`.mp4`/`.mkv`) |
| `--db_dir` | `db` | Куда сохранить FAISS |
| `--whisper_model` | `base` | Размер Whisper: tiny/base/small/medium/large-v3 |
| `--language` | авто | Код языка, напр. `ru` |
| `--device` | `cpu` | `cpu` или `cuda` |
| `--hf_cache_dir` | `.model_cache/hf` | Локальный кэш HuggingFace моделей (SigLIP) |
| `--offline_models` | — | Загружать модели только из локального кэша (без скачивания) |
| `--skip_asr` | — | Пропустить транскрибацию |
| `--skip_scenes` | — | Переиспользовать scenes.json |
| `--scene_metadata_json` | — | JSON с scene metadata overrides (`season_number`, `episode_number`, `characters_in_frame`, `actors_in_frame`, `transcript_text`) |
| `--film_metadata_json` | — | JSON с film metadata (`plot_summary`, `cast_mapping`) |
| `--enable_face_detection` | — | Включить детекцию/идентификацию лиц через InsightFace |
| `--faces_dir` | `data/actors` | Папка с референсными фото актеров |
| `--face_model` | `buffalo_l` | InsightFace model pack |
| `--face_similarity_threshold` | `0.4` | Порог cosine similarity для матчинга актера |
| `--pg_dsn` | — | PostgreSQL DSN для sync метаданных |
| `--pg_schema` | `public` | PostgreSQL schema для таблиц `film_metadata` и `scene_metadata` |

Пример запуска с метаданными и PostgreSQL:
```bash
python build_vectordb.py \
  --enable_face_detection \
  --faces_dir data/actors \
  --scene_metadata_json notebooks/scene_metadata.sample.json \
  --film_metadata_json notebooks/film_metadata.sample.json \
  --pg_dsn "postgresql://video_rag:video_rag@localhost:5432/video_rag"
```

### Локальное хранилище моделей (без повторной загрузки)

1. Один раз скачайте модели в локальный кэш:
```bash
python3 build_vectordb.py \
  --hf_cache_dir .model_cache/hf
```

2. Дальше работайте офлайн (без интернета, если набор моделей не менялся):
```bash
python3 build_vectordb.py \
  --hf_cache_dir .model_cache/hf \
  --offline_models
```

### 3. Поиск через CLI

```bash
python search.py --query "Шрек бежит по болоту"
python search.py --query "погоня" --mode visual --top_k 5
python search.py --query "герой разговаривает с ослом" --vlm
python search.py --query "герой разговаривает с ослом" --vlm --vlm_provider openrouter --vlm_model openai/gpt-4o-mini
```

| Флаг | Описание |
|---|---|
| `--mode` | `fused` (по умолчанию) / `visual` / `speech` |
| `--top_k` | Количество результатов (по умолчанию 5) |
| `--hf_cache_dir` | Локальный кэш HuggingFace моделей |
| `--offline_models` | Загружать SigLIP только из локального кэша |
| `--vlm` | Включить VLM-ранжирование |
| `--vlm_provider` | Провайдер VLM: `ollama` (по умолчанию) или `openrouter` |
| `--vlm_model` | Модель выбранного провайдера |

### 4. Веб-интерфейс

```bash
python app.py
```

Открыть в браузере: `http://localhost:7860`

Опции запуска:
```bash
python app.py --device cuda --vlm_model qwen3.5:9b
python app.py --vlm_provider openrouter --vlm_model openai/gpt-4o-mini
python app.py --hf_cache_dir .model_cache/hf --offline_models
```

### Как выбирать модели в OpenRouter

Практическое правило: оставляем ту же роль модели, что была в Ollama.

| Если в Ollama использовалось | Что брать в OpenRouter | Почему |
|---|---|---|
| `qwen3.5:9b` для VLM rerank | `openai/gpt-4o-mini` или `google/gemini-2.0-flash-001` | Быстро, недорого, хорошо держит короткий формат `REASON/SCORE` |
| Текстовый rerank без кадров (`--vlm_text_only`) | Любая быстрая text-only/omni модель (`openai/gpt-4o-mini`, `anthropic/claude-3.5-haiku`) | Главное: стабильная оценка релевантности, визуал не нужен |
| Сложные визуальные сцены | Более сильная мультимодальная модель (`openai/gpt-4.1`, `anthropic/claude-3.7-sonnet`) | Лучше reasoning по кадрам, но выше задержка/стоимость |

Критерии выбора:
1. Поддержка изображений (если используете кадры, без `--vlm_text_only`).
2. Стабильное следование формату ответа (`REASON` + `SCORE`).
3. Баланс latency/price для количества кандидатов (`--vlm_candidates`).

---

## Документация

- Полная текстовая документация проекта: `docs/project_documentation.md`
- Диаграмма потока данных и трансформаций: `docs/data_flow_diagram.md`

## Формат новых метаданных

`scene_metadata_json`:
```json
[
  {
    "video": "shrek.mp4",
    "scene_id": 12,
    "season_number": null,
    "episode_number": null,
    "characters_in_frame": ["Шрек", "Осёл"],
    "actors_in_frame": ["Mike Myers", "Eddie Murphy"],
    "transcript_text": "..."
  }
]
```

`film_metadata_json`:
```json
{
  "shrek.mp4": {
    "plot_summary": "Краткий синопсис фильма",
    "cast_mapping": {
      "Mike Myers": "Shrek",
      "Eddie Murphy": "Donkey"
    }
  }
}
```

`data/actors` (для InsightFace):
```text
data/actors/
  Mike Myers/
    01.jpg
    02.jpg
  Eddie Murphy/
    ref.png
```

После включения `--enable_face_detection` pipeline:
- детектит лица на keyframes каждой сцены,
- заполняет `actors_in_frame`,
- через `cast_mapping` добавляет `characters_in_frame`,
- автоматически добавляет эту информацию в `transcript_text`/`subtitle` сцены на соответствующем таймлайне.
