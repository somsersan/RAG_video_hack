# Video-RAG — поиск по видео

Система семантического поиска сцен в видеофайлах с помощью мультимодальных эмбеддингов и VLM-ранжирования.

---

## Как это работает

1. **Детекция сцен** — видео нарезается на сцены по изменению контента (PySceneDetect).
2. **ASR** — для каждой сцены извлекается транскрипт речи (faster-whisper).
3. **Эмбеддинги** — ключевые кадры и транскрипты кодируются через SigLIP в два FAISS-индекса.
4. **Поиск** — текстовый запрос кодируется SigLIP; поиск по визуальному и речевому индексу объединяется через Reciprocal Rank Fusion (RRF).
5. **VLM-ранжирование** — топ-кандидаты переранжируются VLM-моделью через Ollama или OpenRouter.

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
├── requirements.txt
├── data/                # Сюда кладём .mp4 файлы
├── output/
│   ├── all_scenes.json  # Найденные сцены
│   └── keyframes/       # Ключевые кадры по видео
├── db/
│   ├── visual.index     # FAISS-индекс по кадрам
│   ├── text.index       # FAISS-индекс по транскриптам
│   └── metadata.json    # Метаданные сцен
└── src/
    ├── scene_detect.py  # Детекция сцен
    ├── asr.py           # Транскрибация (Whisper)
    ├── embed.py         # SigLIP эмбеддинги + FAISS
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

---

## Использование

### 1. Положить видео

Скопируйте `.mp4` файлы в папку `data/`.

### 2. Построить базу данных

```bash
python build_vectordb.py
```

Основные опции:

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--data_dir` | `data` | Папка с .mp4 |
| `--db_dir` | `db` | Куда сохранить FAISS |
| `--whisper_model` | `base` | Размер Whisper: tiny/base/small/medium/large-v3 |
| `--language` | авто | Код языка, напр. `ru` |
| `--device` | `cpu` | `cpu` или `cuda` |
| `--skip_asr` | — | Пропустить транскрибацию |
| `--skip_scenes` | — | Переиспользовать scenes.json |

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
