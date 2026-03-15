# Текстовая документация проекта Video-RAG

## 1. Назначение проекта

`ai-chemp-cu-itmo` — прототип системы семантического поиска моментов в видео.

На вход подается текстовый запрос пользователя (например, "погоня в лесу"), на выходе система возвращает:
1. релевантные сцены;
2. таймкоды (`start_sec`, `end_sec`);
3. пояснение релевантности (при включенном VLM-ранжировании).

Проект ориентирован на локальный запуск и быстрые эксперименты, а не на production-нагрузку.

## 2. Состав репозитория

### 2.1 Основные скрипты

- `build_vectordb.py`  
  Оффлайн-пайплайн индексации: сцены -> ASR -> эмбеддинги -> FAISS.

- `search.py`  
  CLI-поиск по готовой базе (`db/`), поддерживает режимы `visual`, `speech`, `fused` и опциональный VLM re-rank.

- `app.py`  
  Gradio веб-интерфейс поверх того же retrieval пайплайна.

### 2.2 Модули в `src/`

- `src/scene_detect.py`  
  Детекция сцен через PySceneDetect и извлечение keyframe на сцену.

- `src/asr.py`  
  Транскрибация речи через faster-whisper.

- `src/_whisper_worker.py`  
  Воркер-процесс, который запускает Whisper и стримит сегменты JSONL.

- `src/embed.py`  
  Извлечение эмбеддингов SigLIP (визуальных и текстовых), сборка FAISS индексов, сохранение/загрузка БД.

- `src/vlm_rerank.py`  
  Доранжирование кандидатов через VLM-модель (Ollama или OpenRouter).

### 2.3 Рабочие директории

- `data/` — исходные видеофайлы (`.mp4` и `.mkv`);
- `output/` — промежуточные артефакты (`all_scenes.json`, keyframes);
- `db/` — готовая векторная БД (`visual.index`, `text.index`, `metadata.json`);
- `docs/` — проектная документация.

## 3. Архитектура пайплайна

Проект состоит из двух независимых частей:
1. Offline indexing pipeline (подготовка базы поиска).
2. Online retrieval pipeline (обработка пользовательского запроса).

### 3.1 Offline indexing pipeline

Шаг 1. Детекция сцен (`src/scene_detect.py`)
- Видео режется на сцены алгоритмом `ContentDetector`.
- Для каждой сцены вычисляются:
  - `start_sec`,
  - `end_sec`,
  - `keyframe_path` (кадр из середины сцены).
- Формируется список сцен (`all_scenes`), который сохраняется в `output/all_scenes.json`.

Шаг 2. ASR (`src/asr.py`, `src/_whisper_worker.py`)
- Аудио извлекается из видео через `ffmpeg` в mono 16kHz wav.
- Whisper возвращает временные сегменты речи (`start`, `end`, `text`).
- Каждый ASR сегмент сопоставляется сцене по midpoint-схеме:
  midpoint сегмента должен попасть в интервал сцены.
- Тексты агрегируются в `scene["transcript"]`.

Шаг 3. Эмбеддинги и индексы (`src/embed.py`)
- Загружается SigLIP модель и processor.
- Строятся два независимых массива эмбеддингов:
  - `visual_embeddings` по keyframe;
  - `text_embeddings` по transcript.
- Оба массива L2-нормализуются и сохраняются в FAISS `IndexFlatIP`.
- Метаданные сцен сериализуются в `db/metadata.json` с `faiss_id`.

Итог offline этапа: директория `db/` готова к поиску.

### 3.2 Online retrieval pipeline

Шаг 1. Кодирование запроса (`search.encode_query`)
- Текст запроса кодируется через SigLIP text tower в вектор `q_vec`.
- Вектор L2-нормализуется.

Шаг 2. Поиск по индексам
- `visual`: поиск по `db/visual.index`.
- `speech`: поиск по `db/text.index`.
- `fused`: поиск по обоим + объединение через RRF.

Шаг 3. Маппинг в сцены
- Найденные `faiss_id` маппятся через `db/metadata.json`.
- Получаем конкретные сцены и таймкоды.

Шаг 4. Опциональный VLM re-rank
- Для кандидатов извлекаются несколько кадров из интервала сцены.
- Кадры и/или transcript отправляются в VLM-модель выбранного провайдера.
- Модель возвращает `SCORE` и `REASON`, после чего кандидаты сортируются по `vlm_score`.

Шаг 5. Вывод
- CLI: печатает топ результатов с таймкодами.
- Gradio UI: показывает результаты и позволяет вырезать/проиграть соответствующий клип.

## 4. Модель данных

### 4.1 Scene record

Базовая запись сцены (в `all_scenes.json` и `metadata.json`):

```json
{
  "video": "movie.mp4",
  "scene_id": 12,
  "start_sec": 318.52,
  "end_sec": 327.41,
  "keyframe_path": "output/keyframes/movie/scene_0012.jpg",
  "transcript": "..."
}
```

### 4.2 Metadata record с FAISS ID

```json
{
  "faiss_id": 12,
  "video": "movie.mp4",
  "scene_id": 12,
  "start_sec": 318.52,
  "end_sec": 327.41,
  "keyframe_path": "output/keyframes/movie/scene_0012.jpg",
  "transcript": "..."
}
```

### 4.3 Индексы

- `db/visual.index`: векторы кадров сцен;
- `db/text.index`: векторы транскриптов сцен;
- размерность векторов задается моделью SigLIP.

## 5. Извлечение эмбеддингов (детально)

### 5.1 Визуальные эмбеддинги

Источник: `keyframe_path` каждой сцены.

Процесс:
1. Загрузка изображения.
2. Препроцессинг через `AutoProcessor`.
3. Прогон через `model.vision_model`.
4. Извлечение pooled feature (`pooler_output` или fallback).
5. L2-нормализация.
6. Конвертация в `float32`.

Результат: `visual_embeddings` формы `N x D`.

### 5.2 Текстовые эмбеддинги

Источник: `transcript` каждой сцены.

Процесс:
1. Пустые строки заменяются на placeholder (`[без речи]`).
2. Токенизация через `AutoProcessor`.
3. Прогон через `model.text_model`.
4. Извлечение pooled feature.
5. L2-нормализация.
6. Конвертация в `float32`.

Результат: `text_embeddings` формы `N x D`.

### 5.3 Почему используется `IndexFlatIP`

Эмбеддинги заранее L2-нормализованы, поэтому inner product эквивалентен cosine similarity.  
Это позволяет использовать простой и точный `FAISS IndexFlatIP` без дополнительного преобразования метрики.

## 6. Интерфейсы запуска

### 6.1 Подготовка базы

```bash
python build_vectordb.py --data_dir data --output_dir output --db_dir db
```

Ключевые параметры:
- `--scene_threshold`, `--min_scene_len`: качество/частота нарезки сцен;
- `--whisper_model`, `--language`, `--device`: параметры ASR;
- `--img_batch`, `--txt_batch`: размеры батчей эмбеддингов;
- `--skip_scenes`, `--skip_asr`: ускорение повторных прогонов.

### 6.2 CLI поиск

```bash
python search.py --query "герой бежит по лесу" --mode fused --top_k 5
python search.py --query "разговор" --mode speech
python search.py --query "драка" --mode visual --vlm --vlm_provider ollama --vlm_model qwen3.5:9b
python search.py --query "драка" --mode visual --vlm --vlm_provider openrouter --vlm_model openai/gpt-4o-mini
```

### 6.3 Web UI

```bash
python app.py --db_dir db --data_dir data --device cpu --vlm_provider ollama --vlm_model qwen3.5:9b
python app.py --db_dir db --data_dir data --device cpu --vlm_provider openrouter --vlm_model openai/gpt-4o-mini
```

После старта интерфейс доступен на `http://localhost:7860`.

## 7. Внешние зависимости и сервисы

1. Python-пакеты из `requirements.txt`.
2. Бинарник `ffmpeg` в `PATH`.
3. Для VLM re-rank:
- `ollama` provider: запущенный `ollama serve` и загруженная модель (`ollama pull qwen3.5:9b` или другая).
- `openrouter` provider: заданный `OPENROUTER_API_KEY` (через переменную окружения или локальный `.env`, не коммитить в git).
4. Для хранения моделей локально:
- `--hf_cache_dir` для SigLIP (HuggingFace);
- `--whisper_cache_dir` для faster-whisper;
- `--offline_models` для запуска только из локального кэша (без загрузки из сети).

## 8. Ограничения текущей реализации

1. Система ориентирована на локальный прототип и небольшие наборы видео.
2. Точность зависит от качества scene detection и ASR.
3. Отсутствует отдельная работа с OCR/субтитрами.
4. VLM re-rank добавляет задержку и требует внешнего VLM-провайдера (локально через Ollama или через OpenRouter API).
5. Поиск опирается на scene-level гранулярность, а не на точный frame-level grounding.

## 9. Отладка и типовые проблемы

1. `No video files found`  
Проверьте, что видео лежат в `data/` и имеют поддерживаемое расширение (`.mp4` или `.mkv`).

2. Ошибка `ffmpeg`  
Убедитесь, что `ffmpeg` установлен и доступен из командной строки.

3. Ошибки провайдера VLM  
- Для `ollama`: запустите `ollama serve` и проверьте, что модель скачана.  
- Для `openrouter`: проверьте, что `OPENROUTER_API_KEY` задан и валиден.

4. Медленная работа на CPU  
Уменьшите размер Whisper-модели и батчи эмбеддингов, либо используйте `--device cuda`.

## 10. Связанные документы

- Диаграмма данных и трансформаций: `docs/data_flow_diagram.md`.
- Краткий старт и команды: `README.md`.
