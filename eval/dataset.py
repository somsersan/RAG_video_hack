"""
Eval-датасет для Video RAG.

Поддерживает три вида ground truth:
- time_spans: таймкоды моментов (для retrieval-метрик)
- expected_answer: ожидаемый ответ (для generation-метрик, VLM-as-judge)
- оба сразу

Формат JSON:
[
  {
    "query": "драка на крыше",
    "query_type": "visual",
    "video_id": "shrek.mp4",
    "time_spans": [{"start": 120.0, "end": 135.5}],
    "expected_answer": null
  }
]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class TimeSpan:
    start: float
    end: float


@dataclass
class QueryEntry:
    query: str
    query_type: str  # "visual" | "text" | "qa" | "manual"
    video_id: str = ""
    time_spans: list[TimeSpan] | None = None
    expected_answer: str | None = None
    metadata: dict = field(default_factory=dict)


class EvalDataset:
    def __init__(self, entries: list[QueryEntry] | None = None):
        self.entries: list[QueryEntry] = entries or []

    # ── ручное добавление ────────────────────────────────────────────

    def add(
        self,
        query: str,
        video_id: str = "",
        query_type: str = "manual",
        time_spans: list[tuple[float, float]] | tuple[float, float] | None = None,
        expected_answer: str | None = None,
        **extra_metadata,
    ) -> QueryEntry:
        """Добавить вопрос.

        Примеры:
            ds.add("драка на крыше", video_id="shrek.mp4",
                   time_spans=(120.0, 135.5))

            ds.add("почему герой ушёл?", query_type="qa",
                   expected_answer="Он узнал правду")

            ds.add("что решил герой?", time_spans=(300.0, 350.0),
                   expected_answer="Остаться и бороться")
        """
        spans = None
        if time_spans is not None:
            if isinstance(time_spans, tuple) and len(time_spans) == 2 and isinstance(time_spans[0], (int, float)):
                time_spans = [time_spans]
            spans = [TimeSpan(start=s, end=e) for s, e in time_spans]

        entry = QueryEntry(
            query=query,
            query_type=query_type,
            video_id=video_id,
            time_spans=spans,
            expected_answer=expected_answer,
            metadata=extra_metadata,
        )
        self.entries.append(entry)
        return entry

    def add_batch(self, items: list[dict]) -> int:
        """Добавить несколько вопросов разом из списка dict'ов."""
        known_keys = {"query", "video_id", "query_type", "time_spans", "expected_answer"}
        count = 0
        for item in items:
            self.add(
                query=item["query"],
                video_id=item.get("video_id", ""),
                query_type=item.get("query_type", "manual"),
                time_spans=item.get("time_spans"),
                expected_answer=item.get("expected_answer"),
                **{k: v for k, v in item.items() if k not in known_keys},
            )
            count += 1
        return count

    # ── резолв таймкодов → scene IDs (faiss_id) ───────────────────────

    @staticmethod
    def resolve_scene_ids(
        entry: QueryEntry,
        metadata: List[Dict],
        overlap_threshold: float = 0.3,
    ) -> list[int]:
        """Найти faiss_id сцен по таймкодам.

        Сцена считается совпадением, если доля перекрытия с GT span
        >= overlap_threshold от длительности сцены.
        """
        if not entry.time_spans:
            return []
        matched = []
        for scene in metadata:
            scene_start = scene["start_sec"]
            scene_end = scene["end_sec"]
            scene_dur = scene_end - scene_start
            if scene_dur <= 0:
                continue
            # фильтр по video_id если задан
            if entry.video_id and scene.get("video", "") != entry.video_id:
                continue
            for span in entry.time_spans:
                overlap_start = max(scene_start, span.start)
                overlap_end = min(scene_end, span.end)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap / scene_dur >= overlap_threshold:
                    matched.append(scene["faiss_id"])
                    break
        return matched

    # ── фильтрация ───────────────────────────────────────────────────

    def filter(
        self,
        query_type: str | None = None,
        video_id: str | None = None,
    ) -> EvalDataset:
        entries = self.entries
        if query_type is not None:
            entries = [e for e in entries if e.query_type == query_type]
        if video_id is not None:
            entries = [e for e in entries if e.video_id == video_id]
        return EvalDataset(entries)

    def with_time_spans(self) -> EvalDataset:
        return EvalDataset([e for e in self.entries if e.time_spans])

    def with_answers(self) -> EvalDataset:
        return EvalDataset([e for e in self.entries if e.expected_answer])

    # ── сохранение / загрузка ─────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for e in self.entries:
            d: dict = {"query": e.query, "query_type": e.query_type}
            if e.video_id:
                d["video_id"] = e.video_id
            if e.time_spans is not None:
                d["time_spans"] = [{"start": s.start, "end": s.end} for s in e.time_spans]
            if e.expected_answer is not None:
                d["expected_answer"] = e.expected_answer
            if e.metadata:
                d["metadata"] = e.metadata
            data.append(d)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> EvalDataset:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        entries = []
        for d in data:
            spans = None
            if "time_spans" in d:
                spans = [TimeSpan(**s) for s in d["time_spans"]]
            entries.append(QueryEntry(
                query=d["query"],
                query_type=d["query_type"],
                video_id=d.get("video_id", ""),
                time_spans=spans,
                expected_answer=d.get("expected_answer"),
                metadata=d.get("metadata", {}),
            ))
        return cls(entries)

    @classmethod
    def load_or_create(cls, path: str | Path) -> EvalDataset:
        path = Path(path)
        if path.exists():
            return cls.load(path)
        return cls()

    # ── merge ─────────────────────────────────────────────────────────

    def merge(self, other: EvalDataset, deduplicate: bool = True) -> None:
        if deduplicate:
            existing = {(e.video_id, e.query) for e in self.entries}
            for e in other.entries:
                key = (e.video_id, e.query)
                if key not in existing:
                    self.entries.append(e)
                    existing.add(key)
        else:
            self.entries.extend(other.entries)

    # ── helpers ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def summary(self) -> dict:
        from collections import Counter
        types = Counter(e.query_type for e in self.entries)
        videos = Counter(e.video_id for e in self.entries if e.video_id)
        with_spans = sum(1 for e in self.entries if e.time_spans)
        with_answers = sum(1 for e in self.entries if e.expected_answer)
        return {
            "total": len(self.entries),
            "by_type": dict(types),
            "by_video": dict(videos),
            "with_time_spans": with_spans,
            "with_expected_answer": with_answers,
        }

    def __repr__(self) -> str:
        return f"EvalDataset({self.summary()})"