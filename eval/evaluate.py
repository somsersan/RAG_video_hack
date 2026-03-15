"""
Метрики качества Video RAG.

Работает поверх пайплайна ai-chemp-cu-itmo:
- search.py: encode_query, search_index, reciprocal_rank_fusion
- src/embed.py: load_model, load_db

Два режима оценки:
- Retrieval: Recall@K, MRR, nDCG, tIoU, offset — для записей с time_spans
- Generation: VLM-as-judge — для записей с expected_answer
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import faiss
import numpy as np
from transformers import AutoModel, AutoProcessor

from eval.dataset import EvalDataset, TimeSpan
from src.metadata_schema import transcript_text

logger = logging.getLogger(__name__)


def fmt_time(seconds: float) -> str:
    if seconds == float("inf"):
        return "—"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


JUDGE_PROMPT = """\
Ты — строгий оценщик качества ответов системы поиска по фильмам.

Вопрос пользователя: {query}

Ожидаемый ответ: {expected}

Ответ системы: {actual}

Оцени ответ системы по шкале от 1 до 5:
1 — полностью неверный или нерелевантный
2 — частично затрагивает тему, но по сути неверный
3 — в целом верный, но с существенными неточностями или упущениями
4 — верный, с незначительными неточностями
5 — полностью верный и полный

Верни ТОЛЬКО число от 1 до 5, без пояснений.
"""


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    ndcg: float
    mean_tiou: float
    mean_offset_sec: float
    total_queries: int
    per_query: list[dict]

    def __repr__(self) -> str:
        return (
            f"Recall@K={self.recall_at_k:.3f}  "
            f"MRR={self.mrr:.3f}  "
            f"nDCG={self.ndcg:.3f}  "
            f"tIoU={self.mean_tiou:.3f}  "
            f"offset={fmt_time(self.mean_offset_sec)}  "
            f"(n={self.total_queries})"
        )


@dataclass
class GenerationMetrics:
    avg_score: float
    total_queries: int
    per_query: list[dict]

    def __repr__(self) -> str:
        return f"AvgScore={self.avg_score:.2f}/5  (n={self.total_queries})"


@dataclass
class FullMetrics:
    retrieval: RetrievalMetrics | None
    generation: GenerationMetrics | None

    def __repr__(self) -> str:
        parts = []
        if self.retrieval:
            parts.append(f"Retrieval: {self.retrieval}")
        if self.generation:
            parts.append(f"Generation: {self.generation}")
        return "\n".join(parts)


class Evaluator:
    """Evaluator поверх FAISS-индексов и search.py."""

    def __init__(
        self,
        vis_index: faiss.Index,
        txt_index: faiss.Index,
        metadata: List[Dict],
        model: AutoModel,
        processor: AutoProcessor,
        device: str = "cpu",
        top_k: int = 5,
    ):
        self.vis_index = vis_index
        self.txt_index = txt_index
        self.metadata = metadata
        self.model = model
        self.processor = processor
        self.device = device
        self.top_k = top_k

    def evaluate(
        self,
        dataset: EvalDataset,
        top_k: int | None = None,
        overlap_threshold: float = 0.3,
        judge: bool = False,
        judge_model: str = "google/gemini-2.0-flash-001",
        judge_api_key: str = "",
    ) -> FullMetrics:
        if top_k is not None:
            self.top_k = top_k

        retrieval = None
        generation = None

        retrieval_ds = dataset.with_time_spans()
        if len(retrieval_ds) > 0:
            retrieval = self._eval_retrieval(retrieval_ds, overlap_threshold)

        generation_ds = dataset.with_answers()
        if len(generation_ds) > 0 and judge and judge_api_key:
            generation = self._eval_generation(
                generation_ds, judge_model, judge_api_key,
            )

        return FullMetrics(retrieval=retrieval, generation=generation)

    # ── retrieval ─────────────────────────────────────────────────────

    def _eval_retrieval(
        self, dataset: EvalDataset, overlap_threshold: float,
    ) -> RetrievalMetrics:
        recalls, rrs, ndcgs, tious, offsets = [], [], [], [], []
        per_query = []

        for entry in dataset:
            gt_ids = set(EvalDataset.resolve_scene_ids(
                entry, self.metadata, overlap_threshold,
            ))
            if not gt_ids:
                logger.debug(
                    "SKIP (no GT scenes resolved) query=%r  time_spans=%s",
                    entry.query,
                    [(s.start, s.end) for s in entry.time_spans] if entry.time_spans else "—",
                )
                continue

            hits = [(score, int(fid)) for score, fid in self._retrieve(entry.query)]
            retrieved_ids = [fid for _, fid in hits]

            # temporal метрики
            gt_time = entry.time_spans or []
            ret_time = [
                TimeSpan(
                    start=self.metadata[fid]["start_sec"],
                    end=self.metadata[fid]["end_sec"],
                )
                for _, fid in hits if fid < len(self.metadata)
            ]
            best_tiou = self._best_temporal_iou(gt_time, ret_time)
            best_offset = self._best_temporal_offset(gt_time, ret_time)

            # debug
            gt_spans_str = ", ".join(
                f"scene {fid} [{fmt_time(self.metadata[fid]['start_sec'])}–{fmt_time(self.metadata[fid]['end_sec'])}]"
                for fid in sorted(gt_ids) if fid < len(self.metadata)
            )
            ret_spans_str = ", ".join(
                f"scene {fid} [{fmt_time(self.metadata[fid]['start_sec'])}–{fmt_time(self.metadata[fid]['end_sec'])}] (score={score:.4f})"
                for score, fid in hits if fid < len(self.metadata)
            )
            hit_str = "HIT" if best_offset <= 60.0 else "MISS"
            logger.debug(
                "[%s] query=%r\n  GT:        %s\n  Retrieved: %s\n  tIoU=%.3f  offset=%s",
                hit_str, entry.query, gt_spans_str, ret_spans_str, best_tiou, fmt_time(best_offset),
            )

            recall = self._recall(retrieved_ids, gt_ids)
            rr = self._reciprocal_rank(retrieved_ids, gt_ids)
            ndcg = self._ndcg(retrieved_ids, gt_ids)

            recalls.append(recall)
            rrs.append(rr)
            ndcgs.append(ndcg)
            tious.append(best_tiou)
            offsets.append(best_offset)

            per_query.append({
                "query": entry.query,
                "query_type": entry.query_type,
                "ground_truth": sorted(gt_ids),
                "ground_truth_spans": [
                    {"faiss_id": fid, "time": f"{fmt_time(self.metadata[fid]['start_sec'])}–{fmt_time(self.metadata[fid]['end_sec'])}"}
                    for fid in sorted(gt_ids) if fid < len(self.metadata)
                ],
                "retrieved": retrieved_ids,
                "retrieved_details": [
                    {"faiss_id": fid, "time": f"{fmt_time(self.metadata[fid]['start_sec'])}–{fmt_time(self.metadata[fid]['end_sec'])}", "score": score}
                    for score, fid in hits if fid < len(self.metadata)
                ],
                "recall": recall, "rr": rr, "ndcg": ndcg,
                "tiou": best_tiou, "offset_sec": best_offset, "offset": fmt_time(best_offset),
                "hit": hit_str,
            })

        n = len(recalls)
        return RetrievalMetrics(
            recall_at_k=sum(recalls) / n if n else 0,
            mrr=sum(rrs) / n if n else 0,
            ndcg=sum(ndcgs) / n if n else 0,
            mean_tiou=sum(tious) / n if n else 0,
            mean_offset_sec=sum(offsets) / n if n else 0,
            total_queries=n,
            per_query=per_query,
        )

    # ── generation (VLM-as-judge) ─────────────────────────────────────

    def _eval_generation(
        self,
        dataset: EvalDataset,
        judge_model: str,
        api_key: str,
    ) -> GenerationMetrics:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # для generation нужен VLM rerank (используем без VLM — просто transcript)
        scores = []
        per_query = []

        for entry in dataset:
            hits = [(score, int(fid)) for score, fid in self._retrieve(entry.query)]
            # собираем "ответ системы" из транскриптов найденных сцен
            answer_parts = []
            for score, fid in hits[:self.top_k]:
                scene = self.metadata[fid]
                ts = f"[{fmt_time(scene['start_sec'])}–{fmt_time(scene['end_sec'])}]"
                transcript = transcript_text(scene) or "—"
                answer_parts.append(f"{ts} {transcript}")
            actual_answer = "\n".join(answer_parts)

            judge_score = self._judge_answer(
                client, judge_model, entry.query, entry.expected_answer, actual_answer,
            )
            scores.append(judge_score)
            per_query.append({
                "query": entry.query,
                "query_type": entry.query_type,
                "expected": entry.expected_answer,
                "actual": actual_answer,
                "score": judge_score,
            })

        n = len(scores)
        return GenerationMetrics(
            avg_score=sum(scores) / n if n else 0,
            total_queries=n,
            per_query=per_query,
        )

    def _judge_answer(
        self, client, model: str, query: str, expected: str, actual: str,
    ) -> float:
        prompt = JUDGE_PROMPT.format(query=query, expected=expected, actual=actual)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0,
            )
            return float(resp.choices[0].message.content.strip())
        except (ValueError, Exception):
            return 0.0

    # ── retrieval: вызов search.py ────────────────────────────────────

    def _retrieve(self, query: str) -> List[Tuple[float, int]]:
        """Fused search: visual + speech → RRF. Returns [(score, faiss_id), ...]."""
        from search import encode_query, search_index, reciprocal_rank_fusion

        q_vec = encode_query(query, self.model, self.processor, self.device)

        n_search = min(self.top_k * 3, self.vis_index.ntotal)
        _, vis_ids = search_index(self.vis_index, q_vec, n_search)
        _, txt_ids = search_index(self.txt_index, q_vec, n_search)

        hits = reciprocal_rank_fusion([vis_ids.tolist(), txt_ids.tolist()])
        return hits[:self.top_k]

    # ── retrieval metric helpers ──────────────────────────────────────

    @staticmethod
    def _recall(retrieved: list[int], ground_truth: set[int]) -> float:
        if not ground_truth:
            return 0.0
        return len(set(retrieved) & ground_truth) / len(ground_truth)

    @staticmethod
    def _reciprocal_rank(retrieved: list[int], ground_truth: set[int]) -> float:
        for i, rid in enumerate(retrieved):
            if rid in ground_truth:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def _ndcg(retrieved: list[int], ground_truth: set[int]) -> float:
        dcg = 0.0
        for i, rid in enumerate(retrieved):
            if rid in ground_truth:
                dcg += 1.0 / math.log2(i + 2)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(len(ground_truth)))
        return dcg / idcg if idcg > 0 else 0.0

    # ── temporal metric helpers ───────────────────────────────────────

    @staticmethod
    def _tiou(a: TimeSpan, b: TimeSpan) -> float:
        inter_start = max(a.start, b.start)
        inter_end = min(a.end, b.end)
        inter = max(0.0, inter_end - inter_start)
        union = (a.end - a.start) + (b.end - b.start) - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _temporal_offset(a: TimeSpan, b: TimeSpan) -> float:
        if a.start <= b.end and b.start <= a.end:
            return 0.0
        return min(abs(a.start - b.end), abs(b.start - a.end))

    @classmethod
    def _best_temporal_iou(cls, gt_spans: list[TimeSpan], ret_spans: list[TimeSpan]) -> float:
        if not gt_spans or not ret_spans:
            return 0.0
        best_per_gt = []
        for gt in gt_spans:
            best = max(cls._tiou(gt, ret) for ret in ret_spans)
            best_per_gt.append(best)
        return sum(best_per_gt) / len(best_per_gt)

    @classmethod
    def _best_temporal_offset(cls, gt_spans: list[TimeSpan], ret_spans: list[TimeSpan]) -> float:
        if not gt_spans or not ret_spans:
            return float("inf")
        best_per_gt = []
        for gt in gt_spans:
            best = min(cls._temporal_offset(gt, ret) for ret in ret_spans)
            best_per_gt.append(best)
        return sum(best_per_gt) / len(best_per_gt)
