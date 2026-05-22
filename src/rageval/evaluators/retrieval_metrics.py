"""Deterministic retrieval evaluators: recall@k and MRR.

Both functions treat empty *source_chunk_ids* the same way:
score=0.0, label="unknown", with an explicit reason string.
No LLM calls are made.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetricResult:
    metric: str    # "recall_at_k" | "mrr"
    score: float   # 0.0 – 1.0
    label: str     # "pass" | "fail" | "unknown"
    reason: str


def compute_recall_at_k(
    source_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    k: int,
) -> float:
    """Return 1.0 if any source chunk appears in the top-*k* retrieved chunks, else 0.0.

    Returns 0.0 when *source_chunk_ids* is empty (caller should check the label).
    """
    if not source_chunk_ids:
        return 0.0
    top_k = set(retrieved_chunk_ids[:k])
    return 1.0 if any(s in top_k for s in source_chunk_ids) else 0.0


def compute_mrr(
    source_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    k: int | None = None,
) -> float:
    """Return 1/rank of the first relevant chunk (1-based rank), 0.0 if not found.

    *k* optionally caps the search window.
    Returns 0.0 when *source_chunk_ids* is empty (caller should check the label).
    """
    if not source_chunk_ids:
        return 0.0
    source_set = set(source_chunk_ids)
    window = retrieved_chunk_ids[:k] if k is not None else retrieved_chunk_ids
    for rank, chunk_id in enumerate(window, start=1):
        if chunk_id in source_set:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval_for_item(
    source_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
    k: int,
) -> list[RetrievalMetricResult]:
    """Compute recall@k and MRR for one run item.

    When *source_chunk_ids* is empty (no ground truth), both metrics are
    returned with ``score=0.0`` and ``label="unknown"``.
    """
    if not source_chunk_ids:
        reason = "no ground truth: source_chunk_ids is empty"
        return [
            RetrievalMetricResult("recall_at_k", 0.0, "unknown", reason),
            RetrievalMetricResult("mrr", 0.0, "unknown", reason),
        ]

    recall = compute_recall_at_k(source_chunk_ids, retrieved_chunk_ids, k)
    mrr = compute_mrr(source_chunk_ids, retrieved_chunk_ids, k)

    recall_reason = (
        f"source found in top-{k}"
        if recall > 0.0
        else f"source not found in top-{k}"
    )
    mrr_reason = (
        f"first relevant at rank {round(1 / mrr)}"
        if mrr > 0.0
        else "no relevant chunk found within top-k"
    )

    def _label(score: float) -> str:
        return "pass" if score > 0.0 else "fail"

    return [
        RetrievalMetricResult("recall_at_k", recall, _label(recall), recall_reason),
        RetrievalMetricResult("mrr", mrr, _label(mrr), mrr_reason),
    ]
