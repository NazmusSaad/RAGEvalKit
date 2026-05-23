"""Deterministic run summarisation and root-cause classification.

No LLM calls — all logic is rule-based and operates on metric_scores rows
already stored in DuckDB.

Required metrics (all must be present and non-unknown for a "pass"):
  recall_at_k, answer_relevance, faithfulness

Optional / diagnostic only:
  mrr

Root-cause precedence (first matching rule wins):
  1. recall_at_k missing           → missing_metric
  2. recall_at_k label=unknown     → judge_uncertain
  3. recall_at_k score == 0.0      → retrieval_failure
  4. faithfulness missing          → missing_metric
  5. faithfulness label=unknown    → judge_uncertain
  6. faithfulness label=fail       → grounding_failure
  7. answer_relevance missing      → missing_metric
  8. answer_relevance label=unknown→ judge_uncertain
  9. answer_relevance label=fail   → answer_relevance_failure
  10. all required pass             → none
  11. fallback                      → judge_uncertain
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REQUIRED_METRICS: list[str] = ["recall_at_k", "answer_relevance", "faithfulness"]
OPTIONAL_METRICS: list[str] = ["mrr"]

_SUGGESTED_FIXES: dict[str, str] = {
    "retrieval_failure": (
        "Improve retrieval by adjusting chunk size, increasing top_k, "
        "changing the embedding model, or adding reranking."
    ),
    "grounding_failure": (
        "Improve grounding by tightening the prompt to answer only from "
        "context, or by improving retrieved context quality."
    ),
    "answer_relevance_failure": (
        "Improve the generation prompt so the model directly answers "
        "the user question."
    ),
    "judge_uncertain": (
        "Review judge outputs and rerun evaluation with a stronger judge model."
    ),
    "missing_metric": (
        "Run the missing evaluator before summarizing this run."
    ),
    "none": "",
}


@dataclass
class ItemSummary:
    item_id: str
    question_id: str
    overall_label: str          # "pass" | "fail" | "unknown"
    reason: str
    primary_cause: str          # see module docstring for values
    secondary_causes: list[str] = field(default_factory=list)
    suggested_fix: str = ""
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class RunSummary:
    run_id: str
    items: list[ItemSummary]

    @property
    def pass_count(self) -> int:
        return sum(1 for i in self.items if i.overall_label == "pass")

    @property
    def fail_count(self) -> int:
        return sum(1 for i in self.items if i.overall_label == "fail")

    @property
    def unknown_count(self) -> int:
        return sum(1 for i in self.items if i.overall_label == "unknown")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _overall_label(metrics: dict[str, dict]) -> tuple[str, str]:
    """Return (overall_label, reason) for one item."""
    for name in REQUIRED_METRICS:
        m = metrics.get(name)
        if m is None:
            return "unknown", f"missing metric: {name}"
        if m.get("label") == "unknown":
            return "unknown", f"judge uncertain for metric: {name}"

    for name in REQUIRED_METRICS:
        if metrics[name].get("label") == "fail":
            return "fail", f"metric {name} failed"

    return "pass", "all required metrics passed"


def classify_root_cause(metrics: dict[str, dict]) -> tuple[str, list[str]]:
    """Determine primary and secondary root causes from metric scores.

    Returns ``("none", [])`` when all required metrics pass.
    Only meaningful to call when overall label is not "pass".
    """
    recall = metrics.get("recall_at_k")
    faith = metrics.get("faithfulness")
    ar = metrics.get("answer_relevance")

    # Primary: first matching rule
    if recall is None:
        primary = "missing_metric"
    elif recall.get("label") == "unknown":
        primary = "judge_uncertain"
    elif recall.get("score", 1.0) == 0.0:
        primary = "retrieval_failure"
    elif faith is None:
        primary = "missing_metric"
    elif faith.get("label") == "unknown":
        primary = "judge_uncertain"
    elif faith.get("label") == "fail":
        primary = "grounding_failure"
    elif ar is None:
        primary = "missing_metric"
    elif ar.get("label") == "unknown":
        primary = "judge_uncertain"
    elif ar.get("label") == "fail":
        primary = "answer_relevance_failure"
    else:
        primary = "none"

    # Collect all issues across all three required metrics
    potential: list[str] = []

    if recall is None:
        potential.append("missing_metric")
    elif recall.get("label") == "unknown":
        potential.append("judge_uncertain")
    elif recall.get("score", 1.0) == 0.0:
        potential.append("retrieval_failure")

    if faith is None:
        potential.append("missing_metric")
    elif faith.get("label") == "unknown":
        potential.append("judge_uncertain")
    elif faith.get("label") == "fail":
        potential.append("grounding_failure")

    if ar is None:
        potential.append("missing_metric")
    elif ar.get("label") == "unknown":
        potential.append("judge_uncertain")
    elif ar.get("label") == "fail":
        potential.append("answer_relevance_failure")

    # Secondary = unique non-primary issues
    seen = {primary}
    secondary: list[str] = []
    for cause in potential:
        if cause not in seen:
            seen.add(cause)
            secondary.append(cause)

    return primary, secondary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_item(
    item_id: str,
    question_id: str,
    metrics: dict[str, dict],
) -> ItemSummary:
    """Build an :class:`ItemSummary` from the metric scores for one item."""
    overall_label, reason = _overall_label(metrics)

    if overall_label == "pass":
        primary = "none"
        secondary: list[str] = []
    else:
        primary, secondary = classify_root_cause(metrics)

    return ItemSummary(
        item_id=item_id,
        question_id=question_id,
        overall_label=overall_label,
        reason=reason,
        primary_cause=primary,
        secondary_causes=secondary,
        suggested_fix=_SUGGESTED_FIXES.get(primary, ""),
        metrics=metrics,
    )


def build_run_summary(
    run_id: str,
    items_data: list[dict],
    scores_data: list[dict],
) -> RunSummary:
    """Build a :class:`RunSummary` from raw DuckDB rows.

    *items_data*: from ``get_run_items_basic`` (each has ``item_id``, ``question_id``).
    *scores_data*: from ``get_metric_scores_for_run`` (each has ``item_id``, ``metric``,
    ``score``, ``label``).
    """
    scores_by_item: dict[str, dict] = {}
    for row in scores_data:
        iid = row["item_id"]
        if iid not in scores_by_item:
            scores_by_item[iid] = {}
        scores_by_item[iid][row["metric"]] = {
            "score": row["score"],
            "label": row["label"],
        }

    items = [
        summarize_item(
            item_id=item["item_id"],
            question_id=item.get("question_id", ""),
            metrics=scores_by_item.get(item["item_id"], {}),
        )
        for item in items_data
    ]
    return RunSummary(run_id=run_id, items=items)


def mean_metric(run_summary: RunSummary, metric_name: str) -> str:
    """Return mean score for *metric_name* over non-unknown items, or 'N/A'."""
    scores = [
        item.metrics[metric_name]["score"]
        for item in run_summary.items
        if metric_name in item.metrics
        and item.metrics[metric_name].get("label") != "unknown"
    ]
    return f"{sum(scores) / len(scores):.3f}" if scores else "N/A"
