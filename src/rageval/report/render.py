"""Static HTML report renderer for completed evaluation runs.

No LLM calls.  Reads pre-computed data from DuckDB and renders a
self-contained HTML file via a Jinja2 template.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

_METRIC_ORDER = ["recall_at_k", "mrr", "answer_relevance", "faithfulness"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ItemReport:
    item_id: str
    question_id: str
    question: str
    generated_answer: str
    reference_answer: str
    overall_label: str                   # "pass" | "fail" | "unknown"
    metrics: dict[str, dict]             # {metric: {score, label, reason}}
    root_cause: dict | None              # {primary_cause, secondary_causes, suggested_fix}
    contexts: list[dict]                 # [{rank, chunk_id, chunk_text, score}]
    claims: list[dict]                   # [{claim_text, verdict, supporting_chunk_ids, rationale}]


@dataclass
class ReportData:
    run: dict                            # from get_run_by_id
    metric_means: dict[str, float | None]
    label_counts: dict[str, int]         # {"pass": N, "fail": N, "unknown": N}
    root_cause_distribution: dict[str, int]
    items: list[ItemReport]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_run_report_data(con: Any, run_id: str) -> ReportData:
    """Collect all data needed for the report from *con*."""
    from rageval.evaluators.summary import summarize_item
    from rageval.storage.duckdb_dao import (
        get_claims_for_item,
        get_metric_scores_for_run,
        get_retrieved_contexts_for_item,
        get_root_cause_distribution,
        get_root_causes_for_run,
        get_run_by_id,
        get_run_items_for_report,
        get_run_metric_means,
    )

    run = get_run_by_id(con, run_id)
    metric_means = get_run_metric_means(con, run_id)
    root_cause_distribution = get_root_cause_distribution(con, run_id)

    all_scores = get_metric_scores_for_run(con, run_id)
    scores_by_item: dict[str, dict] = {}
    for s in all_scores:
        scores_by_item.setdefault(s["item_id"], {})[s["metric"]] = {
            "score": s["score"],
            "label": s["label"],
            "reason": s["reason"] or "",
        }

    root_causes = get_root_causes_for_run(con, run_id)
    rc_by_item = {rc["item_id"]: rc for rc in root_causes}

    items_data = get_run_items_for_report(con, run_id)

    label_counts: dict[str, int] = {"pass": 0, "fail": 0, "unknown": 0}
    items: list[ItemReport] = []

    for row in items_data:
        item_id = row["item_id"]
        item_metrics = scores_by_item.get(item_id, {})

        item_summary = summarize_item(item_id, row["question_id"], item_metrics)
        lbl = item_summary.overall_label
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

        root_cause = rc_by_item.get(item_id)

        contexts = get_retrieved_contexts_for_item(con, item_id)

        claims_raw = get_claims_for_item(con, item_id)
        claims: list[dict] = []
        for c in claims_raw:
            supporting = c.get("supporting_chunk_ids", [])
            if isinstance(supporting, str):
                try:
                    supporting = json.loads(supporting)
                except (json.JSONDecodeError, ValueError):
                    supporting = []
            claims.append({
                "claim_text": c["claim_text"],
                "verdict": c["verdict"],
                "supporting_chunk_ids": supporting,
                "rationale": c.get("rationale") or "",
            })

        items.append(ItemReport(
            item_id=item_id,
            question_id=row["question_id"],
            question=row["question"],
            generated_answer=row["generated_answer"],
            reference_answer=row["reference_answer"],
            overall_label=lbl,
            metrics=item_metrics,
            root_cause=root_cause,
            contexts=contexts,
            claims=claims,
        ))

    return ReportData(
        run=run,
        metric_means=metric_means,
        label_counts=label_counts,
        root_cause_distribution=root_cause_distribution,
        items=items,
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _fmt_dt(value: Any) -> str:
    if value is None:
        return "—"
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    return str(value)


def _fmt_score(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _ordered_metric_rows(metric_means: dict[str, float | None]) -> list[dict]:
    all_keys = set(metric_means)
    ordered = [m for m in _METRIC_ORDER if m in all_keys]
    rest = sorted(m for m in all_keys if m not in _METRIC_ORDER)
    rows = []
    for m in ordered + rest:
        val = metric_means.get(m)
        rows.append({"name": m, "value": _fmt_score(val)})
    return rows


def render_run_report(data: ReportData) -> str:
    """Render *data* to a self-contained HTML string."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    env.filters["fmt_dt"] = _fmt_dt
    env.filters["fmt_score"] = _fmt_score

    template = env.get_template("run_report.html.j2")
    lc = data.label_counts
    return template.render(
        run=data.run,
        metric_rows=_ordered_metric_rows(data.metric_means),
        pass_count=lc.get("pass", 0),
        fail_count=lc.get("fail", 0),
        unknown_count=lc.get("unknown", 0),
        root_cause_distribution=data.root_cause_distribution,
        items=data.items,
    )
