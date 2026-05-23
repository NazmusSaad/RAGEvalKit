"""Deterministic run comparison module.

No LLM calls — operates entirely on pre-aggregated metric data
already stored in DuckDB.  All metrics are "higher is better."
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Canonical display order; any unlisted metrics are appended alphabetically.
_DISPLAY_ORDER = ["recall_at_k", "mrr", "answer_relevance", "faithfulness"]
_DELTA_THRESHOLD = 1e-9


@dataclass
class MetricDelta:
    metric: str
    baseline_mean: float | None      # None → metric absent / all unknown
    candidate_mean: float | None
    absolute_delta: float | None     # candidate - baseline; None if either is absent
    relative_delta: float | None     # delta / baseline; None if baseline is 0 or absent
    direction: str                   # "improved" | "regressed" | "unchanged" | "n/a"


@dataclass
class RunComparison:
    baseline_run_id: str
    candidate_run_id: str
    metric_deltas: list[MetricDelta]
    baseline_label_counts: dict[str, int]   # {"pass": N, "fail": N, "unknown": N}
    candidate_label_counts: dict[str, int]
    baseline_root_causes: dict[str, int]    # empty if summarize-run not run
    candidate_root_causes: dict[str, int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_delta(metric: str, baseline: float | None, candidate: float | None) -> MetricDelta:
    if baseline is None or candidate is None:
        return MetricDelta(
            metric=metric,
            baseline_mean=baseline,
            candidate_mean=candidate,
            absolute_delta=None,
            relative_delta=None,
            direction="n/a",
        )

    absolute_delta = candidate - baseline
    relative_delta = (
        absolute_delta / baseline
        if abs(baseline) >= _DELTA_THRESHOLD
        else None
    )

    if abs(absolute_delta) < _DELTA_THRESHOLD:
        direction = "unchanged"
    elif absolute_delta > 0:
        direction = "improved"
    else:
        direction = "regressed"

    return MetricDelta(
        metric=metric,
        baseline_mean=baseline,
        candidate_mean=candidate,
        absolute_delta=absolute_delta,
        relative_delta=relative_delta,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_runs(
    baseline_metric_means: dict[str, float | None],
    candidate_metric_means: dict[str, float | None],
    baseline_label_counts: dict[str, int],
    candidate_label_counts: dict[str, int],
    baseline_root_causes: dict[str, int],
    candidate_root_causes: dict[str, int],
    baseline_run_id: str,
    candidate_run_id: str,
    metrics: list[str] | None = None,
) -> RunComparison:
    """Build a :class:`RunComparison` from pre-aggregated data.

    *metrics* controls which metrics appear and in what order.
    When ``None``, all metrics from both runs are included in
    :data:`_DISPLAY_ORDER` priority order.
    """
    if metrics is None:
        all_keys = set(baseline_metric_means) | set(candidate_metric_means)
        ordered = [m for m in _DISPLAY_ORDER if m in all_keys]
        rest = sorted(m for m in all_keys if m not in _DISPLAY_ORDER)
        metrics = ordered + rest

    deltas = [
        _compute_delta(m, baseline_metric_means.get(m), candidate_metric_means.get(m))
        for m in metrics
    ]

    return RunComparison(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        metric_deltas=deltas,
        baseline_label_counts=baseline_label_counts,
        candidate_label_counts=candidate_label_counts,
        baseline_root_causes=baseline_root_causes,
        candidate_root_causes=candidate_root_causes,
    )
