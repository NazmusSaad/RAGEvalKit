"""CI threshold checking module.

No LLM calls.  Operates entirely on pre-aggregated metric means.

Absolute check  — candidate_mean >= threshold_min
                  violation if candidate_mean < min OR candidate_mean is None

Relative check  — drop = baseline_mean - candidate_mean
                  violation if drop > drop_max
                  skip if baseline_mean is None (no baseline to compare)
                  violation if candidate_mean is None and baseline_mean is not None
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rageval.core.config import (
    AbsoluteThresholds,
    RelativeThresholds,
    ThresholdsConfig,
)


@dataclass
class ThresholdViolation:
    metric: str
    check_type: str       # "absolute" | "relative"
    threshold: float
    actual: float | None  # None when metric is missing from the run
    message: str


@dataclass
class CICheckResult:
    passed: bool
    violations: list[ThresholdViolation]
    baseline_run_id: str
    candidate_run_id: str
    thresholds_path: str


# ---------------------------------------------------------------------------
# Alias resolution helpers
# ---------------------------------------------------------------------------

def _abs_min(absolute: AbsoluteThresholds, metric: str) -> float | None:
    """Effective absolute minimum for *metric*, honouring legacy aliases."""
    if metric == "recall_at_k":
        return absolute.recall_at_k_min if absolute.recall_at_k_min is not None else absolute.retrieval_relevance_min
    if metric == "answer_relevance":
        return absolute.answer_relevance_min
    if metric == "faithfulness":
        return absolute.faithfulness_min
    return None


def _rel_max(relative: RelativeThresholds, metric: str) -> float | None:
    """Effective drop maximum for *metric*, honouring legacy aliases."""
    if metric == "recall_at_k":
        return relative.recall_at_k_drop_max if relative.recall_at_k_drop_max is not None else relative.retrieval_relevance_drop_max
    if metric == "answer_relevance":
        return relative.answer_relevance_drop_max
    if metric == "faithfulness":
        return relative.faithfulness_drop_max
    if metric == "mrr":
        return relative.mrr_drop_max
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ci_check(
    baseline_metric_means: dict[str, float | None],
    candidate_metric_means: dict[str, float | None],
    thresholds: ThresholdsConfig,
    baseline_run_id: str,
    candidate_run_id: str,
    thresholds_path: str = "",
) -> CICheckResult:
    """Evaluate all configured thresholds and return a :class:`CICheckResult`."""
    violations: list[ThresholdViolation] = []

    all_metrics = sorted(set(baseline_metric_means) | set(candidate_metric_means))

    for metric in all_metrics:
        baseline_mean = baseline_metric_means.get(metric)
        candidate_mean = candidate_metric_means.get(metric)

        # --- absolute ---
        min_val = _abs_min(thresholds.absolute, metric)
        if min_val is not None:
            if candidate_mean is None:
                violations.append(ThresholdViolation(
                    metric=metric,
                    check_type="absolute",
                    threshold=min_val,
                    actual=None,
                    message=f"{metric}: missing from candidate (required >= {min_val:.3f})",
                ))
            elif candidate_mean < min_val:
                violations.append(ThresholdViolation(
                    metric=metric,
                    check_type="absolute",
                    threshold=min_val,
                    actual=candidate_mean,
                    message=f"{metric}: {candidate_mean:.3f} < {min_val:.3f} (absolute minimum)",
                ))

        # --- relative ---
        max_drop = _rel_max(thresholds.relative, metric)
        if max_drop is not None:
            if baseline_mean is None:
                pass  # no baseline → skip relative check
            elif candidate_mean is None:
                violations.append(ThresholdViolation(
                    metric=metric,
                    check_type="relative",
                    threshold=max_drop,
                    actual=None,
                    message=f"{metric}: missing from candidate (baseline was {baseline_mean:.3f})",
                ))
            else:
                drop = baseline_mean - candidate_mean
                if drop > max_drop:
                    violations.append(ThresholdViolation(
                        metric=metric,
                        check_type="relative",
                        threshold=max_drop,
                        actual=drop,
                        message=f"{metric}: dropped {drop:.3f} > max allowed {max_drop:.3f}",
                    ))

    return CICheckResult(
        passed=len(violations) == 0,
        violations=violations,
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        thresholds_path=thresholds_path,
    )
