"""Unit tests for rageval.ci.check."""
import pytest

from rageval.ci.check import (
    CICheckResult,
    ThresholdViolation,
    _abs_min,
    _rel_max,
    run_ci_check,
)
from rageval.core.config import (
    AbsoluteThresholds,
    PolicyConfig,
    RelativeThresholds,
    ThresholdsConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _thresholds(
    *,
    recall_at_k_min=None,
    answer_relevance_min=None,
    faithfulness_min=None,
    retrieval_relevance_min=None,
    recall_at_k_drop_max=None,
    answer_relevance_drop_max=None,
    faithfulness_drop_max=None,
    mrr_drop_max=None,
    retrieval_relevance_drop_max=None,
) -> ThresholdsConfig:
    return ThresholdsConfig(
        absolute=AbsoluteThresholds(
            recall_at_k_min=recall_at_k_min,
            answer_relevance_min=answer_relevance_min,
            faithfulness_min=faithfulness_min,
            retrieval_relevance_min=retrieval_relevance_min,
        ),
        relative=RelativeThresholds(
            recall_at_k_drop_max=recall_at_k_drop_max,
            answer_relevance_drop_max=answer_relevance_drop_max,
            faithfulness_drop_max=faithfulness_drop_max,
            mrr_drop_max=mrr_drop_max,
            retrieval_relevance_drop_max=retrieval_relevance_drop_max,
        ),
    )


def _run(b_means: dict, c_means: dict, cfg: ThresholdsConfig) -> CICheckResult:
    return run_ci_check(
        baseline_metric_means=b_means,
        candidate_metric_means=c_means,
        thresholds=cfg,
        baseline_run_id="run-b",
        candidate_run_id="run-c",
        thresholds_path="rageval.yaml",
    )


# ---------------------------------------------------------------------------
# Alias helpers: _abs_min
# ---------------------------------------------------------------------------

class TestAbsMin:
    def test_recall_prefers_recall_at_k_min(self):
        abs_ = AbsoluteThresholds(recall_at_k_min=0.8, retrieval_relevance_min=0.5)
        assert _abs_min(abs_, "recall_at_k") == 0.8

    def test_recall_falls_back_to_retrieval_relevance_min(self):
        abs_ = AbsoluteThresholds(recall_at_k_min=None, retrieval_relevance_min=0.5)
        assert _abs_min(abs_, "recall_at_k") == 0.5

    def test_recall_none_when_both_none(self):
        abs_ = AbsoluteThresholds()
        assert _abs_min(abs_, "recall_at_k") is None

    def test_answer_relevance_min(self):
        abs_ = AbsoluteThresholds(answer_relevance_min=0.7)
        assert _abs_min(abs_, "answer_relevance") == 0.7

    def test_faithfulness_min(self):
        abs_ = AbsoluteThresholds(faithfulness_min=0.9)
        assert _abs_min(abs_, "faithfulness") == 0.9

    def test_unknown_metric_returns_none(self):
        abs_ = AbsoluteThresholds(faithfulness_min=0.9)
        assert _abs_min(abs_, "mrr") is None


# ---------------------------------------------------------------------------
# Alias helpers: _rel_max
# ---------------------------------------------------------------------------

class TestRelMax:
    def test_recall_prefers_recall_at_k_drop_max(self):
        rel = RelativeThresholds(recall_at_k_drop_max=0.05, retrieval_relevance_drop_max=0.10)
        assert _rel_max(rel, "recall_at_k") == 0.05

    def test_recall_falls_back_to_retrieval_relevance_drop_max(self):
        rel = RelativeThresholds(recall_at_k_drop_max=None, retrieval_relevance_drop_max=0.10)
        assert _rel_max(rel, "recall_at_k") == 0.10

    def test_recall_none_when_both_none(self):
        rel = RelativeThresholds()
        assert _rel_max(rel, "recall_at_k") is None

    def test_mrr_drop_max(self):
        rel = RelativeThresholds(mrr_drop_max=0.05)
        assert _rel_max(rel, "mrr") == 0.05

    def test_faithfulness_drop_max(self):
        rel = RelativeThresholds(faithfulness_drop_max=0.08)
        assert _rel_max(rel, "faithfulness") == 0.08

    def test_unknown_metric_returns_none(self):
        rel = RelativeThresholds(faithfulness_drop_max=0.05)
        assert _rel_max(rel, "custom_metric") is None


# ---------------------------------------------------------------------------
# run_ci_check: pass cases
# ---------------------------------------------------------------------------

class TestCICheckPass:
    def test_no_thresholds_configured_always_passes(self):
        cfg = _thresholds()
        result = _run({"faithfulness": 0.5}, {"faithfulness": 0.1}, cfg)
        assert result.passed is True
        assert result.violations == []

    def test_candidate_meets_absolute_minimum(self):
        cfg = _thresholds(faithfulness_min=0.80)
        result = _run({}, {"faithfulness": 0.85}, cfg)
        assert result.passed is True

    def test_candidate_exactly_at_absolute_minimum(self):
        cfg = _thresholds(recall_at_k_min=0.70)
        result = _run({}, {"recall_at_k": 0.70}, cfg)
        assert result.passed is True

    def test_candidate_drop_within_relative_limit(self):
        cfg = _thresholds(faithfulness_drop_max=0.10)
        result = _run({"faithfulness": 0.80}, {"faithfulness": 0.75}, cfg)
        assert result.passed is True

    def test_candidate_drop_within_limit(self):
        # drop = 0.04 < 0.05 → should pass
        cfg = _thresholds(faithfulness_drop_max=0.05)
        result = _run({"faithfulness": 0.80}, {"faithfulness": 0.76}, cfg)
        assert result.passed is True

    def test_candidate_improves_over_baseline(self):
        cfg = _thresholds(faithfulness_drop_max=0.05)
        result = _run({"faithfulness": 0.70}, {"faithfulness": 0.90}, cfg)
        assert result.passed is True

    def test_returns_ci_check_result(self):
        assert isinstance(_run({}, {}, _thresholds()), CICheckResult)

    def test_run_ids_stored(self):
        result = _run({}, {}, _thresholds())
        assert result.baseline_run_id == "run-b"
        assert result.candidate_run_id == "run-c"

    def test_thresholds_path_stored(self):
        result = run_ci_check({}, {}, _thresholds(), "b", "c", thresholds_path="my/path.yaml")
        assert result.thresholds_path == "my/path.yaml"


# ---------------------------------------------------------------------------
# run_ci_check: absolute violations
# ---------------------------------------------------------------------------

class TestAbsoluteViolations:
    def test_candidate_below_absolute_minimum(self):
        cfg = _thresholds(faithfulness_min=0.80)
        result = _run({}, {"faithfulness": 0.70}, cfg)
        assert not result.passed
        assert len(result.violations) == 1
        v = result.violations[0]
        assert v.metric == "faithfulness"
        assert v.check_type == "absolute"
        assert abs(v.threshold - 0.80) < 1e-9
        assert abs(v.actual - 0.70) < 1e-9

    def test_missing_candidate_metric_is_violation(self):
        cfg = _thresholds(recall_at_k_min=0.70)
        result = _run({"recall_at_k": 0.80}, {}, cfg)
        assert not result.passed
        v = result.violations[0]
        assert v.check_type == "absolute"
        assert v.actual is None

    def test_multiple_absolute_violations(self):
        cfg = _thresholds(faithfulness_min=0.90, answer_relevance_min=0.85)
        result = _run({}, {"faithfulness": 0.50, "answer_relevance": 0.60}, cfg)
        assert not result.passed
        assert len(result.violations) == 2

    def test_violation_message_is_non_empty(self):
        cfg = _thresholds(faithfulness_min=0.90)
        result = _run({}, {"faithfulness": 0.50}, cfg)
        assert len(result.violations[0].message) > 0

    def test_alias_retrieval_relevance_min_triggers_recall_check(self):
        # Only retrieval_relevance_min set → treated as recall_at_k alias
        cfg = _thresholds(retrieval_relevance_min=0.70)
        result = _run({}, {"recall_at_k": 0.50}, cfg)
        assert not result.passed

    def test_recall_at_k_min_preferred_over_alias(self):
        # Both set → recall_at_k_min=0.90 wins; recall=0.75 → violation
        cfg = _thresholds(recall_at_k_min=0.90, retrieval_relevance_min=0.50)
        result = _run({}, {"recall_at_k": 0.75}, cfg)
        assert not result.passed


# ---------------------------------------------------------------------------
# run_ci_check: relative violations
# ---------------------------------------------------------------------------

class TestRelativeViolations:
    def test_drop_exceeds_max_is_violation(self):
        cfg = _thresholds(faithfulness_drop_max=0.05)
        result = _run({"faithfulness": 0.90}, {"faithfulness": 0.80}, cfg)
        assert not result.passed
        v = result.violations[0]
        assert v.check_type == "relative"
        assert abs(v.actual - 0.10) < 1e-9  # drop = 0.90 - 0.80

    def test_candidate_missing_relative_is_violation_when_baseline_present(self):
        cfg = _thresholds(faithfulness_drop_max=0.05)
        result = _run({"faithfulness": 0.90}, {}, cfg)
        assert not result.passed
        v = result.violations[0]
        assert v.check_type == "relative"
        assert v.actual is None

    def test_baseline_missing_skips_relative_check(self):
        # No baseline → no relative violation even if candidate is present
        cfg = _thresholds(faithfulness_drop_max=0.05)
        result = _run({}, {"faithfulness": 0.50}, cfg)
        assert result.passed

    def test_mrr_drop_max_alias(self):
        cfg = _thresholds(mrr_drop_max=0.05)
        result = _run({"mrr": 0.80}, {"mrr": 0.70}, cfg)
        assert not result.passed
        assert result.violations[0].metric == "mrr"

    def test_alias_retrieval_relevance_drop_max_triggers_recall_check(self):
        cfg = _thresholds(retrieval_relevance_drop_max=0.05)
        result = _run({"recall_at_k": 0.80}, {"recall_at_k": 0.60}, cfg)
        assert not result.passed

    def test_recall_at_k_drop_max_preferred_over_alias(self):
        # recall_at_k_drop_max=0.30 is the effective limit (lenient) → should pass
        cfg = _thresholds(recall_at_k_drop_max=0.30, retrieval_relevance_drop_max=0.05)
        result = _run({"recall_at_k": 0.80}, {"recall_at_k": 0.60}, cfg)
        assert result.passed  # drop=0.20 < 0.30

    def test_violation_message_mentions_metric(self):
        cfg = _thresholds(faithfulness_drop_max=0.05)
        result = _run({"faithfulness": 0.90}, {"faithfulness": 0.80}, cfg)
        assert "faithfulness" in result.violations[0].message

    def test_both_absolute_and_relative_violations(self):
        cfg = _thresholds(faithfulness_min=0.90, faithfulness_drop_max=0.02)
        result = _run({"faithfulness": 0.88}, {"faithfulness": 0.80}, cfg)
        types = {v.check_type for v in result.violations}
        assert "absolute" in types
        assert "relative" in types

    def test_metric_unknown_to_check_module_skipped(self):
        # "custom_metric" has no configured threshold → no violation
        cfg = _thresholds(faithfulness_min=0.80)
        result = _run({"custom_metric": 0.50}, {"custom_metric": 0.10}, cfg)
        assert result.passed
