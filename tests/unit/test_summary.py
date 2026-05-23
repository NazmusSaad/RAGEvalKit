"""Unit tests for run summarisation and root-cause classification."""
import pytest

from rageval.evaluators.summary import (
    ItemSummary,
    RunSummary,
    build_run_summary,
    classify_root_cause,
    mean_metric,
    summarize_item,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics(**kw) -> dict:
    """Build a metrics dict from keyword args like recall_at_k=("pass", 1.0)."""
    return {k: {"label": v[0], "score": v[1]} for k, v in kw.items()}


def _all_pass() -> dict:
    return _metrics(
        recall_at_k=("pass", 1.0),
        answer_relevance=("pass", 0.75),
        faithfulness=("pass", 1.0),
        mrr=("pass", 0.5),
    )


def _with_recall_zero() -> dict:
    return _metrics(
        recall_at_k=("fail", 0.0),
        answer_relevance=("pass", 0.75),
        faithfulness=("pass", 1.0),
    )


def _with_faith_fail() -> dict:
    return _metrics(
        recall_at_k=("pass", 1.0),
        answer_relevance=("pass", 0.75),
        faithfulness=("fail", 0.5),
    )


def _with_ar_fail() -> dict:
    return _metrics(
        recall_at_k=("pass", 1.0),
        answer_relevance=("fail", 0.5),
        faithfulness=("pass", 1.0),
    )


# ---------------------------------------------------------------------------
# overall_label (via summarize_item)
# ---------------------------------------------------------------------------

class TestOverallLabel:
    def test_all_pass_gives_pass(self):
        s = summarize_item("i1", "q1", _all_pass())
        assert s.overall_label == "pass"

    def test_missing_recall_gives_unknown(self):
        metrics = _metrics(answer_relevance=("pass", 0.75), faithfulness=("pass", 1.0))
        s = summarize_item("i1", "q1", metrics)
        assert s.overall_label == "unknown"
        assert "recall_at_k" in s.reason

    def test_missing_answer_relevance_gives_unknown(self):
        metrics = _metrics(recall_at_k=("pass", 1.0), faithfulness=("pass", 1.0))
        s = summarize_item("i1", "q1", metrics)
        assert s.overall_label == "unknown"
        assert "answer_relevance" in s.reason

    def test_missing_faithfulness_gives_unknown(self):
        metrics = _metrics(recall_at_k=("pass", 1.0), answer_relevance=("pass", 0.75))
        s = summarize_item("i1", "q1", metrics)
        assert s.overall_label == "unknown"
        assert "faithfulness" in s.reason

    def test_recall_unknown_label_gives_unknown(self):
        metrics = _metrics(
            recall_at_k=("unknown", 0.0),
            answer_relevance=("pass", 0.75),
            faithfulness=("pass", 1.0),
        )
        s = summarize_item("i1", "q1", metrics)
        assert s.overall_label == "unknown"

    def test_recall_fail_gives_fail(self):
        s = summarize_item("i1", "q1", _with_recall_zero())
        assert s.overall_label == "fail"

    def test_faithfulness_fail_gives_fail(self):
        s = summarize_item("i1", "q1", _with_faith_fail())
        assert s.overall_label == "fail"

    def test_answer_relevance_fail_gives_fail(self):
        s = summarize_item("i1", "q1", _with_ar_fail())
        assert s.overall_label == "fail"

    def test_missing_required_takes_priority_over_fail(self):
        """A missing metric → unknown even if another metric fails."""
        metrics = _metrics(answer_relevance=("fail", 0.25), faithfulness=("pass", 1.0))
        # recall_at_k is missing
        s = summarize_item("i1", "q1", metrics)
        assert s.overall_label == "unknown"

    def test_mrr_missing_does_not_affect_overall(self):
        """MRR is optional; omitting it should not cause unknown."""
        metrics = _metrics(
            recall_at_k=("pass", 1.0),
            answer_relevance=("pass", 0.75),
            faithfulness=("pass", 1.0),
        )
        s = summarize_item("i1", "q1", metrics)
        assert s.overall_label == "pass"


# ---------------------------------------------------------------------------
# classify_root_cause
# ---------------------------------------------------------------------------

class TestClassifyRootCause:
    def test_all_pass_primary_is_none(self):
        primary, secondary = classify_root_cause(_all_pass())
        assert primary == "none"
        assert secondary == []

    def test_missing_recall_primary_missing_metric(self):
        metrics = _metrics(answer_relevance=("pass", 0.75), faithfulness=("pass", 1.0))
        primary, _ = classify_root_cause(metrics)
        assert primary == "missing_metric"

    def test_recall_unknown_primary_judge_uncertain(self):
        metrics = _metrics(
            recall_at_k=("unknown", 0.0),
            answer_relevance=("pass", 0.75),
            faithfulness=("pass", 1.0),
        )
        primary, _ = classify_root_cause(metrics)
        assert primary == "judge_uncertain"

    def test_recall_zero_primary_retrieval_failure(self):
        primary, _ = classify_root_cause(_with_recall_zero())
        assert primary == "retrieval_failure"

    def test_faith_fail_primary_grounding_failure(self):
        primary, _ = classify_root_cause(_with_faith_fail())
        assert primary == "grounding_failure"

    def test_ar_fail_primary_answer_relevance_failure(self):
        primary, _ = classify_root_cause(_with_ar_fail())
        assert primary == "answer_relevance_failure"

    def test_retrieval_failure_secondary_includes_grounding(self):
        """recall@k=0 (retrieval) and faithfulness fails → secondary grounding."""
        metrics = _metrics(
            recall_at_k=("fail", 0.0),
            answer_relevance=("pass", 0.75),
            faithfulness=("fail", 0.4),
        )
        primary, secondary = classify_root_cause(metrics)
        assert primary == "retrieval_failure"
        assert "grounding_failure" in secondary

    def test_grounding_failure_secondary_includes_ar_failure(self):
        metrics = _metrics(
            recall_at_k=("pass", 1.0),
            answer_relevance=("fail", 0.4),
            faithfulness=("fail", 0.4),
        )
        primary, secondary = classify_root_cause(metrics)
        assert primary == "grounding_failure"
        assert "answer_relevance_failure" in secondary

    def test_primary_not_in_secondary(self):
        primary, secondary = classify_root_cause(_with_recall_zero())
        assert primary not in secondary

    def test_secondary_has_no_duplicates(self):
        metrics = _metrics(
            recall_at_k=("fail", 0.0),
            answer_relevance=("fail", 0.4),
            faithfulness=("fail", 0.4),
        )
        _, secondary = classify_root_cause(metrics)
        assert len(secondary) == len(set(secondary))


# ---------------------------------------------------------------------------
# summarize_item — suggested_fix and structure
# ---------------------------------------------------------------------------

class TestSummarizeItem:
    def test_returns_item_summary(self):
        assert isinstance(summarize_item("i1", "q1", _all_pass()), ItemSummary)

    def test_pass_has_none_primary_cause(self):
        s = summarize_item("i1", "q1", _all_pass())
        assert s.primary_cause == "none"
        assert s.secondary_causes == []

    def test_suggested_fix_populated_for_retrieval_failure(self):
        s = summarize_item("i1", "q1", _with_recall_zero())
        assert len(s.suggested_fix) > 0
        assert "retrieval" in s.suggested_fix.lower()

    def test_suggested_fix_empty_for_none(self):
        s = summarize_item("i1", "q1", _all_pass())
        assert s.suggested_fix == ""

    def test_item_id_and_question_id_stored(self):
        s = summarize_item("item123", "q456", _all_pass())
        assert s.item_id == "item123"
        assert s.question_id == "q456"

    def test_metrics_stored(self):
        metrics = _all_pass()
        s = summarize_item("i1", "q1", metrics)
        assert s.metrics == metrics


# ---------------------------------------------------------------------------
# build_run_summary
# ---------------------------------------------------------------------------

class TestBuildRunSummary:
    def _items_data(self):
        return [
            {"item_id": "i1", "question_id": "q1"},
            {"item_id": "i2", "question_id": "q2"},
        ]

    def _scores_data(self):
        rows = []
        for iid in ("i1", "i2"):
            rows += [
                {"item_id": iid, "metric": "recall_at_k", "score": 1.0, "label": "pass"},
                {"item_id": iid, "metric": "answer_relevance", "score": 0.75, "label": "pass"},
                {"item_id": iid, "metric": "faithfulness", "score": 1.0, "label": "pass"},
            ]
        return rows

    def test_returns_run_summary(self):
        rs = build_run_summary("run1", self._items_data(), self._scores_data())
        assert isinstance(rs, RunSummary)

    def test_item_count(self):
        rs = build_run_summary("run1", self._items_data(), self._scores_data())
        assert len(rs.items) == 2

    def test_all_pass_counts(self):
        rs = build_run_summary("run1", self._items_data(), self._scores_data())
        assert rs.pass_count == 2
        assert rs.fail_count == 0
        assert rs.unknown_count == 0

    def test_empty_scores_gives_all_unknown(self):
        rs = build_run_summary("run1", self._items_data(), [])
        assert rs.unknown_count == 2

    def test_run_id_stored(self):
        rs = build_run_summary("myrun", self._items_data(), self._scores_data())
        assert rs.run_id == "myrun"


# ---------------------------------------------------------------------------
# mean_metric
# ---------------------------------------------------------------------------

class TestMeanMetric:
    def _rs(self, scores_by_item):
        items_data = [{"item_id": k, "question_id": ""} for k in scores_by_item]
        scores_data = []
        for iid, metrics in scores_by_item.items():
            for m, (label, score) in metrics.items():
                scores_data.append({"item_id": iid, "metric": m, "score": score, "label": label})
        return build_run_summary("r", items_data, scores_data)

    def test_mean_over_passing_items(self):
        rs = self._rs({"i1": {"recall_at_k": ("pass", 1.0)}, "i2": {"recall_at_k": ("pass", 0.0)}})
        assert mean_metric(rs, "recall_at_k") == "0.500"

    def test_excludes_unknown_items(self):
        rs = self._rs({"i1": {"recall_at_k": ("pass", 1.0)}, "i2": {"recall_at_k": ("unknown", 0.0)}})
        assert mean_metric(rs, "recall_at_k") == "1.000"

    def test_all_unknown_returns_na(self):
        rs = self._rs({"i1": {"recall_at_k": ("unknown", 0.0)}})
        assert mean_metric(rs, "recall_at_k") == "N/A"

    def test_metric_absent_returns_na(self):
        rs = self._rs({"i1": {"faithfulness": ("pass", 1.0)}})
        assert mean_metric(rs, "recall_at_k") == "N/A"
