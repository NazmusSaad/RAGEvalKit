"""Unit tests for rageval.evaluators.compare."""
import pytest

from rageval.evaluators.compare import (
    MetricDelta,
    RunComparison,
    _compute_delta,
    compare_runs,
)

# ---------------------------------------------------------------------------
# _compute_delta
# ---------------------------------------------------------------------------

class TestComputeDelta:
    def test_improved(self):
        d = _compute_delta("recall_at_k", 0.5, 0.8)
        assert d.direction == "improved"
        assert abs(d.absolute_delta - 0.3) < 1e-9

    def test_regressed(self):
        d = _compute_delta("recall_at_k", 0.8, 0.5)
        assert d.direction == "regressed"
        assert abs(d.absolute_delta - (-0.3)) < 1e-9

    def test_unchanged(self):
        d = _compute_delta("recall_at_k", 0.5, 0.5)
        assert d.direction == "unchanged"
        assert abs(d.absolute_delta) < 1e-9

    def test_baseline_none_gives_na(self):
        d = _compute_delta("recall_at_k", None, 0.8)
        assert d.direction == "n/a"
        assert d.absolute_delta is None
        assert d.relative_delta is None

    def test_candidate_none_gives_na(self):
        d = _compute_delta("recall_at_k", 0.8, None)
        assert d.direction == "n/a"
        assert d.absolute_delta is None

    def test_both_none_gives_na(self):
        d = _compute_delta("mrr", None, None)
        assert d.direction == "n/a"

    def test_relative_delta_computed(self):
        d = _compute_delta("recall_at_k", 0.4, 0.6)
        assert d.relative_delta is not None
        assert abs(d.relative_delta - 0.5) < 1e-9

    def test_relative_delta_none_when_baseline_zero(self):
        d = _compute_delta("recall_at_k", 0.0, 0.5)
        assert d.relative_delta is None

    def test_metric_name_preserved(self):
        d = _compute_delta("faithfulness", 0.7, 0.9)
        assert d.metric == "faithfulness"

    def test_returns_metric_delta(self):
        assert isinstance(_compute_delta("mrr", 0.5, 0.6), MetricDelta)


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------

def _base_kwargs(**overrides):
    kwargs = dict(
        baseline_metric_means={"recall_at_k": 0.5, "faithfulness": 0.8},
        candidate_metric_means={"recall_at_k": 0.7, "faithfulness": 0.9},
        baseline_label_counts={"pass": 2, "fail": 1, "unknown": 0},
        candidate_label_counts={"pass": 3, "fail": 0, "unknown": 0},
        baseline_root_causes={"none": 2, "retrieval_failure": 1},
        candidate_root_causes={"none": 3},
        baseline_run_id="run-base",
        candidate_run_id="run-cand",
    )
    kwargs.update(overrides)
    return kwargs


class TestCompareRuns:
    def test_returns_run_comparison(self):
        result = compare_runs(**_base_kwargs())
        assert isinstance(result, RunComparison)

    def test_run_ids_preserved(self):
        result = compare_runs(**_base_kwargs())
        assert result.baseline_run_id == "run-base"
        assert result.candidate_run_id == "run-cand"

    def test_label_counts_preserved(self):
        result = compare_runs(**_base_kwargs())
        assert result.baseline_label_counts == {"pass": 2, "fail": 1, "unknown": 0}
        assert result.candidate_label_counts == {"pass": 3, "fail": 0, "unknown": 0}

    def test_root_causes_preserved(self):
        result = compare_runs(**_base_kwargs())
        assert result.baseline_root_causes == {"none": 2, "retrieval_failure": 1}
        assert result.candidate_root_causes == {"none": 3}

    def test_metric_delta_count(self):
        result = compare_runs(**_base_kwargs())
        assert len(result.metric_deltas) == 2  # recall_at_k, faithfulness

    def test_display_order_respected(self):
        kwargs = _base_kwargs(
            baseline_metric_means={"faithfulness": 0.8, "recall_at_k": 0.5, "mrr": 0.4},
            candidate_metric_means={"faithfulness": 0.9, "recall_at_k": 0.7, "mrr": 0.5},
        )
        result = compare_runs(**kwargs)
        names = [d.metric for d in result.metric_deltas]
        assert names.index("recall_at_k") < names.index("mrr")
        assert names.index("mrr") < names.index("faithfulness")

    def test_explicit_metrics_filter(self):
        result = compare_runs(**_base_kwargs(), metrics=["recall_at_k"])
        assert len(result.metric_deltas) == 1
        assert result.metric_deltas[0].metric == "recall_at_k"

    def test_metric_absent_in_one_run_gives_na(self):
        kwargs = _base_kwargs(
            baseline_metric_means={"recall_at_k": 0.5},
            candidate_metric_means={"recall_at_k": 0.7, "answer_relevance": 0.6},
        )
        result = compare_runs(**kwargs)
        ar_delta = next(d for d in result.metric_deltas if d.metric == "answer_relevance")
        assert ar_delta.direction == "n/a"
        assert ar_delta.baseline_mean is None

    def test_empty_means_produces_empty_deltas(self):
        kwargs = _base_kwargs(
            baseline_metric_means={},
            candidate_metric_means={},
        )
        result = compare_runs(**kwargs)
        assert result.metric_deltas == []

    def test_unlisted_metrics_appended_alphabetically(self):
        kwargs = _base_kwargs(
            baseline_metric_means={"z_metric": 0.5, "recall_at_k": 0.5, "a_metric": 0.3},
            candidate_metric_means={"z_metric": 0.6, "recall_at_k": 0.6, "a_metric": 0.4},
        )
        result = compare_runs(**kwargs)
        names = [d.metric for d in result.metric_deltas]
        # recall_at_k is in _DISPLAY_ORDER → comes first; then a_metric < z_metric alphabetically
        assert names[0] == "recall_at_k"
        assert names.index("a_metric") < names.index("z_metric")
