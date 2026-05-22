"""Unit tests for recall@k and MRR computations — pure functions, no I/O."""
import pytest

from rageval.evaluators.retrieval_metrics import (
    RetrievalMetricResult,
    compute_mrr,
    compute_recall_at_k,
    evaluate_retrieval_for_item,
)


# ---------------------------------------------------------------------------
# compute_recall_at_k
# ---------------------------------------------------------------------------

class TestComputeRecallAtK:
    def test_source_in_top_k(self):
        assert compute_recall_at_k(["c1"], ["c1", "c2", "c3"], k=3) == 1.0

    def test_source_not_in_top_k(self):
        assert compute_recall_at_k(["c4"], ["c1", "c2", "c3"], k=3) == 0.0

    def test_source_at_boundary_included(self):
        assert compute_recall_at_k(["c3"], ["c1", "c2", "c3"], k=3) == 1.0

    def test_source_just_outside_boundary(self):
        assert compute_recall_at_k(["c4"], ["c1", "c2", "c3", "c4"], k=3) == 0.0

    def test_source_at_boundary_k4(self):
        assert compute_recall_at_k(["c4"], ["c1", "c2", "c3", "c4"], k=4) == 1.0

    def test_multiple_sources_one_in_top_k(self):
        assert compute_recall_at_k(["c1", "c9"], ["c1", "c2", "c3"], k=3) == 1.0

    def test_multiple_sources_none_in_top_k(self):
        assert compute_recall_at_k(["c8", "c9"], ["c1", "c2", "c3"], k=3) == 0.0

    def test_multiple_sources_second_in_top_k(self):
        assert compute_recall_at_k(["c8", "c2"], ["c1", "c2", "c3"], k=3) == 1.0

    def test_empty_source_chunk_ids(self):
        assert compute_recall_at_k([], ["c1", "c2"], k=3) == 0.0

    def test_empty_retrieved(self):
        assert compute_recall_at_k(["c1"], [], k=5) == 0.0

    def test_both_empty(self):
        assert compute_recall_at_k([], [], k=3) == 0.0

    def test_k_equals_one_hit(self):
        assert compute_recall_at_k(["c1"], ["c1", "c2", "c3"], k=1) == 1.0

    def test_k_equals_one_miss(self):
        assert compute_recall_at_k(["c2"], ["c1", "c2", "c3"], k=1) == 0.0

    def test_retrieved_longer_than_k(self):
        assert compute_recall_at_k(["c5"], ["c1", "c2", "c3", "c4", "c5"], k=3) == 0.0

    def test_score_is_binary(self):
        s = compute_recall_at_k(["c1", "c2"], ["c1", "c2"], k=5)
        assert s in (0.0, 1.0)


# ---------------------------------------------------------------------------
# compute_mrr
# ---------------------------------------------------------------------------

class TestComputeMRR:
    def test_rank_one(self):
        assert compute_mrr(["c1"], ["c1", "c2", "c3"]) == 1.0

    def test_rank_two(self):
        assert compute_mrr(["c2"], ["c1", "c2", "c3"]) == pytest.approx(0.5)

    def test_rank_three(self):
        assert compute_mrr(["c3"], ["c1", "c2", "c3"]) == pytest.approx(1 / 3)

    def test_rank_four(self):
        assert compute_mrr(["c4"], ["c1", "c2", "c3", "c4"]) == pytest.approx(0.25)

    def test_not_found(self):
        assert compute_mrr(["c9"], ["c1", "c2", "c3"]) == 0.0

    def test_empty_source(self):
        assert compute_mrr([], ["c1", "c2"]) == 0.0

    def test_empty_retrieved(self):
        assert compute_mrr(["c1"], []) == 0.0

    def test_multiple_sources_first_at_rank_one(self):
        assert compute_mrr(["c1", "c9"], ["c1", "c2", "c3"]) == 1.0

    def test_multiple_sources_first_found_at_rank_two(self):
        assert compute_mrr(["c9", "c2"], ["c1", "c2", "c3"]) == pytest.approx(0.5)

    def test_multiple_sources_uses_first_found(self):
        # c3 is at rank 3, c1 is at rank 1 → MRR = 1.0, not 1/3
        assert compute_mrr(["c3", "c1"], ["c1", "c2", "c3"]) == 1.0

    def test_k_excludes_later_source(self):
        assert compute_mrr(["c4"], ["c1", "c2", "c3", "c4"], k=3) == 0.0

    def test_k_includes_source(self):
        assert compute_mrr(["c4"], ["c1", "c2", "c3", "c4"], k=4) == pytest.approx(0.25)

    def test_no_k_uses_full_list(self):
        assert compute_mrr(["c5"], ["c1", "c2", "c3", "c4", "c5"]) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# evaluate_retrieval_for_item
# ---------------------------------------------------------------------------

class TestEvaluateRetrievalForItem:
    def test_returns_two_results(self):
        results = evaluate_retrieval_for_item(["c1"], ["c1", "c2"], k=2)
        assert len(results) == 2

    def test_metrics_are_recall_and_mrr(self):
        results = evaluate_retrieval_for_item(["c1"], ["c1", "c2"], k=2)
        metrics = {r.metric for r in results}
        assert metrics == {"recall_at_k", "mrr"}

    def test_returns_retrieval_metric_result_objects(self):
        results = evaluate_retrieval_for_item(["c1"], ["c1"], k=1)
        assert all(isinstance(r, RetrievalMetricResult) for r in results)

    def test_empty_source_both_unknown(self):
        results = evaluate_retrieval_for_item([], ["c1", "c2"], k=2)
        assert all(r.label == "unknown" for r in results)

    def test_empty_source_scores_zero(self):
        results = evaluate_retrieval_for_item([], ["c1"], k=1)
        assert all(r.score == 0.0 for r in results)

    def test_empty_source_reason_mentions_ground_truth(self):
        results = evaluate_retrieval_for_item([], [], k=3)
        assert all("ground truth" in r.reason for r in results)

    def test_hit_recall_pass_label(self):
        results = evaluate_retrieval_for_item(["c1"], ["c1", "c2"], k=2)
        recall = next(r for r in results if r.metric == "recall_at_k")
        assert recall.label == "pass"
        assert recall.score == 1.0

    def test_miss_recall_fail_label(self):
        results = evaluate_retrieval_for_item(["c9"], ["c1", "c2"], k=2)
        recall = next(r for r in results if r.metric == "recall_at_k")
        assert recall.label == "fail"
        assert recall.score == 0.0

    def test_hit_mrr_pass_label(self):
        results = evaluate_retrieval_for_item(["c2"], ["c1", "c2"], k=2)
        mrr = next(r for r in results if r.metric == "mrr")
        assert mrr.label == "pass"
        assert mrr.score == pytest.approx(0.5)

    def test_miss_mrr_fail_label(self):
        results = evaluate_retrieval_for_item(["c9"], ["c1", "c2"], k=2)
        mrr = next(r for r in results if r.metric == "mrr")
        assert mrr.label == "fail"
        assert mrr.score == 0.0

    def test_reason_strings_non_empty(self):
        for source, retrieved in (
            (["c1"], ["c1"]),
            (["c9"], ["c1"]),
        ):
            for r in evaluate_retrieval_for_item(source, retrieved, k=3):
                assert r.reason

    def test_recall_k_reason_mentions_k(self):
        results = evaluate_retrieval_for_item(["c9"], ["c1", "c2"], k=7)
        recall = next(r for r in results if r.metric == "recall_at_k")
        assert "7" in recall.reason

    def test_mrr_reason_mentions_rank_on_hit(self):
        results = evaluate_retrieval_for_item(["c3"], ["c1", "c2", "c3"], k=5)
        mrr = next(r for r in results if r.metric == "mrr")
        assert "3" in mrr.reason
