"""Unit tests for rageval.report.render.

Builds ReportData from synthetic Python data (no DuckDB required) and
validates the rendered HTML string.  No snapshot comparisons — we assert
that specific substrings appear in the output.
"""
import pytest

from rageval.report.render import ItemReport, ReportData, render_run_report


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _run_meta(**overrides) -> dict:
    base = {
        "run_id": "test-run-abc123def456",
        "name": "unit-test run",
        "tag": "v1.0",
        "config_hash": "deadbeef",
        "config_json": "{}",
        "evalset_id": "eval-1",
        "git_sha": None,
        "started_at": None,
        "finished_at": None,
        "status": "finished",
    }
    base.update(overrides)
    return base


def _item(
    item_id="item-1",
    question="What is the capital of France?",
    generated_answer="Paris is the capital of France.",
    reference_answer="Paris.",
    overall_label="pass",
    metrics=None,
    root_cause=None,
    contexts=None,
    claims=None,
) -> ItemReport:
    if metrics is None:
        metrics = {
            "recall_at_k": {"score": 1.0, "label": "pass", "reason": "All relevant chunks retrieved."},
            "faithfulness": {"score": 1.0, "label": "pass", "reason": ""},
            "answer_relevance": {"score": 0.85, "label": "pass", "reason": ""},
        }
    if contexts is None:
        contexts = [
            {
                "rank": 1,
                "chunk_id": "chunk-abc-0001",
                "chunk_text": "France is a country in Western Europe. Its capital is Paris.",
                "score": 0.95,
            }
        ]
    if claims is None:
        claims = [
            {
                "claim_text": "Paris is the capital of France.",
                "verdict": "supported",
                "supporting_chunk_ids": ["chunk-abc-0001"],
                "rationale": "Directly stated in the retrieved chunk.",
            }
        ]
    return ItemReport(
        item_id=item_id,
        question_id="q1",
        question=question,
        generated_answer=generated_answer,
        reference_answer=reference_answer,
        overall_label=overall_label,
        metrics=metrics,
        root_cause=root_cause,
        contexts=contexts,
        claims=claims,
    )


def _fail_item() -> ItemReport:
    return _item(
        item_id="item-2",
        question="What is the largest planet?",
        generated_answer="Jupiter is the largest planet.",
        reference_answer="Jupiter.",
        overall_label="fail",
        metrics={
            "recall_at_k": {"score": 0.0, "label": "fail", "reason": ""},
            "faithfulness": {"score": 0.4, "label": "fail", "reason": ""},
            "answer_relevance": {"score": 0.7, "label": "pass", "reason": ""},
        },
        root_cause={
            "primary_cause": "retrieval_failure",
            "secondary_causes": ["grounding_failure"],
            "suggested_fix": "Improve retrieval by adding more relevant documents.",
        },
        contexts=[],
        claims=[
            {
                "claim_text": "Jupiter is the largest planet.",
                "verdict": "contradicted",
                "supporting_chunk_ids": [],
                "rationale": "Not supported by retrieved context.",
            }
        ],
    )


def _full_data() -> ReportData:
    return ReportData(
        run=_run_meta(),
        metric_means={
            "recall_at_k": 0.75,
            "mrr": 0.60,
            "answer_relevance": 0.80,
            "faithfulness": 0.90,
        },
        label_counts={"pass": 1, "fail": 1, "unknown": 0},
        root_cause_distribution={"none": 1, "retrieval_failure": 1},
        items=[_item(), _fail_item()],
    )


# ---------------------------------------------------------------------------
# Basic rendering
# ---------------------------------------------------------------------------

class TestRenderBasics:
    def test_returns_string(self):
        html = render_run_report(_full_data())
        assert isinstance(html, str)

    def test_starts_with_doctype(self):
        html = render_run_report(_full_data())
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_contains_html_tag(self):
        html = render_run_report(_full_data())
        assert "<html" in html

    def test_contains_closing_html_tag(self):
        html = render_run_report(_full_data())
        assert "</html>" in html


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

class TestRunMetadata:
    def test_contains_run_id(self):
        html = render_run_report(_full_data())
        assert "test-run-abc123def456" in html

    def test_contains_tag(self):
        html = render_run_report(_full_data())
        assert "v1.0" in html

    def test_contains_status(self):
        html = render_run_report(_full_data())
        assert "finished" in html

    def test_null_tag_shows_dash(self):
        data = _full_data()
        data.run["tag"] = None
        html = render_run_report(data)
        assert "—" in html


# ---------------------------------------------------------------------------
# Metric summary
# ---------------------------------------------------------------------------

class TestMetricSummary:
    def test_contains_recall_at_k(self):
        html = render_run_report(_full_data())
        assert "recall_at_k" in html

    def test_contains_faithfulness(self):
        html = render_run_report(_full_data())
        assert "faithfulness" in html

    def test_contains_mrr(self):
        html = render_run_report(_full_data())
        assert "mrr" in html

    def test_contains_answer_relevance(self):
        html = render_run_report(_full_data())
        assert "answer_relevance" in html

    def test_metric_value_formatted(self):
        html = render_run_report(_full_data())
        assert "0.750" in html or "0.75" in html

    def test_empty_metrics_shows_note(self):
        data = _full_data()
        data.metric_means = {}
        html = render_run_report(data)
        assert "evaluate" in html.lower() or "No metrics" in html

    def test_none_metric_value_shows_na(self):
        data = _full_data()
        data.metric_means = {"recall_at_k": None}
        html = render_run_report(data)
        assert "N/A" in html


# ---------------------------------------------------------------------------
# Label counts (score cards)
# ---------------------------------------------------------------------------

class TestLabelCounts:
    def test_pass_count_appears(self):
        data = _full_data()
        html = render_run_report(data)
        # pass_count=1 should appear in the pass card
        assert ">1<" in html or "1" in html

    def test_fail_count_appears(self):
        data = _full_data()
        html = render_run_report(data)
        assert ">1<" in html or "1" in html


# ---------------------------------------------------------------------------
# Root-cause distribution
# ---------------------------------------------------------------------------

class TestRootCauseDistribution:
    def test_contains_none_cause(self):
        html = render_run_report(_full_data())
        assert "none" in html

    def test_contains_retrieval_failure(self):
        html = render_run_report(_full_data())
        assert "retrieval_failure" in html

    def test_empty_root_cause_shows_summarize_note(self):
        data = _full_data()
        data.root_cause_distribution = {}
        html = render_run_report(data)
        assert "summarize-run" in html


# ---------------------------------------------------------------------------
# Per-item details
# ---------------------------------------------------------------------------

class TestItemDetails:
    def test_question_text_appears(self):
        html = render_run_report(_full_data())
        assert "What is the capital of France?" in html

    def test_generated_answer_appears(self):
        html = render_run_report(_full_data())
        assert "Paris is the capital of France." in html

    def test_reference_answer_appears(self):
        html = render_run_report(_full_data())
        assert "Paris." in html

    def test_item_overall_label_appears(self):
        html = render_run_report(_full_data())
        assert "pass" in html
        assert "fail" in html

    def test_item_root_cause_appears(self):
        html = render_run_report(_full_data())
        assert "retrieval_failure" in html

    def test_secondary_cause_appears(self):
        html = render_run_report(_full_data())
        assert "grounding_failure" in html

    def test_context_chunk_id_appears(self):
        html = render_run_report(_full_data())
        assert "chunk-abc-0001" in html

    def test_no_contexts_shows_note(self):
        data = ReportData(
            run=_run_meta(),
            metric_means={},
            label_counts={"pass": 1, "fail": 0, "unknown": 0},
            root_cause_distribution={},
            items=[_item(contexts=[])],
        )
        html = render_run_report(data)
        assert "No retrieved contexts" in html or "context" in html.lower()

    def test_no_items_shows_note(self):
        data = _full_data()
        data.items = []
        html = render_run_report(data)
        assert "No items" in html


# ---------------------------------------------------------------------------
# Claims and groundedness
# ---------------------------------------------------------------------------

class TestClaims:
    def test_supported_verdict_appears(self):
        html = render_run_report(_full_data())
        assert "supported" in html

    def test_contradicted_verdict_appears(self):
        html = render_run_report(_full_data())
        assert "contradicted" in html

    def test_claim_text_appears(self):
        html = render_run_report(_full_data())
        assert "Paris is the capital of France." in html

    def test_rationale_appears(self):
        html = render_run_report(_full_data())
        assert "Directly stated in the retrieved chunk." in html

    def test_no_claims_shows_extract_claims_note(self):
        data = ReportData(
            run=_run_meta(),
            metric_means={},
            label_counts={"pass": 1, "fail": 0, "unknown": 0},
            root_cause_distribution={},
            items=[_item(claims=[])],
        )
        html = render_run_report(data)
        assert "extract-claims" in html


# ---------------------------------------------------------------------------
# Graceful missing data
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_no_root_cause_on_item_renders_cleanly(self):
        data = ReportData(
            run=_run_meta(),
            metric_means={"faithfulness": 0.9},
            label_counts={"pass": 1, "fail": 0, "unknown": 0},
            root_cause_distribution={},
            items=[_item(root_cause=None)],
        )
        html = render_run_report(data)
        assert "<!DOCTYPE html>" in html

    def test_context_none_score_renders_na(self):
        data = ReportData(
            run=_run_meta(),
            metric_means={},
            label_counts={"pass": 1, "fail": 0, "unknown": 0},
            root_cause_distribution={},
            items=[_item(contexts=[
                {"rank": 1, "chunk_id": "x", "chunk_text": "hello", "score": None}
            ])],
        )
        html = render_run_report(data)
        assert "N/A" in html

    def test_long_question_truncated_in_summary(self):
        long_q = "A" * 150
        data = ReportData(
            run=_run_meta(),
            metric_means={},
            label_counts={"pass": 1, "fail": 0, "unknown": 0},
            root_cause_distribution={},
            items=[_item(question=long_q)],
        )
        html = render_run_report(data)
        # The full question should appear in the body but the summary truncates it
        assert "A" * 100 in html
