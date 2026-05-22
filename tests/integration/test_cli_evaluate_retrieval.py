"""Integration tests for `rageval evaluate-retrieval`.

Uses DummyEmbedder and MockLLMClient — no real API calls.
All storage under tmp_path.
"""
import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rageval.cli.main import app
from rageval.storage.duckdb_dao import get_connection, get_metric_scores_for_run

runner = CliRunner()
TINY_CORPUS = Path(__file__).parents[2] / "examples" / "tiny-corpus"

_CONFIG = textwrap.dedent("""\
    version: 1
    name: test_config
    corpus:
      path: ./docs
    embedding:
      provider: dummy
      model: dummy
      batch_size: 16
    vector_store:
      path: .rageval/chroma
      collection: test_col
    generation:
      provider: mock
      model: mock-model
      temperature: 0.0
      max_tokens: 100
    judge:
      provider: mock
      model: mock
    evalset:
      path: evalsets/test.jsonl
""")

_EMPTY_SOURCE_QUESTIONS = [
    {
        "question_id": f"q{i}",
        "question": f"Question {i}?",
        "reference_answer": f"Answer {i}.",
        "source_chunk_ids": [],  # no ground truth
        "difficulty": "easy",
        "qtype": "factoid",
    }
    for i in range(3)
]


def _write_evalset(path: Path, questions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(q) for q in questions) + "\n", encoding="utf-8"
    )


def _last_run_id(db_path: Path) -> str:
    con = get_connection(db_path)
    try:
        row = con.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else ""


def _do_run(evalset_arg: str = "evalsets/test.jsonl") -> None:
    runner.invoke(
        app,
        ["run", "--config", "configs/test.yaml", "--evalset", evalset_arg],
        catch_exceptions=False,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def run_project(tmp_path, monkeypatch):
    """Project with init + ingest + run; evalset has empty source_chunk_ids."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"], catch_exceptions=False)
    (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)
    runner.invoke(
        app,
        ["ingest", str(TINY_CORPUS), "--config", "configs/test.yaml"],
        catch_exceptions=False,
    )
    _write_evalset(tmp_path / "evalsets" / "test.jsonl", _EMPTY_SOURCE_QUESTIONS)
    _do_run()
    run_id = _last_run_id(tmp_path / ".rageval" / "runs.db")
    return tmp_path, run_id


@pytest.fixture
def run_project_with_sources(tmp_path, monkeypatch):
    """Project where question text = chunk text → DummyEmbedder returns rank-1 retrieval."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"], catch_exceptions=False)
    (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)
    runner.invoke(
        app,
        ["ingest", str(TINY_CORPUS), "--config", "configs/test.yaml"],
        catch_exceptions=False,
    )

    # Get real chunk IDs + texts from DuckDB after ingestion.
    con = get_connection(tmp_path / ".rageval" / "runs.db")
    try:
        rows = con.execute("SELECT chunk_id, text FROM chunks LIMIT 3").fetchall()
    finally:
        con.close()

    # question text == chunk text → sha256 embedding == stored embedding → rank 0
    sourced_questions = [
        {
            "question_id": f"sq{i}",
            "question": text,            # exact match guarantees recall=1, MRR=1
            "reference_answer": "See context.",
            "source_chunk_ids": [chunk_id],
            "difficulty": "easy",
            "qtype": "factoid",
        }
        for i, (chunk_id, text) in enumerate(rows)
    ]
    _write_evalset(tmp_path / "evalsets" / "sourced.jsonl", sourced_questions)
    _do_run("evalsets/sourced.jsonl")

    run_id = _last_run_id(tmp_path / ".rageval" / "runs.db")
    return tmp_path, run_id


# ---------------------------------------------------------------------------
# Happy path — empty source_chunk_ids
# ---------------------------------------------------------------------------

class TestEvaluateRetrievalHappyPath:
    def test_exit_code_zero(self, run_project):
        _, run_id = run_project
        result = runner.invoke(
            app, ["evaluate-retrieval", "--run", run_id], catch_exceptions=False
        )
        assert result.exit_code == 0

    def test_creates_two_metric_rows_per_item(self, run_project):
        project, run_id = run_project
        runner.invoke(
            app, ["evaluate-retrieval", "--run", run_id], catch_exceptions=False
        )
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        assert len(scores) == 6  # 2 metrics × 3 questions

    def test_both_metrics_stored(self, run_project):
        project, run_id = run_project
        runner.invoke(
            app, ["evaluate-retrieval", "--run", run_id], catch_exceptions=False
        )
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        metric_names = {s["metric"] for s in scores}
        assert "recall_at_k" in metric_names
        assert "mrr" in metric_names

    def test_all_labels_unknown_for_empty_sources(self, run_project):
        project, run_id = run_project
        runner.invoke(
            app, ["evaluate-retrieval", "--run", run_id], catch_exceptions=False
        )
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        assert all(s["label"] == "unknown" for s in scores)

    def test_output_mentions_run_id(self, run_project):
        _, run_id = run_project
        result = runner.invoke(
            app, ["evaluate-retrieval", "--run", run_id], catch_exceptions=False
        )
        assert run_id[:12] in result.output or "Run ID" in result.output

    def test_output_shows_unknown_count(self, run_project):
        _, run_id = run_project
        result = runner.invoke(
            app, ["evaluate-retrieval", "--run", run_id], catch_exceptions=False
        )
        assert "3" in result.output  # 3 unknowns

    def test_output_shows_na_for_all_unknowns(self, run_project):
        _, run_id = run_project
        result = runner.invoke(
            app, ["evaluate-retrieval", "--run", run_id], catch_exceptions=False
        )
        assert "N/A" in result.output

    def test_custom_k_reflected_in_output(self, run_project):
        _, run_id = run_project
        result = runner.invoke(
            app,
            ["evaluate-retrieval", "--run", run_id, "--k", "3"],
            catch_exceptions=False,
        )
        assert "3" in result.output


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_re_run_same_row_count(self, run_project):
        project, run_id = run_project
        for _ in range(2):
            runner.invoke(
                app, ["evaluate-retrieval", "--run", run_id], catch_exceptions=False
            )
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        assert len(scores) == 6  # not 12 after two runs


# ---------------------------------------------------------------------------
# Real source_chunk_ids → non-zero scores
# ---------------------------------------------------------------------------

class TestEvaluateRetrievalWithSources:
    def test_scores_are_nonzero(self, run_project_with_sources):
        project, run_id = run_project_with_sources
        runner.invoke(
            app, ["evaluate-retrieval", "--run", run_id], catch_exceptions=False
        )
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        recall_scores = [s["score"] for s in scores if s["metric"] == "recall_at_k"]
        assert any(s > 0.0 for s in recall_scores)

    def test_pass_label_on_hit(self, run_project_with_sources):
        project, run_id = run_project_with_sources
        runner.invoke(
            app, ["evaluate-retrieval", "--run", run_id], catch_exceptions=False
        )
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            scores = get_metric_scores_for_run(con, run_id)
        finally:
            con.close()
        passing = [s for s in scores if s["metric"] == "recall_at_k" and s["score"] > 0]
        assert all(s["label"] == "pass" for s in passing)

    def test_output_shows_numeric_mean(self, run_project_with_sources):
        _, run_id = run_project_with_sources
        result = runner.invoke(
            app, ["evaluate-retrieval", "--run", run_id], catch_exceptions=False
        )
        # At least one hit → mean is numeric, not "N/A"
        assert "N/A" not in result.output


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestEvaluateRetrievalErrors:
    def test_fails_without_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["evaluate-retrieval", "--run", "fake-id"])
        assert result.exit_code != 0
        assert "init" in result.output.lower()

    def test_fails_with_unknown_run_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"], catch_exceptions=False)
        result = runner.invoke(app, ["evaluate-retrieval", "--run", "no-such-run"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()
