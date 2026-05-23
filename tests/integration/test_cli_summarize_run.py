"""Integration tests for `rageval summarize-run`.

Runs the full mock/dev evaluation pipeline:
  init → ingest → generate-evalset → run
  → evaluate-retrieval → evaluate-answer-relevance
  → extract-claims → evaluate-groundedness
  → summarize-run

No real API calls; all evaluators use mock/dummy providers.
"""
import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rageval.cli.main import app
from rageval.storage.duckdb_dao import (
    get_connection,
    get_root_causes_for_run,
    get_run_by_id,
)

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
      model: mock-judge
    evalset:
      path: evalsets/auto.jsonl
""")

_QUESTIONS = [
    {
        "question_id": f"q{i}",
        "question": f"Question {i}?",
        "reference_answer": f"Answer {i}.",
        "source_chunk_ids": [],
        "difficulty": "easy",
        "qtype": "factoid",
    }
    for i in range(3)
]


def _write_evalset(path: Path, questions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(q) for q in questions) + "\n")


def _last_run_id(db_path: Path) -> str:
    con = get_connection(db_path)
    try:
        row = con.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    return row[0] if row else ""


# ---------------------------------------------------------------------------
# Fixture: full pipeline
# ---------------------------------------------------------------------------

@pytest.fixture
def full_pipeline_project(tmp_path, monkeypatch):
    """Runs the full mock evaluation pipeline; returns (tmp_path, run_id)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "test.yaml").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)

    def _invoke(*args):
        result = runner.invoke(app, list(args), catch_exceptions=False)
        assert result.exit_code == 0, f"Command {args} failed:\n{result.output}"
        return result

    _invoke("init")
    _invoke("ingest", str(TINY_CORPUS), "--config", "configs/test.yaml")

    # Write evalset directly (faster than generate-evalset for fixture setup)
    _write_evalset(tmp_path / "evalsets" / "auto.jsonl", _QUESTIONS)

    _invoke("run", "--config", "configs/test.yaml", "--evalset", "evalsets/auto.jsonl")
    run_id = _last_run_id(tmp_path / ".rageval" / "runs.db")

    # Evaluators
    _invoke("evaluate-retrieval", "--run", run_id)
    _invoke("evaluate-answer-relevance", "--run", run_id, "--config", "configs/test.yaml")
    _invoke("extract-claims", "--run", run_id, "--config", "configs/test.yaml")
    _invoke("evaluate-groundedness", "--run", run_id, "--config", "configs/test.yaml")

    return tmp_path, run_id


def _invoke_summarize(run_id: str) -> object:
    return runner.invoke(
        app, ["summarize-run", "--run", run_id], catch_exceptions=False
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestSummarizeRunHappyPath:
    def test_exit_code_zero(self, full_pipeline_project):
        _, run_id = full_pipeline_project
        assert _invoke_summarize(run_id).exit_code == 0

    def test_output_mentions_items_summarized(self, full_pipeline_project):
        _, run_id = full_pipeline_project
        result = _invoke_summarize(run_id)
        assert "Items summarized" in result.output or "3" in result.output

    def test_stores_root_cause_rows(self, full_pipeline_project):
        project, run_id = full_pipeline_project
        _invoke_summarize(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            causes = get_root_causes_for_run(con, run_id)
        finally:
            con.close()
        assert len(causes) == 3  # one per run item

    def test_root_cause_rows_have_required_fields(self, full_pipeline_project):
        project, run_id = full_pipeline_project
        _invoke_summarize(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            causes = get_root_causes_for_run(con, run_id)
        finally:
            con.close()
        for c in causes:
            assert "primary_cause" in c
            assert "secondary_causes" in c
            assert isinstance(c["secondary_causes"], list)

    def test_output_shows_root_cause_distribution(self, full_pipeline_project):
        _, run_id = full_pipeline_project
        result = _invoke_summarize(run_id)
        # distribution table should always mention all known causes
        assert "none" in result.output or "judge_uncertain" in result.output

    def test_output_shows_mean_metrics(self, full_pipeline_project):
        _, run_id = full_pipeline_project
        result = _invoke_summarize(run_id)
        # At least faithfulness and answer_relevance should appear (non-N/A from mock)
        assert "Mean" in result.output or "mean" in result.output.lower()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_re_run_same_root_cause_count(self, full_pipeline_project):
        project, run_id = full_pipeline_project
        _invoke_summarize(run_id)
        _invoke_summarize(run_id)
        con = get_connection(project / ".rageval" / "runs.db")
        try:
            causes = get_root_causes_for_run(con, run_id)
        finally:
            con.close()
        assert len(causes) == 3  # not 6 after two runs


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestSummarizeRunErrors:
    def test_fails_without_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["summarize-run", "--run", "fake-id"])
        assert result.exit_code != 0
        assert "init" in result.output.lower()

    def test_fails_with_unknown_run_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"], catch_exceptions=False)
        result = runner.invoke(app, ["summarize-run", "--run", "no-such-run"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()
