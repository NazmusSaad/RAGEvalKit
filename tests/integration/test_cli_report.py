"""Integration tests for `rageval report`.

Runs the full mock evaluation pipeline then generates an HTML report.
Validates the output file exists and contains expected content markers.
No real API calls — all evaluators use mock/dummy providers.
"""
import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rageval.cli.main import app
from rageval.storage.duckdb_dao import get_connection

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
        "question": f"Integration test question number {i}?",
        "reference_answer": f"Reference answer {i}.",
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


def _invoke(*args) -> object:
    return runner.invoke(app, list(args), catch_exceptions=False)


def _invoke_unchecked(*args) -> object:
    return runner.invoke(app, list(args))


# ---------------------------------------------------------------------------
# Shared fixture: full evaluation pipeline including summarize-run
# ---------------------------------------------------------------------------

@pytest.fixture
def full_run(tmp_path, monkeypatch):
    """Runs the full mock pipeline; returns (tmp_path, run_id)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)

    _invoke("init")
    _invoke("ingest", str(TINY_CORPUS), "--config", "configs/test.yaml")

    evalset_path = tmp_path / "evalsets" / "auto.jsonl"
    _write_evalset(evalset_path, _QUESTIONS)
    config_path = "configs/test.yaml"

    _invoke("run", "--config", config_path, "--evalset", str(evalset_path))
    run_id = _last_run_id(tmp_path / ".rageval" / "runs.db")

    _invoke("evaluate-retrieval", "--run", run_id)
    _invoke("evaluate-answer-relevance", "--run", run_id, "--config", config_path)
    _invoke("extract-claims", "--run", run_id, "--config", config_path)
    _invoke("evaluate-groundedness", "--run", run_id, "--config", config_path)
    _invoke("summarize-run", "--run", run_id)

    return tmp_path, run_id


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestReportHappyPath:
    def test_exit_code_zero(self, full_run):
        tmp_path, run_id = full_run
        result = _invoke_unchecked("report", "--run", run_id, "--output", "report.html")
        assert result.exit_code == 0

    def test_output_file_created(self, full_run):
        tmp_path, run_id = full_run
        _invoke("report", "--run", run_id, "--output", "report.html")
        assert (tmp_path / "report.html").exists()

    def test_output_is_html(self, full_run):
        tmp_path, run_id = full_run
        _invoke("report", "--run", run_id, "--output", "report.html")
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_html_contains_run_id(self, full_run):
        tmp_path, run_id = full_run
        _invoke("report", "--run", run_id, "--output", "report.html")
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert run_id in html

    def test_html_contains_metric_names(self, full_run):
        tmp_path, run_id = full_run
        _invoke("report", "--run", run_id, "--output", "report.html")
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        # At least one metric should appear
        assert "faithfulness" in html or "recall_at_k" in html or "answer_relevance" in html

    def test_html_contains_question_text(self, full_run):
        tmp_path, run_id = full_run
        _invoke("report", "--run", run_id, "--output", "report.html")
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert "Integration test question" in html

    def test_html_contains_root_cause_section(self, full_run):
        tmp_path, run_id = full_run
        _invoke("report", "--run", run_id, "--output", "report.html")
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        # summarize-run was run so root causes should be populated
        assert "Root-Cause" in html or "root" in html.lower()

    def test_html_contains_claim_verdicts(self, full_run):
        tmp_path, run_id = full_run
        _invoke("report", "--run", run_id, "--output", "report.html")
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        # extract-claims + evaluate-groundedness were run → verdicts present
        assert "supported" in html or "contradicted" in html or "unjudged" in html or "not_enough_info" in html

    def test_cli_output_mentions_report_path(self, full_run):
        tmp_path, run_id = full_run
        result = _invoke("report", "--run", run_id, "--output", "report.html")
        assert "report.html" in result.output

    def test_short_flags_work(self, full_run):
        tmp_path, run_id = full_run
        result = _invoke_unchecked("report", "-r", run_id, "-o", "short.html")
        assert result.exit_code == 0
        assert (tmp_path / "short.html").exists()

    def test_creates_parent_directory(self, full_run):
        tmp_path, run_id = full_run
        _invoke("report", "--run", run_id, "--output", "reports/subdir/out.html")
        assert (tmp_path / "reports" / "subdir" / "out.html").exists()


# ---------------------------------------------------------------------------
# Run with no optional data (no summarize-run, no claims)
# ---------------------------------------------------------------------------

class TestReportWithoutOptionalData:
    @pytest.fixture
    def bare_run(self, tmp_path, monkeypatch):
        """A run with only the basic `rageval run` step — no evaluators."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)

        _invoke("init")
        _invoke("ingest", str(TINY_CORPUS), "--config", "configs/test.yaml")
        evalset_path = tmp_path / "evalsets" / "auto.jsonl"
        _write_evalset(evalset_path, _QUESTIONS)

        _invoke("run", "--config", "configs/test.yaml", "--evalset", str(evalset_path))
        run_id = _last_run_id(tmp_path / ".rageval" / "runs.db")
        return tmp_path, run_id

    def test_exit_code_zero_bare_run(self, bare_run):
        tmp_path, run_id = bare_run
        result = _invoke_unchecked("report", "--run", run_id, "--output", "report.html")
        assert result.exit_code == 0

    def test_file_created_bare_run(self, bare_run):
        tmp_path, run_id = bare_run
        _invoke("report", "--run", run_id, "--output", "report.html")
        assert (tmp_path / "report.html").exists()

    def test_no_metrics_note_shown(self, bare_run):
        tmp_path, run_id = bare_run
        _invoke("report", "--run", run_id, "--output", "report.html")
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        # No metrics → note shown
        assert "No metrics" in html or "evaluate" in html.lower()

    def test_no_root_cause_note_shown(self, bare_run):
        tmp_path, run_id = bare_run
        _invoke("report", "--run", run_id, "--output", "report.html")
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert "summarize-run" in html

    def test_no_claims_note_shown(self, bare_run):
        tmp_path, run_id = bare_run
        _invoke("report", "--run", run_id, "--output", "report.html")
        html = (tmp_path / "report.html").read_text(encoding="utf-8")
        assert "extract-claims" in html


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestReportErrors:
    def test_fails_without_run_flag(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _invoke("init")
        result = _invoke_unchecked("report")
        assert result.exit_code != 0

    def test_fails_without_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _invoke_unchecked("report", "--run", "fake-id")
        assert result.exit_code != 0
        assert "init" in result.output.lower()

    def test_fails_with_unknown_run_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _invoke("init")
        result = _invoke_unchecked("report", "--run", "no-such-run-id")
        assert result.exit_code != 0
        assert "not found" in result.output.lower()
