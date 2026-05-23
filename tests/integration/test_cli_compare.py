"""Integration tests for `rageval compare`.

Builds two runs through the full mock pipeline (shared fixture) then
compares them.  No real API calls — all evaluators use mock/dummy providers.
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


def _all_run_ids(db_path: Path) -> list[str]:
    con = get_connection(db_path)
    try:
        rows = con.execute(
            "SELECT run_id FROM runs ORDER BY started_at ASC"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def _run_full_pipeline(runner, app, tmp_path, evalset_path, config_path):
    """Run `rageval run` + all evaluators for one run; return run_id."""

    def _inv(*args):
        result = runner.invoke(app, list(args), catch_exceptions=False)
        assert result.exit_code == 0, f"Command {args} failed:\n{result.output}"
        return result

    _inv("run", "--config", str(config_path), "--evalset", str(evalset_path))
    db_path = tmp_path / ".rageval" / "runs.db"
    run_id = _all_run_ids(db_path)[-1]  # latest

    _inv("evaluate-retrieval", "--run", run_id)
    _inv("evaluate-answer-relevance", "--run", run_id, "--config", str(config_path))
    _inv("extract-claims", "--run", run_id, "--config", str(config_path))
    _inv("evaluate-groundedness", "--run", run_id, "--config", str(config_path))
    _inv("summarize-run", "--run", run_id)
    return run_id


@pytest.fixture
def two_run_project(tmp_path, monkeypatch):
    """Set up a project with two completed runs; returns (tmp_path, baseline_id, candidate_id)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)

    runner.invoke(app, ["init"], catch_exceptions=False)
    runner.invoke(
        app,
        ["ingest", str(TINY_CORPUS), "--config", "configs/test.yaml"],
        catch_exceptions=False,
    )

    evalset_path = tmp_path / "evalsets" / "auto.jsonl"
    _write_evalset(evalset_path, _QUESTIONS)
    config_path = tmp_path / "configs" / "test.yaml"

    baseline_id = _run_full_pipeline(runner, app, tmp_path, evalset_path, config_path)
    candidate_id = _run_full_pipeline(runner, app, tmp_path, evalset_path, config_path)

    return tmp_path, baseline_id, candidate_id


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestCompareHappyPath:
    def test_exit_code_zero(self, two_run_project):
        _, b, c = two_run_project
        result = runner.invoke(
            app, ["compare", "--baseline", b, "--candidate", c], catch_exceptions=False
        )
        assert result.exit_code == 0

    def test_output_contains_baseline_label(self, two_run_project):
        _, b, c = two_run_project
        result = runner.invoke(
            app, ["compare", "--baseline", b, "--candidate", c], catch_exceptions=False
        )
        assert "Baseline" in result.output

    def test_output_contains_candidate_label(self, two_run_project):
        _, b, c = two_run_project
        result = runner.invoke(
            app, ["compare", "--baseline", b, "--candidate", c], catch_exceptions=False
        )
        assert "Candidate" in result.output

    def test_output_has_metric_comparison_table(self, two_run_project):
        _, b, c = two_run_project
        result = runner.invoke(
            app, ["compare", "--baseline", b, "--candidate", c], catch_exceptions=False
        )
        assert "Metric Comparison" in result.output

    def test_output_has_item_labels_table(self, two_run_project):
        _, b, c = two_run_project
        result = runner.invoke(
            app, ["compare", "--baseline", b, "--candidate", c], catch_exceptions=False
        )
        assert "Item Labels" in result.output or "Labels" in result.output

    def test_output_has_root_cause_table(self, two_run_project):
        _, b, c = two_run_project
        result = runner.invoke(
            app, ["compare", "--baseline", b, "--candidate", c], catch_exceptions=False
        )
        # summarize-run was called → root causes should be populated
        assert "Root-Cause" in result.output or "root" in result.output.lower()

    def test_output_mentions_known_metric(self, two_run_project):
        _, b, c = two_run_project
        result = runner.invoke(
            app, ["compare", "--baseline", b, "--candidate", c], catch_exceptions=False
        )
        assert "recall_at_k" in result.output or "faithfulness" in result.output

    def test_short_flags_work(self, two_run_project):
        _, b, c = two_run_project
        result = runner.invoke(
            app, ["compare", "-b", b, "-c", c], catch_exceptions=False
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# No root-causes scenario (summarize-run not called)
# ---------------------------------------------------------------------------

class TestCompareNoRootCauses:
    @pytest.fixture
    def two_runs_no_summarize(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "test.yaml").write_text(_CONFIG)

        runner.invoke(app, ["init"], catch_exceptions=False)
        runner.invoke(
            app,
            ["ingest", str(TINY_CORPUS), "--config", "configs/test.yaml"],
            catch_exceptions=False,
        )
        evalset_path = tmp_path / "evalsets" / "auto.jsonl"
        _write_evalset(evalset_path, _QUESTIONS)

        def _run_bare():
            runner.invoke(
                app,
                ["run", "--config", "configs/test.yaml", "--evalset", str(evalset_path)],
                catch_exceptions=False,
            )
            return _all_run_ids(tmp_path / ".rageval" / "runs.db")[-1]

        b = _run_bare()
        c = _run_bare()
        return tmp_path, b, c

    def test_shows_helpful_message(self, two_runs_no_summarize):
        _, b, c = two_runs_no_summarize
        result = runner.invoke(
            app, ["compare", "--baseline", b, "--candidate", c], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert "summarize-run" in result.output or "Root causes not found" in result.output


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestCompareErrors:
    def test_fails_without_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["compare", "--baseline", "a", "--candidate", "b"])
        assert result.exit_code != 0
        assert "init" in result.output.lower()

    def test_fails_with_unknown_baseline(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(app, ["init"], catch_exceptions=False)
        result = runner.invoke(
            app, ["compare", "--baseline", "no-such-run", "--candidate", "also-fake"]
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "baseline" in result.output.lower()

    def test_fails_with_unknown_candidate(self, two_run_project):
        _, b, _ = two_run_project
        result = runner.invoke(
            app, ["compare", "--baseline", b, "--candidate", "no-such-run"]
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "candidate" in result.output.lower()
